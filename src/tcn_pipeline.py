import copy
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils import clip_grad_norm_
from typing import Callable, Optional, Any



class CausalConv1d(nn.Module):
    """1D convolution with causal padding"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=self.padding
        )
        # Apply weight normalization to the convolutional layer
        self.conv = weight_norm(self.conv)

    def forward(self, x):
        x = self.conv(x)
        # Remove the extra padding on the right to maintain causality
        return x[:, :, :-self.padding] if self.padding > 0 else x

class LayerNorm1d(nn.Module):
    """Layer normalization for 1D tensors"""
    def __init__(self, num_channels):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels)

    def forward(self, x):
        # Transpose from (Batch, Channels, Time) to (Batch, Time, Channels)
        # Apply LayerNorm, then transpose back
        return self.ln(x.transpose(1, 2)).transpose(1, 2)

class TCNBlock(nn.Module):
    """Temporal Convolutional Network block with residual connection"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        # Each block consists of two causal convolutional layers with ReLU and dropout, plus a residual connection
        self.net = nn.Sequential(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation),
            LayerNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            CausalConv1d(out_channels, out_channels, kernel_size, dilation),
            LayerNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # If the number of input and output channels differ, we need a 1x1 convolution to match dimensions for the residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class MultivariateTCN(nn.Module):
    """Multivariate Temporal Convolutional Network for regression"""
    def __init__(self, num_inputs, num_channels, output_dim, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        # Build a stack of TCN blocks according to the specified number of channels and layers
        for i, out_ch in enumerate(num_channels):
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            dilation = 2 ** i
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
        self.tcn = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_dim)

    def forward(self, x):
        # x: (B, T, F)
        x = x.transpose(1, 2)          # -> (B, F, T)
        y = self.tcn(x)                # -> (B, C, T)
        last_step = y[:, :, -1]        # -> (B, C)
        return self.linear(last_step)  # -> (B, output_dim)


class ProbabilisticTCN(MultivariateTCN):
    """TCN that outputs mean and std for probabilistic regression"""
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super().__init__(num_inputs, num_channels, output_dim=2, kernel_size=kernel_size, dropout=dropout)

    def forward(self, x):
        out = super().forward(x)   # (B, 2)
        mu = out[:, 0]
        # Use softplus to ensure the predicted standard deviation is positive, and add a small constant for numerical stability
        sigma = F.softplus(out[:, 1]) + 1e-6
        return mu, sigma


class TCNForecaster:
    """High-level TCN training/prediction API that keeps legacy function calls intact."""

    SUPPORTED_TASKS = {"point", "quantile", "probabilistic"}

    def __init__(
        self,
        task: str = "point",
        quantiles=(0.025, 0.5, 0.975),
        valshare: float = 0.2,
        window_size: int = 96,
        batch_size: int = 64,
        epochs: int = 120,
        n_layers: int = 3,
        n_filters: int = 16,
        kernel_size: int = 5,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        patience: int = 3,
        factor: float = 0.5,
        device: Optional[torch.device] = None,
        use_amp: bool = True,
        show_progress: bool = True,
    ):
        if task not in self.SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task '{task}'. Expected one of: {sorted(self.SUPPORTED_TASKS)}")
        if task == "quantile" and (quantiles is None or len(quantiles) == 0):
            raise ValueError("quantiles must be provided for task='quantile'")

        self.task = task
        self.quantiles = tuple(quantiles) if quantiles is not None else None
        self.valshare = valshare
        self.window_size = window_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.n_layers = n_layers
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.factor = factor
        self.use_amp = use_amp
        self.show_progress = show_progress

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        self.model: Optional[nn.Module] = None
        self.best_stats: Optional[dict[str, Any]] = None

    @staticmethod
    def _as_arrays(*dfs):
        out = []
        for df in dfs:
            if isinstance(df, pd.DataFrame) or isinstance(df, pd.Series):
                out.append(df.to_numpy())
            else:
                out.append(np.asarray(df))
        return tuple(out)

    @staticmethod
    def _create_sequences(X, y, window_size: int):
        X_seq, y_seq = [], []
        for i in range(len(X) - window_size):
            X_seq.append(X[i:i + window_size])
            y_seq.append(y[i + window_size])
        return (
            torch.tensor(np.asarray(X_seq), dtype=torch.float32),
            torch.tensor(np.asarray(y_seq), dtype=torch.float32).reshape(-1, 1),
        )

    def _make_loaders(self, X_train, y_train, X_val, y_val):
        Xtr, ytr = self._create_sequences(X_train, y_train, self.window_size)
        Xva, yva = self._create_sequences(X_val, y_val, self.window_size)
        num_workers = min(4, os.cpu_count() or 1)
        pin_memory = torch.cuda.is_available()
        train_loader = DataLoader(
            TensorDataset(Xtr, ytr),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        )
        val_loader = DataLoader(
            TensorDataset(Xva, yva),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        )
        return train_loader, val_loader

    @staticmethod
    def _quantile_loss(preds, target, quantiles):
        losses = []
        for i, q in enumerate(quantiles):
            e = target - preds[:, i:i + 1]
            losses.append(torch.maximum(q * e, (q - 1) * e))
        return torch.mean(torch.stack(losses))

    @staticmethod
    def _gaussian_nll_loss(mu, sigma, target):
        if not torch.isfinite(mu).all():
            raise RuntimeError("Non-finite mu in probabilistic head")
        if not torch.isfinite(sigma).all():
            raise RuntimeError("Non-finite sigma in probabilistic head")
        if not torch.isfinite(target).all():
            raise RuntimeError("Non-finite target in probabilistic loss")

        sigma = torch.clamp(sigma, min=1e-6, max=1e3)
        nll = 0.5 * (
            np.log(2.0 * np.pi)
            + 2.0 * torch.log(sigma)
            + ((target - mu) / sigma) ** 2
        )
        loss = nll.mean()

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite gaussian NLL loss")
        return loss

    def _fit_loop(
        self,
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        scheduler,
        epoch_end_callback: Optional[Callable[[int, float], None]] = None,
    ):
        best_state = None
        best_val = float("inf")
        best_epoch = -1
        train_losses, val_losses = [], []
        amp_enabled = self.use_amp and (self.device.type == "cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        for epoch in tqdm(range(self.epochs), desc="Training", unit="epoch", leave=False, disable=not self.show_progress):
            model.train()
            tr_loss = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(self.device, non_blocking=True), by.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                    loss = loss_fn(model, bx, by)

                if not torch.isfinite(loss):
                    raise RuntimeError(f"Loss is {loss.item()}, training is diverging")

                try:
                    scaler.scale(loss).backward()
                except RuntimeError as e:
                    if "FIND" in str(e) or "engine" in str(e):
                        raise RuntimeError(f"cuDNN engine selection failed: {e}")
                    raise

                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()
                tr_loss += loss.item()
            tr_loss /= max(len(train_loader), 1)
            train_losses.append(tr_loss)

            model.eval()
            va_loss = 0.0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(self.device, non_blocking=True), by.to(self.device, non_blocking=True)
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                        va_loss += loss_fn(model, bx, by).item()
            va_loss /= max(len(val_loader), 1)
            val_losses.append(va_loss)
            scheduler.step(va_loss)

            if epoch_end_callback is not None:
                epoch_end_callback(epoch, va_loss)

            if va_loss < best_val:
                best_val = va_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())

        if best_state is not None:
            model.load_state_dict(best_state)

        best_stats = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val,
            "best_epoch": best_epoch,
        }
        return model, best_stats

    def _build_model(self, num_inputs: int) -> nn.Module:
        channels = [self.n_filters] * self.n_layers
        if self.task == "probabilistic":
            model = ProbabilisticTCN(
                num_inputs=num_inputs,
                num_channels=channels,
                kernel_size=self.kernel_size,
                dropout=self.dropout,
            )
        else:
            output_dim = 1 if self.task == "point" else len(self.quantiles)
            model = MultivariateTCN(
                num_inputs=num_inputs,
                num_channels=channels,
                output_dim=output_dim,
                kernel_size=self.kernel_size,
                dropout=self.dropout,
            )
        return model.to(self.device)

    def _build_loss_fn(self) -> Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor]:
        if self.task == "point":
            mse = nn.MSELoss()

            def loss_fn(m, bx, by):
                return mse(m(bx), by)

            return loss_fn

        if self.task == "quantile":

            def loss_fn(m, bx, by):
                return self._quantile_loss(m(bx), by, self.quantiles)

            return loss_fn

        def loss_fn(m, bx, by):
            mu, sigma = m(bx)
            return self._gaussian_nll_loss(mu, sigma, by.squeeze(-1))

        return loss_fn

    def fit(
        self,
        X_train,
        y_train,
        epoch_end_callback: Optional[Callable[[int, float], None]] = None,
    ):
        n = len(X_train)
        split_idx = int(n * (1 - self.valshare))
        X_arr, y_arr = self._as_arrays(X_train, y_train)
        X_tr, y_tr = X_arr[:split_idx], y_arr[:split_idx]
        X_va, y_va = X_arr[split_idx:], y_arr[split_idx:]

        train_loader, val_loader = self._make_loaders(X_tr, y_tr, X_va, y_va)

        self.model = self._build_model(num_inputs=X_tr.shape[1])
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            "min",
            patience=self.patience,
            factor=self.factor,
        )
        loss_fn = self._build_loss_fn()

        self.model, self.best_stats = self._fit_loop(
            self.model,
            train_loader,
            val_loader,
            loss_fn,
            optimizer,
            scheduler,
            epoch_end_callback=epoch_end_callback,
        )

        if self.task == "quantile" and self.best_stats is not None:
            self.best_stats["quantiles"] = list(self.quantiles)
        return self

    def _ensure_fitted(self):
        if self.model is None:
            raise RuntimeError("Model is not fitted. Call fit(...) first.")

    def predict(self, X, y):
        """Predict with fitted model. Return format matches the legacy helper functions."""
        self._ensure_fitted()

        X_arr, y_arr = self._as_arrays(X, y)
        y_flat = np.asarray(y_arr).reshape(-1)
        X_seq, _ = self._create_sequences(X_arr, y_flat, self.window_size)
        X_seq = X_seq.to(self.device)

        self.model.eval()
        with torch.no_grad():
            if self.task == "probabilistic":
                mu, sigma = self.model(X_seq)
            else:
                pred = self.model(X_seq)

        if isinstance(y, pd.Series):
            idx = y.index[self.window_size:]
            y_true = y.iloc[self.window_size:]
        else:
            idx = pd.RangeIndex(start=self.window_size, stop=len(y_flat))
            y_true = pd.Series(y_flat[self.window_size:], index=idx)

        if self.task == "point":
            return pd.Series(pred.cpu().numpy().flatten(), index=idx), y_true

        if self.task == "quantile":
            qcols = [f"q{q}" for q in self.quantiles]
            return pd.DataFrame(pred.cpu().numpy(), index=idx, columns=qcols), y_true

        return (
            pd.Series(mu.cpu().numpy().flatten(), index=idx),
            pd.Series(sigma.cpu().numpy().flatten(), index=idx),
            y_true,
        )