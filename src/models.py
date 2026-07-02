import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Suppress FutureWarnings from PyMC and ArviZ

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az
import pymc_bart as pmb
from scipy.stats import gaussian_kde
import pytensor.tensor as pt

class staticModel:
    def __init__(self):
        self.mean = None

    def fit(self, X_train, y_train):
        self.mean = np.mean(y_train)

    def predict(self, X_new):
        return np.full(shape=(len(X_new),), fill_value=self.mean)
    


class energinet_model:
    """Class for Bayesian regression model using additive regression trees."""

    def __init__(
            self,
            N_DRAWS = 2000, # Number of posterior samples to draw
            N_TUNE = 6000,  # Number of tuning steps for the sampler (burn-in)
            N_CHAINS = 4,   # Number of MCMC chains to run
            N_CORES = 4,    # Number of CPU cores to use for parallel sampling
            m = 20,         # Number of trees in the BART model
            seed = 89       # Random seed for reproducibility
        ):
        self.N_DRAWS = N_DRAWS
        self.N_TUNE = N_TUNE
        self.N_CHAINS = N_CHAINS
        self.N_CORES = N_CORES
        self.m = m  
        self.seed = seed

    
    def fit(self,X_data,y_train_scaled):
        """Method for training regression model.
        
        :param X_data: pandas DataFrame or numpy array of covariates/features (predictor) variables data. Rows of observations and variables in columns.
        :param y_train_scaled: pandas DataSeries, DataFrame or numpy array of scaled/normalised univariate target variable.
        :param posterior: Inference object containing posterior distribution of model parameters sampled using the Hamiltonian Monte Carlo method.
        """
        pt.pytensor.config.mode = "NUMBA" # Try this
        self.X_train = X_data
        self.y_train_scaled = y_train_scaled
        self.model = pm.Model()
        with self.model:
            covars= pm.Data('covars', X_data, dims=("obs_id", "train_cols"))
            
            ww = pmb.BART('ww',
                          covars,
                          y_train_scaled,
                          m=self.m, 
                          size=2, # Model both mean and variance (heteroscedasticity) for each observation
                          separate_trees=False, # False means same tree structure for both mean and variance, True means separate tree structure for mean and variance.
                          ) 
            pm.Normal('obs',
                      mu=ww[0],             # Mean parameter
                      sigma=np.exp(ww[1]),  # Standard deviation parameter, using exponential to ensure positivity (heteroscedasticity)
                      observed=y_train_scaled,
                      shape=ww[0].shape,
                      dims="obs_id")


            #NOTE: (From Copilot) The PGBART sampler is used here for sampling the posterior 
            # distribution of the BART model parameters. It is a particle Gibbs sampler 
            # specifically designed for BART models, which can be more efficient than 
            # traditional MCMC samplers for this type of model. The number of particles 
            # can be adjusted based on the complexity of the model and the size of the 
            # dataset, with a default value of 10 particles.

            pgstep = pmb.PGBART(vars=[ww],num_particles=10) # default 10
            
            self.posterior = pm.sample(tune=self.N_TUNE,
                                       draws=self.N_DRAWS,
                                       cores=self.N_CORES,
                                       chains = self.N_CHAINS,
                                       step=[pgstep],
                                       compute_convergence_checks=False,
                                       random_seed=self.seed) # Set random seed for reproducibility

        self.all_trees = list(self.model.ww.owner.op.all_trees)

    def predict(self,X_data):
        """Method for prediction using trained regression model.
        
        :param X_data: pandas DataFrame or numpy array of covariates/features (predictor) variables data. Rows of observations and variables in columns.
        :param posterior_predictive: Inference object containing posterior_predictive distribution of target values.
        :return df_predicted: pandas DataFrame, DataFrame containing predicted sampled target values. Rows of observations and columns of draws from posterior predictive distribution. 
        """
        pt.pytensor.config.mode = "FAST_RUN" # Try this
        if not hasattr(self, 'posterior'):
            raise ValueError('The model needs to be fitted first')

        with self.model:
            pm.set_data({'covars': X_data})
            self.posterior_predictive=pm.sample_posterior_predictive(trace=self.posterior, random_seed=self.seed) # Set random seed for reproducibility
        
        self.df_predicted = self.inference_data_to_pandas_dataframe(self.posterior_predictive,X_data)
        return self.df_predicted

    def inference_data_to_pandas_dataframe(self,inference_data,X_data):
        """Method for extracting posterior predictive observations from Inference object.

        :param inference_data: Inference object, Inference object containing posterior predictive distribution.
        :param X_data: pandas DataFrame, DataFrame of observations for predictor variables for corresponding posterior predictive distribution.
        :return df_predicted: pandas DataFrame, DataFrame containing predicted sampled target values. Rows of observations and columns of draws from posterior predictive distribution.
        """
        data = inference_data.posterior_predictive['obs'].values.transpose(2,0,1).reshape(len(X_data),-1)
        # data = self.model_selection.scaler.rescale_data(data)
        df_predicted = pd.DataFrame(data,index=X_data.index)
        return df_predicted
    
    def get_hdi_intervals(self, coverage=0.95):      
        hdi_data = az.hdi(self.posterior_predictive.posterior_predictive, hdi_prob=coverage)["obs"]

        return hdi_data  
    

class BayesRegressor:
    """Class for Bayesian regression model using additive regression trees."""

    def __init__(
            self, N_DRAWS = 2000, N_TUNE = 6000, N_CHAINS = 4, N_CORES = 4, m = 20, alpha=0.95, beta=2, particles = 10, distribution = "studentT", seed = 89
    ):
        """Constructor for BayesRegressor class.
        :param N_DRAWS: int, number of draws from posterior distribution after tuning.
        :param N_TUNE: int, number of tuning steps for MCMC sampling.
        :param N_CHAINS: int, number of MCMC chains to run in parallel.
        :param N_CORES: int, number of CPU cores to use for parallel MCMC sampling.
        :param m: int, number of trees in the BART model. Default is 20. Higher values can capture more complex relationships but may lead to overfitting and increased computational
        time.
        :param alpha: float, prior parameter for tree depth. Default is 0.95. Lower values lead to shallower trees, while higher values allow for deeper trees.
        :param beta: float, prior parameter for tree depth. Default is 2. Higher values lead to shallower trees, while lower values allow for deeper trees.
        :param distribution: str, the probability distribution for the likelihood function. Default is "studentT".
        :param seed: int, random seed for reproducibility. Default is 89.
        """

        self.N_DRAWS = N_DRAWS
        self.N_TUNE = N_TUNE
        self.N_CHAINS = N_CHAINS
        self.N_CORES = N_CORES
        self.m = m
        self.alpha = alpha
        self.beta = beta
        self.particles = particles
        self.distribution = distribution
        self.seed = seed
        self.ww = None
        self.posterior = None
        
    def fit(self,X_data,y_train_scaled, progressbar=True):
        """Method for training regression model.
        
        :param X_data: pandas DataFrame or numpy array of covariates/features (predictor) variables data. Rows of observations and variables in columns.
        :param y_train_scaled: pandas DataSeries, DataFrame or numpy array of scaled/normalised univariate target variable.
        :param posterior: Inference object containing posterior distribution of model parameters sampled using the Hamiltonian Monte Carlo method.
        """
        pt.pytensor.config.mode = "NUMBA" # Try this. Maybe not needed in the latest version of PyMC and PyMC-BART, which should have improved performance.
        self.X_train = X_data
        self.y_train_scaled = y_train_scaled
        self.model = pm.Model()
        with self.model:
            covars= pm.Data('covars', X_data, dims=("obs_id", "train_cols"))
            
            self.ww = pmb.BART('ww',
                          covars,
                          y_train_scaled,
                          m=self.m,
                          size=2,              # Don't just estimate the mean, also estimate the variance (heteroscedasticity) for each observation.
                          separate_trees=True, # Mean and variance follow the same tree structure, but different leaf values. If true, mean and variance follow different tree structures.
                          alpha=self.alpha,    # Prior parameter for tree depth, default 0.95. Lower values lead to shallower trees.
                          beta=self.beta)      # Prior parameter for tree depth, default 2. Higher values lead to shallower trees.
            
            
            if self.distribution == "studentT":
                pm.StudentT('obs',
                          nu = 2,                    # Degrees of freedom parameter for Student's t distribution, which models the likelihood. Higher values lead to a distribution closer to normal, while lower values allow for heavier tails, providing robustness to outliers.
                          mu=self.ww[0],             # Mean parameter
                          sigma=np.exp(self.ww[1]),  # Standard deviation parameter, using exponential to ensure positivity (heteroscedasticity)
                          observed=y_train_scaled,
                          shape=self.ww[0].shape,
                          dims="obs_id")
                
            elif self.distribution == "normal":
                pm.Normal('obs',
                          mu=self.ww[0],             # Mean parameter
                          sigma=np.exp(self.ww[1]),  # Standard deviation parameter, using exponential to ensure positivity (heteroscedasticity)
                          observed=y_train_scaled,
                          shape=self.ww[0].shape,
                          dims="obs_id")
            else:
                raise ValueError("Invalid distribution specified. Choose 'studentT' or 'normal'.")

            #NOTE: (From Copilot) The PGBART sampler is used here for sampling the posterior 
            # distribution of the BART model parameters. It is a particle Gibbs sampler 
            # specifically designed for BART models, which can be more efficient than 
            # traditional MCMC samplers for this type of model. The number of particles 
            # can be adjusted based on the complexity of the model and the size of the 
            # dataset, with a default value of 10 particles.
            # Force the use of Particle Gibbs sampler and be able to tune it. 
            # Otherwise, the default sampler is (automatic?) NUTS. Maybe not ideal for BART models?
            
            pgstep = pmb.PGBART(vars=[self.ww], num_particles=self.particles) # default 10. 
            
            self.posterior = pm.sample(tune=self.N_TUNE, 
                                       draws=self.N_DRAWS, 
                                       cores=self.N_CORES, 
                                       chains = self.N_CHAINS,
                                       step=[pgstep],
                                       compute_convergence_checks=False, # Disable convergence checks to speed up sampling. Can be enabled for more thorough analysis.
                                       progressbar=progressbar,
                                       discard_tuned_samples=True,  # Discard tuning samples to save memory, since we are not using them for inference.           
                                       idata_kwargs={"log_likelihood": False}, # Don't compute log likelihood to save time and memory, since we are not using it for model comparison or diagnostics.
                                       random_seed=self.seed) # Set random seed for reproducibility
        self.all_trees = list(self.model.ww.owner.op.all_trees)

    def predict(self,X_data, progressbar=True):
        """Method for prediction using trained regression model.
        
        :param X_data: pandas DataFrame or numpy array of covariates/features (predictor) variables data. Rows of observations and variables in columns.
        :param posterior_predictive: Inference object containing posterior_predictive distribution of target values.
        :return df_predicted: pandas DataFrame, DataFrame containing predicted sampled target values. Rows of observations and columns of draws from posterior predictive distribution. 
        """
        pt.pytensor.config.mode = "FAST_RUN" # Try this
        if self.posterior is None:
            raise ValueError('The model needs to be fitted first')

        with self.model:
            pm.set_data({'covars': X_data})
            self.posterior_predictive=pm.sample_posterior_predictive(trace=self.posterior, progressbar=progressbar, random_seed=self.seed)
        
        self.df_predicted = self._inference_data_to_pandas_dataframe(self.posterior_predictive,X_data)
        return self.df_predicted

    def _inference_data_to_pandas_dataframe(self,inference_data,X_data):
        """Method for extracting posterior predictive observations from Inference object.

        :param inference_data: Inference object, Inference object containing posterior predictive distribution.
        :param X_data: pandas DataFrame, DataFrame of observations for predictor variables for corresponding posterior predictive distribution.
        :return df_predicted: pandas DataFrame, DataFrame containing predicted sampled target values. Rows of observations and columns of draws from posterior predictive distribution.
        """
        data = inference_data.posterior_predictive['obs'].values.transpose(2,0,1).reshape(len(X_data),-1)
        df_predicted = pd.DataFrame(data,index=X_data.index if hasattr(X_data, 'index') else np.arange(len(X_data)))
        return df_predicted
    

    def get_hdi_intervals(self, coverage=0.95):      
        hdi_data = az.hdi(self.posterior_predictive.posterior_predictive, hdi_prob=coverage)["obs"]
    
        return hdi_data  
