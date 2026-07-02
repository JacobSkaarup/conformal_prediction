import os
import pandas as pd
import numpy as np
import requests
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  
DATASET_DIR = str(PROJECT_ROOT / "dataset")
ADDITIONAL_DIR = str(PROJECT_ROOT / "dataset" / "additionalFeatures")
CURRENT_DATE = str(pd.Timestamp.today().date())



def clean_missing_data_chunks(df, max_interp_gap_hours=2, min_missing_cols=1):
    """
    Drop rows that belong to large consecutive missing-data chunks and
    interpolate only small gaps.
    """
    df = df.sort_index()

    # Mark rows where at least `min_missing_cols` features are missing.
    missing_rows = df.isna().sum(axis=1) >= min_missing_cols

    if missing_rows.any():
        groups = (missing_rows != missing_rows.shift()).cumsum()
        drop_index = []

        for _, mask in missing_rows.groupby(groups):
            if not mask.iloc[0]:
                continue

            chunk_index = mask.index
            if len(chunk_index) > max_interp_gap_hours:
                drop_index.extend(chunk_index)

        if drop_index:
            df = df.drop(index=drop_index)

    # Only fill short gaps. Larger chunks remain NaN (or are already dropped).
    return df.interpolate(method="time", limit=max_interp_gap_hours, limit_direction="both")


def fetch_from_energidataservice(dataset, start_date, end_date):
    file_name = f'{ADDITIONAL_DIR}/{dataset}_{start_date}_to_{end_date}.csv'
    # Check if the file already exists
    if os.path.exists(file_name):
        return pd.read_csv(file_name)
    else:
        url = f'https://api.energidataservice.dk/dataset/{dataset}?start={start_date}&end={end_date}'

        
        response = requests.get(url)
        if response.status_code == 200:
            result = response.json()
            records = result.get('records', [])
            df = pd.DataFrame.from_records(records)
            df.to_csv(file_name, index=False)
            return df
        else:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
        

def load_data(zone_direction = "dk1 down", target = "volbids"):
    """
    Loads the features and target datasets, performs feature engineering, and merges the datasets together.
    Args:        
        zone_direction (str): The zone and direction for which to load data, either "dk1 down", "dk1 up", "dk2 down", or "dk2 up".
        target (str): The target variable to load, either "volbids" or "FRR".
    Returns:        
        features (pd.DataFrame): The processed features dataset.
        target (pd.Series): The target variable corresponding to the features dataset.
    """

    # Check if combined features and target files exist for faster loading, if not load the original ones
    if os.path.exists(DATASET_DIR + "/" + zone_direction + r"/features_comb.parquet"):
        features = pd.read_parquet(DATASET_DIR + "/" + zone_direction + r"/features_comb.parquet")
        targ = pd.read_parquet(DATASET_DIR + "/" + zone_direction + r"/target_comb.parquet")

        suffix = "_" + zone_direction.split()[0].upper()
        target = targ[target + suffix]

    else:
        features = pd.read_parquet(DATASET_DIR + "/" +  zone_direction +  r"/features.parquet")
        features_updated = pd.read_parquet(DATASET_DIR + "/" +  zone_direction +  r"/features_updated.parquet")
        targ = pd.read_parquet(DATASET_DIR + "/" + zone_direction + r"/target.parquet")
        targ_updated = pd.read_parquet(DATASET_DIR + "/" + zone_direction + r"/target_updated.parquet")

        # Combine original and updated datasets, prioritizing updated values
        features = features.combine_first(features_updated)
        targ = targ.combine_first(targ_updated)

        ##### Net load ####
        features["net_load"] = features["forecast_offline_consumption_MW"] + features["sum_offshore_wind_total_MW_onshore_wind_total_MW"] + features["forecast_offline_solar_MW"]


        ##### Weekday and Weekend ####
        features["weekday"] = features.index.weekday.isin([0, 1, 2, 3, 4]).astype(int)
        # features["weekend"] = features.index.weekday.isin([5, 6]).astype(int)


        ##### Time of year ####
        features["time_year_cos"] = np.cos(2 * np.pi * features.index.dayofyear / 365.25)
        features["time_year_sin"] = np.sin(2 * np.pi * features.index.dayofyear / 365.25)


        ##### Net stress ####
        features["net_stress"] = (np.abs(features["forecast_offline_consumption_MW"].diff()) +
                                  np.abs(features["sum_offshore_wind_total_MW_onshore_wind_total_MW"].diff()) +
                                  np.abs(features["forecast_offline_solar_MW"].diff())
                                 ).fillna(0)
        

        ##### Prices as persistance #####

        # Read in the newest price dataset and process it. Uses 15-minute resolution
        pricesNew = fetch_from_energidataservice('DayAheadPrices', '2025-01-01', CURRENT_DATE)
        pricesNew = pricesNew[["TimeUTC","DayAheadPriceDKK","PriceArea"]]
        pricesNew.rename(columns={"TimeUTC": "time_utc"}, inplace=True)
        pricesNew.set_index("time_utc", inplace=True)
        pricesNew.index = pd.to_datetime(pricesNew.index)
        pricesNew["DayAheadPriceDKK"] = pricesNew["DayAheadPriceDKK"].replace(',', '.', regex=True).astype(float)

        # Split pricesNew into multiple day ahead prices based on the price area
        price_areas = pricesNew["PriceArea"].unique()
        for area in price_areas:
            pricesNew[f"DApriceDKK_{area}"] = pricesNew.loc[pricesNew["PriceArea"] == area, "DayAheadPriceDKK"]

        pricesNew.drop(columns=["DayAheadPriceDKK", "PriceArea"], inplace=True)
        pricesNew = pricesNew.groupby(pricesNew.index).first()
        pricesNew = pricesNew.resample('h').mean()

        # Read in the second price dataset and process it similarly to the first one
        pricesOld = fetch_from_energidataservice('Elspotprices', '2025-01-01', CURRENT_DATE)
        pricesOld = pricesOld[["HourUTC","SpotPriceDKK","PriceArea"]]
        pricesOld.rename(columns={"HourUTC": "time_utc", "SpotPriceDKK": "DayAheadPriceDKK"}, inplace=True)
        pricesOld.set_index("time_utc", inplace=True)
        pricesOld.index = pd.to_datetime(pricesOld.index)
        pricesOld["DayAheadPriceDKK"] = pricesOld["DayAheadPriceDKK"].replace(',', '.', regex=True).astype(float)


        # Split pricesOld into multiple spot prices based on the price area
        for area in price_areas:
            pricesOld[f"DApriceDKK_{area}"] = pricesOld.loc[pricesOld["PriceArea"] == area, "DayAheadPriceDKK"]

        pricesOld.drop(columns=["DayAheadPriceDKK", "PriceArea"], inplace=True)

        # Remove duplicate indexes in pricesOld
        pricesOld = pricesOld.groupby(pricesOld.index).first()
        prices = pd.concat([pricesOld, pricesNew], axis=0)

        # Localize prices to utc and shift 24 hours as persistence model
        prices.index = prices.index.tz_localize('UTC')
        prices = prices.shift(24)

        # Merge features with prices
        features = features.merge(prices, left_index=True, right_index=True, how='left')
        features.index = pd.to_datetime(features.index)


        ##### Price volatility features #####
        # Insert std of prices as features for that day
        for area in price_areas:
            features[f"DApriceDKK_{area}_std"] = np.repeat(features[f"DApriceDKK_{area}"].groupby(features.index.date).std().values, 24)



        ##### Inertia features #####
        inertia = fetch_from_energidataservice('InertiaNordicSyncharea', '2025-01-01', CURRENT_DATE)
        inertia.drop(columns=["HourDK"], inplace=True)
        inertia.rename(columns={"HourUTC": "time_utc"}, inplace=True)
        inertia["time_utc"] = pd.to_datetime(inertia["time_utc"], format="%Y-%m-%dT%H:%M:%S")
        inertia.set_index("time_utc", inplace=True)
        inertia.index = inertia.index.tz_localize('UTC')

        # Convert strings into floats and shift 48 hours as persistence model
        inertia = inertia.apply(lambda x: x.replace(',', '.', regex=True).astype(float))
        inertia = inertia.shift(48)

        features = features.merge(inertia, left_index=True, right_index=True, how='left')

        ###### Gradient features ######
        def compute_gradients(df, columns):
            for col in columns:
                df[col + "_grad"] = df[col].diff()
            return df

        gradient_columns = ["forecast_offline_consumption_MW", "sum_offshore_wind_total_MW_onshore_wind_total_MW", "forecast_offline_solar_MW",
                            "mean_temperature_K", "DApriceDKK_DK1", "DApriceDKK_DK2",
                            "DApriceDKK_NO2", "DApriceDKK_DE", "DApriceDKK_SE3", "DApriceDKK_SE4"
                            ]

        features = compute_gradients(features, gradient_columns)


        ##### Lag features #####
        def create_lag_features(df, lags,  cols):
            df_lagged = df.copy()
            for lag in lags:
                for col in cols:
                    df_lagged[f'{col}_lag_{lag}'] = df[col].shift(lag)

            return df_lagged


        lagged_columns = ["forecast_offline_consumption_MW", "sum_offshore_wind_total_MW_onshore_wind_total_MW", "forecast_offline_solar_MW"]
        features = create_lag_features(features, lags=[1, 2, 3], cols=lagged_columns)
        features.index = pd.to_datetime(features.index)
        targ.index = pd.to_datetime(targ.index)



        ##### Unavailability features, scheduled maintenance #####
        # Source: Entsoe Transparency platform -> outages -> Unavailability of Production and Generation Units
        unavail = pd.DataFrame()

        unavailability_files = sorted((Path(ADDITIONAL_DIR) ).glob("UNAVAILABILITY*.csv"))
        n_unavailability_files = len(unavailability_files)

        if n_unavailability_files == 0:
            raise FileNotFoundError(f"No unavailability files found in {(Path(ADDITIONAL_DIR))}")

        for file_path in unavailability_files:
            temp = pd.read_csv(file_path)
            unavail = pd.concat([unavail, temp], axis=0)

        unavail = unavail[["Time Interval (CET/CEST)", "Installed (MW)", "Available (MW)"]]

        # Split time interval into start and end timestamps
        unavail[["start_time", "end_time"]] = unavail["Time Interval (CET/CEST)"].str.split(" - ", expand=True)
        unavail["start_time"] = pd.to_datetime(unavail["start_time"], format="%d/%m/%Y %H:%M")
        unavail["end_time"] = pd.to_datetime(unavail["end_time"], format="%d/%m/%Y %H:%M")
        unavail["reduced_cap"] = unavail["Installed (MW)"] - unavail["Available (MW)"]
        unavail = unavail[["start_time", "end_time", "reduced_cap"]].dropna()

        # Build hourly aggregate of overlapping reduced capacity using event deltas
        u = unavail.copy()
        u["start_hour"] = u["start_time"].dt.ceil("h")
        u["end_hour"] = u["end_time"].dt.ceil("h")

        events = pd.concat(
            [
                pd.DataFrame({"time": u["start_hour"], "delta": u["reduced_cap"]}),
                pd.DataFrame({"time": u["end_hour"], "delta": -u["reduced_cap"]}),
            ],
            ignore_index=True,
        )

        events = events.groupby("time", as_index=True)["delta"].sum().sort_index()
        dates = pd.date_range(start=u["start_time"].min(), end=u["end_time"].max(), freq="h")
        reduced_cap_tot = events.reindex(dates, fill_value=0).cumsum().to_frame("reduced_cap_tot")

        reduced_cap_tot.index = pd.to_datetime(reduced_cap_tot.index).tz_localize("UTC")
        reduced_cap_tot = reduced_cap_tot.astype(float)
        features = features.merge(reduced_cap_tot, left_index=True, right_index=True, how="left")
        features.index = pd.to_datetime(features.index)


        ##### Final cleaning #####
        # Drop first day due to lagging and persistence
        features = features.loc["2025-01-02":]

        # Drop large missing chunks and only interpolate short gaps (e.g., 1-2 hours).
        features = clean_missing_data_chunks(features, max_interp_gap_hours=2, min_missing_cols=1)


        # Keep only targets that have corresponding features
        common_index = features.index.intersection(targ.index)
        features = features.loc[common_index]
        targ = targ.loc[common_index]
        

        # Add True CM need as target
        suffix = "_" + zone_direction.split()[0].upper()
        targ["CM" + suffix] = targ.iloc[:, 0] - targ.iloc[:, 1]


        # Save the processed features and target datasets for future use
        features.to_parquet(DATASET_DIR + "/" +  zone_direction +  r"/features_comb.parquet")
        targ.to_parquet(DATASET_DIR + "/" + zone_direction + r"/target_comb.parquet")

        
        target = targ[target + suffix]


    return features, target



def remove_high_correlation(X, threshold=0.9):
    """
    Removes features from the dataset that have a correlation higher than the specified threshold.
    Args:
        X (pd.DataFrame): The input features dataset.
        threshold (float): The correlation threshold above which features will be removed.
    Returns:
        pd.DataFrame: The dataset with highly correlated features removed.
        set: The set of removed feature names.
    """
    corr_matrix = X.corr()
    col_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.append(colname)
    return X.drop(columns=col_corr), col_corr


