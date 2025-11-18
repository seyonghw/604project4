# prediction.py
import os
import glob
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz
from statsmodels.tsa.statespace.sarimax import SARIMAX

MODELS_DIR = "/app/models"
TEMP_DIR = "/app/temperature"
OUTPUT_PATH = "/app/output/merged_all_years.csv"

TBASE = 18.0  # same as in fitting.py
TZ_NAME = "America/New_York"


def get_today_str():
    """Return today's date string in America/New_York, format YYYY-MM-DD."""
    tz = pytz.timezone(TZ_NAME)
    now = datetime.now(tz)
    return now.strftime("%Y-%m-%d"), now


def load_zones_from_models():
    """
    Find all <ZONE>_params.npy files in MODELS_DIR and return the list of zones.
    Example files: AECO_params.npy, AEPIMP_params.npy, ...
    """
    pattern = os.path.join(MODELS_DIR, "*_params.npy")
    param_paths = sorted(glob.glob(pattern))
    zones = []

    for path in param_paths:
        fname = os.path.basename(path)
        # AECO_params.npy -> AECO
        if fname.endswith("_params.npy"):
            zone = fname[:-len("_params.npy")]
            zones.append(zone)

    return zones


def prepare_training_data():
    """
    Load merged_all_years.csv and reproduce the feature engineering from fitting.py.
    Returns:
        df: preprocessed DataFrame with columns:
            ['datetime_beginning_ept', 'load_area', 'mw', 'CDH', 'HDH', dow_* ...]
        exog_cols: list of exogenous column names used in the model.
    """
    if not os.path.exists(OUTPUT_PATH):
        raise FileNotFoundError(
            f"{OUTPUT_PATH} not found. Make sure load_data.py has created it."
        )

    # Load merged data
    df = pd.read_csv(OUTPUT_PATH, parse_dates=[0])

    # Ensure we have a proper datetime column (same logic as fitting.py)
    if "datetime_beginning_ept" in df.columns:
        dt = pd.to_datetime(df["datetime_beginning_ept"])
    else:
        dt = pd.to_datetime(df.iloc[:, 0])
        df.insert(0, "datetime_beginning_ept", dt)

    needed_cols = ["datetime_beginning_ept", "load_area", "mw", "temperature_2m"]
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in merged_all_years.csv: {missing}")

    df = df[needed_cols].copy()
    df["datetime_beginning_ept"] = pd.to_datetime(df["datetime_beginning_ept"])

    # Day of week dummies
    df["dow"] = df["datetime_beginning_ept"].dt.dayofweek  # 0=Mon,...,6=Sun
    df = pd.get_dummies(df, columns=["dow"], prefix="dow", dtype=float, drop_first=True)

    # Drop AE and RTO regions (as in fitting.py)
    df = df[~df["load_area"].isin(["AE", "RTO"])]

    # Temperature-based features
    temp = df["temperature_2m"]
    df["CDH"] = (temp - TBASE).clip(lower=0)
    df["HDH"] = (TBASE - temp).clip(lower=0)

    # Restrict to data from 2025 onward (same as in your current fitting.py)
    df = df[df["datetime_beginning_ept"].dt.year >= 2025].copy()
    if df.empty:
        raise ValueError("No data from 2025 onward found in merged_all_years.csv.")

    # Exogenous columns
    exog_cols = ["CDH", "HDH", "dow_1", "dow_2", "dow_3", "dow_4", "dow_5", "dow_6"]

    # Ensure all exog columns exist (some dow_* may be missing)
    for col in exog_cols:
        if col not in df.columns:
            df[col] = 0.0

    return df, exog_cols


def build_exog_from_weather(weather_df, today_local):
    """
    Given a per-zone live weather DataFrame for 'today', build a 96x8 exog matrix
    for SARIMAX forecasting.

    Assumes weather_df contains 24 rows for one day and columns:
      - temp_at_time_t
      - temp_at_time_t_minus_24h
      - temp_at_time_t_minus_48h
      - temp_at_time_t_plus_24h_FORECAST
    (Created in load_data.py)

    We unroll these 4 days into a single 96-long temperature vector, then
    construct CDH, HDH, and DOW dummies just like in fitting.py.
    """
    # Make sure time is sorted, though it should already be
    if "time" in weather_df.columns:
        weather_df["time"] = pd.to_datetime(weather_df["time"])
        weather_df = weather_df.sort_values("time").reset_index(drop=True)

    temp_cols = [
        "temp_at_time_t_minus_48h",
        "temp_at_time_t_minus_24h",
        "temp_at_time_t",
        "temp_at_time_t_plus_24h_FORECAST",
    ]

    missing = [c for c in temp_cols if c not in weather_df.columns]
    if missing:
        raise ValueError(f"Weather data missing expected columns: {missing}")

    # 4 days x 24 hours = 96 temperatures
    temps = weather_df[temp_cols].to_numpy().T.flatten()  # shape (96,)
    df_temp = pd.DataFrame({"temp": temps})

    # Start time = 00:00 two days before today (local time), naive
    today_00 = today_local.replace(hour=0, minute=0, second=0, microsecond=0)
    start_time = today_00 - timedelta(days=2)
    start_time = start_time.replace(tzinfo=None)

    df_temp["datetime"] = pd.date_range(start=start_time, periods=len(df_temp), freq="h")

    # CDH / HDH
    df_temp["CDH"] = (df_temp["temp"] - TBASE).clip(lower=0)
    df_temp["HDH"] = (TBASE - df_temp["temp"]).clip(lower=0)

    # Day-of-week dummies
    df_temp["dow"] = df_temp["datetime"].dt.dayofweek
    df_temp = pd.get_dummies(df_temp, columns=["dow"], prefix="dow", dtype=float, drop_first=True)

    # Ensure all required exog columns exist
    exog_cols = ["CDH", "HDH", "dow_1", "dow_2", "dow_3", "dow_4", "dow_5", "dow_6"]
    for col in exog_cols:
        if col not in df_temp.columns:
            df_temp[col] = 0.0

    exog_future = df_temp[exog_cols]
    return exog_future


def main():
    today_str, now_local = get_today_str()

    zones = load_zones_from_models()
    if not zones:
        # No parameter files -> nothing to predict
        return

    # Prepare the training data once (same preprocessing as fitting.py)
    df_train, exog_cols = prepare_training_data()

    all_zone_daily_loads = []  # list of 24-length lists
    all_zone_peak_hours = []   # list of integers in [0, 23]

    for zone in zones:
        param_path = os.path.join(MODELS_DIR, f"{zone}_params.npy")
        temp_path = os.path.join(TEMP_DIR, f"live_weather_{zone}.csv")

        if not os.path.exists(param_path):
            raise FileNotFoundError(f"Parameter file not found for zone {zone}: {param_path}")
        if not os.path.exists(temp_path):
            raise FileNotFoundError(f"Temperature file not found for zone {zone}: {temp_path}")

        # Load parameter vector
        params = np.load(param_path)

        # Subset training data for this zone
        df_zone = df_train[df_train["load_area"] == zone].copy()
        if df_zone.empty:
            raise ValueError(f"No training data found in merged_all_years.csv for zone {zone}.")

        y_train = df_zone["mw"]
        exog_train = df_zone[exog_cols]

        # Build the SARIMAX model and filter with fixed parameters
        model = SARIMAX(
            y_train,
            order=(1, 0, 0),
            seasonal_order=(1, 1, 1, 24),  # daily seasonal pattern
            exog=exog_train,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        results = model.filter(params)

        # Load live weather data produced by load_data.py
        weather_df = pd.read_csv(temp_path)

        # Build exogenous matrix for the next 96 hours
        exog_future = build_exog_from_weather(weather_df, now_local)
        steps = len(exog_future)  # should be 96

        # Forecast using the filtered model state
        forecast_res = results.get_forecast(steps=steps, exog=exog_future)
        mean_forecast = forecast_res.predicted_mean

        # Last 24 points = predicted loads for the "last day"
        last_24 = np.asarray(mean_forecast[-24:])

        # Round to nearest integer
        daily_loads_int = np.rint(last_24).astype(int).tolist()
        all_zone_daily_loads.append(daily_loads_int)

        # Peak hour: index of max within that last day, 0..23
        peak_hour = int(last_24.argmax())
        all_zone_peak_hours.append(peak_hour)

    # ------------------------------------------------------------------
    # Build output line:
    # "YYYY-MM-DD", L1_00, ..., L1_23, L2_00, ..., L29_23, PH_1, ..., PH_29
    # ------------------------------------------------------------------
    fields = []
    fields.append(f'"{today_str}"')

    # Flatten zone loads in order: zone1's 24, then zone2's 24, ...
    for loads in all_zone_daily_loads:
        fields.extend(str(int(x)) for x in loads)

    # Append peak hours PH_1..PH_n
    for ph in all_zone_peak_hours:
        fields.append(str(int(ph)))

    # Print a single CSV line with NO extra output
    print(",".join(fields))


if __name__ == "__main__":
    main()
