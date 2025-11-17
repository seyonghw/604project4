# prediction.py
import os
import glob
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz
from statsmodels.tsa.statespace.sarimax import SARIMAXResults


MODELS_DIR = "/app/models"
TEMP_DIR = "/app/temperature"

TBASE = 18.0  # same as in fitting.py
TZ_NAME = "America/New_York"


def get_today_str():
    """Return today's date string in America/New_York, format YYYY-MM-DD."""
    tz = pytz.timezone(TZ_NAME)
    now = datetime.now(tz)
    return now.strftime("%Y-%m-%d"), now


def load_zones_from_models():
    """
    Find all sarimax_*.pkl files in MODELS_DIR and return:
    - zones: list of zone codes in a consistent order
    - model_paths: matching list of full paths
    """
    pattern = os.path.join(MODELS_DIR, "sarimax_*.pkl")
    model_paths = sorted(glob.glob(pattern))
    zones = []

    for path in model_paths:
        fname = os.path.basename(path)
        # sarimax_AECO.pkl -> AECO
        if fname.startswith("sarimax_") and fname.endswith(".pkl"):
            zone = fname[len("sarimax_"):-len(".pkl")]
            zones.append(zone)

    return zones, model_paths


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

    # These are the 4 day-wise temperature columns (past 2 days, today, tomorrow-type)
    temp_cols = [
        "temp_at_time_t_minus_48h",
        "temp_at_time_t_minus_24h",
        "temp_at_time_t",
        "temp_at_time_t_plus_24h_FORECAST",
    ]

    missing = [c for c in temp_cols if c not in weather_df.columns]
    if missing:
        raise ValueError(f"Weather data missing expected columns: {missing}")

    # weather_df has shape (24, ...). We want 4 days x 24 hours = 96 temperatures.
    # We interpret the 4 columns as 4 consecutive days and unroll them:
    # Day1(all 24 hours), Day2(all 24 hours), Day3(all 24 hours), Day4(all 24 hours)
    temps = weather_df[temp_cols].to_numpy().T.flatten()  # shape (96,)

    df_temp = pd.DataFrame({"temp": temps})

    # Start time = 00:00 two days before today (local time), but we'll use a naive timestamp
    today_00 = today_local.replace(hour=0, minute=0, second=0, microsecond=0)
    start_time = today_00 - timedelta(days=2)
    start_time = start_time.replace(tzinfo=None)  # make naive

    # Create a datetime column for 96 consecutive hours
    df_temp["datetime"] = pd.date_range(start=start_time, periods=len(df_temp), freq="h")

    # CDH / HDH (same logic as fitting.py, but using df_temp["temp"])
    df_temp["CDH"] = (df_temp["temp"] - TBASE).clip(lower=0)
    df_temp["HDH"] = (TBASE - df_temp["temp"]).clip(lower=0)

    # Day-of-week dummies
    df_temp["dow"] = df_temp["datetime"].dt.dayofweek
    df_temp = pd.get_dummies(df_temp, columns=["dow"], prefix="dow", dtype=float, drop_first=True)

    # Ensure all required dummy columns exist, as in fitting.py
    exog_cols = ["CDH", "HDH", "dow_1", "dow_2", "dow_3", "dow_4", "dow_5", "dow_6"]
    for col in exog_cols:
        if col not in df_temp.columns:
            df_temp[col] = 0.0

    exog_future = df_temp[exog_cols]
    return exog_future


def main():
    today_str, now_local = get_today_str()

    zones, model_paths = load_zones_from_models()
    if not zones:
        # No models -> nothing to predict
        return

    # We assume there should be 29 zones as per the project description,
    # but we just use whatever models are present.
    all_zone_daily_loads = []  # list of 24-length lists for each zone
    all_zone_peak_hours = []   # list of integers in [0, 23]

    for zone in zones:
        model_path = os.path.join(MODELS_DIR, f"sarimax_{zone}.pkl")
        temp_path = os.path.join(TEMP_DIR, f"live_weather_{zone}.csv")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(temp_path):
            raise FileNotFoundError(f"Temperature file not found for zone {zone}: {temp_path}")

        # Load model
        results = SARIMAXResults.load(model_path)

        # Load weather data produced by load_data.py
        weather_df = pd.read_csv(temp_path)

        # Build exogenous matrix for 96 hours
        exog_future = build_exog_from_weather(weather_df, now_local)
        steps = len(exog_future)  # should be 96

        # Forecast
        forecast_res = results.get_forecast(steps=steps, exog=exog_future)
        mean_forecast = forecast_res.predicted_mean

        # Last 24 points = predicted loads for the "last day"
        last_24 = np.asarray(mean_forecast[-24:])

        # Round to nearest integer, as required
        daily_loads_int = np.rint(last_24).astype(int).tolist()
        all_zone_daily_loads.append(daily_loads_int)

        # Peak hour: index of max within that last day, 0..23
        peak_hour = int(last_24.argmax())
        all_zone_peak_hours.append(peak_hour)

    # ------------------------------------------------------------------
    # Build output line:
    # "YYYY-MM-DD", L1_00, ..., L1_23, L2_00, ..., L29_23, PH_1, ..., PH_29
    # (no PDs, as requested)
    # ------------------------------------------------------------------
    fields = []
    fields.append(f'"{today_str}"')

    # Flatten zone loads in order: zone1's 24, then zone2's 24, ...
    for loads in all_zone_daily_loads:
        fields.extend(str(int(x)) for x in loads)

    # Then append peak hours PH_1..PH_29
    for ph in all_zone_peak_hours:
        fields.append(str(int(ph)))

    # Print a single CSV line with NO extra output
    print(",".join(fields))


if __name__ == "__main__":
    main()
