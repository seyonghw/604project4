# fitting.py

import os
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def main():
    # Path inside the Docker container
    data_path = "/app/output/merged_all_years.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"{data_path} not found. Make sure load_data.py has already "
            "created the merged file before running fitting.py."
        )

    print(f"Loading data from {data_path} ...")
    # If your CSV has datetime as the first column / index, this is safe:
    df = pd.read_csv(data_path, parse_dates=[0])
    # Ensure we have a proper datetime column
    if "datetime_beginning_ept" in df.columns:
        dt = pd.to_datetime(df["datetime_beginning_ept"])
    else:
        # If datetime is stored as index / first column without a name
        dt = pd.to_datetime(df.iloc[:, 0])
        df.insert(0, "datetime_beginning_ept", dt)

    # Keep only needed columns
    needed_cols = ["datetime_beginning_ept", "load_area", "mw", "temperature_2m"]
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in merged_all_years.csv: {missing}")

    df = df[needed_cols].copy()
    df["datetime_beginning_ept"] = pd.to_datetime(df["datetime_beginning_ept"])

    # Day of week dummies
    df["dow"] = df["datetime_beginning_ept"].dt.dayofweek  # 0=Mon, ..., 6=Sun
    df = pd.get_dummies(df, columns=["dow"], prefix="dow", dtype=float, drop_first=True)

    # Drop AE and RTO regions
    df = df[~df["load_area"].isin(["AE", "RTO"])]

    # Temperature-based features
    Tbase = 18.0
    temp = df["temperature_2m"]
    df["CDH"] = (temp - Tbase).clip(lower=0)
    df["HDH"] = (Tbase - temp).clip(lower=0)

    # Restrict to data from 2023 onward
    df = df[df["datetime_beginning_ept"].dt.year >= 2023].copy()
    if df.empty:
        raise ValueError("No data from 2023 onward found in merged_all_years.csv.")

    # Exogenous columns (must match the dummies we created)
    exog_cols = ["CDH", "HDH", "dow_1", "dow_2", "dow_3", "dow_4", "dow_5", "dow_6"]

    # Some days may not appear → some dow_* columns may be missing
    for col in exog_cols:
        if col not in df.columns:
            df[col] = 0.0  # safe fallback if a particular day never appears

    # Prepare output directory
    models_dir = "/app/models"
    os.makedirs(models_dir, exist_ok=True)

    zones = sorted(df["load_area"].unique())
    print(f"Fitting SARIMAX models for {len(zones)} zones: {zones}")

    for zone in zones:
        print(f"\n=== Fitting zone: {zone} ===")
        df_zone = df[df["load_area"] == zone].copy()

        if df_zone.empty:
            print(f"  Skipping {zone}: no data from 2023 onward.")
            continue

        y = df_zone["mw"]
        exog = df_zone[exog_cols].copy()

        # Drop any rows with NA in y or exog (just in case)
        mask = y.notna()
        for col in exog_cols:
            mask &= exog[col].notna()

        y = y[mask]
        exog = exog.loc[mask]

        if len(y) < 100:  # arbitrary minimum length guardrail
            print(f"  Skipping {zone}: not enough data points ({len(y)}) to fit SARIMAX.")
            continue

        try:
            model = SARIMAX(
                y,
                order=(1, 0, 0),
                seasonal_order=(1, 1, 1, 24),  # daily seasonal pattern
                exog=exog,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            results = model.fit(disp=False)

            model_path = os.path.join(models_dir, f"sarimax_{zone}.pkl")
            results.save(model_path)
            print(f"  Saved model for {zone} to {model_path}")
        except Exception as e:
            print(f"  Error fitting {zone}: {e}")

    print("\nAll zone fitting attempts finished.")


if __name__ == "__main__":
    main()
