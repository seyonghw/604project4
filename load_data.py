import pandas as pd
import glob
import os
import requests  # For making API calls
import json
from datetime import datetime
import time # To prevent API rate limiting
import pytz


# --- Configuration ---
# A map of PJM zones to their representative locations and timezones.
ZONE_LOCATIONS = {
    "AECO": {"lat": 39.36, "lon": -74.42, "timezone": "America/New_York"},   # Atlantic City Electric (Atlantic City, NJ)
    "AEPAPT": {"lat": 37.27, "lon": -79.94, "timezone": "America/New_York"}, # AEP Appalachian Power (Roanoke, VA)
    "AEPIMP": {"lat": 41.08, "lon": -85.14, "timezone": "America/Indiana/Indianapolis"}, # AEP Indiana Michigan Power (Fort Wayne, IN)
    "AEPKPT": {"lat": 38.48, "lon": -82.64, "timezone": "America/New_York"}, # AEP Kentucky Power (Ashland, KY)
    "AEPOPT": {"lat": 40.80, "lon": -81.38, "timezone": "America/New_York"}, # AEP Ohio Power (Canton, OH)
    "AP": {"lat": 40.30, "lon": -79.54, "timezone": "America/New_York"},     # Allegheny Power (Greensburg, PA)
    "BC": {"lat": 39.29, "lon": -76.61, "timezone": "America/New_York"},     # Alias for BGE (Baltimore, MD)
    "CE": {"lat": 41.50, "lon": -81.69, "timezone": "America/New_York"},     # Cleveland Electric Illuminating (Cleveland, OH)
    "DAY": {"lat": 39.76, "lon": -84.19, "timezone": "America/New_York"},    # Dayton Power and Light (Dayton, OH)
    "DEOK": {"lat": 39.10, "lon": -84.51, "timezone": "America/New_York"},   # Duke Energy Ohio/Kentucky (Cincinnati, OH)
    "DOM": {"lat": 37.54, "lon": -77.43, "timezone": "America/New_York"}, # Dominion Virginia Power (Richmond, VA)
    "DPLCO": {"lat": 39.75, "lon": -75.55, "timezone": "America/New_York"},  # Delmarva Power & Light (Wilmington, DE)
    "DUQ": {"lat": 40.44, "lon": -79.99, "timezone": "America/New_York"},    # Duquesne Light (Pittsburgh, PA)
    "EASTON": {"lat": 38.77, "lon": -76.08, "timezone": "America/New_York"}, # Easton Utilities (Easton, MD)
    "EKPC": {"lat": 37.99, "lon": -84.18, "timezone": "America/New_York"},   # East Kentucky Power Cooperative (Winchester, KY)
    "JC": {"lat": 40.80, "lon": -74.48, "timezone": "America/New_York"},     # Jersey Central Power & Light (Morristown, NJ)
    "ME": {"lat": 40.34, "lon": -75.93, "timezone": "America/New_York"},     # Metropolitan Edison (Reading, PA)
    "OE": {"lat": 41.08, "lon": -81.52, "timezone": "America/New_York"},     # Ohio Edison (Akron, OH)
    "OVEC": {"lat": 39.06, "lon": -83.01, "timezone": "America/New_York"},   # Ohio Valley Electric Corporation (Piketon, OH)
    "PAPWR": {"lat": 40.61, "lon": -75.49, "timezone": "America/New_York"},  # PPL (Pennsylvania Power & Light) (Allentown, PA)
    "PE": {"lat": 39.95, "lon": -75.16, "timezone": "America/New_York"},     # Alias for PECO (Philadelphia, PA)
    "PEPCO": {"lat": 38.91, "lon": -77.04, "timezone": "America/New_York"},  # Potomac Electric Power Company (Washington, D.C.)
    "PLCO": {"lat": 40.61, "lon": -75.49, "timezone": "America/New_York"},   # PPL Electric Utilities (Allentown, PA) - Alias for PAPWR
    "PN": {"lat": 42.13, "lon": -80.09, "timezone": "America/New_York"},     # Penelec (Pennsylvania Electric Co) (Erie, PA)
    "PS": {"lat": 40.73, "lon": -74.17, "timezone": "America/New_York"},     # Public Service Electric and Gas (Newark, NJ) - Alias for PSEG
    "RECO": {"lat": 41.09, "lon": -74.05, "timezone": "America/New_York"},   # Rockland Electric Company (Spring Valley, NY)
    "SMECO": {"lat": 38.52, "lon": -76.80, "timezone": "America/New_York"},  # Southern Maryland Electric Cooperative (Hughesville, MD)
    "UGI": {"lat": 41.25, "lon": -75.88, "timezone": "America/New_York"},    # UGI Utilities (Wilkes-Barre, PA)
    "VMEU": {"lat": 39.49, "lon": -75.03, "timezone": "America/New_York"}    # Vineland Municipal Electric Utility (Vineland, NJ)
}
# --- End of Configuration ---

def find_zone_column(df):
    """Helper function to find the zone column, case-insensitive."""
    possible_names = ['zone', 'ZONE', 'Area', 'AREA', 'region', 'REGION']
    for col in df.columns:
        if col in possible_names:
            return col
    return None

# --- 1. Loading CSV data from OSF file ---
print("--- 1. Loading CSV data from OSF file ---")
DATA_DIR = '/app/data'
print(f"Searching for CSV files in {DATA_DIR} and its subdirectories...")

search_path = os.path.join(DATA_DIR, '**', '*.csv')
csv_files = glob.glob(search_path, recursive=True)

if not csv_files:
    print(f"Error: No CSV files found in {DATA_DIR}.")
    dataframes = {}
else:
    print(f"Found {len(csv_files)} CSV files. Loading each one...")
    dataframes = {}
    all_dates = []  # To store all dates from all files

    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        try:
            df = pd.read_csv(file_path)
            
            # --- This is the new parsing logic ---
            # Get the first column name
            date_col = df.columns[1]
            # Parse the date column. 
            # `pd.to_datetime` is smart enough to handle "1/1/2016 5:00:00 AM"
            parsed_dates = pd.to_datetime(df[date_col], errors='coerce')
            
            # Store the parsed dates and the DataFrame
            all_dates.append(parsed_dates)
            dataframes[file_name] = df
            
            print(f"  - Successfully loaded and parsed dates from {file_name}")
        except Exception as e:
            print(f"Could not load {file_name}: {e}")

    print(f"\nSuccessfully loaded {len(dataframes)} DataFrames.")

# --- 2. Fetching Weather Data for CSV Date Range ---
if not dataframes:
    print("\nNo dataframes loaded, skipping weather fetch.")
else:
    print("\n--- 2. Fetching Weather Data for CSV Date Range ---")
    
    # Combine all dates from all files to find the min and max
    all_dates_combined = pd.concat(all_dates).dropna()
    min_date = all_dates_combined.min()
    max_date = all_dates_combined.max()
    
    # Format dates for the API (YYYY-MM-DD)
    start_str = min_date.strftime('%Y-%m-%d')
    end_str = max_date.strftime('%Y-%m-%d')
    
    print(f"Date range found in CSVs: {start_str} to {end_str}")
    print("Fetching corresponding weather data from Open-Meteo...")

    API_URL = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 39.95,  # Philadelphia
        "longitude": -75.16, # Philadelphia
        "start_date": start_str,
        "end_date": end_str,
        "hourly": "temperature_2m",
        "timezone": "America/New_York"
    }

    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status()  # Raises an error for bad responses (4xx or 5xx)
        weather_data = response.json()
        print("Successfully fetched and parsed weather data.")

        # --- 3. Process and Merge Weather Data ---
        print("\n--- 3. Processing and Merging Weather Data ---")

        # Load weather data into a DataFrame
        hourly_data = weather_data.get('hourly', {})
        if not hourly_data:
            print("Error: No 'hourly' data found in API response.")
        else:
            # Create a clean, time-indexed DataFrame for the weather
            weather_df = pd.DataFrame(hourly_data)
            weather_df['time'] = pd.to_datetime(weather_df['time'])
            weather_df = weather_df.set_index('time')
            print("Created weather DataFrame. Head:")
            print(weather_df.head())

            # Now, let's merge this weather data back into your original CSVs
            # We'll just show an example using the first DataFrame
            
            #first_df_key = list(dataframes.keys())[0]
            #print(f"\nExample merge with first file: '{first_df_key}'")
            
            #original_df = dataframes[first_df_key]
            #date_col = original_df.columns[0]
            
            # Parse dates and set as index (this time on the original df)
            #original_df[date_col] = pd.to_datetime(original_df[date_col])
            #original_df = original_df.set_index(date_col)
            
            # Merge! This joins the temperature to the matching timestamp.
            # 'how=left' keeps all your original data.
            #merged_df = original_df.merge(weather_df, left_index=True, right_index=True, how='left')
            
            #print("Merged DataFrame (head):")
            #print(merged_df.head())

            # Save merged_df locally (to a mounted folder)
            #OUTPUT_DIR = "/app/output"
            #os.makedirs(OUTPUT_DIR, exist_ok=True)

            #output_path = os.path.join(OUTPUT_DIR, f"merged_{first_df_key}")
            #merged_df.to_csv(output_path)

            #print(f"\nMerged data saved to: {output_path}")

            first_file = list(dataframes.keys())[0]
            date_col = dataframes[first_file].columns[1]

            # 1. Concatenate all load dataframes
            df_list = []
            for fname, df in dataframes.items():
                df = df.copy()
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                df_list.append(df)

            combined_df = pd.concat(df_list, ignore_index=True)
            combined_df = combined_df.dropna(subset=[date_col])
            combined_df = combined_df.set_index(date_col).sort_index()

            # 2. Merge with weather_df
            merged_df = combined_df.merge(weather_df, left_index=True, right_index=True, how='left')

            print("Merged full DataFrame (head):")
            print(merged_df.head())

            # 3. Save to disk
            OUTPUT_DIR = "/app/output"
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            output_path = os.path.join(OUTPUT_DIR, "merged_all_years.csv")
            merged_df.to_csv(output_path)
            print(f"\nMerged data (all years) saved to: {output_path}")

            


    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
    except json.JSONDecodeError:
        print("Error: Could not decode JSON response from weather API.")
    except Exception as e:
        print(f"An error occurred during weather processing: {e}")


        # --- 4. Fetching "Live" Data for New Predictions (Per Zone) ---
    print("\n--- 4. Fetching 'Live' Data for New Predictions (Per Zone) ---")

    TEMP_DIR = "/app/temperature"
    os.makedirs(TEMP_DIR, exist_ok=True)

    FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"

    for zone, info in ZONE_LOCATIONS.items():
        print(f"\nFetching live forecast for zone: {zone} ...")

        forecast_params = {
            "latitude": info["lat"],
            "longitude": info["lon"],
            "hourly": "temperature_2m",
            "past_days": 2,       # use past 2 days
            "forecast_days": 2,   # today + tomorrow
            "timezone": info["timezone"]
        }

        try:
            response = requests.get(FORECAST_API_URL, params=forecast_params)
            response.raise_for_status()
            live_data = response.json()

            hourly_live = live_data.get("hourly", {})
            if not hourly_live or "time" not in hourly_live:
                print(f"  ...No 'hourly' data returned for {zone}. Skipping.")
                continue

            # Build DataFrame
            live_df = pd.DataFrame(hourly_live)
            live_df["time"] = pd.to_datetime(live_df["time"])
            live_df["zone"] = zone
            live_df = live_df.set_index("time")

            # Rename and create lag/lead features
            live_df = live_df.rename(columns={"temperature_2m": "temp_at_time_t"})
            live_df["temp_at_time_t_minus_24h"] = live_df["temp_at_time_t"].shift(24)
            live_df["temp_at_time_t_minus_48h"] = live_df["temp_at_time_t"].shift(48)
            live_df["temp_at_time_t_plus_24h_FORECAST"] = live_df["temp_at_time_t"].shift(-24)

            # Get today's date string in the zone's local timezone
            tz = pytz.timezone(info["timezone"])
            today_str = datetime.now(tz).strftime("%Y-%m-%d")

            if today_str not in live_df.index.strftime("%Y-%m-%d"):
                print(f"  ...No rows for today ({today_str}) in index for {zone}. Saving full data instead.")
                zone_df = live_df.copy()
            else:
                # Filter to only today's 24 hours
                zone_df = live_df.loc[today_str].copy()

            # Save per-zone CSV: live_weather_<ZONE>.csv
            out_path = os.path.join(TEMP_DIR, f"live_weather_{zone}.csv")
            zone_df.to_csv(out_path)
            print(f"  Saved live weather for {zone} to: {out_path}")

            time.sleep(1)  # be nice to the API

        except requests.exceptions.RequestException as e:
            print(f"  ...Error fetching live forecast for {zone}: {e}")
        except Exception as e:
            print(f"  ...Unexpected error for {zone}: {e}")


print("\n--- Python script execution finished. ---")
