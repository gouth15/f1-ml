# IMPORT NECESSARY LIBRARIES
from pathlib import Path
import fastf1
from datetime import datetime
import pandas as pd
import json
from fastf1.logger import set_log_level
from logging import ERROR
from rich.progress import track

# SET LOGGER TO ERROR
set_log_level(ERROR)

# SETUP CACHE
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(cache_dir=str(CACHE_DIR))

# CONSTANTS
CURRENT_YEAR = datetime.today().year
SESSIONS = ["Sprint", "Race", "Practice 1", "Practice 2", "Practice 3"]

# LOAD TRACK DETAILS
with open("compound_map.json", "r") as f:
    TRACK_DETAILS = json.load(f)

# LOAD CURRENT SEASON RACE LOCATIONS
current_season = fastf1.get_event_schedule(CURRENT_YEAR)
current_season["Session5DateUtc"] = pd.to_datetime(current_season["Session5DateUtc"]).dt.tz_localize(None)
CURRENT_DATE = pd.Timestamp(datetime.now()).normalize()
COMPLETED_RACES = current_season[current_season["Session5DateUtc"] <= CURRENT_DATE]["Location"].to_list()

# INITIALIZE DATAFRAME
data = pd.DataFrame()

# COLLECT LAP DATA WITH RICH PROGRESS
for location in track(COMPLETED_RACES, description="[bold green]Fetching races..."):
    for session in SESSIONS:
        try:
            current_session = fastf1.get_session(CURRENT_YEAR, location, session)
            current_session.load(messages=False, telemetry=False)
            
            # Load weather and lap data
            weather_data = pd.DataFrame(current_session.weather_data).sort_values("Time")
            lap_data = pd.DataFrame(current_session.laps.pick_quicklaps().pick_accurate()).sort_values("Time")
            
            # Add track info
            lap_data["TrackLength"] = TRACK_DETAILS[location]["value"]
            lap_data["Location"] = location
            lap_data["Compound"] = lap_data.apply(
                lambda row: TRACK_DETAILS[row["Location"]]["tires"].get(row["Compound"], None), axis=1
            )
            
            # Merge weather info
            lap_data = pd.merge_asof(lap_data, weather_data, on="Time", direction="backward")
            data = pd.concat([data, lap_data], ignore_index=True)
        except ValueError:
            continue

# CONVERT TIMEDelta TO SECONDS
for col in data.columns:
    if pd.api.types.is_timedelta64_ns_dtype(data[col]):
        data[col] = data[col].dt.total_seconds()

# DROP UNUSED COLUMNS
drop_cols = [
    "Driver", "Team", "Location", "PitOutTime", "PitInTime", "LapStartDate",
    "Deleted", "DeletedReason", "Sector1SessionTime", "Sector2SessionTime",
    "Sector3SessionTime", "LapStartTime", "Position", "FastF1Generated",
    "IsAccurate", "TrackStatus", "IsPersonalBest", "LapNumber", "AirTemp",
    "Time", "SpeedFL", "Stint", "TrackTemp"
]
data = data.drop(columns=drop_cols)

# TYPE CASTING
data[["SpeedI1", "SpeedI2", "SpeedST"]] = data[["SpeedI1", "SpeedI2", "SpeedST"]].astype("Int64")
data["TyreLife"] = data["TyreLife"].astype("Int64")
data["DriverNumber"] = data["DriverNumber"].astype(int)
data["FreshTyre"] = data["FreshTyre"].astype(int)
data["TrackLength"] = data["TrackLength"].astype(float)
data["Humidity"] = data["Humidity"].astype(float)
data["Rainfall"] = data["Rainfall"].astype(int)

# REMOVE MISSING VALUES
data = data.dropna()

# REMOVE FP1 ROOKIE SESSIONS
DRIVERS = [1, 4, 5, 6, 10, 12, 14, 16, 18, 22, 23, 27, 30, 31, 44, 55, 63, 81, 87, 43, 7]
data = data[data["DriverNumber"].isin(DRIVERS)]

# SAVE CLEANED DATA
data.to_csv("data.csv", index=False)

print(f"Collected data of {data.shape[0]} rows and {data.shape[1]} columns")
