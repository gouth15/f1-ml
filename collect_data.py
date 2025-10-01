# IMPORT NECESSARY LIBRARIES
from pathlib import Path
import fastf1
from datetime import datetime
import pandas as pd
import json
from fastf1.logger import set_log_level
from logging import ERROR
from rich.progress import track
import logging

def collect_f1_data():
    """Collect F1 race data from FastF1 API and save to data.csv"""
    logger = logging.getLogger(__name__)
    
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
    
    # CREATE LOCATION MAPPING FOR MISMATCHED NAMES
    LOCATION_MAPPING = {
        "Miami Gardens": "Miami",
        # Add other mappings as needed
    }

    # LOAD CURRENT SEASON RACE LOCATIONS
    current_season = fastf1.get_event_schedule(CURRENT_YEAR)
    current_season["Session5DateUtc"] = pd.to_datetime(current_season["Session5DateUtc"]).dt.tz_localize(None)
    CURRENT_DATE = pd.Timestamp(datetime.now()).normalize()
    COMPLETED_RACES = current_season[current_season["Session5DateUtc"] <= CURRENT_DATE]["Location"].to_list()
    print(COMPLETED_RACES)

    logger.info(f"Found {len(COMPLETED_RACES)} completed races for {CURRENT_YEAR}")
    logger.info(f"Races: {', '.join(COMPLETED_RACES)}")

    # INITIALIZE DATAFRAME
    data = pd.DataFrame()

    # COLLECT LAP DATA WITH RICH PROGRESS
    for location in track(COMPLETED_RACES, description="[bold green]Fetching races..."):
        logger.info(f"Processing {location}...")
        
        # Map location name if needed
        mapped_location = LOCATION_MAPPING.get(location, location)
        
        # Check if track details exist for this location
        if mapped_location not in TRACK_DETAILS:
            logger.warning(f"Track details not found for '{location}' (mapped to '{mapped_location}'). Skipping this location.")
            continue
        
        for session in SESSIONS:
            try:
                current_session = fastf1.get_session(CURRENT_YEAR, location, session)
                current_session.load(messages=False, telemetry=False)
                
                # Load weather and lap data
                weather_data = pd.DataFrame(current_session.weather_data).sort_values("Time")
                lap_data = pd.DataFrame(current_session.laps.pick_quicklaps().pick_accurate()).sort_values("Time")
                
                # Add track info using mapped location
                lap_data["TrackLength"] = TRACK_DETAILS[mapped_location]["value"]
                lap_data["Location"] = location
                lap_data["Compound"] = lap_data.apply(
                    lambda row: TRACK_DETAILS[mapped_location]["tires"].get(row["Compound"], None), axis=1
                )
                
                # Merge weather info
                lap_data = pd.merge_asof(lap_data, weather_data, on="Time", direction="backward")
                data = pd.concat([data, lap_data], ignore_index=True)
                logger.debug(f"  {session}: {len(lap_data)} laps collected")
            except ValueError as e:
                logger.debug(f"  {session}: Skipped - {e}")
                continue
            except Exception as e:
                logger.warning(f"  {session}: Error processing session - {e}")
                continue

    logger.info(f"Raw data collected: {data.shape[0]} rows")

    # CONVERT TIMEDelta TO SECONDS
    for col in data.columns:
        if pd.api.types.is_timedelta64_ns_dtype(data[col]):
            data[col] = data[col].dt.total_seconds()

    # DROP UNUSED COLUMNS
    drop_cols = [
        "PitOutTime", "PitInTime", "LapStartDate", "Deleted", "DeletedReason", "FastF1Generated", "IsAccurate",
        "Driver", "Team", "Location", "Sector1SessionTime", "Sector2SessionTime", "Sector3SessionTime", "Time",
        "LapStartTime", "Stint", "LapNumber", "Position", "IsPersonalBest", "TrackStatus", "FreshTyre", "Rainfall",
        "WindSpeed", "Pressure", "AirTemp", "WindDirection", "SpeedI2" 
    ]
    data = data.drop(columns=drop_cols)

    # TYPE CASTING
    data[["SpeedI1", "SpeedST"]] = data[["SpeedI1", "SpeedST"]].astype("Int64")
    data["TyreLife"] = data["TyreLife"].astype("Int64")
    data["DriverNumber"] = data["DriverNumber"].astype(int)
    data["TrackLength"] = data["TrackLength"].astype(float)
    data["TrackTemp"] = data["TrackTemp"].astype(float)
    data["Humidity"] = data["Humidity"].astype(float)

    # REMOVE MISSING VALUES
    data_before = len(data)
    data = data.dropna()
    logger.info(f"Removed {data_before - len(data)} rows with missing values")

    # REMOVE FP1 ROOKIE SESSIONS
    DRIVERS = [1, 4, 5, 6, 10, 12, 14, 16, 18, 22, 23, 27, 30, 31, 44, 55, 63, 81, 87, 43, 7]
    data = data[data["DriverNumber"].isin(DRIVERS)]
    logger.info(f"Filtered to {len(DRIVERS)} regular drivers")

    # SAVE CLEANED DATA
    data.to_csv("data.csv", index=False)
    logger.info(f"Saved cleaned data: {data.shape[0]} rows and {data.shape[1]} columns")
    
    return data

if __name__ == "__main__":
    # Setup basic logging for standalone execution
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    collect_f1_data()
