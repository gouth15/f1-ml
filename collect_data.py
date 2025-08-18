# IMPORT THE NEEDED PACKAGES AND LIBRARIES
from pathlib import Path
import fastf1
from datetime import datetime
import pandas as pd
import numpy as np
import json
import fastf1.logger
from logging import ERROR

# SET THE FASTF1 LOGGER TO ERROR
fastf1.logger.set_log_level(ERROR)

# SETUP CACHE FOLDER FOR THE DATA
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok = True)
fastf1.Cache.enable_cache(cache_dir = str(CACHE_DIR))

# ASSIGN ALL THE CONSTANT FIELDS
CURRENT_YEAR = datetime.today().year
CSV_PATH = Path("data.csv")
SESSIONS = ["Sprint", "Race", "Practice 1", "Practice 2", "Practice 3"]
with open("./compound_map.json", "r") as loader: TRACK_DETAILS = json.load(loader)

# LOAD THE RACE LOCATIONS OF CURRENT SEASON
current_season = fastf1.get_event_schedule(CURRENT_YEAR)
CURRENT_DATE = pd.Timestamp(datetime.now()).normalize()
current_season["Session5DateUtc"] = pd.to_datetime(current_season["Session5DateUtc"]).dt.tz_localize(None)
COMPLETED_RACES = current_season[current_season["Session5DateUtc"] <= CURRENT_DATE]["Location"].to_list()

# INITIALISE THE DATAFRAME
data = pd.DataFrame()

# GET ALL THE LAP DATA FOR EACH LOCATION UNDER EACH SESSIONS
for location in COMPLETED_RACES:
    for session in SESSIONS:
        try:
            current_session = fastf1.get_session(CURRENT_YEAR, location, session)
            current_session.load(weather=False, messages=False, telemetry=False)
            is_header = not CSV_PATH.exists()
            current_data = pd.DataFrame(current_session.laps.pick_quicklaps().pick_accurate())
            current_data["TrackLength"] = TRACK_DETAILS[location]["value"]
            current_data["Location"] = location
            data = pd.concat([data, current_data])
        except ValueError:
            continue

# SAVE THE COLLECTED DATA
data.to_csv(CSV_PATH, index=False)

print(f"Colleceted data of {data.shape[0]} rows and {data.shape[1]} columns")

