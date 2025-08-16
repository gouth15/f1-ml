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
TRACK_LENGTHS = {
    "Melbourne": 5.3030, "Shanghai": 5.4510, "Suzuka": 5.8070, "Sakhir": 5.4120, "Jeddah": 6.1740, "Miami": 5.4120,
    "Imola": 4.9090, "Monaco": 3.3370, "Barcelona": 4.6570, "Montréal": 4.3610, "Spielberg": 4.3180, "Silverstone": 5.8910,
    "Spa-Francorchamps": 7.0040, "Budapest": 4.3810, "Zandvoort": 4.2590, "Monza": 5.7930, "Baku": 6.0030, "Marina Bay": 4.9400,
    "Austin": 5.5130, "Mexico City": 4.3040, "São Paulo": 4.3090, "Las Vegas": 6.2010, "Lusail": 5.3800, "Yas Island": 5.2810
}

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
            current_data["TrackLength"] = TRACK_LENGTHS[location]
            data = pd.concat([data, current_data])
        except ValueError:
            continue

# SAVE THE COLLECTED DATA
data.to_csv(CSV_PATH, index=False)

print(f"Colleceted data of {data.shape[0]} rows and {data.shape[1]} columns")

