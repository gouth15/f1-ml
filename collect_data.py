# At first we need to import some packages which are tariff free
# ( I am looking at you Trump ) 
from pathlib import Path
from fastf1 import Cache
import fastf1
import pandas as pd
from datetime import datetime


# Oh bro. We need to cache the data otherwise it's gonna take time like Saubers pit stops
cache_directory = Path("cache")
cache_directory.mkdir(exist_ok = True)
Cache.enable_cache(cache_dir = cache_directory.__str__())

CURRENT_YEAR = datetime.today().year

# normalizing dates so we keep our sanity
season = pd.DataFrame(fastf1.get_event_schedule(CURRENT_YEAR))
today_date = pd.Timestamp(datetime.now()).normalize()
season["Session5DateUtc"] = pd.to_datetime(season["Session5DateUtc"]).dt.tz_localize(None)

completed_races = season[season["Session5DateUtc"] <= today_date]["Location"].to_list()

SESSIONS = ["Sprint", "Race"]

CSV_PATH = Path("f1.csv")

# Need to harcode this stuff. 
TRACK_LENGTHS = {
    "Melbourne": 5.3030,
    "Shanghai": 5.4510,
    "Suzuka": 5.8070,
    "Sakhir": 5.4120,
    "Jeddah": 6.1740,
    "Miami": 5.4120,
    "Imola": 4.9090,
    "Monaco": 3.3370,
    "Barcelona": 4.6570,
    "Montréal": 4.3610,
    "Spielberg": 4.3180,
    "Silverstone": 5.8910,
    "Spa-Francorchamps": 7.0040,
    "Budapest": 4.3810
}

for country in completed_races:
    for session in SESSIONS:
        try:
            current_session = fastf1.get_session(CURRENT_YEAR, country, session)
            # Choosing only laps data cause we dont want our brain to be cooked
            current_session.load(laps = True, weather = False, messages = False, telemetry = False)
            is_header = False if CSV_PATH.exists() else True
            data = pd.DataFrame(current_session.laps)
            data["Country"] = country
            data["Session"] = session
            data["TrackLength"] = TRACK_LENGTHS[country]
            start_fuel = 100.00 if session == "Race" else 33.33
            end_fuel = 2.00
            total_laps = current_session.total_laps
            for idx, lap in data.iterrows():
                # Just my approximate formula to calculate fuel percent in Race laps
                data.loc[idx, "FuelPercent"] = start_fuel - ((start_fuel - end_fuel) / total_laps) * (data.loc[idx, "LapNumber"] - 1)
            data.to_csv(CSV_PATH, mode="a", header=is_header, index=False)
        except ValueError: continue
