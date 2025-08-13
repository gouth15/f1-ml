from pathlib import Path
from fastf1 import Cache
import fastf1
import pandas as pd
from datetime import datetime
import json

class F1DataFetcher:
    def __init__(self, csv_path="f1.csv", track_json="track_lenghts.json"):
        self.cache_directory = Path("cache")
        self.cache_directory.mkdir(exist_ok=True)
        Cache.enable_cache(cache_dir=str(self.cache_directory))
        self.current_year = datetime.today().year
        self.csv_path = Path(csv_path)
        with open(track_json, "r") as f:
            self.track_lengths = json.load(f)

    def fetch_and_store(self):
        season = pd.DataFrame(fastf1.get_event_schedule(self.current_year))
        today_date = pd.Timestamp(datetime.now()).normalize()
        season["Session5DateUtc"] = pd.to_datetime(season["Session5DateUtc"]).dt.tz_localize(None)
        completed_races = season[season["Session5DateUtc"] <= today_date]["Location"].to_list()
        sessions = ["Sprint", "Race"]

        for country in completed_races:
            for session in sessions:
                try:
                    current_session = fastf1.get_session(self.current_year, country, session)
                    current_session.load(laps=True, weather=False, messages=False, telemetry=False)
                    is_header = not self.csv_path.exists()
                    data = pd.DataFrame(current_session.laps.pick_quicklaps().pick_accurate())
                    data["TrackLength"] = self.track_lengths[country]
                    data.to_csv(self.csv_path, mode="a", header=is_header, index=False)
                except ValueError:
                    continue
