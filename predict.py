import pandas as pd
from datetime import datetime

import fastf1   

fastf1.Cache.enable_cache(cache_dir = "cache")

location = "Budapest"
session = "Qualifying"

session = fastf1.get_session(datetime.today().year, location, session)
session.load()
results = session.results

results[["Q1", "Q2", "Q3"]] = (
    results[["Q1", "Q2", "Q3"]]
    .apply(lambda x: x.dt.total_seconds())
)

results["QualifyingTime"] = results[["Q3", "Q2", "Q1"]].bfill(axis=1).iloc[:, 0]

original_data = results[["DriverNumber", "ClassifiedPosition", "QualifyingTime"]]
original_data["DriverNumber"] = original_data["DriverNumber"].astype(int)


drivers = {
  "1": "VER", "4": "NOR", "5": "BOR", "6": "HAD", "7": "DOO", "10": "GAS", "12": "ANT",
  "14": "ALO", "16": "LEC", "18": "STR", "22": "TSU",  "23": "ALB", "27": "HUL", "30": "LAW",
  "31": "OCO", "44": "HAM",   "55": "SAI",  "63": "RUS", "81": "PIA", "87": "BEA", "43": "COL"
}


data = pd.read_csv("data.csv")

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
    "Budapest": 4.3810,
    "Zandvoort": 4.2590,
    "Monza": 5.7930,
    "Baku": 6.0030,
    "Marina Bay": 4.9400,
    "Austin": 5.5130,
    "Mexico City": 4.3040,
    "São Paulo": 4.3090,
    "Las Vegas": 6.2010,
    "Lusail": 5.3800,
    "Yas Island": 5.2810
}


driver_means = (
    data[data["TrackLength"] == TRACK_LENGTHS[location]]  # filter first
    .groupby("DriverNumber")
    .agg({
        "SpeedST": "mean",
        "SpeedFL": "mean",
        "SpeedI1": "mean",
        "SpeedI2": "mean",
        "AveragePace": "mean"
    })
    .reset_index()
)

driver_means["Compound"] = 1
driver_means["TyreLife"] = 1.0
driver_means["FreshTyre"] = True
driver_means["TrackStatus"] = 1
driver_means["TrackLength"] = 4.3810


driver_means = driver_means[
    ["DriverNumber", "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
     "Compound", "TyreLife", "FreshTyre", "TrackStatus", "TrackLength", "AveragePace"]
]

import joblib
model = joblib.load("gd.pkl")


driver_means["PredictedLapTime"] = model.predict(driver_means)

driver_means = driver_means.sort_values("PredictedLapTime")

driver_means["Leader"] = (
    driver_means["PredictedLapTime"] - driver_means["PredictedLapTime"].iloc[0]
).map("+{:.3f}".format)
driver_means["Driver"] = driver_means["DriverNumber"].apply(lambda x: drivers[str(x)])

aggragated_df = pd.merge(original_data, driver_means, on="DriverNumber", how="right")

aggragated_df["Error"] = (
    aggragated_df["QualifyingTime"] - aggragated_df["PredictedLapTime"]
)

aggragated_df["PredictedLapTime"] = driver_means["PredictedLapTime"].apply(lambda x: f"{int(x // 60)}:{(x % 60):06.3f}")

print(aggragated_df[["Driver", "PredictedLapTime", "Error"]].to_string(header=False, index=False))


aggragated_df.to_csv("QualiPred.csv", index=False)
