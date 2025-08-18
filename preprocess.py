import pandas as pd
import numpy as np
from pathlib import Path
import json

# LOAD THE DATA CSV
CSV_PATH = Path("data.csv")
data = pd.read_csv(CSV_PATH)

# REMOVE FP1 ROOKIE SESSION
DRIVERS = [1, 4, 5, 6, 10, 12, 14, 16, 18, 22, 23, 27, 30, 31, 44, 55, 63, 81, 87, 43]
data = data[data["DriverNumber"].isin(DRIVERS)]

# CONVERT THE DTYPES 
data["LapTime"] = data["LapTime"].apply(lambda x: pd.to_timedelta(x).total_seconds())
data["FreshTyre"] = data["FreshTyre"].apply(lambda x: 1 if x else 0)
with open("./compound_map.json", "r") as loader: TRACK_DETAILS = json.load(loader)
data["Compound"] = data.apply(
    lambda row: TRACK_DETAILS[row["Location"]]["tires"].get(row["Compound"], None),
    axis=1
)

# FILTER THE COLUMNS THAT USED FOR TRAINING
INCLUDE= [
    "DriverNumber", "LapTime", "SpeedI1", "SpeedI2",
    "SpeedFL", "Compound", "TyreLife", "FreshTyre", "TrackStatus",
    "TrackLength"]
data = data[INCLUDE]

# REMOVE NA ROWS
data = data.dropna()

# ADD SOME USEFUL DATA
data["AveragePace"] = data["TrackLength"] / (data["LapTime"] / 3600 )

# SAVE THE PREPROCESSED DATA
data.to_csv("new_data.csv", index=False)