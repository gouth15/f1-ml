import pandas as pd
import numpy as np
from pathlib import Path

# LOAD THE DATA CSV
CSV_PATH = Path("data.csv")
data = pd.read_csv(CSV_PATH)

# REMOVE FP1 ROOKIE SESSION
DRIVERS = [1, 4, 5, 6, 10, 12, 14, 16, 18, 22, 23, 27, 30, 31, 44, 55, 63, 81, 87, 43]
data = data[data["DriverNumber"].isin(DRIVERS)]

# FILTER THE COLUMNS THAT USED FOR TRAINING
INCLUDE= [
    "DriverNumber", "LapTime", "SpeedI1", "SpeedI2",
    "SpeedFL", "Compound", "TyreLife", "FreshTyre", "TrackStatus",
    "TrackLength"]
data = data[INCLUDE]

# REMOVE NA ROWS
data = data.dropna()

# CONVERT THE DTYPES 
COMPOUND_MAPPING = { 
    "SOFT": 1, "MEDIUM": 2, "HARD": 3,
    "INTERMEDIATE": 4, "WET": 5 
}
data["LapTime"] = data["LapTime"].apply(lambda x: pd.to_timedelta(x).total_seconds())
data["Compound"] = data["Compound"].apply(lambda x: COMPOUND_MAPPING[x])
data["FreshTyre"] = data["FreshTyre"].apply(lambda x: 1 if x else 0)

# ADD SOME USEFUL DATA
data["AveragePace"] = data["TrackLength"] / (data["LapTime"] / 3600 )

# SAVE THE PREPROCESSED DATA
data.to_csv("new_data.csv", index=False)