from pathlib import Path
import fastf1
from datetime import datetime
import pandas as pd
import fastf1.logger as logger
from logging import ERROR
import numpy as np

logger.set_log_level(ERROR)

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok = True)
fastf1.Cache.enable_cache(cache_dir = str(CACHE_DIR))

CURRENT_YEAR = datetime.today().year

CSV_PATH = Path("data.csv")

current_season = fastf1.get_event_schedule(CURRENT_YEAR)
CURRENT_DATE = pd.Timestamp(datetime.now()).normalize()
current_season["Session5DateUtc"] = pd.to_datetime(current_season["Session5DateUtc"]).dt.tz_localize(None)

print(current_season["Location"])

completed_races = current_season[current_season["Session5DateUtc"] <= CURRENT_DATE]["Location"].to_list()

SESSIONS = ["Sprint", "Race", "Practice 1", "Practice 2", "Practice 3"]

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



data = pd.DataFrame([])

for location in completed_races:
    for session in SESSIONS:
        try:
            current_session = fastf1.get_session(CURRENT_YEAR, location, session)
            current_session.load(weather=False, messages=False, telemetry=False)
            is_header = not CSV_PATH.exists()
            current_data = pd.DataFrame(current_session.laps.pick_quicklaps().pick_accurate())
            current_data["TrackLength"] = TRACK_LENGTHS[location]
            data = pd.concat([data, current_data])
        except ValueError:
            print("Not a sprint weekend. no sprint data")
            continue

print(data.shape)

DRIVERS = [1, 4, 5, 6, 10, 12, 14, 16, 18, 22, 23, 27, 30, 31, 44, 55, 63, 81, 87, 43]

data["DriverNumber"] = data["DriverNumber"].astype(int)

data = data[data["DriverNumber"].isin(DRIVERS)]

print(data.shape)




ROWS_TO_EXCLUDE = ["Time", "Driver", "LapNumber", 
                  "Stint", "Sector1Time", "Sector2Time", "Sector3Time",
                "Sector1SessionTime", "Sector2SessionTime",
                "Sector3SessionTime", "IsPersonalBest", 
                "Team", "LapStartTime", "Position", 
                "FastF1Generated", "IsAccurate", 
                "PitOutTime", "PitInTime", 
                "LapStartDate", "Deleted", "DeletedReason"]



data = data.drop(columns = ROWS_TO_EXCLUDE)

print(data.shape)

data = data.dropna()

print(data.shape)


COMPOUND_MAPPING = {
    "SOFT": 1, "MEDIUM": 2, "HARD": 3,
    "INTERMEDIATE": 4, "WET": 5
}

data["Compound"] = data["Compound"].apply(lambda x: COMPOUND_MAPPING[x])

data["LapTime"] = data["LapTime"].apply(lambda x: pd.to_timedelta(x).total_seconds())

data["AveragePace"] = data["TrackLength"] / (data["LapTime"] / 3600 )


data["TrackStatus"] = data["TrackStatus"].astype(int)


data.to_csv(CSV_PATH, index=False)



TARGET_COLUMN = "LapTime"
FEATURE_COLUMNS = data.columns.to_list()
FEATURE_COLUMNS.remove(TARGET_COLUMN)

print(FEATURE_COLUMNS)
print(TARGET_COLUMN)

print(data.dtypes)

X = data[FEATURE_COLUMNS]
Y = data[TARGET_COLUMN]

print(X)

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

# DATA_SPLIT = 0.3
# RANDOMENESS = 50
# ESTIMATORS = 250

training_metrics = []

# for split in range(20, 36, 5):
#     for random_state in range(30, 61, 5):
#         for estimator in range(150, 401, 50):

#             X_train, X_test, Y_train, Y_test = train_test_split(
#                 X, Y, test_size = split, random_state = random_state
#             )


#             decison_tree_model = GradientBoostingRegressor(
#                 n_estimators = estimator,
#                 random_state = random_state
#             )

#             decison_tree_model.fit(X_train, Y_train)

#             preds = decison_tree_model.predict(X_test)

#             rmse = root_mean_squared_error(Y_test, preds)
#             print(rmse)
#             training_metrics.append({
#                 "DataSplit": split,
#                 "RandomState": random_state,
#                 "Estimators": estimator,
#                 "RMSE": rmse
#             })


X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size = 0.3, random_state = 60
)


decison_tree_model = GradientBoostingRegressor(
    n_estimators = 400,
    random_state = 60
)

decison_tree_model.fit(X_train, Y_train)

preds = decison_tree_model.predict(X_test)

rmse = root_mean_squared_error(Y_test, preds)
print(rmse)
# training_metrics.append({
#     "DataSplit": split,
#     "RandomState": random_state,
#     "Estimators": estimator,
#     "RMSE": rmse
# })

import joblib
joblib.dump(decison_tree_model, "gd.pkl") 

# data = pd.DataFrame(training_metrics)

# data = data.sort_values("RMSE")

# data.to_csv("train_metrics.csv", index=False)



