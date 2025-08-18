import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error
from pathlib import Path

# LOAD THE DATA
CSV_PATH = Path("new_data.csv")
data = pd.read_csv(CSV_PATH)

# SET THE FEATURE AND TARGET COLUMN
TARGET_COLUMN = "LapTime"
FEATURE_COLUMNS = data.columns.to_list()
FEATURE_COLUMNS.remove(TARGET_COLUMN)

X = data[FEATURE_COLUMNS]
Y = data[TARGET_COLUMN]

# SPLIT THE TRAIN AND TEST DATA SET
XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=0.3)

# DEFINE THE MODEL
model = GradientBoostingRegressor(
    n_estimators=500,
)

# FIT THE MODEL
model.fit(XTrain, YTrain)

# PREDICT AND EVALUATE THE ERROR
predictions = model.predict(XTest)
rmse = root_mean_squared_error(YTest, predictions)

# SAVE THE MODEL
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

