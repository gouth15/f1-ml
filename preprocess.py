from pathlib import Path
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib

class F1DataPreprocessor:
    def __init__(self, csv_path="new.csv", rows_json="row_to_include.json", compound_json="compound_map.json"):
        self.csv_path = Path(csv_path)
        with open(rows_json, "r") as f:
            self.rows_to_include = json.load(f)
        with open(compound_json, "r") as f:
            self.compound_map = json.load(f)
        self.encoders = {}
        self.scaler = None

    def convert_tyre_compound(self, compound: str):
        return self.compound_map.get(compound, None)

    def convert_timestamp(self, timestamp):
        return pd.to_timedelta(timestamp).total_seconds()

    def encode_column(self, data, column):
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        self.encoders[column] = le
        return data

    def save_encoders(self, path="encoders.pkl"):
        joblib.dump(self.encoders, path)

    def normalize_columns(self, data, exclude_columns=None):
        """Normalize numeric columns except those in exclude_columns."""
        if exclude_columns is None:
            exclude_columns = []

        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
        cols_to_scale = [col for col in numeric_cols if col not in exclude_columns]

        self.scaler = MinMaxScaler()
        data[cols_to_scale] = self.scaler.fit_transform(data[cols_to_scale])
        return data

    def preprocess(self, save_encoders_path="encoders.pkl", normalize=True, exclude_normalize=None):
        data = pd.read_csv(self.csv_path, usecols=self.rows_to_include).dropna()
        data["Compound"] = data["Compound"].apply(self.convert_tyre_compound)

        for column in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
            data[column] = data[column].apply(self.convert_timestamp)

        data["SpeedSTD"] = data[["SpeedI1", "SpeedI2", "SpeedFL", "SpeedST"]].std(axis=1)
        data["SpeedMean"] = data[["SpeedI1", "SpeedI2", "SpeedFL", "SpeedST"]].mean(axis=1)
        data["AveragePage"] = data["TrackLength"] / (data["LapTime"] / 3600)

        if normalize:
            data = self.normalize_columns(data, exclude_columns=exclude_normalize)

        data.to_csv(self.csv_path, index=False)
        self.save_encoders(save_encoders_path)
