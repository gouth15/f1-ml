import pandas as pd
import fastf1
from datetime import datetime
import torch
from pathlib import Path
from model import NeuralNetwork 
from logging import ERROR
from rich.progress import track
import json
from fastf1.logger import set_log_level
from dataframe_image import export
import logging

def evaluate_model():
    """Evaluate trained model against actual qualifying results"""
    logger = logging.getLogger(__name__)
    
    # ---------------- SETTINGS ----------------
    CACHE_DIR = Path("cache")
    fastf1.Cache.enable_cache(cache_dir=str(CACHE_DIR))
    set_log_level(ERROR)

    CSV_PATH = Path("data.csv")
    MODEL_PATH = Path("./f1-model.pt")  # your saved PyTorch model
    TRACK_DETAILS_PATH = Path("./compound_map.json")

    # Check if required files exist
    if not CSV_PATH.exists():
        raise FileNotFoundError("data.csv not found. Please run data collection first.")
    if not MODEL_PATH.exists():
        raise FileNotFoundError("f1-model.pt not found. Please run model training first.")
    if not TRACK_DETAILS_PATH.exists():
        raise FileNotFoundError("compound_map.json not found.")

    # Load TRACK_DETAILS JSON
    with open(TRACK_DETAILS_PATH, "r") as f:
        TRACK_DETAILS = json.load(f)

    # Extract track lengths
    TRACK_LENGTHS = {k: v["value"] for k, v in TRACK_DETAILS.items()}

    # Load dataset
    data = pd.read_csv(CSV_PATH)
    logger.info(f"Loaded data: {data.shape[0]} rows")

    # Load PyTorch model
    model = NeuralNetwork()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    logger.info("Loaded trained PyTorch model")

    # ---------------- COMPLETED RACES ----------------
    current_year = datetime.today().year
    schedule = fastf1.get_event_schedule(current_year)
    schedule["Session5DateUtc"] = pd.to_datetime(schedule["Session5DateUtc"]).dt.tz_localize(None)
    completed_races = schedule[schedule["Session5DateUtc"] <= pd.Timestamp(datetime.now()).normalize()]["Location"].tolist()
    logger.info(f"Found {len(completed_races)} completed races")

    # ---------------- PER-TRACK METRICS ----------------
    metrics = []

    for location in track(completed_races, description="Processing Qualifying Sessions"):
        logger.info(f"Processing {location}...")
        try:
            # Load session
            session = fastf1.get_session(current_year, location, "Qualifying")
            session.load(messages=False, telemetry=False)
            quali_results = session.results
            
            # Convert Q1-Q3 to seconds
            quali_results[["Q1","Q2","Q3"]] = quali_results[["Q1","Q2","Q3"]].apply(lambda x: x.dt.total_seconds())
            quali_results["QualifyingTime"] = quali_results[["Q3","Q2","Q1"]].bfill(axis=1).iloc[:,0]
            quali_results["DriverNumber"] = quali_results["DriverNumber"].astype(int)
            
            # Driver-level features
            driver_data = (
                data[data["TrackLength"] == TRACK_LENGTHS[location]]
                .groupby("DriverNumber")
                .agg({
                    "Sector1Time":"min",
                    "Sector2Time":"min",
                    "Sector3Time":"min",
                    "SpeedST":"max",
                    "SpeedI1":"max",
                    "SpeedFL": "max"
                }).reset_index()
            )
            
            predicting_data = driver_data.copy()

            # Always use SOFT compound
            predicting_data["Compound"] = TRACK_DETAILS[location]["tires"]["SOFT"]
            
            predicting_data.loc[:, "TyreLife"] = 2.0
            predicting_data.loc[:, "TrackLength"] = TRACK_LENGTHS[location]

            # Track environmental averages
            track_env = (
                data[data["TrackLength"] == TRACK_LENGTHS[location]]
                .agg({
                    "Humidity":"mean",
                    "TrackTemp": "mean"
                })
            )
            for col in track_env.index:
                predicting_data[col] = track_env[col]
            
            # Prepare testing features
            testing_data = predicting_data[[
                'Sector1Time','Sector2Time','Sector3Time',
                'SpeedI1','SpeedST', "SpeedFL", "TyreLife",
                'Compound',
                'TrackLength','Humidity', "TrackTemp"
            ]]
            
            # Predict using PyTorch model
            X_test_tensor = torch.tensor(testing_data.values, dtype=torch.float32)
            with torch.no_grad():
                predicting_data["PredictedLapTime"] = model(X_test_tensor).numpy().flatten()
            
            merged = pd.merge(
                quali_results[["DriverNumber", "QualifyingTime"]],
                predicting_data[["DriverNumber", "PredictedLapTime"]],
                on="DriverNumber",
                how="inner"
            )

            # Drop rows where QualifyingTime is NaN
            merged = merged.dropna(subset=["QualifyingTime"])

            actual = merged["QualifyingTime"]
            predicted = merged["PredictedLapTime"]
            error = actual - predicted
            
            mse = (error**2).mean()
            rmse = mse**0.5
            accuracy = 100 * (1 - (abs(error)/actual).mean())
            
            metrics.append({
                "Track": location,
                "RMSE": rmse,
                "Accuracy(%)": accuracy
            })
            
            logger.info(f"  {location}: RMSE={rmse:.3f}, Accuracy={accuracy:.1f}%")

        except Exception as e:
            logger.warning(f"Skipping {location}: {e}")
            continue

    # ---------------- CREATE METRICS DATAFRAME ----------------
    df_metrics = pd.DataFrame(metrics).sort_values("Track").reset_index(drop=True)
    logger.info(f"Generated metrics for {len(df_metrics)} tracks")

    # ---------------- EXPORT PNG ----------------
    export(df_metrics, "TestReport.png")
    logger.info("Test report saved to TestReport.png")
    
    return df_metrics

if __name__ == "__main__":
    # Setup basic logging for standalone execution
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    evaluate_model()
