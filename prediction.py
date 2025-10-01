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
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_driver_mapping_from_csv(data):
    """Get driver number to name mapping from CSV data"""
    try:
        # Create mapping from DriverNumber to Driver name
        driver_mapping = dict(zip(data['DriverNumber'], data['Driver']))
        logger.info(f"Got driver mapping from CSV data: {len(driver_mapping)} drivers")
        return driver_mapping
    except Exception as e:
        logger.warning(f"Error getting driver mapping from CSV: {e}")
        return {}

def predict_qualifying_times(location=None):
    """Predict qualifying times for a specific location"""
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

    # Get driver mapping from CSV data
    driver_mapping = get_driver_mapping_from_csv(data)
    if not driver_mapping:
        logger.warning("No driver mapping found - will use driver numbers")

    # Load PyTorch model
    model = NeuralNetwork()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    logger.info("Loaded trained PyTorch model")

    # ---------------- LOCATION VALIDATION ----------------
    current_year = datetime.today().year
    
    if location is None:
        # If no location provided, get current/upcoming race
        schedule = fastf1.get_event_schedule(current_year)
        schedule["Session5DateUtc"] = pd.to_datetime(schedule["Session5DateUtc"]).dt.tz_localize(None)
        
        current_date = pd.Timestamp(datetime.now()).normalize()
        upcoming_races = schedule[schedule["Session5DateUtc"] > current_date]
        
        if upcoming_races.empty:
            completed_races = schedule[schedule["Session5DateUtc"] <= current_date]
            if not completed_races.empty:
                target_location = completed_races.iloc[-1]["Location"]
                logger.info(f"No location provided. Using most recent race: {target_location}")
            else:
                raise ValueError("No races found for current year")
        else:
            target_location = upcoming_races.iloc[0]["Location"]
            logger.info(f"No location provided. Using upcoming race: {target_location}")
    else:
        # Validate provided location
        if location not in TRACK_DETAILS:
            available_locations = list(TRACK_DETAILS.keys())
            raise ValueError(f"Location '{location}' not found. Available locations: {available_locations}")
        target_location = location
        logger.info(f"Predicting for specified location: {target_location}")
    
    # Driver mapping is already obtained from CSV data above

    # ---------------- PREDICTION FOR TARGET LOCATION ----------------
    logger.info(f"Processing {target_location}...")
    
    try:
        # Driver-level features from historical data
        driver_data = (
            data[data["TrackLength"] == TRACK_LENGTHS[target_location]]
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
        
        # If no historical data for this track, use overall best times
        if driver_data.empty:
            logger.warning(f"No historical data for {target_location}. Using overall best times.")
            driver_data = (
                data.groupby("DriverNumber")
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

        # Always use SOFT compound for qualifying
        predicting_data["Compound"] = TRACK_DETAILS[target_location]["tires"]["SOFT"]
        
        predicting_data.loc[:, "TyreLife"] = 2.0
        predicting_data.loc[:, "TrackLength"] = TRACK_LENGTHS[target_location]

        # Track environmental averages
        track_env = (
            data[data["TrackLength"] == TRACK_LENGTHS[target_location]]
            .agg({
                "Humidity":"mean",
                "TrackTemp": "mean"
            })
        )
        
        # If no track-specific environmental data, use overall averages
        if track_env.isna().any().any():
            logger.warning(f"No environmental data for {target_location}. Using overall averages.")
            track_env = data.agg({
                "Humidity":"mean",
                "TrackTemp": "mean"
            })
        
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
        
        # Add driver names and create final results
        results = predicting_data[["DriverNumber", "PredictedLapTime"]].copy()
        results["DriverName"] = results["DriverNumber"].map(driver_mapping)
        
        # Fill missing driver names with driver numbers (fallback)
        results["DriverName"] = results["DriverName"].fillna(results["DriverNumber"].astype(str))
        
        # Sort by predicted time (fastest first)
        results = results.sort_values("PredictedLapTime").reset_index(drop=True)
        results["PredictedPosition"] = range(1, len(results) + 1)
        
        # Convert lap time to m:ss.ms format for display
        results["PredictedLapTimeFormatted"] = results["PredictedLapTime"].apply(
            lambda x: f"{int(x // 60)}:{x % 60:06.3f}"
        )
        
        # Calculate gap to leader
        leader_time = results.iloc[0]["PredictedLapTime"]
        results["Gap"] = results["PredictedLapTime"] - leader_time
        results["GapFormatted"] = results["Gap"].apply(
            lambda x: f"+{x:.3f}" if x > 0 else "0.000"
        )
        
        # Reorder columns for better display
        results = results[["PredictedPosition", "DriverNumber", "DriverName", "PredictedLapTime", "PredictedLapTimeFormatted", "Gap", "GapFormatted"]]
        
        logger.info(f"Generated predictions for {len(results)} drivers")
        
        # Display results
        print(f"\n🏁 F1 Qualifying Predictions - {target_location} {current_year}")
        print("=" * 80)
        print(f"{'Pos':<4} {'Driver':<20} {'Predicted Time':<15} {'Gap':<10}")
        print("-" * 80)
        
        for _, row in results.iterrows():
            pos = int(row["PredictedPosition"])
            driver = row["DriverName"]
            # Convert seconds to m:ss.ms format
            total_seconds = row['PredictedLapTime']
            minutes = int(total_seconds // 60)
            seconds = total_seconds % 60
            pred_time = f"{minutes}:{seconds:06.3f}"
            gap = row["GapFormatted"]
            print(f"{pos:<4} {driver:<20} {pred_time:<15} {gap:<10}")
        
        # Save predictions as image
        try:
            from dataframe_image import export
            output_dir = Path("QualiPredictions")
            output_dir.mkdir(exist_ok=True)
            output_file = output_dir / f"{target_location}.png"
            
            # Create a clean dataframe for display
            display_df = results[["PredictedPosition", "DriverName", "PredictedLapTimeFormatted", "GapFormatted"]].copy()
            display_df.columns = ["Pos", "Driver", "Time", "Gap"]
            
            # Save as image
            export(display_df, str(output_file))
            logger.info(f"Predictions saved as image to {output_file}")
        except ImportError:
            logger.warning("dataframe_image package not found. Saving as CSV instead.")
            output_dir = Path("QualiPredictions")
            output_dir.mkdir(exist_ok=True)
            output_file = output_dir / f"{target_location}.csv"
            results.to_csv(output_file, index=False)
            logger.info(f"Predictions saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving image: {e}. Saving as CSV instead.")
            output_dir = Path("QualiPredictions")
            output_dir.mkdir(exist_ok=True)
            output_file = output_dir / f"{target_location}.csv"
            results.to_csv(output_file, index=False)
            logger.info(f"Predictions saved to {output_file}")
        
        return results

    except Exception as e:
        logger.error(f"Error processing {target_location}: {e}")
        raise

if __name__ == "__main__":
    # Setup basic logging for standalone execution
    # Manual location setting - change this value each time you want to predict for a different location
    # Set to None for current/upcoming race, or specify a location name
    LOCATION = "Baku"  # Change this value manually each time
    
    print("F1 Qualifying Time Predictor")
    print("=" * 40)
    
    if LOCATION is None:
        print("Using current/upcoming race...")
    else:
        print(f"Predicting for location: {LOCATION}")
    
    try:
        predict_qualifying_times(LOCATION)
    except Exception as e:
        print(f"Error: {e}")
        print("\nAvailable locations:")
        try:
            import json
            with open("./compound_map.json", "r") as f:
                track_details = json.load(f)
            for loc in sorted(track_details.keys()):
                print(f"  - {loc}")
        except:
            print("  Could not load available locations")