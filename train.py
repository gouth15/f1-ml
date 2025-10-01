# IMPORT LIBRARIES
import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import mutual_info_regression
import logging

import torch
import torch.nn as nn
import torch.optim as optim

from model import NeuralNetwork

def train_model():
    """Train neural network model on F1 data and save it"""
    logger = logging.getLogger(__name__)
    
    # LOAD DATA
    CSV_PATH = Path("data.csv")
    if not CSV_PATH.exists():
        raise FileNotFoundError("data.csv not found. Please run data collection first.")
    
    data = pd.read_csv(CSV_PATH)
    logger.info(f"Loaded data: {data.shape[0]} rows, {data.shape[1]} columns")

    # DEFINE TARGET AND FEATURES
    TARGET_COLUMN = "LapTime"
    FEATURE_COLUMNS = [col for col in data.columns if col not in [TARGET_COLUMN, "DriverNumber", "Driver"]]
    logger.info(f"Features: {FEATURE_COLUMNS}")
    X = data[FEATURE_COLUMNS]
    Y = data[TARGET_COLUMN]

    # SPLIT TRAIN AND TEST SET
    XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=0.30, random_state=42)
    logger.info(f"Training set: {XTrain.shape[0]} samples, {XTrain.shape[1]} features")
    logger.info(f"Test set: {XTest.shape[0]} samples")

    # OPTIONAL: FEATURE IMPORTANCE USING MUTUAL INFORMATION
    logger.info("Calculating feature importance...")
    mi_scores = mutual_info_regression(X, Y)
    feature_importance_df = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "MutualInfo": mi_scores
    }).sort_values(by="MutualInfo", ascending=False)
    feature_importance_df.to_csv("feature_importance.csv", index=False)
    logger.info("Feature importance saved to feature_importance.csv")

    # CONVERT DATA TO TORCH TENSORS
    x_train_tensor = torch.tensor(XTrain.values, dtype=torch.float32)
    x_test_tensor = torch.tensor(XTest.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(YTrain.values, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(YTest.values, dtype=torch.float32).view(-1, 1)

    # TRAIN MODEL
    logger.info("Initializing neural network...")
    model = NeuralNetwork()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    
    EPOCHS = 1000
    logger.info(f"Starting training for {EPOCHS} epochs...")

    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        outputs = model(x_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0 or epoch == EPOCHS:
            logger.info(f"Epoch {epoch}/{EPOCHS} -- Loss: {loss.item():.6f}")

    # EVALUATE MODEL
    logger.info("Evaluating model...")
    with torch.no_grad():
        y_pred = model(x_test_tensor)
        rmse = torch.sqrt(criterion(y_pred, y_test_tensor))
        logger.info(f"Test RMSE: {rmse.item():.6f}")

    # SAVE MODEL
    MODEL_PATH = Path("f1-model.pt")
    torch.save(model.state_dict(), MODEL_PATH)
    logger.info(f"Model saved at {MODEL_PATH}")
    
    return model, rmse.item()

if __name__ == "__main__":
    # Setup basic logging for standalone execution
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    train_model()
