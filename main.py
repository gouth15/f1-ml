#!/usr/bin/env python3
"""
F1 Data Analysis Pipeline - Main Entry Point

This script orchestrates the complete F1 data analysis workflow:
1. Data Collection - Fetch F1 race data from FastF1 API
2. Model Training - Train neural network on collected data
3. Model Evaluation - Compare predictions against actual qualifying results
what
Usage:
    python main.py
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Import our modules
from collect_data import collect_f1_data
from train import train_model
from model_comparison import evaluate_model

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('f1_pipeline.log')
        ]
    )
    return logging.getLogger(__name__)

def check_dependencies():
    """Check if required files and dependencies exist"""
    logger = logging.getLogger(__name__)
    
    # Check for compound_map.json
    if not Path("compound_map.json").exists():
        logger.error("compound_map.json not found! This file is required for track details.")
        return False
    
    # Check for required Python packages
    try:
        import fastf1
        import torch
        import sklearn
        import rich
        import pandas
        import numpy
    except ImportError as e:
        logger.error(f"Missing required package: {e}")
        logger.error("Please install requirements: pip install -r requirements.txt")
        return False
    
    logger.info("All dependencies check passed")
    return True

def run_data_collection():
    """Run data collection step"""
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("STARTING DATA COLLECTION")
    logger.info("=" * 50)
    
    try:
        collect_f1_data()
        logger.info("Data collection completed successfully")
        return True
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        logger.info("This may be due to missing track data or API issues.")
        logger.info("The pipeline will continue if existing data.csv is available.")
        return False

def run_model_training():
    """Run model training step"""
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("STARTING MODEL TRAINING")
    logger.info("=" * 50)
    
    try:
        train_model()
        logger.info("Model training completed successfully")
        return True
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        return False

def run_model_evaluation():
    """Run model evaluation step"""
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("STARTING MODEL EVALUATION")
    logger.info("=" * 50)
    
    try:
        evaluate_model()
        logger.info("Model evaluation completed successfully")
        return True
    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        return False

def main():
    """Main entry point - runs complete pipeline"""
    logger = setup_logging()
    
    logger.info("F1 Data Analysis Pipeline Starting")
    logger.info(f"Timestamp: {datetime.now()}")
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Dependency check failed. Exiting.")
        sys.exit(1)
    
    success_count = 0
    total_steps = 3
    
    # Step 1: Data Collection
    if run_data_collection():
        success_count += 1
    else:
        logger.warning("Data collection had issues but continuing with available data.")
        # Check if we have any data to work with
        if Path("data.csv").exists():
            logger.info("Found existing data.csv, continuing with pipeline.")
            success_count += 1
        else:
            logger.error("No data available. Exiting.")
            sys.exit(1)
    
    # Step 2: Model Training
    if run_model_training():
        success_count += 1
    else:
        logger.error("Model training failed. Exiting.")
        sys.exit(1)
    
    # Step 3: Model Evaluation
    if run_model_evaluation():
        success_count += 1
    else:
        logger.error("Model evaluation failed. Exiting.")
        sys.exit(1)
    
    # Summary
    logger.info("=" * 50)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Steps completed: {success_count}/{total_steps}")
    
    if success_count == total_steps:
        logger.info("✅ All steps completed successfully!")
        logger.info("Generated files:")
        if Path("data.csv").exists():
            logger.info("  - data.csv (collected F1 data)")
        if Path("f1-model.pt").exists():
            logger.info("  - f1-model.pt (trained model)")
        if Path("feature_importance.csv").exists():
            logger.info("  - feature_importance.csv (feature importance analysis)")
        if Path("TestReport.png").exists():
            logger.info("  - TestReport.png (model evaluation results)")
        logger.info("  - f1_pipeline.log (execution log)")
    else:
        logger.warning(f"⚠️  Pipeline completed with {total_steps - success_count} failures")
        sys.exit(1)

if __name__ == "__main__":
    main()