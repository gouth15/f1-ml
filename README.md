# F1 Data Analysis Pipeline

A fun ML project to predict F1 lap times and analyze driver performance using FastF1 data.

## Quick Start

Run the complete pipeline with a single command:

```bash
python main.py
```

This will automatically:
1. Collect F1 race data from FastF1 API
2. Train a neural network model
3. Evaluate the model against actual qualifying results

## Requirements

Install required packages:

```bash
pip install -r requirements.txt
```

## Pipeline Overview

The `main.py` script orchestrates three main steps:

1. **Data Collection** - Fetches F1 race data from FastF1 API
2. **Model Training** - Trains a PyTorch neural network on collected data  
3. **Model Evaluation** - Compares predictions against actual qualifying results

## Output Files

After running `main.py`, you'll get:

- `data.csv` - Cleaned F1 race data
- `f1-model.pt` - Trained neural network model
- `feature_importance.csv` - Feature importance analysis
- `TestReport.png` - Model evaluation report
- `f1_pipeline.log` - Detailed execution log

## Individual Components

You can also run individual components separately:

```bash
# Data collection only
python collect_data.py

# Model training only (requires data.csv)
python train.py

# Model evaluation only (requires trained model)
python model_comparison.py
```

## Project Background

This project started as a fun exploration of F1 data using the FastF1 Python package. The goal is to create an ML model that predicts each driver's pace and suggests optimal tire strategies.

The pipeline focuses on lap data from Sprint and Race sessions (excluding practice sessions to avoid fuel load complications). The model uses features like sector times, speeds, tire compounds, and track conditions to predict lap times.

## Model Details

- **Architecture**: 11 → 32 → 16 → 1 (ReLU activations)
- **Features**: Sector times, speeds, tire compound, track conditions
- **Training**: 1000 epochs with Adam optimizer
- **Evaluation**: RMSE and accuracy metrics per track

## Dependencies

- `fastf1` - F1 data source
- `torch` - Neural network framework
- `scikit-learn` - Machine learning utilities
- `pandas` - Data processing
- `rich` - Progress bars and logging
- `dataframe-image` - Report generation