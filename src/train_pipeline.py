import os
import joblib
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error
)

from data_ingestion import DataIngestion
from data_preprocessing import DataPreprocessing
from feature_engineering import FeatureEngineering
from model_training import ModelTraining
from preprocessing_pipeline import PreprocessingPipeline
from experiment_tracking import setup_mlflow


# =====================================================
# MLFLOW SETUP
# =====================================================

setup_mlflow()

# =====================================================
# DATA INGESTION
# =====================================================

print("=" * 60)
print("DATA INGESTION")
print("=" * 60)

ingestion = DataIngestion(
    "data/raw/car_price.csv"
)

df = ingestion.load_data()

print(f"Dataset Shape: {df.shape}")

# =====================================================
# DATA PREPROCESSING
# =====================================================

print("=" * 60)
print("DATA PREPROCESSING")
print("=" * 60)

preprocessor = DataPreprocessing()

df = preprocessor.preprocess(df)

# =====================================================
# FEATURE ENGINEERING
# =====================================================

print("=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

feature_engineer = FeatureEngineering()

df = feature_engineer.transform(df)

print(df.head())
print(df.shape)

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

trainer = ModelTraining()

(
    X_train,
    X_test,
    y_train,
    y_test
) = trainer.split_data(df)

print(f"X Train Shape : {X_train.shape}")
print(f"X Test Shape  : {X_test.shape}")

# =====================================================
# PREPROCESSING PIPELINE
# =====================================================

preprocessing_pipeline = PreprocessingPipeline()

preprocessor = (
    preprocessing_pipeline
    .get_preprocessor()
)

# =====================================================
# MODEL PIPELINE
# =====================================================

pipeline = trainer.get_pipeline(
    preprocessor
)

# =====================================================
# TRAINING
# =====================================================

print("=" * 60)
print("TRAINING STARTED")
print("=" * 60)

pipeline.fit(
    X_train,
    y_train
)

print("TRAINING COMPLETED")

# =====================================================
# PREDICTIONS
# =====================================================

y_pred = pipeline.predict(X_test)

# =====================================================
# METRICS
# =====================================================

r2 = r2_score(
    y_test,
    y_pred
)

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = root_mean_squared_error(
    y_test,
    y_pred
)

mape = (
    mean_absolute_percentage_error(
        y_test,
        y_pred
    ) * 100
)

# =====================================================
# RESULTS
# =====================================================

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(f"R² Score : {r2:.4f}")
print(f"MAE      : {mae:,.2f}")
print(f"RMSE     : {rmse:,.2f}")
print(f"MAPE     : {mape:.4f}%")

# =====================================================
# MLFLOW LOGGING
# =====================================================

with mlflow.start_run():

    model = pipeline.named_steps["model"]

    mlflow.log_params({
        "n_estimators": model.n_estimators,
        "max_depth": model.max_depth,
        "learning_rate": model.learning_rate,
        "subsample": model.subsample,
        "colsample_bytree": model.colsample_bytree,
        "min_child_weight": model.min_child_weight,
        "gamma": model.gamma,
        "reg_alpha": model.reg_alpha,
        "reg_lambda": model.reg_lambda
    })

    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mape", mape)

    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model",
        serialization_format="cloudpickle"
    )

print("MLflow logging completed")

# =====================================================
# SAVE MODEL
# =====================================================

os.makedirs(
    "artifacts",
    exist_ok=True
)

joblib.dump(
    pipeline,
    "artifacts/model.pkl"
)

print("Model Saved Successfully")

print("=" * 60)
print("PIPELINE COMPLETED")
print("=" * 60)