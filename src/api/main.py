#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import joblib
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from typing import List, Dict, Any
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select
from sklearn.linear_model import LinearRegression  # Added for the manual training feature
from src.database.database import engine, IntelStockPrice

BASE_DIR = os.getcwd()
MODEL_PATH = os.path.join(BASE_DIR, "models", "linear_regression_v1.pkl")

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Server Startup ---
    try:
        ml_models["stock_price_model"] = joblib.load(MODEL_PATH)
        print(f"\n========================================================")
        print(f" SUCCESS: Model successfully loaded from:\n {MODEL_PATH}")
        print(f"========================================================\n")
    except Exception as e:
        print(f"\n========================================================")
        print(f" ERROR: Could not load model file! Details:\n {e}")
        print(f"========================================================\n")

    yield

    # --- Server Shutdown ---
    print("Shutting down server, clearing model from memory...")
    ml_models.clear()

app = FastAPI(
    title="Intel Daily Stock Price Prediction API",
    description="A production-ready FastAPI server utilizing a Linear Regression model to predict Intel daily stock price.",
    version="1.0.0",
    lifespan=lifespan
)

class StockFeatures(BaseModel):
    open_price: float
    high: float
    low: float
    close_price: float
    ma5: float
    ma14: float
    hl_range: float

    model_config = {
        "json_schema_extra": {
            "example": {
                "open_price": 30.50,
                "high": 31.20,
                "low": 30.10,
                "close_price": 30.85,
                "ma5": 30.42,
                "ma14": 29.95,
                "hl_range": 1.10
            }
        }
    }

def get_db():
    with Session(engine) as session:
        yield session

@app.get("/")
def health_check():
    model_loaded = "stock_price_model" in ml_models
    return {
        "status": "healthy" if model_loaded else "unhealthy (model missing)",
        "model_loaded": model_loaded,
        "project": "intel-daily-stock-price-prediction",
        "api_version": "1.0.0"
    }

@app.post("/predict")
def predict_stock(features: List[StockFeatures]):
    if "stock_price_model" not in ml_models:
        raise HTTPException(
            status_code=500, 
            detail="Machine learning model failed to load into memory on startup."
        )

    try:
        data_dicts = [item.model_dump() for item in features]
        X_input = pd.DataFrame(data_dicts)

        X_input = X_input.rename(columns={
            "open_price": "Open", 
            "high": "High",
            "low": "Low",
            "close_price": "Close",
            "ma5": "MA5",
            "ma14": "MA14",
            "hl_range": "High-Low Range"
        })

        model = ml_models["stock_price_model"]
        predictions = model.predict(X_input)

        return {
            "predictions": predictions.tolist()
        }

    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Inference Engine failed on parameters input parsing. Details: {str(e)}"
        )

# --- NEW ROUTE: UPDATE DATASET OVERRIDE ---
@app.post("/api/v1/admin/update-data")
def trigger_data_update():
    """
    Executes the data collection and cleaning scripts in sequence.
    """
    try:
        # Using sys.executable guarantees it uses your exact project virtual env environment
        subprocess.run([sys.executable, "src/database/fetch_data.py"], check=True)
        subprocess.run([sys.executable, "src/database/merge_to_csv.py"], check=True)
        return {"status": "success", "message": "Data pipeline run completed successfully."}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"A processing script failed: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- NEW ROUTE: RETRAIN MODEL OVERRIDE ---
@app.post("/api/v1/admin/retrain-model")
def trigger_model_training():
    """
    Reads the engineered CSV matrix, performs a hot training cycle, and updates live memory cache.
    """
    try:
        csv_path = os.path.join(BASE_DIR, "data", "processed", "intel-daily-stock-price-data-processed.csv")
        if not os.path.exists(csv_path):
            raise HTTPException(status_code=404, detail="Dataset missing. Please update the dataset first.")

        df = pd.read_csv(csv_path)
        feature_columns = ["Open", "High", "Low", "Close", "MA5", "MA14", "High-Low Range"]

        if df.empty or "Target" not in df.columns:
            raise HTTPException(status_code=422, detail="Dataset format invalid or too short for training matrices.")

        X = df[feature_columns]
        y = df["Target"]

        # Fit a new instance of Linear Regression
        new_model = LinearRegression()
        new_model.fit(X, y)

        # Persist binary to system storage
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(new_model, MODEL_PATH)

        # Perform live cache swap
        ml_models["stock_price_model"] = new_model
        return {"status": "success", "message": "Inference architecture successfully updated."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training system malfunction: {str(e)}")

@app.get("/api/v1/stocks", response_model=List[Dict[str, Any]])
def get_processed_stocks(db: Session = Depends(get_db)):
    statement = select(IntelStockPrice).order_by(IntelStockPrice.price_date)
    records = db.exec(statement).all()

    if not records:
        raise HTTPException(
            status_code=404, 
            detail="Database is empty. Please run src/database/fetch_data.py first."
        )

    close_prices = [r.close_price for r in records]
    response_payload = []

    for i, record in enumerate(records):
        ma5 = round(sum(close_prices[i-4:i+1]) / 5, 4) if i >= 4 else None
        ma14 = round(sum(close_prices[i-13:i+1]) / 14, 4) if i >= 13 else None
        hl_range = round(record.high - record.low, 4)
        response_payload.append({
            "id": record.id,
            "date": record.price_date.isoformat(),
            "open_price": record.open_price,
            "high": record.high,
            "low": record.low,
            "close_price": record.close_price,
            "moving_average_5d": ma5,
            "moving_average_14d": ma14,
            "hl_range": hl_range
        })
    return response_payload

