#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import joblib
import os
from contextlib import asynccontextmanager
from typing import List, Dict, Any
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlmodel import Session, select
from src.database.database import engine, IntelStockPrice

# 1. Handle dynamic paths so the server works seamlessly on local computer and github runners
BASE_DIR = os.getcwd()
MODEL_PATH = BASE_DIR + "/models" + "/linear_regression_v1.pkl"

# Model Cache
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Server Startup ---
    # Exception Handler
    try:
        # Load Model
        ml_models["stock_price_model"] = joblib.load(MODEL_PATH)

        # Status Output
        print(f"\n========================================================")
        print(f" SUCCESS: Model successfully loaded from:\n {MODEL_PATH}")
        print(f"========================================================\n")
    except Exception as e:
        # Error Output
        print(f"\n========================================================")
        print(f" ERROR: Could not load model file! Details:\n {e}")
        print(f"========================================================\n")

    yield

    # --- Server Shutdown ---
    print("Shutting down server, clearing model from memory...")
    ml_models.clear()


# Initialize the FastAPI Application
app = FastAPI(
    title="Intel Daily Stock Price Prediction API",
    description="A production-ready FastAPI server utilizing a Linear Regression model to predict Intel daily stock price.",
    version="1.0.0",
    lifespan=lifespan
)


# Stock Features Class 
class StockFeatures(BaseModel):
    open_price: float
    high: float
    low: float
    close_price: float
    ma5: float
    ma14: float
    hl_range: float

    # Backend Configuration Settings Block (/docs)
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
    """
    Standard pulse-check route. Tells monitoring services whether 
    the web app is healthy and if the model binary is actively sitting in memory.
    """
    model_loaded = "stock_price_model" in ml_models
    return {
        "status": "healthy" if model_loaded else "unhealthy (model missing)",
        "model_loaded": model_loaded,
        "project": "intel-daily-stock-price-prediction",
        "api_version": "1.0.0"
    }


@app.post("/predict")
def predict_stock(features: List[StockFeatures]):
    """
    Accepts a JSON payload of historical feature sets, transforms the items 
    into a structured Pandas DataFrame, and parses them directly into 
    your trained linear regression model to output stock price values.
    """
    # Ensures the Model Loaded Successfully
    if "stock_price_model" not in ml_models:
        raise HTTPException(
            status_code=500, 
            detail="Machine learning model failed to load into memory on startup."
        )

    # Exception Handler
    try:
        # Extract Raw Dictionaries
        data_dicts = [item.model_dump() for item in features]

        # Converts Array Into DataFrame
        X_input = pd.DataFrame(data_dicts)

        # Renaming Columns For Matching Model Columns
        X_input = X_input.rename(columns={
            "open_price": "Open", 
            "high": "High",
            "low": "Low",
            "close_price": "Close",
            "ma5": "MA5",
            "ma14": "MA14",
            "hl_range": "High-Low Range"
        })

        # Retrieves the Cached Model
        model = ml_models["stock_price_model"]

        # Model Predict with the Input Values
        predictions = model.predict(X_input)

        # Converts NumPy Array Into JSON
        return {
            "predictions": predictions.tolist()
        }

    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Inference Engine failed on parameters input parsing. Details: {str(e)}"
        )

@app.get("/api/v1/stocks", response_model=List[Dict[str, Any]])
def get_processed_stocks(db: Session = Depends(get_db)):
    """
    Fetches all historical records from the SQLite database, chronologically 
    calculates 5-day and 14-day Moving Averages (MA), and returns the payload.
    """
    statement = select(IntelStockPrice).order_by(IntelStockPrice.price_date)
    records = db.exec(statement).all()

    if not records:
        raise HTTPException(
            status_code=404, 
            detail="Database is empty. Please run src/database/fetch_data.py first."
        )

    # Extract Close Price For MA5 & MA14
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

