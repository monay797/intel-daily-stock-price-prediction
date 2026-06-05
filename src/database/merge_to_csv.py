#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import pandas as pd
from pathlib import Path
from sqlmodel import Session, select
from database import engine, IntelStockPrice

def merge_db_to_csv(csv_filename="intel-daily-stock-price-data-processed.csv"):
    csv_path = "data/processed/" + csv_filename

    # 1. FETCH AND CLEAN DATABASE DATA ONLY
    print("Extracting fresh records from SQLite database...")
    with Session(engine) as session:
        statement = select(IntelStockPrice)
        db_records = session.exec(statement).all()

    if not db_records:
        print("Database is empty. Run fetch_data.py first.")
        return

    # Convert database structures into a temporary Pandas DataFrame
    db_list = []
    for r in db_records:
        db_list.append({
            "Date": r.price_date,
            "Open": r.open_price,
            "High": r.high,
            "Low": r.low,
            "Close": r.close_price,
            "MA5": r.ma5,
            "MA14": r.ma14,
            "High-Low Range": r.hl_range
        })
    df_db = pd.DataFrame(db_list)
    df_db['Date'] = pd.to_datetime(df_db['Date'])

    # 🔥 SPECIFIC REQUIREMENTS: Clean only the database data to protect existing CSV rows
    initial_db_rows = len(df_db)
    df_db = df_db.dropna(subset=['MA5', 'MA14']).reset_index(drop=True)
    db_nulls_dropped = initial_db_rows - len(df_db)

    if db_nulls_dropped > 0:
        print(f"Cleaned {db_nulls_dropped} cold-start rows containing nulls directly from the database batch.")

    # 2. LOAD EXISTING PRISTINE CSV FILE (NO CLEANING APPLIED HERE)

    print(f"Found existing clean CSV file at {csv_path}. Loading data...")
    df_csv = pd.read_csv(csv_path)

    # Standardize old legacy column names if they exist
    if 'price_date' in df_csv.columns and 'Date' not in df_csv.columns:
        df_csv = df_csv.rename(columns={'price_date': 'Date'})

    df_csv['Date'] = pd.to_datetime(df_csv['Date'])

    # Combine both datasets together
    print("Concatenating clean database rows with historical pristine CSV rows...")
    df_combined = pd.concat([df_csv, df_db], ignore_index=True)

    # 3. DEDUPLICATION MATRIX
    initial_row_count = len(df_combined)
    df_combined = df_combined.drop_duplicates(subset=['Date'], keep='last')
    duplicates_removed = initial_row_count - len(df_combined)
    print(f"Deduplication complete: Dropped {duplicates_removed} overlapping rows.")

    # Chronological Sorting
    df_combined = df_combined.sort_values('Date').reset_index(drop=True)

    # Target Feature
    print("🔮 Engineering predictive target column (shifting close_price -1)...")
    df_combined["Target"] = df_combined["Close"].shift(-1)

    # Remove Last Row
    df_combined = df_combined.dropna()

    # Export
    df_combined['Date'] = df_combined['Date'].dt.strftime('%Y-%m-%d')
    df_combined.to_csv(csv_path, index=False)

    print(f"Success! Processed CSV updated safely: {csv_path} ({len(df_combined)} rows total).")

if __name__ == "__main__":
    merge_db_to_csv()

