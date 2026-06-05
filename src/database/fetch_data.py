#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
import pandas as pd
import yfinance as yf
from sqlmodel import Session, select
from database import engine, IntelStockPrice, create_db_and_tables

def populate_stock_data():
    # Create Database and Tables
    create_db_and_tables()

    # Print Statement
    print("Fetching Intel stock data...")

    # Intel Ticker
    ticker = yf.Ticker("INTC")

    # Exception Handler
    try:
        # Fetching the Stock Data Starting from May 1, 2026
        df = ticker.history(period="60d")

        if df.empty:
            print("No Data Retrieved.")
            return

        # Clean up multi-index columns if they exist
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index(level="symbol", drop=True)
        df = df.reset_index()

        # Feature Engineering
        df = df.sort_values('Date').reset_index(drop=True)
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA14'] = df['Close'].rolling(window=14).mean()
        df['High-Low Range'] = df['High'] - df['Low']

        # Opens Session with SQLite Database
        added_count = 0
        with Session(engine) as session:

            print("🔍 Syncing with database states...")
            existing_dates = set(session.exec(select(IntelStockPrice.price_date)).all())

            # Looping Through Each Rows
            for _, row in df.iterrows():
                # Standardize Date Format
                if isinstance(row['Date'], pd.Timestamp):
                    record_date = row['Date'].to_pydatetime().date()
                else:
                    try:
                        record_date = datetime.strptime(str(row['Date']), "%Y-%m-%d").date()
                    except ValueError:
                        record_date = pd.to_datetime(row['Date']).date()

                # Insert Data If Specific Date Does Not Exist
                if record_date not in existing_dates:
                    # Handle initial cold-start NaN rows gracefully
                    ma5_val = round(float(row['MA5']), 4) if not pd.isna(row['MA5']) else None
                    ma14_val = round(float(row['MA14']), 4) if not pd.isna(row['MA14']) else None
                    hl_val = round(float(row['High-Low Range']), 4)

                    stock_entry = IntelStockPrice(
                        price_date=record_date,
                        open_price=round(float(row['Open']), 4),
                        high=round(float(row['High']), 4),
                        low=round(float(row['Low']), 4),
                        close_price=round(float(row['Close']), 4),
                        ma5=ma5_val,
                        ma14=ma14_val,
                        hl_range=hl_val
                    )
                    session.add(stock_entry)
                    existing_dates.add(record_date)
                    added_count += 1

            # Saves New Data Inserted
            session.commit()
            print(f"Success! Added {added_count} new daily stock records to the database.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    populate_stock_data()

