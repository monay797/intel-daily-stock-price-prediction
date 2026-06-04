#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import datetime
import yfinance as yf
import requests_cache
from sqlmodel import Session, select
from database import engine, IntelStockPrice, create_db_and_tables

def populate_stock_data():
    # Create Database and Tables
    create_db_and_tables()

    # Print Statements
    print("Fetching Intel stock data from Yahoo Finance...")
    print("Setting up secure connection to Yahoo Finance...")

    # Cache File Directory
    cache_path = "src/cache/yfinance.cache"

    # Local Cache that Expires after 1 Hour 
    session = requests_cache.CachedSession(str(cache_path), expire_after=3600)

    # Custom User-Agent
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })

    print("Fetching Intel stock data...")

    # Fetching the Stock Data Starting from May 1, 2026
    ticker = yf.Ticker("INTC", session=session)

    try:

        df = ticker.history(start="2026-05-01", interval="1d")

        if df.empty:
            print("No Data Retrieved.")
            return

        # Opens Session with SQLite Database
        with Session(engine) as session:
            added_count = 0

            # Looping Through Each Rows
            for index, row in df.iterrows():
                # Extract the Date
                record_date = index.date()

                # Check Specific Date If Exist
                statement = select(IntelStockPrice).where(IntelStockPrice.price_date == record_date)
                existing_record = session.exec(statement).first()

                # Insert Data If Specific Date Does Not Exist
                if not existing_record:
                    stock_entry = IntelStockPrice(
                        price_date=record_date,
                        open_price=round(row['Open'], 4),
                        high=round(row['High'], 4),
                        low=round(row['Low'], 4),
                        close_price=round(row['Close'], 4)
                    )
                    session.add(stock_entry)
                    added_count += 1

            # Saves New Data Inserted
            session.commit()
            print(f"Success! Added {added_count} new daily stock records to the database.")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Hint: Yahoo Finance might still be rate-limiting your IP. Take a 5-minute break and try again!")

if __name__ == "__main__":
    populate_stock_data()

