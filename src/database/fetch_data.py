#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datetime import datetime
import yfinance as yf
from sqlmodel import Session, select
from database import engine, IntelStockPrice, create_db_and_tables

def populate_stock_data():
    # Create Database and Tables
    create_db_and_tables()

    # Print Statement
    print("Fetching Intel stock data from Yahoo Finance...")

    # Fetching the Stock Data Starting from May 1, 2026
    ticker = yf.Ticker("INTC")
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
            statement = select(IntelStockPrice).where(IntelStockPrice.date == record_date)
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

if __name__ == "__main__":
    populate_stock_data()

