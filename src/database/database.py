#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datetime import date
from typing import Optional
from sqlmodel import Field, SQLModel, create_engine, Session

# Table Structure Class
class IntelStockPrice(SQLModel, table=True):
    # Unique ID - Primary Key
    id: Optional[int] = Field(default=None, primary_key=True)

    # Unique Date
    price_date: date = Field(unique=True, index=True) 

    # Column Features
    open_price: float
    high: float
    low: float
    close_price: float

# Database File Name
DATABASE_FILE = "intel_stock_price.db"

# Database URL
sqlite_url = f"sqlite:///{DATABASE_FILE}"

# Prints Commands to the Terminal
engine = create_engine(sqlite_url, echo=True)

# Helper Function
def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

# Initialize the Database
if __name__ == "__main__":
    print("Initializing database and tables...")
    create_db_and_tables()
    print(f"Database file '{DATABASE_FILE}' has been sucessfully generated.")

