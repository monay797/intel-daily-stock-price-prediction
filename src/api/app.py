#!/usr/bin/env python
# coding: utf-8

# In[37]:


import streamlit as st
import requests

# Set up the title and header of the web page
st.set_page_config(page_title="Intel Stock Price Predictor", page_icon="📈")
st.title("📈 Intel Daily Stock Price Prediction")
st.write("Adjust the metrics below to request a real-time prediction from the machine learning model backend.")

st.markdown("---")

# Create two columns for clean side-by-side data entry fields
col1, col2, col3 = st.columns(3)

with col1:
    open_price = st.number_input("Open Price ($)", value=30.85)
    close_price = st.number_input("Close Price ($)", value=29.50)

with col2:
    low = st.number_input("Low Price ($)", value=28.50)
    high = st.number_input("High Price ($)", value=30.10)
    hl_range = st.number_input("High-Low Range", value=1.10)

with col3:
    ma5 = st.number_input("Moving Average 5-Days", value=29.95)
    ma14 = st.number_input("Moving Average 14-Days", value=30.42)


st.markdown("---")

# Create a big action button
if st.button("Generate Prediction", type="primary"):

    # Pack the exact JSON payload layout your FastAPI backend expects
    payload = [{
        "open_price": open_price,
        "high": high,
        "low": low,
        "close_price": close_price,
        "ma5": ma5,
        "ma14": ma14,
        "hl_range": hl_range
    }]

    with st.spinner("Communicating with FastAPI model server..."):
        try:
            # Change '/predict' if your FastAPI endpoint URL path is named differently
            backend_url = "http://127.0.0.1:8000/predict" 
            response = requests.post(backend_url, json=payload)

            if response.status_code == 200:
                prediction_result = response.json()

                # Render the result in a beautiful green callout box
                st.success("### Prediction Success!")
                st.metric(label="Predicted Target Price", value=round(prediction_result['predictions'][0], 4))
            else:
                st.error(f"Backend returned an error ({response.status_code}): {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the backend server. Is your Uvicorn terminal running?")

