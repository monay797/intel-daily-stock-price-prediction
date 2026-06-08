#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import requests
import yfinance as yf
import plotly.graph_objects as go

# 1. Force wide screen layout
st.set_page_config(page_title="Intel Stock Price Predictor", page_icon="📈", layout="wide")

# 2. Re-engineered CSS Layout Matrix to clear title clipping and provide a "zoomed-in" high fidelity feel
st.markdown("""
    <style>
        /* Restored top padding clearance to keep the title completely safe from clipping */
        .block-container {
            padding-top: 2.2rem !important;
            padding-bottom: 1.5rem !important;
            padding-left: 2.5rem !important;
            padding-right: 2.5rem !important;
        }
        /* Optimized structural margins under elements */
        div.stBlock {
            margin-bottom: 0.4rem !important;
        }
        h2, h3 {
            margin-top: 0.2rem !important;
            margin-bottom: 0.6rem !important;
        }
        /* Scaled up metric size tokens for a clean "zoomed in" appearance */
        [data-testid="stMetricValue"] {
            font-size: 2.1rem !important;
            font-weight: 700 !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 1.0rem !important;
        }
        /* Increased baseline scale of form input values */
        .stNumberInput div div input {
            font-size: 1.05rem !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR: PIPELINE MANAGEMENT CONTROLS ---
with st.sidebar:
    st.markdown("### ⚙️ Pipeline Admin")
    st.caption("Manually trigger backend pipeline engines.")

    if st.button("🔄 Update Dataset", use_container_width=True):
        with st.spinner("Updating..."):
            try:
                res = requests.post("http://127.0.0.1:8000/api/v1/admin/update-data")
                if res.status_code == 200: st.success("Dataset synced!")
                else: st.error("Pipeline error.")
            except Exception as e: st.error("Backend unreachable.")

    if st.button("🧠 Retrain Model", use_container_width=True):
        with st.spinner("Training..."):
            try:
                res = requests.post("http://127.0.0.1:8000/api/v1/admin/retrain-model")
                if res.status_code == 200: st.success("Model hot-swapped!")
                else: st.error("Training error.")
            except Exception as e: st.error("Backend unreachable.")

# Balanced Main Header Container (Added explicit bottom margin separation)
st.markdown("<h2 style='margin: 0 0 0.2rem 0;'>📈 Intel Daily Stock Price Dashboard</h2>", unsafe_allow_html=True)
st.caption("Monitor live performance metrics and generate machine learning inferences. Updates automatically every 60 seconds.")

# --- AUTOMATIC REFRESH ZONE ---
@st.fragment(run_every=60)
def render_live_dashboard():
    # --- REAL-TIME DATA FETCHING (1D INTERVAL) ---
    intc = yf.Ticker("INTC")
    hist = intc.history(period="1mo", interval="1d")

    # Establish cold-start fallbacks
    live_open, live_close, live_low, live_high, live_hl, live_ma5, live_ma14 = [0.0] * 7

    if not hist.empty:
        latest_day = hist.iloc[-1]
        live_open = float(latest_day['Open'])
        live_close = float(latest_day['Close'])
        live_low = float(latest_day['Low'])
        live_high = float(latest_day['High'])
        live_hl = float(latest_day['High'] - latest_day['Low'])

        if len(hist) >= 5:
            live_ma5 = float(hist['Close'].tail(5).mean())
        if len(hist) >= 14:
            live_ma14 = float(hist['Close'].tail(14).mean())

    # --- SIDE-BY-SIDE CORE LAYOUT ---
    col_prediction, col_analytics = st.columns([1, 1], gap="large")

    # Left Column: Input Matrix (Proportionally Expanded)
    with col_prediction:
        st.markdown("### 🔮 ML Model Input Matrix")

        in_col1, in_col2, in_col3, in_col4 = st.columns(4)
        with in_col1:
            open_price = st.number_input("Open ($)", value=live_open, format="%.2f", key="input_open")
            hl_range = st.number_input("HL Range", value=live_hl, format="%.2f", key="input_hl")
        with in_col2:
            close_price = st.number_input("Close ($)", value=live_close, format="%.2f", key="input_close")
            ma5 = st.number_input("MA 5-Day", value=live_ma5, format="%.2f", key="input_ma5")
        with in_col3:
            high = st.number_input("High ($)", value=live_high, format="%.2f", key="input_high")
            ma14 = st.number_input("MA 14-Day", value=live_ma14, format="%.2f", key="input_ma14")
        with in_col4:
            low = st.number_input("Low ($)", value=live_low, format="%.2f", key="input_low")

        if st.button("Generate Prediction", type="primary", use_container_width=True, key="btn_predict"):
            payload = [{
                "open_price": open_price,
                "high": high,
                "low": low,
                "close_price": close_price,
                "ma5": ma5,
                "ma14": ma14,
                "hl_range": hl_range
            }]

            with st.spinner("Processing..."):
                try:
                    backend_url = "http://127.0.0.1:8000/predict" 
                    response = requests.post(backend_url, json=payload)

                    if response.status_code == 200:
                        prediction_result = response.json()
                        st.metric(
                            label="🔮 Predicted Target Price", 
                            value=f"${round(prediction_result['predictions'][0], 4)}"
                        )
                    else:
                        st.error(f"Backend error ({response.status_code})")
                except requests.exceptions.ConnectionError:
                    st.error("Could not reach FastAPI backend.")

    # Right Column: Live Analytics and Dynamically Enlarged Graph
    with col_analytics:
        st.markdown("### 📊 Real-Time Market Feed")

        if not hist.empty:
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2]
            price_change = current_price - prev_close
            percent_change = (price_change / prev_close) * 100

            # Core Metric Row
            stat_col1, stat_col2, stat_col3 = st.columns(3)
            with stat_col1:
                st.metric(
                    label="Current Close", 
                    value=f"${current_price:.2f}", 
                    delta=f"{price_change:+.2f} ({percent_change:+.2f}%)"
                )
            with stat_col2:
                st.metric(label="Today's High", value=f"${live_high:.2f}")
            with stat_col3:
                st.metric(label="Today's Low", value=f"${live_low:.2f}")

            # Plotly Line Graph (Height scaled up to 245px to comfortably fit the zoomed real estate)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist.index.strftime('%Y-%m-%d'), 
                y=hist['Close'], 
                mode='lines+markers',
                name='Daily Close',
                line=dict(color='#00CC96', width=2.5),
                marker=dict(size=5, color='#636EFA'),
                hovertemplate='<b>Date:</b> %{x}<br><b>Close:</b> $%{y:.2f}<extra></extra>'
            ))

            fig.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                height=245,
                xaxis_title="Trading Date",
                yaxis_title="Price ($)",
                hovermode="x unified",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True, key="live_plotly_chart")
        else:
            st.warning("Yahoo Finance data unavailable.")

# Execute layout initialization
render_live_dashboard()

