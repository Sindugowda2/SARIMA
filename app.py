import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="SARIMA Agriculture Forecasting",
    layout="centered"
)

st.title("🌾 SARIMA Model for Agriculture Forecasting")
st.write("Time Series Forecasting using SARIMA Model")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file (Date & Value columns)",
    type=["csv"]
)

# ---------------- DATA LOAD ----------------
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("📊 Raw Dataset")
    st.dataframe(data.head())

    # Column selection
    date_col = st.selectbox("Select Date Column", data.columns)
    value_col = st.selectbox("Select Target Column", data.columns)

    # Data preprocessing
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values(by=date_col)
    data.set_index(date_col, inplace=True)

    ts = data[value_col]

    st.subheader("📈 Time Series Plot")
    st.line_chart(ts)

    # ---------------- SARIMA PARAMETERS ----------------
    st.sidebar.header("SARIMA Parameters")

    p = st.sidebar.slider("p (AR)", 0, 5, 1)
    d = st.sidebar.slider("d (Differencing)", 0, 2, 1)
    q = st.sidebar.slider("q (MA)", 0, 5, 1)

    P = st.sidebar.slider("P (Seasonal AR)", 0, 3, 1)
    D = st.sidebar.slider("D (Seasonal Diff)", 0, 2, 1)
    Q = st.sidebar.slider("Q (Seasonal MA)", 0, 3, 1)
    s = st.sidebar.selectbox("Seasonal Period (s)", [6, 12])

    forecast_steps = st.sidebar.number_input(
        "Forecast Months",
        min_value=1,
        max_value=36,
        value=12
    )

    # ---------------- MODEL TRAIN ----------------
    if st.button("🚀 Train SARIMA Model"):
        with st.spinner("Training SARIMA model..."):
            try:
                model = SARIMAX(
                    ts,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, s),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )

                model_fit = model.fit(disp=False)

                st.success("✅ Model trained successfully!")

                # ---------------- FORECAST ----------------
                forecast = model_fit.forecast(steps=forecast_steps)

                forecast_index = pd.date_range(
                    start=ts.index[-1],
                    periods=forecast_steps + 1,
                    freq="M"
                )[1:]

                forecast_series = pd.Series(
                    forecast.values,
                    index=forecast_index
                )

                st.subheader("📉 Forecast Results")
                st.line_chart(
                    pd.concat([ts, forecast_series], axis=0)
                )

                st.write("### Forecasted Values")
                st.dataframe(forecast_series.reset_index().rename(
                    columns={"index": "Date", 0: "Forecast"}
                ))

            except Exception as e:
                st.error("❌ Error while training model")
                st.code(str(e))

else:
    st.info("⬅️ Upload an agriculture time series CSV file to begin")
