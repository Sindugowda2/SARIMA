import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="SARIMA in Agriculture",
    layout="wide"
)

# ---------------- TITLE ----------------
st.title("🌾 SARIMA Model for Agricultural Forecasting")
st.markdown("Forecast crop production using **Seasonal ARIMA (SARIMA)**")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

# ---------------- DATA LOAD ----------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # ---------------- COLUMN SELECTION ----------------
    st.sidebar.subheader("Select Columns")
    date_col = st.sidebar.selectbox("Date column", df.columns)
    value_col = st.sidebar.selectbox("Value column", df.columns)

    # ---------------- PREPROCESS ----------------
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    ts = df[value_col]

    st.subheader("📈 Time Series Plot")
    fig, ax = plt.subplots()
    ax.plot(ts, label="Actual Data")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    st.pyplot(fig)

    # ---------------- SARIMA PARAMETERS ----------------
    st.sidebar.subheader("SARIMA Parameters")
    p = st.sidebar.number_input("p", 0, 5, 1)
    d = st.sidebar.number_input("d", 0, 2, 1)
    q = st.sidebar.number_input("q", 0, 5, 1)

    P = st.sidebar.number_input("P", 0, 5, 1)
    D = st.sidebar.number_input("D", 0, 2, 1)
    Q = st.sidebar.number_input("Q", 0, 5, 1)
    s = st.sidebar.number_input("Seasonal Period (s)", 1, 24, 12)

    # ---------------- MODEL TRAIN ----------------
    if st.sidebar.button("Train SARIMA Model"):
        with st.spinner("Training model..."):
            try:
                model = SARIMAX(
                    ts,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, s)
                )
                result = model.fit()

                st.success("✅ Model trained successfully!")

                # ---------------- FORECAST ----------------
                steps = st.slider("Forecast steps", 1, 24, 12)
                forecast = result.forecast(steps=steps)

                st.subheader("🔮 Forecast Output")
                st.write(forecast)

                # ---------------- PLOT FORECAST ----------------
                fig2, ax2 = plt.subplots()
                ax2.plot(ts, label="Actual")
                ax2.plot(forecast, label="Forecast", color="red")
                ax2.legend()
                st.pyplot(fig2)

            except Exception as e:
                st.error("❌ Error while training model")
                st.code(str(e))

else:
    st.info("⬅ Upload a CSV file to begin")
