import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Crop Yield Forecasting",
    page_icon="🌾",
    layout="wide"
)

# ---------------- Sidebar ----------------
st.sidebar.title("🌾 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Dashboard", "Prediction", "About", "Contact"]
)

# ---------------- Background Theme ----------------
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #F5FFF5;
    }
    h1, h2, h3 {
        color: #1B5E20;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- DASHBOARD ----------------
if page == "Dashboard":
    st.title("🌾 Crop Yield Forecasting Dashboard")

    st.markdown("""
    ### 📌 Project Overview
    This application predicts **future crop yield trends** using  
    **SARIMA time-series forecasting**.

    **Key Features**
    - Upload agricultural dataset
    - Select state & crop
    - Forecast future yield
    - View graphs & tables
    - Download prediction results
    """)

# ---------------- PREDICTION ----------------
elif page == "Prediction":
    st.title("📈 Crop Yield Prediction")

    uploaded_file = st.file_uploader(
        "Upload CSV Dataset",
        type=["csv"]
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        required_cols = {"State", "Crop", "Crop_Year", "Yield"}
        if not required_cols.issubset(df.columns):
            st.error(
                "Dataset must contain columns: State, Crop, Crop_Year, Yield"
            )
        else:
            st.success("Dataset uploaded successfully!")

            state = st.selectbox("Select State", df["State"].unique())
            crop = st.selectbox("Select Crop", df["Crop"].unique())

            sub = df[(df["State"] == state) & (df["Crop"] == crop)].copy()

            sub["Year"] = pd.to_datetime(
                sub["Crop_Year"].astype(str) + "-01-01"
            )
            sub = sub.groupby("Year")["Yield"].mean().reset_index()
            sub.set_index("Year", inplace=True)

            ts = sub["Yield"].asfreq("YS").interpolate()

            model = SARIMAX(
                ts,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12)
            )
            result = model.fit(disp=False)

            steps = st.slider(
                "Select number of years to forecast",
                1, 10, 5
            )

            forecast = result.get_forecast(steps=steps)
            pred = forecast.predicted_mean
            conf = forecast.conf_int()

            future_dates = pd.date_range(
                start=ts.index[-1] + pd.DateOffset(years=1),
                periods=steps,
                freq="YS"
            )
            pred.index = future_dates
            conf.index = future_dates

            # -------- Plot --------
            st.subheader("📊 Yield Forecast Graph")

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(ts, label="Historical", color="green")
            ax.plot(pred, label="Forecast", color="orange")
            ax.fill_between(
                conf.index,
                conf.iloc[:, 0],
                conf.iloc[:, 1],
                color="orange",
                alpha=0.3
            )
            ax.set_xlabel("Year")
            ax.set_ylabel("Yield")
            ax.legend()
            st.pyplot(fig)

            # -------- Text Output --------
            st.subheader("📋 Year-wise Forecast")
            for year, value in pred.items():
                st.write(
                    f"📅 **{year.year}** → {value:.2f} tons/ha"
                )

            # -------- Table --------
            result_df = pd.DataFrame({
                "Year": pred.index.year,
                "Predicted Yield (tons/ha)": pred.values
            })

            st.subheader("📑 Prediction Table")
            st.dataframe(result_df)

            # -------- Download --------
            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Forecast CSV",
                csv,
                f"{crop}_{state}_forecast.csv",
                "text/csv"
            )

    else:
        st.info("Please upload a dataset to continue.")

# ---------------- ABOUT ----------------
elif page == "About":
    st.title("ℹ️ About This Project")

    st.markdown("""
    **Crop Yield Forecasting System**

    - Uses SARIMA time-series model
    - Helps farmers & planners predict future yields
    - Built using Python & Streamlit
    - Academic Data Science Project
    """)

# ---------------- CONTACT ----------------
elif page == "Contact":
    st.title("📞 Contact")

    st.markdown("""
    **Developer:** Sindu B P  
    **Course:** MCA (Data Science)  
    **Email:** sindusindu304@gmail.com  

    Thank you for visiting this application 🌱
    """)
