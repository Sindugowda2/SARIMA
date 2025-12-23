import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX


# ---- Page Configuration ----
st.set_page_config(page_title="Crop Forecasting Dashboard", layout="wide")

# ---- Sidebar Navigation ----
st.sidebar.title("🌾 Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Prediction", "About", "Contact"])

# 🌄 Custom Background & Theme
page_bg_img = '''
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://img.freepik.com/free-photo/beautiful-rural-landscape-sunset_1112-456.jpg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}

[data-testid="stSidebar"] {
    background-color: rgba(255, 255, 255, 0.85);
    color: #000000;
}

h1, h2, h3, h4 {
    color: #2E8B57;
    text-align: center;
}

div.stButton > button {
    background-color: #2E8B57;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.5em 1em;
    font-size: 16px;
}

div.stButton > button:hover {
    background-color: #228B22;
    color: #fff;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# 🎤 Voice Input Function
def voice_input(prompt_text):
    st.info(prompt_text)
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("🎙️ Listening...")
        audio = recognizer.listen(source, timeout=5)
        try:
            text = recognizer.recognize_google(audio)
            st.success(f"✅ You said: {text}")
            return text
        except sr.UnknownValueError:
            st.error("❌ Sorry, could not understand your voice.")
        except sr.RequestError:
            st.error("⚠️ Voice recognition service error.")
    return None

# 🗣️ Text-to-Speech Function
def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)
    engine.setProperty('voice', engine.getProperty('voices')[1].id)  # female voice
    engine.say(text)
    engine.runAndWait()

# ==========================================
# 🌾 PREDICTION PAGE
# ==========================================
if page == "Prediction":
    st.markdown("<h1 style='color:#1B5E20;'>🌾 AI-Powered Crop Yield Forecasting using SARIMA 🌾</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center; color:#333;'>🎤 Speak. See. Hear. Predict the Future of Agriculture.</h4>", unsafe_allow_html=True)
    st.markdown("---")

    # 📂 Upload Section
    uploaded_file = st.file_uploader("📤 Upload your dataset (CSV format)", type=['csv'])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Dataset successfully uploaded!")

        with st.expander("📊 View Dataset Preview"):
            st.dataframe(df.head())

        required_cols = {'State', 'Crop', 'Crop_Year', 'Yield'}
        if not required_cols.issubset(df.columns):
            st.error(f"⚠️ Your dataset must contain columns: {required_cols}")
        else:
            use_voice = st.checkbox("🎙️ Use Voice Search for Crop and State")
            if use_voice:
                st.write("Say the **state name**:")
                state = voice_input("Speak the state name")
                if state not in df['State'].unique():
                    st.warning("State not found, please type manually below.")
                    state = st.selectbox("🏞️ Select State", df['State'].unique())
                st.write("Say the **crop name**:")
                crop = voice_input("Speak the crop name")
                if crop not in df['Crop'].unique():
                    st.warning("Crop not found, please type manually below.")
                    crop = st.selectbox("🌾 Select Crop", df['Crop'].unique())
            else:
                state = st.selectbox("🏞️ Select State", df['State'].unique())
                crop = st.selectbox("🌾 Select Crop", df['Crop'].unique())

            sub = df[(df['State'] == state) & (df['Crop'] == crop)].copy()
            sub['Year'] = pd.to_datetime(sub['Crop_Year'].astype(str) + '-01-01')
            sub = sub.groupby('Year')['Yield'].mean().reset_index()
            sub.set_index('Year', inplace=True)
            ts = sub['Yield'].asfreq('YS').interpolate()

            model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            result = model.fit(disp=False)

            steps = st.slider("📅 Select number of years to forecast:", 1, 10, 5)
            forecast = result.get_forecast(steps=steps)
            pred = forecast.predicted_mean
            conf = forecast.conf_int()
            future_dates = pd.date_range(start=ts.index[-1] + pd.DateOffset(years=1), periods=steps, freq='YS')
            pred.index = future_dates
            conf.index = future_dates

            # 📈 Plot results
            st.markdown("### 🌾 Forecasted Crop Yield")
            plt.figure(figsize=(10, 5))
            plt.plot(ts, label="Historical", color='#1B5E20')
            plt.plot(pred, label="Forecast", color='#FF8C00', marker='o')
            plt.fill_between(conf.index, conf.iloc[:, 0], conf.iloc[:, 1], color='#FFD580', alpha=0.4)
            plt.title(f"{crop} Yield Forecast for {state}", fontsize=14)
            plt.xlabel("Year")
            plt.ylabel("Yield (tons/ha)")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(plt)

            # 📊 Summary
            st.markdown("### 📊 Yearly Forecast Summary")
            previous_value = ts.iloc[-1]
            for year, value in pred.items():
                change = value - previous_value
                color = "green" if change > 0 else "red"
                trend = "increase" if change > 0 else "decrease"
                line = f"➡️ In **{year.year}**, predicted yield is **<span style='color:{color}; font-weight:bold;'>{value:.2f}</span> tons/ha** (**{trend} of {abs(change):.2f}**)."
                st.markdown(line, unsafe_allow_html=True)
                previous_value = value

            st.markdown("---")
            st.markdown("### 🌟 Insights Summary")
            avg_future = pred.mean()
            best_year = pred.idxmax().year
            worst_year = pred.idxmin().year
            insight_text = (
                f"The **average predicted yield** for next {steps} years is "
                f"**{avg_future:.2f} tons/ha**. The **best year** is {best_year} "
                f"and the **lowest yield year** is {worst_year}."
            )
            st.success(insight_text)
            speak(insight_text)

            result_df = pd.DataFrame({
                "Year": pred.index.year,
                "Predicted_Yield (tons/ha)": pred.values
            })
            st.markdown("### 🌾 Predicted Values Table")
            st.dataframe(result_df.style.background_gradient(cmap='YlGn'))

            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("💾 Download Forecast CSV", data=csv, file_name=f"{crop}_{state}_forecast.csv", mime="text/csv")

            if st.button("❌ Exit Application"):
                st.warning("Application closed. Stop the Streamlit server (Ctrl + C).")
                st.stop()
    else:
        st.info("👆 Please upload your dataset to begin forecasting.")

# ==========================================
# 📊 DASHBOARD PAGE
# ==========================================
elif page == "Dashboard":
    st.title("📊 Forecast Dashboard")
    st.write("Visual summary of yield predictions and trends.")

    # Sample dashboard data
    data = {'Year': [2020, 2021, 2022, 2023, 2024],
            'Yield': [250, 300, 280, 350, 400]}
    df_dash = pd.DataFrame(data)

    fig, ax = plt.subplots()
    ax.plot(df_dash['Year'], df_dash['Yield'], marker='o', color='purple')
    ax.set_xlabel("Year")
    ax.set_ylabel("Yield (tons)")
    ax.set_title("Predicted Crop Yield Over Years")
    st.pyplot(fig)

    col1, col2, col3 = st.columns(3)
    col1.metric("Average Yield", "316 tons")
    col2.metric("Max Yield", "400 tons")
    col3.metric("Growth", "20%")

# ==========================================
# ℹ️ ABOUT PAGE
# ==========================================
elif page == "About":
    st.title("ℹ️ About This Project")
    st.write("""
    This application predicts **crop yield using SARIMA** (Seasonal ARIMA) model.
    It supports **voice-based input** for crop and state names and offers
    **speech feedback** for forecast insights.  
    The project demonstrates AI + Agriculture integration for sustainable farming decisions.
    """)

# ==========================================
# 📬 CONTACT PAGE
# ==========================================
elif page == "Contact":
    st.title("📬 Contact Information")
    st.write("""
    - **Developer:** Sindu B P  
    - **Email:** [sindusindu304@gmail.com](mailto:sindusindu304@gmail.com)  
    - **LinkedIn:** [Sindu’s Profile](https://www.linkedin.com/in/sindu-5b6a92258)  
    - **GitHub:** [SinduGowda2](https://github.com/Sindugowda2)
    """)
