# ------------------------------------------------------------
# ⚡ Electricity Price Prediction Streamlit App (Button Version)
# ------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ------------------------------------------------------------
# Page Setup
# ------------------------------------------------------------
st.set_page_config(page_title="Electricity Price Predictor", page_icon="⚡", layout="centered")
st.title("⚡ Electricity Price Prediction App")

st.markdown(
    "This app uses a trained **LightGBM model** to predict electricity prices based on time-based and numeric features. "
    "Adjust the sliders and click **Predict Price** to see the updated result!"
)

# ------------------------------------------------------------
# Load Model and Scaler
# ------------------------------------------------------------
MODEL_PATH = "best_lgbm.pkl"
SCALER_PATH = "scaler.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file 'best_lgbm.pkl' not found in the current directory.")
    st.stop()

model = joblib.load(MODEL_PATH)

scaler = None
if os.path.exists(SCALER_PATH):
    try:
        scaler = joblib.load(SCALER_PATH)
        st.sidebar.info("Scaler loaded successfully ✅")
    except Exception as e:
        st.sidebar.warning(f"Scaler found but failed to load: {e}")

# ------------------------------------------------------------
# Sidebar Info
# ------------------------------------------------------------
st.sidebar.header("ℹ️ Model Information")
st.sidebar.write("**Model:** LightGBM Regressor")
st.sidebar.write("**Scaler:** StandardScaler (optional)")
st.sidebar.write("**Metric:** R² ≈ 0.994 RMSE ≈ 2.3 MAE ≈ 0.85")
st.sidebar.markdown("---")
st.sidebar.caption("Developed as part of a research project on Electricity Price Forecasting.")

# ------------------------------------------------------------
# User Inputs
# ------------------------------------------------------------
st.subheader("Enter Input Features")

col1, col2 = st.columns(2)
with col1:
    hour = st.slider("Hour of Day (0–23)", 0, 23, 12)
    day_of_week = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 3)
    month = st.slider("Month (1–12)", 1, 12, 6)

with col2:
    demand = st.number_input("Electricity Demand (MW)", min_value=0.0, value=2500.0, step=10.0)
    temperature = st.number_input("Temperature (°C)", min_value=-50.0, max_value=60.0, value=35.0, step=0.5)

# ------------------------------------------------------------
# Feature Engineering
# ------------------------------------------------------------
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)
is_peak = 1 if (7 <= hour <= 10 or 17 <= hour <= 20) else 0
is_night = 1 if (hour < 6 or hour >= 22) else 0

input_dict = {
    "Hour": hour,
    "DayOfWeek": day_of_week,
    "Month": month,
    "Demand": demand,
    "Temperature": temperature,
    "Hour_sin": hour_sin,
    "Hour_cos": hour_cos,
    "IsPeakHour": is_peak,
    "IsNight": is_night
}

input_df = pd.DataFrame([input_dict])

# ------------------------------------------------------------
# Align Features with Model
# ------------------------------------------------------------
try:
    expected_features = list(model.feature_name_)
    for f in expected_features:
        if f not in input_df.columns:
            input_df[f] = 0
    input_df = input_df[expected_features]
except Exception:
    input_df = input_df[input_dict.keys()]

# ------------------------------------------------------------
# Apply Scaler
# ------------------------------------------------------------
if scaler is not None:
    try:
        input_scaled = scaler.transform(input_df)
    except Exception as e:
        st.warning(f"⚠️ Scaler issue: {e}. Using unscaled data.")
        input_scaled = input_df.values
else:
    input_scaled = input_df.values

# ------------------------------------------------------------
# Prediction Trigger Button
# ------------------------------------------------------------
st.markdown("---")
predict_btn = st.button("🔮 Predict Price")

if predict_btn:
    st.info("Checking model input alignment...")

    # --- Debug: Print expected vs actual features
    try:
        model_features = list(model.feature_name_)
        st.write("🧠 Model expects features:", model_features)
        st.write("📊 Input DataFrame columns:", list(input_df.columns))

        # Ensure alignment
        for f in model_features:
            if f not in input_df.columns:
                input_df[f] = 0
        input_df = input_df[model_features]
    except Exception as e:
        st.warning(f"Model has no feature_name_ attr ({e}), using given order.")

    # --- Apply scaler
    if scaler is not None:
        try:
            input_scaled = scaler.transform(input_df)
            st.write("✅ Scaler applied successfully.")
        except Exception as e:
            st.warning(f"⚠️ Scaler failed: {e}. Using raw input.")
            input_scaled = input_df.values
    else:
        input_scaled = input_df.values
        st.write("⚠️ No scaler loaded. Using raw data.")

    # --- Show the final array sent to model
    st.write("🔢 Final numeric input sent to model:")
    st.dataframe(pd.DataFrame(input_scaled, columns=input_df.columns))

    # --- Predict
    try:
        prediction = float(model.predict(input_scaled)[0])
        st.success("✅ Prediction completed successfully!")
        st.metric(label="Predicted Electricity Price (₹/MWh)", value=f"{prediction:.4f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ------------------------------------------------------------
# Debugging Info
# ------------------------------------------------------------
with st.expander("Show Input Data Sent to Model"):
    st.write("📄 Input DataFrame:")
    st.dataframe(input_df)
    st.write("Input shape:", input_df.shape)
