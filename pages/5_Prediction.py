import streamlit as st
import pandas as pd
import importlib
from src.data_loader import load_processed_data
import src.feature_engineering as feature_engineering_module
import src.models as models_module

st.set_page_config(page_title="Prediction", layout="wide")
st.title("Prediction")
st.markdown("### Predict Number of Individuals Involved in a Collision")

# Load data for dropdown options
df = load_processed_data()

if df is None:
    st.error("Processed data not found. Please run Data Cleaning first.")
    st.stop()

# Reload feature engineering and models modules
importlib.reload(feature_engineering_module)
importlib.reload(models_module)

# Load the best trained model
model = models_module.load_best_model()

if model is None:
    st.error("No trained model found. Please go to Modeling page and train models first.")
    st.stop()

st.success("Best model loaded successfully.")

# ====================== USER INPUTS ======================
st.subheader("Enter Collision Details")

col1, col2 = st.columns(2)

with col1:
    road_class = st.selectbox("Road Class", sorted(df['ROAD_CLASS'].dropna().unique()))
    district = st.selectbox("District", sorted(df['DISTRICT'].dropna().unique()))
    visibility = st.selectbox("Visibility", sorted(df['VISIBILITY'].dropna().unique()))
    light = st.selectbox("Light Condition", sorted(df['LIGHT'].dropna().unique()))
    accclass = st.selectbox("Accident Class", sorted(df['ACCLASS'].dropna().unique()))
    impact_type = st.selectbox("Impact Type", sorted(df['IMPACTYPE'].dropna().unique()))

with col2:
    inv_type = st.selectbox("Involvement Type", sorted(df['INVTYPE'].dropna().unique()))
    vehicle_type = st.selectbox("Vehicle Type", sorted(df['VEHTYPE'].dropna().unique()))
    road_surface = st.selectbox("Road Surface Condition", sorted(df['RDSFCOND'].dropna().unique()))
    time_hour = st.slider("Time (Hour of Day)", 0, 23, 14)
    input_date = st.date_input("Date", value=pd.to_datetime("2023-06-01"))

if st.button("Predict Number of Individuals", type="primary"):
    with st.spinner("Making prediction..."):
        try:
            # Prepare input data
            input_df = pd.DataFrame([{
                'ROAD_CLASS': road_class,
                'DISTRICT': district,
                'VISIBILITY': visibility,
                'LIGHT': light,
                'ACCLASS': accclass,
                'IMPACTYPE': impact_type,
                'INVTYPE': inv_type,
                'VEHTYPE': vehicle_type,
                'RDSFCOND': road_surface,
                'TIME': time_hour,
                'DATE': pd.to_datetime(input_date),
                'FATAL_NO': 0   # Important: Add this column for feature engineering
            }])
            
            # Apply feature engineering
            input_df = feature_engineering_module.engineer_features(input_df)

            # Restrict to the exact features used during training
            feature_columns = models_module.FEATURE_COLUMNS
            missing_features = [col for col in feature_columns if col not in input_df.columns]
            if missing_features:
                raise ValueError(f"Missing engineered feature columns: {', '.join(missing_features)}")
            input_df = input_df[feature_columns]
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            
            st.success(f"**Predicted Number of Individuals Involved: {round(prediction)}**")
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.info("Try retraining the model in the Modeling page.")

# ====================== MODEL INFORMATION ======================
with st.expander("Model Information"):
    st.write("**Best Model:** Loaded from `models/best_model.pkl`")
    st.write("**Features Used:** ROAD_CLASS, DISTRICT, VISIBILITY, LIGHT, etc. + Engineered Features")
    st.write("**Target Variable:** NumberOfInvolvedPerson")

st.info("Tip: Make sure you have successfully trained the models in the Modeling page first.")