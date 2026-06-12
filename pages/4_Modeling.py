import importlib
import streamlit as st
import pandas as pd
from src.data_loader import load_processed_data
import src.feature_engineering as feature_engineering_module
import src.models as models_module

st.set_page_config(page_title="Modeling", layout="wide")
st.title("Modeling")
st.markdown("### Model Training, Evaluation, and Selection")

# Load processed data
df = load_processed_data()

if df is None:
    st.error("Processed data not found. Please go to Data Cleaning page and run the cleaning pipeline first.")
    st.stop()

st.write(f"**Dataset Shape:** {df.shape[0]:,} rows and {df.shape[1]} columns")

# ====================== MODEL TRAINING SECTION ======================
st.subheader("Train and Compare Regression Models")

st.markdown("""
This will:
- Apply feature engineering
- Train 4 different regression models
- Compare them using RMSE, MAE, and R²
- Automatically save the best performing model
""")

if st.button("Train Models and Compare", type="primary"):
    with st.spinner("Training models... This may take 1-2 minutes depending on your system."):
        importlib.reload(feature_engineering_module)
        importlib.reload(models_module)
        best_model, results_df = models_module.train_and_compare_models()
        
        if best_model is not None:
            st.success("Model training completed successfully!")
            
            # Show comparison table
            st.subheader("Model Comparison Results")
            st.dataframe(results_df.sort_values(by="RMSE"), use_container_width=True)
            
            # Highlight best model
            best_model_name = results_df.loc[results_df['RMSE'].idxmin(), 'Model']
            st.info(f"**Best Model:** {best_model_name} (selected based on lowest RMSE)")
            
            # Show best model performance
            st.subheader("Best Model Performance")
            best_row = results_df.loc[results_df['RMSE'].idxmin()]
            col1, col2, col3 = st.columns(3)
            col1.metric("RMSE", f"{best_row['RMSE']:.4f}")
            col2.metric("MAE", f"{best_row['MAE']:.4f}")
            col3.metric("R² Score", f"{best_row['R2']:.4f}")

# ====================== CURRENT MODEL STATUS ======================
st.subheader("Current Best Model Status")

model = models_module.load_best_model()

if model is not None:
    st.success("Best model is loaded and ready for prediction.")
    with st.expander("Model Details"):
        st.write("Model Type: Pipeline with Best Regression Model")
        st.write("Saved at: `models/best_model.pkl`")
else:
    st.warning("No trained model found yet. Please train models above.")

# ====================== INSTRUCTIONS ======================
with st.expander("How to Proceed"):
    st.markdown("""
    1. Click **"Train Models and Compare"** to run model training.
    2. Review the performance table (lowest RMSE wins).
    3. Once done, go to the **Prediction** page to use the best model.
    4. You can re-run training anytime to compare again.
    """)

st.success("Modeling page ready. Proceed to Prediction after training the models.")