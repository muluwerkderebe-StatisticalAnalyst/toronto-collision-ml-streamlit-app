import streamlit as st
import pandas as pd
from src.data_loader import load_raw_data
from src.preprocessing import clean_toronto_collisions, save_cleaned_data

st.set_page_config(page_title="Data Cleaning", layout="wide")
st.title("Data Cleaning")
st.markdown("### Replicating the cleaning process from your Jupyter Notebook")

# Load raw data
raw_df = load_raw_data()

if raw_df is None:
    st.error("Raw dataset not found. Please place your file in data/raw/toronto_collisions.csv")
    st.stop()

st.write(f"**Original Dataset Shape:** {raw_df.shape}")

if st.button("Run Data Cleaning Pipeline"):
    with st.spinner("Running data cleaning..."):
        cleaned_df = clean_toronto_collisions(raw_df)
        save_cleaned_data(cleaned_df)
        
        st.subheader("Cleaned Dataset Preview")
        st.dataframe(cleaned_df.head(10), use_container_width=True)
        
        st.subheader("Final Shape")
        st.write(cleaned_df.shape)
        
        st.success("Data cleaning completed successfully.")

# Check if processed file exists
st.subheader("Processed Data Status")
try:
    processed = pd.read_csv("data/processed/collisions_cleaned.csv")
    st.success(f"Processed file found with {processed.shape[0]:,} rows")
    with st.expander("Show Processed Data Sample"):
        st.dataframe(processed.head(), use_container_width=True)
except FileNotFoundError:
    st.warning("No processed file found yet.")