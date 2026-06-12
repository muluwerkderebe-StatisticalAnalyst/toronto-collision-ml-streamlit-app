import pandas as pd
import numpy as np
import os

def clean_toronto_collisions(df):
    """
    Data Cleaning Pipeline based exactly on your Jupyter Notebook steps.
    """
    df = df.copy()
    
    print("Original dataset shape:", df.shape)
    
    # Step 1: Create Target Variable (Exact line from your notebook)
    print("Step 1: Creating target variable NumberOfInvolvedPerson...")
    df['NumberOfInvolvedPerson'] = df.groupby('ACCNUM')['ACCNUM'].transform('count')
    print("Target variable created successfully.")
    
    # Step 2: Missing Data Handling
    print("Step 2: Handling missing values...")
    
    # Features list from your notebook
    features = ['ROAD_CLASS', 'DISTRICT', 'ACCLOC', 'TRAFFCTL', 'VISIBILITY', 
                'LIGHT', 'RDSFCOND', 'ACCLASS', 'IMPACTYPE', 'INVTYPE', 'INVAGE', 
                'INJURY', 'VEHTYPE', 'MANOEUVER', 'DRIVACT', 'DRIVCOND', 'PEDTYPE', 
                'PEDACT', 'PEDCOND', 'CYCLISTYPE', 'CYCACT', 'CYCCOND', 'PEDESTRIAN', 
                'CYCLIST', 'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK', 'TRSN_CITY_VEH', 
                'EMERG_VEH', 'PASSENGER', 'SPEEDING', 'AG_DRIV', 'REDLIGHT', 
                'ALCOHOL', 'DISABILITY']
    
    # Replace NaN and "None" with "No"
    df[features] = df[features].fillna("No").replace("None", "No")
    
    # Handle FATAL_NO
    df['FATAL_NO'] = df['FATAL_NO'].fillna(0)
    
    print("Missing values handled.")
    
    # Step 3: Drop insignificant features
    print("Step 3: Dropping insignificant columns...")
    columns_to_drop = ['_id', 'STREET1', 'STREET2', 'OFFSET', 'INITDIR', 
                      'HOOD_158', 'HOOD_140', 'NEIGHBOURHOOD_158', 'NEIGHBOURHOOD_140']
    
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    print(f"Dropped columns. New shape: {df.shape}")
    
    # Step 4: Drop rows with remaining NaNs in important columns
    print("Step 4: Dropping rows with NaN in key columns...")
    important_cols = ['NumberOfInvolvedPerson'] + [f for f in features if f in df.columns]
    df = df.dropna(subset=important_cols)
    
    print("Final cleaned dataset shape:", df.shape)
    print("Data cleaning completed successfully.")
    
    return df


def save_cleaned_data(df, output_path="data/processed/collisions_cleaned.csv"):
    """Save the cleaned dataset"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")
    return output_path