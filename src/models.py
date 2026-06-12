import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
import joblib
import os
from src.feature_engineering import engineer_features

FEATURE_COLUMNS = [
    'ROAD_CLASS', 'DISTRICT', 'VISIBILITY', 'LIGHT', 'RDSFCOND',
    'ACCLASS', 'IMPACTYPE', 'INVTYPE', 'VEHTYPE', 'TIME',
    'Year', 'Month', 'DayOfWeek', 'IsWeekend', 'TimeOfDay',
    'HasFatality', 'HighRiskCondition'
]

def train_and_compare_models(data_path="data/processed/collisions_cleaned.csv"):
    if not os.path.exists(data_path):
        print("Processed data not found. Please run Data Cleaning first.")
        return None, None

    df = pd.read_csv(data_path)
    
    print("Applying Feature Engineering...")
    df = engineer_features(df)
    
    target = 'NumberOfInvolvedPerson'
    features = FEATURE_COLUMNS
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    categorical_features = ['ROAD_CLASS', 'DISTRICT', 'VISIBILITY', 'LIGHT', 'RDSFCOND', 
                           'ACCLASS', 'IMPACTYPE', 'INVTYPE', 'VEHTYPE', 'TimeOfDay']
    
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)],
        remainder='passthrough'
    )
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "XGBoost": xgb.XGBRegressor(n_estimators=300, learning_rate=0.08, max_depth=7, random_state=42),
        "Neural Network": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }
    
    results = []
    best_model = None
    best_rmse = float('inf')
    best_name = ""
    
    print("\nTraining and Comparing Models...\n")
    
    for name, model in models.items():
        print(f"Training {name}...")
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        # Calculate RMSE without sklearn keyword compatibility issues
        mse = np.mean((y_test - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            "Model": name,
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            "R2": round(r2, 4)
        })
        
        print(f"{name:20} -> RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = pipeline
            best_name = name
    
    results_df = pd.DataFrame(results)
    print("\n=== Model Comparison Results ===")
    print(results_df.sort_values(by="RMSE"))
    
    model_path = "models/best_model.pkl"
    os.makedirs("models", exist_ok=True)
    # Save with compression to reduce repository size while keeping load compatibility
    joblib.dump(best_model, model_path, compress=3)
    print(f"\nBest Model Saved: {best_name} (RMSE: {best_rmse:.4f})")
    
    return best_model, results_df


def load_best_model():
    model_path = "models/best_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None