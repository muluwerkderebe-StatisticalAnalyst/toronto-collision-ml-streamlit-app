import pandas as pd
import os

def load_raw_data():
    path = "data/raw/toronto_collisions.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        # Alternative path check
        alt_path = os.path.join("data", "raw", "toronto_collisions.csv")
        if os.path.exists(alt_path):
            return pd.read_csv(alt_path)
    return None


def load_processed_data():
    path = "data/processed/collisions_cleaned.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    
    # Try absolute path (Windows safe)
    abs_path = os.path.abspath("data/processed/collisions_cleaned.csv")
    if os.path.exists(abs_path):
        return pd.read_csv(abs_path)
    
    return None