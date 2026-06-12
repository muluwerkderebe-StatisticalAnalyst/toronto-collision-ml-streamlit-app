import pandas as pd
import numpy as np

def engineer_features(df):
    """
    Feature Engineering Pipeline - Made robust for both training and prediction.
    """
    df = df.copy()
    
    # ====================== TEMPORAL FEATURES ======================
    if 'DATE' in df.columns:
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
        df['Year'] = df['DATE'].dt.year
        df['Month'] = df['DATE'].dt.month
        df['DayOfWeek'] = df['DATE'].dt.dayofweek
        df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    else:
        df['Year'] = 2023
        df['Month'] = 6
        df['DayOfWeek'] = 0
        df['IsWeekend'] = 0
    
    # Time of Day
    if 'TIME' not in df.columns:
        df['TIME'] = 14
    else:
        df['TIME'] = df['TIME'].fillna(14)

    df['TimeOfDay'] = pd.cut(df['TIME'], 
                            bins=[0, 6, 12, 18, 24], 
                            labels=['Night', 'Morning', 'Afternoon', 'Evening'], 
                            right=False)
    
    # ====================== FATALITY FEATURE (Safe Handling) ======================
    if 'FATAL_NO' not in df.columns:
        df['FATAL_NO'] = 0
    else:
        df['FATAL_NO'] = df['FATAL_NO'].fillna(0)

    df['HasFatality'] = (df['FATAL_NO'] >= 1).astype(int)
    
    # ====================== HIGH RISK CONDITION ======================
    df['HighRiskCondition'] = 0
    if 'VISIBILITY' in df.columns:
        df['HighRiskCondition'] = ((df['VISIBILITY'] == 'Poor') | 
                                  (df.get('LIGHT', '') == 'Dark') | 
                                  (df.get('RDSFCOND', '') == 'Wet')).astype(int)
    
    print("Feature engineering completed. Shape:", df.shape)
    return df