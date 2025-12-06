import pandas as pd
import numpy as np
import os
from datetime import timedelta # Although imported, not strictly used in final code logic

# --------------------------------------------------------------------------------
# --- Configuration Constants ---
# --------------------------------------------------------------------------------

CSV_FILE_NAME = 'sample_cleaned_data.csv'

# Unit conversion factors (kWh, m3 -> MJ)
KWH_TO_MJ = 3.6
M3_TO_MJ = 45.0 

# Prediction period: April 2023 ('2023-04') to March 2024 ('2024-03')
TEST_START_DATE = '2023-04'
TEST_END_DATE = '2024-03'
PREDICTION_LENGTH = 12

# Data period for the required lag (12 months prior)
# Data from April 2022 is needed to forecast April 2023 (m=12)
LAG_START_DATE = '2022-04' 
LAG_END_DATE = '2023-03'

TARGET_COLUMNS = [
    'PV_gene(MJ)', 
    'FC_gene(MJ)', 
    'elect_cons(MJ)', 
    'gas_cons(MJ)'
]

# --------------------------------------------------------------------------------
# --- 1. Data Loading and Preprocessing (Monthly, MJ Unit Conversion) ---
# --------------------------------------------------------------------------------

def load_and_preprocess_data(file_name):
    """Loads CSV, converts data to monthly/MJ units, and reshapes."""
    
    print(f"--- Data Loading: {file_name} ---")
    try:
        df = pd.read_csv(file_name, parse_dates=['datetime'])
        df.set_index('datetime', inplace=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: {file_name} not found. Please run generate_mock_data.py beforehand.")

    # --- Monthly Aggregation and Unit Conversion for HEMS Data ---
    KWH_COLUMNS = ["elect_purchase(kWh)", "elect_sale(kWh)", "PV_gene(kWh)", "FC_gene(kWh)", "elect_cons(kWh)"]
    M3_COLUMNS = ["gas_cons(m3)"]
    RENAME_MAPPING = {
        "elect_purchase(kWh)": "elect_purchase(MJ)", "elect_sale(kWh)": "elect_sale(MJ)",
        "PV_gene(kWh)": "PV_gene(MJ)", "FC_gene(kWh)": "FC_gene(MJ)",
        "elect_cons(kWh)": "elect_cons(MJ)", "gas_cons(m3)": "gas_cons(MJ)"
    }
    
    # Group by ID and resample to monthly sum
    hems_monthly = df.groupby('ID')[KWH_COLUMNS + M3_COLUMNS].resample('MS').sum().reset_index(level='ID')
    
    # Unit Conversion
    for col in [c for c in KWH_COLUMNS if c in hems_monthly.columns]:
        hems_monthly[col] = hems_monthly[col] * KWH_TO_MJ
    for col in [c for c in M3_COLUMNS if c in hems_monthly.columns]:
        hems_monthly[col] = hems_monthly[col] * M3_TO_MJ
    hems_monthly.rename(columns=RENAME_MAPPING, inplace=True)

    # --- Monthly Aggregation for Weather Data (Mean/Sum) ---
    weather_cols_raw = ['temp', 'WNDSPD', 'RHUM', 'PRCRIN_30MIN', 'GLBRAD_30MIN']
    # Group by datetime index to get unique weather readings for each 30-min period, then aggregate
    weather_data_raw = df.groupby(df.index).first()[weather_cols_raw]

    weather_agg_methods = {'temp': 'mean', 'WNDSPD': 'mean', 'RHUM': 'mean', 'PRCRIN_30MIN': 'sum', 'GLBRAD_30MIN': 'sum'}
    weather_monthly_df = weather_data_raw.resample('MS').agg(weather_agg_methods)
    
    WEATHER_RENAME_MAPPING = {
        'temp': 'Monthly_Avg_Temp', 'WNDSPD': 'Monthly_Avg_WindSpeed', 
        'RHUM': 'Monthly_Avg_RelHumidity', 'PRCRIN_30MIN': 'Monthly_Total_Precipitation', 
        'GLBRAD_30MIN': 'Monthly_Total_GlobalRad'
    }
    weather_monthly_df.rename(columns=WEATHER_RENAME_MAPPING, inplace=True)
    
    # --- Merge and Final Shaping ---
    processed_df = hems_monthly.merge(weather_monthly_df, left_index=True, right_index=True, how='left')
    processed_df.dropna(inplace=True)
    processed_df.reset_index(names=['timestamp'], inplace=True)
    
    # Convert timestamp column to 'YYYY-MM' string format
    processed_df['timestamp'] = processed_df['timestamp'].dt.strftime('%Y-%m')
    
    print(f"✅ Data preprocessing completed. Data size: {processed_df.shape}")
    return processed_df

# --------------------------------------------------------------------------------
# --- 2. Seasonal Naive Forecasting Execution ---
# --------------------------------------------------------------------------------

def run_seasonal_naive(df):
    
    # Extract lag data (actual values from 12 months prior)
    lag_df = df[
        (df['timestamp'] >= LAG_START_DATE) & (df['timestamp'] <= LAG_END_DATE)
    ].copy()
    
    # The SN forecast uses only the actual values of TARGET_COLUMNS
    lag_df = lag_df[['ID', 'timestamp'] + TARGET_COLUMNS]

    # --- Creating the Forecast DataFrame ---
    
    # 1. Copy the lag data to form the base of the new prediction DataFrame
    pred_df = lag_df.copy()
    
    # 2. Advance the timestamp column by 12 months (to match the prediction point)
    pred_df['timestamp'] = pd.to_datetime(pred_df['timestamp'], format='%Y-%m')
    pred_df['timestamp'] = (
        pred_df['timestamp'].dt.to_period('M') + PREDICTION_LENGTH
    ).dt.to_timestamp()
    
    # 3. Rename columns to indicate they are the predicted values
    rename_mapping = {col: f'{col}_sn_pred' for col in TARGET_COLUMNS}

    # Keep only the target columns and rename them
    pred_df.rename(columns=rename_mapping, inplace=True)
    pred_df = pred_df[['ID', 'timestamp'] + list(rename_mapping.values())]
    pred_df.reset_index(drop=True, inplace=True)
    
    # Convert timestamp back to 'YYYY-MM' string
    pred_df['timestamp'] = pred_df['timestamp'].dt.strftime('%Y-%m')
    
    # 4. Verify the prediction DataFrame
    print(f"\n--- Seasonal Naive (m=12) Forecasting Execution ---")
    print(f"✅ SN prediction DataFrame created. Size: {pred_df.shape}")
    
    # 5. Save Results (Using .pkl format)
    output_pkl_path = "sn_predictions.pkl"
    pred_df.to_pickle(output_pkl_path)
    print(f"✅ Prediction results shaped and saved to '{output_pkl_path}'.")

    # 6. Display prediction results summary
    print("\n--- SN Prediction Results (First 5 rows) ---")
    display(pred_df.head())
    
    return pred_df

# --------------------------------------------------------------------------------
# --- Execution and Error Check ---
# --------------------------------------------------------------------------------

try:
    processed_data = load_and_preprocess_data(CSV_FILE_NAME)
    if not processed_data.empty:
        run_seasonal_naive(processed_data)
        print("\n=== Seasonal Naive Prediction Execution and Error Check Completed. ===")
    else:
        print("\n=== SN Prediction skipped due to data processing interruption. ===")
except Exception as e:
    print(f"\nFatal Error occurred: {e}")
    print("\n=== Error occurred during Seasonal Naive Prediction execution. ===")
