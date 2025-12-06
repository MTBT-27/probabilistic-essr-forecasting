import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os

# --------------------------------------------------------------------------------
## 1. Configuration and Data Loading
# --------------------------------------------------------------------------------

INPUT_CSV_PATH = 'sample_cleaned_data.csv'

# Unit conversion factors (kWh, m3 -> MJ)
KWH_TO_MJ = 3.6
M3_TO_MJ = 45.0 

print(f"--- 1. Data Loading and Preprocessing (Monthly, MJ Unit Conversion) ---")

try:
    # Load CSV file and set 'datetime' as the index
    original_df = pd.read_csv(INPUT_CSV_PATH, parse_dates=['datetime'])
    original_df.set_index('datetime', inplace=True)
except FileNotFoundError:
    print(f"Error: {INPUT_CSV_PATH} not found. Please run generate_mock_data.py.")
    exit()

# --------------------------------------------------------------------------------
# 2. Monthly Aggregation, Unit Conversion, and Weather Data Processing
#    Replicates the previous processing steps to convert all household data 
#    to monthly, MJ units.
# --------------------------------------------------------------------------------

def process_to_monthly_mj(df):
    
    # Column mapping for unit conversion
    KWH_COLUMNS = ["elect_purchase(kWh)", "elect_sale(kWh)", "PV_gene(kWh)", "FC_gene(kWh)", "elect_cons(kWh)"]
    M3_COLUMNS = ["gas_cons(m3)"]
    RENAME_MAPPING = {
        "elect_purchase(kWh)": "elect_purchase(MJ)", "elect_sale(kWh)": "elect_sale(MJ)",
        "PV_gene(kWh)": "PV_gene(MJ)", "FC_gene(kWh)": "FC_gene(MJ)",
        "elect_cons(kWh)": "elect_cons(MJ)", "gas_cons(m3)": "gas_cons(MJ)"
    }
    
    # Separate HEMS and weather data columns
    hems_cols = [col for col in KWH_COLUMNS + M3_COLUMNS if col in df.columns]
    weather_cols_30min = ['temp', 'WNDSPD', 'RHUM', 'PRCRIN_30MIN', 'GLBRAD_30MIN']
    weather_cols = [col for col in weather_cols_30min if col in df.columns]

    # --- (A) Monthly Aggregation and Unit Conversion for HEMS Data ---
    # Group by ID and resample to monthly sum
    hems_monthly = df.groupby('ID')[hems_cols].resample('MS').sum().reset_index(level='ID')
    
    # Execute unit conversion
    for col in [c for c in KWH_COLUMNS if c in hems_monthly.columns]:
        hems_monthly[col] = hems_monthly[col] * KWH_TO_MJ
    for col in [c for c in M3_COLUMNS if c in hems_monthly.columns]:
        hems_monthly[col] = hems_monthly[col] * M3_TO_MJ
        
    hems_monthly.rename(columns=RENAME_MAPPING, inplace=True)

    # --- (B) Monthly Aggregation for Weather Data (Independent of HEMS data) ---
    # Weather data is identical across all households for a given timestamp.
    # Group by timestamp (index) and take the first value to remove ID dependency.
    weather_data_raw = df.groupby(df.index).first()[weather_cols]

    # Define monthly aggregation methods
    weather_agg_methods = {
        'temp': 'mean', 'WNDSPD': 'mean', 'RHUM': 'mean',
        'PRCRIN_30MIN': 'sum', 'GLBRAD_30MIN': 'sum'
    }
    
    # Resample and aggregate to monthly data
    weather_monthly_df = weather_data_raw.resample('MS').agg(weather_agg_methods)
    
    # Rename weather columns 
    WEATHER_RENAME_MAPPING = {
        'temp': 'Monthly_Avg_Temp', 'WNDSPD': 'Monthly_Avg_WindSpeed', 
        'RHUM': 'Monthly_Avg_RelHumidity', 'PRCRIN_30MIN': 'Monthly_Total_Precipitation', 
        'GLBRAD_30MIN': 'Monthly_Total_GlobalRad'
    }
    weather_monthly_df.rename(columns=WEATHER_RENAME_MAPPING, inplace=True)
    
    # --- (C) Merge and Final Shaping ---
    # Merge monthly HEMS data with monthly weather data
    processed_df = hems_monthly.merge(weather_monthly_df, left_index=True, right_index=True, how='left')
    processed_df.dropna(inplace=True)
    
    # Format the timestamp column
    processed_df.reset_index(names=['timestamp'], inplace=True)
    processed_df['timestamp'] = processed_df['timestamp'].dt.strftime('%Y-%m')
    
    # Reorder columns: ID, timestamp first
    cols = [col for col in processed_df.columns if col not in ['ID', 'timestamp']]
    processed_df = processed_df[['ID', 'timestamp'] + cols]
    
    return processed_df

combined_df_monthly = process_to_monthly_mj(original_df)
print(f"✅ Data preprocessing completed. Data size: {combined_df_monthly.shape}")

# --------------------------------------------------------------------------------
## 3. Selection of Representative Household (Median Consumption)
# --------------------------------------------------------------------------------

print("\n--- 3. Selection of Representative Household ---")

# Calculate total consumption (electricity + gas) per household
df_consumption = combined_df_monthly.groupby('ID')[['elect_cons(MJ)', 'gas_cons(MJ)']].sum().reset_index()
df_consumption['Total_Consumption_MJ'] = df_consumption['elect_cons(MJ)'] + df_consumption['gas_cons(MJ)']
median_consumption = df_consumption['Total_Consumption_MJ'].median()

# Identify the household with the absolute difference closest to the median
df_consumption['Diff_from_Median'] = np.abs(df_consumption['Total_Consumption_MJ'] - median_consumption)
representative_household_id = df_consumption.sort_values('Diff_from_Median').iloc[0]['ID']

print(f"Selected Representative Household ID (Median): {representative_household_id}")

# Extract and shape the representative household's data
representative_data = combined_df_monthly[combined_df_monthly['ID'] == representative_household_id].copy()
representative_data['timestamp'] = pd.to_datetime(representative_data['timestamp'])
representative_data.set_index('timestamp', inplace=True)

# Filter for variables for plotting
plot_vars = ['PV_gene(MJ)', 'elect_cons(MJ)', 'gas_cons(MJ)']
representative_data = representative_data[plot_vars]

# --------------------------------------------------------------------------------
## 4. Plotting Time Series Trends (Thesis Style)
# --------------------------------------------------------------------------------

def plot_separate_timeseries_trends_for_thesis(df, household_id):
    """Plots three time series trends as vertically stacked subplots (for thesis)"""
    
    PLOT_VARS_MAP = {
        'PV_gene(MJ)': {'title': '(a) Photovoltaic Generation (PV)', 'ylabel': 'PV Generation [MJ]'},
        'elect_cons(MJ)': {'title': '(b) Electricity Consumption', 'ylabel': 'Electricity Consumption [MJ]'},
        'gas_cons(MJ)': {'title': '(c) Gas Consumption', 'ylabel': 'Gas Consumption [MJ]'},
    }
    plot_vars
