import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os

# --------------------------------------------------------------------------------
## 1. 設定とデータ読み込み
# --------------------------------------------------------------------------------

INPUT_CSV_PATH = 'sample_cleaned_data.csv'

# 単位変換係数 (元のコードから引用 - kWh, m3 -> MJ)
KWH_TO_MJ = 3.6
M3_TO_MJ = 45.0 

print(f"--- 1. データ読み込みと前処理（月次・MJ単位への変換） ---")

try:
    # CSVファイルの読み込みと 'datetime'をインデックスに設定
    original_df = pd.read_csv(INPUT_CSV_PATH, parse_dates=['datetime'])
    original_df.set_index('datetime', inplace=True)
except FileNotFoundError:
    print(f"エラー: {INPUT_CSV_PATH} が見つかりません。generate_mock_data.pyを実行してください。")
    exit()

# --------------------------------------------------------------------------------
# 2. 月次集計，単位変換，気象データ処理
#    元のコードの処理を再現し，全ての世帯データを月次・MJ単位に変換する
# --------------------------------------------------------------------------------

def process_to_monthly_mj(df):
    
    # 処理前の列名マッピング
    KWH_COLUMNS = ["elect_purchase(kWh)", "elect_sale(kWh)", "PV_gene(kWh)", "FC_gene(kWh)", "elect_cons(kWh)"]
    M3_COLUMNS = ["gas_cons(m3)"]
    RENAME_MAPPING = {
        "elect_purchase(kWh)": "elect_purchase(MJ)", "elect_sale(kWh)": "elect_sale(MJ)",
        "PV_gene(kWh)": "PV_gene(MJ)", "FC_gene(kWh)": "FC_gene(MJ)",
        "elect_cons(kWh)": "elect_cons(MJ)", "gas_cons(m3)": "gas_cons(MJ)"
    }
    
    # HEMSデータと気象データを分離（IDに依存する/しない）
    hems_cols = [col for col in KWH_COLUMNS + M3_COLUMNS if col in df.columns]
    weather_cols_30min = ['temp', 'WNDSPD', 'RHUM', 'PRCRIN_30MIN', 'GLBRAD_30MIN']
    weather_cols = [col for col in weather_cols_30min if col in df.columns]

    # --- (A) HEMSデータの月次集計と単位変換 ---
    # IDごとにグループ化し，月次合計にリサンプリング
    hems_monthly = df.groupby('ID')[hems_cols].resample('MS').sum().reset_index(level='ID')
    
    # 単位変換の実行
    for col in [c for c in KWH_COLUMNS if c in hems_monthly.columns]:
        hems_monthly[col] = hems_monthly[col] * KWH_TO_MJ
    for col in [c for c in M3_COLUMNS if c in hems_monthly.columns]:
        hems_monthly[col] = hems_monthly[col] * M3_TO_MJ
        
    hems_monthly.rename(columns=RENAME_MAPPING, inplace=True)

    # --- (B) 気象データの月次集計（HEMSデータとは独立） ---
    # 気象データは全ての世帯で同じ値を持つため，1つのタイムスタンプ（HEMSデータのIndex）でグループ化し，最初の値を使用
    weather_data_raw = df.groupby(df.index).first()[weather_cols]

    # 月次集計方法の定義
    weather_agg_methods = {
        'temp': 'mean', 'WNDSPD': 'mean', 'RHUM': 'mean',
        'PRCRIN_30MIN': 'sum', 'GLBRAD_30MIN': 'sum'
    }
    
    weather_monthly_df = weather_data_raw.resample('MS').agg(weather_agg_methods)
    
    # 列名変更 (元のコードを再現)
    WEATHER_RENAME_MAPPING = {
        'temp': 'Monthly_Avg_Temp', 'WNDSPD': 'Monthly_Avg_WindSpeed', 
        'RHUM': 'Monthly_Avg_RelHumidity', 'PRCRIN_30MIN': 'Monthly_Total_Precipitation', 
        'GLBRAD_30MIN': 'Monthly_Total_GlobalRad'
    }
    weather_monthly_df.rename(columns=WEATHER_RENAME_MAPPING, inplace=True)
    
    # --- (C) マージと最終整形 ---
    # 月次HEMSデータに月次気象データをマージ
    processed_df = hems_monthly.merge(weather_monthly_df, left_index=True, right_index=True, how='left')
    processed_df.dropna(inplace=True)
    
    # timestamp列の整形
    processed_df.reset_index(names=['timestamp'], inplace=True)
    processed_df['timestamp'] = processed_df['timestamp'].dt.strftime('%Y-%m')
    
    # ID, timestampを先頭に並び替え
    cols = [col for col in processed_df.columns if col not in ['ID', 'timestamp']]
    processed_df = processed_df[['ID', 'timestamp'] + cols]
    
    return processed_df

combined_df_monthly = process_to_monthly_mj(original_df)
print(f"✅ データ前処理が完了しました。データサイズ: {combined_df_monthly.shape}")

# --------------------------------------------------------------------------------
## 3. 代表世帯の選定 (中央値消費量の世帯)
# --------------------------------------------------------------------------------

print("\n--- 3. 代表世帯の選定 ---")

df_consumption = combined_df_monthly.groupby('ID')[['elect_cons(MJ)', 'gas_cons(MJ)']].sum().reset_index()
df_consumption['Total_Consumption_MJ'] = df_consumption['elect_cons(MJ)'] + df_consumption['gas_cons(MJ)']
median_consumption = df_consumption['Total_Consumption_MJ'].median()

# 中央値との差の絶対値が最も小さい世帯を特定
df_consumption['Diff_from_Median'] = np.abs(df_consumption['Total_Consumption_MJ'] - median_consumption)
representative_household_id = df_consumption.sort_values('Diff_from_Median').iloc[0]['ID']

print(f"選定された代表世帯ID (中央値): {representative_household_id}")

# 代表世帯のデータ抽出と整形
representative_data = combined_df_monthly[combined_df_monthly['ID'] == representative_household_id].copy()
representative_data['timestamp'] = pd.to_datetime(representative_data['timestamp'])
representative_data.set_index('timestamp', inplace=True)

# 分析対象変数に絞り込む
plot_vars = ['PV_gene(MJ)', 'elect_cons(MJ)', 'gas_cons(MJ)']
representative_data = representative_data[plot_vars]

# --------------------------------------------------------------------------------
## 4. 時系列トレンドのプロット (論文スタイル)
# --------------------------------------------------------------------------------

def plot_separate_timeseries_trends_for_thesis(df, household_id):
    """3つの時系列トレンドを縦に並べたサブプロットとして描画する（論文用）．"""
    
    PLOT_VARS_MAP = {
        'PV_gene(MJ)': {'title': '(a) Photovoltaic Generation (PV)', 'ylabel': 'PV Generation [MJ]'},
        'elect_cons(MJ)': {'title': '(b) Electricity Consumption', 'ylabel': 'Electricity Consumption [MJ]'},
        'gas_cons(MJ)': {'title': '(c) Gas Consumption', 'ylabel': 'Gas Consumption [MJ]'},
    }
    plot_vars = list(PLOT_VARS_MAP.keys())
    
    fig, axes = plt.subplots(nrows=len(plot_vars), ncols=1, figsize=(10, 12), sharex=True)
    
    for i, var in enumerate(plot_vars):
        ax = axes[i]
        labels = PLOT_VARS_MAP[var]
        
        df[var].plot(
            ax=ax, marker='o', linestyle='-', linewidth=2, color=plt.cm.tab10(i)
        )
        
        ax.set_title(labels['title'], fontsize=14, loc='left')
        ax.set_ylabel(labels['ylabel'], fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        
        if i < len(plot_vars) - 1:
            ax.set_xlabel('')
        else:
            ax.set_xlabel('Date (Month)', fontsize=12)

    fig.suptitle(f"Time Series Trend for Representative Household (ID: {household_id})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

print("\n--- 4. 時系列トレンドのプロット ---")
# [Image of Time Series Trend for Energy and Gas Consumption]
plot_separate_timeseries_trends_for_thesis(representative_data, representative_household_id)


# --------------------------------------------------------------------------------
## 5. ACF/PACF のプロット
# --------------------------------------------------------------------------------

def plot_acf_pacf_for_variable_for_thesis(df, variable_name, lags=18):
    """特定の変数のACFとPACFを論文表記（英語）で縦に並べてプロットする．"""
    
    PLOT_VARS_MAP = {
        'PV_gene(MJ)': {'title_suffix': 'PV Generation'},
        'elect_cons(MJ)': {'title_suffix': 'Electricity Consumption'},
        'gas_cons(MJ)': {'title_suffix': 'Gas Consumption'},
    }
    labels = PLOT_VARS_MAP[variable_name]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. ACF (自己相関関数)
    plot_acf(
        df[variable_name].dropna(), lags=lags, ax=axes[0], 
        title=f'{labels["title_suffix"]} - ACF'
    )
    axes[0].set_xlabel('Lag (Months)', fontsize=12)
    axes[0].set_ylabel('Autocorrelation', fontsize=12) 
    axes[0].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    
    # 2. PACF (偏自己相関関数)
    plot_pacf(
        df[variable_name].dropna(), lags=lags, ax=axes[1], 
        title=f'{labels["title_suffix"]} - PACF',
        method='ywm'
    )
    axes[1].set_xlabel('Lag (Months)', fontsize=12)
    axes[1].set_ylabel('Partial Autocorrelation', fontsize=12)
    axes[1].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    
    fig.suptitle(f"ACF and PACF for Representative Household ({variable_name})", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show()

print("\n--- 5. ACF/PACF のプロット ---")
for var in plot_vars:
    plot_acf_pacf_for_variable_for_thesis(representative_data, var, lags=18)

# --------------------------------------------------------------------------------
## 6. 変数間相関の検証 (モンテカルロシミュレーションの独立性検証用)
# --------------------------------------------------------------------------------

print("\n--- 6. 変数間相関の検証 (全世帯データ) ---")
variables_to_correlate = ["PV_gene(MJ)", "elect_cons(MJ)", "gas_cons(MJ)"]
variable_labels = {"PV_gene(MJ)": "PV Gen.", "elect_cons(MJ)": "Elec. Cons.", "gas_cons(MJ)": "Gas Cons"}

# 全世帯の月次データ（全期間）を集約して相関を計算
df_corr_renamed = combined_df_monthly[variables_to_correlate].rename(columns=variable_labels)
correlation_matrix = df_corr_renamed.corr(method='pearson')

print("\nピアソン相関行列:")
display(correlation_matrix)

# 相関ヒートマップの可視化
# 
plt.figure(figsize=(8, 6))
sns.heatmap(
    correlation_matrix, 
    annot=True,
    cmap='coolwarm',
    vmin=-1,
    vmax=1,
    fmt=".2f"
)
plt.title('Correlation Matrix of Energy Variables (All Households, Monthly)')
plt.show()

print("\n✅ 全てのEDAステップがエラーなく完了しました。")
