import pandas as pd
import numpy as np
import os

# --------------------------------------------------------------------------------
# --- 設定定数 ---
# --------------------------------------------------------------------------------

CSV_FILE_NAME = 'sample_cleaned_data.csv'

# 単位変換係数 (元のコードから引用 - kWh, m3 -> MJ)
KWH_TO_MJ = 3.6
M3_TO_MJ = 45.0 

# 予測期間: 2023年4月 ('2023-04') から 2024年3月 ('2024-03')
TEST_START_DATE = '2023-04'
TEST_END_DATE = '2024-03'
PREDICTION_LENGTH = 12

# 予測に必要なラグ（12ヶ月前）のデータ期間
# 2023-04 の予測には 2022-04 のデータが必要
LAG_START_DATE = '2022-04' 
LAG_END_DATE = '2023-03'

TARGET_COLUMNS = [
    'PV_gene(MJ)', 
    'FC_gene(MJ)', 
    'elect_cons(MJ)', 
    'gas_cons(MJ)'
]

# --------------------------------------------------------------------------------
# --- 1. データ読み込みと前処理（月次・MJ単位への変換） ---
# --------------------------------------------------------------------------------

def load_and_preprocess_data(file_name):
    """CSVを読み込み、月次・MJ単位への変換と整形を行う。"""
    
    print(f"--- データ読み込み: {file_name} ---")
    try:
        df = pd.read_csv(file_name, parse_dates=['datetime'])
        df.set_index('datetime', inplace=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"エラー: {file_name} が見つかりません。事前に generate_mock_data.py を実行してください。")

    # --- HEMSデータの月次集計と単位変換 ---
    KWH_COLUMNS = ["elect_purchase(kWh)", "elect_sale(kWh)", "PV_gene(kWh)", "FC_gene(kWh)", "elect_cons(kWh)"]
    M3_COLUMNS = ["gas_cons(m3)"]
    RENAME_MAPPING = {
        "elect_purchase(kWh)": "elect_purchase(MJ)", "elect_sale(kWh)": "elect_sale(MJ)",
        "PV_gene(kWh)": "PV_gene(MJ)", "FC_gene(kWh)": "FC_gene(MJ)",
        "elect_cons(kWh)": "elect_cons(MJ)", "gas_cons(m3)": "gas_cons(MJ)"
    }
    
    # IDごとにグループ化し，月次合計にリサンプリング
    hems_monthly = df.groupby('ID')[KWH_COLUMNS + M3_COLUMNS].resample('MS').sum().reset_index(level='ID')
    
    # 単位変換
    for col in [c for c in KWH_COLUMNS if c in hems_monthly.columns]:
        hems_monthly[col] = hems_monthly[col] * KWH_TO_MJ
    for col in [c for c in M3_COLUMNS if c in hems_monthly.columns]:
        hems_monthly[col] = hems_monthly[col] * M3_TO_MJ
    hems_monthly.rename(columns=RENAME_MAPPING, inplace=True)

    # --- 気象データの月次集計（平均/合計） ---
    weather_cols_raw = ['temp', 'WNDSPD', 'RHUM', 'PRCRIN_30MIN', 'GLBRAD_30MIN']
    weather_data_raw = df.groupby(df.index).first()[weather_cols_raw]

    weather_agg_methods = {'temp': 'mean', 'WNDSPD': 'mean', 'RHUM': 'mean', 'PRCRIN_30MIN': 'sum', 'GLBRAD_30MIN': 'sum'}
    weather_monthly_df = weather_data_raw.resample('MS').agg(weather_agg_methods)
    WEATHER_RENAME_MAPPING = {
        'temp': 'Monthly_Avg_Temp', 'WNDSPD': 'Monthly_Avg_WindSpeed', 
        'RHUM': 'Monthly_Avg_RelHumidity', 'PRCRIN_30MIN': 'Monthly_Total_Precipitation', 
        'GLBRAD_30MIN': 'Monthly_Total_GlobalRad'
    }
    weather_monthly_df.rename(columns=WEATHER_RENAME_MAPPING, inplace=True)
    
    # --- マージと最終整形 ---
    processed_df = hems_monthly.merge(weather_monthly_df, left_index=True, right_index=True, how='left')
    processed_df.dropna(inplace=True)
    processed_df.reset_index(names=['timestamp'], inplace=True)
    
    # timestamp列を 'YYYY-MM' 文字列に変換
    processed_df['timestamp'] = processed_df['timestamp'].dt.strftime('%Y-%m')
    
    print(f"✅ データ前処理が完了しました。データサイズ: {processed_df.shape}")
    return processed_df

# --------------------------------------------------------------------------------
# --- 2. Seasonal Naive 予測の実行 ---
# --------------------------------------------------------------------------------

def run_seasonal_naive(df):
    
    # 予測期間の実績値のみが必要なため、TEST_END_DATEまでを対象とします。
    # ラグデータ（12ヶ月前の実績値）の抽出
    lag_df = df[
        (df['timestamp'] >= LAG_START_DATE) & (df['timestamp'] <= LAG_END_DATE)
    ].copy()
    
    # SN予測値として使用するデータはTARGET_COLUMNSの実績値のみ
    lag_df = lag_df[['ID', 'timestamp'] + TARGET_COLUMNS]

    # --- 予測データフレームの作成 ---
    
    # 1. ラグデータをコピーし、新しい予測データフレームのベースとする
    pred_df = lag_df.copy()
    
    # 2. timestamp列を12ヶ月（1年）進める (予測時点に合わせる)
    pred_df['timestamp'] = pd.to_datetime(pred_df['timestamp'], format='%Y-%m')
    pred_df['timestamp'] = (
        pred_df['timestamp'].dt.to_period('M') + PREDICTION_LENGTH
    ).dt.to_timestamp()
    
    # 3. 予測値であることを示す列名に変更
    rename_mapping = {col: f'{col}_sn_pred' for col in TARGET_COLUMNS}

    # ターゲット列のみを保持し、列名を変更
    pred_df.rename(columns=rename_mapping, inplace=True)
    pred_df = pred_df[['ID', 'timestamp'] + list(rename_mapping.values())]
    pred_df.reset_index(drop=True, inplace=True)
    
    # timestampを'YYYY-MM'文字列に戻す
    pred_df['timestamp'] = pred_df['timestamp'].dt.strftime('%Y-%m')
    
    # 4. 予測データフレームの確認
    print(f"\n--- Seasonal Naive (m=12) 予測の実行 ---")
    print(f"✅ SN予測データフレームが作成されました。サイズ: {pred_df.shape}")
    
    # 5. 結果の保存 (READMEに従い、ファイル名は `baseline_forecasts.csv` とします)
    output_pkl_path = "sn_predictions.pkl"
    pred_df.to_pickle(output_pkl_path)
    print(f"✅ 予測結果が整形され、'{output_pkl_path}' に保存されました。")

    # 6. 予測結果のサマリーを表示
    print("\n--- SN予測結果（最初の5行） ---")
    display(pred_df.head())
    
    return pred_df

# --------------------------------------------------------------------------------
# --- 実行とエラーチェック ---
# --------------------------------------------------------------------------------

try:
    processed_data = load_and_preprocess_data(CSV_FILE_NAME)
    if not processed_data.empty:
        run_seasonal_naive(processed_data)
        print("\n=== Seasonal Naive予測の実行とエラーチェックが完了しました。===")
    else:
        print("\n=== データ処理が中断されたため、SN予測は実行されませんでした。===")
except Exception as e:
    print(f"\n致命的なエラーが発生しました: {e}")
    print("\n=== Seasonal Naive予測の実行中にエラーが発生しました。===")
