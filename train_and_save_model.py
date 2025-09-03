import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import joblib

# 导入我们新的特征工程函数
from app.feature_pipeline import create_all_features

print("--- 开始训练并保存最终模型 (从原始数据生成所有特征) ---")

# --- 1. 加载原始数据 ---
print("--- 1. 正在加载原始数据文件 ---")
try:
    growth_rate_df = pd.read_csv('data/growth_rate_data.csv')
    weather_df = pd.read_csv('data/interpolated_weather_data.csv')
except FileNotFoundError as e:
    print(f"错误：文件 {e.filename} 未找到。请确保数据文件在 'data/' 目录下。")
    exit()

# --- 2. 数据预处理 ---
print("--- 2. 正在进行数据预处理 ---")
growth_rate_df['utc_timestamp'] = pd.to_datetime(growth_rate_df['utc_timestamp'])
weather_df['period_start_utc'] = pd.to_datetime(weather_df['period_start_utc'])
weather_df.sort_values(by=['farm_id', 'period_start_utc'], inplace=True)
weather_df.set_index('period_start_utc', inplace=True)

# --- 3. 完整特征工程 ---
print("--- 3. 正在为整个数据集生成所有特征（这会比较耗时） ---")
results = []
for index, row in tqdm(growth_rate_df.iterrows(), total=growth_rate_df.shape[0], desc="生成特征"):
    all_features = create_all_features(
        farm_id=row['farm_id'],
        timestamp=row['utc_timestamp'],
        weather_df=weather_df
    )
    all_features['growth_rate'] = row['daily_growth_rate'] # 添加目标变量
    results.append(all_features)

final_df = pd.DataFrame(results)
final_df.dropna(inplace=True)
print(f"特征工程完成！最终数据集维度: {final_df.shape}")

# --- 4. 训练最终模型 ---
print("--- 4. 正在使用所有数据训练最终模型 ---")
target = 'growth_rate'
feature_cols = [col for col in final_df.columns if col not in ['farm_id', 'growth_rate_timestamp', 'growth_rate']]
X = final_df[feature_cols]
y = final_df[target]

# 初始化并训练 scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用我们找到的最佳参数来实例化并训练最终模型
best_rf_params = {
    'max_depth': 12,
    'min_samples_leaf': 6,
    'min_samples_split': 10,
    'n_estimators': 300,
    'random_state': 42,
    'n_jobs': -1
}
final_model = RandomForestRegressor(**best_rf_params)
final_model.fit(X_scaled, y)
print("模型训练完成。")

# --- 5. 保存模型、Scaler 和特征列 ---
model_path = 'app/ml_models/rf_model.joblib'
scaler_path = 'app/ml_models/scaler.joblib'
columns_path = 'app/ml_models/feature_columns.joblib'

joblib.dump(final_model, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(feature_cols, columns_path)

print(f"模型已保存到: {model_path}")
print(f"Scaler已保存到: {scaler_path}")
print(f"特征列已保存到: {columns_path}")