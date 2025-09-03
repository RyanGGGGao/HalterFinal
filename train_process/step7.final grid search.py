#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 23:33:51 2025

@author: gaoyuan10
"""

import pandas as pd
from datetime import timedelta
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tqdm import tqdm # Used to display a progress bar, install with: pip install tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# --- 1. Load All Data Sources ---
print("--- 1. Loading all data files ---")
try:
    original_features_df = pd.read_csv('../data/growth_with_weather_data.csv')
    growth_rate_df = pd.read_csv('../data/growth_rate_data.csv')
    weather_df = pd.read_csv('../data/interpolated_weather_data.csv')
except FileNotFoundError as e:
    print(f"Error: File '{e.filename}' not found. Please ensure all three CSV files are in the same directory.")
    sys.exit(1)

# --- 2. Data Preprocessing ---
print("--- 2. Preprocessing data ---")
original_features_df['growth_rate_timestamp'] = pd.to_datetime(original_features_df['growth_rate_timestamp'])
growth_rate_df['utc_timestamp'] = pd.to_datetime(growth_rate_df['utc_timestamp'])
weather_df['period_start_utc'] = pd.to_datetime(weather_df['period_start_utc'])
weather_df.sort_values(by=['farm_id', 'period_start_utc'], inplace=True)
weather_df.set_index('period_start_utc', inplace=True)

# --- 3. Feature Engineering for Lagged Features (3, 7, 14 days) ---
print("--- 3. Calculating new long-term lagged features (this may take a few minutes) ---")
BASE_TEMP_CELSIUS = 5
SUNSHINE_THRESHOLD_GHI = 25
results = []
for index, row in tqdm(growth_rate_df.iterrows(), total=growth_rate_df.shape[0], desc="Processing growth data"):
    farm_id = row['farm_id']
    growth_timestamp = row['utc_timestamp']
    farm_weather_data = weather_df[weather_df['farm_id'] == farm_id]
    features = {'farm_id': farm_id, 'growth_rate_timestamp': growth_timestamp}
    for days in [3, 7, 14]:
        start_time = growth_timestamp - timedelta(days=days)
        window_weather_data = farm_weather_data.loc[start_time:growth_timestamp]
        if window_weather_data.empty: continue
        temp = window_weather_data['air_temperature_celsius']
        ghi = window_weather_data['global_horizontal_irradiance']
        precip = window_weather_data['precipitation_rate']
        features[f'avg_temp_{days}d'] = temp.mean()
        features[f'min_temp_{days}d'] = temp.min()
        features[f'max_temp_{days}d'] = temp.max()
        features[f'var_temp_{days}d'] = temp.var()
        features[f'avg_ghi_{days}d'] = ghi.mean()
        features[f'max_ghi_{days}d'] = ghi.max()
        features[f'total_sunshine_{days}d'] = (ghi > SUNSHINE_THRESHOLD_GHI).sum() * 0.5
        features[f'total_precip_{days}d'] = precip.sum() * 0.5
        features[f'max_precip_{days}d'] = precip.max()
        daily_gdd = window_weather_data.resample('D')['air_temperature_celsius'].agg(['max', 'min'])
        daily_gdd['gdd'] = ((daily_gdd['max'] + daily_gdd['min']) / 2 - BASE_TEMP_CELSIUS).clip(lower=0)
        features[f'gdd_{days}d'] = daily_gdd['gdd'].sum()
    if 'avg_temp_7d' in features and 'total_sunshine_7d' in features:
        features['inter_temp_sunshine_7d'] = features['avg_temp_7d'] * features['total_sunshine_7d']
    if 'avg_temp_7d' in features and 'total_precip_7d' in features:
        features['inter_temp_precip_7d'] = features['avg_temp_7d'] * features['total_precip_7d']
    results.append(features)
new_features_df = pd.DataFrame(results)
print("\nCalculation of new long-term features is complete.")

# --- 4. Merge Features ---
print("--- 4. Merging 24-hour features with new long-term features ---")
final_df = pd.merge(original_features_df, new_features_df, on=['farm_id', 'growth_rate_timestamp'], how='inner')
final_df.dropna(inplace=True)
print(f"Feature merging complete! Final dataset shape: {final_df.shape}")

# --- 5. Hyperparameter Tuning on the Complete Feature Set ---
print("\n--- 5. Starting GridSearchCV on the complete feature set ---")
target = 'growth_rate'
feature_cols = [col for col in final_df.columns if col not in ['farm_id', 'growth_rate_timestamp', 'growth_rate']]
X = final_df[feature_cols]
y = final_df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define a new parameter grid for the search
param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [12],
    'min_samples_leaf': [6, 8, 10],
    'min_samples_split': [10, 15]
}

# Instantiate the RandomForest Regressor
rf_regressor = RandomForestRegressor(random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid, cv=10,
                           scoring='neg_mean_squared_error',
                           n_jobs=-1, verbose=2)

grid_search.fit(X_train_scaled, y_train)

print("\n--- GridSearchCV Complete ---")
print("Best parameters found on the new feature set: ", grid_search.best_params_)

# Get the best model from the search
best_model = grid_search.best_estimator_

# --- 6. Evaluate Final Model Performance ---
print("\n--- 6. Evaluating final model performance ---")
y_pred = best_model.predict(X_test_scaled) # Use the best_model for predictions
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Test Set RMSE: {rmse:.4f}")
print(f"Test Set MAE: {mae:.4f}")
print(f"Test Set R-squared: {r2:.4f}")

print("\n--- Performance Comparison ---")
previous_best_r2 = 0.4281
print(f"R-squared of the previous best model (24h features only): {previous_best_r2}")
print(f"R-squared of the new model (full feature set, retuned): {r2:.4f}")
if r2 > previous_best_r2:
    improvement = (r2 - previous_best_r2) / previous_best_r2
    print(f"Relative performance improvement: {improvement:.2%}")
else:
    print("Retuning did not yield a significant performance improvement.")

# --- 7. Visualization and Analysis ---
print("\n--- 7. Generating visualizations ---")

# a. Actual vs. Predicted Values Scatter Plot
plt.figure(figsize=(10, 8))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', linewidth=2)
plt.title('Actual vs. Predicted Values (Full Feature Set)', fontsize=16)
plt.xlabel('Actual Growth Rate', fontsize=12)
plt.ylabel('Predicted Growth Rate', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig('actual_vs_predicted_final.png')
print("Saved 'Actual vs. Predicted' scatter plot to actual_vs_predicted_final.png")

# b. Feature Importance Analysis
feature_importances = pd.DataFrame({
    'feature': feature_cols,
    'importance': best_model.feature_importances_ # Use feature importances from the best_model
}).sort_values('importance', ascending=False)

print("\nTop 15 Feature Importances:")
print(feature_importances.head(15))

plt.figure(figsize=(12, 10))
sns.barplot(x='importance', y='feature', data=feature_importances.head(20))
plt.title('Feature Importance (Top 20 - Full Feature Set)', fontsize=16)
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.savefig('feature_importance_final.png')
print("Saved 'Feature Importance' bar chart to feature_importance_final.png")