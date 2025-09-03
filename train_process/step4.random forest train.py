import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import sys

# --- Pandas Display Options ---
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'left')

# --- 1. Load and Preprocess Data ---
try:
    df = pd.read_csv('../data/growth_with_weather_data.csv')
    print(f"Data loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print("Error: 'growth_with_weather_data.csv' not found. Please ensure the file is in the correct directory.")
    sys.exit(1)

# Filter the data
df_filtered = df[df['weather_data_count'] > 40].copy()
print(f"After filtering, {len(df_filtered)} records will be used for modeling.")

# Define features and target variable
features = [
    'date_cos', 'avg_air_temperature_celsius_24h', 'min_air_temperature_celsius_24h',
    'max_air_temperature_celsius_24h', 'median_air_temperature_celsius_24h',
    'var_air_temperature_celsius_24h', 'max_global_horizontal_irradiance_24h',
    'avg_ghi_18h_10h_utc', 'median_ghi_18h_10h_utc', 'total_sunshine_duration_24h',
    'max_precipitation_rate_24h', 'var_precipitation_rate_24h',
    'total_rain_duration_24h', 'growing_degree_days_24h'
]
target = 'growth_rate'

# Prepare data for modeling
X = df_filtered[features]
y = df_filtered[target]

# --- 2. Split Data into Training and Test Sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")


# --- 3. Feature Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --- 4. RandomForest Model Tuning with GridSearchCV ---
print("\n--- Starting GridSearchCV for RandomForest ---")

# Define the parameter grid for RandomForest

# Parameter grid for the first round
# =============================================================================
# param_grid = {
#     'n_estimators': [100, 150, 200],         # number of the tree
#     'max_depth': [5, 10, 15, None],          # max depth
#     'min_samples_split': [2, 5, 10],         # min number of samples when split
#     'min_samples_leaf': [1, 2, 4]            # min number of samples of leaf node
# }
# =============================================================================
# Parameter grid for the second round of search
param_grid = {
    'max_depth': [8, 10, 12],            # Explore around the previous best value of 10
    'n_estimators': [200, 250, 300],     # Explore a higher number of trees
    'min_samples_leaf': [4, 6, 8],       # Explore larger leaf sizes
    'min_samples_split': [10, 12, 15]    # Explore larger split thresholds
}

# Instantiate the RandomForest Regressor
rf_regressor = RandomForestRegressor(random_state=42)

# Set up GridSearchCV
# cv=10 means 10-fold cross-validation
# n_jobs=-1 means using all available CPU cores for parallel processing, which can significantly speed up the search
grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid, cv=10,
                           scoring='neg_mean_squared_error',
                           n_jobs=-1, verbose=2)

grid_search.fit(X_train_scaled, y_train)

# Print the best parameters found
print("\n--- GridSearchCV Complete ---")
print("Best parameters for RandomForest: ", grid_search.best_params_)

# Get the best model from the search
best_model = grid_search.best_estimator_


# --- 5. Evaluate the Final Model on the Independent Test Set ---
print("\n--- Evaluating final model performance on the test set ---")
y_pred_test = best_model.predict(X_test_scaled)

# Calculate evaluation metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print(f"Test Set RMSE (Root Mean Squared Error): {rmse:.4f}")
print(f"Test Set MAE (Mean Absolute Error): {mae:.4f}")
print(f"Test Set R-squared (Coefficient of Determination): {r2:.4f}")