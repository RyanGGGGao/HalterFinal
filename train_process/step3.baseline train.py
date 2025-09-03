import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
# Split the data into 80% for training and 20% for testing to evaluate the model's generalization ability
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")


# --- 3. Feature Scaling ---
# Initialize the StandardScaler
scaler = StandardScaler()
# Fit on the training data and transform it
X_train_scaled = scaler.fit_transform(X_train)
# Use the scaler fitted on the training data to transform the test set, avoiding data leakage
X_test_scaled = scaler.transform(X_test)


# --- 4. Model Optimization: GridSearchCV ---
print("\n--- Starting GridSearchCV to find the best hyperparameters ---")

# Define the parameter grid 
param_grid = {
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'ccp_alpha': [0.0, 0.001, 0.01, 0.1] 
}


# Define the parameter grid for the second round of search
# =============================================================================
# param_grid = {
#     'max_depth': [3, 4, 5, 6, 7],           # Finer search around the previous best value of 5
#     'min_samples_split': [15, 20, 25, 30, 40], # Explore larger values than the previous best of 20
#     'min_samples_leaf': [6, 8, 10, 12, 15],  # Explore larger values than the previous best of 8
#     'ccp_alpha': [0.0, 0.001, 0.005, 0.01]   # Add more values between 0 and 0.01
# }
# =============================================================================

# Initialize the Decision Tree Regressor
dt_regressor = DecisionTreeRegressor(random_state=42)

# Set up and run GridSearchCV on the training set
grid_search = GridSearchCV(estimator=dt_regressor, param_grid=param_grid, cv=10,
                           scoring='neg_mean_squared_error', # Use a single primary metric for refitting
                           n_jobs=-1, verbose=2)

grid_search.fit(X_train_scaled, y_train)

# Print the best parameters found
print("\n--- GridSearchCV Complete ---")
print("Best parameters found: ", grid_search.best_params_)

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


# --- 6. Visualization and Analysis ---
print("\n--- Generating visualizations ---")

# a. Actual vs. Predicted Values Scatter Plot
plt.figure(figsize=(10, 8))
sns.scatterplot(x=y_test, y=y_pred_test, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', linewidth=2)
plt.title('Actual vs. Predicted Values (Test Set)', fontsize=16)
plt.xlabel('Actual Growth Rate', fontsize=12)
plt.ylabel('Predicted Growth Rate', fontsize=12)
plt.grid(True)
plt.show()

# b. Feature Importance Analysis
feature_importances = pd.DataFrame({
    'feature': features,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importances:")
print(feature_importances)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importances)
plt.title('Feature Importance', fontsize=16)
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.show()