import pandas as pd
from datetime import timedelta
import numpy as np

# Define constants for agronomic calculations
BASE_TEMP_CELSIUS = 5
SUNSHINE_THRESHOLD_GHI = 25

def create_all_features(farm_id: str, timestamp: pd.Timestamp, weather_history_df: pd.DataFrame) -> dict:
    """
    Calculates all features (24-hour, 3-day, 7-day, 14-day, and interaction features)
    in real-time for a single prediction, using the provided weather history.

    Args:
        farm_id: The ID of the farm (used for validation).
        timestamp: The specific point in time for the prediction.
        weather_history_df: A DataFrame containing the recent weather history for the farm.

    Returns:
        A dictionary containing all the calculated features.
    """
    # Ensure the provided weather data corresponds to the correct farm_id.
    # In a production application, more robust validation could be added.
    if not weather_history_df.empty and weather_history_df['farm_id'].iloc[0] != farm_id:
        raise ValueError("The provided weather data does not match the farm_id.")

    # Initialize a dictionary to store the features
    features = {}
    
    # --- Date / Seasonal Feature ---
    day_of_year = timestamp.timetuple().tm_yday
    features['date_cos'] = np.cos(2 * np.pi * (day_of_year - 14) / 365)

    # --- Core Logic: Calculate features for different time windows ---
    for days in [1, 3, 7, 14]:
        start_time = timestamp - timedelta(days=days)
        # Filter directly on the provided DataFrame
        window_weather_data = weather_history_df.loc[start_time:timestamp].copy()
        
        if window_weather_data.empty:
            continue

        # Use the '_24h' suffix for the 1-day window to maintain consistency with original feature names
        suffix = f'{int(days*24)}h' if days == 1 else f'{days}d'
        
        temp = window_weather_data['air_temperature_celsius']
        ghi = window_weather_data['global_horizontal_irradiance']
        precip = window_weather_data['precipitation_rate']
        
        features[f'avg_temp_{suffix}'] = temp.mean()
        features[f'total_sunshine_{suffix}'] = (ghi > SUNSHINE_THRESHOLD_GHI).sum() * 0.5
        features[f'total_precip_{suffix}'] = precip.sum() * 0.5
        
        daily_gdd = window_weather_data.resample('D')['air_temperature_celsius'].agg(['max', 'min'])
        daily_gdd['gdd'] = ((daily_gdd['max'] + daily_gdd['min']) / 2 - BASE_TEMP_CELSIUS).clip(lower=0)
        features[f'gdd_{suffix}'] = daily_gdd['gdd'].sum()

        features[f'min_temp_{suffix}'] = temp.min()
        features[f'max_temp_{suffix}'] = temp.max()
        features[f'var_temp_{suffix}'] = temp.var()
        features[f'avg_ghi_{suffix}'] = ghi.mean()
        features[f'max_ghi_{suffix}'] = ghi.max()
        features[f'max_precip_{suffix}'] = precip.max()
        
        # Calculate additional detailed features only for the 24-hour window
        if days == 1:
            features['weather_data_count'] = len(window_weather_data)
            features['median_air_temperature_celsius_24h'] = temp.median()
            features['var_air_temperature_celsius_24h'] = temp.var()
            features['var_precipitation_rate_24h'] = precip.var()
            features['total_rain_duration_24h'] = (precip > 0).sum() * 0.5
            
            # Night-time irradiance features
            window_weather_data['hour'] = window_weather_data.index.hour
            night_time_data = window_weather_data[(window_weather_data['hour'] >= 18) | (window_weather_data['hour'] <= 10)]
            if not night_time_data.empty:
                features['avg_ghi_18h_10h_utc'] = night_time_data['global_horizontal_irradiance'].mean()
                features['median_ghi_18h_10h_utc'] = night_time_data['global_horizontal_irradiance'].median()
            else:
                features['avg_ghi_18h_10h_utc'] = 0
                features['median_ghi_18h_10h_utc'] = 0
    
    # --- Create Interaction Features ---
    if 'avg_temp_7d' in features and 'total_sunshine_7d' in features:
        features['inter_temp_sunshine_7d'] = features['avg_temp_7d'] * features['total_sunshine_7d']
    if 'avg_temp_7d' in features and 'total_precip_7d' in features:
        features['inter_temp_precip_7d'] = features['avg_temp_7d'] * features['total_precip_7d']

    return features