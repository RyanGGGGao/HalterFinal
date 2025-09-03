import pandas as pd
from datetime import timedelta
import numpy as np
import sys
# --- 1. Load Data ---
try:
    growth_rate_df = pd.read_csv('../data/growth_rate_data.csv')
    weather_df = pd.read_csv('../data/interpolated_weather_data.csv')
except FileNotFoundError as e:
    print(f"Error: File '{e.filename}' not found. Please ensure both CSV files are in the same directory.")
    exit()

# --- 2. Data Preprocessing ---
# Convert timestamp columns to datetime objects for calculations
growth_rate_df['utc_timestamp'] = pd.to_datetime(growth_rate_df['utc_timestamp'])
weather_df['period_start_utc'] = pd.to_datetime(weather_df['period_start_utc'])

# --- 3. Feature Engineering ---
# List to store the results for each growth rate record
results = []
# Define constants for agronomic calculations
base_temperature_celsius = 5  # Base temperature for Growing Degree Days (GDD)
sunshine_threshold = 25       # GHI threshold to be considered as 'sunshine'

# Process each grass growth record
for index, row in growth_rate_df.iterrows():
    farm_id = row['farm_id']
    growth_timestamp = row['utc_timestamp']
    
    # Define the 24-hour window for feature calculation
    start_time = growth_timestamp - timedelta(hours=24)
    
    # Filter weather data for the specific farm and the 24-hour window
    relevant_weather_data = weather_df[
        (weather_df['farm_id'] == farm_id) &
        (weather_df['period_start_utc'] >= start_time) &
        (weather_df['period_start_utc'] < growth_timestamp)
    ].copy() # Use .copy() to avoid SettingWithCopyWarning

    # Record the number of weather entries found in the window
    weather_record_count = len(relevant_weather_data)
    
    # Initialize all metrics to None
    avg_temp, min_temp, max_temp, median_temp, var_temp = None, None, None, None, None
    max_ghi, avg_ghi_night, median_ghi_night = None, None, None
    max_precip, var_precip = None, None
    total_rain_duration, total_sunshine_duration = None, None
    growing_degree_days = None

    # Proceed with calculations only if weather data is available
    if weather_record_count > 0:
        # --- Temperature-based features ---
        temp_series = relevant_weather_data['air_temperature_celsius']
        avg_temp = temp_series.mean()
        min_temp = temp_series.min()
        max_temp = temp_series.max()
        median_temp = temp_series.median()
        var_temp = temp_series.var()
        
        # Calculate Growing Degree Days (GDD)
        # Formula: Sum of ((Daily T_avg) - T_base) for temperatures above T_base
        relevant_weather_data['temp_for_gdd'] = temp_series.apply(lambda x: max(0, x - base_temperature_celsius))
        # For 30-min data, sum of degrees above base divided by 48 (number of 30-min intervals in a day)
        growing_degree_days = relevant_weather_data['temp_for_gdd'].sum() / 48
    
        # --- Precipitation and Irradiance features ---
        precip_series = relevant_weather_data['precipitation_rate']
        ghi_series = relevant_weather_data['global_horizontal_irradiance']
        
        max_precip = precip_series.max()
        var_precip = precip_series.var()
        max_ghi = ghi_series.max()
        
        # Filter data for the 18:00 PM to 10:00 AM (UTC) time period
        relevant_weather_data['hour'] = relevant_weather_data['period_start_utc'].dt.hour
        night_time_data = relevant_weather_data[
            (relevant_weather_data['hour'] >= 18) | (relevant_weather_data['hour'] <= 10)
        ]
        
        if not night_time_data.empty:
            avg_ghi_night = night_time_data['global_horizontal_irradiance'].mean()
            median_ghi_night = night_time_data['global_horizontal_irradiance'].median()

        # Calculate total duration of rain in hours (each record is 0.5 hours)
        total_rain_duration = (precip_series > 0).sum() * 0.5

        # Calculate total duration of effective sunshine in hours
        total_sunshine_duration = (ghi_series > sunshine_threshold).sum() * 0.5
    
    # --- Date / Seasonal feature ---
    # Convert timestamp to day of the year
    day_of_year = growth_timestamp.timetuple().tm_yday
    # Transform date into a cosine value to capture seasonality, adjusted for Southern Hemisphere seasons
    date_cos = np.cos(2 * np.pi * (day_of_year - 14) / 365)

    # Append all calculated features to the results list
    results.append({
        # --- Basic Information ---
        'farm_id': farm_id,
        'growth_rate_timestamp': growth_timestamp,
        'growth_rate': row['daily_growth_rate'],
        'weather_data_count': weather_record_count, # Number of weather records in the 24h window, indicating data completeness
        
        # --- Date / Seasonal Feature ---
        'date_cos': date_cos, # Cosine-transformed day of the year to capture seasonal cycles
        
        # --- Temperature Features (previous 24 hours) ---
        'avg_air_temperature_celsius_24h': avg_temp,
        'min_air_temperature_celsius_24h': min_temp,
        'max_air_temperature_celsius_24h': max_temp,
        'median_air_temperature_celsius_24h': median_temp, # Median temperature, robust to outliers
        'var_air_temperature_celsius_24h': var_temp,   # Variance of temperature, measuring its volatility
        
        # --- Irradiance Features (previous 24 hours) ---
        'max_global_horizontal_irradiance_24h': max_ghi, # Peak solar irradiance
        'avg_ghi_18h_10h_utc': avg_ghi_night,          # Average GHI between 18:00 and 10:00 UTC
        'median_ghi_18h_10h_utc': median_ghi_night,      # Median GHI between 18:00 and 10:00 UTC
        'total_sunshine_duration_24h': total_sunshine_duration, # Total hours of effective sunshine
        
        # --- Precipitation Features (previous 24 hours) ---
        'max_precipitation_rate_24h': max_precip, # Maximum precipitation rate, indicating rainfall intensity
        'var_precipitation_rate_24h': var_precip, # Variance of precipitation, measuring its volatility
        'total_rain_duration_24h': total_rain_duration, # Total hours of rainfall
        
        # --- Key Agronomic Indicator (previous 24 hours) ---
        'growing_degree_days_24h': growing_degree_days # Accumulated heat required for plant growth
    })

# --- 4. Save Results ---
# Convert the list of dictionaries to a DataFrame
result_df = pd.DataFrame(results)

# Save the resulting DataFrame to a new CSV file
result_df.to_csv('../data/growth_with_weather_data.csv', index=False)
print("Data processing is complete. Results have been saved to 'growth_with_weather_data.csv'.")
print("\nFirst 5 rows of the generated DataFrame:")
print(result_df.head())