import pandas as pd
import numpy as np
from datetime import timedelta

def interpolate_weather_data(df):
    """
    Conditionally interpolates missing weather data and flags the interpolated rows.

    Interpolation is performed only under the following condition:
    1. The number of consecutive missing records is less than or equal to 3 
       (i.e., a data gap of 1.5 hours or less).
    2. Otherwise, the missing values (NaN) are retained.

    Args:
        df (pd.DataFrame): The raw weather data DataFrame.

    Returns:
        pd.DataFrame: The weather data DataFrame after conditional interpolation.
    """

    # Ensure timestamp column is in datetime format
    df['period_start_utc'] = pd.to_datetime(df['period_start_utc'])

    # Process data grouped by farm_id
    interpolated_dfs = []
    farm_ids = df['farm_id'].unique()

    print("Processing data integrity for each farm...\n")

    for farm_id in farm_ids:
        farm_df = df[df['farm_id'] == farm_id].copy()

        # Sort by timestamp to ensure correct chronological order
        farm_df.sort_values(by='period_start_utc', inplace=True)
        farm_df.reset_index(drop=True, inplace=True)

        # Create a complete time series to identify gaps
        start_date = farm_df['period_start_utc'].min().floor('D')
        end_date = farm_df['period_start_utc'].max().ceil('D')
        full_date_range = pd.date_range(start=start_date, end=end_date, freq='30min')

        full_df = pd.DataFrame(full_date_range, columns=['period_start_utc'])
        full_df['farm_id'] = farm_id

        merged_df = pd.merge(full_df, farm_df, on=['farm_id', 'period_start_utc'], how='left')

        # Before interpolation, create a boolean column to mark the original missing values
        merged_df['is_interpolated'] = merged_df['air_temperature_celsius'].isnull()
        
        # Identify consecutive blocks of missing values
        null_series = merged_df['air_temperature_celsius'].isnull()
        null_groups = null_series.ne(null_series.shift()).cumsum()
        
        # Identify groups of nulls where the consecutive count is greater than 3
        large_null_groups = [
            group_id for group_id, group in merged_df.groupby(null_groups)
            if group['air_temperature_celsius'].isnull().all() and len(group) > 3
        ]

        # Interpolate all relevant numeric columns
        numeric_cols_to_interpolate = ['air_temperature_celsius', 'global_horizontal_irradiance', 'precipitation_rate', 'period_length_mins']
        merged_df[numeric_cols_to_interpolate] = merged_df[numeric_cols_to_interpolate].interpolate(method='linear')

        # Iterate through the large null groups and revert the interpolation
        for group_id in large_null_groups:
            group_indices = merged_df.index[null_groups == group_id]
            start_time = merged_df.loc[group_indices[0], 'period_start_utc']
            end_time = merged_df.loc[group_indices[-1], 'period_start_utc']
            
            # Revert the interpolated values in these large gaps back to NaN
            merged_df.loc[group_indices, numeric_cols_to_interpolate] = np.nan
            # Also set is_interpolated to False, as these were not successfully interpolated
            merged_df.loc[group_indices, 'is_interpolated'] = False
            
            print(f"WARNING: Farm {farm_id} has a gap of {len(group_indices)} consecutive missing records "
                  f"between {start_time} and {end_time}. Interpolation was skipped for this gap.")

        # Fill any remaining NaNs in 'period_length_mins' (for the non-interpolated gaps)
        merged_df['period_length_mins'].fillna(30, inplace=True)

        interpolated_dfs.append(merged_df)

    # Concatenate the DataFrames for all farms
    final_df = pd.concat(interpolated_dfs, ignore_index=True)

    return final_df

# --- Execute Interpolation ---
# The source file is historic_weather_data.csv
input_file = '../data/historic_weather_data.csv'
output_file = '../data/interpolated_weather_data.csv'

try:
    historic_weather_df = pd.read_csv(input_file)
    print(f"Successfully loaded file: {input_file}")

    # Call the function to perform interpolation
    interpolated_df = interpolate_weather_data(historic_weather_df)

    # Inspect the results
    print("\nInfo for the interpolated DataFrame:")
    interpolated_df.info()
    print("\nFirst 5 rows of the interpolated DataFrame:")
    print(interpolated_df.head())

    # Save the results to a new CSV file
    interpolated_df.to_csv(output_file, index=False)
    print(f"\nData has been successfully saved to {output_file}.")

except FileNotFoundError:
    print(f"Error: File '{input_file}' not found. Please check the file path.")