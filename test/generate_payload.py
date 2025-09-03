import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_test_payload(farm_id: str, prediction_timestamp_str: str) -> str:
    """
    Generates a complete JSON payload for API testing, including a 14-day weather history.
    """
    
    prediction_timestamp = pd.to_datetime(prediction_timestamp_str)
    
    # 1. Create a time series of 672 timestamps (14 days * 48 records/day)
    num_records = 14 * 48
    timestamps = [prediction_timestamp - timedelta(minutes=30 * i) for i in range(num_records)]
    timestamps.reverse() # Reverse the list to have chronological order from past to present
    
    weather_history = []
    
    # 2. Generate synthetic weather data for each timestamp
    for ts in timestamps:
        # Simulate diurnal temperature variation using a sine wave
        # Assuming September is spring, with temperatures between 8°C (night) and 18°C (afternoon)
        temp_variation = np.sin(2 * np.pi * (ts.hour + ts.minute/60 - 8) / 24) * 5 
        air_temp = 13 + temp_variation + np.random.randn() * 0.5 # Add some random noise

        # Simulate diurnal irradiance variation (daylight only)
        ghi = 0.0
        if 6 <= ts.hour < 19:
            # Use a sine-like function to simulate sunrise and sunset
            ghi_variation = np.sin(np.pi * (ts.hour + ts.minute/60 - 6) / 13) * 800
            ghi = max(0, ghi_variation + np.random.randint(-50, 50))

        # Simulate sporadic precipitation
        precip = 0.0
        if np.random.rand() < 0.05: # Assume a 5% chance of rain at any given interval
            precip = np.random.uniform(0.1, 5.0)

        weather_history.append({
            "period_start_utc": ts.strftime('%Y-%m-%d %H:%M:%S'),
            "air_temperature_celsius": round(air_temp, 2),
            "global_horizontal_irradiance": round(ghi, 2),
            "precipitation_rate": round(precip, 2)
        })
        
    # 3. Assemble the final JSON payload object
    payload = {
        "farm_id": farm_id,
        "prediction_timestamp_utc": prediction_timestamp_str,
        "weather_history": weather_history
    }
    
    # Convert the Python dictionary to a formatted JSON string
    return json.dumps(payload, indent=2)

if __name__ == "__main__":
    # --- Configure the parameters you want to test here ---
    FARM_ID_TO_TEST = "3a2cce8f-e7b2-4f35-95a0-99c4554cd028"
    # Use a future date as the prediction point for this example
    PREDICTION_TIMESTAMP = "2025-09-03 11:00:00" 

    # Generate and print the JSON payload
    test_json = generate_test_payload(FARM_ID_TO_TEST, PREDICTION_TIMESTAMP)
    print(test_json)