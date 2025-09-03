from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import pandas as pd
import joblib

# Import the updated feature engineering function
from .feature_pipeline import create_all_features

# Initialize the FastAPI application
app = FastAPI(title="Grass Growth Rate Prediction API", version="3.0 - Stateless")

# --- Load Model Artifacts ---
MODEL_PATH = "app/ml_models/rf_model.joblib"
SCALER_PATH = "app/ml_models/scaler.joblib"
COLUMNS_PATH = "app/ml_models/feature_columns.joblib"

# Load the model and other artifacts at application startup
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(COLUMNS_PATH)
except FileNotFoundError as e:
    raise RuntimeError(f"Startup failed: Could not load necessary model or data files - {e}")

# --- Define API Input and Output Models ---
class WeatherRecord(BaseModel):
    """Defines the structure for a single weather record."""
    period_start_utc: str
    air_temperature_celsius: float
    global_horizontal_irradiance: float
    precipitation_rate: float

class PredictionRequest(BaseModel):
    """Defines the structure of the API request body."""
    farm_id: str
    prediction_timestamp_utc: str
    weather_history: List[WeatherRecord] = Field(..., description="A list of 30-minute weather data records, going back at least 14 days from the prediction timestamp.")

class PredictionResponse(BaseModel):
    """Defines the structure of the API response body."""
    farm_id: str
    prediction_timestamp_utc: str
    predicted_growth_rate: float

# --- Create API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the Grass Growth Rate Prediction API v3.0 (Stateless). Visit /docs for API documentation."}

@app.post("/predict", response_model=PredictionResponse)
def predict_growth_rate(request: PredictionRequest):
    try:
        prediction_timestamp = pd.to_datetime(request.prediction_timestamp_utc)
        
        # 1. Convert the incoming JSON data to a pandas DataFrame
        if not request.weather_history:
            raise HTTPException(status_code=400, detail="The 'weather_history' list cannot be empty.")
              
        weather_history_df = pd.DataFrame([record.dict() for record in request.weather_history])
        weather_history_df['period_start_utc'] = pd.to_datetime(weather_history_df['period_start_utc'])
        weather_history_df.set_index('period_start_utc', inplace=True)
        # Add farm_id column for use in the feature pipeline
        weather_history_df['farm_id'] = request.farm_id
        
        # 2. Call the feature engineering pipeline to calculate all features in real-time
        features_dict = create_all_features(
            farm_id=request.farm_id,
            timestamp=prediction_timestamp,
            weather_history_df=weather_history_df
        )
        
        feature_df = pd.DataFrame([features_dict])
        
        # 3. Prepare the features for the model (ensuring the correct column order)
        missing_cols = set(feature_columns) - set(feature_df.columns)
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Could not generate all required features for the given input. Missing: {missing_cols}")
        
        X_predict = feature_df[feature_columns]

        # 4. Standardize the features using the loaded scaler
        X_predict_scaled = scaler.transform(X_predict)
        
        # 5. Make the prediction
        prediction = model.predict(X_predict_scaled)
        
        return PredictionResponse(
            farm_id=request.farm_id,
            prediction_timestamp_utc=request.prediction_timestamp_utc,
            predicted_growth_rate=prediction[0]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")