# Grass Growth Rate Prediction API

This project provides a **FastAPI-based machine learning API** for predicting daily grass growth rate using weather data.  
The API is powered by a trained **RandomForest Regressor**, combined with a comprehensive **feature engineering pipeline** that extracts features from 1-day, 3-day, 7-day, and 14-day weather windows.

The system is designed to be **stateless**: clients must provide the full weather history needed for feature generation, making the service scalable and flexible.

---

## 📂 Project Structure

```
grass_growth_api/
│
├── app/
│   ├── main.py                # FastAPI app & API endpoints
│   ├── feature_pipeline.py    # Feature engineering logic
│   └── ml_models/             # Saved model artifacts
│       ├── rf_model.joblib
│       ├── scaler.joblib
│       └── feature_columns.joblib
│
├── data/
│   ├── growth_rate_data.csv
│   └── historic_weather_data.csv
│
├── train_and_save_model.py    # Train model and save artifacts
│
├── test/
│   ├── generate_payload.py        # Utility to generate synthetic test payloads
│   └── payload.json               # Example test payload
│
├── train_process/             # Detailed Training Process to get the best Model
│   ├── README.md              # Document of Train process and performance improvement
│   ├── step1.interpolate_weather_data.py
│   ├── step2.feature extraction.py
│   ├── step3.baseline train.py
│   ├── step4.random forest train.py
│   └── ...
│
├── requirements.txt           # Dependencies
└── README.md                  # This documentation
```

---

## 🧪 Model Training & Tuning

The detailed **model training and hyperparameter tuning process** is documented in:

📄 [train_process/README.md](train_process/README.md)

Please refer to this document for an in-depth explanation of how the final RandomForest model was optimized.

All related training and tuning scripts are stored in the `train_process/` directory.

---

## ⚙️ Setup and Installation

### 1. Prerequisites

- **Python 3.13.3** (recommended)

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare raw data

Place the following files in the `data/` directory:

- `growth_rate_data.csv`
- `interpolated_weather_data.csv`

---

## 🚀 Deployment Guide

### Step 1: Train and Save the Model

Run the training script to generate all features, train the RandomForest model, and save artifacts (`rf_model.joblib`, `scaler.joblib`, `feature_columns.joblib`) into `app/ml_models/`.

```bash
python train_and_save_model.py
```

Artifacts will be saved to:

- `app/ml_models/rf_model.joblib`
- `app/ml_models/scaler.joblib`
- `app/ml_models/feature_columns.joblib`

---

### Step 2: Start the API Server

Use **uvicorn** to start the FastAPI application:

```bash
uvicorn app.main:app --reload
```

- API will be available at: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Interactive docs (Swagger UI): [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 📡 Using the API

### Endpoint

**POST** `/predict` – Predict grass growth rate.

### Request Body

```json
{
  "farm_id": "farm_id",
  "prediction_timestamp_utc": "YYYY-MM-DD HH:MM:SS",
  "weather_history": [
    {
      "period_start_utc": "YYYY-MM-DD HH:MM:SS",
      "air_temperature_celsius": 12.5,
      "global_horizontal_irradiance": 400.0,
      "precipitation_rate": 0.2
    }，
    {
      "period_start_utc": "YYYY-MM-DD HH:MM:SS",
      "air_temperature_celsius": 12.7,
      "global_horizontal_irradiance": 420.0,
      "precipitation_rate": 0.2
    }
  ]
}
```

- `weather_history` must contain at least **14 days** of 30-min interval records (~672 rows).

### Successful Response

```json
{
  "farm_id": "farm_id",
  "prediction_timestamp_utc": "YYYY-MM-DD HH:MM:SS",
  "predicted_growth_rate": 54.32
}
```

---

## 🧪 Testing

### Option 1: Swagger UI

Visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs), expand `/predict`, and try it out with a JSON payload.

### Option 2: Generate Synthetic Payload (Recommended)

You can generate a ready-to-use JSON payload for testing:

```bash
cd test
python generate_payload.py > payload.json
```

Send the request via curl:

```bash
curl -X 'POST'   'http://127.0.0.1:8000/predict'   -H 'accept: application/json'   -H 'Content-Type: application/json'   -d '@payload.json'
```

---

## 📊 Key Notes

- **Feature Engineering**: Includes daily/weekly aggregates, GDD, sunshine/rain duration, and interaction terms.
- **Model**: RandomForestRegressor with tuned hyperparameters.
- **Stateless Design**: Requires full weather history in the request payload.

---

## 🔮 Next Steps

To move from local development to a **production-ready deployment** of the FastAPI inference model, consider the following:

1. **Containerization**

   - Package the FastAPI app and model artifacts into a Docker image.
   - Ensure the image includes Python 3.13.3, all dependencies from `requirements.txt`, and the trained model files.

2. **Production Server**

   - Replace `uvicorn --reload` with a production-grade ASGI server setup:
     ```bash
     gunicorn -k uvicorn.workers.UvicornWorker app.main:app --bind 0.0.0.0:8000
     ```
   - This provides better performance and stability in production.

3. **Cloud Deployment**

   - Deploy the container to a cloud provider such as AWS (ECS/EKS), GCP (Cloud Run/GKE), or Azure (App Service/AKS).
   - Configure auto-scaling based on request volume.

4. **CI/CD Pipeline**

   - Automate building, testing, and deployment of the API using GitHub Actions, GitLab CI, or other CI/CD tools.

5. **Monitoring & Logging**

   - Integrate monitoring (Prometheus + Grafana, or cloud-native monitoring).
   - Add structured logging (e.g., JSON logs) for easier debugging in production.

6. **Model Lifecycle Management**

   - Version control the trained models in `app/ml_models/`.
   - Expose an endpoint for model metadata (version, training date, evaluation metrics).
   - Optionally support hot-swapping models without restarting the API.

7. **Security & Reliability**
   - Add request validation and rate limiting.
   - Enable HTTPS and authentication (JWT, API key, etc.) if exposed publicly.
   - Add retry logic and error handling for robustness.
