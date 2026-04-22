"""
Employment Rate Forecasting - FastAPI Application
Serves LSTM, GRU, and CNN models for Canadian employment rate forecasting.
"""

from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import pickle
import os
import logging
from datetime import datetime

from forecasting.runtime import configure_runtime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────
app = FastAPI(
    title="Employment Rate Forecasting API",
    description=(
        "Deep Learning API for forecasting Canadian employment rates "
        "using LSTM, GRU, and 1D-CNN models. Supports multi-province, "
        "multi-metric forecasting (Full-time/Part-time × Male/Female)."
    ),
    version="1.0.0",
    contact={"name": "ML Engineering Team"},
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Frontend serving setup
# ─────────────────────────────────────────────
# Get the absolute path of the directory this file is in
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
INDEX_FILE = os.path.join(FRONTEND_DIR, "index.html")

# Log frontend setup for debugging
logger.info(f"Frontend directory: {FRONTEND_DIR}")
logger.info(f"Index file: {INDEX_FILE}")
logger.info(f"Frontend directory exists: {os.path.exists(FRONTEND_DIR)}")
logger.info(f"Index file exists: {os.path.exists(INDEX_FILE)}")

# Serve static files (CSS, JS, images)
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
    logger.info("Static files mounted at /static")
else:
    logger.warning("Frontend directory not found - static files not mounted")

# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

# Serve frontend homepage
@app.get("/")
def serve_frontend():
    """Serve the frontend homepage."""
    if os.path.exists(INDEX_FILE):
        logger.info(f"Serving frontend from: {INDEX_FILE}")
        return FileResponse(INDEX_FILE)
    logger.error(f"Frontend index file not found at: {INDEX_FILE}")
    return {
        "error": "Frontend not found",
        "checked_path": INDEX_FILE,
        "current_workdir": os.getcwd(),
        "base_dir": BASE_DIR,
        "frontend_dir_exists": os.path.exists(FRONTEND_DIR),
        "index_file_exists": os.path.exists(INDEX_FILE)
    }

# ─────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────
class ForecastRequest(BaseModel):
    model_type: str = Field("lstm", description="Model type: 'lstm', 'gru', or 'cnn'")
    steps: int = Field(12, ge=1, le=60, description="Number of months to forecast (1-60)")
    province: Optional[str] = Field(None, description="Filter results to a specific province")

    @validator("model_type")
    def validate_model_type(cls, v):
        allowed = {"lstm", "gru", "cnn"}
        if v.lower() not in allowed:
            raise ValueError(f"model_type must be one of {allowed}")
        return v.lower()


class PredictRequest(BaseModel):
    input_data: List[List[float]] = Field(
        ...,
        description="12 × N_features array of scaled input values (last 12 months)"
    )
    model_type: str = Field("lstm", description="Model type: 'lstm', 'gru', or 'cnn'")

    @validator("model_type")
    def validate_model_type(cls, v):
        allowed = {"lstm", "gru", "cnn"}
        if v.lower() not in allowed:
            raise ValueError(f"model_type must be one of {allowed}")
        return v.lower()

    @validator("input_data")
    def validate_input_shape(cls, v):
        if len(v) != 12:
            raise ValueError("input_data must have exactly 12 time steps (rows)")
        return v


class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]
    timestamp: str
    version: str


# ─────────────────────────────────────────────
# Model registry (lazy-loaded at startup)
# ─────────────────────────────────────────────
MODEL_REGISTRY: Dict[str, Any] = {}
SCALER = None
FEATURE_COLUMNS: List[str] = []
MODEL_FILES = {
    "lstm": "employment_forecast_lstm_final.h5",
    "gru":  "employment_forecast_gru_final.h5",
    "cnn":  "employment_forecast_cnn_final.h5",
}


@app.on_event("startup")
async def load_models():
    """Load all models and scaler on startup."""
    global SCALER, FEATURE_COLUMNS

    configure_runtime()

    try:
        import tensorflow as tf
    except Exception as exc:
        logger.warning(
            "TensorFlow is unavailable during startup; model loading will be skipped. %s",
            exc,
        )
        return

    logger.info("Loading models...")

    for name, path in MODEL_FILES.items():
        if os.path.exists(path):
            try:
                MODEL_REGISTRY[name] = tf.keras.models.load_model(path, compile=False)
                logger.info(f"  ✓ {name.upper()} model loaded from {path}")
            except Exception as e:
                logger.warning(f"  ✗ Failed to load {name}: {e}")
        else:
            logger.warning(f"  ✗ Model file not found: {path}")

    if os.path.exists("scaler.pkl"):
        with open("scaler.pkl", "rb") as f:
            SCALER = pickle.load(f)
        logger.info("  ✓ Scaler loaded")

    # Derive feature columns from example data (for labelling forecasts)
    if os.path.exists("example_data.csv"):
        try:
            from forecasting.data import load_data, pivot_employment_data, handle_missing_values
            df = load_data("example_data.csv")
            df_p = pivot_employment_data(df)
            _, numeric_cols = handle_missing_values(df_p)
            FEATURE_COLUMNS = list(numeric_cols)
        except Exception as e:
            logger.warning(f"Could not derive feature columns: {e}")

    logger.info("Startup complete.")


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/", tags=["Root"])
def root():
    """Welcome endpoint."""
    return {
        "message": "Employment Rate Forecasting API",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Returns API health and model availability."""
    return HealthResponse(
        status="healthy",
        models_loaded={name: name in MODEL_REGISTRY for name in MODEL_FILES},
        timestamp=datetime.utcnow().isoformat() + "Z",
        version="1.0.0",
    )


@app.get("/models", tags=["Models"])
def list_models():
    """List all available models and their status."""
    return {
        "available_models": list(MODEL_REGISTRY.keys()),
        "model_details": {
            "lstm": "Bidirectional LSTM (2 Bi-LSTM layers + 1 LSTM + Dense)",
            "gru":  "Bidirectional GRU (2 Bi-GRU layers + 1 GRU + Dense)",
            "cnn":  "1D Convolutional CNN (3 Conv1D layers + GlobalAvgPool + Dense)",
        },
        "total_loaded": len(MODEL_REGISTRY),
    }


@app.get("/models/{model_name}", tags=["Models"])
def get_model_info(model_name: str = Path(..., description="Model name: lstm, gru, or cnn")):
    """Get detailed information about a specific model."""
    model_name = model_name.lower()
    if model_name not in MODEL_FILES:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found. Choose from: lstm, gru, cnn")

    loaded = model_name in MODEL_REGISTRY
    info = {
        "name": model_name.upper(),
        "loaded": loaded,
        "file": MODEL_FILES[model_name],
    }
    if loaded:
        model = MODEL_REGISTRY[model_name]
        info["input_shape"] = str(model.input_shape)
        info["output_shape"] = str(model.output_shape)
        info["total_parameters"] = model.count_params()
    return info


@app.post("/forecast", tags=["Forecasting"])
def generate_forecast(request: ForecastRequest):
    """
    Generate multi-step employment rate forecasts.
    
    - **model_type**: Choose lstm, gru, or cnn
    - **steps**: Number of future months (1-60)
    - **province**: Optional province filter (e.g., 'Ontario')
    """
    model_type = request.model_type
    if model_type not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model_type}' not loaded. Check /health for model status."
        )
    if SCALER is None:
        raise HTTPException(status_code=503, detail="Scaler not available. Ensure scaler.pkl exists.")
    if not os.path.exists("example_data.csv"):
        raise HTTPException(status_code=503, detail="example_data.csv not found.")

    try:
        from forecasting.data import load_data, pivot_employment_data, handle_missing_values, scale_features
        from forecasting.prediction import forecast_future

        df = load_data("example_data.csv")
        df_p = pivot_employment_data(df)
        df_p["month"] = pd.to_datetime(df_p["month"])
        df_clean, numeric_cols = handle_missing_values(df_p)
        features = df_clean[numeric_cols].values
        scaled_features, _ = scale_features(features)

        last_seq = scaled_features[-12:]
        model = MODEL_REGISTRY[model_type]
        forecasts_scaled = _iterative_forecast(model, last_seq, request.steps)
        forecasts = SCALER.inverse_transform(forecasts_scaled)

        last_date = pd.to_datetime(df_clean["month"].values[-1])
        forecast_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=request.steps, freq="MS"
        )

        result_df = pd.DataFrame(forecasts, columns=list(numeric_cols))
        result_df.insert(0, "month", forecast_dates.strftime("%Y-%m").tolist())

        if request.province:
            prov_cols = [c for c in result_df.columns if request.province.lower() in c.lower()]
            if not prov_cols:
                raise HTTPException(
                    status_code=400,
                    detail=f"Province '{request.province}' not found in data."
                )
            result_df = result_df[["month"] + prov_cols]

        return {
            "model": model_type.upper(),
            "steps": request.steps,
            "province_filter": request.province,
            "forecast": result_df.to_dict(orient="records"),
            "columns": list(result_df.columns),
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Forecast error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", tags=["Forecasting"])
def single_step_predict(request: PredictRequest):
    """
    Single-step prediction from raw input sequence.
    Provide a 12 × N array of scaled feature values.
    """
    model_type = request.model_type
    if model_type not in MODEL_REGISTRY:
        raise HTTPException(status_code=503, detail=f"Model '{model_type}' not loaded.")

    try:
        model = MODEL_REGISTRY[model_type]
        X = np.array(request.input_data)
        X_reshaped = X.reshape(1, X.shape[0], X.shape[1])
        pred_scaled = model.predict(X_reshaped, verbose=0)[0]

        if SCALER is not None:
            pred_orig = SCALER.inverse_transform(pred_scaled.reshape(1, -1))[0]
            prediction = pred_orig.tolist()
        else:
            prediction = pred_scaled.tolist()

        return {
            "model": model_type.upper(),
            "prediction": prediction,
            "feature_columns": FEATURE_COLUMNS if FEATURE_COLUMNS else None,
            "note": "Values represent employment counts in thousands",
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/provinces", tags=["Data"])
def list_provinces():
    """List all Canadian provinces available in the dataset."""
    provinces = [
        "Alberta", "British Columbia", "Manitoba", "New Brunswick",
        "Newfoundland and Labrador", "Nova Scotia", "Ontario",
        "Prince Edward Island", "Quebec", "Saskatchewan"
    ]
    return {"provinces": provinces, "count": len(provinces)}


@app.get("/forecast/{model_name}", tags=["Forecasting"])
def forecast_by_model(
    model_name: str = Path(..., description="Model: lstm, gru, or cnn"),
    steps: int = Query(12, ge=1, le=60, description="Forecast horizon in months"),
    province: Optional[str] = Query(None, description="Optional province filter"),
):
    """Convenience GET endpoint for forecasting by model name."""
    request = ForecastRequest(model_type=model_name, steps=steps, province=province)
    return generate_forecast(request)


# ─────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────
def _iterative_forecast(model, last_sequence: np.ndarray, n_steps: int) -> np.ndarray:
    """Iterative multi-step forecast (auto-regressive)."""
    forecasts = []
    current_seq = last_sequence.copy()
    for _ in range(n_steps):
        x = current_seq.reshape(1, current_seq.shape[0], current_seq.shape[1])
        pred = model.predict(x, verbose=0)[0]
        forecasts.append(pred)
        current_seq = np.vstack([current_seq[1:], pred])
    return np.array(forecasts)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
