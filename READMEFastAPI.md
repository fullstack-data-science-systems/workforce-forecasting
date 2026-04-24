# FastAPI — Employment Rate Forecasting

This document covers everything you need to install, run, and test the FastAPI service.

---

## 1. What is FastAPI?

FastAPI is a modern, high-performance Python web framework for building APIs. It auto-generates interactive documentation (Swagger UI at `/docs`, ReDoc at `/redoc`) and uses Python type hints for input validation.

---

## 2. Installation

```bash
pip install fastapi uvicorn[standard] pydantic
```

---

## 3. Starting the Server

### Development mode (with auto-reload)
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Production mode (multiple workers)
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2
```

### Verify it's running
Open your browser: http://localhost:8000/docs

---

## 4. Endpoints Reference

### GET /health
Returns the health status and which models are loaded.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": {
    "lstm": true,
    "gru": true,
    "cnn": true
  },
  "timestamp": "2025-04-16T14:00:00Z",
  "version": "1.0.0"
}
```

---

### GET /models
Lists all available models.

**cURL:**
```bash
curl http://localhost:8000/models
```

**Response:**
```json
{
  "available_models": ["lstm", "gru", "cnn"],
  "model_details": {
    "lstm": "Bidirectional LSTM (2 Bi-LSTM layers + 1 LSTM + Dense)",
    "gru":  "Bidirectional GRU (2 Bi-GRU layers + 1 GRU + Dense)",
    "cnn":  "1D Convolutional CNN (3 Conv1D layers + GlobalAvgPool + Dense)"
  },
  "total_loaded": 3
}
```

---

### GET /models/{model_name}
Get detailed information about one model.

**cURL:**
```bash
curl http://localhost:8000/models/lstm
```

**Path parameters:**
| Parameter | Values |
|-----------|--------|
| model_name | lstm, gru, cnn |

---

### GET /provinces
Lists all available Canadian provinces.

**cURL:**
```bash
curl http://localhost:8000/provinces
```

**Response:**
```json
{
  "provinces": [
    "Alberta", "British Columbia", "Manitoba", "New Brunswick",
    "Newfoundland and Labrador", "Nova Scotia", "Ontario",
    "Prince Edward Island", "Quebec", "Saskatchewan"
  ],
  "count": 10
}
```

---

### POST /forecast
Generate multi-step employment rate forecasts.

**cURL — LSTM 12 months:**
```bash
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"model_type": "lstm", "steps": 12}'
```

**cURL — GRU, Ontario only:**
```bash
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"model_type": "gru", "steps": 6, "province": "Ontario"}'
```

**Request body:**
```json
{
  "model_type": "lstm",
  "steps": 12,
  "province": "Ontario"
}
```

| Field | Type | Required | Default | Notes |
|-------|------|----------|---------|-------|
| model_type | string | No | "lstm" | lstm, gru, or cnn |
| steps | integer | No | 12 | 1–60 |
| province | string | No | null | Full province name |

**Success Response (200):**
```json
{
  "model": "LSTM",
  "steps": 12,
  "province_filter": "Ontario",
  "forecast": [
    {
      "month": "2025-07",
      "Ontario_FT_F": 3254.12,
      "Ontario_FT_M": 3801.45,
      "Ontario_PT_F": 821.33,
      "Ontario_PT_M": 472.10
    }
  ],
  "columns": ["month", "Ontario_FT_F", "Ontario_FT_M", "Ontario_PT_F", "Ontario_PT_M"],
  "generated_at": "2025-04-16T14:00:00Z"
}
```

**Validation Error (422):**
```json
{
  "detail": [
    {
      "loc": ["body", "model_type"],
      "msg": "value is not a valid enumeration member",
      "type": "value_error.enum"
    }
  ]
}
```

---

### GET /forecast/{model_name}
Convenience GET endpoint for quick testing.

**cURL:**
```bash
curl "http://localhost:8000/forecast/lstm?steps=6&province=Ontario"
```

**Query parameters:**
| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| steps | integer | 12 | 1–60 |
| province | string | null | Optional filter |

---

### POST /predict
Single-step prediction from a raw input sequence (advanced use).

**cURL:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "lstm",
    "input_data": [
      [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
      [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    ]
  }'
```

> Note: input_data must be a 12 × N array where N = number of features (40 for full dataset).

---

## 5. Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| PORT | 8000 | Server port |
| WORKERS | 1 | Number of Uvicorn workers |
| LOG_LEVEL | info | Logging verbosity |

---

## 6. Python Client Example

```python
from api_client import EmploymentForecastClient

client = EmploymentForecastClient(base_url="http://localhost:8000")

# Health check
print(client.health())

# Forecast Ontario for 12 months using LSTM
result = client.forecast(model_type="lstm", steps=12, province="Ontario")
for row in result["forecast"][:3]:
    print(row)
```

---

## 7. DO NOT / DO

| DO | DO NOT |
|----|--------|
| Keep model .h5 files in the same directory as main.py | Move or rename .h5 files |
| Use `uvicorn main:app` to start the server | Delete scaler.pkl |
| Provide steps between 1 and 60 | Provide steps > 60 |
| Use full province names | Abbreviate province names |
