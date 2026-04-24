# Postman Testing Guide — Employment Rate Forecasting API

This guide walks you through importing the Postman collection and testing every endpoint step by step.

---

## 1. Install Postman

- **Windows / Mac / Linux:** Download from https://www.postman.com/downloads/
- Install and create a free account (or skip sign-in for basic use)

---

## 2. Import the Collection

1. Open Postman
2. Click **Import** (top-left)
3. Select **File** tab
4. Click **Choose Files** and select `postman_collection.json`
5. Click **Import**

You should now see **"Employment Rate Forecasting API"** in your Collections panel.

---

## 3. Set Environment Variables

1. Click the **Environments** icon (gear icon, top-right)
2. Click **Add**
3. Name it: `Employment Forecasting Local`
4. Add these variables:

| Variable | Initial Value | Current Value |
|----------|---------------|---------------|
| base_url | http://localhost:8000 | http://localhost:8000 |
| model_type | lstm | lstm |
| steps | 12 | 12 |
| province | Ontario | Ontario |

5. Click **Save**
6. Select this environment from the dropdown (top-right of Postman)

---

## 4. Start the API First

Before running Postman tests, ensure the API is running:

```bash
# In your project directory:
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Or if using Docker:
```bash
docker run -p 8000:8000 employment-forecasting:latest
```

---

## 5. Test Each Endpoint

### Step-by-Step Test Sequence

**Test 1: Root**
1. Expand **Health & Info** folder
2. Click **GET / - Root**
3. Click **Send**
4. Expected: Status `200 OK`, body contains `"message"`

**Test 2: Health Check**
1. Click **GET /health**
2. Click **Send**
3. Expected: `"status": "healthy"`, three models shown in `models_loaded`

**Test 3: List Models**
1. Click **GET /models**
2. Click **Send**
3. Expected: `available_models` array with lstm, gru, cnn

**Test 4: Model Info**
1. Click **GET /models/lstm**
2. Click **Send**
3. Expected: `"loaded": true`, input/output shape info

**Test 5: List Provinces**
1. Click **GET /provinces**
2. Click **Send**
3. Expected: 10 Canadian provinces listed

**Test 6: Forecast (LSTM, 12 months)**
1. Expand **Forecasting** folder
2. Click **POST /forecast - LSTM 12 months**
3. Click **Send**
4. Expected: Status `200`, `"forecast"` array with 12 rows

**Test 7: Forecast (GRU, Ontario, 6 months)**
1. Click **POST /forecast - GRU Ontario 6 months**
2. Click **Send**
3. Expected: Only Ontario columns in results, 6 rows

**Test 8: Forecast (CNN, 24 months)**
1. Click **POST /forecast - CNN 24 months**
2. Click **Send**
3. Expected: `"forecast"` array with 24 rows

**Test 9: GET Forecast (path parameter)**
1. Click **GET /forecast/lstm?steps=6&province=Ontario**
2. Click **Send**
3. Expected: Same as POST forecast

**Test 10: Single-Step Predict**
1. Click **POST /predict - Single Step**
2. Click **Send**
3. Expected: `"prediction"` array with float values

---

## 6. Test Error Scenarios

**Test E1: Invalid model type**
1. Expand **Error Scenarios** folder
2. Click **POST /forecast - Invalid model (400)**
3. Click **Send**
4. Expected: Status `422 Unprocessable Entity`

**Test E2: Steps out of range**
1. Click **POST /forecast - Steps out of range (422)**
2. Click **Send**
3. Expected: Status `422`

**Test E3: Unknown model path**
1. Click **GET /models/invalid - 404**
2. Click **Send**
3. Expected: Status `404 Not Found`

---

## 7. Using cURL from Terminal

You can also test directly from terminal without Postman:

```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/models

# Forecast - LSTM 12 months
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"model_type": "lstm", "steps": 12}'

# Forecast - GRU, Ontario, 6 months
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"model_type": "gru", "steps": 6, "province": "Ontario"}'

# Forecast - CNN, Alberta, 24 months
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"model_type": "cnn", "steps": 24, "province": "Alberta"}'

# GET forecast endpoint
curl "http://localhost:8000/forecast/lstm?steps=12&province=Ontario"

# List provinces
curl http://localhost:8000/provinces

# Unauthorized (if auth added in future)
curl -H "Authorization: Bearer invalid-token" http://localhost:8000/forecast/lstm
```

---

## 8. cURL on Windows (Command Prompt)

Windows Command Prompt uses different quote syntax:

```cmd
curl -X POST http://localhost:8000/forecast ^
  -H "Content-Type: application/json" ^
  -d "{\"model_type\": \"lstm\", \"steps\": 12}"
```

Or use Git Bash on Windows (same syntax as Linux/Mac).

---

## 9. Running All Tests Automatically

1. In Postman, right-click **Employment Rate Forecasting API** collection
2. Click **Run collection**
3. Click **Run Employment Rate Forecasting API**
4. View pass/fail results for all automated test scripts

---

## 10. Expected Response Templates

### Success Response Template
```json
{
  "model": "LSTM",
  "steps": 12,
  "province_filter": null,
  "forecast": [
    {
      "month": "2025-07",
      "Alberta_FT_F": 854.21,
      "Alberta_FT_M": 1024.55
    }
  ],
  "columns": ["month", "Alberta_FT_F", "Alberta_FT_M", "..."],
  "generated_at": "2025-04-16T14:00:00Z"
}
```

### Validation Error Template
```json
{
  "detail": [
    {
      "loc": ["body", "steps"],
      "msg": "ensure this value is less than or equal to 60",
      "type": "value_error.number.not_le"
    }
  ]
}
```

### 404 Not Found Template
```json
{
  "detail": "Model 'bert' not found. Choose from: lstm, gru, cnn"
}
```

---

## 11. DO / DO NOT

| DO | DO NOT |
|----|--------|
| Start the API before running Postman | Run tests without the server running |
| Set environment variables before running | Hardcode the base_url in requests |
| Use the automated test scripts (Test tab) | Skip the health check |
| Test error scenarios to understand failures | Ignore 422 responses |
