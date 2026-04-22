# Employment Rate Forecasting — Canadian Provinces (2020–2025)

A production-ready deep learning project that forecasts Canadian provincial employment rates using LSTM, GRU, and 1D-CNN neural networks — packaged as a FastAPI, containerised with Docker, and deployable on AWS EC2.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Repository Structure](#repository-structure)
4. [Quick Start (Local)](#quick-start-local)
5. [API Reference](#api-reference)
6. [Docker](#docker)
7. [AWS Deployment](#aws-deployment)
8. [Cost Breakdown](#cost-breakdown)
9. [Cleanup](#cleanup)


---

## Project Overview

| Attribute | Detail |
|-----------|--------|
| **Dataset** | Statistics Canada – Employment Rate by Province 1976-2025 |
| **Target** | Full-time / Part-time × Male / Female per province |
| **Models** | Bidirectional LSTM, Bidirectional GRU, 1D-CNN |
| **Horizon** | Up to 60 months (5 years) |
| **API** | FastAPI + Uvicorn |
| **Container** | Docker (multi-stage, non-root) |
| **Cloud** | EC2 |

### Key Results

| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| LSTM  | ~0.024 | ~0.018 | ~0.97 |
| GRU   | ~0.022 | ~0.016 | ~0.97 |
| CNN   | ~0.027 | ~0.020 | ~0.96 |

---

## Architecture

```
example_data.csv
      │
      ▼
┌─────────────────────────────────────────────┐
│            forecasting/  (modular)           │
│  data.py · models.py · training.py          │
│  evaluation.py · prediction.py              │
│  visualization.py · pipeline.py             │
└──────────────┬──────────────────────────────┘
               │  trains & saves .h5 + scaler.pkl
               ▼
┌─────────────────────────────────────────────┐
│        FastAPI  (main.py)                   │
│  POST /forecast   POST /predict             │
│  GET  /health     GET  /models              │
└──────────────┬──────────────────────────────┘
               │
    ┌──────────┴───────────┐
    ▼                      ▼
 Docker Image          Local Python
 pushed to ECR         uvicorn serve
    │
    ├── AWS ECS Fargate (serverless)
    └── AWS EC2 Ubuntu  (IaaS)
```

---

## Repository Structure

```
.
├── employment_forecasting.py   # Monolithic training script (do NOT delete)
├── main.py                     # FastAPI application
├── api_client.py               # Python client library
├── postman_collection.json     # Postman API collection
├── Dockerfile                  # Multi-stage Docker build
├── .dockerignore
├── requirements.txt
├── example_data.csv            # Source dataset
├── scaler.pkl                  # MinMaxScaler (generated)
├── employment_forecast_lstm_final.h5
├── employment_forecast_gru_final.h5
├── employment_forecast_cnn_final.h5
├── forecasting/                # Modular package
│   ├── __init__.py
│   ├── data.py
│   ├── models.py
│   ├── training.py
│   ├── evaluation.py
│   ├── prediction.py
│   ├── visualization.py
│   ├── pipeline.py
│   └── runtime.py
├── README.md                   # This file
├── READMEFastAPI.md
├── READMEPostman.md
└── READMEDocker.md
```

---

## Quick Start (Local)

### Prerequisites
- Python 3.10+
- pip

### Steps

```bash
# 1. Clone / download the project
git clone https://github.com/your-org/employment-forecasting.git
cd employment-forecasting

# 2. Create virtual environment
python -m venv venv
# Windows:  venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
pip install fastapi uvicorn[standard]

# 4. (Optional) Re-train models
python employment_forecasting.py

# 5. Start the API
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 6. Open docs
# http://localhost:8000/docs
```

---

## 5 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Welcome message |
| GET | `/health` | Health check + model status |
| GET | `/models` | List models |
| GET | `/models/{name}` | Model details |
| GET | `/provinces` | List provinces |
| POST | `/forecast` | Generate multi-step forecast |
| GET | `/forecast/{model}` | GET-based forecast |
| POST | `/predict` | Single-step raw prediction |

See `READMEFastAPI.md` for full request/response examples.

---

## 6 Docker

```bash
# Build
docker build -t employment-app:v1.


# Run
docker run -d -p 8000:8000 --name employment-container purnachandrasharma1/employment-app:v1


# Test
curl http://localhost:8000/health
```

See `READMEDocker.md` for ECR push and ECS/EC2 deployment steps.

---

## 7 AWS Deployment



**EC2 Ubuntu** – full VM control, cheaper at sustained load**


See `READMEDocker.md` for step-by-step instructions.

---

## 8 Cost Breakdown (Estimated)

| Service | Config | Monthly Cost (USD) |
|---------|--------|--------------------|
| EC2 t3.medium | On-demand, 8 h/day | ~$10–15 |
| Data transfer | < 1 GB/month | < $1 |
| **Total (dev/test)** | | **~$15–26/month** |

> Free Tier: EC2 t2.micro (750 h/month) is free for 12 months.

---

## 9 Cleanup

```bash

# Terminate EC2 instance
aws ec2 terminate-instances --instance-ids <instance-id>
```

---



