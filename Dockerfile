# ─────────────────────────────────────────────────────────────
# Stage 1: Builder – install Python deps
# ─────────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir fastapi uvicorn[standard] pydantic

# ─────────────────────────────────────────────────────────────
# Stage 2: Runtime image
# ─────────────────────────────────────────────────────────────
FROM python:3.10-slim

LABEL maintainer="ML Engineering Team"
LABEL description="Employment Rate Forecasting API (LSTM/GRU/CNN)"
LABEL version="1.0.0"

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY employment_forecasting.py .
COPY main.py .
COPY forecasting/ ./forecasting/
COPY example_data.csv .

# Copy model files (if present at build time)
COPY --chown=appuser:appuser employment_forecast_lstm_final.h5 ./employment_forecast_lstm_final.h5
COPY --chown=appuser:appuser employment_forecast_gru_final.h5  ./employment_forecast_gru_final.h5
COPY --chown=appuser:appuser employment_forecast_cnn_final.h5  ./employment_forecast_cnn_final.h5

# Non-root user for security
RUN useradd -m appuser
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
