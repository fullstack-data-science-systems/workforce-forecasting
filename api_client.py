"""
Employment Rate Forecasting - Python API Client
Reusable client with error handling, retries, and auth support.
"""

import time
import logging
from typing import Optional, Dict, Any, List

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmploymentForecastClient:
    """
    Python client for the Employment Rate Forecasting API.
    
    Usage:
        client = EmploymentForecastClient(base_url="http://localhost:8000")
        health = client.health()
        forecast = client.forecast(model_type="lstm", steps=12, province="Ontario")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = self._build_session(api_key, max_retries)

    def _build_session(self, api_key: Optional[str], max_retries: int) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update({"Content-Type": "application/json", "Accept": "application/json"})
        if api_key:
            session.headers.update({"Authorization": f"Bearer {api_key}"})
        return session

    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        logger.info(f"GET {url}")
        try:
            resp = self.session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to {url}")
            raise

    def _post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        logger.info(f"POST {url}")
        try:
            resp = self.session.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            raise

    # ─── Public methods ───────────────────────────────────

    def health(self) -> Dict[str, Any]:
        """Check API health and model availability."""
        return self._get("/health")

    def list_models(self) -> Dict[str, Any]:
        """List all available models."""
        return self._get("/models")

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get info for a specific model (lstm, gru, cnn)."""
        return self._get(f"/models/{model_name}")

    def list_provinces(self) -> List[str]:
        """Get list of available Canadian provinces."""
        return self._get("/provinces")["provinces"]

    def forecast(
        self,
        model_type: str = "lstm",
        steps: int = 12,
        province: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate employment rate forecasts.
        
        Args:
            model_type: 'lstm', 'gru', or 'cnn'
            steps: Number of months to forecast (1-60)
            province: Optional province name to filter results
        
        Returns:
            Dict with forecast data and metadata
        """
        payload = {"model_type": model_type, "steps": steps}
        if province:
            payload["province"] = province
        return self._post("/forecast", payload)

    def forecast_get(
        self,
        model_name: str = "lstm",
        steps: int = 12,
        province: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Convenience GET-based forecast."""
        params = {"steps": steps}
        if province:
            params["province"] = province
        return self._get(f"/forecast/{model_name}", params=params)

    def predict_single_step(
        self,
        input_data: List[List[float]],
        model_type: str = "lstm",
    ) -> Dict[str, Any]:
        """
        Single-step prediction from raw input.
        
        Args:
            input_data: 12 × N array of scaled feature values
            model_type: 'lstm', 'gru', or 'cnn'
        """
        payload = {"input_data": input_data, "model_type": model_type}
        return self._post("/predict", payload)


# ─── Convenience functions ────────────────────────────────

def quick_forecast(
    model_type: str = "lstm",
    steps: int = 12,
    province: Optional[str] = None,
    base_url: str = "http://localhost:8000",
) -> Dict[str, Any]:
    """One-liner forecast function."""
    client = EmploymentForecastClient(base_url=base_url)
    return client.forecast(model_type=model_type, steps=steps, province=province)


# ─── Demo ────────────────────────────────────────────────

if __name__ == "__main__":
    client = EmploymentForecastClient(base_url="http://localhost:8000")

    print("=== Health Check ===")
    print(client.health())

    print("\n=== Available Models ===")
    print(client.list_models())

    print("\n=== LSTM 12-Month Forecast (Ontario) ===")
    result = client.forecast(model_type="lstm", steps=12, province="Ontario")
    for row in result["forecast"][:3]:
        print(row)

    print("\n=== GRU 6-Month Forecast ===")
    result = client.forecast(model_type="gru", steps=6)
    print(f"Returned {len(result['forecast'])} rows")
