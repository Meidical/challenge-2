import logging
import os
from typing import Any, Dict

import requests

logger = logging.getLogger(__name__)

IS_DOCKER = os.getenv("IS_DOCKER", "false").lower() == "true"

if IS_DOCKER:
    BENTOML_URL = "http://bentoml:3000"  # Nome do serviço no docker-compose
else:
    BENTOML_URL = "http://localhost:3000"  # Localmente


class BentoMLClient:
    def __init__(self, endpoint: str = BENTOML_URL):
        if not endpoint:
            raise ValueError("Endpoint cannot be empty")
        self.endpoint = endpoint.rstrip('/')
        logger.info(f"BentoML client initialized with endpoint: {endpoint}")

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make a POST request to the BentoML service."""
        url = f"{self.endpoint}/{path}"
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request to {url} failed: {e}")
            raise

    def predict_classification(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.debug("Sending classification prediction request")
        return self._post("predict_classification", {"data": input_data})

    def predict_regression(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.debug("Sending regression prediction request")
        return self._post("predict_regression", {"data": input_data})

    def predict_full(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.debug("Sending full prediction request")
        return self._post("predict_full", {"data": input_data})

    def predict_full_dataframe_classification(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        logger.debug("Sending classification dataset prediction request")
        return self._post("predict_full_dataframe_classification", payload)

    def predict_full_dataframe_regression(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        logger.debug("Sending regression dataset prediction request")
        return self._post("predict_full_dataframe_regression", payload)

    def predict_relapse(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        logger.debug("Sending relapse dataset prediction request")
        return self._post("predict_relapse", payload)
