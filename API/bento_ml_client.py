import logging
from typing import Any, Dict

import bentoml
from boneNarrowClassification import BoneMarrowClassificationInput, BoneMarrowRegressionInput

logger = logging.getLogger(__name__)

ENDPOINT = "127.0.0.1:3000"


class BentoMLClient:
    def __init__(self, endpoint: str = ENDPOINT):
        if not endpoint:
            raise ValueError("Endpoint cannot be empty")
        self.endpoint = endpoint
        try:
            self.client = bentoml.Client(endpoint)
            logger.info(
                f"BentoML client initialized with endpoint: {endpoint}")
        except Exception as e:
            logger.error(f"Failed to initialize BentoML client: {e}")
            raise

    def predict_classification(self, input_data: BoneMarrowClassificationInput) -> Dict[str, Any]:
        if not isinstance(input_data, BoneMarrowClassificationInput):
            raise TypeError(
                "input_data must be of type BoneMarrowClassificationInput")

        try:
            logger.debug(f"Sending classification prediction request")
            response = self.client.post(
                "/predict_classification",
                json=input_data.model_dump()
            )
            logger.info("Classification prediction successful")
            return response.json()
        except Exception as e:
            logger.error(f"Classification prediction failed: {e}")
            raise

    def predict_regression(self, input_data: BoneMarrowRegressionInput) -> Dict[str, Any]:
        if not isinstance(input_data, BoneMarrowRegressionInput):
            raise TypeError(
                "input_data must be of type BoneMarrowRegressionInput")

        try:
            logger.debug(f"Sending regression prediction request")
            response = self.client.post(
                "/predict_regression",
                json=input_data.model_dump()
            )
            logger.info("Regression prediction successful")
            return response.json()
        except Exception as e:
            logger.error(f"Regression prediction failed: {e}")
            raise

    def predict_full(self, input_data: BoneMarrowClassificationInput) -> Dict[str, Any]:
        if not isinstance(input_data, BoneMarrowClassificationInput):
            raise TypeError(
                "input_data must be of type BoneMarrowClassificationInput")

        try:
            logger.debug(f"Sending full prediction request")
            response = self.client.post(
                "/predict_full",
                json=input_data.model_dump()
            )
            logger.info("Full prediction successful")
            return response.json()
        except Exception as e:
            logger.error(f"Full prediction failed: {e}")
            raise
