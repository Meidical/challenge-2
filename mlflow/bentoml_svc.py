import bentoml
from bentoml.io import JSON
import mlflow
import numpy as np
import pandas as pd
from pydantic import BaseModel

# Define the input schema for classification


class BoneMarrowClassificationInput(BaseModel):
    donor_age: float
    donor_age_below_35: str
    donor_ABO: str
    donor_CMV: str
    recipient_age: float
    recipient_age_below_10: str
    recipient_age_int: str
    recipient_gender: str
    recipient_body_mass: float
    recipient_ABO: str
    recipient_rh: str
    recipient_CMV: str
    disease: str
    disease_group: str
    gender_match: str
    ABO_match: str
    CMV_status: float
    HLA_match: str
    HLA_mismatch: str
    antigen: float
    allel: float
    HLA_group_1: str
    risk_group: str
    stem_cell_source: str


# Define the input schema for regression (includes survival_status)
class BoneMarrowRegressionInput(BaseModel):
    donor_age: float
    donor_age_below_35: str
    donor_ABO: str
    donor_CMV: str
    recipient_age: float
    recipient_age_below_10: str
    recipient_age_int: str
    recipient_gender: str
    recipient_body_mass: float
    recipient_ABO: str
    recipient_rh: str
    recipient_CMV: str
    disease: str
    disease_group: str
    gender_match: str
    ABO_match: str
    CMV_status: float
    HLA_match: str
    HLA_mismatch: str
    antigen: float
    allel: float
    HLA_group_1: str
    risk_group: str
    stem_cell_source: str
    is_dead: float


class BoneMarrowClassificationDataset(BaseModel):
    dataset: list[BoneMarrowClassificationInput]


class BoneMarrowRegressionDataset(BaseModel):
    dataset: list[BoneMarrowRegressionInput]


@bentoml.service(
    resources={"cpu": "2", "memory": "500MiB"},
    workers=1,
    traffic={"timeout": 20},
)
class BoneMarrowClassificationService:
    # Load models in __init__
    def __init__(self):

        classification_tag = self.get_model_by_run_id(task="classification")
        regression_tag = self.get_model_by_run_id(task="regression")

        print("Loaded classification model:", classification_tag)
        print("Loaded regression model:", regression_tag)

        # Load classification model
        self.classification_model = bentoml.models.get(
            classification_tag)
        self.classification_model_impl = self.classification_model.load_model()

        # Load regression model
        self.regression_model = bentoml.models.get(
            regression_tag)
        self.regression_model_impl = self.regression_model.load_model()

    # Classification endpoint - predicts survival status with probability
    @staticmethod
    def get_model_by_run_id(task):
        try:
            models = bentoml.models.list()

            # Set MLflow tracking URI
            mlflow.set_tracking_uri("http://127.0.0.1:5000")

            # Get experiment by task
            experiment_name = f"bone-marrow-{task}"
            experiment = mlflow.get_experiment_by_name(experiment_name)

            if experiment is None:
                raise ValueError(f"Experiment '{experiment_name}' not found")

            # Get best run based on task
            if task == "classification":
                metric = "f1_weighted_test"
            else:
                metric = "test_rmse"

            # Search for best run
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"tags.task = '{task}'",
                order_by=[
                    f"metrics.{metric} {'DESC' if task == 'classification' else 'ASC'}"],
                max_results=1
            )

            if runs.empty:
                raise ValueError(f"No runs found for task '{task}'")

            best_run_id = runs.iloc[0].run_id

            # Find model with matching run_id in BentoML
            models = bentoml.models.list()
            for model in models:
                if model.info.labels.get("run_id") == best_run_id:
                    print(f"Found best {task} model: {model.tag}")
                    return str(model.tag)

            raise ValueError(
                f"No BentoML model found with run_id: {best_run_id}")

        except Exception as e:
            print(f"Warning: Could not retrieve best model for {task}: {e}")
            # Return default tags
            if task == "classification":
                return "rf_tuned_optuna_classification:latest"
            else:
                return "rf_tuned_optuna_regression:latest"

    @bentoml.api
    def predict_classification(self, data: BoneMarrowClassificationInput) -> dict:
        # Prepare input data as pandas DataFrame with column names
        input_df = pd.DataFrame([{
            'donor_age': data.donor_age,
            'donor_age_below_35': data.donor_age_below_35,
            'donor_ABO': data.donor_ABO,
            'donor_CMV': data.donor_CMV,
            'recipient_age': data.recipient_age,
            'recipient_age_below_10': data.recipient_age_below_10,
            'recipient_age_int': data.recipient_age_int,
            'recipient_gender': data.recipient_gender,
            'recipient_body_mass': data.recipient_body_mass,
            'recipient_ABO': data.recipient_ABO,
            'recipient_rh': data.recipient_rh,
            'recipient_CMV': data.recipient_CMV,
            'disease': data.disease,
            'disease_group': data.disease_group,
            'gender_match': data.gender_match,
            'ABO_match': data.ABO_match,
            'CMV_status': data.CMV_status,
            'HLA_match': data.HLA_match,
            'HLA_mismatch': data.HLA_mismatch,
            'antigen': data.antigen,
            'allel': data.allel,
            'HLA_group_1': data.HLA_group_1,
            'risk_group': data.risk_group,
            'stem_cell_source': data.stem_cell_source
        }])

        # Predict survival status
        prediction = self.classification_model_impl.predict(input_df)

        # Get probability - the model wrapper should support predict_proba directly
        try:
            # Try to call predict_proba on the wrapper
            if hasattr(self.classification_model_impl, 'predict_proba'):
                proba = self.classification_model_impl.predict_proba(input_df)
            # If not available, try to access the underlying sklearn model
            elif hasattr(self.classification_model_impl, '_model_impl'):
                proba = self.classification_model_impl._model_impl.predict_proba(
                    input_df)
            else:
                raise AttributeError("predict_proba not available")

            return {
                "survival_status": int(prediction[0]),
                "probability_alive": float(proba[0][0]),
                "probability_dead": float(proba[0][1])
            }
        except Exception as e:
            # Fallback if predict_proba is not available
            return {
                "survival_status": int(prediction[0]),
                "probability_alive": None,
                "probability_dead": None,
                "error": f"Could not retrieve probabilities: {str(e)}"
            }

    # Regression endpoint - predicts survival time (requires survival_status)
    @bentoml.api
    def predict_regression(self, data: BoneMarrowRegressionInput) -> dict:
        # Prepare input data as pandas DataFrame with column names
        input_df = pd.DataFrame([{
            'donor_age': data.donor_age,
            'donor_age_below_35': data.donor_age_below_35,
            'donor_ABO': data.donor_ABO,
            'donor_CMV': data.donor_CMV,
            'recipient_age': data.recipient_age,
            'recipient_age_below_10': data.recipient_age_below_10,
            'recipient_age_int': data.recipient_age_int,
            'recipient_gender': data.recipient_gender,
            'recipient_body_mass': data.recipient_body_mass,
            'recipient_ABO': data.recipient_ABO,
            'recipient_rh': data.recipient_rh,
            'recipient_CMV': data.recipient_CMV,
            'disease': data.disease,
            'disease_group': data.disease_group,
            'gender_match': data.gender_match,
            'ABO_match': data.ABO_match,
            'CMV_status': data.CMV_status,
            'HLA_match': data.HLA_match,
            'HLA_mismatch': data.HLA_mismatch,
            'antigen': data.antigen,
            'allel': data.allel,
            'HLA_group_1': data.HLA_group_1,
            'risk_group': data.risk_group,
            'stem_cell_source': data.stem_cell_source,
            'is_dead': data.is_dead
        }])

        # Predict survival time
        result = self.regression_model_impl.predict(input_df)

        return {
            "predicted_survival_time_days": float(result[0])
        }

    # Combined prediction endpoint
    @bentoml.api
    def predict_full(self, data: BoneMarrowClassificationInput) -> dict:
        # First predict survival status
        classification_result = self.predict_classification(data)

        # Use predicted survival status for regression
        regression_input = BoneMarrowRegressionInput(
            **data.model_dump(),
            # survival_status=classification_result["survival_status"]
            is_dead=classification_result["survival_status"]
        )
        regression_result = self.predict_regression(regression_input)

        return {
            "classification": {
                "survival_status": classification_result["survival_status"],
                "probability_alive": classification_result.get("probability_alive"),
                "probability_dead": classification_result.get("probability_dead")
            },
            "regression": {
                "predicted_survival_time_days": regression_result["predicted_survival_time_days"]
            }
        }

    @bentoml.api
    def predict_full_dataframe_regression(self, dataset: list[BoneMarrowRegressionInput]) -> dict:
        # receive a dataframe with multiple rows
        rows = [item.model_dump() for item in dataset]
        input_df = pd.DataFrame(rows)

        # Predict survival time
        result = self.regression_model_impl.predict(input_df)

        # return a dict with the predicted survival time
        return {
            "predictions": result.tolist()
        }

    @bentoml.api
    def predict_full_dataframe_classification(self, dataset: list[BoneMarrowClassificationInput]) -> dict:
        # receive a dataframe with multiple rows
        rows = [item.model_dump() for item in dataset]
        input_df = pd.DataFrame(rows)

        # Predict survival time
        result = self.classification_model_impl.predict(input_df)

        # return a dict with the predicted survival time
        return {
            "predictions": result.tolist()
        }
