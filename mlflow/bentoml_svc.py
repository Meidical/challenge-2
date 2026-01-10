import bentoml
import numpy as np
import pandas as pd
from pydantic import BaseModel

# Define the input schema as a Pydantic model


class BoneMarrowInput(BaseModel):
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
    CMV_status: str
    HLA_match: str
    HLA_mismatch: str
    antigen: int
    allel: int
    HLA_group_1: str
    risk_group: str
    stem_cell_source: str
    tx_post_relapse: str
    CD34_x1e6_per_kg: float
    CD3_x1e8_per_kg: float
    CD3_to_CD34_ratio: float
    ANC_recovery: float
    PLT_recovery: float
    acute_GvHD_II_III_IV: str
    acute_GvHD_III_IV: str
    time_to_acute_GvHD_III_IV: float
    extensive_chronic_GvHD: str
    relapse: str
    survival_time: float
    survival_status: int


# Define the BentoML Service
@bentoml.service(
    resources={"cpu": "2", "memory": "500MiB"},
    workers=1,
    traffic={"timeout": 20},
)
class MyService:
    # Load model in __init__ instead of using runner
    def __init__(self):
        self.model = bentoml.models.get("bone_marrow_model:latest")
        self.model_impl = self.model.load_model()

    # Define Service API and IO schema
    @bentoml.api
    def predict(self, data: BoneMarrowInput) -> np.ndarray:
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
            # Note: MLflow model expects 'is_dead' instead of 'survival_status'
            'is_dead': data.survival_status
        }])

        # Use the model for prediction
        result = self.model_impl.predict(input_df)
        return result
