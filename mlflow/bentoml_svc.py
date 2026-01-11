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
    CMV_status: int
    HLA_match: str
    HLA_mismatch: str
    antigen: int
    allel: int
    HLA_group_1: str
    risk_group: str
    stem_cell_source: str


# Define the BentoML Service
@bentoml.service(
    resources={"cpu": "2", "memory": "500MiB"},
    workers=1,
    traffic={"timeout": 20},
)
class MyService:
    # Load model in __init__ instead of using runner
    def __init__(self):
        self.model = bentoml.models.get("rf_gan_classification:latest")
        self.model_impl = self.model.load_model()

    # Define Service API and IO schema
    @bentoml.api
    def predict(self, data: BoneMarrowInput) -> dict:
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

        # The pipeline includes preprocessing, so pass raw data directly
        result = self.model_impl(input_df)

        # Return a dictionary with the prediction
        return {"prediction": int(result[0])}
