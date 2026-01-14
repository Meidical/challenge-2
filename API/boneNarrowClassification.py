from pydantic import BaseModel


class BoneMarrowClassificationInput(BaseModel):
    donor_age: float = 0
    donor_age_below_35: str = ""
    donor_ABO: str = ""
    donor_CMV: str = ""
    recipient_age: float = 0
    recipient_age_below_10: str = ""
    recipient_age_int: str = ""
    recipient_gender: str = ""
    recipient_body_mass: float = 0
    recipient_ABO: str = ""
    recipient_rh: str = ""
    recipient_CMV: str = ""
    disease: str = ""
    disease_group: str = ""
    gender_match: str = ""
    ABO_match: str = ""
    CMV_status: float = 0
    HLA_match: str = ""
    HLA_mismatch: str = ""
    antigen: float = 0
    allel: float = 0
    HLA_group_1: str = ""
    risk_group: str = ""
    stem_cell_source: str = ""


class BoneMarrowRegressionInput(BaseModel):
    donor_age: float = 0
    donor_age_below_35: str = ""
    donor_ABO: str = ""
    donor_CMV: str = ""
    recipient_age: float = 0
    recipient_age_below_10: str = ""
    recipient_age_int: str = ""
    recipient_gender: str = ""
    recipient_body_mass: float = 0
    recipient_ABO: str = ""
    recipient_rh: str = ""
    recipient_CMV: str = ""
    disease: str = ""
    disease_group: str = ""
    gender_match: str = ""
    ABO_match: str = ""
    CMV_status: float = 0
    HLA_match: str = ""
    HLA_mismatch: str = ""
    antigen: float = 0
    allel: float = 0
    HLA_group_1: str = ""
    risk_group: str = ""
    stem_cell_source: str = ""
    is_dead: float = 0
