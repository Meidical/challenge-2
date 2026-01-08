# Esta classe é responsável pela criação do dataset de entrada para o TOPSIS

    # Estrutura do DataFrame necessária para o funcionamento do TOPSIS:
    # |─────────────|───────────|─────────────────|───────────────────|─────────────────|────────────|─────────────────────────|
    # │ Donor_id    │ HLA Match │ CMV Serostatus  │  Donor Age Group  │ Gender Match    │ ABO Match  │ Expected Survival Time  │
    # ├─────────────┼───────────┼─────────────────┼───────────────────┼─────────────────┼────────────┼─────────────────────────┤
    # │ str         │ int       │ int             │ int               │ int             │ int        │ int                     │
    # └─────────────┴───────────┴─────────────────┴───────────────────┴─────────────────┴────────────┴─────────────────────────┘

import os
import sys

# Add project root to sys.path to allow imports from notebooks package
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from notebooks.dev.match_calc import *
from matrix_for_TOPSIS import dataframe

def aggregate_data(id):

    Donor_id = id

    rows =dataframe.loc[dataframe['Donor_id'] == Donor_id]
    if rows.empty:
        raise ValueError(f"Donor_id {Donor_id} não encontrado")

    # # HLA Match
    # donor_tissue_type = dataframe.loc[dataframe['Donor_id'] == Donor_id, 'donor_tissue_type'].values[0]
    # recipient_tissue_type = dataframe.loc[dataframe['Donor_id'] == Donor_id, 'recipient_tissue_type'].values[0]    
    # HLA_match, _, _ = get_HLA_match(donor_tissue_type, recipient_tissue_type)

    # # CMV Serostatus
    # donor_CMV = dataframe.loc[dataframe['Donor_id'] == Donor_id, 'donor_CMV_Serostatus'].values[0]
    # recipient_CMV = dataframe.loc[dataframe['Donor_id'] == Donor_id, 'recipient_CMV_Serostatus'].values[0]
    # CMV_Serostatus = get_CMV_serostatus(donor_CMV, recipient_CMV)

    # # Donor Age Group
    # donor_age_group = dataframe.loc[dataframe['Donor_id'] == Donor_id, 'donor_age_group'].values[0]

    # # Gender Match
    # donor_gender = dataframe.loc[dataframe['Donor_id'] == Donor_id, 'donor_gender'].values[0]
    # recipient_gender = dataframe.loc[dataframe['Donor_id'] == Donor_id, 'recipient_gender'].values[0]   
    # gender_match = get_gender_match(donor_gender, recipient_gender)   

    # # ABO Match
    # donor_ABO = dataframe.loc[dataframe['Donor_id'] == Donor_id, 'donor_ABO'].values[0]
    # recipient_ABO = dataframe.loc[dataframe['Donor_id'] == Donor_id, 'recipient_ABO'].values[0]
    # ABO_match = get_ABO_match(donor_ABO, recipient_ABO)

    # # Expected Survival Time
    # expected_time_survival = dataframe.loc[dataframe['Donor_id'] == Donor_id, 'expected_time_survival'].values[0]
    

    # return [Donor_id, HLA_match, CMV_Serostatus, donor_age_group, gender_match, ABO_match, expected_time_survival]  