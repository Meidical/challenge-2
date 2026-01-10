# Esta classe é responsável pela criação do dataset de entrada para o TOPSIS

    # Estrutura do DataFrame necessária para o funcionamento do TOPSIS:
    # |─────────────|───────────|───────────|─────────────────|───────────────────|───────────────|────────────|──────────────────────────|-----------|--------------|
    # |recipient_ID │ donor_ID  │ HLA Match │ CMV Serostatus  │  Donor Age Group  │ Gender Match  │ ABO Match  │ Expected Survival Time   │Donor Name |Recipient Name|
    # ├─────────────┼───────────┼───────────┼─────────────────┼───────────────────┼───────────── ─┼────────────┤──────────────────────────┤-----------|--------------|        
    # |    str      │ str       │ int       │ int             │ int               │ int           │ int        │ int                      │str        |str           |
    # └─────────────┴───────────┴───────────┴─────────────────┴───────────────────┴───────────── ─┴────────────┘──────────────────────────┘-----------|--------------|

import os
import sys
import ast
import pandas as pd

# Add project root to sys.path to allow imports from notebooks package
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Lê o dataset de doadores e recetores
def load_dataset(list_path):
    df = pd.read_csv(list_path, sep=';', skip_blank_lines=True, encoding="latin1")
    return df

# Importa a classe responsável por calcular os matches
from notebooks.dev.match_calc import *


# Função Principal
def aggregate_data(id, donor_list_path, recipient_list_path):

    recipient_ID = id

    aggregated_data = {  # Esta variável irá armazenar os dados agregados para o recetor e os doadores
        "Donors": []
    }
    
    df_recipients = load_dataset(recipient_list_path)

    rows_recipients =df_recipients.loc[df_recipients['recipient_ID'] == recipient_ID.upper()]
    print("\n### Recipient Info ###")
    print("ID:", rows_recipients['recipient_ID'].values[0],"\n", "Name:", rows_recipients['recipient_name'].values[0],"\n", "ABO:", rows_recipients['recipient_ABO'].values[0],"\n", "CMV:", rows_recipients['recipient_CMV'].values[0],"\n", "Sex:", rows_recipients['recipient_gender'].values[0],"\n", "Tissue Type:", rows_recipients['tissue_type'].values[0],"\n")        
    if rows_recipients.empty:
        raise ValueError(f"recipient_ID {recipient_ID} não encontrado")

    df_donors = load_dataset(donor_list_path)

    # Em seguida transformamos a idade do doador em grupos de idade, para todos os doadores
    # Grupo 0: 51-60 | Grupo 1: 36-50 | Grupo 2: 18-35
    if (df_donors["donor_age"] < 18).any() or (df_donors["donor_age"] > 60).any():
        raise ValueError("Idades de doadores fora do intervalo permitido (18-60)")
    
    df_donors["donor_age_group"] = pd.cut(        
        df_donors["donor_age"],
        bins=[18, 35, 50, 60],
        labels=[2, 1, 0],
        right=False
    )


    print("### List of donors ###\n", df_donors)


    # Cálculos dos matches em um único ciclo sobre os doadores

    # Preparar dados do recetor usados em todos os cálculos
    recipient_tissue = ast.literal_eval(rows_recipients['tissue_type'].values[0])
    recipient_cmv = int(rows_recipients['recipient_CMV'].values[0].lower() == "present")
    recipient_gender_conv = int(rows_recipients['recipient_gender'].values[0].upper() == "M")

    def convert_ABO(valor):
        mapa = {"O": 0, "A": 1, "B": 2, "AB": 3}
        return mapa.get(valor.upper())

    recipient_ABO = convert_ABO(rows_recipients['recipient_ABO'].values[0])

    for donor in df_donors.iterrows():
        donor = donor[1]  # extrai a série do doador
        donor_ID = donor['donor_ID'] if 'donor_ID' in donor else None
        donor_name = donor['donor_name']

        # HLA Match
        donor_tissue = ast.literal_eval(donor['tissue_type'])
        HLA_match, _, _ = get_HLA_match(donor_tissue, recipient_tissue)

        # CMV Serostatus
        donor_cmv = int(donor['donor_CMV'].lower() == "present")
        CMV_Serostatus = get_CMV_serostatus(donor_cmv, recipient_cmv)

        # Donor Age Group
        donor_age_group = donor["donor_age_group"]        
        #print("\nAge", donor["donor_age"], "Donor Age Group:", donor["donor_age_group"])

        # Gender Match
        donor_gender = int(donor['donor_gender'].upper() == "M")
        gender_match = get_gender_match(donor_gender, recipient_gender_conv)

        # ABO Match
        donor_ABO = convert_ABO(donor['donor_ABO'])
        ABO_match = get_ABO_match(donor_ABO, recipient_ABO)

        # Expected Survival Time
        exp_survival_time = 500

        aggregated_data["Donors"].append({
            "Recipient ID": recipient_ID,
            "Recipient Name": rows_recipients['recipient_name'].values[0],
            "Donor ID": donor_ID,
            "Donor Name": donor_name,
            "HLA Match": HLA_match,
            "CMV Serostatus": CMV_Serostatus,
            "Donor Age Group": donor_age_group,
            "Gender Match": gender_match,
            "ABO Match": ABO_match,
            "Expected Survival Time": exp_survival_time
        })

    aggregated_data = pd.DataFrame(aggregated_data["Donors"]) if aggregated_data["Donors"] else pd.DataFrame()
    if not aggregated_data.empty:
        aggregated_data = aggregated_data[[
            "Recipient ID",
            "Recipient Name",
            "Donor ID",
            "Donor Name",
            "HLA Match",
            "CMV Serostatus",
            "Donor Age Group",
            "Gender Match",
            "ABO Match",
            "Expected Survival Time"
        ]]
    print("\n### Aggregated Data ###")
    print(aggregated_data)
    return aggregated_data


# Execução direta para testes
if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    donors_CSV_PATH = os.path.join(BASE_DIR, "..", "datasets", "raw", "donor_list_raw.csv")      
    recipient_CSV_PATH = os.path.join(BASE_DIR, "..", "datasets", "raw", "recipient_waiting_list_raw.csv")      

    test_id = 'IR001'

    aggregated_data = aggregate_data(test_id, donors_CSV_PATH, recipient_CSV_PATH)
    #print(aggregated_data)
