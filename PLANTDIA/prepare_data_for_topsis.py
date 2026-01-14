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
import requests, json
# pip install bentoml
# pip install requests



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
def aggregate_data(id, donor_list_path, recipient_list_path, stem_cell_source):

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
    recipient_gender_conv = int(rows_recipients['recipient_gender'].values[0].lower() == "male")

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
        HLA_match, missing_allell, missing_antigen = get_HLA_match(donor_tissue, recipient_tissue)

        # CMV Serostatus
        donor_cmv = int(donor['donor_CMV'].lower() == "present")
        CMV_Serostatus = get_CMV_serostatus(donor_cmv, recipient_cmv)

        # Donor Age Group
        donor_age_group = donor["donor_age_group"]        
        #print("\nAge", donor["donor_age"], "Donor Age Group:", donor["donor_age_group"])

        # Gender Match
        donor_gender = int(donor['donor_gender'].lower() == "male")
        gender_match = get_gender_match(donor_gender, recipient_gender_conv)

        # ABO Match
        donor_ABO = convert_ABO(donor['donor_ABO'])
        ABO_match = get_ABO_match(donor_ABO, recipient_ABO)


        ##### Survival Status (is_dead) #####
        url = "http://localhost:3000/predict_classification" # Endpoint do BentoML para predição de sobrevivência

        payload_is_dead = {
            "donor_age": float(donor['donor_age']),
            "donor_age_below_35": "yes" if int(donor_age_group) == 2 else "no",
            "donor_ABO": donor['donor_ABO'],
            "donor_CMV": "present" if int(donor_cmv) == 1 else "absent",
            "recipient_age": float(rows_recipients['recipient_age'].values[0]),
            "recipient_age_below_10": "yes" if float(rows_recipients['recipient_age'].values[0]) < 10 else "no",
            "recipient_age_int": "0_5" if float(rows_recipients['recipient_age'].values[0]) <= 5 else "5_10" if float(rows_recipients['recipient_age'].values[0]) <= 10 else "10_20",
            "recipient_gender": str(rows_recipients['recipient_gender'].values[0]),
            "recipient_body_mass": float(rows_recipients['recipient_body_mass'].values[0]),
            "recipient_ABO": str(rows_recipients['recipient_ABO'].values[0]),
            "recipient_rh": str(rows_recipients['recipient_rh'].values[0]),
            "recipient_CMV": "present" if int(recipient_cmv) == 1 else "absent",
            "disease": str(rows_recipients['disease'].values[0]),
            "disease_group": "nonmalignant" if str(rows_recipients['disease'].values[0]) == "nonmalignant" else "malignant",
            "gender_match": "female_to_male" if int(gender_match) == 1 else "other",
            "ABO_match": "matched" if int(ABO_match) == 1 else "mismatched",
            "CMV_status": float(CMV_Serostatus),
            "HLA_match": f"{int(HLA_match)}/10",
            "HLA_mismatch": "mismatched" if int(HLA_match) <= 8 else "matched",
            "antigen": float(missing_antigen),
            "allel": float(missing_allell),
            "HLA_group_1": "matched",
            "risk_group": str(rows_recipients['risk_group'].values[0]),
            "stem_cell_source": "peripheral_blood" if stem_cell_source.upper() == "PBSC" else "bone_marrow"            
        } 

        # Debug: imprimir payload antes de enviar
        print(f"\n### Payload para {donor_name} ###")
        print(json.dumps(payload_is_dead, indent=2))
        
        response_is_dead = requests.post(url, json={"data": payload_is_dead})

        try:
            response_is_dead.raise_for_status()  # Verifica se houve erro HTTP
            is_dead = response_is_dead.json().get("survival_status")
        except requests.exceptions.RequestException as e:
            print(f"Erro na requisição: {e}")
            print(f"Detalhes do erro do servidor:")
            print(response_is_dead.text)
            is_dead = 0  # Default value
        except ValueError:
            print(f"Erro ao fazer parse do JSON: {response_is_dead.text}")
            is_dead = 0  # Default value
        print("Is Dead Response:")
        print(response_is_dead)



        ##### Expected Survival Time #####
        url = "http://localhost:3000/predict_regression" # Endpoint do BentoML para predição de tempo de sobrevivência
        
        payload_survival_time = payload_is_dead.copy()
        payload_survival_time['is_dead'] = is_dead      

        # Debug: imprimir payload antes de enviar
        print(f"\n### Payload para {donor_name} ###")
        print(json.dumps(payload_survival_time, indent=2))
        
        response = requests.post(url, json={"data": payload_survival_time})

        try:
            response.raise_for_status()  # Verifica se houve erro HTTP
            predicted_survival_time = response.json().get("predicted_survival_time_days")
        except requests.exceptions.RequestException as e:
            print(f"Erro na requisição: {e}")
            print(f"Detalhes do erro do servidor:")
            print(response.text)
            predicted_survival_time = None
        except ValueError:
            print(f"Erro ao fazer parse do JSON: {response.text}")
            predicted_survival_time = None        
        print(response)

      
      # Armazena os dados agregados      
        aggregated_data["Donors"].append({
            "recipient_ID": recipient_ID,
            "Recipient Name": rows_recipients['recipient_name'].values[0],
            "donor_ID": donor_ID,
            "Donor Name": donor_name,
            "HLA Match": HLA_match,
            "CMV Serostatus": CMV_Serostatus,
            "Donor Age Group": donor_age_group,
            "Gender Match": gender_match,
            "ABO Match": ABO_match,
            "Expected Survival Status": "not_survived" if is_dead == 1 else "survived",
            "Expected Survival Time": predicted_survival_time
        })

    aggregated_data = pd.DataFrame(aggregated_data["Donors"]) if aggregated_data["Donors"] else pd.DataFrame()
    if not aggregated_data.empty:
        aggregated_data = aggregated_data[[
            "recipient_ID",
            "Recipient Name",
            "donor_ID",
            "Donor Name",
            "HLA Match",
            "CMV Serostatus",
            "Donor Age Group",
            "Gender Match",
            "ABO Match",
            "Expected Survival Status",
            "Expected Survival Time"
        ]]

    return aggregated_data


# Execução direta para testes
if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    donors_CSV_PATH = os.path.join(BASE_DIR, "..", "datasets", "raw", "donor_list_raw.csv")      
    recipient_CSV_PATH = os.path.join(BASE_DIR, "..", "datasets", "raw", "recipient_waiting_list_raw.csv")      

    test_id = 'IR001'
    stem_cell_source = 'PBSC'

    aggregated_data = aggregate_data(test_id, donors_CSV_PATH, recipient_CSV_PATH, stem_cell_source)
    print(aggregated_data)
