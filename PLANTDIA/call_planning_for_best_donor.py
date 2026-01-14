# Esta classe é responsável por chamar a classe "prepare_data_for_TOPSIS" para preparar os dados necessários para enviar para a classe TOPSIS.

from prepare_data_for_topsis import aggregate_data
from TOPSIS.TOPSIS import calculate_topsis
import os
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
donors_CSV_PATH = os.path.join(BASE_DIR, "..", "datasets", "raw", "donor_list_raw.csv")      
recipient_CSV_PATH = os.path.join(BASE_DIR, "..", "datasets", "raw", "recipient_waiting_list_raw.csv")  


df_recipients = pd.read_csv(recipient_CSV_PATH, sep=';', skip_blank_lines=True, encoding='latin1')


def call_prepare_data_for_topsis(stem_cell_source):
    print("List of IDs from waiting recipients:")
    print(df_recipients['recipient_ID'].tolist())

    while True:      
        recipient_id = input("\nIndique um receptor: ").strip().upper()
        if recipient_id in df_recipients['recipient_ID'].tolist():
            print("\nVocê digitou:", recipient_id, "Name:", df_recipients.loc[df_recipients['recipient_ID'] == recipient_id, 'recipient_name'].values[0])
            aggregated_data = aggregate_data(recipient_id, donors_CSV_PATH, recipient_CSV_PATH, stem_cell_source)
            break

        else:
            print("\nVocê digitou:", recipient_id)
            print("ID de receptor inválido. Certifique-se de que o ID existe na lista de receptores.")
            continue

    return aggregated_data


# Função principal
def call_planning_for_best_donor():
    while True:
        stem_cell_source = input("Indique a fonte de células-tronco (PBSC ou BM): ").strip().upper()
        if stem_cell_source not in ['PBSC', 'BM']:
            print("Fonte de células-tronco inválida. Use 'PBSC' ou 'BM'.")
            continue
        else:            
            break
    dataframe = call_prepare_data_for_topsis(stem_cell_source)    
    df_TOPSIS = calculate_topsis(dataframe, stem_cell_source, verbose=True)
    

    ### Resultado final, que resulta na agregação dos dados dador/receptor com os resultados do TOPSIS ###
    data=dataframe.copy().drop(columns=['recipient_ID', 'donor_ID', 'Donor Name', 'Recipient Name'])

    result = pd.concat([df_TOPSIS, data], axis=1)
    result = result.sort_values(by='TOPSIS Score', ascending=False)
    result.rename(columns={'TOPSIS Score': 'TOPSIS Rank'}, inplace=True)  

    print("\n### Aggregated Data with TOPSIS Results ###")
    print(result)
    return result


# Execução direta para testes
if __name__ == "__main__":
    best_donor = call_planning_for_best_donor()


