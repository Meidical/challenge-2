# Esta classe é responsável por chamar a classe "prepare_data_for_TOPSIS" para preparar os dados necessários para enviar para a classe TOPSIS.

from prepare_data_for_topsis import aggregate_data
from TOPSIS.TOPSIS import calculate_topsis
import os
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
donors_CSV_PATH = os.path.join(BASE_DIR, "..", "datasets", "raw", "donor_list_raw.csv")      
recipient_CSV_PATH = os.path.join(BASE_DIR, "..", "datasets", "raw", "recipient_waiting_list_raw.csv")  


df_recipients = pd.read_csv(recipient_CSV_PATH, sep=';', skip_blank_lines=True, encoding='latin1')


def call_prepare_data_for_topsis():
    print("List of IDs from waiting recipients:")
    print(df_recipients['recipient_ID'].tolist())

    while True:      
        recipient_id = input("\nIndique um receptor: ")
        if recipient_id in df_recipients['recipient_ID'].tolist():
            print("\nVocê digitou:", recipient_id, "Name:", df_recipients.loc[df_recipients['recipient_ID'] == recipient_id, 'recipient_name'].values[0])
            aggregated_data = aggregate_data(recipient_id, donors_CSV_PATH, recipient_CSV_PATH)
            break
                
        else:
            print("\nVocê digitou:", recipient_id)
            print("ID de receptor inválido. Certifique-se de que o ID existe na lista de receptores.")
            continue

    return aggregated_data


# Função principal
def call_planning_for_best_donor(stem_cell_source):
    dataframe = call_prepare_data_for_topsis()    
    result = calculate_topsis(dataframe, stem_cell_source, verbose=True)
    return result


# Execução direta para testes
if __name__ == "__main__":

    while True:
        stem_cell_source = input("Indique a fonte de células-tronco (PBSC ou BM): ").strip().upper()
        if stem_cell_source not in ['PBSC', 'BM']:
            print("Fonte de células-tronco inválida. Use 'PBSC' ou 'BM'.")
            continue
        else:
            best_donor = call_planning_for_best_donor(stem_cell_source)
            print("\n### Best Donor Result ###")
            print(best_donor)
            break

