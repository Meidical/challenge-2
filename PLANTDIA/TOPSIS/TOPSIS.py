# Multi-criteria decision making (MCDM) using Topsis Method

import pandas as pd
import numpy as np

# Funções auxiliares

# Normalização da matriz
def normalize_matrix(matrix, verbose=False):
    matrix_rows = len(matrix)
    matrix_columns = len(matrix[0])
    column_sums= [0] * matrix_columns

    for j in range(matrix_columns):
        for i in range(matrix_rows):
            column_sums[j] += matrix[i][j] ** 2
    column_sums = [value ** 0.5 for value in column_sums]

    normalized_matrix = []
    for i in range(matrix_rows):
        normalized_matrix_rows = []
        for j in range(matrix_columns):
            normalized_matrix_rows.append(matrix[i][j]/column_sums[j])
        normalized_matrix.append(normalized_matrix_rows)
    
    if verbose:
        print('\nNormalized matrix from original matrix: ')
        for i in normalized_matrix:
            for j in i:
                print(f'{j:.4f}',end=' ')
            print()
    return normalized_matrix

# Matriz ponderada
def weighted_matrix(criteria_weight, normalized_matrix, verbose=False):
    normalized_rows = len(normalized_matrix)
    normalized_columns = len(normalized_matrix[0])
    
    weighted_matrix = []
    for i in range(normalized_rows):
        weighted_matrix_rows = []
        for j in range(normalized_columns):
            weighted_matrix_rows.append(normalized_matrix [i][j]* criteria_weight[j])
        weighted_matrix.append(weighted_matrix_rows)
    
    if verbose:
        print('\nWeighted matrix from normalized matrix: ')
        for i in weighted_matrix:
            for j in i:
                print(f'{j:.4f}',end=' ')
            print()
    return weighted_matrix

# Soluções ideais positiva e negativa
def ideal_best_worst(weighted_matrix, criteria_preferences, verbose=False):
    weighted_column = len(weighted_matrix[0])
    positive_ideal= [] 
    negative_ideal = [] 

    for j in range(weighted_column):
        max_value = weighted_matrix[0][j]
        min_value = weighted_matrix[0][j]

        for i in range(len(weighted_matrix)):
            if weighted_matrix[i][j] > max_value:
                max_value = weighted_matrix [i][j]
            if weighted_matrix[i][j] < min_value:
                min_value = weighted_matrix [i][j]
        if criteria_preferences[j] == 1:  
            positive_ideal.append(max_value)
            negative_ideal.append(min_value)
        else:  
            positive_ideal.append(min_value)
            negative_ideal.append(max_value)
    
    if verbose:
        print('\nPositive ideal point for each column: ')
        for i in positive_ideal:
            print(f'{i:.4f}',end=' ')
        print()
        print('\nNegative ideal point for each column: ')
        for i in negative_ideal:
            print(f'{i:.4f}',end=' ')
        print()

    return positive_ideal, negative_ideal

# Cálculo das medidas de separação das soluções ideais
def separation_from_ideal_point(weighted_matrix, positive_ideal, negative_ideal, verbose=False):
    weighted_rows = len(weighted_matrix)
    positive_separation = []
    negative_separation = []

    for i in range(weighted_rows):
        pos_sep = 0
        neg_sep = 0
        for j in range(len(positive_ideal)):
            pos_sep += (weighted_matrix[i][j] - positive_ideal[j]) ** 2
            neg_sep += (weighted_matrix[i][j] - negative_ideal[j]) ** 2
        positive_separation.append(pos_sep ** 0.5)
        negative_separation.append(neg_sep ** 0.5)

    if verbose:
        print('\nPositive separation: ')
        for i in (positive_separation):
            print(f'{i:.4f}')
        print('\nNegative separation: ')
        for i in (negative_separation):
            print(f'{i:.4f}')
    return positive_separation,negative_separation

# Cálculo das similaridades de cada atributo em relação à solução ideal positiva
def similarities_to_PIS(positive_separation, negative_separation, verbose=False):
    num_rows = len(positive_separation)
    relative_similarity = []

    for i in range(num_rows):
        pos_sep = positive_separation[i]
        neg_sep = negative_separation[i]
        similarity = neg_sep/(pos_sep + neg_sep)
        relative_similarity.append(similarity)
    
    if verbose:
        print('\nOrder: ')
        for i in (relative_similarity):
            print(f'{i:.4f}')
    return relative_similarity


# Função principal
#def calculate_topsis(mydata, verbose=False):

def calculate_topsis(verbose=False):

    # Carregar dados temporários para testar
    from matrix_for_TOPSIS import mydata
    mydata

    # Estrutura do DataFrame necessária para o funcionamento do TOPSIS:
    # |─────────────|───────────|─────────────────|───────────────────|─────────────────|────────────|─────────────────────────|
    # │ Donor_id    │ HLA Match │ CMV Serostatus  │  Donor Age Group  │ Gender Match    │ ABO Match  │ Expected Survival Time  │
    # ├─────────────┼───────────┼─────────────────┼───────────────────┼─────────────────┼────────────┼─────────────────────────┤
    # │ str         │ int       │ int             │ int               │ int             │ int        │ int                     │
    # └─────────────┴───────────┴─────────────────┴───────────────────┴─────────────────┴────────────┴─────────────────────────┘


    # Atribuição das preferências dos critérios
    # 1 para maximização e -1 para minimização
    # Maximizar: HLA_match, donor_age_group, ABO_match, expected_time_survival, Gender_match
    # Minimizar: CMV_status
    criteria_preferences = np.array([1, -1, 1, 1, 1, 1])

    # Os pesos serão criados através do AHP?
    criteria_weight = np.array([0.4029, 0.1423, 0.3088, 0.0555, 0.0604, 0.0302])

    # Prepara matriz
    matrix = mydata.iloc[:, 1:].values.astype(int)
    
    # Chama as funções auxiliares
    norm_matrix = normalize_matrix(matrix, verbose=verbose)
    w_matrix = weighted_matrix(criteria_weight, norm_matrix, verbose=verbose)
    pos_ideal, neg_ideal = ideal_best_worst(w_matrix, criteria_preferences, verbose=verbose)
    pos_sep, neg_sep = separation_from_ideal_point(w_matrix, pos_ideal, neg_ideal, verbose=verbose)
    scores = similarities_to_PIS(pos_sep, neg_sep, verbose=verbose)
    
    # Cria resultado final
    first_column = mydata.iloc[:, 0]
    results_series = pd.Series(scores, name='TOPSIS Score')
    df_TOPSIS = pd.concat([first_column, results_series], axis=1)
    df_TOPSIS = df_TOPSIS.sort_values(by='TOPSIS Score', ascending=False)
    df_TOPSIS.rename(columns={'TOPSIS Score': 'TOPSIS Rank'}, inplace=True)
    print("\n=== TOPSIS Results ===")
    return df_TOPSIS


if __name__ == "__main__":
    resultado = calculate_topsis(verbose=False)
    print("\n=== TOPSIS Results ===")
    print(resultado)