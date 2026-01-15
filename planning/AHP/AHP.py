# AHP (Analytic Hierarchy Process)


# Importing necessary packages
import pandas as pd

# Comparações par-a-par (escala de Saaty)
# Intensity of importance | Description
# ------------------------|---------------------------------------
# 9                       | Extreme importance
# 7                       | Very strong or demonstrated importance
# 5                       | Strong importance
# 3                       | Moderate importance
# 1                       | Equal importance
# 2,4,6,8                 | Intermediate values



# Regra clássica: CR < 0.10 → aceitável
# Regra exigente (saúde): CR < 0.05 → muito bom
# CR ≈ 0.01 → quase perfeito


# Identificar os critérios
# Donor_id │ HLA Match │ CMV Serostatus │ Donor Age Group │ Gender Match │ ABO Match │ Expected Survival Time

    # # HLA Match vs outros
    # ('HLA Match', 'Donor Age Group'): 1.5,             # Ligeiramente superior, porque já estamos a filtrar o HLA >= 7
    # ('HLA Match', 'CMV Serostatus'): 3,
    # ('HLA Match', 'ABO Match'): 7,
    # ('HLA Match', 'Gender Match'): 9,
    # ('HLA Match', 'Expected Survival Time'): 9,        # HLA >> Survival_Time (Survival_Time só desempate)

    # # Donor Age Group vs outros
    # ('Donor Age Group', 'CMV Serostatus'): 3,
    # ('Donor Age Group', 'ABO Match'): 5,
    # ('Donor Age Group', 'Gender Match'): 7,
    # ('Donor Age Group', 'Expected Survival Time'): 7,   # Idade >> Survival_Time

    # # CMV Serostatus vs outros
    # ('CMV Serostatus', 'ABO Match'): 3,
    # ('CMV Serostatus', 'Gender Match'): 3,
    # ('CMV Serostatus', 'Expected Survival Time'): 5,    # CMV > Survival_Time

    # # ABO vs outros
    # ('ABO Match', 'Gender Match'): 1,
    # ('ABO Match', 'Expected Survival Time'): 3,         # ABO > Survival_Time

    # # Gender Match vs outros
    # ('Gender Match', 'Expected Survival Time'): 3       # Gender Match > Survival

    ###########################################################################################################
    # Se a transfusão for do sangue, o Gender_match tem peso maior que o ABO_match, trocando os valores acima #
    ###########################################################################################################
    


# Matriz de comparação par-a-par dos critérios
#                              | HLA Match | CMV Serostatus | Donor Age Group | Gender Match | ABO Match | Expected Survival Time |
# | -------------------------- | --------- | -------------- | --------------- | ------------ | --------- | ---------------------- |
# | HLA Match                  |     1     |       3        |       1.5       |     9        |     7     |       9                |
# | CMV Serostatus             |    1/3    |       1        |       1/3       |     3        |     3     |       5                |
# | Donor Age Group            |    2/3    |       3        |        1        |     7        |     5     |       7                |
# | Gender Match               |    1/9    |      1/9       |       1/7       |     1        |     1     |       3                |
# | ABO Match                  |    1/7    |      1/7       |       1/5       |     1        |     1     |       3                |
# | Expected Survival Time     |    1/9    |      1/5       |       1/7       |    1/3       |    1/3    |       1                |


# Encontrar o índice de prioridade para cada um dos critérios
criteria = [
    "HLA Match",
    "CMV Serostatus",
    "Donor Age Group",
    "Gender Match",
    "ABO Match",
    "Expected Survival Time"
]

relations_BM = [
    [1,     3,     3/2,   9,     7,     9],
    [1/3,   1,     1/3,   3,     3,     5],
    [2/3,   3,     1,     7,     5,     7],
    [1/9,   1/9,   1/7,   1,     1,     3],
    [1/7,   1/7,   1/5,   1,     1,     3],
    [1/9,   1/5,   1/7,   1/3,   1/3,   1]
]


relations_PBSC = [
    [1,     3,     3/2,   9,     7,     9],
    [1/3,   1,     1/3,   3,     3,     5],
    [2/3,   3,     1,     7,     5,     7],
    [1/9,   1/9,   1/7,   1,     1,     3],
    [1/9,   1/5,   1/7,   1/3,   1/3,   1], 
    [1/7,   1/7,   1/5,   1,     1,     3]    
]

ahp_df_BM = pd.DataFrame(relations_BM, index=criteria, columns=criteria)
ahp_df_PBSC = pd.DataFrame(relations_PBSC, index=criteria, columns=criteria)

# Função para definir os índices de prioridade
def calculate_priority_indices(ahp_matrix):
    # Normalizar a matriz
    # Somar cada coluna e guardar os resultados
    column_sums = ahp_matrix.sum(axis=0)
    # Dividir cada elemento da matriz pela soma da sua coluna
    normalized_matrix = ahp_matrix / column_sums

    # Calcular os índices de prioridade (média das linhas)
    priority_indices = normalized_matrix.mean(axis=1)

    return priority_indices


# Definir o rátio de consistência (CR)
def calculate_consistency_ratio(ahp_matrix, priority_indices):
    n = ahp_matrix.shape[0]
    print(f"\nNúmero de critérios (n): {n}")

    

# Calcular o vetor de produtos, através da multiplicação da matriz AHP pelos índices de prioridade
    weighted_sum = ahp_matrix.dot(priority_indices)
    print("\nVetor de Produtos:")
    print(weighted_sum) 

#                              | HLA Match | CMV Serostatus | Donor Age Group | Gender Match | ABO Match | Survival Time |
# | -------------------------- | --------- | -------------- | --------------- | ------------ | --------- | --------------|
# | HLA Match                  |     1     |       3        |       1.5       |     9        |     7     |       9       | -> 1*0.404063 + 3*0.144633 + 1.5*0.308705 + 9*0.052773 + 7*0.058589 + 9*0.031237
# | CMV Serostatus             |    1/3    |       1        |       1/3       |     3        |     3     |       5       | -> 1/3*0.404063 + 1*0.144633 + 1/3*0.308705 + 3*0.052773 + 3*0.058589 + 5*0.031237
# | Donor Age Group            |    2/3    |       3        |        1        |     7        |     5     |       7       | -> 2/3*0.404063 + 3*0.144633 + 1*0.308705 + 7*0.052773 + 5*0.058589 + 7*0.031237
# | Gender Match               |    1/9    |      1/9       |       1/7       |     1        |     1     |       3       | -> 1/9*0.404063 + 1/9*0.144633 + 1/7*0.308705 + 1*0.052773 + 1*0.058589 + 3*0.031237
# | ABO Match                  |    1/7    |      1/7       |       1/5       |     1        |     1     |       3       | -> 1/7*0.404063 + 1/7*0.144633 + 1/5*0.308705 + 1*0.052773 + 1*0.058589 + 3*0.031237
# | Expected Survival Time     |    1/9    |      1/5       |       1/7       |    1/3       |    1/3    |       1       | -> 1/9*0.404063 + 1/5*0.144633 + 1/7*0.308705 + 1/3*0.052773 + 1/3*0.058589 + 1*0.031237
# | Índices de prioridade      | 0.404063  |    0.144633    |     0.308705    |   0.052773   | 0.058589  |    0.031237   |  


    
    # Calcular λ_max, ou seja, a média dos quocientes entre o vetor de produtos e os índices de prioridade
    lambda_max = (weighted_sum / priority_indices).mean()

    # Vetor de Produtos:
    # HLA Match                 2.467236
    # CMV Serostatus            0.872495
    # Donor Age Group           1.892998
    # Gender Match              0.310141
    # ABO Match                 0.345200
    # Expected Survival Time    0.186281


    # Índice de Consistência (CI); faz a média dos quocientes entre o vetor de produtos e os índices de prioridade
    ci = (lambda_max - n) / (n - 1)



    # Índices de Random Consistency (RI) para matrizes de ordem 1 a 1, de acordo com Saaty
    # | n  | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   | 11   | 12   | 13   | 14   | 15   |
    # | -- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
    # | RI | 0.00 | 0.00 | 0.58 | 0.90 | 1.12 | 1.24 | 1.32 | 1.41 | 1.45 | 1.49 | 1.51 | 1.48 | 1.56 | 1.57 | 1.59 |

    ri = 1.24  # Usar RI para n = 6 como aproximação

    # Rácio de Consistência (CR)
    cr = ci / ri if ri != 0 else 0

    return cr   




#Funcão Principal a ser chamada para calcular os pesos AHP

def calculate_ahp_weights_BM(stem_cell_source):
    # #Return a callable that provides weights for an ordered list of criteria names.
    # Uses the pairwise matrix defined in this module to compute priority indices
    # (AHP weights). The returned function accepts a list of criterion names and
    # returns the weights in the same order.


    if stem_cell_source == 'BM':
         matrix = pd.DataFrame(relations_BM, index=criteria, columns=criteria)
    elif stem_cell_source == 'PBSC':
         matrix = pd.DataFrame(relations_PBSC, index=criteria, columns=criteria)   
    else:
        raise ValueError("Tipo de tecido inválido. Use 'BM' para medula óssea ou 'PBSC' para sangue.") 
    
    pri = calculate_priority_indices(matrix)

    def weights_for(order):
        return [float(pri[name]) for name in order]

    return weights_for


if __name__ == "__main__":
    # Show the matrix and calculated indices/consistency when running this file directly

    stem_cell_source = 'BM'
    stem_cell_source = 'PBSC'

    if stem_cell_source == 'BM':
        print(ahp_df_BM)
        print("\nÍndices de Prioridade dos Critérios:")
        indices = calculate_priority_indices(ahp_df_BM)
        print(indices)

        cr = calculate_consistency_ratio(ahp_df_BM, indices)
        print(f"\nRácio de Consistência (CR): {cr:.4f}")

        if cr < 0.1:
            print('The model is consistent')
        else:
            print('The model is not consistent')


    elif stem_cell_source == 'PBSC':
        print(ahp_df_PBSC)
        print("\nÍndices de Prioridade dos Critérios:")
        indices = calculate_priority_indices(ahp_df_PBSC)
        print(indices) 
   
        cr = calculate_consistency_ratio(ahp_df_PBSC, indices)
        print(f"\nRácio de Consistência (CR): {cr:.4f}")

        if cr < 0.1:
            print('The model is consistent')
        else:
            print('The model is not consistent')
