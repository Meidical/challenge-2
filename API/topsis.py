import numpy as np
from pandas import DataFrame, Series
from typing import Literal

class Topsis:
    @staticmethod
    def get_deviation_from_ideal_col_TOPSIS(data_criteria_encoded: DataFrame, stem_cell_source: Literal["bone marrow", "pheripheral blood"]):
        data_criteria_encoded = data_criteria_encoded.loc[:, ordered_columns].copy()

        impact = np.array([1, -1, 1, 1, 1, 1])

        weights = get_criteria_weights_AHP(stem_cell_source)

        criteria_matrix = data_criteria_encoded.to_numpy()
        
        matrix_normalized = normalize_matrix(criteria_matrix)
        matrix_weighted = weight_matrix(weights, matrix_normalized)
        
        pos_ideal, neg_ideal = ideal_best_worst(matrix_weighted, impact)
        pos_sep, neg_sep = deviation_from_ideal(matrix_weighted, pos_ideal, neg_ideal)
        deviations = similarity_to_PIS(pos_sep, neg_sep)

        deviation_from_ideal_col = Series(deviations, name="deviation_from_ideal").apply(lambda value: 1 - value)
        return deviation_from_ideal_col


def weight_matrix(weights, matrix_normalized):
    rows_normalized = len(matrix_normalized)
    cols_normalized = len(matrix_normalized[0])

    matrix_weighted = []
    for i in range(rows_normalized):
        matrix_weighted_rows = []
        for j in range(cols_normalized):
            matrix_weighted_rows.append(
                matrix_normalized[i][j] * weights[j])
        matrix_weighted.append(matrix_weighted_rows)

    return matrix_weighted


def ideal_best_worst(matrix_weighted, impact):
    weighted_column = len(matrix_weighted[0])
    positive_ideal = []
    negative_ideal = []

    for j in range(weighted_column):
        max_value = matrix_weighted[0][j]
        min_value = matrix_weighted[0][j]

        for i in range(len(matrix_weighted)):
            if matrix_weighted[i][j] > max_value:
                max_value = matrix_weighted[i][j]
            if matrix_weighted[i][j] < min_value:
                min_value = matrix_weighted[i][j]
        if impact[j] == 1:
            positive_ideal.append(max_value)
            negative_ideal.append(min_value)
        else:
            positive_ideal.append(min_value)
            negative_ideal.append(max_value)

    return positive_ideal, negative_ideal


def deviation_from_ideal(matrix_weighted, positive_ideal, negative_ideal):
    rows_weighted = len(matrix_weighted)
    positive_deviation = []
    negative_deviation = []

    for i in range(rows_weighted):
        pos_sep = 0
        neg_sep = 0

        for j in range(len(positive_ideal)):
            pos_sep += (matrix_weighted[i][j] - positive_ideal[j]) ** 2
            neg_sep += (matrix_weighted[i][j] - negative_ideal[j]) ** 2

        positive_deviation.append(pos_sep ** 0.5)
        negative_deviation.append(neg_sep ** 0.5)

    return positive_deviation, negative_deviation


def similarity_to_PIS(positive_separation, negative_separation):
    num_rows = len(positive_separation)
    relative_similarity = []

    for i in range(num_rows):
        pos_sep = positive_separation[i]
        neg_sep = negative_separation[i]
        similarity = neg_sep/(pos_sep + neg_sep)
        relative_similarity.append(similarity)

    return relative_similarity


def normalize_matrix(matrix):
    matrix_rows = len(matrix)
    matrix_columns = len(matrix[0])
    column_sums = [0] * matrix_columns

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

    return normalized_matrix


def get_criteria_weights_AHP(stem_cell_source: Literal["bone marrow", "pheripheral blood"]):
    if stem_cell_source == "bone marrow":
        relation_matrix = relations_BM
    else:
        relation_matrix = relations_PB

    data_AHP = DataFrame(
        relation_matrix, index=ordered_columns, columns=ordered_columns)

    weights = compute_weights(data_AHP)

    return weights


def compute_weights(data_AHP: DataFrame):
    column_sums = data_AHP.sum(axis=0)

    data_AHP_normalized = data_AHP / column_sums

    weights = data_AHP_normalized.mean(axis=1)

    return np.array(weights)


relations_BM = [
  #HLA | CMV | Age | Gender | ABO | Survival
    [1,     3,     2,   8,     7,     9], #HLA
    [1/3,   1,     1/3,   3,     3,     5], #CMV
    [1/2,   3,     1,     7,     5,     7], #Age
    [1/8,   1/3,   1/7,   1,     1/2,   3], #Gender
    [1/7,   1/3,   1/5,   2,     1,     3], #ABO
    [1/9,   1/5,   1/7,   1/3,   1/3,   1]  #Survival
]

relations_PB = [
  #HLA | CMV | Age | ABO | Gender | Survival
    [1,     3,     2,   7,     8,     9], #HLA
    [1/3,   1,     1/3,   3,     3,     5], #CMV
    [1/2,   3,     1,     5,     7,     7], #Age
    [1/7,   1/3,   1/5,   1/2,   1,     3], #ABO
    [1/8,   1/3,   1/7,   2,     1,     3], #Gender
    [1/9,   1/5,   1/7,   1/3,   1/3,   1]  #Survival
]

ordered_columns = [
    "HLA_match",
    "CMV_status",
    "donor_age_group",
    "gender_match",
    "ABO_match",
    "expected_survival_time"
]
