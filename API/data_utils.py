import ast
from typing import Literal

import numpy as np
import pandas as pd
from pandas import DataFrame


class DataUtils:
    @staticmethod
    def read_df(df_path):
        df = pd.read_csv(df_path, sep=';', encoding="latin1")
        return df

    @staticmethod
    def aggregate_data(recipient_id: str, recipient_waiting_list: pd.DataFrame, donor_list: pd.DataFrame):

        recipient_row = recipient_waiting_list.loc[recipient_waiting_list["recipient_id"] == recipient_id]

        data_aggregated = join_row_to_data(recipient_row, donor_list)

        data_aggregated["donor_age_group"] = pd.cut(
            data_aggregated["donor_age"],
            bins=[18, 35, 50, 60],
            labels=["18-35", "35-50", "50-60"]
        )

        data_aggregated = encode_data(data_aggregated)

        data_aggregated = add_match_features(data_aggregated)

        data_aggregated = encode_data(data_aggregated, mode="decode")

        return data_aggregated

    @staticmethod
    def validate_value(value, default=None, expected_type=None):
        """Validate a value, returning default if NaN, None, or empty string for numeric types."""
        # Handle None
        if value is None:
            return default

        # Handle NaN for float values
        try:
            if np.isnan(value):
                return default
        except (TypeError, ValueError):
            pass  # Not a numeric type, continue

        # Handle empty string for numeric types
        if expected_type in (float, int) and value == '':
            return default

        # Convert to expected type if specified
        if expected_type == str:
            return str(value)
        elif expected_type == float:
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        elif expected_type == int:
            try:
                return int(value)
            except (ValueError, TypeError):
                return default

        return value


def join_row_to_data(row: DataFrame, data: DataFrame):
    data_joined = data.copy()

    for col in row.columns:
        data_joined[col] = row.iloc[0][col]

    return data_joined


ATTRIB_GROUPS = {
    "gender": ["donor_gender", "recipient_gender"],
    "blood_type": ["donor_ABO", "recipient_ABO"],
    "presence": ["donor_CMV", "recipient_CMV"],
    "match": ["ABO_match", "gender_match", "HLA_mismatch"],
    "donor_age_group": ["donor_age_group"]
}

VALUE_MAPPERS = {
    "gender": {"female": 0, "male": 1},
    "blood_type": {"O": 0, "A": 1, "B": 2, "AB": 3},
    "presence": {"absent": 0, "present": 1},
    "match": {"mismatched": 0, "matched": 1},
    "donor_age_group": {"18-35": 2, "35-50": 1, "50-60": 0}
}


def encode_data(data: DataFrame, mode: Literal["encode", "decode"] = "encode"):
    data_encoded = data.copy()

    for mapper_name, attribs in ATTRIB_GROUPS.items():
        mapping = VALUE_MAPPERS[mapper_name]

        if mode == "decode":
            mapping = {value: key for key, value in mapping.items()}

        for attrib in attribs:
            if attrib not in data_encoded.columns:
                continue

            data_encoded[attrib] = data_encoded[attrib].map(mapping)

    return data_encoded


def add_match_features(data_encoded: DataFrame):
    data_complete = data_encoded.copy()

    def compute_HLA(row):
        donor = ast.literal_eval(row["donor_tissue_type"])
        recipient = ast.literal_eval(row["recipient_tissue_type"])
        return get_HLA_match(donor, recipient)

    results = data_complete.apply(compute_HLA, axis=1)
    data_complete[["HLA_match", "allel", "antigen"]
                  ] = DataFrame(results.tolist())

    data_complete['CMV_serostatus'] = data_complete.apply(
        lambda row: get_CMV_serostatus(row['donor_CMV'], row['recipient_CMV']), axis=1)

    data_complete['gender_match'] = data_complete.apply(
        lambda row: get_gender_match(row['donor_gender'], row['recipient_gender']), axis=1)

    data_complete['ABO_match'] = data_complete.apply(
        lambda row: get_ABO_match(row['donor_ABO'], row['recipient_ABO']), axis=1)

    return data_complete


def get_HLA_match(donor_tissue_type, recipient_tissue_type):
    matches, missing_allell, missing_antigen = 0, 0, 0

    for donor_row, recipient_row in zip(donor_tissue_type, recipient_tissue_type):
        donor_row_sorted = sorted(donor_row)
        recipient_row_sorted = sorted(recipient_row)

        for donor_val, recipient_val in zip(donor_row_sorted, recipient_row_sorted):
            if donor_val == recipient_val:
                matches += 1
            else:
                if donor_val.split(':')[0] != recipient_val.split(':')[0]:
                    missing_allell += 1
                if donor_val.split(':')[1] != recipient_val.split(':')[1]:
                    missing_antigen += 1

    return matches, missing_allell, missing_antigen


def get_CMV_serostatus(donor_CMV, recipient_CMV):
    SEROSTATUS_MATRIX = np.array([[0, 1],
                                  [2, 3]])

    return SEROSTATUS_MATRIX[recipient_CMV, donor_CMV]


def get_gender_match(donor_gender, recipient_gender):
    if donor_gender == 0 and recipient_gender == 1:
        return 0

    return 1


def get_ABO_match(donor_ABO, recipient_ABO):
    BLOOD_COMPATIBILITY_MATRIX = np.array([[1, 0, 0, 0],
                                           [1, 1, 0, 0],
                                           [1, 0, 1, 0],
                                           [1, 1, 1, 1]])

    return BLOOD_COMPATIBILITY_MATRIX[recipient_ABO, donor_ABO]
