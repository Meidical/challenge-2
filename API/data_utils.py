import ast
from typing import Literal

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from match_utils import MatchUtils

class DataUtils:
    @staticmethod
    def read_df(df_path):
        df = pd.read_csv(df_path, sep=";", encoding="utf-8")
        return df

    @staticmethod
    def encode_data(data: DataFrame, mode: Literal["encode", "decode"] = "encode"):
        data_encoded = data.copy()

        ATTRIB_GROUPS = {
            "gender": ["donor_gender", "recipient_gender"],
            "blood_type": ["donor_ABO", "recipient_ABO"],
            "presence": ["donor_CMV", "recipient_CMV"],
            "match": ["ABO_match", "HLA_mismatch"],
            "gender_match": ["gender_match"],
            "donor_age_group": ["donor_age_group"],
            "yes_no": ["donor_age_below_35", "recipient_age_below_10", "tx_post_relapse"],
            "disease": ["disease"],
            "malignant": ["disease_group"],
            "level": ["risk_group"],
            "stem_cell_source": ["stem_cell_source"],
        }

        VALUE_MAPPERS = {
            "gender": {"female": 0, "male": 1},
            "blood_type": {"O": 0, "A": 1, "B": 2, "AB": 3},
            "presence": {"absent": 0, "present": 1},
            "match": {"mismatched": 0, "matched": 1},
            "gender_match": {"female_to_male": 0, "other": 1},
            "donor_age_group": {"18-35": 2, "35-50": 1, "50-60": 0},
            "yes_no": {"no": 0, "yes": 1},
            "disease": {"chronic": 1, "AML": 3, "ALL": 4, "nonmalignant": 0, "lymphoma": 2},
            "malignant": {"nonmalignant": 0, "malignant": 1},
            "level": {"low": 1, "high": 1},
            "stem_cell_source": {"pheripheral blood": 0, "bone marrow": 1},
        }

        for mapper_name, attribs in ATTRIB_GROUPS.items():
            mapping = VALUE_MAPPERS[mapper_name]

            if mode == "decode":
                mapping = {value: key for key, value in mapping.items()}

            for attrib in attribs:
                if attrib not in data_encoded.columns:
                    continue

                data_encoded[attrib] = data_encoded[attrib].map(mapping)

        return data_encoded

    @staticmethod
    def aggregate_data(recipient_id: str, recipient_waiting_list: DataFrame, donor_list: DataFrame, donor_id: str = None):

        recipient_row = recipient_waiting_list.loc[recipient_waiting_list["recipient_id"] == recipient_id]

        if donor_id:
            donor_rows = donor_list.loc[donor_list["donor_id"] == donor_id]
        else:
            donor_rows = donor_list

        data_aggregated = join_row_to_data(recipient_row, donor_rows)

        data_aggregated["donor_age_group"] = pd.cut(
            data_aggregated["donor_age"],
            bins=[18, 35, 50, 60],
            labels=["18-35", "35-50", "50-60"]
        )

        data_aggregated = DataUtils.encode_data(data_aggregated)

        data_aggregated = add_match_features(data_aggregated)

        data_aggregated = add_abstracted_features(data_aggregated)

        data_aggregated = DataUtils.encode_data(data_aggregated, mode="decode")

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


def add_match_features(data_encoded: DataFrame):
    data_added = data_encoded.copy()

    def compute_HLA(row):
        donor = ast.literal_eval(row["donor_tissue_type"])
        recipient = ast.literal_eval(row["recipient_tissue_type"])
        HLA_match, allel, antigen = MatchUtils.get_HLA_match(donor, recipient)

        return Series({"HLA_match": HLA_match, "allel": allel, "antigen": antigen})
    
    data_added = data_added.join(data_added.apply(compute_HLA, axis=1))

    data_added["CMV_serostatus"] = data_added.apply(lambda row: MatchUtils.get_CMV_status(row["donor_CMV"], row["recipient_CMV"]), axis=1)
    data_added["CMV_status"] = data_added.apply(lambda row: MatchUtils.get_CMV_status(
        row["donor_CMV"], row["recipient_CMV"]), axis=1)

    data_added["gender_match"] = data_added.apply(lambda row: MatchUtils.get_gender_match(
        row["donor_gender"], row["recipient_gender"]), axis=1)

    data_added["ABO_match"] = data_added.apply(lambda row: MatchUtils.get_ABO_match(
        row["donor_ABO"], row["recipient_ABO"]), axis=1)

    return data_added


def add_abstracted_features(data_encoded: DataFrame):
    data_added = data_encoded.copy()

    data_added["disease_group"] = (
        data_added["disease"] != "nonmalignant").astype("int")

    data_added["donor_age_below_35"] = (
        data_added["donor_age"] < 35).astype("int")

    data_added["recipient_age_below_10"] = (
        data_added["recipient_age"] < 10).astype("int")

    data_added["HLA_mismatch"] = (data_added["HLA_match"] > 8).astype("int")

    return data_added
