import os

import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from boneNarrowClassification import BoneMarrowClassificationInput, BoneMarrowRegressionInput
from bento_ml_client import BentoMLClient

# import logic module
from data_utils import DataUtils
from topsis import Topsis

GLOBAL_PATH = os.path.dirname(os.path.abspath(__file__))
RECIPIENT_CSV_PATH = os.path.join(
    GLOBAL_PATH, "..", "datasets", "dev", "recipient_waiting_list.csv")
DONOR_CSV_PATH = os.path.join(
    GLOBAL_PATH, "..", "datasets", "dev", "donor_list.csv")
PAIR_CSV_PATH = os.path.join(
    GLOBAL_PATH, "..", "datasets", "dev", "transplant_pair_list.csv")

app = Flask(__name__)


@app.route("/recipients", methods=['GET'])
def get_recipients():
    df_recipients = DataUtils.read_df(RECIPIENT_CSV_PATH)
    return df_recipients.to_json(orient='records')


@app.route("/recipient/<recipient_id>/donor-matches", methods=['POST'])
def list_donor_matches(recipient_id: str):
    # Validate POST body
    data = request.get_json()

    if not data:
        return jsonify({"error": "No data provided"}), 400

    # Validate that the payload contains valid stem cell source values
    valid_sources = ["bone marrow", "peripheral blood"]

    if "stem_cell_source" not in data:
        return jsonify({"error": "Missing 'stem_cell_source' in request body"}), 400

    stem_cell_source = data["stem_cell_source"]

    if stem_cell_source not in valid_sources:
        return jsonify({
            "error": f"Invalid stem_cell_source. Must be one of: {valid_sources}"
        }), 400

    # Process the request with validated data
    # TODO: Implement donor matching logic
    print(f"Recipient ID: {recipient_id}")
    print(f"Stem cell source: {stem_cell_source}")

    data_aggregated = DataUtils.aggregate_data(
        recipient_id,
        DataUtils.read_df(RECIPIENT_CSV_PATH),
        DataUtils.read_df(DONOR_CSV_PATH)
    )

    # convet data_aggregated to [BoneMarrowClassificationInput] format
    BoneMarrowClassificationInput_list = []
    for _, row in data_aggregated.iterrows():
        try:
            input_data = row_to_classification_input(row)
            BoneMarrowClassificationInput_list.append(input_data.model_dump())
        except Exception as e:
            print(f"Error processing row: {e}")
            continue

    bentoMl = BentoMLClient()
    is_dead = bentoMl.predict_full_dataframe_classification(
        {"dataset": BoneMarrowClassificationInput_list})

    for item, is_dead_value in zip(BoneMarrowClassificationInput_list, is_dead.get('probabilities_dead', [])):
        item['is_dead'] = is_dead_value

    data_aggregated = bentoMl.predict_full_dataframe_regression(
        {"dataset": BoneMarrowClassificationInput_list})

    deviation_from_ideal_col = Topsis.get_deviation_from_ideal_col_TOPSIS(
        data_aggregated,
        stem_cell_source
    )
    return jsonify(deviation_from_ideal_col.to_dict())


@app.route("/transplant-pairs", methods=['GET'])
def get_pairs():
    df_pairs = DataUtils.read_df(PAIR_CSV_PATH)
    return df_pairs.to_json(orient='records')


@app.route("/transplant-pairs", methods=['POST'])
def create_transplant_pair():
    data = request.get_json()

    recipient_id = data["recipient_id"]
    donor_id = data["donor_id"]

    df_recipients = DataUtils.read_df(RECIPIENT_CSV_PATH)
    df_donors = DataUtils.read_df(DONOR_CSV_PATH)
    df_pairs = DataUtils.read_df(PAIR_CSV_PATH)

    pair_row = DataUtils.aggregate_data(
        recipient_id, df_recipients, df_donors, donor_id)

    prev_id = df_pairs["pair_id"].apply(lambda id: id[2:]).astype(int).max()
    if (np.isnan(prev_id)):
        prev_id = 0

    new_id = prev_id + 1
    pair_id = "IP" + str(new_id).zfill(3)

    pair_row["pair_id"] = pair_id

    df_recipients = df_recipients[df_recipients["recipient_id"]
                                  != recipient_id]
    DataUtils.write_df(RECIPIENT_CSV_PATH, df_recipients)

    df_pairs = pd.concat([df_pairs, pair_row], ignore_index=True)
    DataUtils.write_df(PAIR_CSV_PATH, df_pairs)

    # return df_pairs.to_json(orient='records')
    return jsonify({"msg": f"Row with id: {pair_id}, successfully added!"})


def row_to_classification_input(row) -> BoneMarrowClassificationInput:
    """Convert a DataFrame row to BoneMarrowClassificationInput with validation."""

    return BoneMarrowClassificationInput(
        donor_age=DataUtils.validate_value(row.get('donor_age'), 30.0, float),
        donor_age_below_35=DataUtils.validate_value(
            row.get('donor_age_below_35'), 'yes', str),
        donor_ABO=DataUtils.validate_value(row.get('donor_ABO'), 'A', str),
        donor_CMV=DataUtils.validate_value(
            row.get('donor_CMV'), 'present', str),
        recipient_age=DataUtils.validate_value(
            row.get('recipient_age'), 10.0, float),
        recipient_age_below_10=DataUtils.validate_value(
            row.get('recipient_age_below_10'), 'no', str),
        recipient_age_int=DataUtils.validate_value(
            row.get('recipient_age_int'), '10_20', str),
        recipient_gender=DataUtils.validate_value(
            row.get('recipient_gender'), 'male', str),
        recipient_body_mass=DataUtils.validate_value(
            row.get('recipient_body_mass'), 40.0, float),
        recipient_ABO=DataUtils.validate_value(
            row.get('recipient_ABO'), 'A', str),
        recipient_rh=DataUtils.validate_value(
            row.get('recipient_rh'), 'plus', str),
        recipient_CMV=DataUtils.validate_value(
            row.get('recipient_CMV'), 'present', str),
        disease=DataUtils.validate_value(row.get('disease'), 'ALL', str),
        disease_group=DataUtils.validate_value(
            row.get('disease_group'), 'malignant', str),
        # gender_match=DataUtils.validate_value(
        #    row.get('gender_match'), 'other', str),
        gender_match="other",
        ABO_match=DataUtils.validate_value(
            row.get('ABO_match'), 'matched', str),
        CMV_status=DataUtils.validate_value(row.get('CMV_status'), 3.0, float),
        HLA_match=DataUtils.validate_value(row.get('HLA_match'), '10/10', str),
        HLA_mismatch=DataUtils.validate_value(
            row.get('HLA_mismatch'), 'matched', str),
        antigen=DataUtils.validate_value(row.get('antigen'), 0.0, float),
        allel=DataUtils.validate_value(row.get('allel'), 0.0, float),
        HLA_group_1=DataUtils.validate_value(
            row.get('HLA_group_1'), 'matched', str),
        risk_group=DataUtils.validate_value(row.get('risk_group'), 'low', str),
        stem_cell_source=DataUtils.validate_value(
            row.get('stem_cell_source'), 'peripheral_blood', str)
    )


if __name__ == '__main__':
    app.run(debug=True, port=5001)
