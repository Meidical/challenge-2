from flasgger import Swagger
import os

import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from boneNarrowClassification import BoneMarrowClassificationInput
from bento_ml_client import BentoMLClient

# import logic module
from data_utils import DataUtils
from topsis import Topsis

bentoMl = BentoMLClient()

IS_DOCKER = os.getenv("IS_DOCKER", "false").lower() == "true"

if IS_DOCKER:
    # Paths dentro do container Docker
    GLOBAL_PATH = os.path.dirname(os.path.abspath(__file__))
    RECIPIENT_CSV_PATH = os.path.join(
        GLOBAL_PATH, "datasets", "dev", "recipient_waiting_list.csv")
    DONOR_CSV_PATH = os.path.join(
        GLOBAL_PATH, "datasets", "dev", "donor_list.csv")
    PAIR_CSV_PATH = os.path.join(
        GLOBAL_PATH, "datasets", "dev", "transplant_pair_list.csv")
else:
    # Paths localmente (relativo à pasta do API)
    GLOBAL_PATH = os.path.dirname(os.path.abspath(__file__))
    RECIPIENT_CSV_PATH = os.path.join(
        GLOBAL_PATH, "..", "datasets", "dev", "recipient_waiting_list.csv")
    DONOR_CSV_PATH = os.path.join(
        GLOBAL_PATH, "..", "datasets", "dev", "donor_list.csv")
    PAIR_CSV_PATH = os.path.join(
        GLOBAL_PATH, "..", "datasets", "dev", "transplant_pair_list.csv")

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
swagger = Swagger(app, template_file=os.path.join(BASE_DIR, 'openapi.yaml'))


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

    recipients_dataset = DataUtils.read_df(RECIPIENT_CSV_PATH)
    donors_dataset = DataUtils.read_df(DONOR_CSV_PATH)

    data_aggregated = DataUtils.aggregate_data(
        recipient_id,
        recipients_dataset,
        donors_dataset
    )

    BoneMarrowClassificationInput_list = []
    for _, row in data_aggregated.iterrows():
        try:
            input_data = row_to_classification_input(row)
            BoneMarrowClassificationInput_list.append(input_data.model_dump())
        except Exception as e:
            print(f"Error processing row: {e}")
            continue

    is_dead = bentoMl.predict_full_dataframe_classification(
        {"dataset": BoneMarrowClassificationInput_list})

    for item, is_dead_value in zip(BoneMarrowClassificationInput_list, is_dead.get('probabilities_dead', [])):
        item['is_dead'] = is_dead_value

    expected_survival_time = bentoMl.predict_full_dataframe_regression(
        {"dataset": BoneMarrowClassificationInput_list})

    data_aggregated['expected_survival_time'] = pd.Series(expected_survival_time.get(
        'predictions', []))

    deviation_from_ideal_col = Topsis.get_deviation_from_ideal_col_TOPSIS(
        DataUtils.encode_data(data_aggregated),
        stem_cell_source
    )

    donors_dataset['deviation_from_ideal'] = deviation_from_ideal_col.values
    donors_dataset[["HLA_match", "CMV_status", "gender_match", "ABO_match", "donor_age_group"]] = data_aggregated[
        ["HLA_match", "CMV_status", "gender_match", "ABO_match", "donor_age_group"]]

    return jsonify(donors_dataset.sort_values(by='deviation_from_ideal').to_dict(orient='records'))


@app.route("/transplant-pairs", methods=['GET'])
def get_pairs():
    df_pairs = DataUtils.read_df(PAIR_CSV_PATH)

    df_pairs_with_transplant = df_pairs[df_pairs["predicted_relapse"].notna()].to_json(
        orient='records')
    df_pairs_without_transplant = df_pairs[df_pairs["predicted_relapse"].isna(
    )].to_json(orient='records')

    return jsonify({"data": {
        "with_transplant": df_pairs_with_transplant,
        "without_transplant": df_pairs_without_transplant,
    }})


@app.route("/transplant-pairs", methods=['POST'])
def create_pair():
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
    return jsonify({"pair_id": pair_id})


@app.route("/transplant-pairs/<pair_id>", methods=['POST'])
def fill_pair_transplant_info(pair_id: str):
    data = request.get_json()

    CD34_per_kg = data["CD34_x1e6_per_kg"]
    CD3_per_kg = data["CD3_x1e8_per_kg"]
    tx_post_relapse = data["tx_post_relapse"]
    df_pairs = DataUtils.read_df(PAIR_CSV_PATH)

    # 1) Get the *single* row as a Series
    pair_df = df_pairs.loc[df_pairs["pair_id"] == pair_id]
    if pair_df.empty:
        return jsonify({"error": "pair_id not found"}), 404

    pair_row = pair_df.iloc[0]  # Series, like in your donor-matches loop

    # 2) Build the classification input from that row
    bm_input = row_to_classification_input(pair_row)
    bm_dict = bm_input.model_dump()  # plain Python dict

    # 3) Add transplant-specific fields
    bm_dict["stem_cell_source"] = stem_cell_source.replace(" ", "_")
    bm_dict["CD34_x1e6_per_kg"] = CD34_per_kg
    bm_dict["CD3_x1e8_per_kg"] = CD3_per_kg
    bm_dict["tx_post_relapse"] = tx_post_relapse

    if CD34_per_kg in [0, None] or pd.isna(CD34_per_kg):
        bm_dict["CD3_to_CD34_ratio"] = None
    else:
        bm_dict["CD3_to_CD34_ratio"] = CD3_per_kg / CD34_per_kg

    predicted_relapse = bentoMl.predict_relapse({
        "data": bm_dict})
    return jsonify(predicted_relapse)


def row_to_classification_input(row) -> BoneMarrowClassificationInput:
    """Convert a DataFrame row to BoneMarrowClassificationInput with validation."""

    return BoneMarrowClassificationInput(
        donor_age=DataUtils.validate_value(
            row.get('donor_age'), 30.0, float),
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
        gender_match=DataUtils.validate_value(
            row.get('gender_match'), 'other', str),
        ABO_match=DataUtils.validate_value(
            row.get('ABO_match'), 'matched', str),
        CMV_status=DataUtils.validate_value(
            row.get('CMV_status'), 3.0, float),
        HLA_match=DataUtils.validate_value(
            row.get('HLA_match'), '10/10', str),
        HLA_mismatch=DataUtils.validate_value(
            row.get('HLA_mismatch'), 'matched', str),
        antigen=DataUtils.validate_value(row.get('antigen'), 0.0, float),
        allel=DataUtils.validate_value(row.get('allel'), 0.0, float),
        HLA_group_1=DataUtils.validate_value(
            row.get('HLA_group_1'), 'matched', str),
        risk_group=DataUtils.validate_value(
            row.get('risk_group'), 'low', str),
        stem_cell_source=DataUtils.validate_value(
            row.get('stem_cell_source'), 'peripheral_blood', str)
    )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
