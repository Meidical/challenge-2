from flask import Flask, request, jsonify


# import logic module
from data_utils import DataUtils


RECIPIENT_CSV_PATH = "../datasets/raw/recipient_waiting_list_raw.csv"
DONOR_CSV_PATH = "../datasets/raw/donor_list_raw.csv"

# instance of flask application
app = Flask(__name__)


@app.route("/recipients", methods=['GET'])
def get_recipients():
    df_recipients = DataUtils.read_df(RECIPIENT_CSV_PATH)
    return df_recipients.to_json(orient='records')


@app.route("/recipient/<recipient_id>/donor-matches", methods=['POST'])
def aggregate_endpoint(recipient_id: str):
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

    return jsonify(data_aggregated.to_dict(orient='records'))


if __name__ == '__main__':
    app.run(debug=True, port=5001)
