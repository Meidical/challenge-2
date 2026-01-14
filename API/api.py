# import flask module
import pandas as pd
from flask import Flask

RECIPIENT_CSV_PATH = "../datasets/raw/recipient_waiting_list_raw.csv"

# instance of flask application
app = Flask(__name__)

# home route that returns below text
# when root url is accessed


@app.route("/recipients")
def get_recipients():
    df_recipients = pd.read_csv(RECIPIENT_CSV_PATH, sep=';', encoding="latin1")
    return df_recipients.to_json(orient='records')


if __name__ == '__main__':
    app.run(debug=True, port=5001)


def read_df(df_path):
    df = pd.read_csv(df_path, sep=';', encoding="latin1")
    return df
