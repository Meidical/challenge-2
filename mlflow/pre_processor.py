from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from imblearn.pipeline import Pipeline
import pandas as pd


class PreProcessor:
    def __init__(self):
        self.preprocessor = self.build_preprocessor()

    def build_preprocessor(self):
        bool_cols = [
            "donor_age_below_35",
            "donor_CMV",
            "recipient_age_below_10",
            "recipient_gender",
            "recipient_CMV",
            "disease_group",
            "gender_match",
            "ABO_match",
            "HLA_mismatch",
            "risk_group",
            "stem_cell_source"
        ]

        cat_cols = [
            "CMV_status",
            "disease",
            "HLA_group_1",
            "recipient_age_int",
        ]

        extra_cols = Pipeline([
            ("feature_engineering", FunctionTransformer(
                self.feature_engineering,
                feature_names_out=lambda transformer, input_features: [
                    "donor_age_bin", "recipient_age_bin", "age_gap"]
            )),
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", MinMaxScaler())
        ])

        hla_pipeline = Pipeline([
            ("parser", FunctionTransformer(
                self.parse_hla_match, feature_names_out="one-to-one")),
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", MinMaxScaler())
        ])

        bool_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent", add_indicator=True)),
            ("encoder", OneHotEncoder(drop="if_binary"))
        ])

        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent", add_indicator=True)),
            ("one_hot", OneHotEncoder())
        ])

        columns_to_drop = [
            "donor_ABO",
            "recipient_ABO",
            "recipient_rh",
            "recipient_gender",
            "recipient_age_int",
            "recipient_age_below_10",
            "donor_age_below_35",
            "recipient_CMV",
            "donor_CMV",
            "disease_group"
        ]

        remainder_pipeline = Pipeline([
            ("inputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", MinMaxScaler())
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("extra_cols", extra_cols, ["donor_age", "recipient_age"]),
                ("column_dropper", "drop", columns_to_drop),
                ("hla", hla_pipeline, ["HLA_match"]),
                ("bool", bool_pipeline, bool_cols),
                ("one_hot", cat_pipeline, cat_cols)
            ],
            remainder=remainder_pipeline
        )

        return preprocessor

    def feature_engineering(self, X):
        X = X.copy()

        # Age gap
        X["age_gap"] = (X["donor_age"] - X["recipient_age"]).abs()

        # Donor age bins
        X["donor_age_bin"] = pd.cut(
            X["donor_age"],
            bins=[0, 18, 40, 60, 100],
            labels=False
        )

        # Recipient age bins
        X["recipient_age_bin"] = pd.cut(
            X["recipient_age"],
            bins=[0, 2, 5, 7, 10, 18, 22],
            labels=False
        )

        return X[["donor_age_bin", "recipient_age_bin", "age_gap"]]

    def parse_hla_match(self, X):
        s = X.iloc[:, 0]
        return (
            s
            .astype(str)
            .str.split("/", expand=True)[0]
            .astype(float)
            .to_frame()
        )

    def transform(self, X):
        return self.preprocessor.transform(X)

    def fit_transform(self, X):
        return self.preprocessor.fit_transform(X)