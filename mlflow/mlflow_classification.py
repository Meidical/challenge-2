import logging
import os

import mlflow
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.calibration import cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from mlflow_experiments import EXPERIMENTS, PARAM_GRIDS, MODEL_REGISTRY
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import Metadata
from sdv.sampling import Condition



def load_dataset(file_path: str):
    df = pd.read_excel(file_path)
    return df


def split_data(df, test_size=0.15, random_state=42):
    """Create features and labels of train and test datasets"""

    targets = df[["survival_time", "survival_status"]].rename(
        columns={"survival_status": "is_dead"})
    X = df.loc[:, : "stem_cell_source"].copy()
    y_clf = targets["is_dead"]
    y_reg = targets["survival_time"]
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.15, random_state=42, stratify=y_clf)

    return X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test


def get_training_data(
    X_train,
    y_train,
    experiment
):
    
    if not experiment["gan"]:
        return X_train, y_train
    

    # Apply GAN
    gan_df = X_train.copy()
    gan_df["is_dead"] = y_clf_train.values


    gan_df = gan_df.dropna().reset_index(drop=True)

    metadata = Metadata.detect_from_dataframe(
        data=gan_df,
        table_name="bone_marrow_transplant"
    )

    batch_size = 10

    ctgan = CTGANSynthesizer(
        metadata,
        epochs=experiment["gan_epochs"],
        batch_size=batch_size,
        verbose=True,
        enable_gpu=True
    )

    ctgan.fit(gan_df)

    alive_condition = Condition(
        num_rows=experiment["gan_0"],
        column_values={'is_dead': 0}
    )

    dead_condition = Condition(
        num_rows=experiment["gan_1"],
        column_values={'is_dead': 1}
    )

    synthetic = ctgan.sample_from_conditions(conditions=[alive_condition, dead_condition])
    X_synth = synthetic.drop(columns=["is_dead"])
    y_synth = synthetic["is_dead"]

    X_aug = pd.concat([X_train, X_synth], ignore_index=True)
    y_aug = pd.concat([y_train, y_synth], ignore_index=True)

    return X_aug, y_aug


def build_preprocessor():
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
        ("feature_engineering", FunctionTransformer(feature_engineering, feature_names_out=lambda self,
         input_features: ["donor_age_bin", "recipient_age_bin", "age_gap"])),
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", MinMaxScaler())
    ])

    hla_pipeline = Pipeline([
        ("parser", FunctionTransformer(parse_hla_match, feature_names_out="one-to-one")),
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

    columns_to_drop = ["donor_ABO",
                       "recipient_ABO",
                       "recipient_rh",
                       "recipient_gender",
                       "recipient_age_int",
                       "recipient_age_below_10",
                       "donor_age_below_35",
                       "recipient_CMV",
                       "donor_CMV",
                       "disease_group"]

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

def feature_engineering(X):
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

    return X[["donor_age_bin",
              "recipient_age_bin",
              "age_gap"]]

def parse_hla_match(X):
    s = X.iloc[:, 0]
    return (
        s
        .astype(str)
        .str.split("/", expand=True)[0]
        .astype(float)
        .to_frame()
    )

def build_pipeline(model, use_smote=False):
    preprocessor = build_preprocessor()

    steps = [("preprocessor", preprocessor)]

    if use_smote:
        steps.append(("smote", SMOTE(random_state=42))) 

    steps.append(("classifier", model))     

    return Pipeline(steps)


def tune_model(
    pipeline,
    model_name,
    X_train,
    y_train,
    cv,
    scoring="f1_weighted",
):
    param_grid = PARAM_GRIDS[model_name]

    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=4,
        verbose=10,
        return_train_score=True,
    )

    gs.fit(X_train, y_train)

    return gs

def run_experiment(
    experiment,
    X_train,
    y_train,
    X_test,
    y_test
):
    model = MODEL_REGISTRY[experiment["model"]]

    pipeline = build_pipeline(
        model=model,
        use_smote=experiment["smote"]
    )

    with mlflow.start_run(run_name=experiment["name"]):

        mlflow.set_tags({
            "model": experiment["model"],
            "tuned": experiment["tune"],
            "smote": experiment["smote"],
            "gan": experiment["gan"],
            "experiment_type": "classification",
        })

        if experiment["gan"]:
            mlflow.log_metric("gan_samples_0", experiment["gan_0"])
            mlflow.log_metric("gan_samples_1", experiment["gan_1"])


        mlflow.log_params(experiment)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


        X_tr, y_tr = get_training_data(
            X_train=X_train,
            y_train=y_train,
            experiment = experiment
        )


        if experiment["tune"]:
            gs = tune_model(
                pipeline,
                model_name=experiment["model"],
                X_train=X_tr,
                y_train=y_tr,
                cv=cv
            )
            best_model = gs.best_estimator_

            mlflow.log_params(gs.best_params_)
            mlflow.log_metric("best_cv_score", gs.best_score_)
        else:
            best_model = pipeline.fit(X_tr, y_tr)

        y_pred_train = best_model.predict(X_tr)
        y_pred_cv = cross_val_predict(best_model, X_tr, y_tr, cv=cv, verbose=10, n_jobs=-1)
        y_pred_test = best_model.predict(X_test)

        # Metrics on train set
        mlflow.log_metric("accuracy_train", accuracy_score(y_tr, y_pred_train))
        mlflow.log_metric("f1_weighted_train", f1_score(y_tr, y_pred_train, average="weighted"))

        # Metrics on cv train set
        mlflow.log_metric("accuracy_cv_train", accuracy_score(y_tr, y_pred_cv))
        mlflow.log_metric("f1_weighted_cv_train", f1_score(y_tr, y_pred_cv, average="weighted"))

        # Metrics on test set
        mlflow.log_metric("accuracy_test", accuracy_score(y_test, y_pred_test))
        mlflow.log_metric("f1_weighted_test", f1_score(y_test, y_pred_test, average="weighted"))

    return best_model


print(os.getcwd())
dataset = load_dataset("./mlflow/data/bone_narrow_raw.xlsx")
X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = split_data(dataset)


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("bone-marrow-classification")

for exp in EXPERIMENTS:

    run_experiment(
        experiment=exp,
        X_train=X_train,
        y_train=y_clf_train,
        X_test=X_test,
        y_test=y_clf_test
    )
