import logging
import os

import mlflow
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


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


def set_mlflow_experiment():
    """Create a MLFlow experiment if it doesn't already exist.

    Set the active experiment using the experiment name or id"""

    # Assign the experiment name to a variable
    experiment_name = "bone-narrow-classification"
    logging.info("Checking for existing MLFlow experiment: %s",
                 experiment_name)
    # Check for existing experiment the experiment name
    if mlflow.get_experiment_by_name(experiment_name) is None:
        logging.info(msg="No existing experiment. Creating experiment...")

        # Create experiment and set it as the active experiment
        experiment_id = create_mlflow_experiment(experiment_name)
        mlflow.set_experiment(experiment_id=experiment_id)
        logging.info("Experiment created. Experiment id: %s", experiment_id)

        return experiment_id, experiment_name
    else:
        # Set tracking URI
        logging.info(msg="MLFlow tracking URI...")
        mlflow_tracking_uri = os.getenv(
            "MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
        logging.info("Tracking URI : %s ", mlflow_tracking_uri)
        mlflow.set_tracking_uri(mlflow_tracking_uri)

        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id

        return experiment_name, experiment_id


def create_mlflow_experiment(experiment_name):
    """
    Function to create an MLFlow experiment with a defined
    experiment name.
    """

    # Set MLFLow tracking URI
    mlflow_tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    logging.info("MLFlow Tracking URI URI set as: %s", mlflow_tracking_uri)

    logging.info("Creating experiment...")

    # create experiment using experiment_name
    if experiment_name is not None:
        penguins_experiment = mlflow.create_experiment(
            name=experiment_name
        )
    else:
        penguins_experiment = mlflow.create_experiement(
            name="penguins-experiment"
        )

    return penguins_experiment


def build_pipeline(smote=False):
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
        ("parser", FunctionTransformer(
            parse_hla_match, feature_names_out="one-to-one")),
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

    if smote:
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("smote", SMOTE(sampling_strategy="all", random_state=42)),
            ("model", RandomForestClassifier())
        ])
    else:
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier())
        ])

    return pipeline


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


def random_search(
    estimator,
    X_train,
    y_train,
    X_test,
    y_test,
    parent_run_name="RandomSearch",
    child_run_prefix="Trial",
    log_model_threshold=0.95
):

    param_distributions = {
        'classifier__n_estimators': [50, 100, 150, 200, 250, 300],
        'classifier__max_depth': [None, 5, 10, 15, 20, 25, 30],
        'classifier__min_samples_split': [2, 3, 4, 5, 8, 10],
        'classifier__min_samples_leaf': [1, 2, 3, 4, 5],
        'classifier__max_features': ['sqrt', 'log2', None, 0.5, 0.7],
        'classifier__bootstrap': [True, False]
    }

    param_distributions = {
        'classifier__n_estimators': [25, 50, 75, 100, 150],
        'classifier__max_depth': [None, 3, 5, 7],
        'classifier__max_features': ["sqrt", 5, 10],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2],
    }

    mlflow.autolog(disable=True)

    with mlflow.start_run(run_name=parent_run_name) as parent_run:
        random_search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_distributions,
            n_iter=20,
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1,
            random_state=42,
            return_train_score=True
        )
        random_search.fit(X_train, y_train)

        # Log best parameters and CV score
        mlflow.log_params(random_search.best_params_)
        mlflow.log_metric("best_cv_score", random_search.best_score_)

        # Identify top 5 parameter sets
        cv_results = random_search.cv_results_
        mean_scores = cv_results["mean_test_score"]
        top_5_indices = np.argsort(mean_scores)[-5:][::-1]

        logging.info(f"Retraining top 5 models on full training data...")

        results = []
        for rank, idx in enumerate(top_5_indices):
            candidate_params = cv_results["params"][idx]
            candidate_estimator = clone(estimator)
            candidate_estimator.set_params(**candidate_params)

            # Retrain on full training set
            fitted_model = candidate_estimator.fit(X_train, y_train)

            # Evaluate on test data
            y_pred = fitted_model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)
            test_f1 = f1_score(y_test, y_pred, average='weighted')

            results.append({
                "rank": rank + 1,
                "params": candidate_params,
                "cv_score": mean_scores[idx],
                "test_accuracy": test_accuracy,
                "test_f1_score": test_f1,
                "model": fitted_model
            })

            # Log trial info as nested run
            with mlflow.start_run(run_name=f"{child_run_prefix}_{rank+1}", nested=True):
                mlflow.log_params(candidate_params)
                mlflow.log_metric("mean_cv_score", mean_scores[idx])
                mlflow.log_metric("test_accuracy", test_accuracy)
                mlflow.log_metric("test_f1_score", test_f1)

            logging.info(
                f"Trial {rank + 1}: CV score = {mean_scores[idx]:.4f}, "
                f"Test accuracy = {test_accuracy:.4f}, F1 score = {test_f1:.4f}"
            )

        # Choose the best test-performing model
        best_result = max(results, key=lambda x: x["test_accuracy"])

        logging.info(
            f"Best model found: Rank {best_result['rank']} "
            f"with test accuracy = {best_result['test_accuracy']:.4f}, "
            f"F1 score = {best_result['test_f1_score']:.4f}"
        )

        # Log only the best model if above threshold
        if best_result["test_accuracy"] >= log_model_threshold:
            signature = infer_signature(
                X_train, best_result["model"].predict(X_train))
            mlflow.sklearn.log_model(
                sk_model=best_result["model"],
                name="model",
                signature=signature
            )
            logging.info(
                f"Logged best model to registry "
                f"(test_accuracy: {best_result['test_accuracy']:.4f})"
            )
        else:
            logging.info(
                f"No model met threshold ({log_model_threshold}); "
                f"best accuracy = {best_result['test_accuracy']:.4f}"
            )

        # Log summary metrics
        mlflow.log_metric("best_test_accuracy", best_result["test_accuracy"])
        mlflow.log_metric("best_test_f1_score", best_result["test_f1_score"])

    return random_search, best_result


print(os.getcwd())
dataset = load_dataset("./mlflow/data/bone_narrow_raw.xlsx")

X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = split_data(
    dataset)


print(f"Training features shape: {X_train.shape}")
print(f"Testing features shape: {X_test.shape}")
print(f"Training regression labels shape: {y_reg_train.shape}")
print(f"Testing regression labels shape: {y_reg_test.shape}")
print(f"Training classification labels shape: {y_clf_train.shape}")
print(f"Testing classification labels shape: {y_clf_test.shape}")

experiment_name, experiment_id = set_mlflow_experiment()
print(
    f"Experiment with ID: {experiment_id} and name: {experiment_name}")

pipeline = build_pipeline()
pipeline.fit(X_train, y_clf_train)

print(pipeline.predict(X_train))
random_search_result, best_model_info = random_search(
    estimator=pipeline,
    X_train=X_train,
    y_train=y_clf_train,
    X_test=X_test,
    y_test=y_clf_test,
    parent_run_name="BoneNarrow_Classification_RandomSearch",
    child_run_prefix="BoneNarrow_Trial",
    log_model_threshold=0.80
)
