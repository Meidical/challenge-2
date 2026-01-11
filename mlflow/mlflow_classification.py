import logging
import os
import tempfile
import uuid
import mlflow
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
import bentoml
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.calibration import cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, recall_score, r2_score
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, RepeatedKFold, RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import Metadata
from sdv.sampling import Condition
from mlflow_experiments import EXPERIMENTS, PARAM_GRIDS, MODEL_REGISTRY_CLASSIFICATION, MODEL_REGISTRY_REGRESSION
from mlflow.tracking import MlflowClient


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


def generate_gan(
    X_train,
    y_train,
    experiment,
    task
):

    if task == 'classification':
        if not experiment["gan"]:
            return X_train, y_train

        # Apply GAN
        gan_df = X_train.copy()
        gan_df["is_dead"] = y_train.values

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

        synthetic = ctgan.sample_from_conditions(
            conditions=[alive_condition, dead_condition])

        X_synth = synthetic.drop(columns=["is_dead"])
        y_synth = synthetic["is_dead"]

        X_aug = pd.concat([X_train, X_synth], ignore_index=True)
        y_aug = pd.concat([y_train, y_synth], ignore_index=True)

        mlflow.log_metric("gan_samples_0", experiment["gan_0"])
        mlflow.log_metric("gan_samples_1", experiment["gan_1"])
        mlflow.log_metric("gan_epochs", experiment["gan_epochs"])

        with tempfile.TemporaryDirectory() as tmpdir:
            train_with_gan = X_aug.copy()
            train_with_gan["is_dead"] = y_aug.values

            artifact_path = os.path.join(tmpdir, "train_with_gan.csv")
            metadata_path = os.path.join(tmpdir, "gan_metadata.json")

            train_with_gan.to_csv(artifact_path, index=False)
            mlflow.log_artifact(artifact_path)

            metadata.save_to_json(metadata_path)
            mlflow.log_artifact(metadata_path)

        return X_aug, y_aug

    else:
        return X_train, y_train


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
    task
):
    param_grid = PARAM_GRIDS[model_name]

    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_weighted" if task == "classification" else "neg_root_mean_squared_error",
        cv=cv,
        n_jobs=4,
        verbose=10,
        return_train_score=True,
    )

    gs.fit(X_train, y_train)

    return gs


def import_model(tag, task, model_uri):
    model = bentoml.mlflow.import_model(f'{tag}_{task}', model_uri)
    model_name = ":".join([model.tag.name, model.tag.version])
    return model_name


def load_model(model_name=None):
    if model_name is None:
        model_name = model_name
    bento_model = bentoml.mlflow.load_model(model_name)
    return bento_model


def predict(bento_model, testdata):
    prediction = bento_model.predict(testdata)
    return prediction


def run_experiment(
    experiment,
    X_train,
    y_train,
    X_test,
    y_test,
    task
):
    n_splits = 5

    if task == "classification":
        model = MODEL_REGISTRY_CLASSIFICATION[experiment["model"]]

        mlflow.set_tags({
            "model": experiment["model"],
            "tuned": experiment["tune"],
            "smote": experiment["smote"],
            "gan": experiment["gan"],
            "task": task,
            "cv": n_splits
        })

    else:
        model = MODEL_REGISTRY_REGRESSION[experiment["model"]]

        mlflow.set_tags({
            "model": experiment["model"],
            "tuned": experiment["tune"],
            "task": task,
            "cv": n_splits
        })

    pipeline = build_pipeline(
        model=model,
        use_smote=experiment["smote"]
    )

    # Log datasets
    train_df = X_train.copy()
    train_df["target"] = y_train.values
    train_dataset = mlflow.data.from_pandas(
        train_df,
        targets="target",
        name="train_dataset"
    )

    test_df = X_test.copy()
    test_df["target"] = y_test.values
    test_dataset = mlflow.data.from_pandas(
        test_df,
        targets="target",
        name="test_dataset"
    )

    mlflow.log_input(train_dataset, context="training")
    mlflow.log_input(test_dataset, context="testing")

    mlflow.log_params(experiment)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) if task == "classification" else KFold(
        n_splits=n_splits, shuffle=True, random_state=42)

    X_tr, y_tr = generate_gan(
        X_train=X_train,
        y_train=y_train,
        experiment=experiment,
        task=task
    )

    if experiment["tune"]:
        gs = tune_model(
            pipeline,
            model_name=experiment["model"],
            X_train=X_tr,
            y_train=y_tr,
            cv=cv,
            task=task
        )
        best_model = gs.best_estimator_

        mlflow.log_params(gs.best_params_)
        mlflow.log_metric("best_cv_score", gs.best_score_)

    else:
        best_model = pipeline.fit(X_tr, y_tr)

    y_pred_train = best_model.predict(X_tr)
    y_pred_cv = cross_val_predict(
        best_model, X_tr, y_tr, cv=cv, verbose=10, n_jobs=-1)
    y_pred_test = best_model.predict(X_test)

    if task == "classification":
        # Metrics on train set
        mlflow.log_metric("accuracy_train", accuracy_score(y_tr, y_pred_train))
        mlflow.log_metric("f1_weighted_train", f1_score(
            y_tr, y_pred_train, average="weighted"))

        # Metrics on cv train set
        mlflow.log_metric("accuracy_cv_train", accuracy_score(y_tr, y_pred_cv))
        mlflow.log_metric("f1_weighted_cv_train", f1_score(
            y_tr, y_pred_cv, average="weighted"))

        # Metrics on test set
        mlflow.log_metric("accuracy_test", accuracy_score(y_test, y_pred_test))
        mlflow.log_metric("f1_weighted_test", f1_score(
            y_test, y_pred_test, average="weighted"))
        mlflow.log_metric("recall_test", recall_score(y_test, y_pred_test))
        mlflow.log_metric("precision_test",
                          precision_score(y_test, y_pred_test))

    else:

        # Train
        log_regression_metrics("train", y_tr, y_pred_train)

        # CV
        log_regression_metrics("cv", y_tr, y_pred_cv)

        # Test
        log_regression_metrics("test", y_test, y_pred_test)

        # R² on test
        mlflow.log_metric("test_r2", r2_score(y_test, y_pred_test))

        # Baseline
        baseline_pred = np.full_like(y_test, y_tr.mean())
        baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
        mlflow.log_metric("baseline_rmse_test", baseline_rmse)

    # Log the model
    infer_signature(X_train, y_pred_train)
    return mlflow.sklearn.log_model(best_model, "model")


def log_regression_metrics(prefix, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    mlflow.log_metric(f"{prefix}_rmse", rmse)
    mlflow.log_metric(f"{prefix}_mae", mae)


def run_mlflow(task):
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = split_data(
        dataset)
    mlflow.set_experiment(f"bone-marrow-{task}")

    best_run_id = None

    for exp in EXPERIMENTS[task]:
        with mlflow.start_run(run_name=exp["name"]):
            if task == "regression":
                if exp["use_clf"]:

                    # Get best classification model
                    experiment = mlflow.get_experiment_by_name(
                        f"bone-marrow-classification")

                    runs = mlflow.search_runs(
                        experiment_ids=[experiment.experiment_id],
                        filter_string="tags.task = 'classification'",
                        order_by=["metrics.f1_weighted_test DESC"],
                        max_results=1
                    )

                    model_output = mlflow.get_run(
                        runs.iloc[0].run_id).outputs.model_outputs[0]
                    model_id = model_output.model_id
                    model_path = f"mlflow-artifacts:/{experiment.experiment_id}/models/{model_id}/artifacts"
                    best_clf = mlflow.sklearn.load_model(
                        model_path
                    )

                    mlflow.set_tag("upstream_classifier_run", best_run_id)

                    X_train_reg = X_train.copy()
                    X_test_reg = X_test.copy()

                    if exp["predict_proba"]:
                        X_train_reg["is_dead"] = best_clf.predict_proba(X_train)[
                            :, 1]
                    else:
                        X_train_reg["is_dead"] = best_clf.predict(X_train)

                    X_test_reg["is_dead"] = y_clf_test.values

                else:
                    X_train_reg = X_train
                    X_test_reg = X_test

                # Run regression experiment
                experiment_model = run_experiment(
                    experiment=exp,
                    X_train=X_train_reg,
                    y_train=y_reg_train,
                    X_test=X_test_reg,
                    y_test=y_reg_test,
                    task=task
                )

            else:

                experiment_model = run_experiment(
                    experiment=exp,
                    X_train=X_train,
                    y_train=y_clf_train,
                    X_test=X_test,
                    y_test=y_clf_test,
                    task=task,
                )

            import_model(exp["name"],
                         task,
                         experiment_model.model_uri)


print(os.getcwd())

dataset = load_dataset("./mlflow/data/bone_narrow_raw.xlsx")

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.end_run()
run_mlflow("classification")
run_mlflow("regression")

# b_model = load_model("regression_model")
