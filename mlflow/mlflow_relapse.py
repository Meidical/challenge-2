# Standard library imports
from sklearn.impute import SimpleImputer
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
import urllib.error

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

# Third-party imports - Configuration
import hydra
from omegaconf import DictConfig
import optuna

# Third-party imports - Data manipulation
import numpy as np
import pandas as pd

# Third-party imports - Machine Learning
from sklearn.calibration import cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split
)

# Third-party imports - Imbalanced learning
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Third-party imports - Synthetic data generation
from sdv.metadata import Metadata
from sdv.sampling import Condition
from sdv.single_table import CTGANSynthesizer

# Third-party imports - MLflow
import mlflow
from mlflow.models import infer_signature

# Third-party imports - BentoML
import bentoml

# Local imports
from mlflow_experiments import (
    MODEL_REGISTRY_CLASSIFICATION,
    PARAM_GRIDS
)


class MissingAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        X["donor_CMV_missing"] = (X["donor_CMV"] == "?").astype(int)
        X["recipient_CMV_missing"] = (X["recipient_CMV"] == "?").astype(int)

        X["ABO_match_missing"] = (X["ABO_match"] == "?").astype(int)
        return X


class CellDosageImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.model_ = None
        self.imputer_ = None

    def fit(self, X, y=None):
        X = X.copy()

        # Fill predictors with median
        self.imputer_ = SimpleImputer(strategy="median")
        predictors = X[["CD34_x1e6_per_kg", "recipient_body_mass"]]
        self.imputer_.fit(predictors)

        # Fit model on rows where CD3 is present
        train_data = X.dropna(subset=["CD3_x1e8_per_kg"])
        self.model_ = LinearRegression()
        self.model_.fit(
            self.imputer_.transform(
                train_data[["CD34_x1e6_per_kg", "recipient_body_mass"]]),
            train_data["CD3_x1e8_per_kg"]
        )
        return self

    def transform(self, X):
        X = X.copy()

        # Fill predictors
        X[["CD34_x1e6_per_kg", "recipient_body_mass"]] = self.imputer_.transform(
            X[["CD34_x1e6_per_kg", "recipient_body_mass"]]
        )

        # Predict missing CD3_x1e8_per_kg
        missing = X["CD3_x1e8_per_kg"].isna()
        if missing.any():
            X.loc[missing, "CD3_x1e8_per_kg"] = self.model_.predict(
                self.imputer_.transform(
                    X.loc[missing, ["CD34_x1e6_per_kg", "recipient_body_mass"]]
                )
            )

        # Compute CD3/CD34 ratio
        ratio_missing = X["CD3_to_CD34_ratio"].isna()
        X.loc[ratio_missing, "CD3_to_CD34_ratio"] = X.loc[ratio_missing,
                                                          "CD3_x1e8_per_kg"] / X.loc[ratio_missing, "CD34_x1e6_per_kg"]

        return X


def check_mlflow_server(host="127.0.0.1", port=5001, timeout=2):
    """Check if MLflow server is running"""
    try:
        url = f"http://{host}:{port}/health"
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError):
        return False


def start_mlflow_server(host="127.0.0.1", port=5001):
    """Start MLflow tracking server in background"""
    print(f"Starting MLflow server at {host}:{port}...")

    # Start server in background
    process = subprocess.Popen(
        [
            "mlflow", "server",
            "--host", host,
            "--port", str(port)
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Wait for server to be ready
    max_attempts = 30
    for attempt in range(max_attempts):
        if check_mlflow_server(host, port):
            print(
                f"MLflow server started successfully at http://{host}:{port}")
            return process
        time.sleep(1)
        print(
            f"Waiting for MLflow server to start... ({attempt + 1}/{max_attempts})")

    # If server didn't start, kill the process and raise error
    process.kill()
    raise RuntimeError("Failed to start MLflow server")


def ensure_mlflow_running(host="127.0.0.1", port=5001):
    """Ensure MLflow server is running, start it if not"""
    if check_mlflow_server(host, port):
        print(f"MLflow server is already running at http://{host}:{port}")
        return None
    else:
        return start_mlflow_server(host, port)


def import_model(tag, task, model_uri):
    model = bentoml.mlflow.import_model(f'{tag}_relapse', model_uri)
    model_name = ":".join([model.tag.name, model.tag.version])
    return model_name


def load_dataset(file_path: str):
    df = pd.read_excel(file_path)
    return df


def split_data(df, test_size=0.2):
    X = df.loc[:, : "CD3_to_CD34_ratio"].copy()
    y = df["relapse"]
    y = y.map({'yes': 1, 'no': 0})
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test


def impute_body_mass(data: pd.DataFrame):
    data = data.copy()

    data["recipient_age_group"] = pd.cut(data["recipient_age"], bins=[
                                         0., 1., 5., 10., 15., 20., np.inf], labels=["<1", "1-5", "5-10", "10-15", "15-20", "20+"])

    group_median = data.groupby(["recipient_gender", "recipient_age_group"])[
        "recipient_body_mass"].median()
    global_median = data["recipient_body_mass"].median()

    def fill_mass(row):
        if pd.isna(row["recipient_body_mass"]):
            return group_median.get(
                (row["recipient_gender"], row["recipient_age_group"]),
                global_median
            )
        return row["recipient_body_mass"]

    data["recipient_body_mass"] = data.apply(fill_mass, axis=1)

    data = data.drop(columns="recipient_age_group")
    return data


def encode_booleans(data: pd.DataFrame):
    zero_mapper = {"absent", "no", "female", "mismatched",
                   "female_to_male", "low", "nonmalignant", "peripheral_blood", "?"}
    one_mapper = {"present", "yes", "male", "matched",
                  "other", "high", "malignant", "bone_marrow"}

    def map_values(value):
        if value in zero_mapper:
            return 0
        if value in one_mapper:
            return 1

        return value

    return data.map(map_values)


def feature_engineering(X, cfg):
    X = X.copy()
    features = []

    if cfg.fe.age_gap:
        X["age_gap"] = (X["donor_age"] - X["recipient_age"]).abs()
        features.append("age_gap")

    if cfg.fe.age_bin:
        X["donor_age_bin"] = pd.cut(
            X["donor_age"],
            bins=[0, 18, 40, 60, 100],
            labels=False
        )
        features.append("donor_age_bin")

        X["recipient_age_bin"] = pd.cut(
            X["recipient_age"],
            bins=[0, 2, 5, 7, 10, 18, 22],
            labels=False
        )
        features.append("recipient_age_bin")

    if not features:
        X["_dummy"] = 0
        features.append("_dummy")

    return X[features]


def make_feature_names_out(cfg):
    def _feature_names_out(*args, **kwargs):
        names = []

        if cfg.fe.age_gap:
            names.append("age_gap")

        if cfg.fe.age_bin:
            names.extend(["donor_age_bin", "recipient_age_bin"])

        if not names:
            names.extend(["_dummy"])

        return np.array(names, dtype=object)

    return _feature_names_out


def parse_hla_match(X):
    s = X.iloc[:, 0]
    return (
        s
        .astype(str)
        .str.split("/", expand=True)[0]
        .astype(float)
        .to_frame()
    )


def build_preprocessor(cfg):
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
        "stem_cell_source",
        "tx_post_relapse"
    ]

    cat_cols = [
        "CMV_status",
        "disease",
        "HLA_group_1",
        "recipient_age_int",
    ]

    extra_cols = Pipeline([
        (
            "feature_engineering",
            FunctionTransformer(
                lambda X: feature_engineering(X, cfg),
                feature_names_out=make_feature_names_out(cfg)
            )
        ),

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

    columns_to_drop = [
        col for col, drop in cfg.fe.drop_columns.items()
        if drop
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


def build_pipeline(model, cfg, use_smote=False):
    preprocessor = build_preprocessor(cfg)

    steps = [
        ("cell_dosage_imputer", CellDosageImputer()),
        ("preprocessor", preprocessor),
    ]

    if use_smote:
        steps.append(
            ("smote", SMOTE(random_state=42, sampling_strategy="minority")))

    steps.append(("classifier", model))

    return ImbPipeline(steps)


def suggest_from_grid(trial, param_grid):
    params = {}

    for param_name, values in param_grid.items():
        # numeric range → suggest_int / suggest_float
        if all(isinstance(v, int) for v in values):
            params[param_name] = trial.suggest_int(
                param_name, min(values), max(values)
            )

        elif all(isinstance(v, float) for v in values):
            params[param_name] = trial.suggest_float(
                param_name, min(values), max(values), log=True
            )

        # categorical
        else:
            params[param_name] = trial.suggest_categorical(
                param_name, values
            )

    return params


def generate_gan(
    X_train,
    y_train,
    experiment
):

    if not experiment["gan"]:
        return X_train, y_train

    # Apply GAN
    gan_df = X_train.copy()
    gan_df["relapse"] = y_train.values

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
        column_values={'relapse': 0}
    )

    dead_condition = Condition(
        num_rows=experiment["gan_1"],
        column_values={'relapse': 1}
    )

    synthetic = ctgan.sample_from_conditions(
        conditions=[alive_condition, dead_condition])

    X_synth = synthetic.drop(columns=["relapse"])
    y_synth = synthetic["relapse"]

    X_aug = pd.concat([X_train, X_synth], ignore_index=True)
    y_aug = pd.concat([y_train, y_synth], ignore_index=True)

    mlflow.log_metric("gan_samples_0", experiment["gan_0"])
    mlflow.log_metric("gan_samples_1", experiment["gan_1"])
    mlflow.log_metric("gan_epochs", experiment["gan_epochs"])

    with tempfile.TemporaryDirectory() as tmpdir:
        train_with_gan = X_aug.copy()
        train_with_gan["relapse"] = y_aug.values

        artifact_path = os.path.join(tmpdir, "train_with_gan.csv")
        metadata_path = os.path.join(tmpdir, "gan_metadata.json")

        train_with_gan.to_csv(artifact_path, index=False)
        mlflow.log_artifact(artifact_path)

        metadata.save_to_json(metadata_path)
        mlflow.log_artifact(metadata_path)

    return X_aug, y_aug


def optuna_objective(
    trial,
    pipeline,
    model_name,
    X_train,
    y_train,
    cv
):
    param_grid = PARAM_GRIDS[model_name]
    params = suggest_from_grid(trial, param_grid)

    pipeline.set_params(**params)

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1,
    )

    return scores.mean()


def tune_model(
    pipeline,
    model_name,
    X_train,
    y_train,
    cv,
    tuning_method,
    n_trials=50
):
    if tuning_method == "grid":
        gs = GridSearchCV(
            pipeline,
            PARAM_GRIDS[model_name],
            scoring="f1_weighted",
            cv=cv,
            n_jobs=4,
            verbose=10,
        )
        gs.fit(X_train, y_train)
        return gs.best_estimator_, gs.best_score_

    elif tuning_method == "optuna":
        study = optuna.create_study(direction="maximize")

        study.optimize(
            lambda trial: optuna_objective(
                trial, pipeline, model_name, X_train, y_train, cv
            ),
            n_trials=n_trials
        )

        mlflow.log_metric("optuna_trials", n_trials)

        best_pipeline = pipeline.set_params(**study.best_params)
        best_pipeline.fit(X_train, y_train)

        return best_pipeline, study.best_value


def run_experiment(
    experiment,
    X_train,
    y_train,
    X_test,
    y_test,
    cfg
):
    n_splits = 5

    model = MODEL_REGISTRY_CLASSIFICATION[experiment["model"]]

    mlflow.set_tags({
        "model": experiment["model"],
        "tuned": experiment["tune"],
        "smote": experiment["smote"],
        "gan": experiment["gan"],
    })

    pipeline = build_pipeline(
        model=model,
        use_smote=experiment["smote"],
        cfg=cfg
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

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    X_tr, y_tr = generate_gan(
        X_train=X_train,
        y_train=y_train,
        experiment=experiment
    )
    # print(X_train.isna().sum())
    # assert X_tr.isna().sum().sum() == 0, "NaNs still exist!"

    if experiment["tune"]:
        best_model, score = tune_model(
            pipeline,
            model_name=experiment["model"],
            X_train=X_tr,
            y_train=y_tr,
            cv=cv,
            tuning_method=experiment["tuning_method"]
        )

        mlflow.log_params({
            k: v for k, v in best_model.get_params().items()
            if isinstance(v, (str, int, float, bool))
        })
        mlflow.log_metric("best_cv_score", score)

    else:
        best_model = pipeline.fit(X_tr, y_tr)

    y_pred_train = best_model.predict(X_tr)
    y_pred_cv = cross_val_predict(
        best_model, X_tr, y_tr, cv=cv, verbose=10, n_jobs=-1)
    y_pred_test = best_model.predict(X_test)

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

    # Log the model with the fitted preprocessor
    signature = infer_signature(X_train, y_pred_train)
    return mlflow.sklearn.log_model(
        best_model,
        "model",
        signature=signature
    )


def build_run_name(exp, task):
    parts = [exp["model"]]

    if exp.get("tune"):
        parts.append("tuned")
    else:
        parts.append("base")

    if exp.get("tuning_method") == "grid":
        parts.append("grid")
    else:
        parts.append("optuna")

    if exp.get("smote"):
        parts.append("smote")

    if exp.get("gan"):
        parts.append("gan")

    return "_".join(parts)


def run_mlflow(df, cfg):
    X_train, X_test, y_train, y_test = split_data(df)

    mlflow.set_experiment(f"bone-marrow-relapse")

    exp = {
        "model": cfg.model,
        "tune": cfg.tune,
        "smote": cfg.smote,
        "gan": cfg.gan,
        "gan_epochs": cfg.gan_params.epochs,
        "gan_0": cfg.gan_params.gan_0,
        "gan_1": cfg.gan_params.gan_1,
        "use_clf": cfg.regression.use_clf,
        "predict_proba": cfg.regression.predict_proba,
        "tuning_method": cfg.tuning.method
    }

    run_name = build_run_name(exp, cfg.task)

    with mlflow.start_run(run_name=run_name):

        # Log full Hydra config
        mlflow.log_params({
            "model": cfg.model,
            "task": cfg.task,
            "tune": cfg.tune,
            "smote": cfg.smote,
            "gan": cfg.gan,
            "cv_folds": cfg.cv.folds,
            "tuning_method": cfg.tuning.method
        })

        experiment_model = run_experiment(
            experiment=exp,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            cfg=cfg
        )

        import_model(run_name, cfg.task, experiment_model.model_uri)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(os.getcwd())

    mlflow_run = {
        "host": "127.0.0.1",
        "port": 5000
    }

    # Ensure MLflow server is running
    ensure_mlflow_running(host=mlflow_run["host"], port=mlflow_run["port"])

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(
        f"http://{mlflow_run['host']}:{mlflow_run['port']}")

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "data", "bone_narrow_raw.xlsx")

    df = load_dataset(dataset_path)

    run_mlflow(df, cfg)


if __name__ == "__main__":
    main()
