# Standard library imports
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
import urllib.error

# Third-party imports - Configuration
import hydra
from omegaconf import DictConfig
import optuna

# Third-party imports - Data manipulation
import numpy as np
import pandas as pd

# Third-party imports - Preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import (
    MinMaxScaler, 
    OneHotEncoder
)

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

# Third-party imports - Explainability
import shap
import matplotlib.pyplot as plt

# Local imports
from mlflow_experiments import (
    MODEL_REGISTRY_CLASSIFICATION,
    MODEL_REGISTRY_REGRESSION,
    PARAM_GRIDS,
    MODEL_FAMILIES
)
from pre_processor import PreProcessor

def check_mlflow_server(host="127.0.0.1", port=5000, timeout=2):
    """Check if MLflow server is running"""
    try:
        url = f"http://{host}:{port}/health"
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError):
        return False


def start_mlflow_server(host="127.0.0.1", port=5000):
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


def ensure_mlflow_running(host="127.0.0.1", port=5000):
    """Ensure MLflow server is running, start it if not"""
    if check_mlflow_server(host, port):
        print(f"MLflow server is already running at http://{host}:{port}")
        return None
    else:
        return start_mlflow_server(host, port)


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
        "stem_cell_source"
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


def build_pipeline(model, preprocessor, use_smote=False):
    steps = [("preprocessor", preprocessor)]

    if use_smote:
        steps.append(("smote", SMOTE(random_state=42)))

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


def optuna_objective(
    trial,
    pipeline,
    model_name,
    X_train,
    y_train,
    cv,
    task
):
    param_grid = PARAM_GRIDS[model_name]
    params = suggest_from_grid(trial, param_grid)

    pipeline.set_params(**params)

    scoring = (
        "f1_weighted"
        if task == "classification"
        else "neg_root_mean_squared_error"
    )

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=4,
    )

    return scores.mean()


def tune_model(
    pipeline,
    model_name,
    X_train,
    y_train,
    cv,
    task,
    tuning_method,
    n_trials=50
):
    if tuning_method == "grid":
        gs = GridSearchCV(
            pipeline,
            PARAM_GRIDS[model_name],
            scoring="f1_weighted" if task == "classification" else "neg_root_mean_squared_error",
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
                trial, pipeline, model_name, X_train, y_train, cv, task
            ),
            n_trials=n_trials
        )

        mlflow.log_metric("optuna_trials", n_trials)

        best_pipeline = pipeline.set_params(**study.best_params)
        best_pipeline.fit(X_train, y_train)

        return best_pipeline, study.best_value


def import_model(tag, task, model_uri):
    model = bentoml.mlflow.import_model(f'{tag}_{task}', model_uri, labels={
                                        "run_id": mlflow.active_run().info.run_id})
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


def get_model_family(model_name: str):
    for family, models in MODEL_FAMILIES.items():
        if model_name in models:
            return family
    return "unsupported"


def log_shap_explanations(
    pipeline,
    X_train,
    task,
    model_name,
    max_samples=1000
):
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["classifier"]

    model_family = get_model_family(model_name)
    if model_family == "unsupported":
        print("Unsupported model")
        return

    X_transformed = preprocessor.transform(X_train)

    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    feature_names = preprocessor.get_feature_names_out()

    # --- Subsample for speed ---
    if X_transformed.shape[0] > max_samples:
        idx = np.random.choice(X_transformed.shape[0], max_samples, replace=False)
        X_transformed = X_transformed[idx]

    # --- Create explainer + SHAP values ---
    if model_family == "tree":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)

        if task == "classification":
            # Binary classification handling
            if isinstance(shap_values, list):
                shap_to_plot = shap_values[1]
            else:
                # shape = (n_samples, n_features, n_classes)
                shap_to_plot = shap_values[:, :, 1]
        else:
            # Regression → single output
            shap_to_plot = shap_values

    else:  # linear models
        background = shap.sample(X_transformed, min(100, X_transformed.shape[0]))

        explainer = shap.LinearExplainer(
            model,
            background,
            feature_perturbation="interventional"
        )

        shap_to_plot = explainer.shap_values(X_transformed)

    # --- Plot + log ---
    with tempfile.TemporaryDirectory() as tmpdir:
        bar_path = os.path.join(tmpdir, f"shap_{task}_bar.png")
        beeswarm_path = os.path.join(tmpdir, f"shap_{task}_beeswarm.png")

        # Bar plot (global importance)
        shap.summary_plot(
            shap_to_plot,
            X_transformed,
            feature_names=feature_names,
            plot_type="bar",
            max_display=30,
            show=False
        )
        plt.savefig(bar_path, bbox_inches="tight")
        plt.close()

        # Beeswarm (distribution + direction)
        shap.summary_plot(
            shap_to_plot,
            X_transformed,
            feature_names=feature_names,
            plot_type="dot",
            max_display=30,
            show=False
        )
        plt.savefig(beeswarm_path, bbox_inches="tight")
        plt.close()

        mlflow.log_artifact(bar_path, artifact_path="shap")
        mlflow.log_artifact(beeswarm_path, artifact_path="shap")


def run_experiment(
    experiment,
    X_train,
    y_train,
    X_test,
    y_test,
    task,
    cfg
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
            "cv": n_splits,
        })

    else:
        model = MODEL_REGISTRY_REGRESSION[experiment["model"]]

        mlflow.set_tags({
            "model": experiment["model"],
            "tuned": experiment["tune"],
            "task": task,
            "cv": n_splits,
        })

    # Create preprocessor instance
    preprocessor = build_preprocessor(cfg)

    pipeline = build_pipeline(
        model=model,
        preprocessor=preprocessor,
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
        best_model, score = tune_model(
            pipeline,
            model_name=experiment["model"],
            X_train=X_tr,
            y_train=y_tr,
            cv=cv,
            task=task,
            tuning_method=experiment["tuning_method"]
        )

        mlflow.log_params({
            k: v for k, v in best_model.get_params().items()
            if isinstance(v, (str, int, float, bool))
        })

        mlflow.log_metric("best_cv_score", score)

    else:
        best_model = pipeline.fit(X_tr, y_tr)

    if experiment["explain"]:
        log_shap_explanations(
            pipeline=best_model,
            X_train=X_tr,
            task=task,
            model_name=experiment["model"]
        )


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
        log_regression_metrics("train", y_tr, y_pred_train)
        log_regression_metrics("cv", y_tr, y_pred_cv)
        log_regression_metrics("test", y_test, y_pred_test)
        mlflow.log_metric("test_r2", r2_score(y_test, y_pred_test))

        # Baseline
        baseline_pred = np.full_like(y_test, y_tr.mean())
        baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
        mlflow.log_metric("baseline_rmse_test", baseline_rmse)

    # Log the model with the fitted preprocessor
    signature = infer_signature(X_train, y_pred_train)
    return mlflow.sklearn.log_model(
        best_model,
        "model",
        signature=signature
    )


def log_regression_metrics(prefix, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    mlflow.log_metric(f"{prefix}_rmse", rmse)
    mlflow.log_metric(f"{prefix}_mae", mae)


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

    if task == "regression" and exp.get("use_clf"):
        parts.append("wclf")
        if exp.get("predict_proba"):
            parts.append("proba")

    if not exp["age_gap"]:
        parts.append("noagegap")

    if not exp["age_bin"]:
        parts.append("noagebin")


    return "_".join(parts)


def run_mlflow(df, cfg):
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = split_data(
        df)

    mlflow.set_experiment(f"bone-marrow-{cfg.task}")

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
        "tuning_method": cfg.tuning.method,
        "explain": cfg.shap,
        "age_bin": cfg.fe.age_gap,
        "age_gap": cfg.fe.age_bin
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

        if cfg.task == "regression" and cfg.regression.use_clf:
            experiment = mlflow.get_experiment_by_name(
                "bone-marrow-classification")

            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="tags.task = 'classification'",
                order_by=["metrics.f1_weighted_test DESC"],
                max_results=1
            )

            best_run_id = runs.iloc[0].run_id
            run_info = mlflow.get_run(best_run_id)

            model_output = run_info.outputs.model_outputs[0]
            model_id = model_output.model_id

            model_path = (
                f"mlflow-artifacts:/{experiment.experiment_id}/models/"
                f"{model_id}/artifacts"
            )

            best_clf = mlflow.sklearn.load_model(model_path)

            mlflow.set_tag("upstream_classifier_run", best_run_id)

            X_train = X_train.copy()
            X_test = X_test.copy()

            if cfg.regression.predict_proba:
                X_train["is_dead"] = best_clf.predict_proba(X_train)[:, 1]
            else:
                X_train["is_dead"] = best_clf.predict(X_train)

            X_test["is_dead"] = y_clf_test.values

            y_train = y_reg_train
            y_test = y_reg_test

        else:
            y_train = y_clf_train if cfg.task == "classification" else y_reg_train
            y_test = y_clf_test if cfg.task == "classification" else y_reg_test

        experiment_model = run_experiment(
            experiment=exp,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            task=cfg.task,
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
