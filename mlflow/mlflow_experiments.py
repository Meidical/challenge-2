from xgboost import XGBClassifier, XGBRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import ElasticNet, LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

MODEL_REGISTRY_CLASSIFICATION = {
    "rf": RandomForestClassifier(random_state=42),
    "lr": LogisticRegression(random_state=42, max_iter=500),
    "et": ExtraTreesClassifier(random_state=42),
    "svm": SVC(random_state=42, probability=True),
    "xgb": XGBClassifier(random_state=42, n_jobs=1),
    "lgbm": LGBMClassifier(random_state=42)
}

MODEL_REGISTRY_REGRESSION = {
    "rf": RandomForestRegressor(random_state=42),
    "rr": Ridge(random_state=42),
    "et": ExtraTreesRegressor(random_state=42),
    "svm": SVR(),
    "xgb": XGBRegressor(random_state=42, n_jobs=1),
    "lgbm": LGBMRegressor(random_state=42),
    "enet": ElasticNet(random_state=42, max_iter=10_000)
}


PARAM_GRIDS = {
    "rf": {
        'classifier__n_estimators': [25, 50, 75, 100, 150],
        'classifier__max_depth': [None, 3, 5, 7],
        'classifier__max_features': ["sqrt", 5, 10],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2],
    },

    "rf_2": {
        "classifier__n_estimators": [50, 100, 150, 300],
        "classifier__max_depth": [None, 5, 10, 20],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 5],
        "classifier__max_features": ["sqrt", 0.5],
    },

    "et": {
        "classifier__n_estimators": [100, 300, 500],
        "classifier__max_depth": [None, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 5],
        "classifier__max_features": ["sqrt", 0.5],
    },

    "lr": {
        "classifier__C": [0.1, 1, 10],
        "classifier__penalty": ["l2"],
        "classifier__solver": ["lbfgs"]
    },

    "rr": {
        "classifier__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
        "classifier__fit_intercept": [True, False],
        "classifier__solver": ["auto"]
    },

    "xgb": {
        "classifier__max_depth": [2, 3, 4],
        "classifier__learning_rate": [0.03, 0.05, 0.1],
        "classifier__n_estimators": [50, 100, 150, 200, 300],
        "classifier__subsample": [0.7, 0.8, 1.0],
        "classifier__colsample_bytree": [0.7, 0.8, 1.0],
    },
    "svm": {
        'classifier__C': [0.1, 1, 10, 100, 1000],
        'classifier__gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        'classifier__kernel': ['rbf']
    },
    "lgbm": {
        "classifier__num_leaves": [7, 15, 31],
        "classifier__learning_rate": [0.03, 0.05, 0.1],
        "classifier__n_estimators": [50, 100, 150, 200],
        "classifier__subsample": [0.7, 0.8, 1.0],
        "classifier__colsample_bytree": [0.7, 0.8, 1.0],
    },
    "enet": {
        "classifier__alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
        "classifier__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    }
}

MODEL_FAMILIES = {
    "tree": ['rf', "et", "xgb", "lgbm"],
    "linear": ['rr', 'lr', 'enet']
}