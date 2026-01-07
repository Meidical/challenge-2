from xgboost import XGBClassifier, XGBRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

EXPERIMENTS = {
    "classification": [
        # Base experiments
        {
            "name": "rf_base",
            "model": "rf",
            "tune": False,
            "smote": False,
            "gan": False
        },
        {
            "name": "rf_tuned",
            "model": "rf",
            "tune": True,
            "smote": False,
            "gan": False,
        },
        {
            "name": "rf_smote",
            "model": "rf",
            "tune": True,
            "smote": True,
            "gan": False,
        },
        {
            "name": "rf_gan",
            "model": "rf",
            "tune": True,
            "smote": False,
            "gan": True,
            "gan_0": 10,
            "gan_1": 20,
            "gan_epochs": 300
        },
        {
            "name": "lr_base",
            "model": "lr",
            "tune": False,
            "smote": False,
            "gan": False
        },
        {
            "name": "lr_tuned",
            "model": "lr",
            "tune": True,
            "smote": False,
            "gan": False
        },
        {
            "name": "lr_smote",
            "model": "lr",
            "tune": True,
            "smote": True,
            "gan": False
        },
        {
            "name": "et_base",
            "model": "et",
            "tune": False,
            "smote": False,
            "gan": False
        },
        {
            "name": "et_tuned",
            "model": "et",
            "tune": True,
            "smote": False,
            "gan": False
        },
        {
            "name": "et_smote",
            "model": "et",
            "tune": True,
            "smote": True,
            "gan": False
        },
        {
            "name": "svm_base",
            "model": "svm",
            "tune": False,
            "smote": False,
            "gan": False
        },
        {
            "name": "svm_tuned",
            "model": "svm",
            "tune": True,
            "smote": False,
            "gan": False
        },
        {
            "name": "svm_smote",
            "model": "svm",
            "tune": True,
            "smote": True,
            "gan": False
        },
        {
            "name": "xgb_base",
            "model": "xgb",
            "tune": False,
            "smote": False,
            "gan": False
        },
        {
            "name": "xgb_tuned",
            "model": "xgb",
            "tune": True,
            "smote": False,
            "gan": False
        },
        {
            "name": "xgb_smote",
            "model": "xgb",
            "tune": True,
            "smote": True,
            "gan": False
        },
        {
            "name": "lgbm_base",
            "model": "lgbm",
            "tune": False,
            "smote": False,
            "gan": False
        },
        {
            "name": "lgbm_tuned",
            "model": "lgbm",
            "tune": True,
            "smote": False,
            "gan": False
        },
        {
            "name": "lgbm_smote",
            "model": "lgbm",
            "tune": True,
            "smote": True,
            "gan": False
        },
        # Other experiments
        {
            "name": "lr_gan_500_3",
            "model": "lr",
            "tune": True,
            "smote": False,
            "gan": True,
            "gan_0": 50,
            "gan_1": 66,
            "gan_epochs": 300,
        }
    ],
    "regression": [
        {
            "name": "rf_base",
            "model": "rf",
            "tune": False,
            "smote": False,
            "gan": False
        }
    ]
}

MODEL_REGISTRY_CLASSIFICATION = {
    "rf": RandomForestClassifier(random_state=42),
    "lr": LogisticRegression(random_state=42, max_iter=500),
    "et": ExtraTreesClassifier(random_state=42),
    "svm": SVC(random_state=42, probability=True),
    "xgb": XGBClassifier(random_state=42, n_jobs=1),
    "lgbm": LGBMClassifier(random_state=42),
}

MODEL_REGISTRY_REGRESSION = {
    "rf": RandomForestRegressor(random_state=42),
    "lr": LinearRegression(),
    "et": ExtraTreesRegressor(random_state=42),
    "svm": SVR(),
    "xgb": XGBRegressor(random_state=42, n_jobs=1),
    "lgbm": LGBMRegressor(random_state=42),
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
    }
}
