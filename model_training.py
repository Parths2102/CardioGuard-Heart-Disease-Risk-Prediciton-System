import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


DATA_URL = "https://raw.githubusercontent.com/plotly/datasets/master/heart.csv"
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    """
    Load the heart disease dataset.
    By default this pulls a public heart dataset compatible with UCI fields.
    """
    df = pd.read_csv(DATA_URL)
    return df


def build_preprocessor(df: pd.DataFrame):
    """Create a preprocessing pipeline for numeric and categorical features."""
    # Assume standard heart dataset columns
    target_col = "target"
    feature_cols = [c for c in df.columns if c != target_col]

    numeric_features = [
        "age",
        "trestbps",  # resting blood pressure
        "chol",
        "thalach",
        "oldpeak",
    ]

    categorical_features = [
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal",
    ]

    # Fallback: infer if some columns are missing
    numeric_features = [c for c in numeric_features if c in feature_cols]
    categorical_features = [c for c in categorical_features if c in feature_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor, numeric_features, categorical_features, target_col


def build_models():
    """Define the four base models."""
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced_subsample"
        ),
        "svm": SVC(kernel="rbf", probability=True, class_weight="balanced"),
    }

    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
        )

    return models


def evaluate_models(X_train, X_test, y_train, y_test, preprocessor, models):
    """Train and evaluate each model, returning metrics and best model."""
    results = {}
    best_name = None
    best_f1 = -1.0
    best_pipeline = None

    for name, model in models.items():
        clf = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = None
        if hasattr(clf.named_steps["model"], "predict_proba"):
            y_prob = clf.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
        }
        results[name] = metrics

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_name = name
            best_pipeline = clf

    return results, best_name, best_pipeline


def build_ensemble(preprocessor, models, X_train, y_train):
    """Optional: majority voting ensemble of the models that support predict_proba."""
    voting_estimators = []
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            voting_estimators.append((name, model))

    if not voting_estimators:
        return None

    ensemble = VotingClassifier(
        estimators=voting_estimators,
        voting="soft",
    )

    ensemble_clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", ensemble),
        ]
    )
    ensemble_clf.fit(X_train, y_train)
    return ensemble_clf


def main():
    print("Loading dataset...")
    df = load_dataset()
    preprocessor, num_features, cat_features, target_col = build_preprocessor(df)

    X = df[num_features + cat_features]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Building models...")
    models = build_models()

    print("Training and evaluating models...")
    results, best_name, best_pipeline = evaluate_models(
        X_train, X_test, y_train, y_test, preprocessor, models
    )

    print("\nModel performance:")
    for name, metrics in results.items():
        print(
            f"{name}: "
            f"accuracy={metrics['accuracy']:.3f}, "
            f"precision={metrics['precision']:.3f}, "
            f"recall={metrics['recall']:.3f}, "
            f"f1={metrics['f1']:.3f}"
        )

    if best_pipeline is None:
        raise RuntimeError("No best model identified.")

    best_model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(best_pipeline, best_model_path)
    print(f"\nBest model '{best_name}' saved to {best_model_path}")

    # Optional ensemble
    print("\nTraining ensemble (majority voting)...")
    ensemble_clf = build_ensemble(preprocessor, build_models(), X_train, y_train)
    if ensemble_clf is not None:
        ensemble_path = MODELS_DIR / "ensemble_model.joblib"
        joblib.dump(ensemble_clf, ensemble_path)
        print(f"Ensemble model saved to {ensemble_path}")
    else:
        print("Ensemble model could not be built (no probability-supporting estimators).")


if __name__ == "__main__":
    main()

