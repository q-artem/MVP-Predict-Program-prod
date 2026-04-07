import os
import sys
import joblib
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

MODEL_DIR = "models"
RANDOM_STATE = 42
TOP_N = 5


# =========================
# УТИЛИТЫ
# =========================
def get_paths(model_type):
    if model_type == 0:
        return {
            "name": "binary",
            "model": f"{MODEL_DIR}/binary_model.pkl",
            "columns": f"{MODEL_DIR}/binary_columns.pkl",
            "objective": "binary",
            "max_depth": 6,
        }
    elif model_type == 1:
        return {
            "name": "quality",
            "model": f"{MODEL_DIR}/quality_model.pkl",
            "columns": f"{MODEL_DIR}/quality_columns.pkl",
            "objective": "multiclass",
            "max_depth": 4,
        }
    else:
        raise ValueError("model_type must be 0 (binary) or 1 (quality)")


def prepare_features(df, saved_columns=None):
    X = df.copy()

    cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns
    for col in cat_cols:
        X[col] = X[col].astype("category")

    if saved_columns is not None:
        for col in saved_columns:
            if col not in X.columns:
                X[col] = pd.NA

        extra_cols = [c for c in X.columns if c not in saved_columns]
        if extra_cols:
            X = X.drop(columns=extra_cols)

        X = X[saved_columns]

    return X


def save_model(model, columns, paths):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, paths["model"])
    joblib.dump(columns, paths["columns"])


def load_model(paths):
    if not os.path.exists(paths["model"]):
        raise FileNotFoundError(f"Model not found: {paths['model']}")
    if not os.path.exists(paths["columns"]):
        raise FileNotFoundError(f"Columns not found: {paths['columns']}")

    model = joblib.load(paths["model"])
    columns = joblib.load(paths["columns"])
    return model, columns


# =========================
# TRAIN (с нуля)
# =========================
def train(model_type, train_fraction, input_csv, out_csv, target_col):
    paths = get_paths(model_type)

    df = pd.read_csv(input_csv, engine="pyarrow")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X = prepare_features(X)

    if train_fraction < 1.0:
        X_train, X_hold, y_train, y_hold = train_test_split(
            X, y,
            train_size=train_fraction,
            random_state=RANDOM_STATE,
            stratify=y
        )

        holdout_df = X_hold.copy()
        holdout_df[target_col] = y_hold
        holdout_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

        print(f"Saved holdout: {out_csv}")

    else:
        X_train = X
        y_train = y

    model = LGBMClassifier(
        n_estimators=100,
        max_depth=paths["max_depth"],
        learning_rate=0.1,
        objective=paths["objective"],
        random_state=RANDOM_STATE,
        verbose=-1,
        force_col_wise=True
    )

    model.fit(X_train, y_train)

    save_model(model, X.columns.tolist(), paths)

    print(f"{paths['name']} model trained and saved.")


# =========================
# RETRAIN (100%)
# =========================
def retrain(model_type, input_csv, target_col):
    paths = get_paths(model_type)

    df = pd.read_csv(input_csv, engine="pyarrow")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X = prepare_features(X)

    model = LGBMClassifier(
        n_estimators=100,
        max_depth=paths["max_depth"],
        learning_rate=0.1,
        objective=paths["objective"],
        random_state=RANDOM_STATE,
        verbose=-1,
        force_col_wise=True
    )

    model.fit(X, y)

    save_model(model, X.columns.tolist(), paths)

    print(f"{paths['name']} model retrained and saved.")


# =========================
# PREDICT + SHAP
# =========================
def predict(model_type, input_csv, output_csv):
    paths = get_paths(model_type)

    model, columns = load_model(paths)

    df = pd.read_csv(input_csv, engine="pyarrow")
    X = prepare_features(df, saved_columns=columns)

    preds = model.predict(X)
    df["PREDICTED"] = preds

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if model_type == 0:
            df["CONFIDENCE"] = np.where(preds == 1, proba[:, 1], 1 - proba[:, 1])
        else:
            df["CONFIDENCE"] = proba.max(axis=1)

    # ===== SHAP =====
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    feature_names = X.columns.tolist()

    for i in range(len(df)):
        if model_type == 0:
            sample = shap_values[1][i] if isinstance(shap_values, list) else shap_values[i]
        else:
            cls = preds[i]
            cls_idx = list(model.classes_).index(cls)

            if isinstance(shap_values, list):
                sample = shap_values[cls_idx][i]
            else:
                sample = shap_values[i, :, cls_idx]

        order = np.argsort(np.abs(sample))[::-1]

        for j in range(min(TOP_N, len(order))):
            idx = order[j]
            df.loc[i, f"top_factor_{j+1}"] = feature_names[idx]
            df.loc[i, f"top_impact_{j+1}"] = float(sample[idx])

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Saved predictions: {output_csv}")


# =========================
# CLI
# =========================
def print_help():
    print("Usage:")
    print("  train <0|1> <fraction> <input.csv> <out.csv> <target>")
    print("  retrain <0|1> <input.csv> <target>")
    print("  predict <0|1> <input.csv> <output.csv>")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_help()
        sys.exit()

    cmd = sys.argv[1]

    if cmd == "train" and len(sys.argv) == 7:
        train(
            int(sys.argv[2]),
            float(sys.argv[3]),
            sys.argv[4],
            sys.argv[5],
            sys.argv[6]
        )

    elif cmd == "retrain" and len(sys.argv) == 5:
        retrain(
            int(sys.argv[2]),
            sys.argv[3],
            sys.argv[4]
        )

    elif cmd == "predict" and len(sys.argv) == 5:
        predict(
            int(sys.argv[2]),
            sys.argv[3],
            sys.argv[4]
        )

    else:
        print_help()