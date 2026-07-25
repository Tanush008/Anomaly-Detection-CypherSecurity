import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import xgboost as xgb
import shap
import joblib

from features import FEATURE_COLUMNS, FRIENDLY_NAMES


def train_isolation_forest(feat_df, train_mask, contamination=0.08):
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    model.fit(feat_df.loc[train_mask, FEATURE_COLUMNS])
    raw = -model.decision_function(feat_df[FEATURE_COLUMNS])  # score ALL rows, fit on train only
    anomaly_score = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    return model, anomaly_score


def train_classifier(feat_df, labels, train_mask, eval_mask):
    """Fits on train_mask only. Returns two reports:
      - internal_report: train/test split WITHIN the train window (sanity check)
      - out_of_time_report: predictions on eval_mask, the window the model never saw —
        this is the credible number, quote THIS one, not the internal one.
    """
    le = LabelEncoder()
    le.fit(labels)  # fit on the full label set so every class is known, even if rare in train
    y_all = le.transform(labels)

    X_train, y_train = feat_df.loc[train_mask, FEATURE_COLUMNS], y_all[train_mask.values]
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42,
        stratify=y_train if pd.Series(y_train).nunique() > 1 else None,
    )

    model = xgb.XGBClassifier(
        n_estimators=250, max_depth=4, learning_rate=0.1,
        objective="multi:softprob", num_class=len(le.classes_),
        eval_metric="mlogloss", random_state=42,
    )
    counts = pd.Series(y_tr).value_counts()
    weights = pd.Series(y_tr).map(lambda c: len(y_tr) / (len(counts) * counts[c]))
    model.fit(X_tr, y_tr, sample_weight=weights.values)

    internal_report = classification_report(
        y_val, model.predict(X_val), labels=list(range(len(le.classes_))),
        target_names=le.classes_, zero_division=0,
    )

    X_eval, y_eval = feat_df.loc[eval_mask, FEATURE_COLUMNS], y_all[eval_mask.values]
    out_of_time_report = classification_report(
        y_eval, model.predict(X_eval), labels=list(range(len(le.classes_))),
        target_names=le.classes_, zero_division=0,
    )
    out_of_time_report_dict = classification_report(
        y_eval, model.predict(X_eval), labels=list(range(len(le.classes_))),
        target_names=le.classes_, zero_division=0, output_dict=True,
    )

    return model, le, internal_report, out_of_time_report, out_of_time_report_dict


def build_shap_explainer(model):
    return shap.TreeExplainer(model)


def batch_shap_top_features(explainer, feat_df, top_k=2):
    """Vectorized SHAP over the whole matrix at once (looping row-by-row through
    TreeExplainer is what made SHAP unusably slow in the first draft — batching
    it is what makes 'feature attribution per alert' actually feasible to ship)."""
    shap_values = explainer.shap_values(feat_df[FEATURE_COLUMNS])
    arr = np.array(shap_values)  # (n_classes, n_rows, n_features) or (n_rows, n_features, n_classes)
    if arr.ndim == 3 and arr.shape[0] != feat_df.shape[0]:
        arr = np.transpose(arr, (1, 2, 0))  # -> (n_rows, n_features, n_classes)
    impact = np.abs(arr).sum(axis=2) if arr.ndim == 3 else np.abs(arr)  # (n_rows, n_features)

    top_features = []
    for row in impact:
        order = np.argsort(row)[::-1][:top_k]
        names = [FRIENDLY_NAMES[FEATURE_COLUMNS[i]] for i in order if row[i] > 1e-6]
        top_features.append(names)
    return top_features


def save_artifacts(iso_model, clf_model, label_encoder, out_dir="models_saved"):
    joblib.dump(iso_model, f"{out_dir}/isolation_forest.joblib")
    joblib.dump(clf_model, f"{out_dir}/xgb_classifier.joblib")
    joblib.dump(label_encoder, f"{out_dir}/label_encoder.joblib")
