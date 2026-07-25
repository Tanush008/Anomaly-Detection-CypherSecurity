import numpy as np

WEIGHTS = {"rules": 0.30, "iso": 0.25, "sequence": 0.20, "clf": 0.25}

RULE_LABELS = {
    "flag_impossible_travel": "Impossible travel between logins",
    "flag_brute_force": "Brute-force login pattern",
    "flag_lateral_movement": "Access to resource outside usual pattern",
    "flag_new_device": "Login from a new/unrecognized device",
    "flag_odd_hour": "Login at an unusual hour for this entity",
    "flag_mismatched_fingerprint": "Device fingerprint mismatch on known device_id",
}


def compute_risk_scores(feat_df, anomaly_score, sequence_score, clf_model, label_encoder):
    from features import FEATURE_COLUMNS
    proba = clf_model.predict_proba(feat_df[FEATURE_COLUMNS])
    pred_idx = proba.argmax(axis=1)
    classes = list(label_encoder.classes_)
    pred_label = label_encoder.inverse_transform(pred_idx)
    pred_conf = proba.max(axis=1)

    none_idx = classes.index("normal_baseline") if "normal_baseline" in classes else None
    attack_conf = np.where(pred_idx == none_idx, 1 - pred_conf, pred_conf) if none_idx is not None else pred_conf
    predicted_pattern = np.where(pred_idx == none_idx, "normal_baseline", pred_label) if none_idx is not None else pred_label

    rules_norm = feat_df["rule_flag_count"].clip(upper=5) / 5.0
    risk = (
        WEIGHTS["rules"] * rules_norm
        + WEIGHTS["iso"] * anomaly_score
        + WEIGHTS["sequence"] * sequence_score.values
        + WEIGHTS["clf"] * attack_conf
    )
    return (risk * 100).round(1), predicted_pattern, attack_conf


def explain(feat_row, sequence_score_val, shap_top_features):
    """shap_top_features: list of friendly feature-name strings for this row,
    already computed in batch (see models.batch_shap_top_features)."""
    reasons = [RULE_LABELS[c] for c in RULE_LABELS if feat_row.get(c)]
    if feat_row.get("is_coldstart"):
        reasons.append("Limited history for this entity — scored against population baseline (cold start)")
    if sequence_score_val > 0.7:
        reasons.append("Unusual pattern across recent event sequence (gradual drift, sequence model)")

    parts = []
    if reasons:
        parts.append("Rules: " + "; ".join(reasons))
    else:
        parts.append("No deterministic rule fired")
    if shap_top_features:
        parts.append("Model attribution: " + ", ".join(shap_top_features))
    return " | ".join(parts)
