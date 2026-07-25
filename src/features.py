import pandas as pd

FEATURE_COLUMNS = [
    "entity_type_user", "entity_type_service_account", "entity_type_edge_device",
    "login_failed", "is_coldstart",
    "session_duration_z", "session_duration_trend",
    "distinct_resources_recent",
    "flag_impossible_travel", "flag_brute_force", "flag_lateral_movement",
    "flag_new_device", "flag_odd_hour", "flag_mismatched_fingerprint",
    "rule_flag_count",
]

FRIENDLY_NAMES = {
    "entity_type_user": "entity is a human user",
    "entity_type_service_account": "entity is a service account",
    "entity_type_edge_device": "entity is an edge device",
    "login_failed": "login failed",
    "is_coldstart": "limited history for this entity (cold start)",
    "session_duration_z": "session duration far from typical",
    "session_duration_trend": "session duration trending up vs. recent history",
    "distinct_resources_recent": "unusually many distinct resources touched recently",
    "flag_impossible_travel": "impossible travel between logins",
    "flag_brute_force": "brute-force login pattern",
    "flag_lateral_movement": "resource outside usual pattern",
    "flag_new_device": "new/unrecognized device",
    "flag_odd_hour": "unusual hour for this entity",
    "flag_mismatched_fingerprint": "device fingerprint mismatch",
    "rule_flag_count": "number of rule flags triggered",
}


def build_ml_features(df):
    """Returns (feat_df, df) — df is re-sorted by entity+time to match feat_df's
    row order 1:1 (callers should use the returned df from here on, not the
    one they passed in)."""
    df = df.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)
    dur_mean, dur_std = df["session_duration"].mean(), max(df["session_duration"].std(), 1e-6)

    trend = pd.Series(0.0, index=df.index)
    breadth = pd.Series(0.0, index=df.index)
    for entity_id, g in df.groupby("entity_id"):
        idx = g.index
        durations = g["session_duration"]
        prior_mean = durations.shift(1).rolling(5, min_periods=1).mean()
        trend.loc[idx] = (durations - prior_mean).fillna(0.0).values

        resources = g["resource_accessed"].reset_index(drop=True)
        breadth.loc[idx] = [resources.iloc[max(0, i - 4):i + 1].nunique() for i in range(len(resources))]

    trend_std = max(trend.std(), 1e-6)
    etype = pd.get_dummies(df["entity_type"], prefix="entity_type")
    for col in ["entity_type_user", "entity_type_service_account", "entity_type_edge_device"]:
        if col not in etype.columns:
            etype[col] = 0

    feat = pd.DataFrame({
        "entity_type_user": etype["entity_type_user"].astype(int),
        "entity_type_service_account": etype["entity_type_service_account"].astype(int),
        "entity_type_edge_device": etype["entity_type_edge_device"].astype(int),
        "login_failed": (~df["login_success"]).astype(int),
        "is_coldstart": df["flag_coldstart"].astype(int),
        "session_duration_z": ((df["session_duration"] - dur_mean) / dur_std).abs(),
        "session_duration_trend": (trend / trend_std).clip(-5, 5),
        "distinct_resources_recent": breadth,
        "flag_impossible_travel": df["flag_impossible_travel"].astype(int),
        "flag_brute_force": df["flag_brute_force"].astype(int),
        "flag_lateral_movement": df["flag_lateral_movement"].astype(int),
        "flag_new_device": df["flag_new_device"].astype(int),
        "flag_odd_hour": df["flag_odd_hour"].astype(int),
        "flag_mismatched_fingerprint": df["flag_mismatched_fingerprint"].astype(int),
        "rule_flag_count": df["rule_flag_count"],
    }, index=df.index)
    return feat, df
