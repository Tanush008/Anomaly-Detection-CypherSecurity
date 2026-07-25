import math
import numpy as np
import pandas as pd

MIN_EVENTS_FOR_OWN_BASELINE = 5
DRIFT_WINDOW_DAYS = 14


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dlambda = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _daily_stats(g):
    """One row of stats per (entity, day)."""
    g = g.copy()
    g["day"] = g["timestamp"].dt.floor("D")
    out = g.groupby("day").agg(
        hour_mean_day=("timestamp", lambda s: s.dt.hour.mean()),
        n_events_day=("timestamp", "count"),
        fail_rate_day=("login_success", lambda s: 1 - s.mean()),
    ).reset_index()
    return out


def build_baselines(df):
    """
    Returns:
      per_entity_daily: dict[entity_id] -> DataFrame indexed by day with TRAILING
        window baseline (hour_mean, hour_std, n_hist_events, usual_resources,
        usual_devices) computed using only that entity's own PRIOR events.
      population_baseline: dict[entity_type] -> dict of population-level stats,
        used as the cold-start fallback.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["day"] = df["timestamp"].dt.floor("D")

    # population-level baseline per entity_type (cold-start fallback)
    population_baseline = {}
    for etype, g in df.groupby("entity_type"):
        hours = g["timestamp"].dt.hour
        population_baseline[etype] = {
            "hour_mean": hours.mean(),
            "hour_std": max(hours.std(), 1.0),
            "usual_resources": set(g["resource_accessed"].value_counts().head(5).index),
            "fail_rate": 1 - g["login_success"].mean(),
        }

    # per-entity trailing-window baseline, recomputed per day (concept drift)
    per_entity_daily = {}
    for entity_id, g in df.groupby("entity_id"):
        g = g.sort_values("timestamp")
        days = sorted(g["day"].unique())
        day_records = {}
        for day in days:
            window_start = day - pd.Timedelta(days=DRIFT_WINDOW_DAYS)
            hist = g[(g["day"] >= window_start) & (g["day"] < day)]
            n_hist = len(hist)
            if n_hist >= MIN_EVENTS_FOR_OWN_BASELINE:
                hours = hist["timestamp"].dt.hour
                day_records[day] = {
                    "hour_mean": hours.mean(),
                    "hour_std": max(hours.std(), 1.0),
                    "usual_resources": set(hist["resource_accessed"].value_counts().head(4).index),
                    "usual_devices": set(hist["device_id"].unique()),
                    "n_hist_events": n_hist,
                    "source": "own_history",
                }
            else:
                # COLD START fallback: not enough personal history yet
                etype = g["entity_type"].iloc[0]
                pb = population_baseline[etype]
                day_records[day] = {
                    "hour_mean": pb["hour_mean"],
                    "hour_std": pb["hour_std"],
                    "usual_resources": pb["usual_resources"],
                    "usual_devices": set(g[g["day"] < day]["device_id"].unique()),
                    "n_hist_events": n_hist,
                    "source": "population_coldstart",
                }
        per_entity_daily[entity_id] = day_records

    return per_entity_daily, population_baseline


def apply_rules(df, per_entity_daily, max_speed_kmh=900, brute_force_window_s=120, brute_force_count=6):
    """Deterministic, explainable rule checks — fast, no training data needed."""
    df = df.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["day"] = df["timestamp"].dt.floor("D")

    n = len(df)
    flags = {k: np.zeros(n, dtype=bool) for k in [
        "flag_impossible_travel", "flag_brute_force", "flag_lateral_movement",
        "flag_new_device", "flag_odd_hour", "flag_mismatched_fingerprint",
        "flag_coldstart",
    ]}

    for entity_id, g in df.groupby("entity_id"):
        idx = g.index.to_list()

        for i in range(1, len(idx)):
            prev, cur = idx[i - 1], idx[i]
            dt_h = (df.at[cur, "timestamp"] - df.at[prev, "timestamp"]).total_seconds() / 3600
            if dt_h <= 0:
                continue
            dist = haversine_km(df.at[prev, "lat"], df.at[prev, "lon"], df.at[cur, "lat"], df.at[cur, "lon"])
            if dist / dt_h > max_speed_kmh and dist > 300:
                flags["flag_impossible_travel"][cur] = True

        fails = g[g["login_success"] == False]
        fail_times, fail_idx = fails["timestamp"].tolist(), fails.index.tolist()
        for i, t in enumerate(fail_times):
            window = [t2 for t2 in fail_times if 0 <= (t - t2).total_seconds() <= brute_force_window_s]
            if len(window) >= brute_force_count:
                flags["flag_brute_force"][fail_idx[i]] = True

        base_by_day = per_entity_daily[entity_id]
        for i in idx:
            day = df.at[i, "day"]
            base = base_by_day.get(day) or list(base_by_day.values())[0]
            if df.at[i, "resource_accessed"] not in base["usual_resources"]:
                flags["flag_lateral_movement"][i] = True
            if df.at[i, "device_id"] not in base["usual_devices"]:
                flags["flag_new_device"][i] = True
            hour = df.at[i, "timestamp"].hour
            z = abs(hour - base["hour_mean"]) / base["hour_std"]
            if z > 3:
                flags["flag_odd_hour"][i] = True
            if base["source"] == "population_coldstart":
                flags["flag_coldstart"][i] = True

    # device spoofing proxy: same device_id, but fingerprint's OS token changed
    fp_by_device = {}
    for i in df.index:
        dev, fp = df.at[i, "device_id"], str(df.at[i, "device_fingerprint"])
        os_token = fp.split("|")[0]
        if dev in fp_by_device and fp_by_device[dev] != os_token:
            flags["flag_mismatched_fingerprint"][i] = True
        fp_by_device.setdefault(dev, os_token)

    for k, v in flags.items():
        df[k] = v
    df["rule_flag_count"] = df[[k for k in flags if k != "flag_coldstart"]].sum(axis=1)
    return df
