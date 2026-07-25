import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from generate_logs import generate
from baseline_profiling import build_baselines, apply_rules
from features import build_ml_features
from models import train_isolation_forest, train_classifier, build_shap_explainer, batch_shap_top_features, save_artifacts
from sequence_model import compute_sequence_scores
from risk_engine import compute_risk_scores, explain

DATA_PATH = "data/access_logs.csv"
SCORED_PATH = "data/scored_logs.csv"
EVAL_FRACTION = 0.27  # ~last 27% of days held out as the out-of-time eval window


def precision_at_top_k_percent(df, k_percent=1.0):
    n_top = max(1, int(len(df) * k_percent / 100))
    top = df.sort_values("risk_score", ascending=False).head(n_top)
    precision = (top["signal_type"] != "Benign").mean()
    recall = (top["signal_type"] == "Anomaly").sum() / max(1, (df["signal_type"] == "Anomaly").sum())
    return n_top, precision, recall


def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models_saved", exist_ok=True)

    if not os.path.exists(DATA_PATH):
        print("Generating synthetic logs...")
        df = generate(out_path=DATA_PATH)
    else:
        print(f"Loading existing logs from {DATA_PATH}")
        df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

    print("\nBuilding baselines (cold-start fallback + trailing-window drift)...")
    per_entity_daily, population_baseline = build_baselines(df)

    print("Applying rule engine...")
    df = apply_rules(df, per_entity_daily)

    print("Building ML feature matrix (entity-type + trend/breadth features)...")
    feat_df, df = build_ml_features(df)  # df re-sorted to match feat_df's row order

    cutoff_day = df["timestamp"].min() + (df["timestamp"].max() - df["timestamp"].min()) * (1 - EVAL_FRACTION)
    train_mask = df["timestamp"] < cutoff_day
    eval_mask = ~train_mask
    df["split"] = train_mask.map({True: "train", False: "eval"})
    print(f"\nTemporal split: train={train_mask.sum()} rows (before {cutoff_day.date()}), "
          f"eval={eval_mask.sum()} rows (on/after {cutoff_day.date()}) — eval is fully held out from fitting")

    print("\nTraining Isolation Forest on TRAIN window only (baseline profiling model)...")
    iso_model, anomaly_score = train_isolation_forest(feat_df, train_mask)

    print("Training XGBoost on TRAIN window only (anomaly-type classification)...")
    clf_model, label_encoder, internal_report, oot_report, oot_report_dict = train_classifier(
        feat_df, df["pattern"], train_mask, eval_mask
    )
    print("\nInternal validation report (held-out rows WITHIN the train window):\n", internal_report)
    print("\n*** OUT-OF-TIME report (eval window, never seen during fitting — quote THIS one) ***\n", oot_report)

    print("Training LSTM autoencoder on TRAIN-window benign sequences only...")
    sequence_score, autoencoder = compute_sequence_scores(df, feat_df, train_mask)

    print("Computing fused risk scores for all rows...")
    risk_100, predicted_pattern, attack_conf = compute_risk_scores(
        feat_df, anomaly_score, sequence_score, clf_model, label_encoder
    )

    print("Computing SHAP feature attribution (batched, all rows)...")
    explainer = build_shap_explainer(clf_model)
    shap_top = batch_shap_top_features(explainer, feat_df, top_k=2)

    df["anomaly_score"] = anomaly_score.round(3)
    df["sequence_score"] = sequence_score.round(3)
    df["risk_score"] = risk_100
    df["predicted_pattern"] = predicted_pattern
    df["model_confidence"] = attack_conf.round(3)
    df["explanation"] = [
        explain(feat_df.iloc[i], sequence_score.iloc[i], shap_top[i]) for i in range(len(df))
    ]

    eval_df = df[eval_mask]
    n_top, precision, recall = precision_at_top_k_percent(eval_df, k_percent=1.0)
    n_top5, precision5, recall5 = precision_at_top_k_percent(eval_df, k_percent=5.0)
    print(f"\n--- OUT-OF-TIME alert-budget evaluation (EVAL-window events only) ---")
    print(f"Top 1% ({n_top} rows): precision={precision:.3f}, recall of true anomalies={recall:.3f}")
    print(f"Top 5% ({n_top5} rows): precision={precision5:.3f}, recall of true anomalies={recall5:.3f}")
    print("(Recall at a 1% budget looks low mainly because brute-force/credential-stuffing episodes each "
          "contribute many correlated raw rows — in production these would be correlated into one alert "
          "per episode before applying the budget, not scored as independent events. Noted as a limitation.)")

    n_top_all, precision_all, _ = precision_at_top_k_percent(df, k_percent=1.0)
    print(f"\n(For reference only, NOT the headline number — in-sample precision @ top 1% "
          f"across all rows including train: {precision_all:.3f}. Expect this to look better "
          f"than the out-of-time number; that gap is normal and worth naming in your report.)")

    coldstart_rows = df["flag_coldstart"].sum()
    print(f"\nCold-start rows (population baseline fallback): {coldstart_rows} ({coldstart_rows/len(df)*100:.1f}%)")

    df = df.sort_values("risk_score", ascending=False)
    df.to_csv(SCORED_PATH, index=False)
    save_artifacts(iso_model, clf_model, label_encoder)
    autoencoder.save("models_saved/sequence_autoencoder.keras")

    with open("models_saved/evaluation_summary.txt", "w") as f:
        f.write(f"Temporal split: train before {cutoff_day.date()}, eval on/after\n")
        f.write(f"Train rows: {train_mask.sum()}  Eval rows: {eval_mask.sum()}\n\n")
        f.write("INTERNAL validation report (within train window):\n" + internal_report + "\n\n")
        f.write("OUT-OF-TIME report (held-out eval window — the credible number):\n" + oot_report + "\n\n")
        f.write(f"Precision @ top 1% (out-of-time, eval only): {precision:.3f}\n")
        f.write(f"Recall of anomalies in top 1% (out-of-time): {recall:.3f}\n")
        f.write(f"Precision @ top 5% (out-of-time, eval only): {precision5:.3f}\n")
        f.write(f"Recall of anomalies in top 5% (out-of-time): {recall5:.3f}\n")
        f.write(f"Precision @ top 1% (in-sample, all rows, reference only): {precision_all:.3f}\n")
        f.write(f"Cold-start rows: {coldstart_rows} ({coldstart_rows/len(df)*100:.1f}%)\n")

    import json
    metrics = {
        "split": {
            "cutoff_date": str(cutoff_day.date()),
            "train_rows": int(train_mask.sum()),
            "eval_rows": int(eval_mask.sum()),
        },
        "classification_report_out_of_time": oot_report_dict,
        "alert_budget": {
            "top_1pct": {"n_events": n_top, "precision": round(precision, 3), "recall": round(recall, 3)},
            "top_5pct": {"n_events": n_top5, "precision": round(precision5, 3), "recall": round(recall5, 3)},
            "top_1pct_in_sample_reference": {"n_events": n_top_all, "precision": round(precision_all, 3)},
        },
        "cold_start": {
            "rows": int(coldstart_rows),
            "pct_of_total": round(coldstart_rows / len(df) * 100, 1),
        },
        "total_rows": len(df),
    }
    with open("models_saved/evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved scored logs to {SCORED_PATH}")
    print(f"Saved evaluation summary to models_saved/evaluation_summary.txt")
    print(f"Saved structured metrics to models_saved/evaluation_metrics.json (read by report/generate_report.py)")
    print("\nNow run:  streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
