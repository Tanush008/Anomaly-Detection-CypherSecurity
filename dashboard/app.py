import os
import time
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Behavioral Anomaly Detection", layout="wide")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "scored_logs.csv")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=["timestamp"])


if not os.path.exists(DATA_PATH):
    st.error("No scored logs found. Run `python src/train.py` from the project root first.")
    st.stop()

df = load_data()

st.title("Behavioral anomaly detection — analyst dashboard")
st.caption(
    "Learns per-entity/entity-type baselines (users, service accounts, edge devices) "
    "and flags deviations across four fused signals: rule engine, baseline profiling "
    "(Isolation Forest), sequence-aware detection (LSTM autoencoder), and attack classification (XGBoost). "
    "All models were fit on an early time window and evaluated on a later, held-out window — "
    "the metrics below are from data the models never trained on."
)

eval_df = df[df["split"] == "eval"]

# ---- Summary metrics ----
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total events", f"{len(df):,}")
c2.metric("Eval-window events (held out)", f"{len(eval_df):,}")
c3.metric("Unique entities", df["entity_id"].nunique())
c4.metric("True anomalies (eval window)", int((eval_df["signal_type"] == "Anomaly").sum()))
c5.metric("Cold-start events", int(df["flag_coldstart"].sum()))

col_a, col_b = st.columns([2, 1])
with col_a:
    counts = df[df["predicted_pattern"] != "normal_baseline"]["predicted_pattern"].value_counts().reset_index()
    counts.columns = ["pattern", "count"]
    st.plotly_chart(px.bar(counts, x="pattern", y="count", title="Detected alerts by pattern (all rows)"), use_container_width=True)
with col_b:
    n_top = max(1, int(len(eval_df) * 0.01))
    top = eval_df.sort_values("risk_score", ascending=False).head(n_top)
    precision_top1 = (top["signal_type"] != "Benign").mean()
    st.metric("Precision @ top 1% (out-of-time, eval only)", f"{precision_top1*100:.1f}%")
    st.caption(f"Of the {n_top} highest-risk EVAL-window events — the models never saw these while "
               "training — this fraction were truly Anomaly/Edge case. This is the credible number; "
               "an in-sample-only score would look better but wouldn't mean anything.")

st.divider()

# ---- Live replay simulation ----
st.subheader("Live feed (simulated replay)")
st.caption(
    "Curated for demo purposes: one example of each detected pattern (plus a few normal "
    "events for context), replayed in chronological order — a literal 'last 30 events' feed "
    "would be almost all normal_baseline, since that's ~91% of all traffic."
)
if st.toggle("Start live replay", value=False):
    feed_placeholder = st.empty()

    anomalous = df[df["signal_type"] != "Benign"]
    one_per_pattern = anomalous.sort_values("risk_score", ascending=False).groupby("predicted_pattern").head(1)
    normal_context = df[df["signal_type"] == "Benign"].sample(n=min(10, len(df)), random_state=1)
    recent = pd.concat([one_per_pattern, normal_context]).sort_values("timestamp")

    lines = []
    for _, row in recent.iterrows():
        color = "🔴" if row["risk_score"] >= 60 else ("🟡" if row["risk_score"] >= 30 else "🟢")
        lines.append(
            f"{color} `{row['timestamp']}` **{row['entity_id']}** ({row['entity_type']}) → "
            f"{row['resource_accessed']} — risk **{row['risk_score']}** ({row['predicted_pattern']})"
        )
        feed_placeholder.markdown("  \n".join(lines))  # accumulates, so the whole mix stays visible
        time.sleep(0.4)

st.divider()

# ---- Alert table ----
st.subheader("Alerts")
f1, f2, f3 = st.columns([1, 1, 1])
min_risk = f1.slider("Minimum risk score", 0, 100, 30)
entity_type_filter = f2.multiselect("Entity type", df["entity_type"].unique().tolist(), default=df["entity_type"].unique().tolist())
split_filter = f3.multiselect("Data split", ["train", "eval"], default=["train", "eval"])
filtered = df[
    (df["risk_score"] >= min_risk) & (df["entity_type"].isin(entity_type_filter)) & (df["split"].isin(split_filter))
].sort_values("risk_score", ascending=False)

st.dataframe(
    filtered[[
        "timestamp", "split", "entity_id", "entity_type", "geo_location", "resource_accessed",
        "auth_method", "predicted_pattern", "risk_score", "explanation",
    ]].head(200),
    use_container_width=True, height=350,
)

st.divider()

# ---- Drill-down ----
st.subheader("Investigate an entity")
entity_options = sorted(df["entity_id"].unique())
selected = st.selectbox("Select entity", entity_options)

entity_df = df[df["entity_id"] == selected].sort_values("timestamp")
top_alert = entity_df.sort_values("risk_score", ascending=False).iloc[0]

cA, cB = st.columns(2)
cA.markdown(f"**Highest risk event:** score `{top_alert['risk_score']}` — pattern: `{top_alert['predicted_pattern']}`")
cA.info(f"Why flagged: {top_alert['explanation']}")
cB.markdown(f"**Device fingerprint:** `{top_alert['device_fingerprint']}`")
cB.markdown(f"**Command sequence:** `{top_alert['command_sequence']}`")
cB.markdown(f"**Session duration:** {top_alert['session_duration']} min")

timeline_fig = px.scatter(
    entity_df, x="timestamp", y="risk_score", color="predicted_pattern",
    hover_data=["resource_accessed", "geo_location", "explanation"],
    title=f"Risk score timeline — {selected}",
)
st.plotly_chart(timeline_fig, use_container_width=True)
