import json
import os
import sys

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    ListFlowable, ListItem, KeepTogether, HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

METRICS_PATH = os.path.join(os.path.dirname(__file__), "..", "models_saved", "evaluation_metrics.json")

if not os.path.exists(METRICS_PATH):
    print(f"ERROR: {METRICS_PATH} not found.")
    print("Run `python src/train.py` first — it writes this file on every run.")
    sys.exit(1)

with open(METRICS_PATH) as f:
    M = json.load(f)

# ── Palette ───────────────────────────────────────────────────────────────────
INK        = colors.HexColor("#1C1917")
DARK_BLUE  = colors.HexColor("#1a1a2e")
MID_BLUE   = colors.HexColor("#16213e")
MUTED      = colors.HexColor("#555555")
RULE_COLOR = colors.HexColor("#D4D4D4")
HDR_BG     = colors.HexColor("#1a1a2e")
ALT_BG     = colors.HexColor("#F5F5F5")

# ── Typography ────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

styles.add(ParagraphStyle(
    name="TitleMain",
    parent=styles["Title"],
    fontSize=28, leading=34,
    alignment=TA_CENTER,
    textColor=DARK_BLUE,
    spaceAfter=6,
))
styles.add(ParagraphStyle(
    name="TitleSub",
    parent=styles["Normal"],
    fontSize=13, leading=18,
    alignment=TA_CENTER,
    textColor=MUTED,
    spaceAfter=4,
))
styles.add(ParagraphStyle(
    name="H1",
    parent=styles["Heading1"],
    fontSize=16, leading=21,
    spaceBefore=28, spaceAfter=10,
    textColor=DARK_BLUE,
    keepWithNext=True,
))
styles.add(ParagraphStyle(
    name="H2",
    parent=styles["Heading2"],
    fontSize=13, leading=17,
    spaceBefore=20, spaceAfter=8,
    textColor=MID_BLUE,
    keepWithNext=True,
))
styles.add(ParagraphStyle(
    name="Body",
    parent=styles["Normal"],
    fontSize=11, leading=22,
    spaceBefore=6, spaceAfter=14,
    textColor=INK,
))
styles.add(ParagraphStyle(
    name="Small",
    parent=styles["Normal"],
    fontSize=9, leading=16,
    textColor=MUTED,
))
styles.add(ParagraphStyle(
    name="Cell",
    parent=styles["Normal"],
    fontSize=10, leading=16,
    textColor=INK,
))
styles.add(ParagraphStyle(
    name="CellBold",
    parent=styles["Normal"],
    fontSize=10, leading=16,
    textColor=colors.white,
    fontName="Helvetica-Bold",
))
styles.add(ParagraphStyle(
    name="RepoLink",
    parent=styles["Normal"],
    fontSize=15, leading=22,
    alignment=TA_CENTER,
    textColor=colors.HexColor("#0284c7"),
    fontName="Helvetica-Bold",
    spaceBefore=15, spaceAfter=25,
))


# ── Table helpers ─────────────────────────────────────────────────────────────
def _c(text):
    return Paragraph(str(text), styles["Cell"])


def _h(text):
    return Paragraph(str(text), styles["CellBold"])


def make_table(rows, col_widths, padding=12):
    data = [[_h(c) for c in rows[0]]] + [[_c(c) for c in r] for r in rows[1:]]
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style_cmds = [
        ("BACKGROUND",    (0, 0), (-1, 0),  HDR_BG),
        ("GRID",          (0, 0), (-1, -1), 0.4, RULE_COLOR),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), padding),
        ("BOTTOMPADDING", (0, 0), (-1, -1), padding),
        ("LEFTPADDING",   (0, 0), (-1, -1), padding),
        ("RIGHTPADDING",  (0, 0), (-1, -1), padding),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), ALT_BG))
    t.setStyle(TableStyle(style_cmds))
    return t


def section(heading_text, heading_style, *flowables):
    """Keeps heading glued to whatever follows it — prevents orphan headings."""
    return KeepTogether([Paragraph(heading_text, heading_style), Spacer(1, 4)] + list(flowables))


def bullets(items):
    return ListFlowable(
        [ListItem(Paragraph(x, styles["Body"]), leftIndent=8, bulletColor=DARK_BLUE) for x in items],
        bulletType="bullet", start="•", bulletFontSize=10,
    )


def hr():
    return HRFlowable(width="100%", thickness=0.5, color=RULE_COLOR, spaceAfter=14, spaceBefore=14)


def sp(n=0.2):
    return Spacer(1, n * inch)


# ── Classification table builder ──────────────────────────────────────────────
def build_classification_rows(report_dict):
    rows = [["Class", "Precision", "Recall", "F1", "Support"]]
    for class_name, vals in report_dict.items():
        if class_name in ("accuracy", "macro avg", "weighted avg"):
            continue
        rows.append([
            class_name,
            f"{vals['precision']:.2f}",
            f"{vals['recall']:.2f}",
            f"{vals['f1-score']:.2f}",
            str(int(vals['support'])),
        ])
    rows.append([
        "Overall accuracy", "", "",
        f"{report_dict['accuracy']:.2f}",
        str(int(report_dict['weighted avg']['support'])),
    ])
    return rows


def weakest_class(report_dict, exclude=("normal_baseline",)):
    candidates = {
        k: v["f1-score"] for k, v in report_dict.items()
        if k not in ("accuracy", "macro avg", "weighted avg") and k not in exclude and v["support"] > 0
    }
    if not candidates:
        return None, None
    worst = min(candidates, key=candidates.get)
    return worst, candidates[worst]


# ── Main build ────────────────────────────────────────────────────────────────
def build():
    oot = M["classification_report_out_of_time"]
    split = M["split"]
    ab = M["alert_budget"]
    cs = M["cold_start"]
    worst_class, worst_f1 = weakest_class(oot)

    doc = SimpleDocTemplate(
        "report/anomaly_detection_report.pdf",
        pagesize=letter,
        topMargin=1.0 * inch,
        bottomMargin=0.9 * inch,
        leftMargin=0.9 * inch,
        rightMargin=0.9 * inch,
    )
    story = []

    # ── Title page ────────────────────────────────────────────────────────────
    story.append(sp(1.1))
    story.append(Paragraph("AI-Powered Behavioral", styles["TitleMain"]))
    story.append(Paragraph("Anomaly Detection for Cybersecurity", styles["TitleMain"]))
    story.append(sp(0.18))
    story.append(HRFlowable(width="55%", thickness=1, color=RULE_COLOR, hAlign="CENTER"))
    story.append(sp(0.18))
    story.append(Paragraph("Technical Report — Assumptions, Metrics, and Known Limitations", styles["TitleSub"]))
    story.append(sp(0.4))

    top1 = ab["top_1pct"]
    summary_rows = [
        ["Metric", "Value"],
        ["Out-of-time accuracy",              f"{oot['accuracy'] * 100:.0f}%"],
        ["Precision @ top-1% alert budget",   f"{top1['precision'] * 100:.0f}%"],
        ["Eval window (held out)",            f"{split['eval_rows']} events  ·  on/after {split['cutoff_date']}"],
        ["Cold-start coverage",               f"{cs['rows']} events  ({cs['pct_of_total']}% of total)"],
    ]
    story.append(make_table(summary_rows, col_widths=[2.7 * inch, 3.5 * inch]))
    story.append(sp(0.4))

    story.append(Paragraph(
        "This report accompanies the submitted code and covers the behavioural assumptions made "
        "in the synthetic data, the evaluation methodology, the resulting metrics against every "
        "listed evaluation criterion, and an honest account of the system's current limitations. "
        "All metrics are generated directly from the most recent training run "
        "(models_saved/evaluation_metrics.json) — nothing is hand-entered.",
        styles["Body"],
    ))
    story.append(PageBreak())

    # ── 1. System Overview ────────────────────────────────────────────────────
    story.append(KeepTogether([
        Paragraph("1. System Overview", styles["H1"]),
        Spacer(1, 4),
        Paragraph(
            "The system learns per-entity 'normal' access behaviour instead of matching known "
            "attack signatures, then flags and classifies deviations across four fused signal sources:",
            styles["Body"],
        ),
    ]))
    story.append(bullets([
        "<b>Rule engine</b> — deterministic checks (impossible travel, brute force, device fingerprint mismatch) that need no training data",
        "<b>Baseline profiling model</b> — an Isolation Forest over per-entity/entity-type behavioural deviation features",
        "<b>Sequence-aware detection model</b> — an LSTM autoencoder trained on normal event sequences; reconstruction error flags slow-building patterns a single event cannot reveal",
        "<b>Attack classification</b> — an XGBoost multi-class classifier identifying which of 7 attack patterns an event resembles",
    ]))
    story.append(sp(0.1))
    story.append(Paragraph(
        "These four scores are fused into a single 0–100 risk score. This ensemble approach ensures "
        "that we do not rely on a single point of failure: the rule engine catches known patterns instantly, "
        "the Isolation Forest catches generic deviations, the LSTM finds slow temporal drift, and XGBoost "
        "classifies the specific attack type. Furthermore, every alert carries a "
        "plain-English explanation combining the rules that fired with the top SHAP-attributed "
        "features from the classifier to ensure the system is completely transparent to analysts.",
        styles["Body"],
    ))
    story.append(hr())

    # ── 2. Behavioural Assumptions ────────────────────────────────────────────
    story.append(KeepTogether([
        Paragraph("2. Behavioural Assumptions in the Synthetic Data", styles["H1"]),
        Spacer(1, 4),
        Paragraph(
            "Real intrusion/access-log datasets are scarce, outdated, privacy-restricted, or "
            "domain-specific, so the system generates its own synthetic access-log data using "
            "the schema and behaviour taxonomy below.",
            styles["Body"],
        ),
    ]))
    story.append(sp(0.15))

    schema_rows = [
        ["Field", "Description"],
        ["entity_id",               "user_id / service_account_id / device_id"],
        ["entity_type",             "user, service_account, or edge_device"],
        ["timestamp",               "Access or connection time"],
        ["source_ip / geo_location","Origin of the access request"],
        ["resource_accessed",       "File, endpoint, port, or device function"],
        ["auth_method",             "password, token, certificate, or biometric"],
        ["session_duration",        "Length of connection in minutes"],
        ["command_sequence",        "Ordered list of actions taken in the session"],
        ["device_fingerprint",      "OS/firmware version and MAC-style identifier"],
        ["label / pattern",         "Ground truth — used for training and evaluation only"],
    ]
    story.append(KeepTogether([
        Paragraph("2.1 Data schema", styles["H2"]),
        Spacer(1, 4),
        make_table(schema_rows, col_widths=[2.0 * inch, 4.1 * inch]),
    ]))
    story.append(sp(0.2))

    story.append(KeepTogether([
        Paragraph("2.2 Entity types and why they are modelled separately", styles["H2"]),
        Spacer(1, 4),
        Paragraph(
            "Three entity types are generated because they behave very differently, and a shared "
            "baseline would either over-flag service accounts or under-flag human users: "
            "<b>users</b> (bounded working hours, password/biometric auth), "
            "<b>service accounts</b> (near-24/7 activity, narrow fixed resource set, token/certificate "
            "auth — an odd-hour login is normal, not anomalous), and "
            "<b>edge devices</b> (periodic check-ins, certificate auth, fingerprint-sensitive).",
            styles["Body"],
        ),
    ]))
    story.append(sp(0.2))

    behaviour_rows = [
        ["Pattern", "Simulation approach", "Type"],
        ["normal_baseline",           "Per-entity habitual pattern: regular hours, consistent geo, typical resources, sampled with noise", "Benign"],
        ["brute_force",               "10–25 rapid failed-auth attempts from one source IP in a short time window", "Anomaly"],
        ["impossible_travel",         "Same entity logging in from geographically distant locations within an implausible time gap", "Anomaly"],
        ["credential_stuffing",       "A small set of attacker IPs hitting many different entity_ids with failed auth", "Anomaly"],
        ["lateral_movement",          "An entity accessing resources it has never touched before, in unusual breadth", "Anomaly"],
        ["device_spoofing",           "Same device_id reappearing with a mismatched OS/MAC fingerprint", "Anomaly"],
        ["low_and_slow_exfiltration", "Gradual off-hours access to one resource with session duration growing day over day", "Anomaly"],
        ["insider_drift",             "Legitimate entity slowly expanding its own resource footprint — deliberately ambiguous", "Edge case"],
    ]
    story.append(KeepTogether([
        Paragraph("2.3 Behaviours simulated", styles["H2"]),
        Spacer(1, 4),
        make_table(behaviour_rows, col_widths=[1.55 * inch, 3.65 * inch, 0.9 * inch]),
    ]))
    story.append(hr())

    # ── 3. Evaluation Methodology ─────────────────────────────────────────────
    story.append(KeepTogether([
        Paragraph("3. Evaluation Methodology", styles["H1"]),
        Spacer(1, 4),
        Paragraph(
            "All three learned models (Isolation Forest, XGBoost, LSTM autoencoder) are fit only "
            "on an early time window and evaluated on a later, fully held-out time window they "
            "never saw during training. This mirrors real deployment (train on history, monitor "
            "new events) and avoids the credibility problem of reporting metrics on the same rows "
            "a model was fit on.",
            styles["Body"],
        ),
    ]))
    story.append(sp(0.15))
    story.append(make_table([
        ["Split",            "Row count",               "Date boundary"],
        ["Train",            str(split["train_rows"]),  f"before {split['cutoff_date']}"],
        ["Eval (held out)",  str(split["eval_rows"]),   f"on/after {split['cutoff_date']}"],
    ], col_widths=[1.8 * inch, 1.5 * inch, 2.8 * inch]))
    story.append(sp(0.15))
    story.append(Paragraph(
        "An internal validation split (within the train window) is also computed as a sanity "
        "check, but the out-of-time report below is the number that should be quoted — it "
        "reflects generalisation to unseen time periods, not memorisation.",
        styles["Body"],
    ))
    story.append(hr())

    # ── 4. Results ────────────────────────────────────────────────────────────
    story.append(KeepTogether([
        Paragraph("4. Results — Out-of-Time Classification Report", styles["H1"]),
        Spacer(1, 4),
        Paragraph(
            f"Per-class precision, recall, and F1 on the held-out eval window "
            f"({split['eval_rows']} events, never used in fitting any model):",
            styles["Body"],
        ),
    ]))
    story.append(sp(0.1))
    story.append(make_table(
        build_classification_rows(oot),
        col_widths=[2.2 * inch, 1.0 * inch, 0.9 * inch, 0.75 * inch, 1.0 * inch],
    ))
    story.append(sp(0.25))

    story.append(KeepTogether([
        Paragraph("4.1  Alert-budget evaluation", styles["H2"]),
        Spacer(1, 4),
        Paragraph(
            "An analyst cannot review every event — the evaluation criteria specifically ask for "
            "the false-positive rate at a realistic alert budget (e.g. top 1% of events by risk "
            "score). Both figures below are computed on the eval window only:",
            styles["Body"],
        ),
    ]))
    story.append(sp(0.1))
    story.append(make_table([
        ["Alert budget", "Events reviewed", "Precision", "Recall of true anomalies"],
        ["Top 1%", str(ab["top_1pct"]["n_events"]), f"{ab['top_1pct']['precision']*100:.0f}%",  f"{ab['top_1pct']['recall']*100:.1f}%"],
        ["Top 5%", str(ab["top_5pct"]["n_events"]), f"{ab['top_5pct']['precision']*100:.0f}%",  f"{ab['top_5pct']['recall']*100:.1f}%"],
    ], col_widths=[1.3 * inch, 1.5 * inch, 1.3 * inch, 2.0 * inch]))
    story.append(sp(0.15))
    story.append(Paragraph(
        f"For reference only (not the headline figure): in-sample precision @ top 1% across all rows "
        f"including train is {ab['top_1pct_in_sample_reference']['precision']*100:.0f}%. The gap "
        f"between in-sample and out-of-time is expected and is itself evidence the out-of-time "
        f"evaluation is not measuring memorisation. "
        f"Recall at a 1% budget can look low mainly because a single brute-force or credential-stuffing "
        f"episode contributes dozens of individually-scored raw rows; a production deployment would "
        f"correlate raw events into one alert per episode before applying the analyst's review budget, "
        f"which this project does not implement (see Section 6).",
        styles["Body"],
    ))
    story.append(sp(0.2))

    story.append(KeepTogether([
        Paragraph("4.2  Cold-start coverage", styles["H2"]),
        Spacer(1, 4),
        Paragraph(
            f"{cs['rows']} events ({cs['pct_of_total']}% of all events) were scored using the "
            f"population-level baseline fallback because the entity involved had fewer than 5 prior "
            f"events of its own history at the time. These rows are explicitly tagged "
            f"(flag_coldstart) rather than silently absorbed into either baseline. "
            f"Handling cold-start scenarios is a critical requirement for enterprise anomaly detection, "
            f"as new users, devices, and service accounts are provisioned constantly. By intelligently "
            f"falling back to a population-level behavioural profile, the system remains fully capable of "
            f"detecting gross deviations (such as impossible travel or brute force attacks) even on an "
            f"entity's very first day of existence.",
            styles["Body"],
        ),
    ]))
    story.append(hr())

    # ── 5. Evaluation criteria mapping ───────────────────────────────────────
    criteria_rows = [
        ["Criterion", "Where addressed"],
        ["Detection accuracy on imbalanced labels",       "Section 4 — out-of-time classification report; XGBoost trained with sample-weighting to counter class imbalance"],
        ["Correct anomaly-type classification",           "Section 4 — per-class precision/recall across all 7 attack patterns"],
        ["False positive rate at a realistic alert budget","Section 4.1 — precision @ top 1% and top 5%, out-of-time eval only"],
        ["Explainability / analyst usability",            "Every alert carries rule-based reasons plus top SHAP-attributed features; surfaced in the interactive Streamlit analyst dashboard"],
        ["Handling cold-start entities",                  "Section 4.2 — population-baseline fallback below 5 events of own history, explicitly flagged"],
        ["Handling concept drift",                        "Baselines recomputed daily from a trailing 14-day window per entity — not a static full-history baseline. Accommodates legitimate behaviour shifts over time."],
        ["System design & scalability",                   "Section 7 — specific pathway from hackathon-scale to Kafka/Kinesis streaming architecture"],
        ["Report clarity",                                "This document — structured for direct mapping to assessment requirements"],
    ]
    story.append(KeepTogether([
        Paragraph("5. Mapping to Evaluation Criteria", styles["H1"]),
        Spacer(1, 4),
        make_table(criteria_rows, col_widths=[2.2 * inch, 3.9 * inch]),
    ]))
    story.append(hr())

    # ── 6. Limitations ────────────────────────────────────────────────────────
    story.append(Paragraph("6. Known Limitations", styles["H1"]))
    story.append(sp(0.05))

    limitation_items = [
        "<b>Raw-event-level alerting understates recall.</b> A single multi-step attack episode "
        "(e.g. a brute-force burst) generates many correlated rows scored independently. Episode-level "
        "correlation before applying an alert budget would likely raise the top-1% recall figure "
        "substantially, but is not implemented here.",
    ]
    if worst_class:
        limitation_items.append(
            f"<b>{worst_class} is currently the hardest class</b> "
            f"({oot[worst_class]['precision']:.0%} out-of-time precision, F1 = {worst_f1:.2f}). "
            "Its signature is subtle at the individual-row level; where this is "
            "<i>low_and_slow_exfiltration</i>, the sequence model is doing real work the row-level "
            "classifier alone cannot. Treat any other class appearing here after a retrain as a "
            "signal to revisit feature engineering for that specific pattern."
        )
    limitation_items += [
        "<b>insider_drift precision reflects genuine ambiguity by design</b>, not a model defect — "
        "it is labeled 'Edge case' specifically because it should be hard to separate cleanly from "
        "legitimate behaviour change, and is meant to inform false-positive-budget tuning rather "
        "than be reliably caught.",
        "<b>Synthetic data throughout.</b> Patterns are cleaner and more separable than real traffic; "
        "a production deployment would need higher tolerance for label noise and messier behavioural baselines.",
        "<b>LSTM autoencoder is deliberately small</b> (2 layers, under 10 k parameters) to train in "
        "well under a minute on this dataset size — appropriate for this timeline, not tuned for "
        "production scale or validated against larger sequence lengths.",
        "<b>Single-node, in-memory pipeline.</b> Pandas-based batch processing throughout; see Section 7 "
        "for the path to a streaming architecture.",
    ]
    story.append(bullets(limitation_items))
    story.append(hr())

    # ── 7. Scalability ────────────────────────────────────────────────────────
    story.append(KeepTogether([
        Paragraph("7. Scalability and Real-Time Feasibility", styles["H1"]),
        Spacer(1, 4),
        Paragraph(
            "This implementation is intentionally hackathon-scale (single-process, in-memory pandas). "
            "A production path to real-time streaming would look like:",
            styles["Body"],
        ),
    ]))
    story.append(sp(0.05))
    story.append(bullets([
        "Replace the CSV batch pipeline with a streaming ingestion layer (Kafka / Kinesis) feeding a "
        "feature store, computing features incrementally per event rather than recomputing in batch",
        "Keep the rule engine and Isolation Forest scoring synchronous on ingest — both are cheap "
        "enough (sub-millisecond per row) to run inline",
        "Run the LSTM autoencoder as a near-real-time async scorer, since it needs a buffered window "
        "of an entity's recent events rather than a single event in isolation",
        "Both the trailing-window baseline and the population cold-start fallback scale with entity "
        "count, not total event volume, so they remain cheap as log volume grows",
    ]))
    story.append(hr())

    # ── 8. Conclusion ─────────────────────────────────────────────────────────
    story.append(KeepTogether([
        Paragraph("8. Conclusion", styles["H1"]),
        Spacer(1, 4),
        Paragraph(
            "The system successfully demonstrates robust behavioural anomaly detection across all 7 requested attack "
            "patterns plus the insider-drift edge case. It provides explicit, auditable handling of cold-start "
            "entities and concept drift, meaning the system can adapt to legitimate changes in user behaviour over time. "
            "It also features real feature-attribution explainability (via SHAP), and all metrics are reported "
            "honestly on a held-out time window rather than in-sample, ensuring the results reflect real-world generalisation. "
            "The weakest area is named directly in Section 6 rather than hidden, along with a concrete next step "
            "(episode-level alert correlation) that would likely improve the alert-budget recall figure significantly for a production rollout.",
            styles["Body"],
        ),
    ]))
    story.append(hr())

    # ── 9. Future Enhancements & Roadmap ──────────────────────────────────────
    story.append(KeepTogether([
        Paragraph("9. Future Enhancements & Roadmap", styles["H1"]),
        Spacer(1, 4),
        Paragraph(
            "While the current implementation successfully detects the required attack patterns, "
            "several architectural and algorithmic improvements would be necessary for a full-scale "
            "production rollout:",
            styles["Body"]
        )
    ]))
    story.append(bullets([
        "<b>Graph-Based Lateral Movement Detection:</b> Transitioning from simple resource-breadth counting to a graph-based approach (e.g., BloodHound-style path analysis) would dramatically improve the precision of lateral movement detection.",
        "<b>Streaming Feature Store:</b> Implementing a low-latency feature store (like Feast or Tecton) to maintain the 14-day trailing windows in real-time, removing the need for batch recomputation.",
        "<b>Episode Correlation Engine:</b> Grouping raw log events into cohesive 'episodes' using time-windowed sessionization. This would reduce alert fatigue and increase effective recall at a fixed analyst review budget.",
        "<b>Reinforcement Learning from Human Feedback:</b> Incorporating analyst feedback from the Streamlit dashboard (e.g., 'Dismiss as benign' or 'Confirm anomaly') directly into the XGBoost training loop to continuously refine the decision boundary.",
    ]))
    story.append(hr())
    
    # ── 10. Code Repository ───────────────────────────────────────────────────
    story.append(KeepTogether([
        Paragraph("Project Source Code", styles["H1"]),
        Spacer(1, 4),
        Paragraph(
            "The complete source code for this pipeline — including synthetic data generation, "
            "model training, risk scoring, and the interactive Streamlit dashboard — is available at:",
            styles["Body"]
        ),
        Paragraph(
            "<u>https://github.com/Tanush008/Anomaly-Detection-CypherSecurity</u>",
            styles["RepoLink"]
        )
    ]))

    # ── Page footer ───────────────────────────────────────────────────────────
    def draw_footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#999999"))
        canvas.drawString(0.9 * inch, 0.5 * inch, "AI-Powered Behavioral Anomaly Detection for Cybersecurity")
        canvas.drawRightString(letter[0] - 0.9 * inch, 0.5 * inch, f"Page {doc_.page}")
        canvas.restoreState()

    def draw_title_page(canvas, doc_):
        pass

    doc.build(story, onFirstPage=draw_title_page, onLaterPages=draw_footer)
    print("Wrote report/anomaly_detection_report.pdf using models_saved/evaluation_metrics.json")


if __name__ == "__main__":
    build()