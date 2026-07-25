import json
import os
import sys

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    ListFlowable, ListItem, KeepTogether,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER

METRICS_PATH = os.path.join(os.path.dirname(__file__), "..", "models_saved", "evaluation_metrics.json")

if not os.path.exists(METRICS_PATH):
    print(f"ERROR: {METRICS_PATH} not found.")
    print("Run `python src/train.py` first — it writes this file on every run.")
    sys.exit(1)

with open(METRICS_PATH) as f:
    M = json.load(f)

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="H1", parent=styles["Heading1"], spaceBefore=18, spaceAfter=8,
                           textColor=colors.HexColor("#1a1a2e"), keepWithNext=True))
styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"], spaceBefore=12, spaceAfter=6,
                           textColor=colors.HexColor("#16213e"), keepWithNext=True))
styles.add(ParagraphStyle(name="Body", parent=styles["Normal"], spaceBefore=4, spaceAfter=8, leading=15))
styles.add(ParagraphStyle(name="Small", parent=styles["Normal"], fontSize=9, textColor=colors.HexColor("#555555"), leading=12))
styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER))
styles.add(ParagraphStyle(name="SubtitleCenter", parent=styles["Normal"], alignment=TA_CENTER, textColor=colors.HexColor("#555555"), fontSize=11))

# Cell styles for table content — this is what makes cells actually WRAP
# instead of overflow. Every cell in every table below is built with cell()/
# header_cell(), never a bare string.
styles.add(ParagraphStyle(name="Cell", parent=styles["Normal"], fontSize=9, leading=11.5))
styles.add(ParagraphStyle(name="CellHeader", parent=styles["Normal"], fontSize=9, leading=11.5,
                           textColor=colors.white, fontName="Helvetica-Bold"))

TABLE_HEADER_BG = colors.HexColor("#1a1a2e")
TABLE_ALT_BG = colors.HexColor("#f2f2f7")


def cell(text):
    return Paragraph(str(text), styles["Cell"])


def header_cell(text):
    return Paragraph(str(text), styles["CellHeader"])


def wrapped_table(rows, col_widths):
    """rows: list of lists of PLAIN strings/numbers. First row is treated as
    the header. Every cell is wrapped in a Paragraph so long text wraps
    within its column instead of overflowing it."""
    data = [[header_cell(c) for c in rows[0]]] + [[cell(c) for c in r] for r in rows[1:]]
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEADER_BG),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), TABLE_ALT_BG))
    t.setStyle(TableStyle(style))
    return t


def bullets(items):
    return ListFlowable(
        [ListItem(Paragraph(x, styles["Body"]), leftIndent=6) for x in items],
        bulletType="bullet", start="•",
    )


def heading_with(title_style, title_text, *following_flowables):
    """Groups a heading with whatever comes right after it so the pair can
    never be split across a page break (see keepWithNext note above — this
    adds a second layer of protection for cases with a table right after
    the heading, where keepWithNext alone is less reliable)."""
    return KeepTogether([Paragraph(title_text, title_style), Spacer(1, 4)] + list(following_flowables))


def build_classification_rows(report_dict):
    rows = [["Class", "Precision", "Recall", "F1", "Support"]]
    for class_name, vals in report_dict.items():
        if class_name in ("accuracy", "macro avg", "weighted avg"):
            continue
        rows.append([
            class_name, f"{vals['precision']:.2f}", f"{vals['recall']:.2f}",
            f"{vals['f1-score']:.2f}", str(int(vals['support'])),
        ])
    rows.append(["Overall accuracy", "", "", f"{report_dict['accuracy']:.2f}", str(int(report_dict['weighted avg']['support']))])
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


def build():
    oot = M["classification_report_out_of_time"]
    split = M["split"]
    ab = M["alert_budget"]
    cs = M["cold_start"]
    worst_class, worst_f1 = weakest_class(oot)

    doc = SimpleDocTemplate(
        "report/anomaly_detection_report.pdf", pagesize=letter,
        topMargin=0.9 * inch, bottomMargin=0.8 * inch, leftMargin=0.85 * inch, rightMargin=0.85 * inch,
    )
    story = []

    # ---------- Title ----------
    story.append(Spacer(1, 1.2 * inch))
    story.append(Paragraph("AI-Powered Behavioral Anomaly Detection", styles["TitleCenter"]))
    story.append(Paragraph("for Cybersecurity", styles["TitleCenter"]))
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph("Technical Report — Assumptions, Metrics, and Known Limitations", styles["SubtitleCenter"]))
    story.append(Spacer(1, 2.5 * inch))
    story.append(Paragraph(
        "This report accompanies the submitted code and covers the behavioural assumptions made "
        "in the synthetic data, the evaluation methodology, the resulting metrics against every "
        "listed evaluation criterion, and an honest account of the system's current limitations. "
        "All metrics below are generated directly from the most recent training run "
        "(models_saved/evaluation_metrics.json), not hand-entered.",
        styles["Body"],
    ))
    story.append(PageBreak())

    # ---------- 1. Overview ----------
    story.append(Paragraph("1. System Overview", styles["H1"]))
    story.append(Paragraph(
        "The system learns per-entity 'normal' access behaviour instead of matching known attack "
        "signatures, then flags and classifies deviations across four fused signal sources:",
        styles["Body"],
    ))
    story.append(bullets([
        "<b>Rule engine</b> — deterministic checks (impossible travel, brute force, device fingerprint mismatch) that need no training data",
        "<b>Baseline profiling model</b> — an Isolation Forest over per-entity/entity-type behavioural deviation features",
        "<b>Sequence-aware detection model</b> — an LSTM autoencoder trained on normal event sequences; reconstruction error flags slow-building patterns a single event can't reveal",
        "<b>Attack classification</b> — an XGBoost multi-class classifier identifying which of 7 attack patterns an event resembles",
    ]))
    story.append(Paragraph(
        "These four scores are fused into a single 0-100 risk score, and every alert carries a "
        "plain-English explanation combining the rules that fired with the top SHAP-attributed "
        "features from the classifier.",
        styles["Body"],
    ))

    # ---------- 2. Assumptions ----------
    story.append(Paragraph("2. Behavioural Assumptions in the Synthetic Data", styles["H1"]))
    story.append(Paragraph(
        "Real intrusion/access-log datasets are scarce, outdated, privacy-restricted, or domain-specific, "
        "so the system generates its own synthetic access-log data using the schema and behaviour taxonomy below.",
        styles["Body"],
    ))

    schema_rows = [
        ["Field", "Description"],
        ["entity_id", "user_id / service_account_id / device_id"],
        ["entity_type", "user, service_account, or edge_device"],
        ["timestamp", "access or connection time"],
        ["source_ip / geo_location", "origin of the access"],
        ["resource_accessed", "file, endpoint, port, or device function"],
        ["auth_method", "password, token, certificate, or biometric"],
        ["session_duration", "length of connection (minutes)"],
        ["command_sequence", "ordered list of actions taken in the session"],
        ["device_fingerprint", "OS/firmware version + MAC-style identifier"],
        ["label / pattern", "ground truth, used for training/evaluation only"],
    ]
    story.append(heading_with(styles["H2"], "2.1 Data schema",
                               wrapped_table(schema_rows, col_widths=[1.9 * inch, 4.1 * inch])))
    story.append(Spacer(1, 10))

    story.append(Paragraph("2.2 Entity types and why they're modeled separately", styles["H2"]))
    story.append(Paragraph(
        "Three entity types are generated because they behave very differently, and a shared baseline "
        "would either over-flag service accounts or under-flag human users: <b>users</b> (bounded "
        "working hours, password/biometric auth), <b>service accounts</b> (near-24/7 activity, "
        "narrow fixed resource set, token/certificate auth — an odd-hour login is normal, not anomalous), "
        "and <b>edge devices</b> (periodic check-ins, certificate auth, fingerprint-sensitive).",
        styles["Body"],
    ))
    story.append(PageBreak())

    behaviour_rows = [
        ["Pattern", "Simulation approach", "Signal type"],
        ["normal_baseline", "Per-entity habitual pattern: regular hours, consistent geo, typical resources, sampled with noise", "Benign"],
        ["brute_force", "10-25 rapid failed-auth attempts from one source in a short window", "Anomaly"],
        ["impossible_travel", "Same entity logging in from geographically distant locations within an implausible time gap", "Anomaly"],
        ["credential_stuffing", "A small set of attacker IPs hitting many different entity_ids with failed auth", "Anomaly"],
        ["lateral_movement", "An entity accessing resources it has never touched before, in unusual breadth", "Anomaly"],
        ["device_spoofing", "Same device_id reappearing with a mismatched OS/MAC fingerprint", "Anomaly"],
        ["low_and_slow_exfiltration", "Gradual, off-hours access to one resource with session duration growing day over day", "Anomaly"],
        ["insider_drift", "Legitimate entity slowly expanding its own resource footprint — deliberately ambiguous", "Edge case"],
    ]
    story.append(heading_with(styles["H2"], "2.3 Behaviours simulated",
                               wrapped_table(behaviour_rows, col_widths=[1.5 * inch, 3.7 * inch, 0.8 * inch])))
    story.append(PageBreak())

    # ---------- 3. Evaluation methodology ----------
    story.append(Paragraph("3. Evaluation Methodology", styles["H1"]))
    story.append(Paragraph(
        "All three learned models (Isolation Forest, XGBoost, LSTM autoencoder) are fit only on an "
        "early time window and evaluated on a later, fully held-out time window they never saw during "
        "training. This mirrors real deployment (train on history, monitor new events) and avoids the "
        "credibility problem of reporting metrics on the same rows a model was fit on.",
        styles["Body"],
    ))
    story.append(wrapped_table([
        ["Split", "Row count", "Cutoff"],
        ["Train", str(split["train_rows"]), f"before {split['cutoff_date']}"],
        ["Eval (held out)", str(split["eval_rows"]), f"on/after {split['cutoff_date']}"],
    ], col_widths=[1.8 * inch, 1.4 * inch, 3 * inch]))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "An internal validation split (within the train window) is also computed as a sanity check, "
        "but the out-of-time report below is the number that should be quoted — it reflects "
        "generalization to unseen time periods, not memorization.",
        styles["Body"],
    ))

    # ---------- 4. Metrics ----------
    story.append(Paragraph("4. Results — Out-of-Time Classification Report", styles["H1"]))
    story.append(Paragraph(
        f"Per-class precision, recall, and F1 on the held-out eval window "
        f"({split['eval_rows']} events, never used in fitting any model):",
        styles["Body"],
    ))
    story.append(wrapped_table(
        build_classification_rows(oot),
        col_widths=[2.0 * inch, 1.0 * inch, 0.9 * inch, 0.7 * inch, 1.0 * inch],
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph("4.1 Alert-budget evaluation (false positive rate at realistic analyst capacity)", styles["H2"]))
    story.append(Paragraph(
        "An analyst cannot review every event — the evaluation criteria specifically ask for the "
        "false-positive rate at a realistic alert budget (e.g. top 1% of events by risk score). "
        "Both figures below are computed on the eval window only:",
        styles["Body"],
    ))
    story.append(wrapped_table([
        ["Alert budget", "Events reviewed", "Precision", "Recall of true anomalies"],
        ["Top 1%", str(ab["top_1pct"]["n_events"]), f"{ab['top_1pct']['precision']*100:.0f}%", f"{ab['top_1pct']['recall']*100:.1f}%"],
        ["Top 5%", str(ab["top_5pct"]["n_events"]), f"{ab['top_5pct']['precision']*100:.0f}%", f"{ab['top_5pct']['recall']*100:.1f}%"],
    ], col_widths=[1.3 * inch, 1.5 * inch, 1.3 * inch, 1.9 * inch]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        f"For reference only (not the headline figure): in-sample precision @ top 1% across all rows "
        f"including train is {ab['top_1pct_in_sample_reference']['precision']*100:.0f}%. Expect this to "
        f"look better than the out-of-time number — that gap is normal and is itself evidence the "
        f"out-of-time evaluation isn't just measuring memorization. Recall at a 1% budget can look "
        f"low mainly because a single brute-force or credential-stuffing episode contributes "
        f"dozens of individually-scored raw rows; a production deployment would correlate raw events "
        f"into one alert per episode before applying the analyst's review budget, which this project "
        f"does not implement (see Limitations, section 6).",
        styles["Body"],
    ))
    story.append(PageBreak())

    story.append(Paragraph("4.2 Cold-start coverage", styles["H2"]))
    story.append(Paragraph(
        f"{cs['rows']} events ({cs['pct_of_total']}% of all events) were scored using the "
        f"population-level baseline fallback because the entity involved had fewer than 5 prior "
        f"events of its own history at the time. These rows are explicitly tagged "
        f"(flag_coldstart) rather than silently absorbed into either baseline.",
        styles["Body"],
    ))

    # ---------- 5. Evaluation criteria mapping ----------
    criteria_rows = [
        ["Criterion", "Where addressed"],
        ["Detection accuracy on imbalanced labels", "Section 4 — out-of-time classification report; XGBoost trained with sample-weighting to counter class imbalance"],
        ["Correct anomaly-type classification", "Section 4 — per-class precision/recall across all 7 attack patterns"],
        ["False positive rate at a realistic alert budget", "Section 4.1 — precision @ top 1% and top 5%, out-of-time"],
        ["Explainability / analyst usability", "Every alert carries rule-based reasons plus top SHAP-attributed features; surfaced in the analyst dashboard"],
        ["Handling cold-start entities", "Section 4.2 — population-baseline fallback below 5 events of own history, explicitly flagged"],
        ["Handling concept drift", "Baselines recomputed daily from a trailing 14-day window per entity, not a static full-history baseline"],
        ["System design & scalability", "Section 7"],
        ["Report clarity", "This document"],
    ]
    story.append(heading_with(styles["H1"], "5. Mapping to Evaluation Criteria",
                               wrapped_table(criteria_rows, col_widths=[2.1 * inch, 4.2 * inch])))
    story.append(PageBreak())

    # ---------- 6. Limitations ----------
    story.append(Paragraph("6. Known Limitations", styles["H1"]))
    limitation_items = [
        "<b>Raw-event-level alerting understates recall.</b> A single multi-step attack episode (e.g. a brute-force burst) generates many correlated rows scored independently; episode-level correlation before applying an alert budget would likely raise the top-1% recall figure substantially, but is not implemented here.",
    ]
    if worst_class:
        limitation_items.append(
            f"<b>{worst_class} is currently the hardest class</b> "
            f"({oot[worst_class]['precision']:.0%} out-of-time precision, F1={worst_f1:.2f}). "
            "Its signature can be subtle at the individual-row level; where this is "
            "<i>low_and_slow_exfiltration</i>, the sequence model is doing real work the row-level "
            "classifier alone can't. Where this is a different class after a retrain, treat that as a "
            "signal to revisit feature engineering for that specific pattern before resubmitting."
        )
    limitation_items += [
        "<b>insider_drift precision reflects genuine ambiguity by design</b>, not a model defect — it is labeled 'Edge case' specifically because it should be hard to separate cleanly from legitimate behaviour change, and is meant to inform false-positive-budget tuning rather than be reliably caught.",
        "<b>Synthetic data throughout.</b> Patterns are cleaner and more separable than real traffic; a production deployment would need higher tolerance for label noise and messier behavioural baselines.",
        "<b>LSTM autoencoder is deliberately small</b> (2 layers, under 10k parameters) to train in well under a minute on this dataset size — appropriate for this timeline, not tuned for production scale or validated against larger sequence lengths.",
        "<b>Single-node, in-memory pipeline.</b> Pandas-based batch processing throughout; see Section 7 for the path to a streaming architecture.",
    ]
    story.append(bullets(limitation_items))

    # ---------- 7. Scalability ----------
    story.append(Paragraph("7. Scalability and Real-Time Feasibility", styles["H1"]))
    story.append(Paragraph(
        "This implementation is intentionally hackathon-scale (single-process, in-memory pandas). "
        "A production path to real-time streaming would look like:",
        styles["Body"],
    ))
    story.append(bullets([
        "Replace the CSV batch pipeline with a streaming ingestion layer (Kafka/Kinesis) feeding a feature store, computing features incrementally per event rather than recomputing in batch",
        "Keep the rule engine and Isolation Forest scoring synchronous on ingest — both are cheap enough (sub-millisecond per row) to run inline",
        "Run the LSTM autoencoder as a near-real-time async scorer, since it needs a buffered window of an entity's recent events rather than a single event in isolation",
        "Both the trailing-window baseline and the population cold-start fallback scale with entity count, not total event volume, so they remain cheap as log volume grows",
    ]))

    # ---------- 8. Conclusion ----------
    story.append(Paragraph("8. Conclusion", styles["H1"]))
    story.append(Paragraph(
        "The system demonstrates behavioural anomaly detection across all 7 requested attack "
        "patterns plus the insider-drift edge case, with explicit, auditable handling of cold-start "
        "entities and concept drift, real feature-attribution explainability, and metrics reported "
        "honestly on a held-out time window rather than in-sample. The weakest area is named "
        "directly in Section 6 rather than hidden, along with a concrete next step (episode-level "
        "alert correlation) that would likely improve the alert-budget recall figure.",
        styles["Body"],
    ))

    doc.build(story)
    print("Wrote report/anomaly_detection_report.pdf using models_saved/evaluation_metrics.json")


if __name__ == "__main__":
    build()
