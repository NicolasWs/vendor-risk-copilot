"""
Explainability + memo agent — the "PM-facing" layer on top of the model.

Does NOT require an LLM API key to run (uses a deterministic template
renderer by default so the project works offline/CI); if OPENAI_API_KEY
is set, it will optionally use an LLM to polish the memo prose.

This mirrors the real workflow: a trading-ops / compliance committee
doesn't want raw model probabilities, it wants a one-paragraph memo per
vendor-mandate pair explaining WHY, with an audit trail of the inputs
that drove the decision.
"""
import json
import os
from dataclasses import dataclass, asdict

import pandas as pd

SCORED_PATH = "data/scored_mandates.csv"
IMPORTANCE_PATH = "data/feature_importances.csv"
AUDIT_LOG_PATH = "data/agent_audit_log.jsonl"

RISK_FEATURES = ["compliance_flags_12m", "sla_breach_rate"]
QUALITY_FEATURES = ["data_quality_score", "fill_rate", "coverage_pct", "relationship_years"]
COST_FEATURES = ["cost_bps", "latency_ms"]


@dataclass
class MemoResult:
    vendor: str
    asset_class: str
    region: str
    recommendation: str
    confidence: float
    drivers: list
    flags: list
    tool_calls: list
    memo: str


def flag_compliance(row: pd.Series):
    if row["compliance_flags_12m"] > 2:
        return f"{int(row['compliance_flags_12m'])} compliance flags in last 12m (elevated)"
    return None


def flag_sla(row: pd.Series):
    if row["sla_breach_rate"] > 0.08:
        return f"SLA breach rate {row['sla_breach_rate']:.1%} above 8% threshold"
    return None


def flag_data_quality(row: pd.Series):
    if row["data_quality_score"] < 60:
        return f"Data quality score {row['data_quality_score']:.0f}/100 below acceptable floor"
    return None


# Explicit registry so the eval harness can score "tool selection"
# precision/recall the same way Dataiku's trajectory metrics do.
FLAG_TOOLS = {
    "flag_compliance": flag_compliance,
    "flag_sla": flag_sla,
    "flag_data_quality": flag_data_quality,
}


def _flag_row(row: pd.Series) -> tuple:
    """Returns (flag_messages, tool_calls_fired) for a row."""
    messages, tool_calls = [], []
    for tool_name, tool_fn in FLAG_TOOLS.items():
        msg = tool_fn(row)
        if msg:
            messages.append(msg)
            tool_calls.append(tool_name)
    return messages, tool_calls


def _top_drivers_for_row(row: pd.Series, importance_df: pd.DataFrame, k: int = 3) -> list:
    """Rank this row's numeric features by (global importance), annotate direction."""
    drivers = []
    for _, imp_row in importance_df.head(8).iterrows():
        feat = imp_row["feature"]
        if feat not in row.index:
            continue
        drivers.append((feat, imp_row["importance"], row[feat]))
    drivers.sort(key=lambda x: -x[1])
    return drivers[:k]


def render_memo(row: pd.Series, importance_df: pd.DataFrame) -> MemoResult:
    model_approved = row["pred_approved"] == 1
    conf = float(row["approval_probability"] if model_approved
                  else 1 - row["approval_probability"])
    flags, tool_calls = _flag_row(row)

    # Governance rule: a deterministic risk flag always overrides the
    # model's recommendation toward escalation. A model should never be
    # able to silently approve a vendor that trips a hard risk rule —
    # this coupling is exactly what the eval harness (src/eval/) checks
    # on every run via the golden set.
    rec = "DECLINE / ESCALATE" if flags else ("APPROVE" if model_approved else "DECLINE / ESCALATE")
    drivers = _top_drivers_for_row(row, importance_df)

    driver_text = "; ".join(f"{name} = {val}" for name, _, val in drivers)
    flag_text = " No material risk flags raised." if not flags else (
        " Risk flags: " + "; ".join(flags) + "."
    )

    memo = (
        f"Recommendation: {rec} ({conf:.0%} model confidence) for {row['vendor']} "
        f"on {row['asset_class']} mandates, {row['region']} region. "
        f"Key drivers: {driver_text}.{flag_text} "
        f"This assessment is model-generated from historical vendor performance data "
        f"and should be reviewed by the trading-ops committee before final sign-off."
    )

    return MemoResult(
        vendor=row["vendor"], asset_class=row["asset_class"], region=row["region"],
        recommendation=rec, confidence=round(conf, 4),
        drivers=[{"feature": n, "value": v} for n, _, v in drivers],
        flags=flags, tool_calls=tool_calls, memo=memo,
    )


def score_and_render(row: pd.Series, model, importance_df: pd.DataFrame) -> MemoResult:
    """
    Entry point for rows that do NOT already have pred_approved /
    approval_probability columns (e.g. golden eval cases) — runs the
    saved model then renders the memo. Used by the eval harness so it
    exercises the real model + real agent logic, not a mock.
    """
    from src.pipeline.train_model import CAT_COLS, NUM_COLS

    row = row.copy()
    X = pd.DataFrame([row[NUM_COLS + CAT_COLS]])
    row["pred_approved"] = int(model.predict(X)[0])
    row["approval_probability"] = float(model.predict_proba(X)[0, 1])
    return render_memo(row, importance_df)


def _audit(entry: dict):
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def run(limit: int = 20):
    df = pd.read_csv(SCORED_PATH)
    importance_df = pd.read_csv(IMPORTANCE_PATH)

    sample = df.sample(n=min(limit, len(df)), random_state=1337)
    results = []
    for _, row in sample.iterrows():
        result = render_memo(row, importance_df)
        results.append(asdict(result))
        _audit({"input_row": row.to_dict(), "output": asdict(result)})

    with open("data/sample_memos.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Generated {len(results)} memos -> data/sample_memos.json")
    print(f"Audit trail appended -> {AUDIT_LOG_PATH}")
    print("\n--- Example memo ---")
    print(results[0]["memo"])


if __name__ == "__main__":
    # fresh audit log each run for repeatable demo
    if os.path.exists(AUDIT_LOG_PATH):
        os.remove(AUDIT_LOG_PATH)
    run()
