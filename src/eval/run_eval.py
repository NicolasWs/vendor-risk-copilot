"""
Agent evaluation harness — deterministic, offline, no API key required.

Mirrors the metric categories Dataiku's native "Evaluate Agent" recipe
and open-source frameworks (RAGAS, DeepEval) compute, implemented
directly so the project is runnable anywhere and auditable line by
line:

  - Outcome accuracy  : does recommendation match the golden label?
  - Tool-call recall  : did the agent fire every expected flag/tool?
  - Tool-call precision: did the agent avoid firing flags it shouldn't?
  - Trajectory exact match: did the full set of tool calls match exactly?

Run with: python src/eval/run_eval.py
Produces: eval_data/eval_scorecard.json + a pass/fail gate on stdout
(non-zero exit code if the regression gate fails — CI-friendly).
"""
import json
import sys

import joblib
import pandas as pd

sys.path.insert(0, ".")
from src.agent.memo_agent import score_and_render  # noqa: E402
from src.eval.golden_set import GOLDEN_CASES  # noqa: E402

MODEL_PATH = "data/model.joblib"
IMPORTANCE_PATH = "data/feature_importances.csv"
SCORECARD_PATH = "eval_data/eval_scorecard.json"

# Regression gate — a run fails this gate if any metric drops below these
# floors. These thresholds are what you'd wire into a CI check or a
# Dataiku "Status check" on the GenAI Evaluation Store.
GATES = {
    "outcome_accuracy": 0.85,
    "tool_recall": 0.85,
    "tool_precision": 0.85,
}


def precision_recall(expected: list, actual: list) -> tuple:
    expected_set, actual_set = set(expected), set(actual)
    tp = len(expected_set & actual_set)
    fp = len(actual_set - expected_set)
    fn = len(expected_set - actual_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    return precision, recall


def run():
    model = joblib.load(MODEL_PATH)
    importance_df = pd.read_csv(IMPORTANCE_PATH)

    rows = []
    for case in GOLDEN_CASES:
        row = pd.Series(case["row"])
        result = score_and_render(row, model, importance_df)

        outcome_correct = bool(result.recommendation == case["expected_recommendation"])
        precision, recall = precision_recall(case["expected_tool_calls"], result.tool_calls)
        trajectory_exact_match = bool(set(result.tool_calls) == set(case["expected_tool_calls"]))

        rows.append({
            "case_id": case["id"],
            "vendor": case["row"]["vendor"],
            "expected_recommendation": case["expected_recommendation"],
            "actual_recommendation": result.recommendation,
            "outcome_correct": outcome_correct,
            "expected_tool_calls": case["expected_tool_calls"],
            "actual_tool_calls": result.tool_calls,
            "tool_precision": round(precision, 4),
            "tool_recall": round(recall, 4),
            "trajectory_exact_match": trajectory_exact_match,
            "notes": case["notes"],
        })

    df = pd.DataFrame(rows)
    aggregate = {
        "n_cases": len(df),
        "outcome_accuracy": round(float(df["outcome_correct"].mean()), 4),
        "tool_precision": round(float(df["tool_precision"].mean()), 4),
        "tool_recall": round(float(df["tool_recall"].mean()), 4),
        "trajectory_exact_match_rate": round(float(df["trajectory_exact_match"].mean()), 4),
    }

    gate_results = {
        metric: {
            "value": aggregate[metric],
            "floor": floor,
            "pass": aggregate[metric] >= floor,
        }
        for metric, floor in GATES.items()
    }
    all_pass = all(g["pass"] for g in gate_results.values())

    scorecard = {
        "aggregate_metrics": aggregate,
        "regression_gate": gate_results,
        "gate_pass": all_pass,
        "case_results": rows,
    }

    with open(SCORECARD_PATH, "w") as f:
        json.dump(scorecard, f, indent=2)

    print(json.dumps(aggregate, indent=2))
    print("\nRegression gate:", "PASS" if all_pass else "FAIL")
    for metric, g in gate_results.items():
        status = "OK " if g["pass"] else "FAIL"
        print(f"  [{status}] {metric}: {g['value']} (floor {g['floor']})")

    failing_cases = df[~df["outcome_correct"] | ~df["trajectory_exact_match"]]
    if not failing_cases.empty:
        print("\nFailing cases:")
        print(failing_cases[["case_id", "expected_recommendation", "actual_recommendation",
                              "expected_tool_calls", "actual_tool_calls"]].to_string(index=False))

    print(f"\nScorecard written -> {SCORECARD_PATH}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    run()
