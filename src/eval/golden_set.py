"""
Golden test set for the vendor/risk memo agent.

Each case defines a synthetic input row + the EXPECTED agent behavior,
mirroring the shape of a Dataiku "Evaluate Agent" recipe golden dataset
(query / expected output / expected tool calls / reference), adapted to
this agent's deterministic scoring context: input row -> recommendation,
flags, drivers.

"Tool calls" here map to the checks the memo agent is expected to run
on each row (flag_compliance, flag_sla, flag_data_quality) — kept as an
explicit list so the eval harness can score tool-selection precision/
recall the same way Dataiku's trajectory metrics do.
"""

GOLDEN_CASES = [
    {
        "id": "clean_strong_approve",
        "row": dict(
            vendor="ClearPath Analytics", asset_class="Equities", region="EMEA",
            cost_bps=3.2, latency_ms=45, fill_rate=0.98, coverage_pct=0.95,
            compliance_flags_12m=0, data_quality_score=92, relationship_years=9,
            sla_breach_rate=0.01,
        ),
        "expected_recommendation": "APPROVE",
        "expected_tool_calls": [],
        "notes": "Clean profile across every metric; no flags expected.",
    },
    {
        "id": "compliance_breach_decline",
        "row": dict(
            vendor="Ironview Trading", asset_class="Derivatives", region="APAC",
            cost_bps=6.0, latency_ms=90, fill_rate=0.9, coverage_pct=0.8,
            compliance_flags_12m=4, data_quality_score=70, relationship_years=3,
            sla_breach_rate=0.03,
        ),
        "expected_recommendation": "DECLINE / ESCALATE",
        "expected_tool_calls": ["flag_compliance"],
        "notes": "4 compliance flags alone should drive a decline + flag.",
    },
    {
        "id": "sla_breach_flag",
        "row": dict(
            vendor="Vantor Securities", asset_class="FX", region="Americas",
            cost_bps=4.5, latency_ms=70, fill_rate=0.93, coverage_pct=0.85,
            compliance_flags_12m=0, data_quality_score=80, relationship_years=5,
            sla_breach_rate=0.12,
        ),
        "expected_recommendation": "DECLINE / ESCALATE",
        "expected_tool_calls": ["flag_sla"],
        "notes": "SLA breach rate 12% > 8% threshold should be flagged.",
    },
    {
        "id": "poor_data_quality_flag",
        "row": dict(
            vendor="Harrowgate Execution", asset_class="Fixed Income", region="EMEA",
            cost_bps=5.0, latency_ms=100, fill_rate=0.9, coverage_pct=0.75,
            compliance_flags_12m=1, data_quality_score=42, relationship_years=4,
            sla_breach_rate=0.04,
        ),
        "expected_recommendation": "DECLINE / ESCALATE",
        "expected_tool_calls": ["flag_data_quality"],
        "notes": "Data quality 42 < 60 floor should be flagged.",
    },
    {
        "id": "multi_flag_worst_case",
        "row": dict(
            vendor="Baskerville Markets", asset_class="Commodities", region="APAC",
            cost_bps=9.0, latency_ms=180, fill_rate=0.7, coverage_pct=0.5,
            compliance_flags_12m=5, data_quality_score=35, relationship_years=1,
            sla_breach_rate=0.2,
        ),
        "expected_recommendation": "DECLINE / ESCALATE",
        "expected_tool_calls": ["flag_compliance", "flag_sla", "flag_data_quality"],
        "notes": "Worst-case vendor: all three flags should fire together.",
    },
    {
        "id": "borderline_moderate",
        "row": dict(
            vendor="Fenwick Liquidity Partners", asset_class="Equities", region="EMEA",
            cost_bps=4.0, latency_ms=60, fill_rate=0.94, coverage_pct=0.82,
            compliance_flags_12m=1, data_quality_score=68, relationship_years=6,
            sla_breach_rate=0.05,
        ),
        "expected_recommendation": "APPROVE",
        "expected_tool_calls": [],
        "notes": "Below all flag thresholds; should pass despite 1 historical flag.",
    },
]
