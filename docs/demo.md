# Demo script (3 minutes)

**Framing (0:00-0:30):** "Trading desks re-approve brokers and data
vendors every quarter on spreadsheets. I built a governed AI agent that
does this in minutes, with a full audit trail — the same pattern banks
are now buying from Dataiku + Snowflake for FSI risk scoring."

**Data & model (0:30-1:15):** Show `vendor_mandates.csv` — walk through
one row. Run `train_model.py` live or show the printed metrics (87.9%
accuracy / 0.965 AUC). Show `feature_importances.csv` — "compliance
flags and data quality dominate the decision, which matches how a real
committee would reason."

**Agent & memo (1:15-2:15):** Run `memo_agent.py`. Read one APPROVE and
one DECLINE memo aloud. Point out: (1) plain-English driver
explanation, (2) explicit risk flags, (3) mandatory "review before
sign-off" line — this is a recommendation engine, not an autonomous
approver. Open `agent_audit_log.jsonl` — "every single decision is
logged with its full input, so compliance can reconstruct any
recommendation six months later."

**Product framing (2:15-3:00):** Show `docs/PRD.md` roadmap slide.
"v1 is this repo. v2 adds real anonymized data and human-override
capture. v3 wires this into the actual committee workflow tool with
enterprise guardrails — cost caps, PII redaction, sign-off gates."
Close: "This is the same type of engagement I delivered at Natixis on
broker/data-provider selection — here it's public, synthetic, and
PM-narrated so you can see both the build and the thinking."
