# PRD: Vendor & Risk Decisioning Copilot

## Problem
Trading-ops and compliance committees at asset managers / banks
re-approve brokers and market-data vendors on manual spreadsheet
reviews. This is slow (days per cycle), inconsistent across reviewers,
and leaves no structured audit trail for regulators — a growing gap as
AI-Act / SR 11-7-style model-governance expectations extend to vendor
risk decisions.

## Users
- **Primary:** Trading-ops / vendor-risk analysts who prepare the
  committee pack each quarter.
- **Secondary:** Compliance officers who need an auditable decision
  trail; desk heads who want faster, more consistent vendor reviews.

## Solution (v1 scope, this repo)
A model scores each vendor/mandate pair on approval likelihood from
historical performance data (cost, execution quality, coverage,
compliance history). An agent layer converts the score into a
plain-English memo with explicit decision drivers and risk flags, and
logs every decision for audit.

## Success metrics
- **Time-to-decision**: committee pack prep time, target -50%
  (manual spreadsheet collation → automated scored shortlist)
- **Consistency**: variance in approval outcome for similar vendor
  profiles across reviewers, target reduction via documented drivers
- **Audit coverage**: 100% of recommendations backed by a logged,
  reproducible input/output record
- **Override rate**: % of agent recommendations the committee
  overrides — tracked as a model-trust signal, not a failure metric,
  and fed back into retraining

## Roadmap
- **v1 (this repo):** batch scoring + memo generation + audit log on
  synthetic data; Dataiku DSS equivalent flow for no-code stakeholders.
- **v2:** real (anonymized/masked) client data; SHAP-based local
  explanations instead of global feature importance; human-in-the-loop
  override capture feeding a feedback dataset.
- **v3:** live agent integrated into the committee workflow tool
  (e.g. triggered from a ticket/Slack command), with LLM-Mesh-style
  model routing and guardrails (cost caps, PII redaction, approval
  sign-off gates) matching enterprise governance patterns (Dataiku
  GenAI Registry / Safe Guard, or equivalent in-house controls).

## Out of scope (v1)
- Real-time / streaming scoring (batch only)
- Multi-agent orchestration (single scoring model + single memo agent)
- Production authentication/authorization (demo-only, local data)

## Risks & mitigations
- **Model opacity** → mitigated by feature-importance-driven memos and
  mandatory "review before sign-off" language in every recommendation.
- **Synthetic-data realism** → latent-factor generator correlates
  cost/quality/compliance the way real vendor data behaves, documented
  in `generate_data.py`; flagged in README as synthetic throughout.
- **Over-automation risk** → v1 intentionally produces a *recommendation*,
  not an automatic approval; override rate tracked as a core metric from
  day one.
