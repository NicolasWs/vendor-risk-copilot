"""
Synthetic broker / data-vendor dataset generator.

Mirrors a real-world "broker & data provider selection" exercise for a
capital-markets desk: each row is one (vendor, mandate) evaluation with
cost, execution quality, coverage, and compliance signals.

No real client/vendor data is used anywhere in this project.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(1337)

VENDOR_NAMES = [
    "Meridian Capital Markets", "Northfield Data Services", "Askew & Pryce",
    "Vantor Securities", "ClearPath Analytics", "Solace Prime Brokerage",
    "Ironview Trading", "Baskerville Markets", "Quillon Data", "Harrowgate Execution",
    "Fenwick Liquidity Partners", "Oberon Risk Analytics",
]

ASSET_CLASSES = ["Equities", "Fixed Income", "FX", "Derivatives", "Commodities"]
REGIONS = ["EMEA", "APAC", "Americas"]


def generate(n_rows: int = 1200, seed: int = 1337) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_rows):
        vendor = rng.choice(VENDOR_NAMES)
        asset_class = rng.choice(ASSET_CLASSES)
        region = rng.choice(REGIONS)

        # Vendor "true quality" latent factor drives correlated metrics
        base_quality = rng.normal(0, 1)

        cost_bps = max(0.1, rng.normal(8 - base_quality * 1.5, 2.0))
        latency_ms = max(1, rng.normal(120 - base_quality * 20, 30))
        fill_rate = np.clip(rng.normal(0.92 + base_quality * 0.03, 0.04), 0.5, 0.999)
        coverage_pct = np.clip(rng.normal(0.8 + base_quality * 0.05, 0.1), 0.2, 1.0)
        compliance_flags_12m = max(0, int(rng.poisson(lam=max(0.05, 1.2 - base_quality))))
        data_quality_score = np.clip(rng.normal(75 + base_quality * 10, 10), 0, 100)
        relationship_years = max(0, rng.normal(6 + base_quality, 3))
        sla_breach_rate = np.clip(rng.normal(0.05 - base_quality * 0.01, 0.02), 0, 0.3)

        # Label: would an ops/trading committee approve this vendor for the mandate?
        approval_score = (
            -0.35 * cost_bps
            - 0.01 * latency_ms
            + 25 * fill_rate
            + 8 * coverage_pct
            - 3.0 * compliance_flags_12m
            + 0.06 * data_quality_score
            + 0.15 * relationship_years
            - 20 * sla_breach_rate
            + rng.normal(0, 2.5)
        )
        approved = int(approval_score > np.median([approval_score]) or approval_score > 15)
        # recompute with a fixed threshold learned empirically below instead
        rows.append(dict(
            vendor=vendor,
            asset_class=asset_class,
            region=region,
            cost_bps=round(cost_bps, 2),
            latency_ms=round(latency_ms, 1),
            fill_rate=round(fill_rate, 4),
            coverage_pct=round(coverage_pct, 4),
            compliance_flags_12m=compliance_flags_12m,
            data_quality_score=round(data_quality_score, 1),
            relationship_years=round(relationship_years, 1),
            sla_breach_rate=round(sla_breach_rate, 4),
            _approval_score=approval_score,
        ))

    df = pd.DataFrame(rows)
    threshold = df["_approval_score"].quantile(0.55)  # ~45% approved, realistic selectivity
    df["approved"] = (df["_approval_score"] > threshold).astype(int)
    df = df.drop(columns=["_approval_score"])
    return df


if __name__ == "__main__":
    out = generate()
    out.to_csv("data/vendor_mandates.csv", index=False)
    print(f"Wrote {len(out)} rows to data/vendor_mandates.csv")
    print(out["approved"].value_counts(normalize=True))
