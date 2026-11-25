"""Model monthly GPU cost optimization opportunities."""

import csv
from pathlib import Path

MONTHLY_HOURS = 100
HOURLY_RATE = 0.89
BASELINE = MONTHLY_HOURS * HOURLY_RATE

RECOMMENDATIONS = [
    ("0-30 days", "Terminate idle GPUs", 0.20, "Low"),
    ("0-30 days", "Use free development runtimes", 0.10, "Low"),
    ("0-30 days", "Eliminate redundant runs", 0.15, "Low"),
    ("1-6 months", "Optimize sequence packing", 0.20, "Medium"),
    ("1-6 months", "Optimize training kernels", 0.25, "Medium"),
    ("1-6 months", "Checkpoint-based recovery", 0.10, "Medium"),
    ("1-6 months", "Interruptible GPU instances", 0.40, "Medium"),
    ("6+ months", "Centralized GPU scheduling", 0.30, "High"),
    ("6+ months", "Automated GPU selection", 0.35, "High"),
    ("6+ months", "Shared artifact caching", 0.10, "Medium"),
    ("6+ months", "Budget-aware orchestration", 0.20, "High"),
]


def calculate_savings(baseline, fraction):
    if baseline < 0 or not 0 <= fraction <= 1:
        raise ValueError("Invalid savings assumptions")
    return round(baseline * fraction, 2)


def main():
    Path("results").mkdir(exist_ok=True)
    path = Path("results/monthly-savings.csv")

    with path.open("w", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow([
            "timeframe", "recommendation", "baseline_usd",
            "assumed_savings_pct", "monthly_savings_usd",
            "complexity", "evidence_type",
        ])

        for phase, action, fraction, complexity in RECOMMENDATIONS:
            writer.writerow([
                phase, action, BASELINE, fraction * 100,
                calculate_savings(BASELINE, fraction),
                complexity, "modeled",
            ])

    print(f"Monthly baseline: ${BASELINE:.2f}")
    print(f"Generated {len(RECOMMENDATIONS)} recommendations: {path}")


if __name__ == "__main__":
    main()
