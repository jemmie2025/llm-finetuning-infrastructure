"""Estimate single-GPU QLoRA compute costs."""

import csv
import math
from pathlib import Path


def estimate_cost(tokens, epochs, throughput, rate, overhead=0.15):
    values = (tokens, epochs, throughput, rate, overhead)

    if not all(math.isfinite(value) for value in values):
        raise ValueError("Inputs must be finite")

    if min(tokens, epochs, throughput, rate) <= 0:
        raise ValueError("Workload, throughput and rate must be positive")

    if overhead < 0:
        raise ValueError("Overhead cannot be negative")

    hours = tokens * epochs / throughput / 3600
    billed_hours = hours * (1 + overhead)

    return billed_hours, billed_hours * rate


def main():
    Path("results").mkdir(exist_ok=True)
    rows = []

    with open("data/gpu-pricing.csv") as source:
        for gpu in csv.DictReader(source):
            for epochs in (1, 3):
                for overhead in (0.0, 0.15, 0.30):
                    hours, cost = estimate_cost(
                        tokens=10_000_000,
                        epochs=epochs,
                        throughput=float(
                            gpu["assumed_tokens_per_second"]
                        ),
                        rate=float(gpu["usd_per_hour"]),
                        overhead=overhead,
                    )

                    rows.append({
                        **gpu,
                        "dataset_tokens": 10_000_000,
                        "epochs": epochs,
                        "overhead": overhead,
                        "estimated_hours": round(hours, 4),
                        "estimated_cost_usd": round(cost, 4),
                        "evidence_type": "modeled",
                    })

    path = Path("results/qlora-cost-comparison.csv")

    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} cost scenarios: {path}")


if __name__ == "__main__":
    main()
