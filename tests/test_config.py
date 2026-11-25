import csv
import json
import math
from pathlib import Path

import pytest
import yaml


@pytest.mark.parametrize("method", ["lora", "qlora"])
def test_training_config(method):
    config = yaml.safe_load(
        Path(f"configs/{method}.yaml").read_text()
    )
    assert config["model"]["method"] == method
    assert config["model"]["trust_remote_code"] is False
    assert config["lora"]["r"] > 0
    assert config["training"]["learning_rate"] > 0
    assert config["training"]["batch_size"] > 0
    assert config["training"]["gradient_checkpointing"] is True


def test_qlora_quantization():
    config = yaml.safe_load(
        Path("configs/qlora.yaml").read_text()
    )
    quant = config["model"]["quantization"]
    assert quant["bits"] == 4
    assert quant["type"] == "nf4"
    assert quant["double_quantization"] is True


def load_dataset(path):
    return [
        json.loads(line)["messages"]
        for line in Path(path).read_text().splitlines()
    ]


def test_dataset_integrity():
    train = load_dataset("data/train.jsonl")
    evaluation = load_dataset("data/eval.jsonl")

    assert len(train) == 80
    assert len(evaluation) == 10

    for messages in train + evaluation:
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert all(message["content"] for message in messages)

    assert {x[0]["content"] for x in train}.isdisjoint(
        {x[0]["content"] for x in evaluation}
    )


def test_training_evidence():
    metrics = json.loads(
        Path("results/lora-metrics.json").read_text()
    )
    assert metrics["method"] == "lora"
    assert metrics["training_steps"] == 20
    assert math.isfinite(metrics["training_loss"])
    assert metrics["runtime_seconds"] > 0
    assert 0 < metrics["trainable_parameters"] < metrics["total_parameters"]
    assert 0 < metrics["peak_allocated_vram_gib"] <= (
        metrics["peak_reserved_vram_gib"]
    )


def test_evaluation_evidence():
    report = json.loads(
        Path("results/lora-evaluation.json").read_text()
    )
    assert report["adapter_reload"] == "passed"
    assert report["samples"] == len(report["responses"]) == 10

    for method in ("baseline", "adapted"):
        rate = sum(
            row[f"{method}_exact_match"]
            for row in report["responses"]
        ) / report["samples"]

        assert report[f"{method}_exact_match_rate"] == pytest.approx(rate)


@pytest.mark.parametrize("filename,count", [
    ("qlora-cost-comparison.csv", 30),
    ("monthly-savings.csv", 11),
])
def test_modeled_results(filename, count):
    with Path(f"results/{filename}").open() as source:
        rows = list(csv.DictReader(source))

    assert len(rows) == count
    assert all(row["evidence_type"] == "modeled" for row in rows)
