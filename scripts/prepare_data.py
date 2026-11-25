"""Generate deterministic synthetic DevOps instruction datasets."""

import json
from pathlib import Path

TRAIN = [
    ("Check disk utilization", "df -h"),
    ("Check available memory", "free -h"),
    ("List running Docker containers", "docker ps"),
    ("List Docker images", "docker images"),
    ("List Kubernetes pods", "kubectl get pods"),
    ("List Kubernetes nodes", "kubectl get nodes"),
    ("Validate Terraform configuration", "terraform validate"),
    ("Check Git working tree status", "git status"),
    ("Check listening network sockets", "ss -tuln"),
    ("Check GPU utilization", "nvidia-smi"),
]

EVAL = [
    ("Check current directory", "pwd"),
    ("List Kubernetes services", "kubectl get services"),
    ("Display Git commit history", "git log --oneline"),
    ("Check Docker disk usage", "docker system df"),
    ("Check system uptime", "uptime"),
]

TEMPLATES = [
    "{}.",
    "Which command can I use to {}?",
    "Provide the CLI command to {}.",
    "How do I {}?",
    "Give the terminal command to {}.",
    "What command should an engineer use to {}?",
    "Return only the command to {}.",
    "Identify the command required to {}.",
]


def write_dataset(path, pairs, templates):
    records = [
        {
            "messages": [
                {"role": "user", "content": template.format(task)},
                {"role": "assistant", "content": answer},
            ]
        }
        for task, answer in pairs
        for template in templates
    ]

    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    print(f"{path}: {len(records)} samples")


if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    write_dataset(Path("data/train.jsonl"), TRAIN, TEMPLATES)
    write_dataset(Path("data/eval.jsonl"), EVAL, TEMPLATES[:2])
