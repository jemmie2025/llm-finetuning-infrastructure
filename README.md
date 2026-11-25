# LLM Fine-Tuning Infrastructure

Engineering assessment of LoRA, QLoRA and full fine-tuning,
covering GPU efficiency, framework selection, cost modeling
and operational risk.

## Stack

PyTorch, Transformers, TRL, PEFT, bitsandbytes, YAML,
pytest and GitHub Actions.

## Validated Results

| Metric | Result |
|---|---:|
| Model | Qwen3-1.7B |
| GPU | Tesla T4 |
| Trainable parameters | 0.1863% |
| Training runtime | 43.94 s |
| Peak allocated VRAM | 3.323 GiB |
| Exact match | 0% to 30% |
| Adapter reload | Passed |
| Automated tests | 26 passed |
| QLoRA cost scenarios | 30 (modeled) |
| FinOps recommendations | 11 (modeled) |

## Local Validation

python -m pip install -r requirements-dev.txt
make check

## GPU Execution

Run on a CUDA-compatible environment.

    python -m pip install -r requirements.txt
    python scripts/train.py --config configs/lora.yaml --dataset data/train.jsonl
    python scripts/evaluate.py --adapter outputs/lora --dataset data/eval.jsonl

Use `--resume` with a compatible checkpoint for recovery.
Revised precision and resume support require GPU revalidation.

## Documentation

- [Technical report](docs/report.md)
- [Training configuration](configs/lora.yaml)
- [QLoRA configuration](configs/qlora.yaml)
- [Execution evidence](results/)

## Scope

LoRA training and evaluation were executed on a real GPU.
QLoRA costs are modeled; full fine-tuning and distributed
training remain untested.
