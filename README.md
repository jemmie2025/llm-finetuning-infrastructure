# LLM Fine-Tuning Infrastructure

Engineering assessment of LoRA, QLoRA and full fine-tuning,
covering GPU efficiency, framework selection, FinOps and
operational risk.

## Technology Stack

- Training: PyTorch, Transformers, TRL, PEFT and bitsandbytes.
- GPU environment: Google Colab (NVIDIA Tesla T4).
- Development: WSL2 (Ubuntu).
- Configuration: YAML.
- Testing and CI: pytest and GitHub Actions.

## Models

| Model | Method | Status |
|---|---|---|
| Qwen3-1.7B | LoRA | Trained and evaluated |
| Qwen2.5-7B-Instruct | QLoRA | Configured and cost-modeled |

## Validated Results

| Metric | Result |
|---|---:|
| Execution environment | Google Colab |
| GPU | NVIDIA Tesla T4 |
| Training steps | 20 |
| Trainable parameters | 0.1863% |
| Training runtime | 43.94 s |
| Peak allocated VRAM | 3.323 GiB |
| Training throughput | 1.848 samples/s |
| Strict exact match (10 samples) | 0% to 30% |
| Adapter reload | Passed |
| Automated tests | 26 passed |

## Cost Analysis

- 30 modeled QLoRA cost scenarios across five GPUs.
- 11 monthly cost-saving recommendations.
- Savings, throughput and runtime assumptions documented.

## Local Validation

```bash
python -m pip install -r requirements-dev.txt
make check
sha256sum -c results/SHA256SUMS
```

## GPU Execution

LoRA training and evaluation were performed on Google Colab
using an NVIDIA Tesla T4 GPU.

Reproduction requires a compatible CUDA-enabled PyTorch
environment.

```bash
python -m pip install -r requirements.txt
python scripts/train.py --config configs/lora.yaml --dataset data/train.jsonl
python scripts/evaluate.py --adapter outputs/lora --dataset data/eval.jsonl
```

Checkpoint resume and revised precision selection require
GPU revalidation.

## Documentation

- [Technical report](docs/report.md)
- [LoRA configuration](configs/lora.yaml)
- [QLoRA configuration](configs/qlora.yaml)
- [Execution evidence](results/)

## Limitations

LoRA training and evaluation were GPU-validated.
QLoRA costs are modeled. Full fine-tuning, distributed
training and production recovery remain untested.

Known dependency vulnerabilities restrict production adoption.
