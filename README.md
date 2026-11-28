# LLM Fine-Tuning Smoke Tests and QLoRA Cost Modeling

GPU-based validation of LoRA, QLoRA and full fine-tuning,
comparative assessment of TRL and Unsloth, and 7B QLoRA
infrastructure cost estimation.

## Technology Stack

| Component | TRL repeat environment | Unsloth environment |
|---|---|---|
| PyTorch | 2.14.1+cu130 | 2.10.0+cu126 |
| Transformers | 4.52.4 | 5.17.0 |
| TRL | 0.19.0 | 1.15.0 |
| PEFT | 0.15.2 | 0.22.0 |
| bitsandbytes | 0.49.0 | 0.50.2 |
| Accelerate | 1.7.0 | 1.16.0 |
| Datasets | 3.6.0 | 4.8.5 |
| Unsloth | Not used | 2026.11.10 |
| GPU | Tesla T4 | Tesla T4 |

PyTorch was supplied by the respective GPU environments.
The historical LoRA run did not retain a complete dependency lock.

## GPU Smoke Tests

Qwen2.5-0.5B-Instruct, 80 training samples and 20 steps.

| Method | Runtime | Peak allocated VRAM |
|---|---:|---:|
| Full fine-tuning | 30.03 s | 9.230 GiB |
| LoRA | 27.83 s | 1.040 GiB |
| QLoRA | 40.68 s | 1.050 GiB |
| Unsloth LoRA | 38.54 s | 0.992 GiB |

Additional LoRA and 4-bit QLoRA smoke tests were
completed using Qwen3-1.7B.

These single-run experiments validate training execution
and resource consumption, not production model quality.

## Model Evaluation

The Qwen3-1.7B LoRA evaluation used 100 synthetic held-out samples.

| Metric | Result |
|---|---:|
| Baseline exact match | 39% |
| LoRA-adapted exact match | 50% |
| Improvement | 11 percentage points |

Evaluation used greedy decoding and strict exact matching.

## Cost Modeling

- 30 estimated 7B QLoRA training cost scenarios across five GPUs.
- 11 independent infrastructure cost-saving recommendations.
- Documented throughput, pricing and runtime assumptions.

## Security Assessment

- TRL dependency audit: 44 findings across five packages,
  representing 37 distinct advisories.
- Unsloth environment (Google Colab): seven findings across
  three packages, representing five distinct advisories.
- PyTorch and Torchvision were excluded from the Unsloth audit.

Advisory identifiers, severity ratings and reported fixes
are documented in the technical report.

## Limitations

- Experiments demonstrate functionality, not convergence.
- Framework comparisons are limited by differing dependency
  versions and training optimizations.
- Unsloth reported two non-finite gradient norms.
- The synthetic evaluation does not establish generalization.
- 7B QLoRA costs are estimated, not experimentally measured.
- Reported security fixes have not been applied.

## Validation

    python -m pip install -r requirements-dev.txt
    make check
    sha256sum -c results/SHA256SUMS

## Documentation

- [Technical report](docs/report.md)
- [Training configurations](configs/)
- [Experiment source](experiment-source/)
- [Experimental results](results/)
