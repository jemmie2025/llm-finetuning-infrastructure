# LLM Fine-Tuning Infrastructure Assessment

## Method Comparison

| Method | Estimated 7B VRAM | Trade-off |
|---|---|---|
| LoRA | 18-32 GiB | Adapter training; reduced optimizer memory |
| QLoRA | 8-16 GiB | 4-bit base; lower memory requirements |
| Full | 120-160 GiB | All parameters updated; high compute cost |

Memory estimates depend on batching, sequence length and optimizer.

## Decision Matrix

| Goal | Method | Tool |
|---|---|---|
| Task adaptation | LoRA | HF TRL + PEFT |
| GPU efficiency | QLoRA | Unsloth |
| Repeatable training | LoRA/QLoRA | Axolotl |
| Distributed training | Full/LoRA | Axolotl |
| PyTorch research | LoRA/Full | torchtune |

### Framework Limitations

| Framework | Limitation |
|---|---|
| TRL | Requires explicit integration and dependency management |
| Axolotl | Configuration and GPU compatibility complexity |
| Unsloth | Hardware, model and optimized kernel compatibility |
| torchtune | Active development discontinued in 2025 |

Only TRL and PEFT were execution-validated.

## Validated Experiment

Qwen3-1.7B trained on Tesla T4 using PyTorch, TRL and PEFT.

| Metric | Result |
|---|---:|
| Training steps | 20 |
| Trainable parameters | 0.1863% |
| Runtime | 43.94 s |
| Peak allocated VRAM | 3.323 GiB |
| Throughput | 1.848 samples/s |
| Training loss | 4.3719 |
| Exact match | 0% to 30% |
| Adapter reload | Passed |

Configuration: [LoRA](../configs/lora.yaml).
Evidence: [training](../results/lora-metrics.json),
[evaluation](../results/lora-evaluation.json).

Reported BF16 does not imply native T4 BF16 acceleration.
Results validate functionality, not production model quality.

## Cost Analysis

Qwen2.5-7B QLoRA: NF4, 10M tokens, 1 or 3 epochs.

Cost = tokens × epochs / throughput / 3600 ×
(1 + overhead) × hourly rate.

[30 cost scenarios](../results/qlora-cost-comparison.csv)
cover five GPUs across RunPod and Lambda.

Throughput and runtime overhead are assumed.
Storage, transfer and taxes are excluded.

Pricing verified on 2026-11-25 against official
[RunPod](https://www.runpod.io/pricing) and
[Lambda](https://lambda.ai/pricing) published rates.
All five rates matched; regional availability remains unverified.

[Pricing inputs](../data/gpu-pricing.csv).

## FinOps Roadmap

Baseline: $89/month for 100 GPU-hours.

| Timeline | Actions | Monthly savings per action | Complexity |
|---|---|---|---|
| 0-30 days | Idle termination; free runtimes; deduplication | $17.80; $8.90; $13.35 | Low |
| 1-6 months | Packing; optimized kernels; recovery; spot GPUs | $17.80; $22.25; $8.90; $35.60 | Medium |
| 6+ months | Scheduling; GPU selection; orchestration | $26.70; $31.15; $17.80 | High |
| 6+ months | Artifact caching | $8.90 | Medium |

[Savings model](../results/monthly-savings.csv).
Estimates are independent, not additive or realized.

## Compliance and Risk

| Priority | Gap | Required control |
|---|---|---|
| Critical | Data and credential exposure | Data approval, secret scanning |
| High | Supply chain and GPU exhaustion | Dependency scans, resource limits |
| High | Quality and cost overruns | Evaluation gates, budget enforcement |
| Medium | Recovery and reproducibility | Restore tests, revision pinning |

Roadmap: secure inputs immediately, validate recovery and
access controls within six months, then centralize governance.

### Audit Preparation Checklist

- [x] Training configurations and datasets retained.
- [x] GPU metrics and evaluation evidence captured.
- [x] Adapter reload and automated tests passed.
- [x] Published GPU pricing verified.
- [ ] Immutable model revisions and dependency lock.
- [ ] Security scans and license review.
- [ ] Artifact integrity and recovery testing.
- [ ] Production access controls and audit logs.

## Performance and Scalability

- Database: indexed job metadata and query-plan analysis.
- Caching: versioned models and tokenized datasets.
- Load balancing: GPU-aware scheduling.
- Resources: quantization, packing and checkpointing.
- Bottlenecks: GPU, CPU, I/O and throughput profiling.
- Scalability: distributed FSDP or DeepSpeed.
- Monitoring: DCGM, Prometheus and Grafana; track VRAM,
  utilization, loss, throughput, failures and spending.

These are proposed production improvements.

## Rollback and Recovery

| Failure | Recovery action |
|---|---|
| Model quality regression | Disable adapter or restore last approved adapter |
| Training interruption | Resume from last valid checkpoint |
| Configuration regression | Revert to last validated Git revision |
| Dependency failure | Recreate last known-good environment |
| Cost overrun | Stop training and enforce budget limits |

Retain versioned adapters, configurations, checkpoints and
evaluation results. Validate artifact integrity, inference
quality and resource consumption before restoring service.

Adapter disabling was exercised during evaluation.
Production rollback and checkpoint recovery remain untested.

## Conclusion

LoRA execution and adapter reload validated; 26 tests passed.
QLoRA costs are modeled. Full and distributed training,
production controls and cross-framework benchmarks remain untested.

## References

- [LoRA](https://arxiv.org/abs/2106.09685)
- [QLoRA](https://arxiv.org/abs/2305.14314)
- [TRL](https://huggingface.co/docs/trl/)
- [Axolotl](https://docs.axolotl.ai/)
- [Unsloth](https://unsloth.ai/docs)
- [torchtune](https://github.com/pytorch/torchtune)
- [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework)
