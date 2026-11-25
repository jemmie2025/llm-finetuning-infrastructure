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

TRL and PEFT were validated; other frameworks were researched.

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

Throughput is assumed; storage, transfer and taxes are excluded.

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

Audit evidence: configs, datasets, metrics, evaluation and tests.
Pending: security scans, license review, integrity checks,
recovery testing and production audit logs.

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
