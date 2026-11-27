# LLM Fine-Tuning Infrastructure Assessment

## Objective

Validate LoRA, QLoRA and full fine-tuning on GPU
infrastructure, compare TRL and Unsloth, and estimate
7B QLoRA training costs.

## Fine-Tuning Method Comparison

Environment: NVIDIA Tesla T4, Qwen2.5-0.5B-Instruct,
80 synthetic training samples and 20 training steps.

| Method | Trainable parameters | Runtime | Peak allocated VRAM |
|---|---:|---:|---:|
| Full fine-tuning | 494,032,768 | 30.03 s | 9.230 GiB |
| LoRA | 1,081,344 | 27.83 s | 1.040 GiB |
| QLoRA | 1,081,344 | 40.68 s | 1.050 GiB |

LoRA achieved approximately 89% lower peak allocated
GPU memory than full fine-tuning.

QLoRA reduced peak reserved memory from 1.209 GiB
to 1.152 GiB but increased runtime relative to LoRA.

### Qwen3-1.7B Comparison

| Method | Runtime | Peak allocated VRAM |
|---|---:|---:|
| LoRA | 41.45 s | 3.323 GiB |
| QLoRA | 55.74 s | 2.539 GiB |

QLoRA reduced peak allocated memory by 23.6%,
with a corresponding increase in training runtime.

These results demonstrate workload-specific resource
trade-offs rather than universal performance advantages.

## Framework Comparison

| Metric | TRL | Unsloth |
|---|---:|---:|
| Runtime | 27.83 s | 38.54 s |
| Peak allocated VRAM | 1.040 GiB | 0.992 GiB |
| Throughput (samples/s) | 2.944 | 2.164 |
| Training steps | 20 | 20 |

Both stacks completed training and demonstrated
parameter updates. Unsloth consumed slightly less
GPU memory but required longer execution time.

Different dependency versions and training optimizations
limit direct performance attribution. Two non-finite
gradient norms were observed during Unsloth training.

## Decision Matrix

| Method or framework | Recommended use |
|---|---|
| LoRA | Resource-efficient model adaptation |
| QLoRA | Memory-constrained model training |
| Full fine-tuning | Workloads requiring all parameters to be updated |
| TRL | Configurable and reproducible training workflows |
| Unsloth | Optimized training where compatibility and stability are established |

## Model Evaluation

| Evaluation | Baseline | LoRA-adapted |
|---|---:|---:|
| Original (10 samples) | 0% | 30% |
| Expanded (100 samples) | 39% | 50% |

The original scores represent baseline and adapted
model performance, not variation across repeated runs.

Greedy decoding and strict exact matching were used.
The limited training budget may have contributed to
low accuracy.

The expanded synthetic evaluation demonstrated an
11-percentage-point improvement using a newly trained
adapter. The two evaluations are not directly comparable.

All 20-step training runs are functional smoke tests,
not evidence of convergence or production model quality.

## 7B QLoRA Cost Modeling

Assumptions: 10 million tokens, one epoch and 15%
runtime overhead.

| GPU | Estimated duration | Estimated cost |
|---|---:|---:|
| RTX 3090 | 9.13 h | $4.56 |
| RTX 4090 | 5.32 h | $4.74 |
| L40S | 4.56 h | $4.97 |
| H100 PCIe | 2.66 h | $7.69 |
| A10 | 12.78 h | $16.48 |

The cost model covers 30 scenarios across five GPUs.
Estimates use recorded hourly rates and assumed throughput.
Storage, transfer and taxes are excluded.

Cost-saving recommendations prioritize GPU utilization,
idle resource termination and workload-appropriate
GPU selection. Savings are estimated, not realized.

- [Pricing inputs](../data/gpu-pricing.csv)
- [Cost scenarios](../results/qlora-cost-comparison.csv)
- [Savings estimates](../results/monthly-savings.csv)

## Security Assessment

The dependency audit performed on 2026-11-27 reported
44 findings across five packages. Deduplication by
package, version and CVE identified 37 distinct advisories.

| Dependency | Scanned version | Distinct advisories |
|---|---|---:|
| Transformers | 4.52.4 | 17 |
| Accelerate | 1.7.0 | 1 |
| Datasets | 3.6.0 | 1 |
| fsspec | 2025.3.0 | 1 |
| filelock | 4.1.0 | 17 |
| Total | | 37 |

### Vulnerability Inventory

| Package | Version | CVE | GHSA | Severity (CVSS) | Reported fixes |
|---|---|---|---|---|---|
| accelerate | 1.7.0 | [CVE-2026-69112](https://nvd.nist.gov/vuln/detail/CVE-2026-69112) | [GHSA-4j2p-28q2-5m79](https://github.com/advisories/GHSA-4j2p-28q2-5m79) | Medium (6.9) | No fix listed |
| datasets | 3.6.0 | [CVE-2026-66007](https://nvd.nist.gov/vuln/detail/CVE-2026-66007) | [GHSA-379c-qx7v-6h59](https://github.com/advisories/GHSA-379c-qx7v-6h59) | Medium (6.9) | 5.0.1 |
| filelock | 4.1.0 | [CVE-2026-112906](https://nvd.nist.gov/vuln/detail/CVE-2026-112906) | [GHSA-2g95-wg33-c5x2](https://github.com/advisories/GHSA-2g95-wg33-c5x2) | Medium (4.7) | 4.2.1 |
| filelock | 4.1.0 | [CVE-2026-112907](https://nvd.nist.gov/vuln/detail/CVE-2026-112907) | [GHSA-567q-5qvw-pf92](https://github.com/advisories/GHSA-567q-5qvw-pf92) | Low (2.5) | 4.2.1 |
| filelock | 4.1.0 | [CVE-2026-112908](https://nvd.nist.gov/vuln/detail/CVE-2026-112908) | [GHSA-6627-m3xw-93mp](https://github.com/advisories/GHSA-6627-m3xw-93mp) | Medium (5.3) | 4.2.1 |
| filelock | 4.1.0 | [CVE-2026-112909](https://nvd.nist.gov/vuln/detail/CVE-2026-112909) | [GHSA-69hj-f68j-2879](https://github.com/advisories/GHSA-69hj-f68j-2879) | Medium (6.1) | 4.2.1 |
| filelock | 4.1.0 | [CVE-2026-112910](https://nvd.nist.gov/vuln/detail/CVE-2026-112910) | [GHSA-8mx5-7m5m-r77g](https://github.com/advisories/GHSA-8mx5-7m5m-r77g) | Medium (5.3) | 4.2.1 |
| filelock | 4.1.0 | [CVE-2026-112911](https://nvd.nist.gov/vuln/detail/CVE-2026-112911) | [GHSA-9c4x-gm8v-wgc4](https://github.com/advisories/GHSA-9c4x-gm8v-wgc4) | Medium (6.3) | 4.2.1 |
| filelock | 4.1.0 | [CVE-2026-112912](https://nvd.nist.gov/vuln/detail/CVE-2026-112912) | [GHSA-9j6q-wvj5-jgjg](https://github.com/advisories/GHSA-9j6q-wvj5-jgjg) | Low (2.5) | 4.2.1 |
| filelock | 4.1.0 | [CVE-2026-112913](https://nvd.nist.gov/vuln/detail/CVE-2026-112913) | [GHSA-9vhv-jxww-xfm5](https://github.com/advisories/GHSA-9vhv-jxww-xfm5) | Low (3.3) | 4.2.1 |
| filelock | 4.1.0 | [CVE-2026-112914](https://nvd.nist.gov/vuln/detail/CVE-2026-112914) | [GHSA-crv4-6mhj-j5jh](https://github.com/advisories/GHSA-crv4-6mhj-j5jh) | Medium (5.3) | 4.2.1 |
| filelock | 4.1.0 | [CVE-2026-112915](https://nvd.nist.gov/vuln/detail/CVE-2026-112915) | [GHSA-cwqr-hh33-gh44](https://github.com/advisories/GHSA-cwqr-hh33-gh44) | Medium (5.5) | 4.2.1 |
| filelock | 4.1.0 | [CVE-2026-112934](https://nvd.nist.gov/vuln/detail/CVE-2026-112934) | [GHSA-fj95-fr6c-h845](https://github.com/advisories/GHSA-fj95-fr6c-h845) | Medium (6.1) | 4.2.1 |
| filelock | 4.1.0 | [CVE-2026-112935](https://nvd.nist.gov/vuln/detail/CVE-2026-112935) | [GHSA-fq56-3cx3-fjxp](https://github.com/advisories/GHSA-fq56-3cx3-fjxp) | Medium (4.7) | 4.2.1 |
| filelock | 4.1.0 | [CVE-2026-112937](https://nvd.nist.gov/vuln/detail/CVE-2026-112937) | [GHSA-jqc4-5cvw-7w85](https://github.com/advisories/GHSA-jqc4-5cvw-7w85) | Medium (6.1) | 4.2.1 |
| filelock | 4.1.0 | [CVE-2026-112938](https://nvd.nist.gov/vuln/detail/CVE-2026-112938) | [GHSA-m462-7rxw-wvpg](https://github.com/advisories/GHSA-m462-7rxw-wvpg) | Medium (5.8) | 4.2.1 |
| filelock | 4.1.0 | [CVE-2026-112939](https://nvd.nist.gov/vuln/detail/CVE-2026-112939) | [GHSA-rfpc-qcw5-3pc4](https://github.com/advisories/GHSA-rfpc-qcw5-3pc4) | Medium (4.7) | 4.2.1 |
| filelock | 4.1.0 | [CVE-2026-112940](https://nvd.nist.gov/vuln/detail/CVE-2026-112940) | [GHSA-v4x8-3wr5-3wq8](https://github.com/advisories/GHSA-v4x8-3wr5-3wq8) | High (7.1) | 4.2.1 |
| filelock | 4.1.0 | [CVE-2026-112941](https://nvd.nist.gov/vuln/detail/CVE-2026-112941) | [GHSA-wfhq-q4j2-c37m](https://github.com/advisories/GHSA-wfhq-q4j2-c37m) | Low (3.3) | 4.2.1 |
| fsspec | 2025.3.0 | [CVE-2026-104851](https://nvd.nist.gov/vuln/detail/CVE-2026-104851) | [GHSA-27vj-qcqg-25rc](https://github.com/advisories/GHSA-27vj-qcqg-25rc) | High (8.8) | 2026.6.0 |
| transformers | 4.52.4 | [CVE-2025-14920](https://nvd.nist.gov/vuln/detail/CVE-2025-14920) | Not listed | High (7.8) | No fix listed |
| transformers | 4.52.4 | [CVE-2025-14921](https://nvd.nist.gov/vuln/detail/CVE-2025-14921) | Not listed | High (7.8) | No fix listed |
| transformers | 4.52.4 | [CVE-2025-14924](https://nvd.nist.gov/vuln/detail/CVE-2025-14924) | Not listed | High (7.8) | No fix listed |
| transformers | 4.52.4 | [CVE-2025-14926](https://nvd.nist.gov/vuln/detail/CVE-2025-14926) | Not listed | High (7.8) | No fix listed |
| transformers | 4.52.4 | [CVE-2025-14927](https://nvd.nist.gov/vuln/detail/CVE-2025-14927) | Not listed | High (7.8) | No fix listed |
| transformers | 4.52.4 | [CVE-2025-14928](https://nvd.nist.gov/vuln/detail/CVE-2025-14928) | Not listed | High (7.8) | No fix listed |
| transformers | 4.52.4 | [CVE-2025-14929](https://nvd.nist.gov/vuln/detail/CVE-2025-14929) | Not listed | High (7.8) | No fix listed |
| transformers | 4.52.4 | [CVE-2025-14930](https://nvd.nist.gov/vuln/detail/CVE-2025-14930) | Not listed | High (7.8) | No fix listed |
| transformers | 4.52.4 | [CVE-2025-5197](https://nvd.nist.gov/vuln/detail/CVE-2025-5197) | [GHSA-9356-575x-2w9m](https://github.com/advisories/GHSA-9356-575x-2w9m) | Medium (5.3) | 4.53.0 |
| transformers | 4.52.4 | [CVE-2025-6051](https://nvd.nist.gov/vuln/detail/CVE-2025-6051) | [GHSA-rcv9-qm8p-9p6j](https://github.com/advisories/GHSA-rcv9-qm8p-9p6j) | Medium (5.3) | 4.53.0 |
| transformers | 4.52.4 | [CVE-2025-6638](https://nvd.nist.gov/vuln/detail/CVE-2025-6638) | [GHSA-59p9-h35m-wg4g](https://github.com/advisories/GHSA-59p9-h35m-wg4g) | Medium (5.3) | 4.53.0 |
| transformers | 4.52.4 | [CVE-2025-6921](https://nvd.nist.gov/vuln/detail/CVE-2025-6921) | [GHSA-4w7r-h757-3r74](https://github.com/advisories/GHSA-4w7r-h757-3r74) | Medium (5.3) | 4.53.0 |
| transformers | 4.52.4 | [CVE-2026-1839](https://nvd.nist.gov/vuln/detail/CVE-2026-1839) | [GHSA-69w3-r845-3855](https://github.com/advisories/GHSA-69w3-r845-3855) | Medium (6.5) | 5.0.0, 5.0.0rc3 |
| transformers | 4.52.4 | [CVE-2026-4372](https://nvd.nist.gov/vuln/detail/CVE-2026-4372) | [GHSA-29pf-2h5f-8g72](https://github.com/advisories/GHSA-29pf-2h5f-8g72) | High (7.8) | 5.3.0 |
| transformers | 4.52.4 | [CVE-2026-5241](https://nvd.nist.gov/vuln/detail/CVE-2026-5241) | [GHSA-fgcw-684q-jj6r](https://github.com/advisories/GHSA-fgcw-684q-jj6r) | High (8.0) | 5.5.0 |
| transformers | 4.52.4 | [CVE-2026-80047](https://nvd.nist.gov/vuln/detail/CVE-2026-80047) | [GHSA-x9r9-c232-4q39](https://github.com/advisories/GHSA-x9r9-c232-4q39) | High (7.8) | No fix listed |
| transformers | 4.52.4 | [CVE-2026-9856](https://nvd.nist.gov/vuln/detail/CVE-2026-9856) | [GHSA-xrqw-3rrv-vx5w](https://github.com/advisories/GHSA-xrqw-3rrv-vx5w) | High (7.1) | 5.10.0 |

### Remediation Recommendations

Prioritize high-severity findings, particularly those
involving untrusted model loading, arbitrary code execution
and filesystem access. Adopt compatible patched releases
where available and monitor upstream advisories where
no fixed release is listed.

Reported fixes are recommendations, not applied remediation.

### Assessment Scope

Severity ratings use the GitHub Advisory Database,
published security bulletins and advisory descriptions
included in the audit. CVSS versions may differ by advisory.

CVE-2026-1839 has additional PyTorch version prerequisites.
Scanner findings establish potential dependency exposure,
not confirmed exploitability in this workload.

No findings were reported for the scanned PEFT 0.15.2
and bitsandbytes 0.49.0 versions.

The audit covers resolved package versions from the project
requirements. It does not establish complete runtime security
or cover the separate Unsloth environment.

### Advisory Sources

- [GitHub Advisory Database](https://github.com/advisories)
- [National Vulnerability Database](https://nvd.nist.gov/)
- [Transformers security bulletin](https://www.ibm.com/support/pages/node/7272658)

## Conclusion

LoRA, QLoRA and full fine-tuning were validated through
GPU smoke tests. TRL and Unsloth were compared using
measured resource consumption and execution time.

LoRA provided the lowest measured runtime among the
tested training methods. QLoRA demonstrated memory
savings on the 1.7B model.

The expanded evaluation recorded an 11-percentage-point
improvement in strict exact-match accuracy.

The results support infrastructure and training-method
selection within the tested scope.

## References

- [LoRA](https://arxiv.org/abs/2106.09685)
- [QLoRA](https://arxiv.org/abs/2305.14314)
- [TRL](https://huggingface.co/docs/trl/)
- [Unsloth](https://unsloth.ai/docs)
- [Experimental evidence](../results/)
