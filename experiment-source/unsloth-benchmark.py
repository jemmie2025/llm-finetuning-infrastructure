
from unsloth import FastLanguageModel
import torch
import json
import time
import math
import hashlib
import inspect

from pathlib import Path
from importlib.metadata import version
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
revision = "7ae557604adf67be50417f59c2c2f167def9a775"

assert torch.cuda.is_available()

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_id,
    revision=revision,
    use_exact_model_name=True,
    max_seq_length=512,
    dtype=torch.float16,
    load_in_4bit=False,
    trust_remote_code=False,
)

assert not getattr(model, "is_loaded_in_4bit", False)

model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj"
    ],
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing=True,
    random_state=42,
)

model.config.use_cache = False

dataset = load_dataset(
    "json",
    data_files="data/train.jsonl",
    split="train",
)

assert len(dataset) == 80

config = dict(
    output_dir="outputs/lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_steps=20,
    learning_rate=2e-5,
    optim="adamw_torch",
    fp16=True,
    bf16=False,
    seed=42,
    logging_steps=1,
    logging_nan_inf_filter=False,
    save_strategy="no",
    report_to="none",
    packing=False,
)

fields = SFTConfig.__dataclass_fields__

if "max_length" in fields:
    config["max_length"] = 512
else:
    config["max_seq_length"] = 512

args = SFTConfig(**config)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=args,
)

trainable, total = model.get_nb_trainable_parameters()

name, parameter = next(
    (n, p)
    for n, p in model.named_parameters()
    if p.requires_grad and "lora_B" in n
)

before = parameter.detach().cpu().clone()

torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()

start = time.perf_counter()
result = trainer.train()
torch.cuda.synchronize()
elapsed = time.perf_counter() - start

assert result.global_step == 20
assert math.isfinite(result.training_loss)
assert not torch.equal(
    before, parameter.detach().cpu()
)

metrics = {
    "framework": "Unsloth + TRL",
    "model": model_id,
    "requested_model_revision": revision,
    "method": "lora",
    "gpu": torch.cuda.get_device_name(0),
    "samples": len(dataset),
    "dataset_sha256": hashlib.sha256(
        Path("data/train.jsonl").read_bytes()
    ).hexdigest(),
    "steps": result.global_step,
    "learning_rate": 2e-5,
    "trainable_parameters": trainable,
    "total_parameters": total,
    "weights_updated": True,
    "training_loss": result.training_loss,
    "runtime_seconds": round(elapsed, 2),
    "peak_allocated_vram_gib": round(
        torch.cuda.max_memory_allocated() / 1024**3, 3
    ),
    "peak_reserved_vram_gib": round(
        torch.cuda.max_memory_reserved() / 1024**3, 3
    ),
    "throughput_samples_per_second":
        result.metrics.get("train_samples_per_second"),
    "versions": {
        p: version(p) for p in [
            "unsloth", "unsloth-zoo", "torch",
            "transformers", "trl", "peft",
            "bitsandbytes", "accelerate", "datasets"
        ]
    }
}

Path("results").mkdir(exist_ok=True)

Path("results/unsloth-lora.json").write_text(
    json.dumps(metrics, indent=2) + "\n"
)

print(json.dumps(metrics, indent=2))
print("UNSLOTH TRAINING: PASSED")
