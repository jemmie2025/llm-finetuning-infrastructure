
import sys
import json
import math
import time
import torch
import hashlib

from pathlib import Path
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from trl import SFTConfig, SFTTrainer

method, model_id, revision = sys.argv[1:]

assert torch.cuda.is_available()
set_seed(42)

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    revision=revision,
    trust_remote_code=False,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

kwargs = {
    "revision": revision,
    "trust_remote_code": False,
    "device_map": {"": 0},
    "torch_dtype": (
        torch.float32 if method == "full"
        else torch.float16
    ),
}

if method == "qlora":
    kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

model = AutoModelForCausalLM.from_pretrained(
    model_id, **kwargs
)

model.config.use_cache = False

if method == "qlora":
    assert model.is_loaded_in_4bit

adapter = None

if method != "full":
    adapter = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj"
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

dataset = load_dataset(
    "json",
    data_files="data/train.jsonl",
    split="train",
)

args = SFTConfig(
    output_dir=f"outputs/{method}",
    max_length=512,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_steps=20,
    learning_rate=2e-5,
    optim="adamw_torch",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={
        "use_reentrant": False
    },
    fp16=True,
    bf16=False,
    seed=42,
    logging_steps=1,
    logging_nan_inf_filter=False,
    save_strategy="no",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=adapter,
)

if method == "full":
    total = sum(
        p.numel() for p in trainer.model.parameters()
    )
    trainable = sum(
        p.numel() for p in trainer.model.parameters()
        if p.requires_grad
    )
    assert trainable == total
else:
    trainable, total = (
        trainer.model.get_nb_trainable_parameters()
    )

name, parameter = next(
    (n, p)
    for n, p in trainer.model.named_parameters()
    if p.requires_grad and (
        "q_proj.weight" in n if method == "full"
        else "lora_B" in n
    )
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
), "Selected trainable weights did not change"

metrics = {
    "framework": "HF TRL",
    "model": model_id,
    "model_revision": revision,
    "method": method,
    "gpu": torch.cuda.get_device_name(0),
    "torch_version": str(torch.__version__),
    "samples": len(dataset),
    "dataset_sha256": hashlib.sha256(
        Path("data/train.jsonl").read_bytes()
    ).hexdigest(),
    "steps": result.global_step,
    "learning_rate": 2e-5,
    "mixed_precision": "fp16",
    "trainable_parameters": trainable,
    "total_parameters": total,
    "checked_parameter": name,
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
}

Path("results").mkdir(exist_ok=True)

Path(f"results/{method}.json").write_text(
    json.dumps(metrics, indent=2) + "\n"
)

print(json.dumps(metrics, indent=2))
