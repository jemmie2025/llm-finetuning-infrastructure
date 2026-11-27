"""Configuration-driven single-GPU LoRA/QLoRA training."""

import argparse
import json
import math
import time
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    model_cfg = config["model"]
    adapter_cfg = config["lora"]
    train_cfg = config["training"]

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        set_seed,
    )
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required.")

    set_seed(train_cfg["seed"])
    bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    dtype = torch.bfloat16 if bf16 else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["id"],
        trust_remote_code=False,
        revision=model_cfg["revision"],
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": {"": 0},
        "trust_remote_code": False,
        "revision": model_cfg["revision"],
    }

    if model_cfg["method"] == "qlora":
        quant = model_cfg["quantization"]
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant["type"],
            bnb_4bit_use_double_quant=quant["double_quantization"],
            bnb_4bit_compute_dtype=dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["id"], **model_kwargs
    )
    import bitsandbytes as bnb

    quantized_layers = sum(
        isinstance(layer, bnb.nn.Linear4bit)
        for layer in model.modules()
    )

    assert model.is_loaded_in_4bit
    assert quantized_layers > 0

    model.config.use_cache = False

    adapter = LoraConfig(
        r=adapter_cfg["r"],
        lora_alpha=adapter_cfg["alpha"],
        lora_dropout=adapter_cfg["dropout"],
        target_modules=adapter_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    dataset = load_dataset(
        "json", data_files=args.dataset, split="train"
    )

    training_args = SFTConfig(
        output_dir=train_cfg["output_dir"],
        max_length=train_cfg["max_length"],
        per_device_train_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=(
            train_cfg["gradient_accumulation_steps"]
        ),
        learning_rate=train_cfg["learning_rate"],
        num_train_epochs=train_cfg["epochs"],
        max_steps=train_cfg["max_steps"],
        optim=train_cfg["optimizer"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=bf16,
        fp16=not bf16,
        seed=train_cfg["seed"],
        logging_steps=1,
        logging_nan_inf_filter=False,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=2,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=adapter,
    )

    trainer.model.print_trainable_parameters()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    result = trainer.train(resume_from_checkpoint=args.resume)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    if not math.isfinite(result.training_loss):
        raise RuntimeError("Non-finite training loss detected.")

    trainable, total = trainer.model.get_nb_trainable_parameters()

    metrics = {
        "loaded_in_4bit": True,
        "quantized_layers": quantized_layers,
        "model": model_cfg["id"],
        "method": model_cfg["method"],
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "precision": str(dtype),
        "dataset_samples": len(dataset),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "training_steps": result.global_step,
        "training_loss": result.training_loss,
        "runtime_seconds": round(elapsed, 2),
        "peak_allocated_vram_gib": round(
            torch.cuda.max_memory_allocated() / 1024**3, 3
        ),
        "peak_reserved_vram_gib": round(
            torch.cuda.max_memory_reserved() / 1024**3, 3
        ),
        "trainer_metrics": result.metrics,
    }

    trainer.save_model(train_cfg["output_dir"])
    tokenizer.save_pretrained(train_cfg["output_dir"])

    Path("results").mkdir(exist_ok=True)
    path = Path(f"results/{model_cfg['method']}-metrics.json")
    path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
