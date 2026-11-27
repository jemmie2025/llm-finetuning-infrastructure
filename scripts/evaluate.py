"""Compare base and fine-tuned model responses."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="outputs/lora")
    parser.add_argument("--dataset", default="data/eval.jsonl")
    parser.add_argument(
        "--output", default="results/lora-evaluation.json"
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    if output_path.exists():
        raise SystemExit(f"Refusing to overwrite: {output_path}")

    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required.")

    config = PeftConfig.from_pretrained(args.adapter)
    dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported(including_emulation=False)
        else torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)

    base = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        torch_dtype=dtype,
        device_map={"": 0},
        trust_remote_code=False,
    )

    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    def generate(prompt):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        ).to(model.device)

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        return tokenizer.decode(
            output[0, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

    results = []

    for line in Path(args.dataset).read_text().splitlines():
        messages = json.loads(line)["messages"]
        prompt = messages[0]["content"]
        expected = messages[1]["content"]

        with model.disable_adapter():
            baseline = generate(prompt)

        adapted = generate(prompt)

        results.append({
            "prompt": prompt,
            "expected": expected,
            "baseline": baseline,
            "adapted": adapted,
            "baseline_exact_match": baseline == expected,
            "adapted_exact_match": adapted == expected,
        })

    report = {
        "adapter_reload": "passed",
        "samples": len(results),
        "baseline_exact_match_rate": sum(
            r["baseline_exact_match"] for r in results
        ) / len(results),
        "adapted_exact_match_rate": sum(
            r["adapted_exact_match"] for r in results
        ) / len(results),
        "responses": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x") as f:
        f.write(json.dumps(report, indent=2) + "\n")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
