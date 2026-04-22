"""SFT training for cluster_v2023 — forked from train_sft.py.

Carries two fixes flagged by third-party review (Kimi + Codex,
2026-04-22) on a step-time degradation bug that resisted five rounds
of ad-hoc tuning in the shared script:

  P1 (line ~120): drop `dtype=torch.bfloat16` from from_pretrained.
      Combined with `quantization_config=bnb_config` it forced
      accelerate to load the 14B checkpoint in bf16 (~28 GB) before
      bnb replaced layers with Linear4bit. The pre-quant bf16 copy
      lingered in VRAM → +~20 GB ghost memory.
      bnb_4bit_compute_dtype already governs matmul precision.

  P2 (SFTConfig): add gradient_checkpointing_kwargs={"use_reentrant":
      False}. transformers>=4.35 defaults use_reentrant=True, which
      the PEFT docs explicitly warn against for LoRA adapters. The
      reentrant checkpoint graph doesn't preserve adapter hooks, so
      either activations aren't freed (per-step leak → +100s/step
      linear growth we observed) or GC silently no-ops (full 30-40 GB
      activation). See PEFT docs + transformers issue 28867.

The shared scripts/train_sft.py is NOT modified — cluster v1 and
finance may have their own tuning around it. This fork applies the
fixes only to cluster_v2023's SFT run. See decisions-cluster-v2023.md
Part 7 for full diagnostic context.
"""

import argparse
import json
import logging
import os
import sys
import time

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

logger = logging.getLogger(__name__)


def setup_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_sft_data(path: str, system_prompt: str | None = None) -> Dataset:
    """Load SFT data and convert to HF Dataset.

    If ``system_prompt`` is provided, prepend it as a ``system`` role message
    to every conversation that does not already start with one. The Finance
    domain exports data without a system message (the domain-specific prompt
    is injected at training time to match inference).
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    records = []
    for item in raw:
        messages = list(item["messages"])
        if system_prompt and (not messages or messages[0].get("role") != "system"):
            messages = [{"role": "system", "content": system_prompt}] + messages
        records.append({"messages": messages})

    ds = Dataset.from_list(records)
    logger.info(f"Loaded {len(ds)} SFT samples from {path}"
                + (" (system prompt injected)" if system_prompt else ""))
    return ds


def main():
    parser = argparse.ArgumentParser(description="SFT training with QLoRA")
    parser.add_argument("--model-path", default="Qwen/Qwen3-14B")
    parser.add_argument("--data-path", default="outputs/sft_final/sft_data.json")
    parser.add_argument("--output-dir", default="outputs/sft_model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--system-prompt-file", default=None,
        help="Path to a text file holding the system prompt to prepend to "
             "every SFT conversation (e.g. configs/finance_system_prompt.txt).",
    )
    args = parser.parse_args()

    setup_logging(args.output_dir)
    t0 = time.perf_counter()

    logger.info("=" * 60)
    logger.info("SFT Training — QLoRA on Qwen3-14B")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Data: {args.data_path}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Epochs: {args.epochs}, BS: {args.batch_size}, "
                f"Grad accum: {args.grad_accum}, LR: {args.lr}")
    logger.info(f"LoRA r={args.lora_r}, alpha={args.lora_alpha}")
    logger.info(f"Max seq len: {args.max_seq_len}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}, "
                f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with 4-bit quantization
    logger.info("Loading model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # P1: no `dtype=torch.bfloat16` here — bnb_4bit_compute_dtype already
    # governs matmul precision, and passing dtype= alongside
    # quantization_config caused accelerate to materialize a bf16 copy of
    # the 14B weights (~28 GB) that lingered in VRAM after bnb replaced
    # layers with Linear4bit. See module docstring.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    logger.info(f"Trainable params: {trainable:,} / {total:,} "
                f"({trainable / total * 100:.2f}%)")

    # Load data
    sys_prompt = None
    if args.system_prompt_file:
        with open(args.system_prompt_file, encoding="utf-8") as f:
            sys_prompt = f.read().rstrip()
        logger.info(f"System prompt loaded from {args.system_prompt_file} "
                    f"({len(sys_prompt)} chars)")
    dataset = load_sft_data(args.data_path, system_prompt=sys_prompt)

    # Split 90/10 for train/eval
    split = dataset.train_test_split(test_size=0.1, seed=args.seed)
    train_ds = split["train"]
    eval_ds = split["test"]
    logger.info(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    # Training arguments
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        bf16=True,
        logging_steps=5,
        # eval during training has been observed to leak tensors into the
        # CUDA caching pool, producing +1-3s/step compounding degradation
        # starting around step 200 (see decisions Part 7 P7-9). Final
        # eval via trainer.evaluate() at the end is still safe because no
        # further training step follows it.
        eval_strategy="no",
        save_strategy="epoch",
        save_total_limit=None,  # keep every epoch checkpoint for post-hoc best-of-k eval
        seed=args.seed,
        report_to="none",
        gradient_checkpointing=True,
        # P2: transformers>=4.35 defaults use_reentrant=True, which breaks
        # gradient flow for PEFT LoRA adapters. Forcing False per PEFT docs.
        # See module docstring.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,
        dataloader_num_workers=0,  # Windows server
        max_length=args.max_seq_len,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    # Train
    logger.info("Starting training...")
    train_result = trainer.train()

    elapsed = time.perf_counter() - t0
    logger.info("=" * 60)
    logger.info(f"Training complete in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    logger.info(f"Train loss: {train_result.training_loss:.4f}")
    logger.info(f"Train samples/sec: {train_result.metrics.get('train_samples_per_second', 0):.2f}")

    # Save
    logger.info("Saving model...")
    trainer.save_model(os.path.join(args.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))

    # Final eval
    logger.info("Running final evaluation...")
    eval_results = trainer.evaluate()
    logger.info(f"Eval loss: {eval_results['eval_loss']:.4f}")
    logger.info(f"Eval perplexity: {torch.exp(torch.tensor(eval_results['eval_loss'])):.2f}")

    # Save metrics
    metrics = {
        "train_loss": train_result.training_loss,
        "eval_loss": eval_results["eval_loss"],
        "eval_perplexity": torch.exp(torch.tensor(eval_results["eval_loss"])).item(),
        "trainable_params": trainable,
        "total_params": total,
        "train_samples": len(train_ds),
        "eval_samples": len(eval_ds),
        "elapsed_seconds": elapsed,
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Done!")


if __name__ == "__main__":
    main()
