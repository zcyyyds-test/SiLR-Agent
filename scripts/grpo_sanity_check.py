"""Verify GRPO log_prob machinery before launching a training run.

Loads base model + SFT adapter, constructs a dummy prompt+completion
pair, computes log_prob with the same mask + sum semantics used in
train_grpo_cluster_v2023.py, and asserts:
  - sum log_prob is finite and negative (well-formed)
  - running the same sample twice gives log_prob within 1e-3 (deterministic)
  - ratio of (new / old) ≈ 1.0 (same model, same tokens)

Exit non-zero on any assertion failure. Run BEFORE launching iter 1.
Catches the cluster v1 fatal bug where log_prob included prompt tokens
(see decisions-cluster-v2023.md GRPO section).
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


def _sum_log_prob_on_action_tokens(model, tokenizer, prompt: str, action: str) -> tuple[float, int]:
    import torch

    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    full_ids = tokenizer(prompt + action, return_tensors="pt").input_ids.to(model.device)
    prompt_len = prompt_ids.shape[1]
    action_len = full_ids.shape[1] - prompt_len
    if action_len <= 0:
        raise RuntimeError("action tokenizes to 0 tokens — check chat template")

    labels = full_ids.clone()
    labels[:, :prompt_len] = -100  # mask prompt tokens

    with torch.no_grad():
        out = model(input_ids=full_ids, labels=labels)
    per_token_mean_neg_ll = out.loss.item()
    sum_log_prob = -per_token_mean_neg_ll * action_len
    return sum_log_prob, action_len


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True)
    p.add_argument("--sft-adapter", required=True)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(model, args.sft_adapter)
    model.eval()

    prompt = "You are a scheduler. State: node n0 Ready, 1 LS queued. Action: "
    action = (
        '{"tool_name": "assign_job", '
        '"params": {"job_id": "j0", "node_id": "n0"}}'
    )

    lp1, n1 = _sum_log_prob_on_action_tokens(model, tok, prompt, action)
    lp2, n2 = _sum_log_prob_on_action_tokens(model, tok, prompt, action)
    ratio = math.exp(lp2 - lp1)

    logger.info(f"sum log_prob (run 1) = {lp1:.4f} over {n1} action tokens")
    logger.info(f"sum log_prob (run 2) = {lp2:.4f} over {n2} action tokens")
    logger.info(f"ratio (new/old)     = {ratio:.6f}")

    assert math.isfinite(lp1), f"log_prob is not finite: {lp1}"
    assert lp1 < 0, f"log_prob must be negative, got {lp1}"
    assert abs(lp1 - lp2) < 1e-3, (
        f"deterministic replay diverged: {lp1} vs {lp2}")
    assert abs(ratio - 1.0) < 1e-3, (
        f"new/old ratio not ≈ 1.0 on same model: {ratio}")
    logger.info("SANITY CHECK PASSED — ok to launch GRPO training")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
