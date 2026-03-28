"""QLoRA DPO training with Unsloth + TRL.

Loads a SFT checkpoint and trains on preference pairs (chosen/rejected).
Generic skeleton — no domain-specific frozen prompts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


@dataclass
class DPOConfig:
    """Configuration for DPO training."""

    model_name: str = ""
    sft_checkpoint: str = "outputs/sft"
    output_dir: str = "outputs/dpo"
    data_dir: str = "data/training"
    system_prompt: str = ""

    # QLoRA
    load_in_4bit: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.0
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # DPO
    beta: float = 0.1
    loss_type: str = "sigmoid"

    # Training
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    max_seq_length: int = 4096
    max_prompt_length: int = 2048
    bf16: bool = True
    gradient_checkpointing: bool = True

    # Logging
    logging_steps: int = 10
    save_strategy: str = "epoch"
    save_total_limit: int = 2
    report_to: str = "none"

    @classmethod
    def from_yaml(cls, path: str | Path) -> DPOConfig:
        if yaml is None:
            raise ImportError("pyyaml required: pip install pyyaml")
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def run_dpo(config: DPOConfig) -> Path:
    """Run QLoRA DPO training on top of SFT checkpoint.

    Pipeline:
    1. Load base model + merge SFT LoRA adapter
    2. Apply fresh LoRA for DPO training
    3. Load DPO dataset via TrainingDataLoader
    4. Train with TRL DPOTrainer
    5. Save DPO adapter

    Returns:
        Path to saved DPO adapter directory.
    """
    if not config.model_name:
        raise ValueError("model_name must be set in DPOConfig")
    if not config.system_prompt:
        raise ValueError("system_prompt must be set in DPOConfig")

    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # -- 1. Load base + SFT adapter --
    model, tokenizer = _load_sft_model(config)

    # -- 2. Apply fresh LoRA for DPO --
    model, unsloth_gc = _apply_lora(model, config)

    # -- 3. Load DPO data --
    from .data_loader import TrainingDataLoader

    loader = TrainingDataLoader(config.data_dir)
    dataset = loader.load_dpo_dataset(system_prompt=config.system_prompt)
    logger.info(f"DPO dataset: {len(dataset)} pairs")

    # -- 4. Train --
    from trl import DPOTrainer, DPOConfig as TRLDPOConfig

    gc_enabled = config.gradient_checkpointing and not unsloth_gc

    training_args = TRLDPOConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        bf16=config.bf16,
        gradient_checkpointing=gc_enabled,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        save_total_limit=config.save_total_limit,
        report_to=config.report_to,
        beta=config.beta,
        loss_type=config.loss_type,
        max_length=config.max_seq_length,
        max_prompt_length=config.max_prompt_length,
        optim="adamw_8bit",
    )

    # Workaround: TRL 0.17+ expects model.warnings_issued but PEFT doesn't expose it
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    logger.info("Starting DPO training...")
    trainer.train()

    # -- 5. Save adapter --
    adapter_path = output_path / "adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    logger.info(f"DPO adapter saved to {adapter_path}")

    return adapter_path


def _load_sft_model(config: DPOConfig):
    """Load base model with SFT LoRA merged in."""
    sft_adapter = Path(config.sft_checkpoint) / "adapter"

    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model_name,
            max_seq_length=config.max_seq_length,
            dtype=None,
            load_in_4bit=config.load_in_4bit,
        )
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(sft_adapter))
        model = model.merge_and_unload()
        logger.info(f"Loaded {config.model_name} + merged SFT adapter via Unsloth")
        return model, tokenizer

    except Exception as e:
        logger.warning(f"Unsloth failed ({type(e).__name__}: {e}), using transformers fallback")

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    import torch

    hf_name = config.model_name
    if hf_name.startswith("unsloth/"):
        base = hf_name.replace("unsloth/", "")
        hf_name = f"Qwen/{base}" if base.startswith("Qwen") else base

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ) if config.load_in_4bit else None

    model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(model, str(sft_adapter))
    model = model.merge_and_unload()
    logger.info(f"Loaded {hf_name} + merged SFT adapter via transformers")
    return model, tokenizer


def _apply_lora(model, config: DPOConfig) -> tuple:
    """Apply fresh LoRA adapters for DPO training. Returns (model, unsloth_gc)."""
    try:
        from unsloth import FastLanguageModel

        model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )
        return model, True

    except Exception as e:
        logger.warning(f"Unsloth LoRA failed ({type(e).__name__}), using peft fallback")

    from peft import LoraConfig, get_peft_model

    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    return model, False
