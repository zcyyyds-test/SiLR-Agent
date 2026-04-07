"""QLoRA SFT training with Unsloth + TRL.

Generic SFT skeleton — model name and data directory are configurable.
No frozen domain-specific prompts.
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
class SFTConfig:
    """Configuration for SFT training."""

    model_name: str = ""
    output_dir: str = "outputs/sft"
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

    # Training
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    max_seq_length: int = 4096
    bf16: bool = True
    gradient_checkpointing: bool = True

    # Data
    upsample_min: int = 0

    # Logging
    logging_steps: int = 10
    save_strategy: str = "epoch"
    save_total_limit: int = 2
    report_to: str = "none"

    @classmethod
    def from_yaml(cls, path: str | Path) -> SFTConfig:
        if yaml is None:
            raise ImportError("pyyaml required: pip install pyyaml")
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def run_sft(config: SFTConfig) -> Path:
    """Run QLoRA SFT training.

    Pipeline:
    1. Load base model in 4-bit via Unsloth (fallback: transformers + peft)
    2. Apply LoRA adapters
    3. Load SFT dataset via TrainingDataLoader
    4. Apply chat template
    5. Train with TRL SFTTrainer
    6. Save LoRA adapter

    Returns:
        Path to saved LoRA adapter directory.
    """
    if not config.model_name:
        raise ValueError("model_name must be set in SFTConfig")
    if not config.system_prompt:
        raise ValueError("system_prompt must be set in SFTConfig")

    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # -- 1. Load model --
    model, tokenizer = _load_model(config)

    # -- 2. Apply LoRA --
    model, unsloth_gc = _apply_lora(model, config)

    # -- 3. Load data --
    from .data_loader import TrainingDataLoader

    loader = TrainingDataLoader(config.data_dir)
    dataset = loader.load_sft_dataset(
        system_prompt=config.system_prompt,
        upsample_min=config.upsample_min,
    )
    logger.info(f"SFT dataset: {len(dataset)} conversations")

    # -- 4. Format with chat template --
    def format_chat(example: dict) -> dict:
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = dataset.map(format_chat, remove_columns=dataset.column_names)

    # -- 5. Train --
    from trl import SFTTrainer, SFTConfig as TRLSFTConfig

    gc_enabled = config.gradient_checkpointing and not unsloth_gc

    training_args = TRLSFTConfig(
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
        optim="adamw_8bit",
        max_length=config.max_seq_length,
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    logger.info("Starting SFT training...")
    trainer.train()

    # -- 6. Save adapter --
    adapter_path = output_path / "adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    logger.info(f"SFT adapter saved to {adapter_path}")

    return adapter_path


def _load_model(config: SFTConfig):
    """Load base model, trying Unsloth first then falling back to transformers."""
    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model_name,
            max_seq_length=config.max_seq_length,
            dtype=None,
            load_in_4bit=config.load_in_4bit,
        )
        logger.info(f"Loaded {config.model_name} via Unsloth (4-bit)")
        return model, tokenizer

    except Exception as e:
        logger.warning(f"Unsloth failed ({type(e).__name__}: {e}), falling back to transformers")

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import torch

    hf_name = config.model_name
    if hf_name.startswith("unsloth/"):
        base = hf_name.replace("unsloth/", "")
        if base.startswith("Qwen"):
            hf_name = f"Qwen/{base}"
        else:
            hf_name = base

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

    logger.info(f"Loaded {hf_name} via transformers (4-bit fallback)")
    return model, tokenizer


def _apply_lora(model, config: SFTConfig) -> tuple:
    """Apply LoRA adapters. Returns (model, unsloth_gc)."""
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
        logger.info(f"Applied LoRA via Unsloth (r={config.lora_r})")
        return model, True

    except Exception as e:
        logger.warning(f"Unsloth LoRA failed ({type(e).__name__}), falling back to peft")

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
    model.print_trainable_parameters()
    logger.info(f"Applied LoRA via peft (r={config.lora_r})")
    return model, False
