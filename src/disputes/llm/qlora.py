"""QLoRA fine-tune of a specialist drafting model.

**This is not run by CI and it is not run by ``make eval``.** It needs a GPU and
``transformers``, ``peft``, ``bitsandbytes`` and ``trl``, none of which are in
the default install. It is here because the dataset builder and the curriculum
in this package exist to feed it, and a training script that was never written
would make those two pointless.

What the configuration encodes, and why:

* **4-bit NF4 with double quantisation.** NF4 is information-theoretically
  matched to normally-distributed weights, which is what a trained network's
  weights approximately are; double quantisation reclaims the quantisation
  constants themselves, which are not free at this scale.
* **bfloat16 compute dtype.** The weights are stored in 4 bits but the matmul
  runs in bf16. Running the compute in fp16 on a model this size is where the
  loss spikes come from.
* **Adapters on the attention *and* MLP projections.** Attention-only adapters
  are the common default and underfit a task that changes the output *format*
  as much as the content, which drafting with a fixed citation convention does.
* **Completion-only loss.** The prompt carries the retrieved sections and the
  computed deadlines verbatim, and training the model to reproduce them teaches
  it to memorise the corpus instead of to use it.

The evaluation after training is the same verifier the workflow uses: citation
resolution and the invented-date check. A fine-tune that lowers perplexity while
inventing dates is a worse model for this task, and perplexity will not say so.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..config import ARTIFACT_DIR


@dataclass
class QLoraConfig:
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    output_dir: str = str(ARTIFACT_DIR / "qlora-disputes")

    # quantisation
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"

    # adapters
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",   # attention
        "gate_proj", "up_proj", "down_proj",      # MLP - format-changing tasks need these
    )

    # optimisation
    epochs: int = 2
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    max_seq_length: int = 4096
    optim: str = "paged_adamw_8bit"
    gradient_checkpointing: bool = True
    bf16: bool = True

    # curriculum
    curriculum: bool = True
    shuffle_within_bucket: bool = True
    n_buckets: int = 5
    seed: int = 17

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingPlan:
    """What the run would do, computable without a GPU.

    Emitted so the configuration is reviewable and the cost is known before
    anyone spends four hours finding out.
    """

    config: QLoraConfig
    n_examples: int
    trainable_parameters: int = 0
    total_parameters: int = 0
    steps_per_epoch: int = 0
    total_steps: int = 0
    buckets: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_model": self.config.base_model,
            "examples": self.n_examples,
            "effective_batch_size": self.config.per_device_batch_size * self.config.gradient_accumulation_steps,
            "steps_per_epoch": self.steps_per_epoch,
            "total_steps": self.total_steps,
            "trainable_parameters": self.trainable_parameters,
            "total_parameters": self.total_parameters,
            "trainable_fraction": (
                round(self.trainable_parameters / self.total_parameters, 6) if self.total_parameters else None
            ),
            "curriculum_buckets": self.buckets,
            "config": self.config.to_dict(),
        }


def plan(n_examples: int, cfg: QLoraConfig | None = None, hidden_size: int = 4096, n_layers: int = 32) -> TrainingPlan:
    """Estimate the run without loading anything.

    LoRA adds ``2 * r * d`` parameters per adapted projection. With r=32 across
    seven projections in 32 layers of a 7B model that is roughly 0.6% of the
    weights trainable, which is the whole reason this fits on one card.
    """
    cfg = cfg or QLoraConfig()
    effective_batch = cfg.per_device_batch_size * cfg.gradient_accumulation_steps
    steps_per_epoch = max(1, n_examples // effective_batch)

    per_projection = 2 * cfg.lora_r * hidden_size
    trainable = per_projection * len(cfg.target_modules) * n_layers
    total = 7_000_000_000

    return TrainingPlan(
        config=cfg,
        n_examples=n_examples,
        trainable_parameters=trainable,
        total_parameters=total,
        steps_per_epoch=steps_per_epoch,
        total_steps=steps_per_epoch * cfg.epochs,
        buckets=[{"bucket": i, "share": round(1 / cfg.n_buckets, 3)} for i in range(cfg.n_buckets)],
    )


def curriculum_order(examples: list[dict[str, Any]], cfg: QLoraConfig) -> list[dict[str, Any]]:
    """Bucketed curriculum, not a strict sort.

    A strict difficulty sort correlates the ordering with the label - measured
    at eta-squared 0.42 on this dataset in ``curriculum.py`` - and a single pass
    over class-blocked data ends fitted to whatever came last. Bucketing into a
    few difficulty bands and shuffling *within* each band keeps the easy-first
    progression while breaking the class blocking that does the damage.
    """
    import random

    if not cfg.curriculum:
        ordered = list(examples)
        random.Random(cfg.seed).shuffle(ordered)
        return ordered

    ranked = sorted(examples, key=lambda e: e.get("difficulty", 0.5))
    size = max(1, len(ranked) // cfg.n_buckets)
    rng = random.Random(cfg.seed)
    out: list[dict[str, Any]] = []
    for start in range(0, len(ranked), size):
        bucket = ranked[start : start + size]
        if cfg.shuffle_within_bucket:
            rng.shuffle(bucket)
        out.extend(bucket)
    return out


def train(dataset_path: Path, cfg: QLoraConfig | None = None) -> dict[str, Any]:  # pragma: no cover - needs a GPU
    cfg = cfg or QLoraConfig()
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError(
            "QLoRA training needs the optional extras: pip install -e '.[finetune]' "
            f"(missing: {exc.name})"
        ) from exc

    examples = [json.loads(line) for line in Path(dataset_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    ordered = curriculum_order(examples, cfg)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantisation = BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=getattr(torch, cfg.bnb_4bit_compute_dtype),
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, quantization_config=quantisation, device_map="auto", torch_dtype=torch.bfloat16)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.gradient_checkpointing)
    model = get_peft_model(model, LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules), bias="none", task_type="CAUSAL_LM"))

    dataset = Dataset.from_list([{"messages": e["messages"]} for e in ordered])
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=SFTConfig(
            output_dir=cfg.output_dir,
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.per_device_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            lr_scheduler_type=cfg.lr_scheduler_type,
            warmup_ratio=cfg.warmup_ratio,
            max_length=cfg.max_seq_length,
            optim=cfg.optim,
            gradient_checkpointing=cfg.gradient_checkpointing,
            bf16=cfg.bf16,
            logging_steps=10,
            save_strategy="epoch",
            seed=cfg.seed,
            # Curriculum order is meaningless if the sampler reshuffles it.
            dataset_kwargs={"skip_prepare_dataset": False},
            group_by_length=False,
            completion_only_loss=True,
        ),
    )
    result = trainer.train()
    trainer.save_model(cfg.output_dir)
    return {"output_dir": cfg.output_dir, "metrics": result.metrics, "examples": len(ordered)}
