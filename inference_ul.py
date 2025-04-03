#!/usr/bin/env python
# coding=utf-8

import os
import torch
import logging
from dataclasses import dataclass, field
from typing import Optional
import transformers
from transformers import (
    HfArgumentParser,
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
)
from datasets import load_dataset
from loraprune.lora import LoraConfig
from loraprune.peft_model import get_peft_model

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    lora_weights: str = field(default=None, metadata={"help": "Path to LoRA weights"})
    tokenizer_name: Optional[str] = field(default=None)

@dataclass
class DataArguments:
    forget_dataset: str = field(default=None, metadata={"help": "Dataset to check unlearning"})
    forget_subset: Optional[str] = field(default=None, metadata={"help": "Specific subset of forget dataset"})
    retain_dataset: str = field(default=None, metadata={"help": "Dataset to check retention"})
    retain_subset: Optional[str] = field(default=None, metadata={"help": "Specific subset of retain dataset"})
    block_size: Optional[int] = field(default=None)

@dataclass
class EvalArguments:
    output_dir: str = field(default="eval_results")
    per_device_eval_batch_size: int = field(default=8)
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_target_modules: str = field(default="q_proj,v_proj")

def load_dataset_split(dataset_path: str, subset: str = None):
    """
    Load full dataset or specific subset
    """
    if subset:
        return load_dataset(dataset_path, subset)
    return load_dataset(dataset_path)

def evaluate(
    model_args,
    data_args,
    eval_args,
):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    )
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Configure LoRA
    lora_target_modules = eval_args.lora_target_modules.split(',')
    config = LoraConfig(
        r=eval_args.lora_r,
        lora_alpha=eval_args.lora_alpha,
        target_modules=lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
        peft_type="LORA"
    )
    model = get_peft_model(model, config)
    
    # Load LoRA weights
    if model_args.lora_weights:
        model.load_state_dict(torch.load(model_args.lora_weights))
    
    model.eval()
    
    # Load datasets
    forget_data = load_dataset_split(data_args.forget_dataset, data_args.forget_subset)
    retain_data = load_dataset_split(data_args.retain_dataset, data_args.retain_subset)
    
    # Prepare data collator
    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    # Evaluate on forget set
    logger.info("Evaluating on forget set...")
    forget_loader = torch.utils.data.DataLoader(
        forget_data["test"] if "test" in forget_data else forget_data["validation"],
        batch_size=eval_args.per_device_eval_batch_size,
        collate_fn=data_collator
    )
    
    forget_loss = 0
    forget_steps = 0
    with torch.no_grad():
        for batch in forget_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            forget_loss += outputs.loss.item()
            forget_steps += 1
    forget_perplexity = torch.exp(torch.tensor(forget_loss / forget_steps))
    
    # Evaluate on retain set
    logger.info("Evaluating on retain set...")
    retain_loader = torch.utils.data.DataLoader(
        retain_data["test"] if "test" in retain_data else retain_data["validation"],
        batch_size=eval_args.per_device_eval_batch_size,
        collate_fn=data_collator
    )
    
    retain_loss = 0
    retain_steps = 0
    with torch.no_grad():
        for batch in retain_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            retain_loss += outputs.loss.item()
            retain_steps += 1
    retain_perplexity = torch.exp(torch.tensor(retain_loss / retain_steps))
    
    # Compute metrics
    metrics = {
        "forget_loss": forget_loss / forget_steps,
        "forget_perplexity": forget_perplexity.item(),
        "retain_loss": retain_loss / retain_steps,
        "retain_perplexity": retain_perplexity.item(),
        "unlearning_efficiency": (forget_loss / forget_steps) / (retain_loss / retain_steps)
    }
    
    # Save results
    os.makedirs(eval_args.output_dir, exist_ok=True)
    with open(os.path.join(eval_args.output_dir, "eval_results.txt"), "w") as f:
        for key, value in metrics.items():
            f.write(f"{key} = {value}\n")
            logger.info(f"{key}: {value}")
    
    return metrics

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, EvalArguments))
    model_args, data_args, eval_args = parser.parse_args_into_dataclasses()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Evaluate
    metrics = evaluate(model_args, data_args, eval_args)

if __name__ == "__main__":
    main()