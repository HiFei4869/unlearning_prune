import os
import sys
from typing import List, Optional

import fire
import torch
import transformers
from datasets import load_dataset
from loraprune.trainer_ul import LoRAPruneTrainer  # Changed to use unlearning version
from loraprune.utils_ul import freeze  # Changed to use unlearning version
from loraprune.lora import LoraConfig  # Changed to use unlearning version

from peft import (
    prepare_model_for_kbit_training,
)
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft.peft_model import get_peft_model_state_dict, set_peft_model_state_dict
from loraprune.peft_model import get_peft_model

IGNORE_INDEX = -100


def load_dataset_split(dataset_path: str, subset: str = None):
    """Load full dataset or specific subset"""
    if subset:
        return load_dataset(dataset_path, subset)
    if dataset_path.endswith(".json"):
        return load_dataset("json", data_files=dataset_path)
    return load_dataset(dataset_path)


def train(
    # model/data params
    base_model: str = "",  # the required argument
    forget_dataset: str = "",  # dataset to unlearn
    retain_dataset: str = "",  # dataset to retain
    forget_subset: Optional[str] = None,  # optional subset of forget dataset
    retain_subset: Optional[str] = None,  # optional subset of retain dataset
    output_dir: str = "output_dir",
    # training hyperparams
    nsamples: int = 25000,
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    # pruning hyperparams
    ratio: float = 0.5,
    init_ratio: float = 0,
    warmup_iters: float = 0.1,
    cooldown_iters: float = 0.1,
    prune_freq: int = 10,
    prune_metric: str = 'lora',  # options: lora|grad|magnitude
    # unlearning hyperparams
    unlearning_threshold: float = 0.5,  # threshold for unlearning score
    min_retain_performance: float = 0.9,  # minimum performance to maintain on retain set
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "down_proj",
        "up_proj"
    ],
    # llm hyperparams
    train_on_inputs: bool = True,
    add_eos_token: bool = True,
    group_by_length: bool = False,
    resume_from_checkpoint: str = None,
    save_steps: int = 0,
    save_total_limit: int = 0,
):
    """
    Train function with unlearning support.
    """
    print(
        f"Unlearning pruning with params:\n"
        f"base_model: {base_model}\n"
        f"forget_dataset: {forget_dataset}\n"
        f"retain_dataset: {retain_dataset}\n"
        f"forget_subset: {forget_subset}\n"
        f"retain_subset: {retain_subset}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"group_by_length: {group_by_length}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
        f"prune_metric: {prune_metric}\n"
        f"unlearning_threshold: {unlearning_threshold}\n"
        f"min_retain_performance: {min_retain_performance}\n"
    )

    assert base_model, "Please specify a --base_model, e.g. --base_model='meta-llama/Llama-2-7b-hf'"
    assert forget_dataset or retain_dataset, "Please specify at least one of --forget_dataset or --retain_dataset"
    
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # Ensure prompt is a string
        if not isinstance(prompt, str):
            raise ValueError(f"Expected string input, got {type(prompt)}")
            
        # Use encode instead of batch_encode_plus for single inputs
        encoding = tokenizer.encode(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding="max_length",
            return_tensors=None,
        )
        
        # Convert to list if it's not already
        input_ids = encoding if isinstance(encoding, list) else encoding.ids
        
        if (
            input_ids[-1] != tokenizer.eos_token_id
            and len(input_ids) < cutoff_len
            and add_eos_token
        ):
            input_ids.append(tokenizer.eos_token_id)
            
        # Create attention mask (1 for all tokens)
        attention_mask = [1] * len(input_ids)
        
        # Pad if needed
        if len(input_ids) < cutoff_len:
            padding_length = cutoff_len - len(input_ids)
            input_ids.extend([tokenizer.pad_token_id] * padding_length)
            attention_mask.extend([0] * padding_length)
            
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.copy()
        }
        return result

    def generate_prompt(data_point):
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["question"]}

### Response:
{data_point["answer"]}"""

    def generate_and_tokenize_prompt(data_point):
        # Debug: Print data point structure before tokenization
        print("\n===== DEBUG: TOKENIZATION =====")
        print(f"Data point type: {type(data_point)}")
        print(f"Data point keys: {list(data_point.keys())}")
        for key, value in data_point.items():
            print(f"  {key}: {type(value)} = {repr(value)[:100]}")
        
        full_prompt = generate_prompt(data_point)
        print(f"Generated prompt type: {type(full_prompt)}")
        print(f"Generated prompt (first 100 chars): {repr(full_prompt)[:100]}")
        
        try:
            tokenized_full_prompt = tokenize(full_prompt, add_eos_token)
            print(f"Tokenized prompt keys: {list(tokenized_full_prompt.keys())}")
            for key, value in tokenized_full_prompt.items():
                print(f"  {key}: {type(value)} = {type(value[0]) if value else None}")
                
            if not train_on_inputs:
                user_prompt = generate_prompt({**data_point, "answer": ""})
                tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
                user_prompt_len = len(tokenized_user_prompt["input_ids"])
                tokenized_full_prompt["labels"] = [
                    IGNORE_INDEX
                ] * user_prompt_len + tokenized_full_prompt["labels"][
                    user_prompt_len:
                ]  # could be sped up
            print("Tokenization successful!")
            return tokenized_full_prompt
        except Exception as e:
            print(f"Error during tokenization: {str(e)}")
            # Return a dummy tokenized prompt with the right structure
            # This helps us identify which examples are causing problems
            return {
                "input_ids": [0] * cutoff_len,
                "attention_mask": [0] * cutoff_len,
                "labels": [0] * cutoff_len,
                "error": str(e)
            }

    # Load and preprocess datasets
    if forget_dataset:
        print(f"Loading forget dataset: {forget_dataset}")
        forget_data = load_dataset_split(forget_dataset, forget_subset)
    else:
        forget_data = None
        print("No forget dataset found.")

    if retain_dataset:
        print(f"Loading retain dataset: {retain_dataset}")
        retain_data = load_dataset_split(retain_dataset, retain_subset)
    else:
        retain_data = None
        print("No retain dataset found.")

    # Configure LoRA
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        peft_type="LORA"
    )
    

    # print("\n===== DEBUG: DATASET STRUCTURE =====")
    # if forget_data:
    #     print("Forget dataset example:")
    #     example = forget_data["train"][0]
    #     print(f"Keys: {list(example.keys())}")
    #     for key, value in example.items():
    #         print(f"  {key}: {type(value)} = {value}")
    
    # if retain_data:
    #     print("\nRetain dataset example:")
    #     example = retain_data["train"][0]
    #     print(f"Keys: {list(example.keys())}")
    #     for key, value in example.items():
    #         print(f"  {key}: {type(value)} = {value}")
    # print("=====================================\n")

    model = get_peft_model(model, config)
    # Process datasets
    def process_dataset(data, split="train", max_samples=None):
        if not data:
            return None
        
        if val_set_size > 0 and split == "train":
            train_val = data["train"].train_test_split(
                test_size=0.2, shuffle=True, seed=42
            )
            processed_data = train_val["train"]
            if max_samples:
                processed_data = processed_data.select(range(min(max_samples, len(processed_data))))
            return processed_data.map(
                generate_and_tokenize_prompt,
                remove_columns=["question", "answer"],  # Remove original columns to avoid tensor conversion issues
                batched=False  # Process one example at a time
            )
        elif val_set_size > 0 and split == "val":
            train_val = data["train"].train_test_split(
                test_size=0.2, shuffle=True, seed=42
            )
            processed_data = train_val["test"]
            if max_samples:
                processed_data = processed_data.select(range(min(max_samples, len(processed_data))))
            return processed_data.map(
                generate_and_tokenize_prompt,
                remove_columns=["question", "answer"],  # Remove original columns to avoid tensor conversion issues
                batched=False  # Process one example at a time
            )
        else:
            processed_data = data["train"]
            if max_samples:
                processed_data = processed_data.select(range(min(max_samples, len(processed_data))))
            return processed_data.map(
                generate_and_tokenize_prompt,
                remove_columns=["question", "answer"],  # Remove original columns to avoid tensor conversion issues
                batched=False  # Process one example at a time
            )

    # Process forget and retain datasets
    forget_train = process_dataset(forget_data, "train", nsamples) if forget_data else None
    forget_val = process_dataset(forget_data, "val") if forget_data and val_set_size > 0 else None
    retain_train = process_dataset(retain_data, "train", nsamples) if retain_data else None
    retain_val = process_dataset(retain_data, "val") if retain_data and val_set_size > 0 else None

    # Initialize trainer with unlearning support
    trainer = LoRAPruneTrainer(
        model=model,
        forget_dataset=forget_train,
        retain_dataset=retain_train,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=0,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="no",  # Disable evaluation since we don't want to use eval dataset
            save_strategy="steps",
            eval_steps=None,  # Remove eval steps since evaluation is disabled
            save_steps=save_steps,
            output_dir=output_dir,
            save_total_limit=save_total_limit,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        # Pruning parameters
        ratio=ratio,
        init_ratio=init_ratio,
        warmup_iters=warmup_iters,
        cooldown_iters=cooldown_iters,
        prune_freq=prune_freq,
        prune_metric=prune_metric,
        # Unlearning parameters
        unlearning_threshold=unlearning_threshold,
        min_retain_performance=min_retain_performance,
    )

    model.config.use_cache = False

    # Handle state dict for saving
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    # Train the model
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save the final model
    model.save_pretrained(output_dir)

    print("\nTraining completed successfully!")
    
if __name__ == "__main__":
    fire.Fire(train)