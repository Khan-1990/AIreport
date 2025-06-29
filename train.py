import os
import torch
import argparse
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

def main(model_path, data_path, output_dir):
    """
    Main function to run the fine-tuning process from a local model path.
    """
    # Token is only needed if scripts were to download, but good practice to have available
    hf_token = os.environ.get("HF_TOKEN")

    # --- 1. Configure Model Quantization (QLoRA) ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # --- 2. Load Base Model and Tokenizer from Local Path ---
    print(f"Loading base model from local path: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, # Load from the local directory on the Northflank volume
        quantization_config=bnb_config,
        device_map={"": 0},
        token=hf_token # Token might still be useful for tokenizer or config checks
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    print(f"Loading tokenizer from local path: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- 3. Configure PEFT (LoRA) ---
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    # --- 4. Load Dataset ---
    dataset_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.jsonl') or f.endswith('.json')]
    print(f"Loading dataset from files: {dataset_files}")
    dataset = load_dataset("json", data_files=dataset_files, split="train")
    print(f"Dataset loaded successfully with {len(dataset)} examples.")

    # --- 5. Set Training Arguments ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit",
        logging_steps=25,
        learning_rate=2e-4,
        bf16=True,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="none",
    )

    # --- 6. Initialize and Start Trainer ---
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=2048,
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
    )
    
    print("\n--- Starting Fine-Tuning ---")
    trainer.train()
    print("--- Fine-Tuning Completed ---")

    # --- 7. Save Final Model Adapter ---
    print(f"Saving fine-tuned model adapter to: {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("--- Model Adapter Saved Successfully ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune an LLM.")
    parser.add_argument("--model_path", type=str, required=True, help="The local directory path to the base model.")
    parser.add_argument("--data_path", type=str, required=True, help="The path to the folder containing training data JSON files.")
    parser.add_argument("--output_dir", type=str, required=True, help="The directory to save the fine-tuned model adapter.")
    
    args = parser.parse_args()
    main(args.model_path, args.data_path, args.output_dir)
```
---
```python