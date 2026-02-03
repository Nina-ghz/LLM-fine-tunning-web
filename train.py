import torch
import shutil
import os
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)

# Configuration
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET_FILE = "dataset.jsonl"
OUTPUT_DIR = "lora_adapter"

def main():
    print(f"PyTorch Version: {torch.__version__}")

    # 1. Setup Model (4-bit Quantization for GPU efficiency)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading model... (This may take a minute)")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)

    # 2. LoRA Config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. Process Data
    print(f"Loading dataset from {DATASET_FILE}...")
    data = load_dataset("json", data_files=DATASET_FILE, split="train")

    def format_prompt(sample):
        full_text = f"User: {sample['instruction']}\nAssistant: {sample['response']}{tokenizer.eos_token}"
        return tokenizer(full_text, padding="max_length", truncation=True, max_length=128)

    tokenized_data = data.map(format_prompt)

    # 4. Train
    print("Starting Training...")
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_data,
        args=TrainingArguments(
            output_dir="checkpoints",
            per_device_train_batch_size=4,
            gradient_accumulation_steps=1,
            learning_rate=2e-4,
            max_steps=50,
            fp16=True,
            logging_steps=10,
            optim="paged_adamw_8bit"
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()

    # 5. Save Adapter and Tokenizer
    print(f"Saving model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()