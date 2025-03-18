# Import necessary libraries
import os
from dotenv import load_dotenv  # To load environment variables from .env
import torch
import transformers  # FIXED: Missing import
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Define the model ID for the pre-trained Google Gemma model
model_id = "google/gemma-2b"

# Configure quantization settings for the model using BitsAndBytes
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Load the model in 4-bit precision
    bnb_4bit_quant_type="nf4",  # Use NormalFloat4 (NF4) quantization
    bnb_4bit_compute_dtype=torch.bfloat16  # Use bfloat16 for computation
)

# Load the tokenizer and model for the Gemma model
tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",  # Automatically map to available devices (GPU/CPU)
    token=HF_TOKEN
)

# Test the model with a sample input
def test_model():
    text = "Quote: Imagination is more,"
    device = model.device  # FIXED: Use model's device instead of hardcoded "cuda:0"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=20)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

test_model()

# Load the dataset for fine-tuning
data = load_dataset("Abirate/english_quotes")
data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

# Define a formatting function to preprocess examples for training
def formatting_func(example):
    text = f"Quote: {example['quote'][0]}\nAuthor: {example['author'][0]}"
    return {"text": text}  # FIXED: Return dictionary instead of a list

# Configure LoRA (Low-Rank Adaptation) for efficient fine-tuning
lora_config = LoraConfig(
    r=8,  # Rank of the low-rank matrices
    lora_alpha=32,  # FIXED: Added alpha parameter
    lora_dropout=0.1,  # FIXED: Added dropout for stability
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM"
)

# Initialize the Supervised Fine-Tuning Trainer
trainer = SFTTrainer(
    model=model,  # The Gemma model
    train_dataset=data["train"],  # Use the quotes dataset
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,  # Small batch size (needed for 2B model)
        gradient_accumulation_steps=4,  # Accumulate gradients over 4 steps
        warmup_steps=2,  # Gradual learning rate warmup
        max_steps=100,  # Train for 100 steps
        learning_rate=2e-4,  # Learning rate
        fp16=True,  # Use 16-bit floating point
        logging_steps=1,  # Log every step
        output_dir="outputs",  # Save model outputs here
        optim="paged_adamw_8bit"  # Use memory-efficient optimizer
    ),
    peft_config=lora_config,  # Use LoRA fine-tuning
    formatting_func=formatting_func,  # Format the data
)

# Start training
trainer.train()
