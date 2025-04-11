# -*- coding: utf-8 -*-
"""
ft_gpt2_v2.py
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    get_linear_schedule_with_warmup,
    DataCollatorForLanguageModeling,
    GPT2Config
)
from datasets import load_dataset
from tqdm.auto import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import json

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataset Validation ---
def validate_json_files(dataset_path):
    valid_formats = 0
    for json_file in dataset_path.glob("*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    valid_formats += 1
                else:
                    f.seek(0)
                    if sum(1 for line in f) > 0:
                        valid_formats += 1
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON format in {json_file}: {e}")
    print(f"Validated {valid_formats} JSON file(s) successfully!")

# --- Story Formatting ---
def format_story(example):
    scenes_text = "\n".join(
        [f"Scene {scene['scene']}: {scene['text']}" 
         for scene in example.get("scenes", [])]
    )
    return {"text": f"Genre: {example.get('genre', 'unknown')}\n{scenes_text}"}

# --- Custom Data Collator ---
class CustomDataCollator(DataCollatorForLanguageModeling):
    def torch_call(self, examples):
        batch = super().torch_call(examples)
        # Explicit label shifting for causal LM
        batch["labels"] = batch["input_ids"].clone()
        return batch

# --- Dataset Setup ---
dataset_path = Path("C:/Users/mnafi/Documents/AI_Final_Project/Project_code/storyteller_ai/gen_data")  # Path to your story dataset
validate_json_files(dataset_path)

# Load dataset without streaming for initial validation
full_dataset = load_dataset(
    "json",
    data_files={"train": [str(p) for p in dataset_path.glob("*.json")]}
)["train"]

# Split dataset
split = full_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
valid_dataset = split["test"]

# Format datasets
train_dataset = train_dataset.map(format_story)
valid_dataset = valid_dataset.map(format_story)

# --- Tokenization ---
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding=False,  # Let collator handle padding
        return_tensors=None  # Return Python dict
    )

tokenized_train = train_dataset.map(
    tokenize_function,
    batched=True,
    batch_size=100,
    remove_columns=train_dataset.column_names  # Remove all original columns
)

tokenized_valid = valid_dataset.map(
    tokenize_function,
    batched=True,
    batch_size=100,
    remove_columns=valid_dataset.column_names  # Remove all original columns
)

# --- Data Collator ---
data_collator = CustomDataCollator(tokenizer=tokenizer, mlm=False)

# --- Data Sanity Check ---
sample_batch = data_collator([tokenized_train[0]])
print("\nData Sanity Check:")
print("Input IDs shape:", sample_batch["input_ids"].shape)
print("Labels shape:", sample_batch["labels"].shape)
print("Unique labels:", torch.unique(sample_batch["labels"]))

# --- Model Setup ---
config = GPT2Config.from_pretrained("gpt2")
config.use_cache = False
model = GPT2LMHeadModel.from_pretrained("gpt2", config=config).to(device)

# --- Training Setup ---
optimizer = AdamW(model.parameters(), lr=1e-5)  # Reduced learning rate
num_epochs = 3 # 40 mins
accumulation_steps = 4
steps_per_epoch = 250  # Reduced for testing
total_steps = steps_per_epoch * num_epochs
validation_interval = 200
batch_sizet=2

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=total_steps
)

# --- Metric Functions ---
def calculate_accuracy(logits, labels):
    predictions = torch.argmax(logits, dim=-1)
    mask = (labels != -100) & (labels != tokenizer.pad_token_id)
    valid_tokens = mask.sum()
    if valid_tokens == 0:
        return torch.tensor(0.0).to(device)
    correct = (predictions[mask] == labels[mask]).float().sum()
    return correct / valid_tokens

def evaluate(model, dataloader):
    model.eval()
    total_loss, total_acc, total_batches = 0.0, 0.0, 0
    
    with torch.no_grad():
        #for batch in tqdm(dataloader, desc="Validating"):
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            
            loss = outputs.loss
            acc = calculate_accuracy(outputs.logits, inputs["labels"])
            
            if not torch.isnan(loss) and loss.item() < 100:  # Loss sanity check
                total_loss += loss.item()
                total_acc += acc.item()
                total_batches += 1
            
            if total_batches >= 20:  # Smaller validation batches
                break
    
    return {
        "loss": total_loss / total_batches if total_batches > 0 else float('nan'),
        "accuracy": total_acc / total_batches if total_batches > 0 else float('nan')
    }

# --- Training Execution ---
model.train()
model.gradient_checkpointing_enable()

metrics = {
    "train_loss": [],
    "train_acc": [],
    "valid_loss": [],
    "valid_acc": [],
    "lr": [],
    "steps": []
}

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    train_loader = DataLoader(tokenized_train, batch_size=batch_sizet, collate_fn=data_collator)
    
    pbar = tqdm(total=steps_per_epoch, desc="Training")
    optimizer.zero_grad()
    
    for step, batch in enumerate(train_loader):
        # Forward pass
        inputs = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**inputs)
        
        # Loss check
        if torch.isnan(outputs.loss):
            print(f"\nNaN detected at step {step}! Stopping training.")
            break
        
        # Calculate metrics
        loss = outputs.loss / accumulation_steps
        acc = calculate_accuracy(outputs.logits, inputs["labels"])
        
        # Store metrics
        metrics["train_loss"].append(loss.item() * accumulation_steps)
        metrics["train_acc"].append(acc.item())
        metrics["lr"].append(scheduler.get_last_lr()[0])
        metrics["steps"].append(step + (epoch * steps_per_epoch))
        
        # Backward pass
        loss.backward()

        # Optimization step
        if (step + 1) % accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            pbar.update(accumulation_steps)
            pbar.set_postfix({
                "loss": f"{metrics['train_loss'][-1]:.4f}",
                "acc": f"{metrics['train_acc'][-1]:.2%}",
                "lr": f"{metrics['lr'][-1]:.2e}"
            })

        # Validation
        if step % validation_interval == 0 and step > 0:
            valid_loader = DataLoader(tokenized_valid, batch_size=batch_sizet, collate_fn=data_collator)
            valid_metrics = evaluate(model, valid_loader)
            
            if not np.isnan(valid_metrics["loss"]):
                metrics["valid_loss"].append(valid_metrics["loss"])
                metrics["valid_acc"].append(valid_metrics["accuracy"])
                
                print(f"\nStep {step} Validation:")
                print(f"Loss: {valid_metrics['loss']:.4f} | Accuracy: {valid_metrics['accuracy']:.2%}")

        if step >= steps_per_epoch:
            break

    pbar.close()
    torch.save(model.state_dict(), f"v2_storyteller_epoch{epoch+1}.pt")

# --- Final Metrics ---
def safe_mean(data, window_size=100):
    slice_data = data[-window_size:]
    return np.nanmean(slice_data) if len(slice_data) > 0 else float('nan')

final_metrics = {
    "train_loss": safe_mean(metrics["train_loss"]),
    "train_acc": safe_mean(metrics["train_acc"]),
    "valid_loss": safe_mean(metrics["valid_loss"], 10),
    "valid_acc": safe_mean(metrics["valid_acc"], 10)
}

print("\nTraining Complete!")
print(f"Final Training Loss: {final_metrics['train_loss']:.4f}")
print(f"Final Training Accuracy: {final_metrics['train_acc']:.2%}")
print(f"Final Validation Loss: {final_metrics['valid_loss']:.4f}")
print(f"Final Validation Accuracy: {final_metrics['valid_acc']:.2%}")

# Group by epoch
def group_by_epoch(data, steps_per_epoch):
    return [np.mean(data[i * steps_per_epoch: (i + 1) * steps_per_epoch]) for i in range(num_epochs)]

train_loss_per_epoch = group_by_epoch(metrics["train_loss"], steps_per_epoch)
train_acc_per_epoch = group_by_epoch(metrics["train_acc"], steps_per_epoch)
lr_per_epoch = group_by_epoch(metrics["lr"], steps_per_epoch)

# Safe validation loss/acc - already every 200 steps, so sample every epoch
valid_loss_per_epoch = metrics["valid_loss"][:num_epochs]
valid_acc_per_epoch = metrics["valid_acc"][:num_epochs]

# Plot
epochs = list(range(1, num_epochs + 1))
plt.figure(figsize=(12, 8))

# Loss Plot
plt.subplot(3, 1, 1)
plt.plot(epochs, train_loss_per_epoch, marker='o', label="Train Loss")
plt.plot(epochs, valid_loss_per_epoch, marker='x', label="Valid Loss")
plt.title("Training/Validation Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# Accuracy Plot
plt.subplot(3, 1, 2)
plt.plot(epochs, train_acc_per_epoch, marker='o', label="Train Acc")
plt.plot(epochs, valid_acc_per_epoch, marker='x', label="Valid Acc")
plt.title("Token Prediction Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

# Learning Rate Plot
plt.subplot(3, 1, 3)
plt.plot(epochs, lr_per_epoch, marker='o', color="green")
plt.title("Learning Rate per Epoch")
plt.xlabel("Epoch")
plt.ylabel("LR")
plt.grid(True)

plt.tight_layout()
plt.savefig("Variant2_metrics_per_epoch.png")
plt.show()
