# -*- coding: utf-8 -*-
"""
ft_gpt2_v1.py
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

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataset Loading & Filtering ---
dataset_folder = Path('C:/Users/mnafi/Documents/AI_Final_Project/Project_code/Dataset/BookCorpus/books1/epubtxt')
full_dataset = load_dataset("text", data_dir=str(dataset_folder), streaming=True)["train"]

# Filter out empty texts before tokenization
full_dataset = full_dataset.filter(lambda x: len(x["text"].strip()) > 0)

# Split into train/validation (90/10)
validation_ratio = 0.1
buffer_size = 10000

train_dataset = full_dataset.shuffle(buffer_size).skip(int(buffer_size * validation_ratio))
valid_dataset = full_dataset.take(int(buffer_size * validation_ratio))

# --- Model & Tokenizer Setup ---
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

config = GPT2Config.from_pretrained("gpt2")
config.use_cache = False
model = GPT2LMHeadModel.from_pretrained("gpt2", config=config).to(device)

# --- Fixed Tokenization ---
def tokenize_function(examples):
    # Handle empty text edge case
    if len(examples["text"].strip()) == 0:
        return {"input_ids": [], "attention_mask": []}
    
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding='max_length',
        return_tensors=None  # Critical fix: return lists instead of tensors
    )
    return tokenized

# Process individual examples
tokenized_train = train_dataset.map(tokenize_function, batched=False, remove_columns=["text"])
tokenized_valid = valid_dataset.map(tokenize_function, batched=False, remove_columns=["text"])

# --- Fixed Data Loading ---
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

def stream_data_loader(dataset, batch_size=4):
    batch = []
    for example in dataset:
        # Skip empty examples
        if not example["input_ids"]:
            continue
            
        # Convert to tensors with proper dimensions
        tensor_example = {
            "input_ids": torch.tensor(example["input_ids"]),
            "attention_mask": torch.tensor(example["attention_mask"]),
            "labels": torch.tensor(example["input_ids"])  # For language modeling
        }
        batch.append(tensor_example)
        
        if len(batch) >= batch_size:
            collated = data_collator(batch)
            # Check for valid batches
            if not (collated["labels"] == -100).all():
                yield collated
            batch = []
    
    # Handle remaining samples
    if batch:
        collated = data_collator(batch)
        if not (collated["labels"] == -100).all():
            yield collated

# --- Training Setup ---
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 3
accumulation_steps = 4
#steps_per_epoch = 1000  # Reduced for testing   # 10-20 min/epoch
steps_per_epoch = 5000 # 45 mins/epoch
#steps_per_epoch = 25000  # Process full dataset # 10.4 hours/epoch
total_steps = steps_per_epoch * num_epochs
validation_interval = 500
batch_sizet=4

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=500,
    num_training_steps=total_steps
)

# --- Metric Functions ---
def calculate_accuracy(logits, labels):
    predictions = torch.argmax(logits, dim=-1)
    mask = labels != -100  # Ignore padding tokens
    if mask.sum() == 0:
        return torch.tensor(0.0).to(device)
    correct = (predictions[mask] == labels[mask]).float().sum()
    total = mask.float().sum()
    return correct / total

def evaluate(model, dataloader):
    model.eval()
    total_loss, total_acc, total_batches = 0, 0, 0
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            
            # Skip invalid batches
            if (inputs["labels"] == -100).all():
                continue
                
            outputs = model(**inputs)
            loss = outputs.loss
            acc = calculate_accuracy(outputs.logits, inputs["labels"])
            
            total_loss += loss.item()
            total_acc += acc.item()
            total_batches += 1
            
            if total_batches >= 50:  # Limit validation steps
                break
    
    if total_batches == 0:
        return {"loss": float('nan'), "accuracy": float('nan')}
    
    return {
        "loss": total_loss / total_batches,
        "accuracy": total_acc / total_batches
    }

# --- Training Loop ---
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
    shuffled_train = tokenized_train.shuffle(buffer_size)
    train_loader = stream_data_loader(shuffled_train, batch_size=batch_sizet)
    
    pbar = tqdm(total=steps_per_epoch, desc="Training")
    optimizer.zero_grad()
    
    for step, batch in enumerate(train_loader):
        # Training step
        inputs = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**inputs)
        
        # Calculate metrics
        loss = outputs.loss / accumulation_steps
        acc = calculate_accuracy(outputs.logits, inputs["labels"])
        
        # Store metrics
        metrics["train_loss"].append(loss.item() * accumulation_steps)
        metrics["train_acc"].append(acc.cpu().item())
        metrics["lr"].append(scheduler.get_last_lr()[0])
        metrics["steps"].append(step + (epoch * steps_per_epoch))
        
        loss.backward()

        # Optimization step with gradient clipping
        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
            valid_loader = stream_data_loader(tokenized_valid, batch_size=batch_sizet)
            valid_metrics = evaluate(model, valid_loader)
            
            if not np.isnan(valid_metrics["loss"]):
                metrics["valid_loss"].append(valid_metrics["loss"])
                metrics["valid_acc"].append(valid_metrics["accuracy"])
                print(f"\nStep {step} Validation:")
                print(f"Loss: {valid_metrics['loss']:.4f} | Accuracy: {valid_metrics['accuracy']:.2%}")
            else:
                print("\nValidation skipped - no valid batches")

        if step >= steps_per_epoch:
            break

    pbar.close()
    torch.save(model.state_dict(), f"v1_finetuned_epoch{epoch+1}.pt")

# --- Final Metrics ---
final_metrics = {
    "train_loss": np.nanmean(metrics["train_loss"][-validation_interval:]),
    "train_acc": np.nanmean(metrics["train_acc"][-validation_interval:]),
    "valid_loss": np.nanmean(metrics["valid_loss"][-5:]),
    "valid_acc": np.nanmean(metrics["valid_acc"][-5:]),
}

if not np.isnan(final_metrics["valid_loss"]):
    final_metrics["perplexity"] = torch.exp(torch.tensor(final_metrics["valid_loss"])).item()
else:
    final_metrics["perplexity"] = float('nan')

print("\nTraining Complete!")
print(f"Final Training Loss: {final_metrics['train_loss']:.4f} | Accuracy: {final_metrics['train_acc']:.2%}")
print(f"Final Validation Loss: {final_metrics['valid_loss']:.4f} | Accuracy: {final_metrics['valid_acc']:.2%}")
print(f"Validation Perplexity: {final_metrics['perplexity']:.4f}")


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
plt.savefig("Variant1_metrics_per_epoch.png")
plt.show()