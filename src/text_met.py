# -*- coding: utf-8 -*-
"""
text_met.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""


import csv
import evaluate
import torch
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModelForCausalLM

nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt_tab', quiet=True)

# --------------------------------------------------
# 1. Perplexity Calculation (Intrinsic Metric)
# --------------------------------------------------
def calculate_perplexity(generated_text, model, tokenizer):
    inputs = tokenizer(generated_text, return_tensors="pt", padding=True, truncation=True)
    inputs["labels"] = inputs["input_ids"]
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    if outputs.loss is None:
        return None  # Handle edge cases
    
    return torch.exp(outputs.loss).item()

# --------------------------------------------------
# 2. BLEU Score (Reference-Based Metric)
# --------------------------------------------------
def calculate_bleu(generated_text, reference_text):
    gen_tokens = word_tokenize(generated_text.lower().strip())
    ref_tokens = [word_tokenize(reference_text.lower().strip())]
    
    return sentence_bleu(
        ref_tokens, gen_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=SmoothingFunction().method1
    )

# --------------------------------------------------
# 3. ROUGE Score (Reference-Based Metric)
# --------------------------------------------------
def calculate_rouge(generated_text, reference_text, rouge):
    generated_clean = ' '.join(generated_text.split())
    reference_clean = ' '.join(reference_text.split())
    
    scores = rouge.compute(
        predictions=[generated_clean],
        references=[reference_clean],
        use_stemmer=True,
        use_aggregator=False
    )
    
    return {key: values[0] for key, values in scores.items()}

# --------------------------------------------------
# Main Execution
# --------------------------------------------------
if __name__ == "__main__":
    # Configuration
    GENERATED_DIR = "C:/Users/mnafi/Documents/AI_Final_Project/Project_code/storyteller_ai/gen_story/"
    REFERENCE_PATH = "C:/Users/mnafi/Documents/AI_Final_Project/Project_code/storyteller_ai/ref_story/The_Hare_and_the_Tortoise.txt"
    #REFERENCE_PATH = "C:/Users/mnafi/Documents/AI_Final_Project/Project_code/storyteller_ai/ref_story/The_magical_forest_ref.txt"
    #REFERENCE_PATH = "C:/Users/mnafi/Documents/AI_Final_Project/Project_code/storyteller_ai/ref_story/The_magical_forest_ref.txt"
    OUTPUT_CSV = "C:/Users/mnafi/Documents/AI_Final_Project/Project_code/storyteller_ai/per_met/text_performance_metrics.csv"

    # Load reference story
    def load_reference():
        try:
            with open(REFERENCE_PATH, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error loading reference: {str(e)}")
            exit()

    reference_story = load_reference()
    
    # Initialize metrics tools
    try:
        print("\n Loading evaluation models...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        perplexity_model = AutoModelForCausalLM.from_pretrained("gpt2")
        rouge = evaluate.load('rouge', keep_in_memory=True)
    except Exception as e:
        print(f"Error initializing models: {str(e)}")
        exit()

    # Process files
    print(f"\n Processing files...")
    results = []
    for filename in os.listdir(GENERATED_DIR):
        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(GENERATED_DIR, filename)
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                generated_text = f.read().strip()
                
            if not generated_text:
                print(f"Skipping empty file: {filename}")
                continue

            # Calculate metrics
            metrics = {
                "filename": filename,
                "perplexity": calculate_perplexity(generated_text, perplexity_model, tokenizer),
                "bleu": calculate_bleu(generated_text, reference_story),
                **calculate_rouge(generated_text, reference_story, rouge)
            }
            
            results.append(metrics)
            print(f"\n Processed: {filename}")
            
        except Exception as e:
            print(f" Error processing {filename}: {str(e)}")

    # Save results
    if results:
        keys = results[0].keys()
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n Results saved to {OUTPUT_CSV}")
    else:
        print(" No valid files processed")

    
#set CUDA_LAUNCH_BLOCKING=1
#python text_met.py
#cd C:\Users\mnafi\Documents\AI_Final_Project\Project_code\storyteller_ai\src