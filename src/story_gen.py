# -*- coding: utf-8 -*-
"""
story_gen.py
"""
import torch
import transformers
from transformers import pipeline, AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import re

# Configuration
OUTPUT_DIR = "C:/Users/mnafi/Documents/AI_Final_Project/Project_code/storyteller_ai/gen_story"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model configurations
MODELS = {
    "gpt2-large": {
        "model_name": "gpt2-large",
        "device_map": "auto",
        "quantize": False,
        "max_length": 1024
    },
    "llama3-8b-ollama-api": {
        "model_name": "llama3.1:8b",  # Use the actual model name you have installed
        "stream": False   # Set to True if you want streaming
    },
    "deepseek-7b-api": {
        "model_name": "deepseek-r1:7b",  # Use the actual model name you have installed
        "stream": False   # Set to True if you want streaming
    },
    "gemma3-4b-api": {
        "model_name": "gemma3:4b",  # Use the actual model name you have installed
        "stream": False   # Set to True if you want streaming
    },
    "gpt2-finetuned-v1": {
        "model_name": "gpt2",  # Base model architecture
        "checkpoint": "C:/Users/mnafi/Documents/AI_Final_Project/Project_code/storyteller_ai/finetune_model/Variant_1/v1_finetuned_epoch3.pt",  # Path to your fine-tuned weights
        "device_map": "auto",
        "quantize": False,
        "max_length": 512,
        "custom_load": True  # Flag for custom loading
    },
    "gpt2-finetuned-v2": {
        "model_name": "gpt2",  # Base model architecture
        "checkpoint": "C:/Users/mnafi/Documents/AI_Final_Project/Project_code/storyteller_ai/finetune_model/Variant_2/v2_storyteller_epoch3.pt",  # Path to your fine-tuned weights
        "device_map": "auto",
        "quantize": False,
        "max_length": 512,
        "custom_load": True  # Flag for custom loading
    }
}


# =============================================
# Model selection - start
# =============================================

# For quick testing (low VRAM usage):
#SELECTED_MODELS = ["gpt2-large"]

import ollama
# llama model from local ollama interface:
#SELECTED_MODELS = ["llama3-8b-ollama-api"]  # Choose your preferred method

# deepseek model from local ollama interface:
#SELECTED_MODELS = ["deepseek-7b-api"]  # Choose your preferred method

# gemma3 model from local ollama interface:
#SELECTED_MODELS = ["gemma3-4b-api"]  # Choose your preferred method

# Fine tuned v1 GPT 2 model SELECTED_MODELS 
#SELECTED_MODELS = ["gpt2-finetuned-v1"]  # Add this to your selection list

# Fine tuned v2 GPT 2 model SELECTED_MODELS 
SELECTED_MODELS = ["gpt2-finetuned-v2"]  # Add this to your selection list


# =============================================
# Model selection - end
# =============================================

# Your prompt
#genres = ["Science Fiction", "Fantasy", "Mystery", "Horror", "Romance", "Adventure"]


# =============================================
# Case study selection - start
# =============================================

#Case Study 1: Fairytale for Children
#genres = ["Adventure"]
#Title = ["The Hare and the Tortoise"]


#Case Study 2: Scientific Fantasy Story
#genres = ["Science Fiction"]
#Title = ["Big hero 6"]

#Case Study 3: Romance Fantasy Story
#genres = ["Romance"]
#Title = ["Beauty and the Beast"]

# =============================================
# Case study selection -end
# =============================================


prompt_template = """Write a short story 'Title: The Hare and the Tortoise' in the Adventure genre with exactly 10 scenes. 
Clearly separate scenes using 'Scene {num}:' (replace {num} with scene numbers). 
Ensure a complete narrative arc."""

# Quantization config for 4-bit models
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 4x memory reduction
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"  # Optimal quantization
)

def sanitize_model_name(model_name):
    """Convert model names to safe filenames"""
    return re.sub(r"[\W_]+", "_", model_name)

def generate_and_save(model_config):
    try:
        
        print(f"\n Loading {model_config['model_name']}...")
        
        
       # =============================================
       # OLLAMA API IMPLEMENTATION
       # =============================================
        if "llama3" in model_config["model_name"]:
                        
            from llama_api import generate_llama_story
            
            # Generate story with dynamic prompt
            llama_response = generate_llama_story(prompt_template)
            
            if llama_response:
                # Save output with genre in filename
                filename = f"ollama_llama3_story.txt"
                output_path = os.path.join(OUTPUT_DIR, filename)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(llama_response)
                    
                # Free up GPU memory
                torch.cuda.empty_cache()
            
        elif "deepseek" in model_config["model_name"]:
            
            # #Api call
            from deepseek_api import generate_deepseek_story
            
            
            deepseek_response = generate_deepseek_story(prompt_template)
            
            if deepseek_response:
                # Save output with genre in filename
                filename = f"ollama_deepseek_story.txt"
                output_path = os.path.join(OUTPUT_DIR, filename)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(deepseek_response)
                    
                # Free up GPU memory
                torch.cuda.empty_cache()
        
        elif "gemma3" in model_config["model_name"]:
            
            # #Api call
            from gemma3_api import generate_gemma3_story
            
            
            gemma3_response = generate_gemma3_story(prompt_template)
            
            if gemma3_response:
                # Save output with genre in filename
                filename = f"ollama_gemma3_story.txt"
                output_path = os.path.join(OUTPUT_DIR, filename)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(gemma3_response)
                    
                # Free up GPU memory
                torch.cuda.empty_cache()


        # =============================================
        # HUGGING FACE IMPLEMENTATION 
        # =============================================
        
        else:
            
            # Free up GPU memory manually using torch.cuda.empty_cache()
            torch.cuda.empty_cache()

            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_config["model_name"])
            
            # Special cases
            
            if "gpt2-finetuned-v1" in model_config["model_name"]:
                print(" Loading custom fine-tuned model...")
                # Load base model
                model = AutoModelForCausalLM.from_pretrained(
                    model_config["model_name"],
                    device_map=model_config["device_map"]
                )
                # Load fine-tuned weights
                model.load_state_dict(torch.load(model_config["checkpoint"]))
                
            if "gpt2-finetuned-v2" in model_config["model_name"]:
                print(" Loading custom fine-tuned model...")
                # Load base model
                model = AutoModelForCausalLM.from_pretrained(
                    model_config["model_name"],
                    device_map=model_config["device_map"]
                )
                # Load fine-tuned weights
                model.load_state_dict(torch.load(model_config["checkpoint"]))
    
    
            else:
                # Original loading for other models
                model_params = {
                    "pretrained_model_name_or_path": model_config["model_name"],
                    "device_map": model_config["device_map"],
                    "trust_remote_code": model_config.get("trust_remote_code", False)
                }
    
                if model_config["quantize"]:
                    model_params["quantization_config"] = quantization_config
                    
                if "torch_dtype" in model_config:
                    model_params["torch_dtype"] = model_config["torch_dtype"]
    
                model = AutoModelForCausalLM.from_pretrained(**model_params)
            
            # Create pipeline
            generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device_map=model_config["device_map"]
            )
    
            print(f"⚡ Generating with {model_config['model_name']}...")
            result = generator(
                prompt_template,
                max_length=model_config["max_length"],
                temperature=0.9,
                top_p=0.92,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
    
            # Save output
            filename = f"{sanitize_model_name(model_config['model_name'])}_story.txt"
            output_path = os.path.join(OUTPUT_DIR, filename)
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result[0]['generated_text'])
                
            print(f" Saved to {output_path}")

    except Exception as e:
        print(f" Error with {model_config['model_name']}: {str(e)}")
    finally:
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()

# Run generation for selected models
for model_key in SELECTED_MODELS:
    if model_key in MODELS:
        generate_and_save(MODELS[model_key])
    else:
        print(f" Warning: {model_key} not found in model configurations")

print("\n Generation complete!")
