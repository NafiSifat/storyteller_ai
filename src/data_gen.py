# -*- coding: utf-8 -*-
"""
data_gen.py
"""

import json
import os
import ollama
from tqdm import tqdm

genres = ["Science Fiction", "Fantasy", "Mystery", "Horror", "Romance", "Adventure"]

prompt_template = """Write a short story in the {genre} genre with exactly 10 scenes. 
Clearly separate scenes using 'Scene {{num}}:' (replace {{num}} with scene numbers). 
Ensure a complete narrative arc."""

output_dir = "C:/Users/mnafi/Documents/AI_Final_Project/Project_code/storyteller_ai/gen_data"

# Create output directory if needed
os.makedirs(output_dir, exist_ok=True)


# New: Create a list to hold all stories
full_dataset = []

#for i in tqdm(range(1, 1001), desc="Generating Stories"):
for i in tqdm(range(1, 501), desc="Generating Stories"):
    genre = genres[i % len(genres)]
    prompt = prompt_template.format(genre=genre)
    
    response = ollama.generate(
        model='deepseek-r1:7b',
        prompt=prompt,
        options={'temperature': 0.7, 'num_predict': 2048}
    )
    story_text = response['response']

    # Extract scenes with improved parsing
    scenes = []
    scene_headers = [f"Scene {num}:" for num in range(1, 11)]
    
    for header in scene_headers:
        if header in story_text:
            start_idx = story_text.index(header) + len(header)
            end_idx = story_text.find("Scene", start_idx)
            
            scene_content = story_text[start_idx:end_idx].strip() if end_idx != -1 else story_text[start_idx:].strip()
            scene_number = int(header.split()[1].replace(":", ""))
            
            scenes.append({
                "scene": scene_number,
                "text": scene_content
            })

    story_data = {
        "title": f"Generated Story {i}",
        "genre": genre,
        "scenes": scenes
    }
    
    full_dataset.append(story_data)  # Add to dataset list

# Save as single JSON Lines file (recommended)
with open(os.path.join(output_dir, "full_dataset.jsonl"), "w", encoding="utf-8") as f:
    for story in full_dataset:
        f.write(json.dumps(story, ensure_ascii=False) + "\n")

# Alternative: Save as single JSON array
with open(os.path.join(output_dir, "full_dataset.json"), "w", encoding="utf-8") as f:
    json.dump(full_dataset, f, ensure_ascii=False, indent=4)

print("Dataset generation complete!")





















