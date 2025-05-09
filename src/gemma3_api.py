# -*- coding: utf-8 -*-
"""
gemma3_api.py
"""

import requests
import json
import torch

def generate_gemma3_story(prompt):
    # Free up GPU memory
    torch.cuda.empty_cache()
    
    url = 'http://localhost:11434/api/generate'
    payload = {
        "model": "gemma3:4b",
        "prompt": prompt,
        "stream": False
    }
    
    response = requests.post(url, data=json.dumps(payload), headers={'Content-Type': 'application/json'})
    
    if response.status_code == 200:
        list_dict_words = []
        for line in response.text.split("\n"):
            if line.strip():
                try:
                    data = json.loads(line)
                    list_dict_words.append(data)
                except json.JSONDecodeError:
                    continue
        gemma3_response = " ".join([word.get('response', '') for word in list_dict_words])
        return gemma3_response
    else:
        print(f"❌ Request failed with status code: {response.status_code}")
        return None