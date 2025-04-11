# -*- coding: utf-8 -*-
"""
Image_gen.py
"""

import os
import glob
import time
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image, ImageDraw, ImageFont
import textwrap

# Configuration
TEXT_DIR = "C:/Users/mnafi/Documents/AI_Final_Project/Project_code/storyteller_ai/gen_story/"  # Directory containing text files
OUTPUT_DIR = "C:/Users/mnafi/Documents/AI_Final_Project/Project_code/storyteller_ai/gen_img"  # Root output directory
MAX_IMAGES_PER_STORY = 13

IMAGE_SIZE = (512, 512)  # Base image size
TEXT_BOX_WIDTH = 400      # Width for text panel
FONT_SIZE = 20          # Larger text
LINE_SPACING = 10       # More space between lines
MARGIN = 20             # Larger margins
WRAP_WIDTH = 50         # Characters per line (in wrapper)
BACKGROUND_COLOR = (255, 255, 255)  # White
TEXT_COLOR = (0, 0, 0)              # Black


#MODEL_NAME = "runwayml/stable-diffusion-v1-5"  # Base model -> a model that tends to produce 
                                                # more creative and varied art

MODEL_NAME = "stabilityai/stable-diffusion-2-1" #v2.1 is generally better for applications 
                                                # requiring higher photorealism and more precise prompt adherence.

#MODEL_NAME = "stabilityai/stable-diffusion-3"

#MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"


TORCH_DTYPE = torch.float16  # Use FP16 to save VRAM
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def create_pipeline():
    """Create and return the Stable Diffusion pipeline"""
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=TORCH_DTYPE,
        safety_checker=None,  # Disable safety checker to save VRAM
    ).to(DEVICE)
    
    # Optimizations for 8GB VRAM
    pipe.enable_attention_slicing()
    return pipe

def create_composite_image(image, prompt):
    """Create side-by-side image with text panel"""
    # Create new composite image
    composite_width = IMAGE_SIZE[0] + TEXT_BOX_WIDTH
    composite_height = IMAGE_SIZE[1]
    composite = Image.new("RGB", (composite_width, composite_height), BACKGROUND_COLOR)
    
    # Paste generated image on the left
    composite.paste(image, (0, 0))
    
    # Prepare text panel on the right
    try:
        font = ImageFont.truetype("arial.ttf", FONT_SIZE)
    except:
        font = ImageFont.load_default()
    
    # Set up text drawing
    draw = ImageDraw.Draw(composite)
    text_area_width = TEXT_BOX_WIDTH - 2*MARGIN
    text_area_height = IMAGE_SIZE[1] - 2*MARGIN
    
    # Wrap text to fit in the text panel
    wrapper = textwrap.TextWrapper(width=40)  # Characters per line
    wrapped_text = wrapper.fill(text=prompt)
    
    # Calculate text position
    x = IMAGE_SIZE[0] + MARGIN
    y = MARGIN
    
    # Draw text line by line
    for line in wrapped_text.split('\n'):
        if y + FONT_SIZE > text_area_height + MARGIN:
            line += "..."  # Add ellipsis if text overflows
            draw.text((x, y), line, font=font, fill=TEXT_COLOR)
            break
        draw.text((x, y), line, font=font, fill=TEXT_COLOR)
        y += FONT_SIZE + LINE_SPACING
    
    return composite



# def process_story_file(file_path, output_folder, pipe):
#     """Process a single story file and generate images"""
#     with open(file_path, "r") as f:
#         content = f.read().splitlines()  # Split into lines
    
#     # Take first MAX_IMAGES_PER_STORY lines as prompts
#     prompts = [line.strip() for line in content if line.strip()][:MAX_IMAGES_PER_STORY]
    
#     for i, prompt in enumerate(prompts):
#         try:
#             # Generate image
#             image = pipe(
#                 prompt,
#                 height=IMAGE_SIZE[1],
#                 width=IMAGE_SIZE[0],
#                 num_inference_steps=25
#             ).images[0]
            
#             # Create composite image with text panel
#             composite = create_composite_image(image, prompt)
            
#             # Save image
#             img_path = os.path.join(output_folder, f"image_{i+1:02d}.png")
#             composite.save(img_path)
#             print(f"Saved: {img_path}")
            
#             # Short delay to manage VRAM
#             time.sleep(1)
            
#         except Exception as e:
#             print(f"Error generating image for prompt: {prompt}\nError: {str(e)}")



def process_story_file(file_path, output_folder, pipe):
    """Process a single story file and generate images"""
    with open(file_path, "r") as f:
        content = f.read().splitlines()

    # Build a list of the prompt lines, excluding any line that starts with "**Title:**"
    # or "**Scene" (including up to "**Scene 10:**").
    prompts = []
    for line in content:
        line_stripped = line.strip()
        # Skip empty lines, the Title line, or lines that begin with **Scene
        if not line_stripped:
            continue
        if line_stripped.startswith("**Title:**"):
            # Remove the label and strip it again
            title_text = line_stripped.replace("**Title:**", "").strip()
            # Truncate to 10 words
            words = title_text.split()
            truncated_title = " ".join(words[:10])
            
            continue
        if line_stripped.startswith("**Scene"):  # covers Scene 1, Scene 2, ... Scene 10
            continue
        # If it passed those checks, it's text we want to keep
        prompts.append(line_stripped)
    
    # Take the first MAX_IMAGES_PER_STORY lines as prompts
    prompts = prompts[:MAX_IMAGES_PER_STORY]

    for i, prompt in enumerate(prompts):
        try:
            # Generate image
            image = pipe(
                prompt,
                height=IMAGE_SIZE[1],
                width=IMAGE_SIZE[0],
                num_inference_steps=25
            ).images[0]
            
            # Create composite image with text panel
            composite = create_composite_image(image, prompt)
            
            # Save image
            img_path = os.path.join(output_folder, f"image_{i+1:02d}.png")
            composite.save(img_path)
            print(f"Saved: {img_path}")
            
            # Short delay to manage VRAM
            time.sleep(1)
            
        except Exception as e:
            print(f"Error generating image for prompt: {prompt}\nError: {str(e)}")


def add_prompt_to_image(image, prompt):
    """Add prompt text to the bottom of the image"""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    
    # Calculate text position
    text_position = (10, img.height - 30)
    
    # Add text
    draw.text(text_position, prompt, fill="white")
    
    return img

def main():
    # Create output directory if not exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize pipeline
    pipe = create_pipeline()
    
    # Process all text files in TEXT_DIR
    for story_file in glob.glob(os.path.join(TEXT_DIR, "*.txt")):
        # Create output folder for each story
        story_name = os.path.splitext(os.path.basename(story_file))[0]
        output_folder = os.path.join(OUTPUT_DIR, story_name)
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"\nProcessing: {story_file}")
        process_story_file(story_file, output_folder, pipe)
        
    print("\nImage generation complete!")

if __name__ == "__main__":
    main()
    # Free up GPU memory manually using torch.cuda.empty_cache()
    torch.cuda.empty_cache()