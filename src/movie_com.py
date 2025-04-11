# -*- coding: utf-8 -*-
"""
movie_com.py
"""

import os
from moviepy.editor import *
from gtts import gTTS # Google Text-to-Speech
import time
from PIL import Image
import numpy as np
import re

from moviepy.config import change_settings

# Configure ImageMagick path
change_settings({"IMAGEMAGICK_BINARY": r"C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe"})


# Configuration
IMAGE_DIR = "C:/Users/mnafi/Documents/AI_Final_Project/Project_code/storyteller_ai/gen_img"  # Root folder with story folders
STORY_DIR = "C:/Users/mnafi/Documents/AI_Final_Project/Project_code/storyteller_ai/gen_story/"  # Folder with original text files
OUTPUT_DIR = "C:/Users/mnafi/Documents/AI_Final_Project/Project_code/storyteller_ai/gen_video/" # Where to save final videos
TEMP_AUDIO_DIR = "C:/Users/mnafi/Documents/AI_Final_Project/Project_code/storyteller_ai/gen_audio/"  # For temporary voice files

# Video settings
FPS = 24
VIDEO_SIZE = (1920, 1080)  # Full HD
FONT = "Arial"
FONT_SIZE = 40
TEXT_COLOR = "green"
TEXT_POS = ("center", "bottom")
TEXT_DURATION_OFFSET = 1  # Seconds to show text after audio ends
IMAGE_RESAMPLING = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS


def safe_delete(path, retries=3, delay=1):
    """Safely delete files with retries and delays"""
    for i in range(retries):
        try:
            os.remove(path)
            return True
        except PermissionError:
            if i < retries - 1:
                time.sleep(delay)
            continue
    return False


# Modified image processing function
def process_image(img_path, target_size):
    img = Image.open(img_path)
    img = img.resize(target_size, IMAGE_RESAMPLING)
    return img

def create_narration(text, filename):
    """Convert text to speech using gTTS"""
    tts = gTTS(text=text, lang='en', slow=False)
    tts.save(filename)
    time.sleep(1)  # Avoid rate limiting

def create_video_for_story(story_name):
    """Create video for a single story"""
    # Path setup
    story_file = os.path.join(STORY_DIR, f"{story_name}.txt")
    image_folder = os.path.join(IMAGE_DIR, story_name)
    output_path = os.path.join(OUTPUT_DIR, f"{story_name}.mp4")
    
    # # Read story content
    # with open(story_file, 'r') as f:
    #     lines = [line.strip() for line in f.readlines() if line.strip()]
    
    
    # --- Read and parse story content ---
    # We only want to keep lines that are not labels (like "**Scene 1:**") or,
    # if text is on the same line after the label, we extract just that text.
    raw_lines = []
    with open(story_file, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()
    
    parsed_lines = []
    # Regex to match something like "**Title:**" or "**Scene 1:**" at the start of a line
    label_pattern = re.compile(r"^\*\*(Title|Scene\s*\d+)\:\*\*")
    
    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            continue  # skip empty lines
        
        # If a line starts with something like **Title:** or **Scene X:**
        # we remove just that label portion and keep any text that might follow on the same line.
        if label_pattern.search(stripped):
            # Replace the label part with empty, then strip
            # e.g. "**Scene 1:** Some text" -> "Some text"
            no_label = label_pattern.sub("", stripped).lstrip(": ").strip()
            
            # if there's still text after removing the label, keep it
            if no_label:
                parsed_lines.append(no_label)
        else:
            # Otherwise, it's just a normal line, keep it as-is
            parsed_lines.append(stripped)

    # At this point, parsed_lines should have no label lines, only the text
    
    
    # Create clips and audio files list
    clips = []
    audio_files = []
    
    # Process each image and corresponding text
    for idx, line in enumerate(parsed_lines[:13]):  # use parsed_lines[:12]
        img_path = os.path.join(image_folder, f"image_{idx+1:02d}.png")
        audio_path = os.path.join(TEMP_AUDIO_DIR, f"{story_name}_{idx}.mp3")
        
        if not os.path.exists(img_path):
            print(f"\nMissing image: {img_path}")
            continue

        # Create narration audio first
        create_narration(line, audio_path)
        
        # Calculate duration after creating audio
        audio_duration = AudioFileClip(audio_path).duration
        duration = audio_duration + TEXT_DURATION_OFFSET

        # --- Process the image (resize) ---
        try:
            processed_img = process_image(img_path, VIDEO_SIZE)
            img_clip = ImageClip(np.array(processed_img)).set_duration(duration)
        except Exception as img_error:
            print(f"Error processing image {img_path}: {str(img_error)}")
            continue
        
        # Create audio clip
        #audio = AudioFileClip(audio_path)
        
        # Create audio clip
        try:
            audio = AudioFileClip(audio_path)
        except Exception as audio_error:
            print(f"Audio error: {audio_path} - {str(audio_error)}")
            continue

        
        # Create text clip
        try:
            txt_clip = TextClip(line,
                                fontsize=FONT_SIZE,
                                color=TEXT_COLOR,
                                font=FONT,
                                method='caption',
                                size=(VIDEO_SIZE[0], None))
            txt_clip = txt_clip.set_position(TEXT_POS).set_duration(duration)
        except Exception as text_error:
            print(f"Error creating text clip for line {idx+1}: {str(text_error)}")
            continue

        
        # Combine elements
        try:
            final_clip = CompositeVideoClip([img_clip, txt_clip.set_audio(audio)])
            final_clip = final_clip.set_duration(duration)  # Explicit duration set
            
        except Exception as e:
            print(f"Error creating clip for line {idx+1}: {str(e)}")
            continue
        

        
        clips.append(final_clip)


    # Concatenate all clips
    if clips:
        try:
            final_video = concatenate_videoclips(clips, method="compose")
            # Write output file
            final_video.write_videofile(
                output_path,
                fps=FPS,
                codec='libx264',
                audio_codec='aac',
                threads=4,
                ffmpeg_params=[
                '-crf', '23',            # Quality control (23 is default)
                '-movflags', '+faststart',  # Web optimization
                '-pix_fmt', 'yuv420p',   # Universal compatibility
                '-profile:v', 'main',    # H.264 profile
                '-level', '3.1'          # H.264 level
                ]
            )
        finally:
            # Properly close all clips after rendering
            final_video.close()
            for clip in clips:
                clip.close()
    else:
        print(f"\nNo valid clips created for {story_name}")
    
    # Cleanup temporary files
    if os.path.exists(audio_path):
        os.remove(audio_path)
    
    
     # Cleanup any remaining audio files
    for idx in range(len(lines[:13])):
        audio_path = os.path.join(TEMP_AUDIO_DIR, f"{story_name}_{idx}.mp3")
        safe_delete(audio_path)
        
        
def main():
    # Create directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)
    
    # Process all stories
    # for story_file in os.listdir(STORY_DIR):
    #     if story_file.endswith(".txt"):
    #         story_name = os.path.splitext(story_file)[0]
    #         print(f"\nProcessing: {story_name}")
    #         create_video_for_story(story_name)
    
    # Hardcoded story name (without .txt extension)
    story_name = "ollama_llama3_story"
    
    print(f"\nProcessing: {story_name}")
    create_video_for_story(story_name)
    
    print("\nVideo creation complete!")

if __name__ == "__main__":
    main()