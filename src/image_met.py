# -*- coding: utf-8 -*-
"""
image_met.py
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.nn.functional import adaptive_avg_pool2d
from scipy import linalg
from PIL import Image
from torchvision.models import inception_v3

# Configuration
IMAGE_DIR = "C:/Users/mnafi/Documents/AI_Final_Project/Project_code/storyteller_ai/gen_img"   # Root directory with generated images
RESULTS_CSV = "C:/Users/mnafi/Documents/AI_Final_Project/Project_code/storyteller_ai/per_met/image_met.csv"  # Output CSV file
BATCH_SIZE = 13 # Number of images processed at once.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # DEVICE is set to GPU if available, otherwise CPU.

# Initialize Inception model
inception_model = inception_v3(pretrained=True, transform_input=False).to(DEVICE)
inception_model.eval()

def get_activations(images, model, batch_size=20):
    """Calculate activations for FID and Inception Score"""
    activations = [] # List to store activations (features) for each batch.
    preds = [] # List to store the softmax predictions for each batch.
    
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(DEVICE) # Select a batch and move it to the DEVICE.
            output = model(batch) # Run the batch through the model.
            
            # If the output is only 2D (logits), skip pooling/  Check if the output is 2-dimensional (i.e., final logits).
            if output.dim() == 2:
                # If output is 2D (batch, classes), we assume it is already in the right shape.
                act = output.cpu()
            else:
                # Otherwise, if the output has spatial dimensions (e.g., [batch, channels, height, width]),
                # apply adaptive average pooling to convert it to [batch, channels].
                act = adaptive_avg_pool2d(output, (1, 1)) \
                        .squeeze(-1) \
                        .squeeze(-1) \
                        .cpu()
            
            activations.append(act)
            # Compute softmax on the output to obtain class probabilities and move them to CPU.
            preds.append(torch.nn.functional.softmax(output, dim=1).cpu())
    
    # Concatenate all batch activations and predictions into single tensors.
    return torch.cat(activations), torch.cat(preds)


def calculate_fid(act1, act2):
    ### Calculate Frechet Inception Distance ###
    
    # Compute mean and covariance for the two sets of activations.
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    
    # Compute the squared difference between the means.
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # Compute the square root of the product of the covariance matrices.
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    
    # If the result is a complex number (due to numerical issues), take only the real part.
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    # The FID is the sum of the squared difference and the trace of the covariance adjustments.
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def calculate_inception_score(preds, splits=10):
    # Ensure at least 2 images per split
    max_splits = min(splits, preds.shape[0] // 2)  # Avoid empty splits
    scores = []
    for i in range(max_splits):
        part = preds[i * (preds.shape[0] // max_splits) : (i+1) * (preds.shape[0] // max_splits)]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)

def process_folder(folder_path, real_activations=None):
    ### Process a folder of images and return metrics ###
    
    # Get list of image files (only PNG, JPG, or JPEG) in the folder.
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        return None # If no images are found, return None.

    # Load and preprocess images
    processed_images = []
    for img_file in image_files:
        img = Image.open(os.path.join(folder_path, img_file)).convert('RGB')
        img = img.resize((299, 299))  # Resize to the size expected by Inception.
        img = np.array(img).transpose((2, 0, 1)) / 255.0 # Convert image to numpy array, reorder dimensions (C,H,W) and normalize.
        img = (img - np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)) / np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        #img = torch.tensor(img).unsqueeze(0).float() # Convert to tensor, add a batch dimension, and cast to float.
        img = torch.tensor(img).float()  # Shape: [3, 299, 299]
        processed_images.append(img)
    
    # Create single tensor of all images or Concatenate all processed image tensors into one tensor
    #all_images = torch.cat(processed_images)
    
    # Stack along new batch dimension -> shape: (N, 3, 299, 299)
    all_images = torch.stack(processed_images, dim=0)
    
    # Calculate activations and predictions
    act, preds = get_activations(all_images, inception_model, BATCH_SIZE)
    
    # Calculate metrics -> Initialize a dictionary to store computed metrics.
    metrics = {
        'folder': os.path.basename(folder_path),
        'num_images': len(image_files)
    }
    
    # Inception Score -> Calculate the Inception Score from the predictions
    is_mean, is_std = calculate_inception_score(preds.numpy())
    metrics['inception_score_mean'] = is_mean
    metrics['inception_score_std'] = is_std
    
    # FID Score (requires real images statistics) -> If real image activations are provided, calculate FID.
    if real_activations is not None:
        fid = calculate_fid(real_activations, act.numpy())
        metrics['fid'] = fid
    
    return metrics

def main():
    # Collect all folders with generated images
    folders = [os.path.join(IMAGE_DIR, d) for d in os.listdir(IMAGE_DIR) 
              if os.path.isdir(os.path.join(IMAGE_DIR, d))]
    
    results = []
    for folder in folders:
        print(f"Processing: {folder}")
        metrics = process_folder(folder)
        if metrics:
            results.append(metrics)
    
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"Metrics saved to {RESULTS_CSV}")

if __name__ == "__main__":
    main()
    
#Inception Score (IS): Higher is better (typical range: 1-300)
#Fréchet Inception Distance (FID): Lower is better (typical range: 0-300)