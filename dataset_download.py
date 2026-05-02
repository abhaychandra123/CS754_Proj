#!/usr/bin/env python3
"""
Downloads a subset of the ImageNet 128x128 dataset for evaluation.
"""

import argparse
import os
import numpy as np
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Download dataset for evaluation")
    parser.add_argument("--num-images", type=int, default=50, help="Number of images to download")
    parser.add_argument("--save-dir", type=str, default="dataset/images", help="Directory to save images")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use")
    
    args = parser.parse_args()
    
    print(f"Loading dataset from benjamin-paine/imagenet-1k-128x128 ({args.split} split)...")
    
    # We use streaming to avoid downloading the entire 1.2M image dataset
    ds = load_dataset("benjamin-paine/imagenet-1k-128x128", split=args.split, streaming=True)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"Downloading and saving {args.num_images} images to {args.save_dir}...")
    
    count = 0
    with tqdm(total=args.num_images) as pbar:
        for item in ds:
            image = item['image']
            
            # Ensure we're working with standard RGB images
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            file_path = os.path.join(args.save_dir, f"img_{count:04d}.png")
            image.save(file_path)
            
            count += 1
            pbar.update(1)
            
            if count >= args.num_images:
                break
                
    print(f"Successfully downloaded {count} images to {args.save_dir}/")

if __name__ == "__main__":
    main()
