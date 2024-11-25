#!/usr/bin/env python3
import os
import json
import torch
import argparse
from glob import glob
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List
from transformers import AutoTokenizer

# Maximum sequence length for tokenization
MAX_LENGTH = 512

def load_jsonl_files(base_path: str, split: str) -> List[Dict]:
    """Load and combine all JSONL files for a specific split."""
    split_path = os.path.join(base_path, "python/final/jsonl", split)
    print(f"Looking for files in: {split_path}")
    all_samples = []
    jsonl_files = glob(os.path.join(split_path, f"python_{split}_*.jsonl"))
    print(f"Found {len(jsonl_files)} JSONL files")
    
    if not jsonl_files:
        raise ValueError(f"No JSONL files found in {split_path}")
    
    for file_path in tqdm(jsonl_files, desc=f"Loading {split} files"):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    sample = json.loads(line)
                    all_samples.append(sample)
                except json.JSONDecodeError:
                    print(f"Error decoding line in {file_path}")
                except Exception as e:
                    print(f"Error processing line in {file_path}: {str(e)}")
    
    print(f"Total samples loaded: {len(all_samples)}")
    return all_samples

def process_sample(tokenizer, sample: Dict, model_type: str) -> Dict:
    """Process a single sample from the dataset."""
    try:
        code = sample.get('code', '')
        docstring = sample.get('docstring', '')
        
        if not code or not docstring:
            return None
        
        # Clean up the docstring and code
        docstring = docstring.strip()
        code = code.strip()
        
        if len(docstring) < 10 or len(code) < 10:
            return None
        
        # Generate the appropriate prompt based on model type
        if model_type == "teacher":
            prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Write a Python function based on this description:
{docstring}

### Response:
Here's the implementation:
{code}"""
        else:  # student
            prompt = f"""# Function description:
{docstring}

{code}"""
        
        # Tokenize the prompt
        model_inputs = tokenizer(
            prompt,
            truncation=True,
            padding='max_length',
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        
        # Create labels
        labels = tokenizer(
            code,
            truncation=True,
            padding='max_length',
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        
        return {
            "input_ids": model_inputs["input_ids"].squeeze(),
            "attention_mask": model_inputs["attention_mask"].squeeze(),
            "labels": labels["input_ids"].squeeze()
        }
            
    except Exception as e:
        print(f"Error processing sample: {str(e)}")
        return None

def process_and_save_split(input_path: str, output_path: str, split: str, tokenizer, model_type: str):
    """Process an entire split and save it to disk."""
    # Create output directory if it doesn't exist
    split_output_path = os.path.join(output_path, split)
    os.makedirs(split_output_path, exist_ok=True)
    
    # Load raw data
    raw_data = load_jsonl_files(input_path, split)
    
    # Process samples
    processed_samples = []
    for idx, sample in enumerate(tqdm(raw_data, desc=f"Processing {split} samples")):
        processed = process_sample(tokenizer, sample, model_type)
        if processed is not None:
            # Save every 10000 samples to a new file to avoid memory issues
            if len(processed_samples) >= 10000:
                file_num = idx // 10000
                torch.save(
                    processed_samples,
                    os.path.join(split_output_path, f"{split}_{file_num}.pt")
                )
                processed_samples = []
            processed_samples.append(processed)
    
    # Save any remaining samples
    if processed_samples:
        file_num = len(raw_data) // 10000
        torch.save(
            processed_samples,
            os.path.join(split_output_path, f"{split}_{file_num}.pt")
        )
    
    print(f"Finished processing {split} split")

def main():
    parser = argparse.ArgumentParser(description='Preprocess CodeSearchNet dataset')
    parser.add_argument('--input_path', type=str, required=True, default="/Users/bryanjangeesingh/Downloads/python/python/final/jsonl",
                      help='Path to the raw CodeSearchNet dataset')
    parser.add_argument('--output_path', type=str, required=True, default="/Users/bryanjangeesingh/Downloads/processed_dataset",
                      help='Path to save processed dataset')
    parser.add_argument('--model_type', type=str, choices=['teacher', 'student'],
                      required=True, help='Type of model to preprocess for')
    parser.add_argument('--model_name', type=str, 
                      default='Salesforce/codegen-350M-mono',
                      help='Name of the model to use for tokenization')
    args = parser.parse_args()
    
    # Initialize tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Process each split
    for split in ['train', 'valid', 'test']:
        process_and_save_split(
            args.input_path,
            args.output_path,
            split,
            tokenizer,
            args.model_type
        )

if __name__ == "__main__":
    main()