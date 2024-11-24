from datasets import load_dataset
from typing import Dict, List

def process_sample(sample: Dict) -> Dict:
    """Process a single sample from the CodeSearchNet dataset."""
    # Format the input as a coding task
    prompt = f"### Instructions: Write a Python function based on the following description.\n\n### Description: {sample['docstring']}\n\n### Code:"
    
    # The target is the actual code
    target = sample['code']
    
    return {
        "input": prompt,
        "output": target
    }

def load_dataset_from_hub() -> Dict[str, List]:
    """Load and process the CodeSearchNet dataset."""
    # Load the Python subset of CodeSearchNet
    dataset = load_dataset("code_search_net", "python")
    
    # Process train split
    train_samples = [
        process_sample(sample) 
        for sample in dataset["train"]
        if sample['docstring'] and sample['code']  # Filter out empty samples
    ]
    
    # Process validation split
    val_samples = [
        process_sample(sample) 
        for sample in dataset["validation"]
        if sample['docstring'] and sample['code']  # Filter out empty samples
    ]
    
    return {
        "train": train_samples,
        "validation": val_samples
    }

if __name__ == "__main__":
    # Test the dataset loading
    dataset = load_dataset_from_hub()
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Validation samples: {len(dataset['validation'])}")
    
    # Print a sample
    print("\nSample input:")
    print(dataset['train'][0]['input'])
    print("\nSample output:")
    print(dataset['train'][0]['output'])
