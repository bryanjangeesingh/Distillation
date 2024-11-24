from datasets import load_dataset
from typing import Dict, List
from abc import ABC, abstractmethod

class BaseDatasetProcessor(ABC):
    """Base class for dataset processing with different prompting styles."""
    
    @abstractmethod
    def generate_prompt(self, code: str, docstring: str = None) -> str:
        """Generate model-specific prompt format."""
        pass
    
    def process_sample(self, sample: Dict) -> Dict:
        """Process a single sample from the CodeSearchNet dataset."""
        return {
            "input": self.generate_prompt(sample['code'], sample['docstring']),
            "output": sample['code']
        }
    
    def load_dataset_from_hub(self) -> Dict[str, List]:
        """Load and process the CodeSearchNet dataset."""
        dataset = load_dataset("code_search_net", "python")
        
        train_samples = [
            self.process_sample(sample) 
            for sample in dataset["train"]
            if sample['docstring'] and sample['code']
        ]
        
        val_samples = [
            self.process_sample(sample) 
            for sample in dataset["validation"]
            if sample['docstring'] and sample['code']
        ]
        
        return {
            "train": train_samples,
            "validation": val_samples
        }

class StudentDatasetProcessor(BaseDatasetProcessor):
    """Dataset processor for student model (CodeLlama)."""
    
    def generate_prompt(self, code: str, docstring: str = None) -> str:
        """Format input for CodeLlama."""
        return f"# Complete the following Python function:\n\n{docstring if docstring else ''}\n{code}"

class TeacherDatasetProcessor(BaseDatasetProcessor):
    """Dataset processor for teacher model (WizardCoder)."""
    
    def generate_prompt(self, code: str, docstring: str = None) -> str:
        """Format input for WizardCoder."""
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

        ### Instruction:
        Create a Python script for this problem:
        {docstring if docstring else ''}

        {code}

        ### Response:"""
        return prompt

if __name__ == "__main__":
    # Test both dataset processors
    student_processor = StudentDatasetProcessor()
    teacher_processor = TeacherDatasetProcessor()
    
    # Load datasets with different prompting styles
    student_dataset = student_processor.load_dataset_from_hub()
    teacher_dataset = teacher_processor.load_dataset_from_hub()
    
    # Print samples from both datasets
    print("Student Model Sample:")
    print(student_dataset['train'][0]['input'])
    print("\nTeacher Model Sample:")
    print(teacher_dataset['train'][0]['input'])
