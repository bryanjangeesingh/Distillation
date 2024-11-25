from dataclasses import dataclass
 
@dataclass
class dataset:
    file: str = None
    training_size: float = 1
    encoder_decoder: bool = False
    processed_data_path: str = "/nobackup/users/brytech/processed_dataset"  # Add this line

    # Distillation
    generated_by: str = None