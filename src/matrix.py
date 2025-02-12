import yaml
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
import json
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Any, Tuple, Optional
from .errors import ValidationError, get_context

class NeuroMatrix(Dataset):
    """
    Custom dataset class for NEURO data format.
    Supports both training data and model weights.
    """
    
    def __init__(self, data: List[Dict[str, Any]], meta: Optional[Dict[str, Any]] = None):
        self.data = data
        self.meta = meta or {}
        self.model_config = {}
        self._validate_data()
        self._prepare_tensors()
    
    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> 'NeuroMatrix':
        """
        Creates a NeuroMatrix from a dictionary representation.
        
        Args:
            data_dict: Dictionary containing data and metadata
            
        Returns:
            A new NeuroMatrix instance
        """
        if not isinstance(data_dict, dict):
            raise ValidationError(
                "Invalid data format",
                context=get_context(),
                suggestions=["Data should be a dictionary with 'meta' and 'data' keys"],
                details={"provided_type": type(data_dict).__name__}
            )
            
        meta = data_dict.get('meta', {})
        data = data_dict.get('data', [])
        
        if not isinstance(data, list):
            raise ValidationError(
                "Invalid data format",
                context=get_context(),
                suggestions=["Data should be a list of input-output pairs"],
                details={"provided_type": type(data).__name__}
            )
            
        return cls(data, meta)
    
    @classmethod
    def from_tensors(cls, inputs: torch.Tensor, targets: torch.Tensor) -> 'NeuroMatrix':
        """
        Creates a NeuroMatrix from PyTorch tensors.
        
        Args:
            inputs: Input tensor
            targets: Target tensor
            
        Returns:
            A new NeuroMatrix instance
        """
        if not isinstance(inputs, torch.Tensor) or not isinstance(targets, torch.Tensor):
            raise ValidationError(
                "Invalid tensor types",
                context=get_context(),
                suggestions=["Both inputs and targets must be PyTorch tensors"],
                details={
                    "inputs_type": type(inputs).__name__,
                    "targets_type": type(targets).__name__
                }
            )
            
        if len(inputs) != len(targets):
            raise ValidationError(
                "Mismatched tensor lengths",
                context=get_context(),
                suggestions=["Input and target tensors must have the same length"],
                details={
                    "inputs_length": len(inputs),
                    "targets_length": len(targets)
                }
            )
            
        # Convert tensors to list of dictionaries
        data = []
        for x, y in zip(inputs, targets):
            data.append({
                'input': x.tolist(),
                'output': y.tolist() if y.dim() > 0 else [y.item()]
            })
            
        meta = {
            'created': 'from_tensors',
            'input_shape': list(inputs.shape[1:]),
            'output_shape': list(targets.shape[1:]) if targets.dim() > 1 else [1]
        }
        
        return cls(data, meta)
    
    def _validate_data(self):
        """Validates the data format."""
        if not isinstance(self.data, list):
            raise ValidationError(
                "Invalid data format",
                context=get_context(),
                suggestions=["Data must be a list of dictionaries"],
                details={"provided_type": type(self.data).__name__}
            )
            
        for i, item in enumerate(self.data):
            if not isinstance(item, dict):
                raise ValidationError(
                    f"Invalid data item at index {i}",
                    context=get_context(),
                    suggestions=["Each item must be a dictionary with 'input' and 'output' keys"],
                    details={"item_type": type(item).__name__}
                )
                
            if 'input' not in item or 'output' not in item:
                raise ValidationError(
                    f"Missing required keys in data item at index {i}",
                    context=get_context(),
                    suggestions=["Each item must have 'input' and 'output' keys"],
                    details={"available_keys": list(item.keys())}
                )
    
    def _prepare_tensors(self):
        """Converts data to PyTorch tensors."""
        self.inputs = []
        self.targets = []
        
        for item in self.data:
            x = torch.tensor(item['input'], dtype=torch.float32)
            y = torch.tensor(item['output'], dtype=torch.float32)
            self.inputs.append(x)
            self.targets.append(y)
            
        self.inputs = torch.stack(self.inputs)
        self.targets = torch.stack(self.targets)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]
    
    def to_torch_dataset(self) -> 'NeuroMatrix':
        """Returns self as it's already a PyTorch Dataset."""
        return self

    def add_meta(self, name, created, description):
        self.meta = {
            'name': name,
            'created': created,
            'description': description
        }

    def add_model_config(self, config):
        self.model_config = config

    def add_data_point(self, input_data, output_data):
        self.data.append({
            'input': input_data,
            'output': output_data
        })

    @staticmethod
    def load(file_path):
        """Load matrix data from a file."""
        with open(file_path, 'r') as f:
            try:
                # Try JSON first
                data = json.load(f)
            except json.JSONDecodeError:
                # If JSON fails, try YAML
                f.seek(0)  # Reset file pointer
                data = yaml.safe_load(f)
            
        matrix = NeuroMatrix()
        
        # Load metadata if present
        if 'meta' in data:
            matrix.add_meta(
                data['meta'].get('name', ''),
                data['meta'].get('created', ''),
                data['meta'].get('description', '')
            )
        
        # Load model config if present
        if 'model_config' in data:
            matrix.add_model_config(data['model_config'])
        
        # Load data
        matrix.data = data.get('data', [])
        
        return matrix

    def save(self, file_path):
        """Save matrix data to a file."""
        data = {
            'meta': self.meta,
            'model_config': self.model_config,
            'data': self.data
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    def validate(self):
        """Validate the matrix format"""
        # Check if meta information exists
        if not self.meta:
            raise ValueError("Missing meta information")

        # Check if data exists
        if not self.data:
            raise ValueError("No data points found")

        # Check if all data points have input and output
        for i, point in enumerate(self.data):
            if 'input' not in point or 'output' not in point:
                raise ValueError(f"Data point {i} missing input or output")

        # Check if model configuration exists
        if not self.model_config:
            raise ValueError("Missing model configuration")

        return True


# Example usage
if __name__ == '__main__':
    # Create a new dataset
    matrix = NeuroMatrix()
    matrix.add_meta("Sample Dataset", "Example of binary classification")
    
    # Add some sample data
    matrix.add_data_point([0.5, 0.8, 0.3], [1.0])
    matrix.add_data_point([0.2, 0.4, 0.1], [0.0])
    
    # Set model configuration
    layers = [
        {'type': 'dense', 'neurons': 64, 'activation': 'relu'},
        {'type': 'dense', 'neurons': 1, 'activation': 'sigmoid'}
    ]
    matrix.set_model_config(input_size=3, layers=layers)
    
    # Save to file
    matrix.save('example.nrm')
    
    # Load and validate
    loaded_matrix = NeuroMatrix.load('example.nrm')
    loaded_matrix.validate()
    
    # Convert to PyTorch dataset
    torch_dataset = loaded_matrix.to_torch_dataset()
    print("Dataset created successfully!") 