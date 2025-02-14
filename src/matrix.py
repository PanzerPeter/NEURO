import yaml
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
import json
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Any, Tuple, Optional
from .errors import ValidationError, get_context
import time
import math

class NeuroMatrix(Dataset):
    """
    Custom dataset class for NEURO data format.
    Supports both training data and model weights.
    """
    
    def __init__(self, data: List[Dict[str, Any]], meta: Optional[Dict[str, Any]] = None):
        """Initialize NeuroMatrix with data and metadata."""
        if not isinstance(data, list):
            raise ValidationError(
                "Data must be a list",
                context=get_context(),
                suggestions=["Provide data as a list of dictionaries"]
            )
        self.data = data
        self.meta = meta or {}
        self.model_config = {
            'input_size': None,
            'output_size': None,
            'batch_size': 32
        }
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
        """Creates a NeuroMatrix from PyTorch tensors."""
        if not isinstance(inputs, torch.Tensor) or not isinstance(targets, torch.Tensor):
            raise ValidationError(
                "Invalid tensor types",
                context=get_context(),
                suggestions=["Both inputs and targets must be PyTorch tensors"]
            )
            
        if len(inputs) != len(targets):
            raise ValidationError(
                "Mismatched tensor lengths",
                context=get_context(),
                suggestions=["Input and target tensors must have the same length"]
            )
            
        # Check for inf/nan values
        if torch.any(torch.isinf(inputs)) or torch.any(torch.isnan(inputs)):
            raise ValidationError(
                "Input tensor contains inf or nan values",
                context=get_context(),
                suggestions=["Ensure input values are finite numbers"]
            )
            
        if torch.any(torch.isinf(targets)) or torch.any(torch.isnan(targets)):
            raise ValidationError(
                "Target tensor contains inf or nan values",
                context=get_context(),
                suggestions=["Ensure target values are finite numbers"]
            )
            
        # Convert tensors to list of dictionaries
        data = []
        for x, y in zip(inputs, targets):
            data.append({
                'input': x.tolist(),
                'output': y.tolist() if y.dim() > 0 else [float(y)]
            })
            
        meta = {
            'created': 'from_tensors',
            'input_shape': list(inputs.shape[1:]),
            'output_shape': list(targets.shape[1:]) if targets.dim() > 1 else [1],
            'original_target_shape': list(targets.shape),
            'original_target_dim': targets.dim()
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
        """Converts data to PyTorch tensors with proper shape handling."""
        if not self.data:
            raise ValidationError(
                "Cannot create tensors from empty data",
                context=get_context(),
                suggestions=["Add data points before creating tensors"]
            )
        
        # First validate dimensions
        input_dim = len(self.data[0]['input'])
        output_dim = len(self.data[0]['output']) if isinstance(self.data[0]['output'], list) else 1
        
        inputs = []
        targets = []
        
        for i, item in enumerate(self.data):
            if len(item['input']) != input_dim:
                raise ValidationError(
                    f"Mismatched input dimensions at index {i}",
                    context=get_context(),
                    suggestions=["Ensure all inputs have the same dimension"]
                )
            
            curr_output = item['output']
            if not isinstance(curr_output, list):
                curr_output = [float(curr_output)]
            
            if len(curr_output) != output_dim:
                raise ValidationError(
                    f"Mismatched output dimensions at index {i}",
                    context=get_context(),
                    suggestions=["Ensure all outputs have the same dimension"]
                )
            
            inputs.append(torch.tensor(item['input'], dtype=torch.float32))
            targets.append(torch.tensor(curr_output, dtype=torch.float32))
        
        self.inputs = torch.stack(inputs)
        self.targets = torch.stack(targets)
        
        # Store original shapes in meta
        self.meta['input_shape'] = list(self.inputs.shape[1:])
        self.meta['output_shape'] = list(self.targets.shape[1:])
        self.meta['original_target_dim'] = len(self.data[0]['output']) if isinstance(self.data[0]['output'], list) else 0
        
        # Update model config if not set
        if not self.model_config['input_size']:
            self.model_config['input_size'] = input_dim
        if not self.model_config['output_size']:
            self.model_config['output_size'] = output_dim
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a data point by index with proper tensor handling."""
        if isinstance(idx, slice):
            return NeuroMatrix(self.data[idx], self.meta)
        
        inputs = self.inputs[idx]
        targets = self.targets[idx]
        
        return {
            'input': inputs.tolist(),
            'output': targets.tolist() if targets.dim() > 0 else [float(targets)]
        }
    
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
            
        if not isinstance(data, dict) or 'data' not in data:
            raise ValidationError(
                "Invalid file format",
                context=get_context(),
                suggestions=["File must contain a dictionary with 'data' key"]
            )
            
        matrix = NeuroMatrix(data['data'], data.get('meta'))
        
        # Load model config if present
        if 'model_config' in data:
            matrix.model_config = data['model_config']
        
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
        """Validate the matrix format with proper error handling."""
        if not self.meta:
            self.meta = {'created_at': time.time()}
        
        if not self.data:
            raise ValidationError(
                "No data points found",
                context=get_context(),
                suggestions=["Add data points before validation"]
            )
        
        for i, point in enumerate(self.data):
            if 'input' not in point or 'output' not in point:
                raise ValidationError(
                    f"Data point {i} missing input or output",
                    context=get_context(),
                    suggestions=["Ensure all data points have 'input' and 'output' keys"]
                )
            
            if point['input'] is None or not point['input']:
                raise ValidationError(
                    f"Empty input in data point {i}",
                    context=get_context(),
                    suggestions=["Ensure input values are not empty"]
                )
            
            if point['output'] is None or not point['output']:
                raise ValidationError(
                    f"Empty output in data point {i}",
                    context=get_context(),
                    suggestions=["Ensure output values are not empty"]
                )
            
            # Check for invalid values (inf, nan)
            try:
                input_tensor = torch.tensor(point['input'], dtype=torch.float32)
                if torch.any(torch.isinf(input_tensor)) or torch.any(torch.isnan(input_tensor)):
                    raise ValidationError(
                        f"Invalid input values (inf/nan) in data point {i}",
                        context=get_context(),
                        suggestions=["Ensure all input values are finite numbers"]
                    )
            except (ValueError, TypeError):
                raise ValidationError(
                    f"Invalid input values in data point {i}",
                    context=get_context(),
                    suggestions=["Ensure all input values are valid numbers"]
                )
            
            try:
                if isinstance(point['output'], list):
                    output_tensor = torch.tensor(point['output'], dtype=torch.float32)
                    if torch.any(torch.isinf(output_tensor)) or torch.any(torch.isnan(output_tensor)):
                        raise ValidationError(
                            f"Invalid output values (inf/nan) in data point {i}",
                            context=get_context(),
                            suggestions=["Ensure all output values are finite numbers"]
                        )
                else:
                    output_val = float(point['output'])
                    if math.isinf(output_val) or math.isnan(output_val):
                        raise ValidationError(
                            f"Invalid output value (inf/nan) in data point {i}",
                            context=get_context(),
                            suggestions=["Ensure output value is a finite number"]
                        )
            except (ValueError, TypeError):
                raise ValidationError(
                    f"Invalid output values in data point {i}",
                    context=get_context(),
                    suggestions=["Ensure all output values are valid numbers"]
                )
        
        return True

    def to_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert the matrix data back to PyTorch tensors."""
        if not hasattr(self, 'inputs') or not hasattr(self, 'targets'):
            self._prepare_tensors()
        
        # Get original target shape and dimension from meta
        original_shape = self.meta.get('original_target_shape')
        original_dim = self.meta.get('original_target_dim', 1)
        
        # Handle inputs
        inputs = self.inputs
        
        # Handle targets based on original dimensionality
        if original_shape and len(original_shape) > 1:
            # Multi-dimensional case
            targets = self.targets.reshape(original_shape)
        elif original_dim == 0:
            # Scalar case (0-dimensional)
            targets = self.targets.squeeze()
        elif original_dim == 1:
            # Vector case (1-dimensional)
            targets = self.targets.squeeze(-1)
        else:
            # Default case
            targets = self.targets
            
        return inputs, targets

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert the matrix data to NumPy arrays."""
        inputs, targets = self.to_tensors()
        return inputs.numpy(), targets.numpy()

    def normalize(self):
        """Normalize the input data to have zero mean and unit variance."""
        if not hasattr(self, 'inputs'):
            self._prepare_tensors()
            
        # Convert to double precision for better numerical stability
        inputs = self.inputs.double()
        
        # Calculate mean and center the data
        mean = torch.mean(inputs, dim=0)
        centered = inputs - mean
        
        # Calculate variance and std
        variance = torch.mean(centered ** 2, dim=0)
        std = torch.sqrt(variance)
        
        # Handle zero std case
        std = torch.where(std == 0, torch.ones_like(std), std)
        
        # Normalize
        self.inputs = (centered / std).float()  # Convert back to float32
        
        # Store normalization parameters
        self.meta['normalization'] = {
            'mean': mean.float().tolist(),
            'std': std.float().tolist()
        }
        
        # Update data points with normalized inputs
        for i, point in enumerate(self.data):
            point['input'] = self.inputs[i].tolist()
        
        return self

    def create_batches(self, batch_size: int) -> DataLoader:
        """Create a DataLoader with proper tensor handling."""
        dataset = TensorDataset(self.inputs, self.targets)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def concatenate(self, other: 'NeuroMatrix') -> 'NeuroMatrix':
        """
        Concatenate this matrix with another NeuroMatrix.
        
        Args:
            other: Another NeuroMatrix instance
            
        Returns:
            A new NeuroMatrix containing data from both matrices
        """
        if not isinstance(other, NeuroMatrix):
            raise ValidationError(
                "Invalid concatenation type",
                context=get_context(),
                suggestions=["Can only concatenate with another NeuroMatrix"],
                details={"provided_type": type(other).__name__}
            )
            
        combined_data = self.data + other.data
        result = NeuroMatrix(combined_data)
        
        # Merge configurations
        result.model_config = self.model_config.copy()
        result.meta = {
            'created_at': time.time(),
            'combined_from': [self.meta.get('created_at'), other.meta.get('created_at')]
        }
        
        return result

    def get_statistics(self) -> Dict[str, Any]:
        """
        Calculate basic statistics of the dataset.
        
        Returns:
            Dictionary containing statistics
        """
        inputs, targets = self.to_tensors()
        
        return {
            'num_samples': len(self),
            'input_shape': list(inputs.size()[1:]),
            'input_mean': inputs.mean(dim=0).tolist(),
            'input_std': inputs.std(dim=0).tolist(),
            'output_mean': targets.mean(dim=0).tolist(),
            'output_std': targets.std(dim=0).tolist()
        }

    def split(self, train_ratio=0.8):
        """Split data into training and validation sets."""
        split_idx = int(len(self.data) * train_ratio)
        
        train_data = self.data[:split_idx]
        val_data = self.data[split_idx:]
        
        train_matrix = NeuroMatrix(train_data)
        val_matrix = NeuroMatrix(val_data)
        
        # Copy configuration
        train_matrix.model_config = self.model_config.copy()
        val_matrix.model_config = self.model_config.copy()
        
        return train_matrix, val_matrix

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