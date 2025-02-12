import yaml
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
import json
from torch.nn.utils.rnn import pad_sequence

class NeuroMatrix:
    def __init__(self, data=None):
        self.data = []
        self.meta = {}
        self.model_config = {}
        if data is not None:
            self.data = data

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

    def to_torch_dataset(self):
        """Convert the matrix data to a PyTorch dataset."""
        if not self.data:
            raise ValueError("Empty matrix cannot be converted to dataset")

        # Add dtype auto-detection
        sample_input = self.data[0]['input']
        dtype = torch.float if isinstance(sample_input[0], float) else torch.long
        
        # Convert inputs and targets to tensors
        inputs = [torch.tensor(d['input'], dtype=dtype) for d in self.data]
        targets = [torch.tensor(d['output'], dtype=dtype).squeeze() for d in self.data]  # Squeeze to remove extra dimensions
        
        # Stack tensors into batches
        inputs = torch.stack(inputs)
        targets = torch.stack(targets)
        
        # Create dataset
        dataset = TensorDataset(inputs, targets)
        return DataLoader(dataset, batch_size=32, shuffle=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

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