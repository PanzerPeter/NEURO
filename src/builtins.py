"""
NEURO Built-in Functions and Classes

This module implements the built-in functionality of the NEURO language.
It provides core classes and functions available to all NEURO programs.

Key responsibilities:
- Implement NeuralNetwork base class
- Define layer classes (Dense, Conv2D, etc.)
- Implement activation functions
- Provide data handling functions
- Define optimizer classes
- Implement loss functions
- Support for model operations

This module provides the core AI functionality of NEURO.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Union, Tuple
from .errors import NeuroLayerError, NeuroShapeError, NeuroValueError

class NeuralNetwork(nn.Module):
    """Base class for all neural networks in NEURO."""
    
    def __init__(self, input_size: Union[int, Tuple[int, ...]], output_size: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.layers: nn.ModuleList = nn.ModuleList()
        self.loss_fn = None
        self.optimizer = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        for layer in self.layers:
            x = layer(x)
        return x
    
    def add_layer(self, layer: nn.Module) -> None:
        """Add a layer to the network."""
        self.layers.append(layer)
    
    def train_model(self, data: 'DataLoader', **kwargs) -> Dict:
        """Train the network."""
        if self.loss_fn is None:
            raise NeuroValueError("Loss function not set")
        if self.optimizer is None:
            raise NeuroValueError("Optimizer not set")
        
        epochs = kwargs.get('epochs', 10)
        batch_size = kwargs.get('batch_size', 32)
        validation_data = kwargs.get('validation_data', None)
        
        # Set model to training mode
        self.train(True)
        
        history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        for epoch in range(epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(data, batch_size)
            history['loss'].append(train_loss)
            history['accuracy'].append(train_acc)
            
            # Validation phase
            if validation_data is not None:
                val_loss, val_acc = self.evaluate(validation_data)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_acc)
        
        return history
    
    def evaluate(self, data: 'DataLoader') -> Tuple[float, float]:
        """Evaluate the network."""
        # Set model to evaluation mode
        self.train(False)
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in data:
                outputs = self(inputs)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return total_loss / len(data), correct / total
    
    def save(self, path: str) -> None:
        """Save the model to a file."""
        torch.save({
            'model_state': self.state_dict(),
            'input_size': self.input_size,
            'output_size': self.output_size
        }, path)
    
    def load(self, path: str) -> None:
        """Load the model from a file."""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state'])
    
    def train_mode(self) -> None:
        """Set the model to training mode."""
        self.train(True)
    
    def eval_mode(self) -> None:
        """Set the model to evaluation mode."""
        self.eval()
    
    def _train_epoch(self, data: 'DataLoader', batch_size: int) -> Tuple[float, float]:
        """Train a single epoch."""
        total_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in data.batch(batch_size):
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self(inputs)
            loss = self.loss_fn(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        return total_loss / len(data), correct / total

class CustomLayer(nn.Module):
    """Custom layer implementation for NEURO.
    
    This class represents a custom layer defined by the user with the 'layer' keyword.
    It can contain multiple sub-layers and acts as a Sequential container.
    """
    
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all sub-layers."""
        for layer in self.layers:
            x = layer(x)
        return x
    
    def add_layer(self, layer: nn.Module) -> None:
        """Add a sub-layer to this custom layer."""
        self.layers.append(layer)

class DataLoader:
    """Data loading and preprocessing utility."""
    
    def __init__(self, data: torch.Tensor, targets: torch.Tensor):
        if len(data) != len(targets):
            raise NeuroShapeError(
                expected_shape=(len(targets),),
                got_shape=(len(data),)
            )
        self.data = data
        self.targets = targets
        self.current_idx = 0
    
    def __len__(self) -> int:
        return len(self.data)
    
    def batch(self, batch_size: int):
        """Create batches of data."""
        for i in range(0, len(self), batch_size):
            yield (
                self.data[i:i + batch_size],
                self.targets[i:i + batch_size]
            )
    
    def shuffle(self) -> None:
        """Shuffle the data."""
        indices = torch.randperm(len(self))
        self.data = self.data[indices]
        self.targets = self.targets[indices]
    
    def split(self, ratio: float) -> Tuple['DataLoader', 'DataLoader']:
        """Split the data into two parts."""
        if not 0 < ratio < 1:
            raise NeuroValueError("Split ratio must be between 0 and 1")
        
        split_idx = int(len(self) * ratio)
        return (
            DataLoader(self.data[:split_idx], self.targets[:split_idx]),
            DataLoader(self.data[split_idx:], self.targets[split_idx:])
        )

# Layer factories
def Dense(units: int, activation: Optional[str] = None, input_size: Optional[int] = None) -> nn.Module:
    """Create a dense (fully connected) layer."""
    # Validate units must be positive
    if units <= 0:
        raise NeuroValueError("'units' must be a positive integer")
        
    layers = []
    # If input_size is provided, use it, otherwise assume units for both dimensions
    # This is a simplification - in a real model, we'd use proper shape inference
    in_features = input_size if input_size is not None else units
    layers.append(nn.Linear(in_features, units))
    
    if activation is not None:
        if activation == "relu":
            layers.append(nn.ReLU())
        elif activation == "sigmoid":
            layers.append(nn.Sigmoid())
        elif activation == "tanh":
            layers.append(nn.Tanh())
        elif activation == "softmax":
            layers.append(nn.Softmax(dim=1))
        else:
            raise NeuroValueError(f"Unknown activation function: {activation}")
    
    return nn.Sequential(*layers)

def Dropout(rate: float) -> nn.Module:
    """Create a dropout layer."""
    if not 0 <= rate < 1:
        raise NeuroValueError("Dropout rate must be between 0 and 1")
    return nn.Dropout(p=rate)

def BatchNorm() -> nn.Module:
    """Create a batch normalization layer."""
    return nn.BatchNorm1d()

# Loss functions
class Loss:
    """Create a loss function."""
    def __init__(self, type: str):
        if type == "mse":
            self.loss = nn.MSELoss()
        elif type == "cross_entropy":
            self.loss = nn.CrossEntropyLoss()
        elif type == "binary_crossentropy":
            self.loss = nn.BCELoss()
        else:
            raise NeuroValueError(f"Unknown loss function: {type}")
        self.type = type
        
    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)
        
    def get_function(self):
        """Get the PyTorch loss function."""
        return self.loss

# Optimizers
def Optimizer(type: str, learning_rate: float, **kwargs) -> optim.Optimizer:
    """Create an optimizer."""
    # Üres lista a paramétereknél, ha nincs megadva model vagy paraméterek
    params = kwargs.pop('params', [])
    
    # PyTorch optimizerek 'lr' paramétert használnak 'learning_rate' helyett
    if type == "adam":
        return optim.Adam(params, lr=learning_rate, **kwargs)
    elif type == "sgd":
        return optim.SGD(params, lr=learning_rate, **kwargs)
    elif type == "rmsprop":
        return optim.RMSprop(params, lr=learning_rate, **kwargs)
    else:
        raise NeuroValueError(f"Unknown optimizer: {type}")

# Data loading functions
def load_matrix(path: str) -> DataLoader:
    """Load data from a .nrm file."""
    from .matrix import NeuroMatrix
    matrix = NeuroMatrix.load(path)
    return matrix.to_dataloader()

def load_dataset(name: str) -> DataLoader:
    """Load a built-in dataset."""
    if name == "mnist":
        from torchvision.datasets import MNIST
        from torchvision.transforms import ToTensor
        
        dataset = MNIST(root='./data', train=True, transform=ToTensor(), download=True)
        data = torch.stack([sample[0] for sample in dataset])
        targets = torch.tensor([sample[1] for sample in dataset])
        return DataLoader(data, targets)
    else:
        raise NeuroValueError(f"Unknown dataset: {name}") 