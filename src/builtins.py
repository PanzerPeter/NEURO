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
from types import SimpleNamespace

class NeuralNetwork(nn.Module):
    """Base class for all neural networks in NEURO."""
    
    def __init__(self, input_size: Union[int, Tuple[int, ...]], output_size: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.layers: nn.ModuleList = nn.ModuleList()
        self.loss_fn = None
        self.optimizer = None
        self._gradient_checkpointing_enabled = False
        self._mixed_precision_enabled = False
        self._branches = {}  # For tracking named branches
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        if self._gradient_checkpointing_enabled and self.training:
            return self._forward_with_checkpointing(x)
        
        for layer in self.layers:
            x = layer(x)
        return x
    
    def _forward_with_checkpointing(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with gradient checkpointing to save memory."""
        import torch.utils.checkpoint as checkpoint
        
        # Process layers in chunks to save memory
        chunks = self._chunk_layers(3)  # Process 3 layers at a time
        
        for chunk in chunks:
            # Create a dummy forward function for this chunk
            def custom_forward(*inputs):
                result = inputs[0]
                for layer in chunk:
                    result = layer(result)
                return result
            
            # Apply checkpointing to this chunk
            x = checkpoint.checkpoint(custom_forward, x)
        
        return x
    
    def _chunk_layers(self, chunk_size: int) -> List[List[nn.Module]]:
        """Split layers into chunks for gradient checkpointing."""
        chunks = []
        current_chunk = []
        
        for layer in self.layers:
            current_chunk.append(layer)
            if len(current_chunk) == chunk_size:
                chunks.append(current_chunk)
                current_chunk = []
        
        # Add the remaining layers
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def add_layer(self, layer: nn.Module) -> None:
        """Add a layer to the network."""
        self.layers.append(layer)
    
    def add_branch(self, name: str, branch: 'CustomLayer') -> None:
        """Add a named branch to the network."""
        self._branches[name] = branch
        self.add_layer(branch)
    
    def get_branch(self, name: str) -> Optional['CustomLayer']:
        """Get a named branch."""
        return self._branches.get(name)
    
    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing to reduce memory usage."""
        self._gradient_checkpointing_enabled = True
    
    def disable_gradient_checkpointing(self) -> None:
        """Disable gradient checkpointing."""
        self._gradient_checkpointing_enabled = False
    
    def is_gradient_checkpointing_enabled(self) -> bool:
        """Check if gradient checkpointing is enabled."""
        return self._gradient_checkpointing_enabled
    
    def enable_mixed_precision(self) -> None:
        """Enable mixed precision training."""
        self._mixed_precision_enabled = True
    
    def disable_mixed_precision(self) -> None:
        """Disable mixed precision training."""
        self._mixed_precision_enabled = False
    
    def is_mixed_precision_enabled(self) -> bool:
        """Check if mixed precision training is enabled."""
        return self._mixed_precision_enabled
    
    def get_memory_usage(self) -> int:
        """Get current GPU memory usage in bytes.
        
        Returns:
            Memory usage in bytes, or 0 if GPU is not available.
        """
        if not torch.cuda.is_available():
            return 0
        
        # Get current GPU memory usage
        return torch.cuda.memory_allocated()
    
    def get_branch_gradient_norm(self, branch_name: str) -> Optional[float]:
        """Get the gradient norm for a specific branch.
        
        Args:
            branch_name: Name of the branch
            
        Returns:
            Gradient norm or None if branch doesn't exist
        """
        branch = self.get_branch(branch_name)
        if branch is None:
            return None
        
        total_norm = 0.0
        for p in branch.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        
        return total_norm ** 0.5
    
    def get_branch_metrics(self, branch_name: str) -> Dict[str, float]:
        """Get performance metrics for a specific branch.
        
        This is a stub that would need to be properly implemented based on
        how branch-specific metrics are tracked during training.
        
        Args:
            branch_name: Name of the branch
            
        Returns:
            Dictionary of metrics or empty dict if branch doesn't exist
        """
        branch = self.get_branch(branch_name)
        if branch is None:
            return {}
        
        # This would need to be implemented based on the specific metrics
        # that are tracked during training.
        return {'loss': 0.0, 'accuracy': 0.0}
    
    def train_model(self, data: 'DataLoader', **kwargs) -> Dict:
        """Train the network."""
        if self.loss_fn is None:
            raise NeuroValueError("Loss function not set")
        if self.optimizer is None:
            raise NeuroValueError("Optimizer not set")
        
        epochs = kwargs.get('epochs', 10)
        batch_size = kwargs.get('batch_size', 32)
        validation_data = kwargs.get('validation_data', None)
        gradient_checkpoint = kwargs.get('gradient_checkpoint', False)
        
        # Set up gradient checkpointing if requested
        if gradient_checkpoint:
            self.enable_gradient_checkpointing()
        
        # Set model to training mode
        self.train(True)
        
        # Set up mixed precision training if enabled
        if self._mixed_precision_enabled:
            scaler = torch.cuda.amp.GradScaler()
        else:
            scaler = None
        
        # Prepare history tracking
        history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        # Run callbacks before training if any
        if 'before_training_callback' in kwargs:
            kwargs['before_training_callback'](self)
        
        for epoch in range(epochs):
            # Training phase
            if self._mixed_precision_enabled:
                train_loss, train_acc = self._train_epoch_mixed_precision(
                    data, batch_size, scaler, 
                    after_batch_callback=kwargs.get('after_batch_callback')
                )
            else:
                train_loss, train_acc = self._train_epoch(
                    data, batch_size,
                    after_batch_callback=kwargs.get('after_batch_callback')
                )
            
            history['loss'].append(train_loss)
            history['accuracy'].append(train_acc)
            
            # Validation phase
            if validation_data is not None:
                val_loss, val_acc = self.evaluate(validation_data)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_acc)
            
            # Create metrics object for callbacks
            metrics = SimpleNamespace(
                epoch=epoch,
                loss=train_loss,
                accuracy=train_acc,
                val_loss=val_loss if validation_data is not None else None,
                val_accuracy=val_acc if validation_data is not None else None
            )
            
            # Run after epoch callback if provided
            if 'after_epoch_callback' in kwargs:
                result = kwargs['after_epoch_callback'](self, metrics)
                # Check if the callback returned "StopTraining"
                if result == "StopTraining":
                    print(f"Training stopped early at epoch {epoch+1}")
                    break
        
        return history
    
    def _train_epoch(self, data: 'DataLoader', batch_size: int, after_batch_callback=None) -> Tuple[float, float]:
        """Train a single epoch."""
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(data.batch(batch_size)):
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
            
            # Call after_batch callback if provided
            if after_batch_callback:
                batch_metrics = SimpleNamespace(
                    batch=batch_idx,
                    loss=loss.item(),
                    batch_size=targets.size(0)
                )
                after_batch_callback(self, batch_metrics)
            
            # Free memory
            del outputs, loss
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return total_loss / len(data), correct / total
    
    def _train_epoch_mixed_precision(self, data: 'DataLoader', batch_size: int, scaler, after_batch_callback=None) -> Tuple[float, float]:
        """Train a single epoch with mixed precision."""
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(data.batch(batch_size)):
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass with autocasting
            with torch.cuda.amp.autocast():
                outputs = self(inputs)
                loss = self.loss_fn(outputs, targets)
            
            # Backward pass and optimize with scaled gradients
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Call after_batch callback if provided
            if after_batch_callback:
                batch_metrics = SimpleNamespace(
                    batch=batch_idx,
                    loss=loss.item(),
                    batch_size=targets.size(0)
                )
                after_batch_callback(self, batch_metrics)
            
            # Free memory
            del outputs, loss
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return total_loss / len(data), correct / total
    
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

def BatchNorm(num_features: Optional[int] = None, momentum: float = 0.1, eps: float = 1e-5, dim: int = 1) -> nn.Module:
    """Create a batch normalization layer.
    
    Args:
        num_features: Number of features. If None, will be inferred at runtime
        momentum: Momentum factor for running stats
        eps: Small constant for numerical stability
        dim: Dimensionality (1 for BatchNorm1d, 2 for BatchNorm2d)
    
    Returns:
        PyTorch BatchNorm module
    """
    if dim == 1:
        if num_features is None:
            raise NeuroValueError("'num_features' must be specified for BatchNorm1d")
        return nn.BatchNorm1d(num_features=num_features, momentum=momentum, eps=eps)
    elif dim == 2:
        if num_features is None:
            raise NeuroValueError("'num_features' must be specified for BatchNorm2d")
        return nn.BatchNorm2d(num_features=num_features, momentum=momentum, eps=eps)
    else:
        raise NeuroValueError(f"Unsupported BatchNorm dimension: {dim}")

def Conv2D(filters: int, kernel_size: Union[int, Tuple[int, int]], 
           stride: Union[int, Tuple[int, int]] = 1, 
           padding: Union[int, str] = 0,
           dilation: Union[int, Tuple[int, int]] = 1,
           groups: int = 1,
           bias: bool = True,
           activation: Optional[str] = None,
           input_channels: Optional[int] = None) -> nn.Module:
    """Create a 2D convolutional layer.
    
    Args:
        filters: Number of output channels
        kernel_size: Size of the convolving kernel
        stride: Stride of the convolution
        padding: Padding added to all sides of the input
        dilation: Spacing between kernel elements
        groups: Number of blocked connections from input to output
        bias: Whether to use a bias term
        activation: Activation function to use
        input_channels: Number of input channels. If None, will be inferred at runtime
    
    Returns:
        PyTorch Conv2d module with optional activation
    """
    if filters <= 0:
        raise NeuroValueError("'filters' must be a positive integer")
    
    if input_channels is None:
        raise NeuroValueError("'input_channels' must be specified for Conv2D layer")
    
    # Create layer
    layers = []
    layers.append(nn.Conv2d(
        in_channels=input_channels,
        out_channels=filters,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias
    ))
    
    # Add activation
    if activation is not None:
        if activation == "relu":
            layers.append(nn.ReLU())
        elif activation == "sigmoid":
            layers.append(nn.Sigmoid())
        elif activation == "tanh":
            layers.append(nn.Tanh())
        elif activation == "leaky_relu":
            layers.append(nn.LeakyReLU())
        else:
            raise NeuroValueError(f"Unknown activation function: {activation}")
    
    return nn.Sequential(*layers)

def MaxPool(pool_size: Union[int, Tuple[int, int]], 
            stride: Optional[Union[int, Tuple[int, int]]] = None,
            padding: Union[int, Tuple[int, int]] = 0,
            dilation: Union[int, Tuple[int, int]] = 1) -> nn.Module:
    """Create a 2D max pooling layer.
    
    Args:
        pool_size: Size of the window to take a max over
        stride: Stride of the window. Default value is pool_size
        padding: Implicit zero padding on both sides
        dilation: Controls the stride of elements in the window
    
    Returns:
        PyTorch MaxPool2d module
    """
    if stride is None:
        stride = pool_size
        
    return nn.MaxPool2d(
        kernel_size=pool_size,
        stride=stride,
        padding=padding,
        dilation=dilation
    )

def GlobalAveragePooling2D(keepdims: bool = False) -> nn.Module:
    """Create a global average pooling layer for 2D inputs.
    
    Args:
        keepdims: Whether to keep the spatial dimensions as 1
    
    Returns:
        A module that performs global average pooling
    """
    class GlobalAvgPool2d(nn.Module):
        def __init__(self, keepdims: bool = False):
            super().__init__()
            self.keepdims = keepdims
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Input: B x C x H x W
            # Output: B x C (or B x C x 1 x 1 if keepdims=True)
            pooled = torch.mean(x, dim=(2, 3), keepdim=self.keepdims)
            return pooled
    
    return GlobalAvgPool2d(keepdims=keepdims)

def LSTM(units: int, 
         input_size: Optional[int] = None,
         return_sequences: bool = False,
         bidirectional: bool = False,
         num_layers: int = 1,
         dropout: float = 0.0) -> nn.Module:
    """Create an LSTM layer.
    
    Args:
        units: Number of features in the hidden state
        input_size: Size of each input feature. If None, will be inferred
        return_sequences: If True, return the full sequence, otherwise only the last output
        bidirectional: If True, becomes a bidirectional LSTM
        num_layers: Number of recurrent layers
        dropout: Dropout between LSTM layers (not applied to last layer)
    
    Returns:
        A module wrapping the PyTorch LSTM
    """
    if units <= 0:
        raise NeuroValueError("'units' must be a positive integer")
    
    if input_size is None:
        raise NeuroValueError("'input_size' must be specified for LSTM layer")
    
    class LSTMWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=units,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0
            )
            self.return_sequences = return_sequences
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            output, (h_n, c_n) = self.lstm(x)
            if self.return_sequences:
                return output
            else:
                # Return last time step output
                if self.lstm.bidirectional:
                    # For bidirectional, concatenate last outputs from both directions
                    forward = output[:, -1, :units]
                    backward = output[:, 0, units:]
                    return torch.cat([forward, backward], dim=1)
                else:
                    return output[:, -1, :]
    
    return LSTMWrapper()

def GRU(units: int, 
        input_size: Optional[int] = None,
        return_sequences: bool = False,
        bidirectional: bool = False,
        num_layers: int = 1,
        dropout: float = 0.0) -> nn.Module:
    """Create a GRU layer.
    
    Args:
        units: Number of features in the hidden state
        input_size: Size of each input feature. If None, will be inferred
        return_sequences: If True, return the full sequence, otherwise only the last output
        bidirectional: If True, becomes a bidirectional GRU
        num_layers: Number of recurrent layers
        dropout: Dropout between GRU layers (not applied to last layer)
    
    Returns:
        A module wrapping the PyTorch GRU
    """
    if units <= 0:
        raise NeuroValueError("'units' must be a positive integer")
    
    if input_size is None:
        raise NeuroValueError("'input_size' must be specified for GRU layer")
    
    class GRUWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=units,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0
            )
            self.return_sequences = return_sequences
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            output, h_n = self.gru(x)
            if self.return_sequences:
                return output
            else:
                # Return last time step output
                if self.gru.bidirectional:
                    # For bidirectional, concatenate last outputs from both directions
                    forward = output[:, -1, :units]
                    backward = output[:, 0, units:]
                    return torch.cat([forward, backward], dim=1)
                else:
                    return output[:, -1, :]
    
    return GRUWrapper()

def Reshape(shape: Tuple[int, ...]) -> nn.Module:
    """Create a reshape layer.
    
    Args:
        shape: Target shape (excluding batch dimension)
    
    Returns:
        Module that reshapes inputs to the target shape
    """
    class ReshapeLayer(nn.Module):
        def __init__(self, shape: Tuple[int, ...]):
            super().__init__()
            self.shape = shape
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.view(x.size(0), *self.shape)
    
    return ReshapeLayer(shape)

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

def Branch(name: str) -> CustomLayer:
    """Create a new branch with a specific name.
    
    Args:
        name: Name of the branch
    
    Returns:
        A custom layer that represents the branch
    """
    branch = CustomLayer()
    branch.name = name
    return branch

def Concatenate(branches: List[str] = None, dim: int = 1) -> nn.Module:
    """Concatenate multiple branches along a specified dimension.
    
    Args:
        branches: List of branch names to concatenate
        dim: Dimension along which to concatenate
    
    Returns:
        A module that concatenates inputs along the specified dimension
    """
    class ConcatenateLayer(nn.Module):
        def __init__(self, branches, dim=1):
            super().__init__()
            self.branches = branches
            self.dim = dim
            self.branch_outputs = {}
        
        def register_branch_output(self, name, output):
            """Register the output of a branch for later concatenation."""
            self.branch_outputs[name] = output
        
        def forward(self, x):
            """Concatenate registered branch outputs with input."""
            if not self.branches:
                return x
            
            tensors_to_concat = []
            for branch_name in self.branches:
                if branch_name in self.branch_outputs:
                    tensors_to_concat.append(self.branch_outputs[branch_name])
                else:
                    raise NeuroValueError(f"Branch '{branch_name}' not found or has no output")
            
            if not tensors_to_concat:
                return x
            
            # Add the input tensor to concat
            tensors_to_concat.append(x)
            
            # Concatenate tensors
            return torch.cat(tensors_to_concat, dim=self.dim)
    
    return ConcatenateLayer(branches, dim)

def Backbone(model=None, trainable: bool = True):
    """Create a backbone using a pre-trained model.
    
    Args:
        model: Pre-trained model to use as backbone
        trainable: Whether the backbone is trainable
    
    Returns:
        A module that wraps the backbone model
    """
    class BackboneWrapper(nn.Module):
        def __init__(self, model, trainable):
            super().__init__()
            self.model = model
            self.trainable = trainable
            
            # Make model non-trainable if needed
            if not trainable:
                for param in self.model.parameters():
                    param.requires_grad = False
        
        def forward(self, x):
            return self.model(x)
        
        @property
        def trainable(self):
            """Get trainable state."""
            return self._trainable
        
        @trainable.setter
        def trainable(self, value):
            """Set trainable state."""
            self._trainable = value
            for param in self.model.parameters():
                param.requires_grad = value
    
    return BackboneWrapper(model, trainable)

class MultiOutputModel(nn.Module):
    """A model that can have multiple outputs."""
    
    def __init__(self, outputs: List[nn.Module]):
        """Initialize with list of output modules."""
        super().__init__()
        self.outputs = nn.ModuleList(outputs)
    
    def forward(self, x) -> List[torch.Tensor]:
        """Forward pass returning multiple outputs."""
        return [output(x) for output in self.outputs]

def create_multi_output_model(base_model: NeuralNetwork, output_modules: Dict[str, nn.Module]) -> nn.Module:
    """Create a multi-output model from a base model and output modules.
    
    Args:
        base_model: Base model to extract features
        output_modules: Dict of named output modules
    
    Returns:
        Multi-output model
    """
    class NamedMultiOutputModel(nn.Module):
        def __init__(self, base_model, output_modules):
            super().__init__()
            self.base_model = base_model
            self.output_modules = nn.ModuleDict(output_modules)
        
        def forward(self, x):
            features = self.base_model(x)
            return {name: module(features) for name, module in self.output_modules.items()}
    
    return NamedMultiOutputModel(base_model, output_modules) 