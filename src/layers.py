import abc
import torch
import torch.nn as nn

class Layer(nn.Module):
    """Base class for all NEURO layers."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Default forward just passes input through
        # Subclasses should override this
        return x

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Allows the layer instance to be called like a function."""
        return self.forward(x)

    def get_parameters(self) -> list[torch.Tensor]:
        """Returns a list of learnable parameters for the layer.
        
        Base implementation returns an empty list. Subclasses with parameters 
        should override this method.
        """
        return []

    def __repr__(self) -> str:
        """Provides a string representation of the layer."""
        return f"{self.__class__.__name__}()"

# --- Concrete Layer Implementations ---

class DenseLayer(Layer):
    """Represents a fully connected (dense) layer."""
    def __init__(self, units, activation=None, input_shape=None):
        super().__init__()
        # We need input_shape to define the layer, 
        # but it's often determined dynamically in PyTorch
        self.units = units
        self.activation_name = activation
        # Actual nn.Linear layer will be created in the NeuralNetwork model

class Conv2dLayer(Layer):
    """Represents a 2D convolutional layer."""
    def __init__(self, out_channels, kernel_size, in_channels=None, stride=1, padding=0, activation=None):
        super().__init__()
        self.in_channels = in_channels # May be inferred later
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation_name = activation
        # Actual nn.Conv2d layer will be created in the NeuralNetwork model 