import abc
import torch

class Layer(abc.ABC):
    """Abstract base class for all neural network layers."""

    def __init__(self):
        # Placeholder for potential future shared initialization
        pass

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the layer."""
        raise NotImplementedError

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