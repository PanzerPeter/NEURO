import torch
import torch.nn as nn
from typing import Dict, Any

# Import Neuro types
from .neuro_types import (
    NeuroType, LossType, TensorType, NEURO_FLOAT, NEURO_INT, AnyType, 
    TensorFloat32, TensorInt32, NEURO_ANY
)

class BaseLoss(nn.Module):
    """Base class for Neuro losses to ensure common methods."""
    def __init__(self):
        super().__init__()

    def get_config(self) -> Dict[str, Any]:
        """Return the loss configuration."""
        # Base implementation returns empty dict, subclasses should override
        return {}

    def get_type_signature(self) -> LossType:
        """Return the type signature for the loss function."""
        # Default: expects Any predictions and targets
        # Subclasses should override with specific types
        return LossType(pred_type=NEURO_ANY, target_type=NEURO_ANY)

    def __repr__(self) -> str:
        config = self.get_config()
        config_str = ", ".join(f"{k}={v!r}" for k, v in config.items())
        return f"{self.__class__.__name__}({config_str})"

class BCELoss(BaseLoss):
    """
    Binary Cross-Entropy Loss.
    """
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCELoss()

    def get_config(self) -> Dict[str, Any]:
        # BCE has no configurable parameters in this simple version
        return {}

    def get_type_signature(self) -> LossType:
        # BCE expects float predictions (probabilities) and float/int targets
        # Shapes typically (batch,) or (batch, 1)
        # Allow flexible batch dimension with None
        pred_type = TensorFloat32(shape=(None,)) # Or (None, 1)? Let's use (None,)
        target_type = TensorType(shape=(None,), dtype=NEURO_FLOAT) # Accepts float targets
        # Could also allow Int targets: TensorInt32(shape=(None,))
        # Let's make target type slightly more general for now
        # target_type = TensorType(shape=(None,), dtype=AnyType()) # Or check compatible numeric? 
        
        return LossType(pred_type=pred_type, target_type=target_type)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculates the BCE loss.

        Args:
            predictions: The model's predictions (logits or probabilities).
                         Should have values between 0 and 1.
            targets: The ground truth labels (0 or 1).

        Returns:
            The calculated loss value.
        """
        # Ensure predictions are probabilities (apply sigmoid if they are logits)
        # For simplicity, assuming predictions are already probabilities here.
        # In a real scenario, ensure input is appropriate or preprocess.
        if not ((predictions >= 0) & (predictions <= 1)).all():
             # Applying sigmoid if the values are not in [0, 1] range, assuming they are logits
             predictions = torch.sigmoid(predictions)

        # Ensure target dtype is float for BCELoss
        targets = targets.float()

        # Ensure predictions and targets have the same shape
        if predictions.shape != targets.shape:
            # Attempt to broadcast or reshape if necessary and meaningful
            # Example: if targets are [N] and predictions are [N, 1], squeeze predictions
            if predictions.ndim == targets.ndim + 1 and predictions.shape[-1] == 1:
                predictions = predictions.squeeze(-1)
            elif targets.ndim == predictions.ndim + 1 and targets.shape[-1] == 1:
                 targets = targets.squeeze(-1)
            # Add more sophisticated shape handling if needed
            if predictions.shape != targets.shape:
                 raise ValueError(f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}")


        return self.loss_fn(predictions, targets)

# Example usage (optional, for testing within the file)
if __name__ == '__main__':
    loss_func = BCELoss()
    # Example with probabilities
    preds_prob = torch.tensor([0.1, 0.9, 0.8, 0.3])
    targets_bin = torch.tensor([0, 1, 1, 0])
    loss_val_prob = loss_func(preds_prob, targets_bin)
    print(f"Loss (probabilities): {loss_val_prob.item()}")

    # Example with logits (requires sigmoid)
    preds_logits = torch.tensor([-2.2, 2.2, 1.4, -0.85]) # approx sigmoid -> [0.1, 0.9, 0.8, 0.3]
    loss_val_logits = loss_func(preds_logits, targets_bin)
    print(f"Loss (logits): {loss_val_logits.item()}")

    # Example with shape mismatch (needs handling)
    preds_prob_extra_dim = torch.tensor([[0.1], [0.9], [0.8], [0.3]])
    loss_val_shape = loss_func(preds_prob_extra_dim, targets_bin)
    print(f"Loss (shape handling): {loss_val_shape.item()}")

    try:
        preds_wrong_shape = torch.tensor([[0.1, 0.2], [0.9, 0.8], [0.8, 0.7], [0.3, 0.4]])
        loss_func(preds_wrong_shape, targets_bin)
    except ValueError as e:
        print(f"Caught expected error for wrong shape: {e}") 