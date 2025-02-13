"""
NEURO Debugger Module
Provides debugging tools for NEURO models.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
import time
import cProfile
import pstats
from io import StringIO
import numpy as np
from .errors import NeuroError, get_context

class ModelDebugger:
    def __init__(self, model: nn.Module):
        self.model = model
        self.watches: Dict[str, List[torch.Tensor]] = {}
        self.break_conditions: List[str] = []
        self.profiling = False
        self.profiler = None
        self.gradient_history = []
        self.activation_history = []
        self._hooks = []
        self.loss_hook = None

    def watch(self, targets: List[str]):
        """Set up watches for gradients, activations, or other metrics."""
        for target in targets:
            if target == "gradients":
                self._setup_gradient_watching()
            elif target == "activations":
                self._setup_activation_watching()
            self.watches[target] = []

    def set_break_conditions(self, conditions: List[str]):
        """Set conditions that will pause execution."""
        self.break_conditions = conditions
        for condition in conditions:
            if condition == "nan":
                self._setup_nan_detection()
            elif condition == "loss_spike":
                self.loss_hook = self._setup_loss_monitoring()
                return self.loss_hook  # Return the hook for testing purposes

    def enable_profiling(self):
        """Enable performance profiling."""
        self.profiling = True
        self.profiler = cProfile.Profile()
        self.profiler.enable()

    def disable_profiling(self):
        """Disable and show profiling results."""
        if self.profiling and self.profiler:
            self.profiler.disable()
            s = StringIO()
            stats = pstats.Stats(self.profiler, stream=s).sort_stats('cumulative')
            stats.print_stats()
            print(s.getvalue())
            self.profiling = False
            self.profiler = None

    def _setup_gradient_watching(self):
        """Set up hooks to watch gradients."""
        def gradient_hook(grad):
            # Convert to numpy and flatten
            grad_np = grad.detach().cpu().numpy()
            self.gradient_history.append(grad_np.ravel())  # Use ravel() for consistent 1D arrays
            if "nan" in self.break_conditions and torch.isnan(grad).any():
                raise NeuroError(
                    "NaN gradient detected",
                    context=get_context(),
                    suggestions=["Check learning rate", "Check for gradient explosion"]
                )
            return grad

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(gradient_hook)
                self._hooks.append(hook)

    def _setup_activation_watching(self):
        """Set up hooks to watch activations."""
        def activation_hook(module, input, output):
            # Convert to numpy and flatten
            output_np = output.detach().cpu().numpy()
            self.activation_history.append(output_np.ravel())  # Use ravel() for consistent 1D arrays
            if "nan" in self.break_conditions and torch.isnan(output).any():
                raise NeuroError(
                    "NaN activation detected",
                    context=get_context(),
                    suggestions=["Check model architecture", "Check input normalization"]
                )

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ReLU)):  # Only watch specific layers
                hook = module.register_forward_hook(activation_hook)
                self._hooks.append(hook)

    def _setup_nan_detection(self):
        """Set up NaN detection in both forward and backward passes."""
        def nan_hook(module, grad_input, grad_output):
            if any(g is not None and torch.isnan(g).any() for g in grad_input):
                raise NeuroError(
                    "NaN detected in gradients",
                    context=get_context(),
                    suggestions=[
                        "Reduce learning rate",
                        "Add gradient clipping",
                        "Check for division by zero"
                    ]
                )

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):  # Only watch specific layers
                hook = module.register_full_backward_hook(nan_hook)  # Use full backward hook
                self._hooks.append(hook)

    def _setup_loss_monitoring(self):
        """Set up loss spike detection."""
        self.loss_history = []
        
        def loss_hook(loss):
            if self.loss_history:
                prev_loss = self.loss_history[-1]
                if loss > prev_loss * 2:  # Loss more than doubled
                    raise NeuroError(
                        "Loss spike detected",
                        context=get_context(),
                        suggestions=[
                            "Reduce learning rate",
                            "Check for outliers in batch",
                            "Consider gradient clipping"
                        ]
                    )
            self.loss_history.append(float(loss))
            return loss

        return loss_hook

    def get_watch_summary(self) -> Dict[str, Any]:
        """Get summary of watched variables."""
        summary = {}
        
        if "gradients" in self.watches and self.gradient_history:
            try:
                # Stack 1D arrays and compute statistics
                grads = np.stack(self.gradient_history)
                summary["gradients"] = {
                    "mean": float(np.mean(grads)),
                    "std": float(np.std(grads)),
                    "max": float(np.max(grads)),
                    "min": float(np.min(grads))
                }
            except ValueError:
                # If stacking fails, compute statistics on concatenated array
                grads = np.concatenate(self.gradient_history)
                summary["gradients"] = {
                    "mean": float(np.mean(grads)),
                    "std": float(np.std(grads)),
                    "max": float(np.max(grads)),
                    "min": float(np.min(grads))
                }
        
        if "activations" in self.watches and self.activation_history:
            try:
                # Stack 1D arrays and compute statistics
                acts = np.stack(self.activation_history)
                summary["activations"] = {
                    "mean": float(np.mean(acts)),
                    "std": float(np.std(acts)),
                    "max": float(np.max(acts)),
                    "min": float(np.min(acts))
                }
            except ValueError:
                # If stacking fails, compute statistics on concatenated array
                acts = np.concatenate(self.activation_history)
                summary["activations"] = {
                    "mean": float(np.mean(acts)),
                    "std": float(np.std(acts)),
                    "max": float(np.max(acts)),
                    "min": float(np.min(acts))
                }
        
        return summary

    def cleanup(self):
        """Remove all hooks and clear history."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self.gradient_history.clear()
        self.activation_history.clear()
        if hasattr(self, 'loss_history'):
            self.loss_history.clear()

class DebugContext:
    """Context manager for debugging blocks."""
    def __init__(self, model: nn.Module, **debug_options):
        self.debugger = ModelDebugger(model)
        self.debug_options = debug_options

    def __enter__(self):
        if 'watch' in self.debug_options:
            self.debugger.watch(self.debug_options['watch'])
        if 'break_on' in self.debug_options:
            self.debugger.set_break_conditions(self.debug_options['break_on'])
        if self.debug_options.get('profile', False):
            self.debugger.enable_profiling()
        return self.debugger

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.debug_options.get('profile', False):
            self.debugger.disable_profiling()
        self.debugger.cleanup()
        return False  # Don't suppress exceptions 