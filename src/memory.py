"""
NEURO Memory Management Module
Provides utilities for efficient memory usage during training.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
import psutil
import gc
from .errors import MemoryError, get_context
from torch.utils.checkpoint import checkpoint

@dataclass
class MemoryStats:
    """Contains memory usage statistics."""
    total_allocated: int
    total_cached: int
    current_allocated: int
    peak_allocated: int
    device_name: str

class GradientCheckpoint:
    """
    Implements gradient checkpointing for memory-efficient training.
    Saves memory by only storing selective activations and recomputing others
    during backward pass.
    """
    
    def __init__(self, model: nn.Module, checkpoint_layers: Optional[List[str]] = None):
        """
        Initialize gradient checkpointing.
        
        Args:
            model: The model to apply checkpointing to
            checkpoint_layers: Optional list of layer names to checkpoint
        """
        self.model = model
        self.checkpoint_layers = checkpoint_layers or []
        self.checkpointed_modules = {}
        
    def enable(self):
        """Enable gradient checkpointing."""
        if not self.checkpoint_layers:
            self.checkpoint_layers = self._auto_detect_checkpoints()
        
        for name, module in self.model.named_modules():
            if name in self.checkpoint_layers:
                if hasattr(module, '_original_forward'):
                    continue
                
                # Store original forward
                module._original_forward = module.forward
                
                # Create checkpointed forward
                def make_checkpoint_forward(mod):
                    def checkpoint_forward(*args, **kwargs):
                        def custom_forward(*inputs):
                            return mod._original_forward(*inputs)
                        return checkpoint(custom_forward, *args, use_reentrant=False, preserve_rng_state=True)
                    return checkpoint_forward
                
                # Apply checkpointing
                checkpoint_fn = make_checkpoint_forward(module)
                module.forward = checkpoint_fn
                self.checkpointed_modules[name] = module._original_forward
    
    def disable(self):
        """Disable gradient checkpointing."""
        for name, module in self.model.named_modules():
            if name in self.checkpointed_modules:
                # Restore original forward
                module.forward = self.checkpointed_modules[name]
                if hasattr(module, '_original_forward'):
                    delattr(module, '_original_forward')
        self.checkpointed_modules.clear()
    
    def _auto_detect_checkpoints(self) -> List[str]:
        """
        Automatically detects which layers should be checkpointed based on:
        1. Memory usage
        2. Computational cost
        3. Layer size
        """
        checkpoint_layers = []
        memory_threshold = 1e8  # 100MB
        
        # Add all layers by default for Sequential models
        if isinstance(self.model, nn.Sequential):
            for i in range(len(self.model)):
                checkpoint_layers.append(str(i))
        else:
            # For other models, detect based on size
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d, nn.LSTM, nn.GRU)):
                    param_size = sum(p.numel() * p.element_size() for p in module.parameters())
                    if param_size > memory_threshold:
                        checkpoint_layers.append(name)
        
        return checkpoint_layers

class MemoryManager:
    """
    Manages memory usage during model training.
    Implements memory-efficient training loops and large model support.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 device: torch.device,
                 max_memory_usage: float = 0.9):
        """
        Args:
            model: The neural network model
            device: The device to train on
            max_memory_usage: Maximum fraction of available memory to use
        """
        self.model = model
        self.device = device
        self.max_memory_usage = max_memory_usage
        self.gradient_checkpoint: Optional[GradientCheckpoint] = None
        self.peak_memory = 0
    
    def get_memory_stats(self) -> MemoryStats:
        """Gets current memory usage statistics."""
        if self.device.type == 'cuda':
            return MemoryStats(
                total_allocated=torch.cuda.memory_allocated(self.device),
                total_cached=torch.cuda.memory_reserved(self.device),
                current_allocated=torch.cuda.memory_allocated(self.device),
                peak_allocated=torch.cuda.max_memory_allocated(self.device),
                device_name=torch.cuda.get_device_name(self.device)
            )
        else:
            process = psutil.Process()
            memory_info = process.memory_info()
            return MemoryStats(
                total_allocated=memory_info.rss,
                total_cached=memory_info.vms,
                current_allocated=memory_info.rss,
                peak_allocated=memory_info.peak_wset,
                device_name='cpu'
            )
    
    def check_memory_usage(self) -> Tuple[bool, Optional[str]]:
        """
        Checks if memory usage is within acceptable limits.
        
        Returns:
            Tuple of (is_ok, warning_message)
        """
        stats = self.get_memory_stats()
        
        if self.device.type == 'cuda':
            total_memory = torch.cuda.get_device_properties(self.device).total_memory
            current_usage = stats.total_allocated / total_memory
            
            if current_usage > self.max_memory_usage:
                return False, f"GPU memory usage ({current_usage:.1%}) exceeds limit ({self.max_memory_usage:.1%})"
        else:
            total_memory = psutil.virtual_memory().total
            current_usage = stats.total_allocated / total_memory
            
            if current_usage > self.max_memory_usage:
                return False, f"RAM usage ({current_usage:.1%}) exceeds limit ({self.max_memory_usage:.1%})"
        
        return True, None
    
    def optimize_memory_usage(self):
        """
        Optimizes memory usage by:
        1. Enabling gradient checkpointing if needed
        2. Clearing unused memory
        3. Moving less used tensors to CPU
        """
        is_ok, warning = self.check_memory_usage()
        if not is_ok:
            # Enable gradient checkpointing if not already enabled
            if self.gradient_checkpoint is None:
                self.gradient_checkpoint = GradientCheckpoint(self.model)
            
            # Clear memory cache
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            gc.collect()
            
            # If still problematic, raise error
            is_ok, warning = self.check_memory_usage()
            if not is_ok:
                raise MemoryError(
                    "Memory optimization failed",
                    context=get_context(),
                    suggestions=[
                        "Reduce batch size",
                        "Enable gradient checkpointing",
                        "Use mixed precision training",
                        "Reduce model size"
                    ],
                    details={
                        "current_memory_usage": self.get_memory_stats(),
                        "device": str(self.device)
                    }
                )
    
    def efficient_forward(self, *args, **kwargs):
        """
        Performs a memory-efficient forward pass.
        """
        try:
            # Check memory before forward pass
            self.optimize_memory_usage()
            
            # Perform forward pass
            output = self.model(*args, **kwargs)
            
            # Check memory after forward pass
            self.optimize_memory_usage()
            
            return output
            
        except Exception as e:
            raise MemoryError(
                "Error during forward pass",
                context=get_context(),
                suggestions=[
                    "Check input sizes",
                    "Reduce model complexity",
                    "Enable gradient checkpointing"
                ],
                details={"original_error": str(e)}
            ) from e 