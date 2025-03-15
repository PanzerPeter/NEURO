"""
NEURO Matrix Implementation

This module implements the NeuroMatrix format (.nrm) for NEURO.
It provides functionality for handling AI datasets and model configurations.

Key responsibilities:
- Define NeuroMatrix data structure
- Load and save .nrm files
- Data validation and preprocessing
- Memory-efficient data handling
- Integration with PyTorch tensors
- Support for common data operations
- Dataset splitting and batching

This module is crucial for data handling in NEURO.
"""

import json
import torch
import numpy as np
from typing import Dict, List, Tuple, Union, Any, Optional, Iterator
from datetime import datetime
from .errors import NeuroValueError, NeuroShapeError
import pickle

class NeuroMatrix:
    """Matrix data structure for NEURO language."""
    
    def __init__(self):
        """Initialize an empty matrix."""
        self._inputs = None
        self._outputs = None
        self._normalized = False
        self._input_shape = None
        self._output_shape = None
        self._num_samples = 0
        self._metadata = {}
    
    @property
    def inputs(self):
        """Get input data."""
        return self._inputs
    
    @property
    def outputs(self):
        """Get output data."""
        return self._outputs
    
    @property
    def input_shape(self):
        """Get input shape."""
        return self._input_shape
    
    @property
    def output_shape(self):
        """Get output shape."""
        return self._output_shape
    
    @property
    def num_samples(self):
        """Get number of samples."""
        return self._num_samples
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get matrix metadata."""
        return self._metadata
    
    @metadata.setter
    def metadata(self, value: Dict[str, Any]) -> None:
        """Set matrix metadata.
        
        Args:
            value: Dictionary of metadata
        """
        if not isinstance(value, dict):
            raise TypeError("Metadata must be a dictionary")
        self._metadata = value.copy()
    
    def add_data(self, inputs: np.ndarray, outputs: np.ndarray) -> None:
        """Add data to the matrix.
        
        Args:
            inputs: Input data array
            outputs: Output data array
            
        Raises:
            NeuroShapeError: If input and output shapes are incompatible
            NeuroValueError: If data contains invalid values
        """
        # Validate input types
        if not isinstance(inputs, np.ndarray) or not isinstance(outputs, np.ndarray):
            raise NeuroValueError("Inputs and outputs must be numpy arrays")

        # Validate shapes
        if len(inputs) != len(outputs):
            raise NeuroShapeError(
                "Number of input and output samples must match",
                expected_shape=(len(inputs),),
                actual_shape=(len(outputs),)
            )

        # Initialize or append data
        if self._inputs is None:
            self._inputs = inputs
            self._outputs = outputs
        else:
            if inputs.shape[1:] != self._inputs.shape[1:]:
                raise NeuroShapeError(
                    "Input shape mismatch",
                    expected_shape=self._inputs.shape[1:],
                    actual_shape=inputs.shape[1:]
                )
            if outputs.shape[1:] != self._outputs.shape[1:]:
                raise NeuroShapeError(
                    "Output shape mismatch",
                    expected_shape=self._outputs.shape[1:],
                    actual_shape=outputs.shape[1:]
                )
            self._inputs = np.vstack([self._inputs, inputs])
            self._outputs = np.vstack([self._outputs, outputs])

        self._input_shape = self._inputs.shape[1:]
        self._output_shape = self._outputs.shape[1:]
        self._num_samples = len(self._inputs)
        self._normalized = False
    
    def normalize(self) -> None:
        """Normalize input data."""
        if not self._normalized and self._inputs is not None:
            self._inputs = (self._inputs - np.mean(self._inputs, axis=0)) / np.std(self._inputs, axis=0)
            self._normalized = True
    
    def shuffle(self) -> None:
        """Randomly shuffle the data."""
        if self._num_samples == 0:
            return

        indices = np.random.permutation(self._num_samples)
        self._inputs = self._inputs[indices]
        self._outputs = self._outputs[indices]
    
    def to_torch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert data to PyTorch tensors.
        
        Returns:
            Tuple of (input_tensor, output_tensor)
        """
        if self._inputs is None or self._outputs is None:
            raise ValueError("No data available to convert")

        return (
            torch.from_numpy(self._inputs.astype(np.float32)),
            torch.from_numpy(self._outputs.astype(np.float32))
        )
    
    def create_dataloader(self, batch_size: int = 32) -> torch.utils.data.DataLoader:
        """Create a PyTorch DataLoader.
        
        Args:
            batch_size: Size of each batch
            
        Returns:
            DataLoader instance
        """
        inputs, outputs = self.to_torch()
        dataset = torch.utils.data.TensorDataset(inputs, outputs)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def train_test_split(self, test_size: float = 0.2) -> Tuple['NeuroMatrix', 'NeuroMatrix']:
        """Split data into training and test sets."""
        if self._inputs is None or self._outputs is None:
            raise NeuroValueError("No data to split")
        
        # Create indices
        indices = np.random.permutation(self._num_samples)
        test_samples = int(self._num_samples * test_size)
        test_idx = indices[:test_samples]
        train_idx = indices[test_samples:]
        
        # Create new matrices
        train_matrix = NeuroMatrix()
        test_matrix = NeuroMatrix()
        
        # Split data
        train_matrix.add_data(self._inputs[train_idx], self._outputs[train_idx])
        test_matrix.add_data(self._inputs[test_idx], self._outputs[test_idx])
        
        return train_matrix, test_matrix
    
    def train_val_test_split(
        self,
        val_size: float = 0.2,
        test_size: float = 0.2,
        shuffle: bool = True
    ) -> Tuple['NeuroMatrix', 'NeuroMatrix', 'NeuroMatrix']:
        """Split data into training, validation and test sets.
        
        Args:
            val_size: Fraction of data for validation
            test_size: Fraction of data for testing
            shuffle: Whether to shuffle before splitting
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        if shuffle:
            self.shuffle()

        # Calculate split indices
        # For 100 samples with val_size=0.2, test_size=0.2:
        # - Train should be 60 samples (60%)
        # - Val should be 20 samples (20%)
        # - Test should be 20 samples (20%)
        train_size = 1.0 - val_size - test_size
        train_end = int(self._num_samples * train_size)
        val_end = train_end + int(self._num_samples * val_size)

        train_data = NeuroMatrix()
        val_data = NeuroMatrix()
        test_data = NeuroMatrix()

        # Split the data
        train_data.add_data(self._inputs[:train_end], self._outputs[:train_end])
        val_data.add_data(self._inputs[train_end:val_end], self._outputs[train_end:val_end])
        test_data.add_data(self._inputs[val_end:], self._outputs[val_end:])

        return train_data, val_data, test_data
    
    def save(self, filepath: str) -> None:
        """Save matrix data to file.
        
        Args:
            filepath: Path to save file
        """
        data = {
            'inputs': self._inputs,
            'outputs': self._outputs,
            'normalized': self._normalized,
            'metadata': self._metadata
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'NeuroMatrix':
        """Load matrix data from file.
        
        Args:
            filepath: Path to load file
            
        Returns:
            NeuroMatrix instance
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        matrix = cls()
        matrix._inputs = data['inputs']
        matrix._outputs = data['outputs']
        matrix._normalized = data['normalized']
        matrix._metadata = data.get('metadata', {})
        matrix._input_shape = matrix._inputs.shape[1:] if matrix._inputs is not None else None
        matrix._output_shape = matrix._outputs.shape[1:] if matrix._outputs is not None else None
        matrix._num_samples = len(matrix._inputs) if matrix._inputs is not None else 0
        return matrix

    def __len__(self) -> int:
        return len(self._inputs) if self._inputs is not None else 0

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._inputs is None:
            raise NeuroValueError("No data available")
        return self._inputs[idx], self._outputs[idx]

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        if self._inputs is None:
            raise NeuroValueError("No data available")
        for i in range(len(self)):
            yield self[i]

class NeuroDataLoader:
    """Data loader for batch iteration."""
    
    def __init__(self, matrix: NeuroMatrix, batch_size: int = 32):
        """Initialize the data loader."""
        self.matrix = matrix
        self.batch_size = batch_size
        self.current_idx = 0
    
    def __iter__(self):
        """Return self as iterator."""
        self.current_idx = 0
        return self
    
    def __next__(self):
        """Get next batch."""
        if self.current_idx >= len(self.matrix):
            raise StopIteration
        
        end_idx = min(self.current_idx + self.batch_size, len(self.matrix))
        batch_inputs = self.matrix.inputs[self.current_idx:end_idx]
        batch_outputs = self.matrix.outputs[self.current_idx:end_idx]
        
        self.current_idx = end_idx
        return batch_inputs, batch_outputs
    
    def __len__(self):
        """Get number of batches."""
        return (len(self.matrix) + self.batch_size - 1) // self.batch_size 