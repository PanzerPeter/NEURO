"""
Defines the type system for the Neuro language components.
"""

from typing import TypeAlias, NewType, Any, Tuple, Optional, Union, Protocol, runtime_checkable

# === Base Types ===

class NeuroType:
    """Base class for all types in the Neuro system."""
    def __repr__(self) -> str:
        return self.__class__.__name__

    def __eq__(self, other: object) -> bool:
        # Default equality check based on class type
        if not isinstance(other, NeuroType):
            return NotImplemented
        return self.__class__ == other.__class__

    def is_compatible(self, other: 'NeuroType') -> bool:
        """Checks if this type is compatible with another type."""
        # Default: compatible if they are the same type
        return self == other

class AnyType(NeuroType):
    """Represents any possible Neuro type, useful for generics or uninferrable types."""
    def is_compatible(self, other: 'NeuroType') -> bool:
        return True # AnyType is compatible with everything

class VoidType(NeuroType):
    """Represents the absence of a value (e.g., a statement that doesn't return)."""
    pass

class NumericType(NeuroType):
    """Base type for numeric values (int, float)."""
    pass

class IntType(NumericType):
    """Represents an integer type."""
    pass

class FloatType(NumericType):
    """Represents a floating-point type."""
    pass

class BoolType(NeuroType):
    """Represents a boolean type."""
    pass

class StringType(NeuroType):
    """Represents a string type."""
    pass


# === Tensor/Data Types ===

class TensorType(NeuroType):
    """Represents a tensor/matrix type, potentially with shape and dtype."""
    def __init__(self, shape: Optional[Tuple[Optional[int], ...]] = None, dtype: NeuroType = AnyType()):
        self.shape = shape # None indicates unknown dimension, empty tuple for scalar?
        self.dtype = dtype

    def __repr__(self) -> str:
        shape_str = f"shape={self.shape}" if self.shape is not None else "shape=?"
        dtype_str = f"dtype={self.dtype!r}"
        return f"{self.__class__.__name__}({shape_str}, {dtype_str})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TensorType):
            return NotImplemented
        # For equality, require exact shape and dtype match (unless AnyType)
        return (self.shape == other.shape and
                (self.dtype == other.dtype or isinstance(self.dtype, AnyType) or isinstance(other.dtype, AnyType)))

    def is_compatible(self, other: 'NeuroType') -> bool:
        """Check compatibility based on shape and dtype."""
        if isinstance(other, AnyType):
            return True
        if not isinstance(other, TensorType):
            return False

        # Dtype compatibility
        if not self.dtype.is_compatible(other.dtype) and not isinstance(self.dtype, AnyType) and not isinstance(other.dtype, AnyType):
             return False

        # Shape compatibility (more complex logic might be needed here)
        # Basic check: same number of dimensions if both known, allow None dimensions to match anything
        if self.shape is None or other.shape is None:
            return True # Assume compatible if one shape is unknown

        if len(self.shape) != len(other.shape):
            return False

        for s_dim, o_dim in zip(self.shape, other.shape):
            if s_dim is not None and o_dim is not None and s_dim != o_dim:
                return False
        return True

class DataType(NeuroType):
    """Represents loaded data, potentially split and batched."""
    def __init__(self, element_type: NeuroType = AnyType(), has_splits: bool = False):
        self.element_type = element_type # Type of individual data points (e.g., Tuple[Tensor, Tensor])
        self.has_splits = has_splits # Indicates if data has train/val/test splits

    def __repr__(self) -> str:
         return f"DataType(element={self.element_type!r}, splits={self.has_splits})"


# === Component Types ===

class LayerType(NeuroType):
    """Represents a neural network layer type."""
    # Could potentially store input/output type signatures here
    def __init__(self, input_type: NeuroType = AnyType(), output_type: NeuroType = AnyType()):
         self.input_type = input_type
         self.output_type = output_type

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(in={self.input_type!r}, out={self.output_type!r})"

    # Compatibility might involve checking if output of one layer matches input of another


class ModelType(NeuroType):
    """Represents a neural network model type."""
    # Could store the overall input/output signature of the model
    def __init__(self, input_type: NeuroType = AnyType(), output_type: NeuroType = AnyType()):
         self.input_type = input_type
         self.output_type = output_type

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(in={self.input_type!r}, out={self.output_type!r})"


class LossType(NeuroType):
    """Represents a loss function type."""
    # Could specify expected input types (predictions, targets)
    def __init__(self, pred_type: NeuroType = AnyType(), target_type: NeuroType = AnyType()):
        self.pred_type = pred_type
        self.target_type = target_type

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(pred={self.pred_type!r}, target={self.target_type!r})"


class OptimizerType(NeuroType):
    """Represents an optimizer type."""
    pass # Optimizers mainly interact with model parameters, less direct type signature needed?


# === Function/Callable Types ===

class FunctionType(NeuroType):
    """Represents a function or method type signature."""
    def __init__(self, param_types: list[NeuroType], return_type: NeuroType):
        # TODO: Support named parameters, optional parameters, variable args?
        self.param_types = param_types # List of expected argument types
        self.return_type = return_type

    def __repr__(self) -> str:
        params_str = ", ".join(map(repr, self.param_types))
        return f"FunctionType(params=[{params_str}], returns={self.return_type!r})"

    def is_compatible(self, other: 'NeuroType') -> bool:
        # Check if another function type has compatible signature (basic check)
        if not isinstance(other, FunctionType):
            return False
        if len(self.param_types) != len(other.param_types):
            return False
        if not self.return_type.is_compatible(other.return_type):
             return False
        # Check parameter compatibility
        for self_param, other_param in zip(self.param_types, other.param_types):
            # Contravariance for parameters? For now, require compatibility.
             if not self_param.is_compatible(other_param):
                 return False
        return True


# === Aliases/Constants for convenience ===
NEURO_INT = IntType()
NEURO_FLOAT = FloatType()
NEURO_BOOL = BoolType()
NEURO_STRING = StringType()
NEURO_ANY = AnyType()
NEURO_VOID = VoidType()

# Example specific tensor type
TensorFloat32 = lambda shape: TensorType(shape=shape, dtype=NEURO_FLOAT)
TensorInt32 = lambda shape: TensorType(shape=shape, dtype=NEURO_INT)

# Potentially add types for Activation Functions, DataLoaders, etc. later 