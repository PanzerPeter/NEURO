import abc
import torch
import torch.nn as nn
from typing import Tuple, Optional, Union, Dict, Any
import math

# Import Neuro types for signature definition
from .neuro_types import NeuroType, LayerType, TensorType, NEURO_ANY, NEURO_FLOAT, AnyType
from src.errors import NeuroTypeError # Add NeuroTypeError import

class Layer(nn.Module):
    """Base class for all NEURO layers."""
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def get_config(self) -> dict:
        """Return the layer's configuration dictionary."""
        raise NotImplementedError

    # Helper to parse kernel/stride/padding which can be int or tuple
    def _parse_2d_param(self, param: Union[int, Tuple[int, int]], name: str) -> Tuple[int, int]:
        if isinstance(param, int):
            return (param, param)
        elif isinstance(param, tuple) and len(param) == 2 and all(isinstance(x, int) for x in param):
            return param
        else:
            # This should ideally be caught during param validation earlier
            raise ValueError(f"Invalid 2D parameter format: {param}. Expected int or tuple[int, int] for {name}.")

    def get_type_signature(self, input_type: NeuroType) -> LayerType:
        """
        Calculate and return the LayerType (input/output) based on the 
        expected input type and layer configuration.

        Args:
            input_type: The NeuroType expected as input to this layer.

        Returns:
            A LayerType instance describing the expected input and calculated 
            output types of this specific layer instance.
        """
        # Default implementation: assumes layer doesn't change the type
        # Subclasses MUST override this for type/shape transformations
        if not isinstance(input_type, (TensorType, AnyType)):
             # Basic check: Layers typically operate on Tensors
             # Raise or return a LayerType indicating incompatibility?
             # For now, let's assume AnyType if input is not Tensor
             pass # Keep input_type as is, maybe Any

        # This default is often wrong, subclasses need to provide specifics!
        return LayerType(input_type=input_type, output_type=input_type)

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
        # Use get_config for a more informative repr
        config = self.get_config()
        config_str = ", ".join(f"{k}={v!r}" for k, v in config.items())
        return f"{self.__class__.__name__}({config_str})"

# --- Concrete Layer Implementations ---

class DenseLayer(Layer):
    """Represents a fully connected (dense) layer."""
    def __init__(self, units: int, activation: Optional[str] = None, input_shape: Optional[Tuple[int, ...]] = None):
        super().__init__()
        # We need input_shape to define the layer, 
        # but it's often determined dynamically in PyTorch
        # For type checking, we rely on the input_type passed to get_type_signature
        self.units = units
        self.activation_name = activation
        # Store input_shape if provided, might be useful but not strictly necessary for type sig
        self._input_shape_hint = input_shape 

    def get_config(self) -> dict:
        config = {'units': self.units}
        if self.activation_name:
            config['activation'] = self.activation_name
        # Maybe include input_shape_hint if needed?
        # if self._input_shape_hint:
        #    config['input_shape'] = self._input_shape_hint
        return config

    def get_type_signature(self, input_type: NeuroType) -> LayerType:
        # Handle AnyType input: Assume it will resolve to a compatible Tensor
        if isinstance(input_type, AnyType):
            # Assume a 2D tensor input with unknown features for the type signature
            effective_input_type = TensorType(shape=(None, None), dtype=NEURO_FLOAT) 
        elif isinstance(input_type, TensorType):
            if input_type.shape is not None and len(input_type.shape) != 2:
                 # Dense expects 2D input (Batch, Features)
                 # TODO: Handle Flatten implicitly? For now, require 2D.
                 raise NeuroTypeError(f"Dense layer expects a 2D input tensor (Batch, Features), but received shape {input_type.shape}.")
            # If checks pass, input_type is a valid TensorType for Dense
            effective_input_type = input_type
        else:
            # If not AnyType or TensorType, it's an error
            raise NeuroTypeError(f"Dense layer expects a TensorType input, but received {input_type!r}.")

        # Calculate output shape based on the *effective* input type
        output_shape: Optional[Tuple[Optional[int], ...]] = None
        if effective_input_type.shape is not None:
            # Assume input shape is (batch, features_in)
            # Output shape will be (batch, units)
            batch_dim = effective_input_type.shape[0] # Keep batch dim (could be None)
            output_shape = (batch_dim, self.units)
        else:
            # If input shape unknown, output shape is also partially unknown
            output_shape = (None, self.units)

        # Dense layers typically output floats
        output_dtype = NEURO_FLOAT
        output_type = TensorType(shape=output_shape, dtype=output_dtype)

        # TODO: Consider activation function's potential impact on output type/range (e.g., Sigmoid -> [0,1])
        # This might be too detailed for this level of type checking.

        return LayerType(input_type=effective_input_type, output_type=output_type)


class Conv2dLayer(Layer):
    """Represents a 2D convolutional layer."""
    def __init__(self, out_channels: int, kernel_size: Union[int, Tuple[int, int]], in_channels: Optional[int] = None, stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int], str] = 0, activation: Optional[str] = None):
        super().__init__()
        self.in_channels = in_channels # May be inferred later
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation_name = activation
        # Actual nn.Conv2d layer will be created in the NeuralNetwork model

    def get_config(self) -> dict:
        config = {
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
        }
        if self.in_channels is not None:
             config['in_channels'] = self.in_channels
        if self.activation_name:
            config['activation'] = self.activation_name
        return config

    def get_type_signature(self, input_type: NeuroType) -> LayerType:
        # Conv2D expects input like (batch, channels_in, height_in, width_in)
        if not isinstance(input_type, TensorType) or input_type.shape is None or len(input_type.shape) != 4:
            # Mark incompatible or return generic signature
            raise NeuroTypeError(f"Conv2D layer expects a 4D input tensor (Batch, Channels, Height, Width), but received {input_type!r}.")
            # effective_input_type = TensorType(shape=(None, None, None, None), dtype=NEURO_ANY)
            # output_type = TensorType(shape=(None, self.out_channels, None, None), dtype=NEURO_FLOAT)
        else:
            effective_input_type = input_type
            b, c_in, h_in, w_in = input_type.shape
            
            # Validate in_channels if provided
            if self.in_channels is not None and c_in is not None and self.in_channels != c_in:
                raise NeuroTypeError(f"Conv2D layer defined with in_channels={self.in_channels} but received input with {c_in} channels.")

            # Calculate output shape
            try:
                kH, kW = self._parse_2d_param(self.kernel_size, "kernel_size")
                sH, sW = self._parse_2d_param(self.stride, "stride")
                
                # Handle padding ('same', 'valid', int, or tuple)
                if isinstance(self.padding, str):
                    padding_str = self.padding.lower()
                    if padding_str == 'valid':
                        pH, pW = 0, 0
                    elif padding_str == 'same':
                        if h_in is None or w_in is None:
                            # Cannot calculate 'same' padding without concrete input size
                            raise NeuroTypeError("Cannot calculate 'same' padding for Conv2D layer without a concrete input height and width.")
                        if sH is None or sW is None or kH is None or kW is None:
                             raise NeuroTypeError("Cannot calculate 'same' padding for Conv2D layer without concrete stride and kernel size.")
                        # Calculate padding needed for 'same'
                        # Output size H_out = ceil(H_in / sH)
                        # P = floor(((H_out - 1) * sH + kH - H_in) / 2)
                        h_out_same = math.ceil(h_in / sH)
                        w_out_same = math.ceil(w_in / sW)
                        pH = math.floor(((h_out_same - 1) * sH + kH - h_in) / 2)
                        pW = math.floor(((w_out_same - 1) * sW + kW - w_in) / 2)
                        if pH < 0 or pW < 0:
                             # This can happen with stride > 1 if not handled carefully
                             # For simplicity, error out if calculated padding is negative
                             raise NeuroTypeError(f"'same' padding calculation resulted in negative padding ({pH}, {pW}). Check input size, kernel size, and stride.")

                    else:
                        # Invalid padding string
                        raise NeuroTypeError(f"Invalid padding string '{self.padding}' for Conv2D layer. Use 'same' or 'valid'.")
                else:
                    # Numeric padding
                    pH, pW = self._parse_2d_param(self.padding, "padding")

            except (ValueError, TypeError) as e:
                # Invalid kernel/stride/padding format from _parse_2d_param
                raise NeuroTypeError(f"Invalid parameter format for Conv2D layer: {e}") from e
            except NeuroTypeError: # Re-raise specific errors from 'same' padding calc
                raise
            except Exception as e: # Catch unexpected errors during param parsing
                raise NeuroTypeError(f"Internal error parsing parameters for Conv2D: {e}") from e

            # Calculate output H, W if possible
            h_out: Optional[int] = None
            if h_in is not None and kH is not None and sH is not None and pH is not None:
                if h_in + 2 * pH < kH:
                    raise NeuroTypeError(f"Conv2D input height ({h_in}) + padding ({pH}*2) is smaller than kernel height ({kH}).")
                h_out = math.floor(((h_in + 2 * pH - kH) / sH) + 1)
                if h_out <= 0:
                    raise NeuroTypeError(f"Conv2D output height calculated to be non-positive ({h_out}). Check input size, kernel, padding, and stride.")

            w_out: Optional[int] = None
            if w_in is not None and kW is not None and sW is not None and pW is not None:
                 if w_in + 2 * pW < kW:
                     raise NeuroTypeError(f"Conv2D input width ({w_in}) + padding ({pW}*2) is smaller than kernel width ({kW}).")
                 w_out = math.floor(((w_in + 2 * pW - kW) / sW) + 1)
                 if w_out <= 0:
                    raise NeuroTypeError(f"Conv2D output width calculated to be non-positive ({w_out}). Check input size, kernel, padding, and stride.")

            output_shape = (b, self.out_channels, h_out, w_out)
            # Conv layers typically preserve or output floats
            # Use input dtype unless it's Any, then default to Float
            output_dtype = NEURO_FLOAT if isinstance(input_type.dtype, AnyType) else input_type.dtype
            output_type = TensorType(shape=output_shape, dtype=output_dtype)

        return LayerType(input_type=effective_input_type, output_type=output_type)

# ... (Other layer classes need get_config and get_type_signature) ...

class BatchNormLayer(Layer):
    """Represents a Batch Normalization layer."""
    def __init__(self, momentum=0.1, eps=1e-5, dim=None): # dim can be used to distinguish 1D/2D
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.dim = dim # Store dimension (e.g., 1 for Dense, 2 for Conv2D)
        # Actual nn.BatchNorm1d/2d created in NeuralNetwork model

    def get_config(self) -> dict:
        config = {'momentum': self.momentum, 'eps': self.eps}
        if self.dim is not None:
            config['dim'] = self.dim
        return config

    def get_type_signature(self, input_type: NeuroType) -> LayerType:
        # BatchNorm preserves shape and dtype
        # The main check is dimensionality (e.g., BatchNorm1D vs 2D)
        # For now, assume it preserves the input type exactly.
        # A more robust check might validate len(input_type.shape).
        return LayerType(input_type=input_type, output_type=input_type)

class FlattenLayer(Layer):
    """Represents a Flatten layer."""
    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim
        # Actual nn.Flatten created in NeuralNetwork model

    def get_config(self) -> dict:
        return {'start_dim': self.start_dim, 'end_dim': self.end_dim}

    def get_type_signature(self, input_type: NeuroType) -> LayerType:
        if not isinstance(input_type, TensorType) or input_type.shape is None:
            # Cannot determine output shape without input shape
            effective_input_type = TensorType()
            output_type = TensorType(shape=(None, None)) # Usually (Batch, Features)
        else:
            effective_input_type = input_type
            # Simplified: Flatten usually results in (Batch, Features)
            # Calculate Features based on start/end dim and input shape
            # For now, just represent as (Batch, UnknownFeatures)
            batch_dim = input_type.shape[0] if len(input_type.shape) > 0 else None
            output_shape = (batch_dim, None)
            output_type = TensorType(shape=output_shape, dtype=input_type.dtype)

        return LayerType(input_type=effective_input_type, output_type=output_type)

class MaxPool2dLayer(Layer):
    """Represents a 2D Max Pooling layer."""
    def __init__(self, kernel_size: Union[int, Tuple[int, int]], stride: Optional[Union[int, Tuple[int, int]]] = None, padding: Union[int, Tuple[int, int], str] = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size # Default stride = kernel_size
        self.padding = padding
        # Actual nn.MaxPool2d created in NeuralNetwork model

    def get_config(self) -> dict:
        return {
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
        }

    def get_type_signature(self, input_type: NeuroType) -> LayerType:
        # MaxPool2D expects input like (batch, channels_in, height_in, width_in)
        if not isinstance(input_type, TensorType) or input_type.shape is None or len(input_type.shape) != 4:
            raise NeuroTypeError(f"MaxPool2D layer expects a 4D input tensor (Batch, Channels, Height, Width), but received {input_type!r}.")

        effective_input_type = input_type
        b, c_in, h_in, w_in = input_type.shape

        # Calculate output shape - uses same logic as Conv2D
        try:
            kH, kW = self._parse_2d_param(self.kernel_size, "kernel_size")
            sH, sW = self._parse_2d_param(self.stride, "stride")

            if isinstance(self.padding, str):
                 # Pooling layers typically don't support 'same'/'valid' strings in PyTorch type interfaces
                 # We could implement calculation similar to Conv2D 'same', but for clarity let's require numeric padding here.
                 # PyTorch nn.MaxPool2d itself only accepts int/tuple for padding.
                 raise NeuroTypeError(f"String padding ('{self.padding}') not supported for MaxPool2D layer type checking. Use integer or tuple padding.")
                 # pH, pW = None, None # Mark as error for now
            else:
                pH, pW = self._parse_2d_param(self.padding, "padding")

        except (ValueError, TypeError) as e:
             raise NeuroTypeError(f"Invalid parameter format for MaxPool2D layer: {e}") from e
        except NeuroTypeError: # Re-raise specific errors
             raise
        except Exception as e: # Catch unexpected errors
             raise NeuroTypeError(f"Internal error parsing parameters for MaxPool2D: {e}") from e

        # Calculate output H, W
        h_out: Optional[int] = None
        if h_in is not None and kH is not None and sH is not None and pH is not None:
            if h_in + 2 * pH < kH:
                 raise NeuroTypeError(f"MaxPool2D input height ({h_in}) + padding ({pH}*2) is smaller than kernel height ({kH}).")
            # MaxPool2D uses ceil_mode=False by default in PyTorch. Formula: floor((Hin + 2*P - K) / S) + 1
            h_out = math.floor(((h_in + 2 * pH - kH) / sH) + 1)
            if h_out <= 0:
                raise NeuroTypeError(f"MaxPool2D output height calculated to be non-positive ({h_out}). Check input size, kernel, padding, and stride.")

        w_out: Optional[int] = None
        if w_in is not None and kW is not None and sW is not None and pW is not None:
            if w_in + 2 * pW < kW:
                 raise NeuroTypeError(f"MaxPool2D input width ({w_in}) + padding ({pW}*2) is smaller than kernel width ({kW}).")
            w_out = math.floor(((w_in + 2 * pW - kW) / sW) + 1)
            if w_out <= 0:
                 raise NeuroTypeError(f"MaxPool2D output width calculated to be non-positive ({w_out}). Check input size, kernel, padding, and stride.")

        # MaxPool preserves channel count
        output_shape = (b, c_in, h_out, w_out)
        # MaxPool preserves dtype
        output_dtype = input_type.dtype
        output_type = TensorType(shape=output_shape, dtype=output_dtype)

        return LayerType(input_type=effective_input_type, output_type=output_type)

class AvgPool2dLayer(Layer):
    """Represents a 2D Average Pooling layer."""
    def __init__(self, kernel_size: Union[int, Tuple[int, int]], stride: Optional[Union[int, Tuple[int, int]]] = None, padding: Union[int, Tuple[int, int], str] = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        # Actual nn.AvgPool2d created in NeuralNetwork model

    def get_config(self) -> dict:
        return {
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
        }

    def get_type_signature(self, input_type: NeuroType) -> LayerType:
        # Same signature logic as MaxPool2d
        if not isinstance(input_type, TensorType) or input_type.shape is None or len(input_type.shape) != 4:
            raise NeuroTypeError(f"AvgPool2D layer expects a 4D input tensor (Batch, Channels, Height, Width), but received {input_type!r}.")

        effective_input_type = input_type
        b, c_in, h_in, w_in = input_type.shape

        try:
            kH, kW = self._parse_2d_param(self.kernel_size, "kernel_size")
            sH, sW = self._parse_2d_param(self.stride, "stride")

            if isinstance(self.padding, str):
                raise NeuroTypeError(f"String padding ('{self.padding}') not supported for AvgPool2D layer type checking. Use integer or tuple padding.")
            else:
                pH, pW = self._parse_2d_param(self.padding, "padding")

        except (ValueError, TypeError) as e:
             raise NeuroTypeError(f"Invalid parameter format for AvgPool2D layer: {e}") from e
        except NeuroTypeError: # Re-raise specific errors
             raise
        except Exception as e: # Catch unexpected errors
             raise NeuroTypeError(f"Internal error parsing parameters for AvgPool2D: {e}") from e


        h_out: Optional[int] = None
        if h_in is not None and kH is not None and sH is not None and pH is not None:
             if h_in + 2 * pH < kH:
                  raise NeuroTypeError(f"AvgPool2D input height ({h_in}) + padding ({pH}*2) is smaller than kernel height ({kH}).")
             # AvgPool2D formula: floor((Hin + 2*P - K) / S) + 1
             h_out = math.floor(((h_in + 2 * pH - kH) / sH) + 1)
             if h_out <= 0:
                 raise NeuroTypeError(f"AvgPool2D output height calculated to be non-positive ({h_out}). Check input size, kernel, padding, and stride.")


        w_out: Optional[int] = None
        if w_in is not None and kW is not None and sW is not None and pW is not None:
            if w_in + 2 * pW < kW:
                 raise NeuroTypeError(f"AvgPool2D input width ({w_in}) + padding ({pW}*2) is smaller than kernel width ({kW}).")
            w_out = math.floor(((w_in + 2 * pW - kW) / sW) + 1)
            if w_out <= 0:
                 raise NeuroTypeError(f"AvgPool2D output width calculated to be non-positive ({w_out}). Check input size, kernel, padding, and stride.")

        # AvgPool preserves channel count
        output_shape = (b, c_in, h_out, w_out)
        # AvgPool preserves dtype (usually float anyway)
        output_dtype = input_type.dtype
        output_type = TensorType(shape=output_shape, dtype=output_dtype)

        return LayerType(input_type=effective_input_type, output_type=output_type)

