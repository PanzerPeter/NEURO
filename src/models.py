import torch
import torch.nn as nn
from collections import OrderedDict
from src.errors import NeuroTypeError # Import the custom error type

# Placeholder for layer mapping (will be expanded)
LAYER_REGISTRY = {
    # 'Dense': nn.LazyLinear, # Using LazyLinear initially might simplify input size handling
    # 'Dropout': nn.Dropout,
    # Add other layer types here
}

# Activation function mapping
ACTIVATION_REGISTRY = {
    'relu': nn.ReLU,
    'sigmoid': nn.Sigmoid,
    # Add other activation functions here as needed (tanh, softmax, etc.)
}

class NeuralNetwork(nn.Module):
    """
    Represents a neural network defined using the NEURO language.
    Dynamically builds the network architecture based on parsed AST nodes.
    """
    def __init__(self, name, params, layer_nodes):
        super().__init__()
        self.model_name = name
        self.input_size = params.get('input_size') # Store for potential future use/validation
        self.output_size = params.get('output_size') # Store for potential future use/validation
        
        print(f"Initializing PyTorch NeuralNetwork: {name}")
        print(f"  Input Size: {self.input_size}, Output Size: {self.output_size}")
        
        # Build the sequential model from layer definitions
        layers = OrderedDict()
        for i, layer_node in enumerate(layer_nodes):
            layer_type = layer_node.layer_type
            layer_params = layer_node.params
            
            print(f"  Adding Layer {i}: Type={layer_type}, Params={layer_params}")
            
            # --- Basic Layer Implementation (Placeholders) ---
            # This needs to be significantly expanded
            
            # Example: Handling Dense layer (very basic)
            if layer_type == 'Dense':
                units = layer_params.get('units')
                if units is None:
                    # Raise a NeuroTypeError for missing required parameters
                    raise NeuroTypeError(f"Missing required parameter 'units' for layer type Dense (Layer {i})")
                
                # Using LazyLinear avoids needing input_features immediately
                layer_module = torch.nn.LazyLinear(out_features=int(units))
                layers[f'dense_{i}'] = layer_module
                
                activation_name = layer_params.get('activation')
                if activation_name:
                    # Look up and add the activation function module
                    act_module_class = ACTIVATION_REGISTRY.get(activation_name.lower()) # Case-insensitive lookup
                    if act_module_class:
                        layers[f'activation_{i}'] = act_module_class()
                        print(f"    Added Activation: {activation_name}")
                    else:
                        # Raise an error if the activation function isn't recognized
                        raise NeuroTypeError(f"Unknown activation function: '{activation_name}' for layer type Dense (Layer {i}). Supported: {list(ACTIVATION_REGISTRY.keys())}")
            
            # Example: Handling Dropout (very basic)
            elif layer_type == 'Dropout':
                 rate = layer_params.get('rate')
                 if rate is None:
                     raise NeuroTypeError(f"Missing required parameter 'rate' for layer type Dropout (Layer {i})")
                 layer_module = torch.nn.Dropout(p=float(rate))
                 layers[f'dropout_{i}'] = layer_module
                 
            else:
                # Raise an error for unknown/unsupported layer types
                raise NeuroTypeError(f"Unknown or unsupported layer type: {layer_type} (Layer {i})")

        # Create the sequential container
        self.network = nn.Sequential(layers)
        print(f"  Finished building network sequence: {self.network}")

    def forward(self, x):
        """Defines the forward pass of the network."""
        return self.network(x)

    def __str__(self):
        # Provide a string representation including the model structure
        return f"NeuralNetwork(name='{self.model_name}')\n{super().__str__()}" 