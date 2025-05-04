import torch
import torch.nn as nn
import torch.optim as optim # Import optim
from collections import OrderedDict
from src.errors import NeuroTypeError, NeuroSyntaxError # Import the custom error type
from .losses import BCELoss # Assuming losses.py is in the same directory
# We might need more losses later
# from torch.nn import MSELoss 
from .optimizers import Adam # Assuming optimizers.py is in the same directory
# We might need more optimizers later
# from torch.optim import SGD

# Placeholder for layer mapping (will be expanded)
# LAYER_REGISTRY = {
#     # 'Dense': nn.LazyLinear, # Using LazyLinear initially might simplify input size handling
#     # 'Dropout': nn.Dropout,
#     # Add other layer types here
# }

# Activation function mapping
ACTIVATION_REGISTRY = {
    'relu': nn.ReLU,
    'sigmoid': nn.Sigmoid,
    # Add other activation functions here as needed (tanh, softmax, etc.)
    'tanh': nn.Tanh,       # Added Tanh
    'softmax': nn.Softmax  # Added Softmax (consider dim parameter)
}

def _parse_int_tuple(value, param_name, layer_info):
    """Helper to parse tuple parameters like kernel_size, stride, padding."""
    if isinstance(value, int):
        return value
    elif isinstance(value, (tuple, list)) and all(isinstance(x, int) for x in value):
        return tuple(value)
    elif isinstance(value, str):
        try:
            # Attempt to parse comma-separated string like "3,3"
            parts = [int(p.strip()) for p in value.split(',')]
            if len(parts) == 1:
                return parts[0] # Single number
            elif len(parts) == 2: 
                return tuple(parts) # Pair like (3,3)
            else:
                 raise NeuroTypeError(f"Invalid tuple format for '{param_name}' in {layer_info}. Expected int or two comma-separated ints, got '{value}'")
        except ValueError:
             raise NeuroTypeError(f"Invalid integer value in '{param_name}' for {layer_info}: '{value}'")
    else:
         raise NeuroTypeError(f"Invalid type for '{param_name}' in {layer_info}. Expected int, tuple/list of ints, or string like '3,3', got {type(value)}")

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
        
        layers = OrderedDict()
        current_in_channels = None # Track channels for Conv layers
        is_first_conv = True

        for i, layer_node in enumerate(layer_nodes):
            layer_type = layer_node.layer_type
            layer_params = layer_node.params
            layer_name_base = f"layer_{i}" # Base name for the layer and its activation
            layer_info = f"Layer {i} ('{layer_type}')"
            
            print(f"  Processing {layer_info}: Params={layer_params}")
            
            if layer_type is None:
                # Skip nodes explicitly marked with None type - likely handled elsewhere (e.g., activation within another layer)
                print(f"    Skipping layer node with type None (potentially activation handled by previous layer)")
                continue
                
            activation_name = layer_params.get('activation') # Check for activation within the params
            activation_module = None
            if activation_name:
                act_module_class = ACTIVATION_REGISTRY.get(activation_name.lower())
                if act_module_class:
                    # Special handling for Softmax dimension if needed
                    if activation_name.lower() == 'softmax':
                        # Defaulting to dim=1 for typical classification tasks
                        # This might need to be configurable via layer_params in the future
                        softmax_dim = layer_params.get('dim', 1) 
                        activation_module = act_module_class(dim=int(softmax_dim))
                        print(f"    Found Activation for this layer: {activation_name} (dim={softmax_dim})")
                    else:
                        activation_module = act_module_class()
                        print(f"    Found Activation for this layer: {activation_name}")
                else:
                    raise NeuroTypeError(f"Unknown activation function: '{activation_name}' specified for {layer_info}. Supported: {list(ACTIVATION_REGISTRY.keys())}")
            
            # --- Layer Implementations ---
            if layer_type == 'Dense':
                units = layer_params.get('units')
                if units is None:
                    raise NeuroTypeError(f"Missing required parameter 'units' for {layer_info}")
                
                # If Dense follows Conv, add Flatten unless explicitly told not to
                # This requires tracking the output shape of the previous layer, which is complex
                # Using LazyLinear avoids needing Flatten explicitly for now.
                layer_module = nn.LazyLinear(out_features=int(units))
                layers[f'{layer_name_base}_dense'] = layer_module
                print(f"    Added Dense Layer: units={units}")
                current_in_channels = None # Reset channel tracking after Dense
                
            elif layer_type == 'Conv2D':
                # Required params
                out_channels = layer_params.get('out_channels')
                kernel_size_raw = layer_params.get('kernel_size')
                if out_channels is None or kernel_size_raw is None:
                     raise NeuroTypeError(f"Missing required parameter 'out_channels' or 'kernel_size' for {layer_info}")
                
                out_channels = int(out_channels)
                kernel_size = _parse_int_tuple(kernel_size_raw, 'kernel_size', layer_info)
                
                # Optional params
                stride_raw = layer_params.get('stride', 1)
                padding_raw = layer_params.get('padding', 0)
                in_channels_param = layer_params.get('in_channels')

                stride = _parse_int_tuple(stride_raw, 'stride', layer_info)
                padding = _parse_int_tuple(padding_raw, 'padding', layer_info)

                # Determine input channels
                if in_channels_param is not None:
                    in_channels = int(in_channels_param)
                    if current_in_channels is not None and current_in_channels != in_channels and not is_first_conv:
                         print(f"Warning: Explicit 'in_channels={in_channels}' provided for {layer_info}, overriding inferred value {current_in_channels}")
                elif current_in_channels is not None:
                    in_channels = current_in_channels
                    print(f"    Inferring in_channels={in_channels} from previous layer output for {layer_info}")
                else:
                    # Cannot infer in_channels for the first Conv layer if not provided
                    # For now, raise error. Could use LazyConv2d but want explicit channels.
                     raise NeuroTypeError(f"Missing required parameter 'in_channels' for the first Conv2D layer ({layer_info}) if not inferrable.")

                layer_module = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding
                )
                layers[f'{layer_name_base}_conv2d'] = layer_module
                print(f"    Added Conv2D Layer: in={in_channels}, out={out_channels}, kernel={kernel_size}, stride={stride}, padding={padding}")
                current_in_channels = out_channels # Update for next Conv layer
                is_first_conv = False
                 
            elif layer_type == 'Dropout':
                 rate = layer_params.get('rate')
                 if rate is None:
                     raise NeuroTypeError(f"Missing required parameter 'rate' for {layer_info}")
                 layer_module = nn.Dropout(p=float(rate))
                 layers[f'{layer_name_base}_dropout'] = layer_module
                 print(f"    Added Dropout Layer: rate={rate}")
                 # Dropout doesn't change channel count
                 
            else:
                raise NeuroTypeError(f"Unknown or unsupported layer type: {layer_type} ({layer_info})")

            # Add activation function AFTER the main layer module if it was found
            if activation_module:
                 layers[f'{layer_name_base}_activation'] = activation_module
                 print(f"    Added Activation Module: {activation_name}")

        self.network = nn.Sequential(layers)
        print(f"  Finished building network sequence: {self.network}")

    def forward(self, x):
        """Defines the forward pass of the network."""
        # Need to handle potential shape mismatches, e.g., Conv2D output to Dense input
        # For now, assume input shape is correct for the first layer. 
        # LazyLinear helps after Conv layers. Manual Flatten might be needed otherwise.
        return self.network(x)

    def train_model(self, X_train, y_train, loss_fn_name='bce', optimizer_name='adam', epochs=10, learning_rate=0.001, batch_size=32):
        """
        Basic training loop for the neural network.
        Now supports mse loss and sgd optimizer.

        Args:
            X_train (torch.Tensor): Training features.
            y_train (torch.Tensor): Training labels.
            loss_fn_name (str): Name of the loss function ('bce', 'mse').
            optimizer_name (str): Name of the optimizer ('adam', 'sgd').
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for the optimizer.
            batch_size (int): Size of mini-batches for training.

        Returns:
            list[float]: A list containing the average loss for each epoch.
        """
        print(f"\nStarting training for {epochs} epochs...")
        history = [] # Initialize history list
        
        # --- Select Loss Function ---
        loss_fn_name_lower = loss_fn_name.lower()
        if loss_fn_name_lower == 'bce':
            loss_fn = BCELoss() 
        elif loss_fn_name_lower == 'mse':
            loss_fn = nn.MSELoss() # Use torch MSELoss
        elif loss_fn_name_lower == 'crossentropy': # Added CrossEntropy
             loss_fn = nn.CrossEntropyLoss() # Use torch CrossEntropyLoss
        else:
            # Updated error message to include crossentropy
            raise ValueError(f"Unsupported loss function: {loss_fn_name}. Supported: ['bce', 'mse', 'crossentropy']")
            
        # --- Select Optimizer ---
        optimizer_name_lower = optimizer_name.lower()
        if optimizer_name_lower == 'adam':
            optimizer = Adam(self.parameters(), lr=learning_rate) 
        elif optimizer_name_lower == 'sgd':
             optimizer = optim.SGD(self.parameters(), lr=learning_rate) # Use torch SGD
        elif optimizer_name_lower == 'rmsprop': # Added RMSprop
             optimizer = optim.RMSprop(self.parameters(), lr=learning_rate) # Use torch RMSprop
        else:
            # Updated error message to include rmsprop
            raise ValueError(f"Unsupported optimizer: {optimizer_name}. Supported: ['adam', 'sgd', 'rmsprop']")

        # Ensure model is in training mode
        self.train()

        num_samples = X_train.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size

        for epoch in range(epochs):
            epoch_loss = 0.0
            permutation = torch.randperm(num_samples)
            
            for i in range(0, num_samples, batch_size):
                indices = permutation[i:i + batch_size]
                batch_X, batch_y = X_train[indices], y_train[indices]
                
                optimizer.zero_grad()
                outputs = self(batch_X)
                loss = loss_fn(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.detach().item()

            avg_epoch_loss = epoch_loss / num_batches
            history.append(avg_epoch_loss) # Store average epoch loss
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.4f}")
            
        print("Training finished.")
        return history # Return the recorded history

    def __str__(self):
        # Provide a string representation including the model structure
        return f"NeuralNetwork(name='{self.model_name}')\n{super().__str__()}" 