import torch
import torch.nn as nn
import torch.optim as optim # Import optim
from collections import OrderedDict
import yaml # For potentially saving/loading metadata alongside model state
import os # For path manipulation in save/load
from src.errors import NeuroTypeError, NeuroSyntaxError # Import the custom error type
from .losses import BCELoss
from .optimizers import Adam

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
        last_layer_dim = None # Track output dimension of the last layer (e.g., 1 for Dense, 2 for Conv)

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
                last_layer_dim = 1 # Dense output is typically 1D (feature vector)
                
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
                last_layer_dim = 2 # Conv2D output is 2D (feature map)
                
            elif layer_type == 'Dropout':
                 rate = layer_params.get('rate')
                 if rate is None:
                     raise NeuroTypeError(f"Missing required parameter 'rate' for {layer_info}")
                 layer_module = nn.Dropout(p=float(rate))
                 layers[f'{layer_name_base}_dropout'] = layer_module
                 print(f"    Added Dropout Layer: rate={rate}")
                 # Dropout doesn't change channel count or dimensionality
                 
            elif layer_type == 'BatchNorm':
                momentum = float(layer_params.get('momentum', 0.1))
                eps = float(layer_params.get('eps', 1e-5))
                # Determine dimension based on previous layer or explicit param
                explicit_dim = layer_params.get('dim')
                target_dim = None
                if explicit_dim is not None:
                    try:
                        target_dim = int(explicit_dim)
                        if target_dim not in [1, 2]: # Assuming 1D and 2D for now
                           raise ValueError()
                    except ValueError:
                         raise NeuroTypeError(f"Invalid 'dim' parameter '{explicit_dim}' for {layer_info}. Must be 1 or 2.")
                elif last_layer_dim is not None:
                    target_dim = last_layer_dim
                    print(f"    Inferring BatchNorm dim={target_dim} based on previous layer output for {layer_info}")
                else:
                    # Cannot infer if it's the first layer or previous dim unknown
                     raise NeuroTypeError(f"Cannot determine dimension for {layer_info}. Add 'dim=1' or 'dim=2' parameter or ensure it follows a layer with known output dimension.")
               
                # Get the number of features/channels from the PREVIOUS layer's output
                # This is tricky without knowing the exact shapes. Using LazyBatchNorm might be an option,
                # but for now, we'll require the features based on the previous layer definition.
                if target_dim == 1:
                    # Expecting Dense layer before. Need output units of previous Dense.
                    # This requires looking back at the previous layer module, which is complex here.
                    # Using LazyBatchNorm1d simplifies this. 
                    layer_module = nn.LazyBatchNorm1d(eps=eps, momentum=momentum)
                    print(f"    Added LazyBatchNorm1d Layer: momentum={momentum}, eps={eps}")
                elif target_dim == 2:
                    # Expecting Conv2D layer before. Need out_channels.
                    if current_in_channels is None: # Use current_in_channels as it holds the *output* channels of the last Conv layer
                         raise NeuroTypeError(f"Cannot determine number of features for BatchNorm2d in {layer_info}. Ensure it follows a Conv2D layer.")
                    num_features = current_in_channels
                    layer_module = nn.BatchNorm2d(num_features=num_features, eps=eps, momentum=momentum)
                    print(f"    Added BatchNorm2d Layer: features={num_features}, momentum={momentum}, eps={eps}")
                else:
                     raise NeuroTypeError(f"Unsupported BatchNorm dimension {target_dim} inferred for {layer_info}.") # Should not happen due to earlier checks
                    
                layers[f'{layer_name_base}_batchnorm'] = layer_module
                # BatchNorm doesn't typically change dimensionality
                
            elif layer_type == 'Flatten':
                start_dim = int(layer_params.get('start_dim', 1))
                end_dim = int(layer_params.get('end_dim', -1))
                layer_module = nn.Flatten(start_dim=start_dim, end_dim=end_dim)
                layers[f'{layer_name_base}_flatten'] = layer_module
                print(f"    Added Flatten Layer: start_dim={start_dim}, end_dim={end_dim}")
                current_in_channels = None # Flatten makes channel tracking irrelevant
                last_layer_dim = 1 # Flatten output is 1D
                
            elif layer_type == 'MaxPool2D':
                 kernel_size_raw = layer_params.get('kernel_size')
                 if kernel_size_raw is None:
                     raise NeuroTypeError(f"Missing required parameter 'kernel_size' for {layer_info}")
                 kernel_size = _parse_int_tuple(kernel_size_raw, 'kernel_size', layer_info)
                 
                 # Default stride is kernel_size in PyTorch MaxPool2d
                 stride_raw = layer_params.get('stride', kernel_size) 
                 padding_raw = layer_params.get('padding', 0)
                 
                 stride = _parse_int_tuple(stride_raw, 'stride', layer_info)
                 padding = _parse_int_tuple(padding_raw, 'padding', layer_info)
                 
                 layer_module = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
                 layers[f'{layer_name_base}_maxpool2d'] = layer_module
                 print(f"    Added MaxPool2D Layer: kernel={kernel_size}, stride={stride}, padding={padding}")
                 # Pooling preserves channel count but changes spatial dimensions
                 last_layer_dim = 2 # Output remains 2D spatially

            elif layer_type == 'AvgPool2D':
                 kernel_size_raw = layer_params.get('kernel_size')
                 if kernel_size_raw is None:
                     raise NeuroTypeError(f"Missing required parameter 'kernel_size' for {layer_info}")
                 kernel_size = _parse_int_tuple(kernel_size_raw, 'kernel_size', layer_info)
                 
                 # Default stride is kernel_size in PyTorch AvgPool2d
                 stride_raw = layer_params.get('stride', kernel_size)
                 padding_raw = layer_params.get('padding', 0)
                 
                 stride = _parse_int_tuple(stride_raw, 'stride', layer_info)
                 padding = _parse_int_tuple(padding_raw, 'padding', layer_info)
                 
                 layer_module = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
                 layers[f'{layer_name_base}_avgpool2d'] = layer_module
                 print(f"    Added AvgPool2D Layer: kernel={kernel_size}, stride={stride}, padding={padding}")
                 # Pooling preserves channel count but changes spatial dimensions
                 last_layer_dim = 2 # Output remains 2D spatially

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

    def train_model(self, X_train, y_train, X_val=None, y_val=None, 
                    loss_fn_name='bce', optimizer_name='adam', 
                    epochs=10, learning_rate=0.001, batch_size=32):
        """
        Basic training loop for the neural network.
        Supports multiple losses/optimizers and optional validation data.

        Args:
            X_train (torch.Tensor): Training features.
            y_train (torch.Tensor): Training labels.
            X_val (torch.Tensor, optional): Validation features. Defaults to None.
            y_val (torch.Tensor, optional): Validation labels. Defaults to None.
            loss_fn_name (str): Name of the loss function ('bce', 'mse', 'crossentropy').
            optimizer_name (str): Name of the optimizer ('adam', 'sgd', 'rmsprop').
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for the optimizer.
            batch_size (int): Size of mini-batches for training.

        Returns:
            list[float]: A list containing the average loss for each epoch.
            dict: A dictionary containing training history (e.g., {'loss': [...], 'val_loss': [...]}).
        """
        print(f"\nStarting training for {epochs} epochs...")
        # Initialize history list
        history = {'loss': [], 'val_loss': []} # Use a dict for multiple metrics
        
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

        # Ensure model is in training mode initially
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
            # history.append(avg_epoch_loss) # Store average epoch loss
            history['loss'].append(avg_epoch_loss)
            
            # --- Validation Phase (if validation data provided) ---
            epoch_val_loss = None
            if X_val is not None and y_val is not None:
                self.eval() # Set model to evaluation mode
                num_val_samples = X_val.shape[0]
                num_val_batches = (num_val_samples + batch_size - 1) // batch_size
                val_loss_total = 0.0
                
                with torch.no_grad(): # Disable gradient calculations
                    for i in range(0, num_val_samples, batch_size):
                        # Note: Using full batches for validation is common, but mini-batching works too
                        # Using mini-batching here for consistency and memory efficiency if val set is large.
                        batch_X_val = X_val[i:i + batch_size]
                        batch_y_val = y_val[i:i + batch_size]
                        
                        val_outputs = self(batch_X_val)
                        val_loss = loss_fn(val_outputs, batch_y_val)
                        val_loss_total += val_loss.detach().item()
                        
                epoch_val_loss = val_loss_total / num_val_batches
                history['val_loss'].append(epoch_val_loss)
                self.train() # Set model back to training mode
            else:
                 history['val_loss'].append(None) # Append None if no validation this epoch

            # Print epoch summary
            epoch_summary = f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_epoch_loss:.4f}"
            if epoch_val_loss is not None:
                epoch_summary += f" - Val Loss: {epoch_val_loss:.4f}"
            print(epoch_summary)

        print(f"--- Finished training ---")
        return history # Return the recorded history dictionary

    def save(self, filepath):
        """Saves the model's state_dict and architecture info to a file.

        Args:
            filepath (str): Path to save the model file (e.g., 'my_model.nrmodel').
                          The directory will be created if it doesn't exist.
        """
        print(f"Saving model '{self.model_name}' to {filepath}...")
        save_dir = os.path.dirname(filepath)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
        # We need to save more than just state_dict to reconstruct the exact architecture,
        # especially with Lazy layers. Saving the layer definitions (type & params)
        # is crucial. 
        # For now, we save state_dict and essential info.
        # A more robust solution would serialize the layer structure used in __init__.
        save_data = {
            'model_name': self.model_name,
            'input_size': self.input_size,
            'output_size': self.output_size,
            # TODO: Persist the layer_nodes structure used during __init__ for robust loading
            # 'layer_nodes': self.layer_nodes, # Assuming layer_nodes was stored on self
            'model_state_dict': self.state_dict(),
            '__version__': '0.1.0' # Add a version for future compatibility
        }
        try:
            torch.save(save_data, filepath)
            print(f"Model saved successfully to {filepath}")
        except Exception as e:
            print(f"Error saving model to {filepath}: {e}")
            # Potentially raise a custom NeuroSaveError

    @staticmethod
    def load(filepath):
        """Loads a model from a file.

        Args:
            filepath (str): Path to the saved model file.

        Returns:
            NeuralNetwork: The loaded model instance.
        
        Raises:
            FileNotFoundError: If the filepath doesn't exist.
            KeyError: If the saved file is missing required keys.
            Exception: For other loading errors.
        """
        print(f"Loading model from {filepath}...")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        try:
            # Load onto CPU first, can be moved to GPU later if needed
            checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
            
            # Basic validation
            required_keys = ['model_name', 'model_state_dict'] # Add more as save format evolves
            if not all(key in checkpoint for key in required_keys):
                raise KeyError(f"Saved model file '{filepath}' is missing required keys. Found: {list(checkpoint.keys())}")

            # --- Reconstructing the Model --- 
            # This is the complex part. Without the original layer definitions 
            # (like the layer_nodes list passed to __init__), we cannot perfectly 
            # reconstruct the model, especially with Lazy modules. 
            # 
            # WORKAROUND (SIMPLE): Assume the user has defined an identical 
            # NeuralNetwork structure *before* calling load, and we just load the weights.
            # This is common in PyTorch workflows but less ideal for a standalone language.
            # 
            # A BETTER APPROACH (REQUIRES saving layer_nodes in .save()):
            # 1. Get layer_nodes from checkpoint['layer_nodes']
            # 2. Create a new NeuralNetwork instance using these layer_nodes:
            #    model = NeuralNetwork(name=checkpoint['model_name'], 
            #                        params={'input_size': checkpoint.get('input_size'), ...}, 
            #                        layer_nodes=checkpoint['layer_nodes'])
            # 3. Load the state dict: model.load_state_dict(checkpoint['model_state_dict'])
            #
            # For now, we cannot implement the better approach as layer_nodes isn't saved.
            # We will raise an error indicating this limitation.
            
            # print(f"Warning: Model loading currently assumes the model architecture")
            # print(f"         is defined *before* calling load. Only weights are loaded.")
            # print(f"         Reconstructing architecture from file is not yet supported.")
            
            # Placeholder for demonstration - THIS WILL LIKELY FAIL without prior model definition
            # model_name = checkpoint['model_name']
            # print(f"Loaded model name: {model_name}")
            # model = NeuralNetwork(name=model_name, params={}, layer_nodes=[]) # Dummy creation
            # model.load_state_dict(checkpoint['model_state_dict'])
            
            # Raise an informative error until reconstruction is implemented
            raise NotImplementedError("Model reconstruction from file is not yet implemented. " 
                                    "Define the model architecture in your script first, then load the state_dict into it.")
            
            # Once reconstruction works, return the model:
            # model.eval() # Set to evaluation mode after loading
            # print(f"Model '{model.model_name}' loaded successfully.")
            # return model
            
        except FileNotFoundError: # Already handled but good practice
            raise
        except Exception as e:
            print(f"Error loading model from {filepath}: {e}")
            # Potentially raise a custom NeuroLoadError
            raise # Re-raise the exception

    def __str__(self):
        # Provide a string representation including the model structure
        return f"NeuralNetwork(name='{self.model_name}')\n{super().__str__()}" 