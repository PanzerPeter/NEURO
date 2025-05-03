import torch
import torch.nn as nn
import torch.optim as optim # Import optim
from collections import OrderedDict
from src.errors import NeuroTypeError # Import the custom error type
from .losses import BCELoss # Assuming losses.py is in the same directory
# We might need more losses later
# from torch.nn import MSELoss 
from .optimizers import Adam # Assuming optimizers.py is in the same directory
# We might need more optimizers later
# from torch.optim import SGD

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
    'tanh': nn.Tanh,       # Added Tanh
    'softmax': nn.Softmax  # Added Softmax (consider dim parameter)
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
        
        layers = OrderedDict()
        for i, layer_node in enumerate(layer_nodes):
            layer_type = layer_node.layer_type
            layer_params = layer_node.params
            layer_name_base = f"layer_{i}" # Base name for the layer and its activation
            
            print(f"  Processing Layer {i}: Type={layer_type}, Params={layer_params}")
            
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
                    raise NeuroTypeError(f"Unknown activation function: '{activation_name}' specified for layer type {layer_type} (Layer {i}). Supported: {list(ACTIVATION_REGISTRY.keys())}")
            
            # --- Layer Implementations ---
            if layer_type == 'Dense':
                units = layer_params.get('units')
                if units is None:
                    raise NeuroTypeError(f"Missing required parameter 'units' for layer type Dense (Layer {i})")
                
                layer_module = torch.nn.LazyLinear(out_features=int(units))
                layers[f'{layer_name_base}_dense'] = layer_module
                print(f"    Added Dense Layer: units={units}")
                
            elif layer_type == 'Dropout':
                 rate = layer_params.get('rate')
                 if rate is None:
                     raise NeuroTypeError(f"Missing required parameter 'rate' for layer type Dropout (Layer {i})")
                 layer_module = torch.nn.Dropout(p=float(rate))
                 layers[f'{layer_name_base}_dropout'] = layer_module
                 print(f"    Added Dropout Layer: rate={rate}")
                 
            else:
                raise NeuroTypeError(f"Unknown or unsupported layer type: {layer_type} (Layer {i})")

            # Add activation function AFTER the main layer module if it was found
            if activation_module:
                 layers[f'{layer_name_base}_activation'] = activation_module
                 print(f"    Added Activation Module: {activation_name}")

        self.network = nn.Sequential(layers)
        print(f"  Finished building network sequence: {self.network}")

    def forward(self, x):
        """Defines the forward pass of the network."""
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