import torch
import torch.nn as nn
from collections import OrderedDict
from src.errors import NeuroTypeError # Import the custom error type
from .losses import BCELoss # Assuming losses.py is in the same directory
from .optimizers import Adam # Assuming optimizers.py is in the same directory

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

        Args:
            X_train (torch.Tensor): Training features.
            y_train (torch.Tensor): Training labels.
            loss_fn_name (str): Name of the loss function ('bce').
            optimizer_name (str): Name of the optimizer ('adam').
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for the optimizer.
            batch_size (int): Size of mini-batches for training.
        """
        print(f"\nStarting training for {epochs} epochs...")
        
        # --- Select Loss Function ---
        if loss_fn_name.lower() == 'bce':
            loss_fn = BCELoss() # Use the imported class
        else:
            raise ValueError(f"Unsupported loss function: {loss_fn_name}. Supported: ['bce']")
            
        # --- Select Optimizer ---
        if optimizer_name.lower() == 'adam':
            optimizer = Adam(self.parameters(), lr=learning_rate) # Use the imported class
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}. Supported: ['adam']")

        # Ensure model is in training mode
        self.train()

        num_samples = X_train.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size # Calculate number of batches

        for epoch in range(epochs):
            epoch_loss = 0.0
            permutation = torch.randperm(num_samples) # Shuffle data each epoch
            
            for i in range(0, num_samples, batch_size):
                indices = permutation[i:i + batch_size]
                batch_X, batch_y = X_train[indices], y_train[indices]
                
                # 1. Zero gradients
                optimizer.zero_grad()

                # 2. Forward pass
                outputs = self(batch_X) # Call the forward method

                # 3. Calculate loss
                loss = loss_fn(outputs, batch_y)

                # 4. Backward pass
                loss.backward()

                # 5. Optimizer step
                optimizer.step()
                
                # Use .detach() before .item() to avoid warning
                epoch_loss += loss.detach().item()

            avg_epoch_loss = epoch_loss / num_batches
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.4f}")
            
        print("Training finished.")

    def __str__(self):
        # Provide a string representation including the model structure
        return f"NeuralNetwork(name='{self.model_name}')\n{super().__str__()}" 