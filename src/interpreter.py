"""
NEURO Interpreter Module
Executes NEURO code with enhanced error handling and memory management.
"""

import torch
import torch.nn as nn
import random
from typing import Any, Dict, Optional
from .parser import NeuroParser
from .matrix import NeuroMatrix
from .errors import (
    NeuroError, 
    ValidationError, 
    ConfigurationError,
    get_context
)
from .memory import MemoryManager

class ResidualBlock(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        
    def forward(self, x):
        return x + self.module(x)

class LabelSmoothing(torch.nn.Module):
    def __init__(self, smoothing=0.1, ignore_index=0):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing
        
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            num_classes = pred.size(-1)
            if num_classes > 1:
                true_dist.fill_(self.smoothing / (num_classes - 1))
            else:
                true_dist.fill_(self.smoothing)
            target = target.long()
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            mask = target == self.ignore_index
            true_dist[mask.unsqueeze(1).expand_as(true_dist)] = 0
            
        return torch.mean(torch.sum(-true_dist * pred, dim=-1)[~mask])

class SelfAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=True):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=batch_first
        )
    
    def forward(self, x):
        return self.attention(x, x, x)[0]

class SequenceModel(torch.nn.Module):
    def __init__(self, layers, input_size, output_size, sequence_length):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.input_size = input_size
        self.output_size = output_size
        self.sequence_length = sequence_length
        self.max_grad_norm = None
        self.scheduler = None
        self.has_embedding = any(isinstance(layer, torch.nn.Embedding) for layer in layers)

    def forward(self, x, teacher_forcing_ratio=0.5):
        # Ensure input has correct dimensions [batch_size, seq_len]
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension
        if x.dim() == 3:
            x = x.squeeze(0)  # Remove extra dimension
        
        batch_size = x.size(0)
        outputs = []
        
        # Convert to long tensor if using embedding
        if self.has_embedding:
            x = x.long()
        
        # Process through layers
        hidden = None
        for t in range(self.sequence_length):
            # Get input for current timestep
            if t == 0 or (random.random() < teacher_forcing_ratio and t < x.size(1)):
                decoder_input = x[:, t:t+1]
            else:
                # Use previous prediction
                _, topi = output.max(1)
                decoder_input = topi.unsqueeze(1)
            
            # Forward pass through layers
            output = decoder_input
            for layer in self.layers:
                if isinstance(layer, (torch.nn.LSTM, torch.nn.GRU)):
                    if hidden is None:
                        # Initialize hidden state
                        num_directions = 2 if layer.bidirectional else 1
                        h = torch.zeros(layer.num_layers * num_directions, batch_size, layer.hidden_size)
                        if isinstance(layer, torch.nn.LSTM):
                            c = torch.zeros(layer.num_layers * num_directions, batch_size, layer.hidden_size)
                            hidden = (h, c)
                        else:
                            hidden = h
                    output, hidden = layer(output, hidden)
                else:
                    output = layer(output)
            
            outputs.append(output)
        
        # Stack outputs [batch_size, seq_len, vocab_size]
        outputs = torch.cat(outputs, dim=1)
        return outputs

    def evaluate(self, data, beam_size=1):
        self.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        
        with torch.no_grad():
            for inputs, targets in data:
                # Ensure inputs have batch dimension
                if inputs.dim() == 1:
                    inputs = inputs.unsqueeze(0)
                if targets.dim() == 1:
                    targets = targets.unsqueeze(0)
                if inputs.dim() == 3:
                    inputs = inputs.squeeze(0)
                if targets.dim() == 3:
                    targets = targets.squeeze(0)
                
                # Forward pass with beam search
                if beam_size > 1:
                    outputs = self.beam_search(inputs, beam_size)
                else:
                    outputs = self(inputs, teacher_forcing_ratio=0)
                
                # Calculate loss and accuracy
                loss = criterion(outputs.view(-1, self.output_size), targets.view(-1))
                total_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = outputs.max(2)
                mask = targets != 0  # Don't count padding tokens
                total += mask.sum().item()
                correct += ((predicted == targets) & mask).sum().item()
        
        accuracy = correct / total if total > 0 else 0
        avg_loss = total_loss / len(data)
        
        print(f"Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        return accuracy

    def beam_search(self, input_seq, beam_size):
        batch_size = input_seq.size(0)
        
        # Initialize beam with start tokens
        beams = [(torch.zeros(batch_size, 1, dtype=torch.long), 0.0)]
        
        # Generate sequence
        for t in range(self.sequence_length):
            candidates = []
            
            # Expand each beam
            for sequence, score in beams:
                # Forward pass
                output = self(sequence, teacher_forcing_ratio=0)
                log_probs = output[:, -1].log_softmax(dim=-1)
                
                # Get top k candidates
                values, indices = log_probs.topk(beam_size)
                
                for k in range(beam_size):
                    new_seq = torch.cat([sequence, indices[:, k:k+1]], dim=1)
                    new_score = score + values[:, k].mean().item()
                    candidates.append((new_seq, new_score))
            
            # Select top k beams
            beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]
        
        # Return best sequence
        best_sequence = beams[0][0]
        
        # Convert to one-hot
        outputs = torch.zeros(batch_size, self.sequence_length, self.output_size)
        outputs.scatter_(2, best_sequence.unsqueeze(-1), 1)
        return outputs

    def clip_gradients(self, max_norm):
        self.max_grad_norm = max_norm

    def set_scheduler(self, scheduler_type, optimizer, warmup_steps=None):
        if scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=warmup_steps)
        elif scheduler_type == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=warmup_steps)

class NeuroSequential(torch.nn.Sequential):
    def __init__(self, *args, output_size=None):
        super().__init__(*args)
        self.output_size = output_size
        self.max_grad_norm = None
        self.scheduler = None

    def forward(self, x):
        return super().forward(x)

    def clip_gradients(self, max_norm):
        self.max_grad_norm = max_norm

    def set_scheduler(self, scheduler_type, optimizer, warmup_steps=None):
        if scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=warmup_steps)
        elif scheduler_type == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=warmup_steps)

    def evaluate(self, data, beam_size=1):
        self.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        criterion = torch.nn.BCEWithLogitsLoss()
        
        with torch.no_grad():
            for inputs, targets in data:
                # Forward pass
                outputs = self(inputs)
                if outputs.size(-1) == 1:  # Binary classification
                    outputs = outputs.squeeze(-1)
                
                # Calculate loss
                loss = criterion(outputs, targets.float())
                total_loss += loss.item()
                
                # Calculate accuracy
                predictions = (outputs > 0.5).float()  # Binary threshold at 0.5
                correct += (predictions == targets).sum().item()
                total += targets.size(0)
        
        accuracy = correct / total if total > 0 else 0
        avg_loss = total_loss / len(data)
        
        print(f"Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        return accuracy

class NeuroInterpreter:
    def __init__(self, device: Optional[str] = None):
        self.parser = NeuroParser()
        self.variables: Dict[str, Any] = {}
        self.device = torch.device(device if device else 
                                 "cuda" if torch.cuda.is_available() else "cpu")
        self.memory_manager: Optional[MemoryManager] = None
        self.current_model = None
        self.last_layer_size = None
    
    def interpret(self, source_code: str) -> Any:
        """
        Interprets NEURO source code with enhanced error handling.
        
        Args:
            source_code: The NEURO source code to interpret
            
        Returns:
            The result of the last executed statement
        """
        try:
            ast = self.parser.parse(source_code)
            return self.execute_ast(ast)
        except Exception as e:
            if isinstance(e, NeuroError):
                raise
            raise NeuroError(
                f"Error interpreting code: {str(e)}",
                context=get_context(),
                suggestions=[
                    "Check syntax for errors",
                    "Verify all variables are defined",
                    "Ensure model configuration is valid"
                ]
            ) from e
    
    def execute_ast(self, ast: tuple) -> Any:
        """
        Executes an Abstract Syntax Tree node.
        
        Args:
            ast: The AST node to execute
            
        Returns:
            The result of executing the AST node
        """
        try:
            node_type = ast[0]
            
            if node_type == 'program':
                return self.execute_program(ast[1])
            elif node_type == 'neural_network':
                return self.create_neural_network(ast[1], ast[2], ast[3], ast[4])
            elif node_type == 'method_call':
                return self.execute_method_call(ast[1], ast[2], ast[3])
            elif node_type == 'assignment':
                return self.execute_assignment(ast[1], ast[2])
            elif node_type == 'binop':
                return self.execute_binop(ast[1], ast[2], ast[3])
            elif node_type == 'number':
                return float(ast[1])
            elif node_type == 'string':
                return str(ast[1])
            elif node_type == 'id':
                return self.variables.get(ast[1])
            elif node_type == 'print':
                return self.execute_print(ast[1])
            elif node_type == 'print_formatted':
                return self.execute_print_formatted(ast[1], ast[2])
            elif node_type == 'load_matrix':
                return self.execute_load_matrix(ast[1])
            elif node_type == 'save_model':
                return self.execute_save_model(ast[1], ast[2])
            elif node_type == 'load_model':
                return self.execute_load_model(ast[1])
            elif node_type == 'layer':
                return self.execute_layer(ast[1], ast[2], ast[3], ast[4])
            elif node_type == 'config':
                return self.execute_config(ast[1], ast[2])
            else:
                raise ValueError(f"Unknown node type: {node_type}")
        
        except Exception as e:
            if isinstance(e, NeuroError):
                raise
            raise NeuroError(
                f"Error executing AST node: {str(e)}",
                context=get_context(),
                suggestions=[
                    "Check model architecture",
                    "Verify layer configurations",
                    "Ensure data types are compatible"
                ],
                details={"ast_node": str(ast)}
            ) from e
    
    def create_neural_network(self, name: str, params: list, layers: list, config: Optional[list] = None) -> nn.Module:
        """
        Creates a neural network with validation.
        
        Args:
            name: Network name
            params: Network parameters
            layers: Layer configurations
            config: Optional network configuration
            
        Returns:
            The constructed model
        """
        try:
            # Extract and validate input size
            input_size = int(self._get_param(params, 'input_size'))
            output_size = int(self._get_param(params, 'output_size'))
            
            if input_size <= 0 or output_size <= 0:
                raise ValidationError(
                    "Invalid model dimensions",
                    context=get_context(),
                    suggestions=["Input and output sizes must be positive"],
                    details={
                        "input_size": input_size,
                        "output_size": output_size
                    }
                )
            
            # Check for unreasonable dimensions
            if input_size > 1e5 or output_size > 1e5:
                raise NeuroError(
                    "Model dimensions too large",
                    context=get_context(),
                    suggestions=[
                        "Reduce input/output dimensions",
                        "Consider using a more efficient architecture",
                        "Split the model into smaller components"
                    ],
                    details={
                        "input_size": input_size,
                        "output_size": output_size,
                        "max_allowed": 1e5
                    }
                )
            
            # Initialize layer tracking
            self.last_layer_size = input_size
            
            # Build the model
            model = self._build_model(params, layers)
            
            # Set up memory management
            memory_manager = MemoryManager(
                model,
                self.device,
                max_memory_usage=0.9
            )
            
            # Attach memory manager to model
            model.memory_manager = memory_manager
            self.memory_manager = memory_manager
            
            # Store in variables
            self.variables[name] = model
            self.current_model = model
            
            # Initialize loss function if not set
            if not hasattr(self, 'loss_function'):
                self.loss_function = nn.BCEWithLogitsLoss()
            
            return model
            
        except RuntimeError as e:
            if "memory" in str(e).lower():
                raise MemoryError("Not enough memory to create model")
            raise
    
    def _build_model(self, params: list, layers: Optional[list]) -> nn.Module:
        """
        Builds a neural network model with validation.
        
        Args:
            params: Network parameters
            layers: Layer configurations
            
        Returns:
            The constructed model
        """
        try:
            # Extract parameters
            input_size = self._get_param(params, 'input_size')
            output_size = self._get_param(params, 'output_size')
            
            # Validate parameters
            if input_size <= 0 or output_size <= 0:
                raise ValidationError(
                    "Invalid model dimensions",
                    context=get_context(),
                    suggestions=["Input and output sizes must be positive"],
                    details={"input_size": input_size, "output_size": output_size}
                )
            
            # Create model with custom Sequential class
            model = NeuroSequential()
            model.max_grad_norm = None  # Initialize gradient clipping
            model.scheduler = None      # Initialize scheduler
            
            # Add layers if specified
            if layers:
                for layer in layers:
                    layer_type = layer[1]
                    layer_params = layer[2]
                    model.append(self._create_layer(layer_type, layer_params))
            
            return model
            
        except Exception as e:
            if isinstance(e, NeuroError):
                raise
            raise ConfigurationError(
                "Error building model",
                context=get_context(),
                suggestions=[
                    "Check layer parameters",
                    "Verify layer compatibility",
                    "Ensure sufficient memory"
                ],
                details={"error_location": "model_building"}
            ) from e
    
    def _create_layer(self, layer_type: str, params: list) -> nn.Module:
        """
        Creates a single neural network layer with validation.
        
        Args:
            layer_type: Type of layer to create
            params: Layer parameters
            
        Returns:
            The created layer
        """
        try:
            if layer_type == 'Dense':
                units = int(self._get_param(params, 'units'))
                activation = self._get_param(params, 'activation', 'relu')
                
                # Validate units
                if units <= 0:
                    raise ValidationError(
                        "Number of units must be positive",
                        context=get_context(),
                        suggestions=["Use a positive integer for units"],
                        details={"units": units}
                    )
                
                layer = nn.Sequential(
                    nn.Linear(self._get_last_size(), units),
                    self._get_activation(activation)
                )
                self._update_last_size(units)
                return layer
            
            elif layer_type == 'LSTM':
                hidden_size = int(self._get_param(params, 'hidden_size'))
                num_layers = int(self._get_param(params, 'num_layers', 1))
                bidirectional = bool(self._get_param(params, 'bidirectional', False))
                
                layer = nn.LSTM(
                    input_size=self._get_last_size(),
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    bidirectional=bidirectional,
                    batch_first=True
                )
                self._update_last_size(hidden_size * (2 if bidirectional else 1))
                return layer
            
            elif layer_type == 'GRU':
                hidden_size = int(self._get_param(params, 'hidden_size'))
                num_layers = int(self._get_param(params, 'num_layers', 1))
                bidirectional = bool(self._get_param(params, 'bidirectional', False))
                
                layer = nn.GRU(
                    input_size=self._get_last_size(),
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    bidirectional=bidirectional,
                    batch_first=True
                )
                self._update_last_size(hidden_size * (2 if bidirectional else 1))
                return layer
            
            elif layer_type == 'Attention':
                num_heads = int(self._get_param(params, 'num_heads', 8))
                dropout = float(self._get_param(params, 'dropout', 0.1))
                
                layer = SelfAttention(
                    embed_dim=self._get_last_size(),
                    num_heads=num_heads,
                    dropout=dropout
                )
                return layer  # Size remains unchanged
            
            elif layer_type == 'Embedding':
                vocab_size = int(self._get_param(params, 'vocab_size'))
                embedding_dim = int(self._get_param(params, 'embedding_dim'))
                
                if vocab_size is None or embedding_dim is None:
                    raise ValidationError(
                        "Embedding layer requires vocab_size and embedding_dim parameters",
                        context=get_context(),
                        suggestions=["Add vocab_size and embedding_dim to layer parameters"]
                    )
                
                layer = nn.Embedding(vocab_size, embedding_dim)
                self._update_last_size(embedding_dim)
                return layer
            
            elif layer_type == 'Conv2D':
                in_channels = int(self._get_param(params, 'in_channels'))
                out_channels = int(self._get_param(params, 'out_channels'))
                kernel_size = int(self._get_param(params, 'kernel_size', 3))
                
                if in_channels is None or out_channels is None:
                    raise ValidationError(
                        "Conv2D layer requires in_channels and out_channels parameters",
                        context=get_context(),
                        suggestions=["Add in_channels and out_channels to layer parameters"]
                    )
                
                layer = nn.Conv2d(in_channels, out_channels, kernel_size)
                self._update_last_size(out_channels)
                return layer
            
            elif layer_type == 'MaxPool':
                kernel_size = int(self._get_param(params, 'kernel_size', 2))
                layer = nn.MaxPool2d(kernel_size)
                return layer  # Size changes but depends on input shape
            
            elif layer_type == 'Dropout':
                rate = float(self._get_param(params, 'rate', 0.5))
                return nn.Dropout(rate)  # Size remains unchanged
            
            elif layer_type == 'Flatten':
                return nn.Flatten()  # Size changes but depends on input shape
            
            elif layer_type == 'normalize':
                num_features = int(self._get_param(params, 'features', self._get_last_size()))
                return nn.LayerNorm(num_features)  # Size remains unchanged
            
            else:
                raise ValidationError(
                    f"Unknown layer type: {layer_type}",
                    context=get_context(),
                    suggestions=["Check layer type"]
                )
            
        except Exception as e:
            if isinstance(e, NeuroError):
                raise
            raise ConfigurationError(
                f"Failed to create {layer_type} layer",
                context=get_context(),
                suggestions=[
                    "Check layer parameters",
                    "Verify activation function",
                    "Ensure dimensions match"
                ],
                details={
                    "layer_type": layer_type,
                    "parameters": str(params)
                }
            ) from e
    
    def _get_param(self, params: list, name: str, default: Any = None) -> Any:
        """
        Safely extracts a parameter value.
        
        Args:
            params: List of parameters
            name: Parameter name to find
            default: Default value if not found
            
        Returns:
            The parameter value
        """
        for param in params:
            if param[0] == 'named_param' and param[1] == name:
                return self._evaluate_expression(param[2])
        if default is not None:
            return default
        raise ValidationError(
            f"Required parameter '{name}' not found",
            context=get_context(),
            suggestions=[f"Add {name}=value to parameters"],
            details={"available_params": str(params)}
        )
    
    def _get_activation(self, name: str) -> nn.Module:
        """
        Gets an activation function by name.
        
        Args:
            name: Name of the activation function
            
        Returns:
            The activation module
        """
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'softmax': nn.Softmax(dim=-1),
            'log_softmax': nn.LogSoftmax(dim=-1)
        }
        
        if name not in activations:
            raise ValidationError(
                f"Unknown activation function: {name}",
                context=get_context(),
                suggestions=[
                    f"Available activations: {', '.join(activations.keys())}"
                ]
            )
        
        return activations[name]
    
    def _get_last_size(self) -> int:
        """
        Get the size of the last layer or input size.
        
        Returns:
            The size of the last layer
        """
        if self.last_layer_size is None:
            raise ValidationError(
                "Layer size not initialized",
                context=get_context(),
                suggestions=["Ensure input_size is specified in NeuralNetwork"],
                details={"current_size": None}
            )
        return self.last_layer_size

    def _update_last_size(self, size: int):
        """
        Update the size of the last layer.
        
        Args:
            size: New layer size
        """
        self.last_layer_size = size

    def execute_program(self, statements):
        result = None
        for statement in statements:
            result = self.execute_ast(statement)
        return result

    def execute_method_call(self, obj_expr, method_name, params):
        """Execute a method call on an object."""
        obj = self.execute_ast(obj_expr)
        if obj is None:
            raise ValueError(f"Object not found")
        
        # Convert params list to dictionary
        param_dict = {'positional': []}  # Initialize with positional list
        for param in params:
            if param[0] == 'param':
                # Handle positional parameters
                param_dict['positional'].append(self.execute_ast(param[1]))
            elif param[0] == 'named_param':
                # Handle named parameters
                param_dict[param[1]] = self.execute_ast(param[2])
        
        if method_name == 'train':
            # Get training parameters with defaults
            epochs = int(param_dict.get('epochs', 10))
            learning_rate = float(param_dict.get('learning_rate', 0.001))
            teacher_forcing_ratio = float(param_dict.get('teacher_forcing_ratio', 0.5))
            scheduler = param_dict.get('scheduler', None)
            warmup_steps = int(param_dict.get('warmup_steps', 10))
            data = param_dict['positional'][0] if param_dict['positional'] else None
            
            if not isinstance(data, NeuroMatrix):
                raise ValidationError(
                    "Training data must be a NeuroMatrix object",
                    context=get_context(),
                    suggestions=["Convert your data to NeuroMatrix format"],
                    details={"data_type": type(data).__name__}
                )
            
            # Convert data to PyTorch dataset
            dataset = data.to_torch_dataset()
            
            # Initialize optimizer and scheduler
            optimizer = torch.optim.Adam(obj.parameters(), lr=learning_rate)
            if scheduler == 'cosine':
                obj.set_scheduler('cosine', optimizer, warmup_steps)
            elif scheduler == 'step':
                obj.set_scheduler('step', optimizer, warmup_steps)
            
            # Training loop
            criterion = self.loss_function or torch.nn.BCEWithLogitsLoss()  # Default to BCE for binary classification
            obj.train()
            
            for epoch in range(epochs):
                total_loss = 0
                batches = 0
                
                for inputs, targets in dataset:
                    optimizer.zero_grad()
                    
                    # Forward pass with teacher forcing
                    if isinstance(obj, SequenceModel):
                        outputs = obj(inputs, teacher_forcing_ratio)
                    else:
                        outputs = obj(inputs)
                        if outputs.size(-1) == 1:  # Binary classification
                            outputs = outputs.squeeze(-1)
                            targets = targets.squeeze(-1)  # Match target shape
                    
                    # Calculate loss
                    loss = criterion(outputs, targets.float())  # Convert targets to float for BCE
                    
                    # Backward pass with gradient clipping
                    loss.backward()
                    if obj.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(obj.parameters(), obj.max_grad_norm)
                    
                    optimizer.step()
                    if obj.scheduler is not None:
                        obj.scheduler.step()
                    
                    total_loss += loss.item()
                    batches += 1
                
                avg_loss = total_loss / batches
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
            return obj
        
        elif method_name == 'evaluate':
            data = param_dict['positional'][0] if 'positional' in param_dict else None
            beam_size = int(param_dict.get('beam_size', 1))
            
            if not isinstance(data, NeuroMatrix):
                raise ValueError("Evaluation data must be a NeuroMatrix object")
            
            dataset = data.to_torch_dataset()
            return obj.evaluate(dataset, beam_size)
        
        elif method_name == 'clip_gradients':
            max_norm = float(param_dict.get('max_norm', 1.0))
            obj.clip_gradients(max_norm)
            return obj
        
        else:
            raise ValueError(f"Unknown method: {method_name}")

    def execute_assignment(self, var_name, value):
        result = self.execute_ast(value)
        self.variables[var_name] = result
        return result

    def execute_binop(self, op, left, right):
        left_val = self.execute_ast(left)
        right_val = self.execute_ast(right)
        
        if op == 'PLUS':
            return left_val + right_val
        elif op == 'MINUS':
            return left_val - right_val
        elif op == 'TIMES':
            return left_val * right_val
        elif op == 'DIVIDE':
            return left_val / right_val
        else:
            raise ValueError(f"Unknown operator: {op}")

    def execute_print(self, expr):
        value = self.execute_ast(expr)
        print(value)
        return value

    def execute_print_formatted(self, format_str, expr):
        value = self.execute_ast(expr)
        print(format_str, value)
        return value

    def execute_load_matrix(self, filename):
        matrix = NeuroMatrix.load(filename)
        # Store the current matrix for model configuration updates
        self.current_matrix = matrix
        return matrix

    def execute_save_model(self, model_name, filename):
        model = self.variables.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found")
        torch.save(model, filename)  # Save the entire model, not just state_dict
        return None

    def execute_load_model(self, filename):
        model = torch.load(filename)  # Load the entire model
        model.eval()  # Set to evaluation mode
        return model

    def execute_config(self, config_type, params):
        """Execute configuration statements (loss and optimizer)"""
        if config_type == 'loss':
            self.loss_function = self.get_loss_function(params)
        elif config_type == 'optimizer':
            self.optimizer_config = params
        return None

    def _evaluate_expression(self, expr: tuple) -> Any:
        """
        Evaluates an expression tuple.
        
        Args:
            expr: Expression tuple to evaluate
            
        Returns:
            The evaluated expression result
        """
        try:
            expr_type = expr[0]
            
            if expr_type == 'number':
                return float(expr[1])
            elif expr_type == 'string':
                return str(expr[1])
            elif expr_type == 'id':
                # Handle boolean literals
                if expr[1].lower() == 'true':
                    return True
                elif expr[1].lower() == 'false':
                    return False
                
                if expr[1] not in self.variables:
                    raise ValidationError(
                        f"Undefined variable: {expr[1]}",
                        context=get_context(),
                        suggestions=["Check variable name", "Ensure variable is defined before use"],
                        details={"variable_name": expr[1]}
                    )
                return self.variables[expr[1]]
            elif expr_type == 'binop':
                left = self._evaluate_expression(expr[2])
                right = self._evaluate_expression(expr[3])
                
                if expr[1] == '+':
                    return left + right
                elif expr[1] == '-':
                    return left - right
                elif expr[1] == '*':
                    return left * right
                elif expr[1] == '/':
                    if right == 0:
                        raise ValidationError(
                            "Division by zero",
                            context=get_context(),
                            suggestions=["Check denominator value"],
                            details={"operator": "/"}
                        )
                    return left / right
                elif expr[1] == '==':
                    return left == right
                elif expr[1] == '!=':
                    return left != right
                elif expr[1] == '<':
                    return left < right
                elif expr[1] == '>':
                    return left > right
                elif expr[1] == '<=':
                    return left <= right
                elif expr[1] == '>=':
                    return left >= right
                elif expr[1] == 'and':
                    return left and right
                elif expr[1] == 'or':
                    return left or right
            else:
                raise ValidationError(
                    f"Unknown expression type: {expr_type}",
                    context=get_context(),
                    suggestions=["Check expression syntax"],
                    details={"expression_type": expr_type}
                )
            
        except Exception as e:
            if isinstance(e, NeuroError):
                raise
            raise ValidationError(
                f"Error evaluating expression: {str(e)}",
                context=get_context(),
                suggestions=["Check expression syntax", "Verify data types"],
                details={"expression": str(expr)}
            ) from e

    def get_loss_function(self, params: list) -> nn.Module:
        """
        Get a loss function by name.
        
        Args:
            params: Loss function parameters
            
        Returns:
            The loss function module
        """
        loss_type = self._get_param(params, 'type')
        
        loss_functions = {
            'mse': nn.MSELoss(),
            'mae': nn.L1Loss(),
            'bce': nn.BCELoss(),
            'bce_with_logits': nn.BCEWithLogitsLoss(),
            'cross_entropy': nn.CrossEntropyLoss(),
            'nll': nn.NLLLoss(),
            'kl_div': nn.KLDivLoss(),
            'smooth_l1': nn.SmoothL1Loss()
        }
        
        if loss_type not in loss_functions:
            raise ValidationError(
                f"Unknown loss function: {loss_type}",
                context=get_context(),
                suggestions=[
                    f"Available loss functions: {', '.join(loss_functions.keys())}"
                ]
            )
        
        return loss_functions[loss_type]

    def get_optimizer(self, params: list) -> torch.optim.Optimizer:
        """
        Get an optimizer by name.
        
        Args:
            params: Optimizer parameters
            
        Returns:
            The optimizer instance
        """
        if self.current_model is None:
            raise ValidationError(
                "No model available for optimization",
                context=get_context(),
                suggestions=["Create a model before configuring optimizer"]
            )
        
        optimizer_type = self._get_param(params, 'type')
        learning_rate = float(self._get_param(params, 'learning_rate', 0.001))
        
        optimizers = {
            'sgd': lambda: torch.optim.SGD(
                self.current_model.parameters(),
                lr=learning_rate
            ),
            'adam': lambda: torch.optim.Adam(
                self.current_model.parameters(),
                lr=learning_rate
            ),
            'adamw': lambda: torch.optim.AdamW(
                self.current_model.parameters(),
                lr=learning_rate
            ),
            'rmsprop': lambda: torch.optim.RMSprop(
                self.current_model.parameters(),
                lr=learning_rate
            ),
            'adagrad': lambda: torch.optim.Adagrad(
                self.current_model.parameters(),
                lr=learning_rate
            )
        }
        
        if optimizer_type not in optimizers:
            raise ValidationError(
                f"Unknown optimizer: {optimizer_type}",
                context=get_context(),
                suggestions=[
                    f"Available optimizers: {', '.join(optimizers.keys())}"
                ]
            )
        
        return optimizers[optimizer_type]()

# Example usage
if __name__ == '__main__':
    # Test input
    test_input = '''
    model = NeuralNetwork(input_size=128, output_size=10)
    model.train(data, epochs=10)
    accuracy = model.evaluate(test_data)
    '''
    
    # Create interpreter and run code
    interpreter = NeuroInterpreter()
    result = interpreter.interpret(test_input)
    print("Execution result:", result) 