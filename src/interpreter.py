"""
NEURO Interpreter Module
Executes NEURO code with enhanced error handling and memory management.
"""

import torch
import torch.nn as nn
import random
from typing import Any, Dict, Optional, List, Callable
from .parser import NeuroParser
from .matrix import NeuroMatrix
from .errors import (
    NeuroError, 
    ValidationError, 
    ConfigurationError,
    get_context
)
from .memory import MemoryManager
import torchvision.models as models

class CustomLayer:
    """Decorator for custom layer definitions."""
    def __init__(self, func: Callable):
        self.func = func
        self.name = func.__name__
        
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

class PretrainedBackbone(nn.Module):
    """Wrapper for pretrained models used as backbones."""
    def __init__(self, model_name: str, trainable: bool = False):
        super().__init__()
        try:
            # Handle model creation with weights
            if model_name.lower() == "resnet18":
                self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                # Get the weights enum class for other models
                weights_enum = getattr(models, f"{model_name}_Weights", None)
                if weights_enum is None:
                    raise ConfigurationError(
                        f"No weights found for model {model_name}",
                        context=get_context(),
                        suggestions=[
                            "Check available models in torchvision.models",
                            "Use a valid model name"
                        ]
                    )
                self.model = getattr(models, model_name)(weights=weights_enum.DEFAULT)
            
            # Remove the final classification layer
            if hasattr(self.model, 'fc'):
                self.model.fc = nn.Identity()
            elif hasattr(self.model, 'classifier'):
                self.model.classifier = nn.Identity()
                
            if not trainable:
                for param in self.model.parameters():
                    param.requires_grad = False
                    
            # Move model to the same device as the input
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(device)
            
        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(
                f"Error initializing pretrained model {model_name}: {str(e)}",
                context=get_context(),
                suggestions=[
                    "Check available models in torchvision.models",
                    "Use a valid model name",
                    "Ensure you have internet connection for downloading weights"
                ]
            )
    
    def forward(self, x):
        # Ensure input is on the same device as model
        x = x.to(next(self.model.parameters()).device)
        # Forward pass through the backbone
        features = self.model(x)
        # Add spatial dimensions if they were removed
        if features.dim() == 2:
            features = features.unsqueeze(-1).unsqueeze(-1)
        return features

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

class BranchingModule(nn.Module):
    """Module that supports multiple branches and concatenation."""
    def __init__(self):
        super().__init__()
        self.branches = nn.ModuleDict()
        self.merge_ops = {}
        self.current_branch = None
    
    def add_branch(self, name: str) -> None:
        """Add a new branch."""
        self.branches[name] = nn.ModuleList()
        self.current_branch = name
    
    def add_layer(self, layer: nn.Module) -> None:
        """Add a layer to current branch."""
        if self.current_branch is None:
            raise ValidationError(
                "No active branch",
                context=get_context(),
                suggestions=["Create a branch before adding layers"]
            )
        self.branches[self.current_branch].append(layer)
    
    def set_merge(self, branches: List[str], operation: str = 'concat') -> None:
        """Set how branches should be merged."""
        if operation not in ['concat', 'add', 'multiply']:
            raise ValidationError(
                f"Unknown merge operation: {operation}",
                context=get_context(),
                suggestions=["Use 'concat', 'add', or 'multiply'"]
            )
        self.merge_ops['final'] = (branches, operation)
    
    def forward(self, x):
        """Forward pass through branches and merge."""
        branch_outputs = {}
        
        # Process each branch
        for name, layers in self.branches.items():
            branch_x = x
            for layer in layers:
                branch_x = layer(branch_x)
            branch_outputs[name] = branch_x
        
        # Merge branches
        if 'final' in self.merge_ops:
            branches, operation = self.merge_ops['final']
            outputs = [branch_outputs[b] for b in branches]
            
            if operation == 'concat':
                return torch.cat(outputs, dim=1)
            elif operation == 'add':
                return sum(outputs)
            else:  # multiply
                result = outputs[0]
                for o in outputs[1:]:
                    result = result * o
                return result
        
        # If no merge operation, return last branch output
        return list(branch_outputs.values())[-1]

class TrainingCallback:
    """Base class for training callbacks."""
    def on_training_begin(self, model, data):
        pass
    
    def on_epoch_begin(self, epoch, logs=None):
        pass
    
    def on_batch_begin(self, batch, logs=None):
        pass
    
    def on_batch_end(self, batch, logs=None):
        pass
    
    def on_epoch_end(self, epoch, logs=None):
        pass
    
    def on_training_end(self, logs=None):
        pass

class EarlyStopping(TrainingCallback):
    """Early stopping callback."""
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('loss')
        if current_loss is None:
            return
        
        if self.best_loss is None:
            self.best_loss = current_loss
        elif current_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = current_loss
            self.counter = 0

class ModelCheckpoint(TrainingCallback):
    """Model checkpoint callback."""
    def __init__(self, filepath, save_best_only=False, monitor='loss'):
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.best = float('inf')
        self.model = None
    
    def on_training_begin(self, model, data):
        self.model = model
    
    def on_epoch_end(self, epoch, logs=None):
        if not self.model:
            return
            
        current = logs.get(self.monitor)
        if current is None:
            return
        
        if self.save_best_only:
            if current < self.best:
                self.best = current
                torch.save(self.model.state_dict(), self.filepath)
        else:
            torch.save(self.model.state_dict(), 
                      self.filepath.format(epoch=epoch, **logs))

class TensorBoardLogger(TrainingCallback):
    """TensorBoard logging callback."""
    def __init__(self, log_dir):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir)
    
    def on_epoch_end(self, epoch, logs=None):
        for name, value in logs.items():
            self.writer.add_scalar(name, value, epoch)
    
    def on_training_end(self, logs=None):
        self.writer.close()

def before_training(func):
    """Decorator for preprocessing data before training."""
    func._is_before_training = True
    return func

def after_epoch(func):
    """Decorator for post-epoch processing."""
    func._is_after_epoch = True
    return func

class NeuroInterpreter:
    def __init__(self, device: Optional[str] = None):
        self.parser = NeuroParser()
        self.variables: Dict[str, Any] = {}
        self.device = torch.device(device if device else 
                                 "cuda" if torch.cuda.is_available() else "cpu")
        self.memory_manager: Optional[MemoryManager] = None
        self.current_model = None
        self.last_layer_size = None
    
    def create_matrix(self, inputs: torch.Tensor, targets: torch.Tensor) -> 'NeuroMatrix':
        """
        Create a NeuroMatrix from input and target tensors.
        
        Args:
            inputs: Input tensor
            targets: Target tensor
            
        Returns:
            NeuroMatrix object
        """
        from .matrix import NeuroMatrix
        return NeuroMatrix.from_tensors(inputs, targets)

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
                # Neural network node should have parameters and layers
                params = ast[1] if len(ast) > 1 else []
                layers = ast[2] if len(ast) > 2 else []
                return self.create_neural_network(params, layers)
            elif node_type == 'method_call':
                return self.execute_method_call(ast[1], ast[2], ast[3])
            elif node_type == 'assignment':
                return self.execute_assignment(ast[1], ast[2])
            elif node_type == 'decorated':
                return self.execute_decorated(ast[1], ast[2])
            elif node_type == 'custom_layer':
                return self.execute_custom_layer(ast[1], ast[2], ast[3])
            elif node_type == 'branch':
                return self.execute_branch(ast[1], ast[2])
            elif node_type == 'binop':
                return self.execute_binop(ast[1], ast[2], ast[3])
            elif node_type == 'unary_op':
                return self.execute_unary_op(ast[1], ast[2])
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
                return self.execute_layer(ast[1], ast[2])
            elif node_type == 'config':
                return self.execute_config(ast[1], ast[2])
            elif node_type == 'list':
                return [self.execute_ast(item) for item in ast[1]]
            elif node_type == 'dict':
                return {k: self.execute_ast(v) for k, v in ast[1]}
            elif node_type == 'return':
                return self.execute_return(ast[1])
            elif node_type == 'for':
                return self.execute_for_loop(ast[1], ast[2], ast[3])
            elif node_type == 'function_call':
                return self.execute_function_call(ast[1], ast[2])
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
    
    def create_neural_network(self, params: list, layers: Optional[list]) -> nn.Module:
        """
        Creates a neural network with validation.
        
        Args:
            params: Network parameters
            layers: Layer configurations
            
        Returns:
            The constructed model
        """
        try:
            # Extract and validate input size
            input_size = None
            output_size = None
            for param in params:
                if param[0] == 'named_param':
                    if param[1] == 'input_size':
                        input_size = self.execute_ast(param[2])
                    elif param[1] == 'output_size':
                        output_size = self.execute_ast(param[2])
            
            if input_size is None or output_size is None:
                raise ValidationError(
                    "Missing required parameters: input_size and output_size",
                    context=get_context(),
                    suggestions=["Specify both input_size and output_size"]
                )
            
            # Convert to integers and validate
            try:
                input_size = int(input_size)
                output_size = int(output_size)
            except (TypeError, ValueError):
                raise ValidationError(
                    "Input and output sizes must be integers",
                    context=get_context(),
                    suggestions=["Use integer values for dimensions"],
                    details={
                        "input_size": input_size,
                        "output_size": output_size
                    }
                )
            
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
            
            # Initialize layer tracking
            self.last_layer_size = input_size
            
            # Build the model
            model = self._build_model(input_size, output_size, layers or [])
            
            # Set up memory management
            memory_manager = MemoryManager(
                model,
                self.device,
                max_memory_usage=0.9
            )
            
            # Attach memory manager to model
            model.memory_manager = memory_manager
            self.memory_manager = memory_manager
            
            return model
            
        except Exception as e:
            if isinstance(e, NeuroError):
                raise
            raise ConfigurationError(
                f"Error creating neural network: {str(e)}",
                context=get_context(),
                suggestions=[
                    "Check parameter values",
                    "Verify layer configurations",
                    "Ensure sufficient memory"
                ]
            ) from e
    
    def _build_model(self, input_size: float, output_size: float, layers: list) -> nn.Module:
        """
        Builds a neural network model with validation.
        
        Args:
            input_size: Size of input layer
            output_size: Size of output layer
            layers: Layer configurations
            
        Returns:
            The constructed model
        """
        try:
            # Create model with custom Sequential class
            model = NeuroSequential()
            model.max_grad_norm = None  # Initialize gradient clipping
            model.scheduler = None      # Initialize scheduler
            
            # Add layers if specified
            if layers:
                for layer in layers:
                    if layer[0] == 'layer':
                        layer_type = layer[1]
                        layer_params = layer[2] if len(layer) > 2 else []
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
                ]
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
                in_features = int(self._get_param(params, 'in_features', self._get_last_size()))
                
                # Validate units
                if units <= 0:
                    raise ValidationError(
                        "Number of units must be positive",
                        context=get_context(),
                        suggestions=["Use a positive integer for units"],
                        details={"units": units}
                    )
                
                layer = nn.Sequential(
                    nn.Linear(in_features, units),
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
            elif param[0] == 'string':
                # Handle string parameters
                param_dict['positional'].append(param[1])
        
        if method_name == 'train':
            # Get training parameters with defaults
            epochs = int(param_dict.get('epochs', 10))
            learning_rate = float(param_dict.get('learning_rate', 0.001))
            teacher_forcing_ratio = float(param_dict.get('teacher_forcing_ratio', 0.5))
            scheduler = param_dict.get('scheduler', None)
            warmup_steps = int(param_dict.get('warmup_steps', 10))
            callbacks = param_dict.get('callbacks', [])
            data = param_dict['positional'][0] if param_dict['positional'] else None
            
            if not isinstance(data, NeuroMatrix):
                raise ValidationError(
                    "Training data must be a NeuroMatrix object",
                    context=get_context(),
                    suggestions=["Convert your data to NeuroMatrix format"],
                    details={"data_type": type(data).__name__}
                )
            
            # Initialize optimizer and scheduler
            optimizer = torch.optim.Adam(obj.parameters(), lr=learning_rate)
            if scheduler == 'cosine':
                obj.set_scheduler('cosine', optimizer, warmup_steps)
            elif scheduler == 'step':
                obj.set_scheduler('step', optimizer, warmup_steps)
            
            # Initialize callbacks
            callback_instances = []
            for callback in callbacks:
                if isinstance(callback, dict):
                    callback_type = callback.pop('type')
                    if callback_type == 'EarlyStopping':
                        callback_instances.append(EarlyStopping(**callback))
                    elif callback_type == 'ModelCheckpoint':
                        callback_instances.append(ModelCheckpoint(**callback))
                    elif callback_type == 'TensorBoardLogger':
                        callback_instances.append(TensorBoardLogger(**callback))
            
            # Training loop with callbacks
            criterion = self.loss_function or torch.nn.BCEWithLogitsLoss()
            obj.train()
            
            # Call before_training hooks
            for attr_name in dir(obj):
                attr = getattr(obj, attr_name)
                if hasattr(attr, '_is_before_training'):
                    data = attr(data)
            
            # Training begin callbacks
            for callback in callback_instances:
                callback.on_training_begin(obj, data)
            
            for epoch in range(epochs):
                total_loss = 0
                batches = 0
                
                # Epoch begin callbacks
                for callback in callback_instances:
                    callback.on_epoch_begin(epoch)
                
                for batch_idx, (inputs, targets) in enumerate(data):
                    # Batch begin callbacks
                    batch_logs = {'batch': batch_idx}
                    for callback in callback_instances:
                        callback.on_batch_begin(batch_idx, batch_logs)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass with teacher forcing
                    if isinstance(obj, SequenceModel):
                        outputs = obj(inputs, teacher_forcing_ratio)
                    else:
                        outputs = obj(inputs)
                        if outputs.size(-1) == 1:
                            outputs = outputs.squeeze(-1)
                            targets = targets.squeeze(-1)
                    
                    # Calculate loss
                    loss = criterion(outputs, targets.float())
                    
                    # Backward pass with gradient clipping
                    loss.backward()
                    if obj.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(obj.parameters(), obj.max_grad_norm)
                    
                    optimizer.step()
                    if obj.scheduler is not None:
                        obj.scheduler.step()
                    
                    total_loss += loss.item()
                    batches += 1
                    
                    # Batch end callbacks
                    batch_logs.update({'loss': loss.item()})
                    for callback in callback_instances:
                        callback.on_batch_end(batch_idx, batch_logs)
                
                avg_loss = total_loss / batches
                
                # Epoch end callbacks
                epoch_logs = {'loss': avg_loss, 'epoch': epoch}
                for callback in callback_instances:
                    callback.on_epoch_end(epoch, epoch_logs)
                
                # Call after_epoch hooks
                for attr_name in dir(obj):
                    attr = getattr(obj, attr_name)
                    if hasattr(attr, '_is_after_epoch'):
                        if attr(obj, data) == 'StopTraining':
                            print(f"Training stopped early at epoch {epoch+1}")
                            break
                
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
            # Training end callbacks
            for callback in callback_instances:
                callback.on_training_end({'final_loss': avg_loss})
            
            return obj
        
        elif method_name == 'evaluate':
            data = param_dict['positional'][0] if param_dict['positional'] else None
            beam_size = int(param_dict.get('beam_size', 1))
            
            if not isinstance(data, NeuroMatrix):
                raise ValueError("Evaluation data must be a NeuroMatrix object")
            
            dataset = data.to_torch_dataset()
            return obj.evaluate(dataset, beam_size)
        
        elif method_name == 'clip_gradients':
            max_norm = float(param_dict.get('max_norm', 1.0))
            obj.clip_gradients(max_norm)
            return obj
        
        elif method_name == 'save':
            # Handle save method
            if not param_dict['positional']:
                raise ValidationError(
                    "Save method requires a filename parameter",
                    context=get_context(),
                    suggestions=["Provide a filename to save the model"]
                )
            filename = param_dict['positional'][0]
            format = param_dict.get('format', 'pt')
            
            try:
                if format == 'torchscript':
                    # Save as TorchScript
                    scripted_model = torch.jit.script(obj)
                    scripted_model.save(filename)
                else:
                    # Save model state dict
                    torch.save(obj.state_dict(), filename)
                return None
            except Exception as e:
                raise ValidationError(
                    f"Error saving model: {str(e)}",
                    context=get_context(),
                    suggestions=[
                        "Check write permissions for the target directory",
                        "Ensure sufficient disk space",
                        "Verify the path is valid"
                    ]
                )
        
        else:
            raise ValueError(f"Unknown method: {method_name}")

    def execute_assignment(self, var_name, value):
        result = self.execute_ast(value)
        self.variables[var_name] = result
        return result

    def execute_binop(self, op: str, left: Any, right: Any) -> Any:
        """Execute a binary operation."""
        # Evaluate operands
        left_val = self.execute_ast(left)
        right_val = self.execute_ast(right)

        # Handle arithmetic operations
        if op in ['+', '-', '*', '/', '%', '**']:
            if isinstance(left_val, (int, float)) and isinstance(right_val, (int, float)):
                if op == '+':
                    return left_val + right_val
                elif op == '-':
                    return left_val - right_val
                elif op == '*':
                    return left_val * right_val
                elif op == '/':
                    if right_val == 0:
                        raise NeuroError(
                            "Division by zero",
                            context=get_context(),
                            suggestions=["Check your arithmetic expressions"]
                        )
                    return left_val / right_val
                elif op == '%':
                    return left_val % right_val
                elif op == '**':
                    return left_val ** right_val
            else:
                raise NeuroError(
                    f"Invalid operands for {op}: {type(left_val)} and {type(right_val)}",
                    context=get_context(),
                    suggestions=[
                        "Ensure operands are numbers",
                        "Check variable types"
                    ]
                )

        # Handle comparison operations
        elif op in ['==', '!=', '<', '>', '<=', '>=']:
            if op == '==':
                return left_val == right_val
            elif op == '!=':
                return left_val != right_val
            elif op == '<':
                return left_val < right_val
            elif op == '>':
                return left_val > right_val
            elif op == '<=':
                return left_val <= right_val
            elif op == '>=':
                return left_val >= right_val

        # Handle logical operations
        elif op in ['and', 'or']:
            if op == 'and':
                return left_val and right_val
            elif op == 'or':
                return left_val or right_val

        raise NeuroError(
            f"Unknown operator: {op}",
            context=get_context(),
            suggestions=[
                "Check model architecture",
                "Verify layer configurations",
                "Ensure data types are compatible"
            ]
        )

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
        """Save model weights safely."""
        try:
            model = self.variables.get(model_name)
            if model is None:
                raise ValidationError(
                    f"Model {model_name} not found",
                    context=get_context(),
                    suggestions=["Check if the model name is correct"]
                )
            
            # Save only the model weights
            torch.save(model.state_dict(), filename)
            return None
        
        except Exception as e:
            raise ValidationError(
                f"Error saving model: {str(e)}",
                context=get_context(),
                suggestions=[
                    "Check write permissions for the target directory",
                    "Ensure sufficient disk space",
                    "Verify the path is valid"
                ]
            )

    def execute_load_model(self, filename):
        """Load a model safely with weights only."""
        try:
            # Load the model architecture first
            model = self.current_model
            if model is None:
                raise ValidationError(
                    "No model architecture defined for loading weights",
                    context=get_context(),
                    suggestions=[
                        "Create a model before loading weights",
                        "Define the model architecture explicitly"
                    ]
                )
            
            # Load weights safely
            state_dict = torch.load(filename, weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()  # Set to evaluation mode
            return model
        
        except Exception as e:
            raise ValidationError(
                f"Error loading model: {str(e)}",
                context=get_context(),
                suggestions=[
                    "Verify the model file exists",
                    "Check if the model architecture matches the saved weights",
                    "Ensure the file contains valid model weights"
                ]
            )

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
            elif expr_type == 'boolean':
                return bool(expr[1])
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

    def execute_decorated(self, decorator: tuple, statement: tuple) -> Any:
        """Execute a decorated statement."""
        decorator_name = decorator[1]
        decorator_args = decorator[2]
        
        if decorator_name == 'custom_layer':
            return self.register_custom_layer(statement)
        elif decorator_name == 'pretrained':
            return self.load_pretrained_model(decorator_args, statement)
        else:
            raise NeuroError(f"Unknown decorator: {decorator_name}")
            
    def execute_custom_layer(self, name: str, input_var: str, args: tuple, body: tuple) -> Any:
        """Execute a custom layer definition."""
        def layer_func(x, *params):
            # Create new scope for the layer
            old_vars = self.variables.copy()
            self.variables[input_var] = x
            if args:
                self.variables[args[0]] = params[0]
                
            result = self.execute_ast(body)
            
            # Restore old scope
            self.variables = old_vars
            return result
            
        self.variables[name] = layer_func
        return layer_func
        
    def execute_branch(self, name: str, body: tuple) -> Any:
        """Execute a model branch definition."""
        return self.create_branch(name, lambda: self.execute_ast(body))

    def execute_unary_op(self, op: str, operand: Any) -> Any:
        """Execute a unary operation."""
        val = self.execute_ast(operand)

        if op == '-':
            if isinstance(val, (int, float)):
                return -val
            else:
                raise NeuroError(
                    f"Invalid operand for unary minus: {type(val)}",
                    context=get_context(),
                    suggestions=["Ensure operand is a number"]
                )
        elif op == 'not':
            return not val

        raise NeuroError(
            f"Unknown unary operator: {op}",
            context=get_context(),
            suggestions=["Check operator syntax"]
        )

    def execute_return(self, value: tuple) -> Any:
        """Execute a return statement."""
        return self.execute_ast(value)

    def execute_for_loop(self, var_name: str, limit: tuple, body: list) -> None:
        """Execute a for loop."""
        limit_val = self.execute_ast(limit)
        for i in range(limit_val):
            self.variables[var_name] = i
            for stmt in body:
                self.execute_ast(stmt)

    def execute_function_call(self, function_name: str, params: list) -> Any:
        """Execute a function call with the given parameters."""
        if function_name == 'loss':
            return self.create_loss(params)
        elif function_name == 'optimizer':
            return self.create_optimizer(params)
        elif function_name == 'load_model':
            return self.execute_load_model(params)
        else:
            raise NeuroError(
                f"Unknown function: {function_name}",
                context=get_context(),
                suggestions=[
                    "Check function name spelling",
                    "Verify function is defined",
                    "Ensure function is imported"
                ]
            )

    def load_pretrained_model(self, model_name: tuple, statement: tuple) -> Any:
        """
        Load a pretrained model.
        
        Args:
            model_name: The name of the pretrained model
            statement: The statement to execute with the pretrained model
            
        Returns:
            The result of executing the statement with the pretrained model
        """
        try:
            model_name = self.execute_ast(model_name)
            
            # Import torchvision for pretrained models
            try:
                import torchvision.models as models
            except ImportError:
                raise ConfigurationError(
                    "Failed to import torchvision",
                    context=get_context(),
                    suggestions=[
                        "Install torchvision",
                        "Check Python environment"
                    ]
                )
            
            # Get the model class
            model_class = getattr(models, model_name.lower(), None)
            if model_class is None:
                raise ConfigurationError(
                    f"Failed to load pretrained model: {model_name}",
                    context=get_context(),
                    suggestions=[
                        "Check model name",
                        "Verify model is available in torchvision"
                    ]
                )
            
            # Load pretrained model
            model = model_class(pretrained=True)
            
            # Execute the statement with the model
            if isinstance(statement, tuple) and statement[0] == 'assignment':
                var_name = statement[1]
                self.variables[var_name] = model
                return model
            
            return model
            
        except Exception as e:
            if isinstance(e, NeuroError):
                raise
            raise ConfigurationError(
                f"Failed to load pretrained model: {str(e)}",
                context=get_context(),
                suggestions=[
                    "Check model name",
                    "Verify model is available",
                    "Check network connection"
                ]
            ) from e

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