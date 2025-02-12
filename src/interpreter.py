import torch
import numpy as np
import random
from .parser import NeuroParser
from .matrix import NeuroMatrix

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
            target = target.long()  # Convert target to long dtype
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
        # Use the input as query, key, and value
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
    def __init__(self):
        self.parser = NeuroParser()
        self.variables = {}  # Symbol table for variables
        self.models = {}    # Storage for neural network models
        self.loss_function = None
        self.optimizer_config = None

    def interpret(self, code):
        try:
            ast = self.parser.parse(code)
            if ast is None:
                raise ValueError("Failed to parse the code. Please check the syntax.")
            return self.execute(ast)
        except Exception as e:
            print(f"\nError executing code: {str(e)}")
            print("Traceback:")
            import traceback
            traceback.print_exc()
            return None

    def execute(self, node):
        if isinstance(node, tuple):
            node_type = node[0]
            
            if node_type == 'program':
                return self.execute_program(node[1])
            elif node_type == 'neural_network':
                return self.execute_neural_network(node[1], node[2], node[3], node[4])
            elif node_type == 'method_call':
                return self.execute_method_call(node[1], node[2], node[3])
            elif node_type == 'assignment':
                return self.execute_assignment(node[1], node[2])
            elif node_type == 'binop':
                return self.execute_binop(node[1], node[2], node[3])
            elif node_type == 'number':
                return float(node[1])
            elif node_type == 'string':
                return str(node[1])
            elif node_type == 'id':
                return self.variables.get(node[1])
            elif node_type == 'print':
                return self.execute_print(node[1])
            elif node_type == 'print_formatted':
                return self.execute_print_formatted(node[1], node[2])
            elif node_type == 'load_matrix':
                return self.execute_load_matrix(node[1])
            elif node_type == 'save_model':
                return self.execute_save_model(node[1], node[2])
            elif node_type == 'load_model':
                return self.execute_load_model(node[1])
            elif node_type == 'layer':
                return self.execute_layer(node[1], node[2], node[3], node[4])
            elif node_type == 'config':
                return self.execute_config(node[1], node[2])
            else:
                raise ValueError(f"Unknown node type: {node_type}")
        
        elif isinstance(node, list):
            result = None
            for statement in node:
                result = self.execute(statement)
            return result
        
        elif isinstance(node, (int, float)):
            return float(node)
        
        elif isinstance(node, str):
            return str(node)
        
        else:
            raise ValueError(f"Cannot execute node of type {type(node)}")

    def execute_program(self, statements):
        result = None
        for statement in statements:
            result = self.execute(statement)
        return result

    def execute_neural_network(self, var_name, params, layers=None, config=None):
        # Get network parameters
        input_size = None
        output_size = None
        sequence_length = None
        
        for param in params:
            if param[0] == 'named_param':
                if param[1] == 'input_size':
                    input_size = int(self.execute(param[2]))
                elif param[1] == 'output_size':
                    output_size = int(self.execute(param[2]))
                elif param[1] == 'sequence_length':
                    sequence_length = int(self.execute(param[2]))
        
        if input_size is None or output_size is None:
            raise ValueError("Neural network requires input_size and output_size")

        # Create model layers
        if layers is None:
            # Default architecture if no layers specified
            model = NeuroSequential(
                torch.nn.Linear(input_size, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, output_size),
                output_size=output_size
            )
        else:
            # Check if this is a sequence model (has LSTM, GRU, Attention, or Embedding layers)
            is_sequence_model = any(layer[1] in ['LSTM', 'GRU', 'ATTENTION', 'EMBEDDING'] 
                                  for layer in layers)
            
            # Build layer list
            layer_list = []
            current_size = input_size
            has_embedding = any(layer[1] == 'EMBEDDING' for layer in layers)
            
            for layer in layers:
                layer_module = self.execute_layer(layer[1], layer[2], current_size, sequence_length)
                if isinstance(layer_module, tuple):
                    layer_list.extend(layer_module)
                    current_size = layer_module[0].out_features
                else:
                    layer_list.append(layer_module)
                    if hasattr(layer_module, 'out_features'):
                        current_size = layer_module.out_features
                    elif hasattr(layer_module, 'hidden_size'):
                        current_size = layer_module.hidden_size * (2 if getattr(layer_module, 'bidirectional', False) else 1)
                    elif isinstance(layer_module, torch.nn.Embedding):
                        current_size = layer_module.embedding_dim
            
            # Add final output layer if needed
            if current_size != output_size:
                layer_list.append(torch.nn.Linear(current_size, output_size))
            
            # Create the appropriate model type
            if is_sequence_model:
                model = SequenceModel(layer_list, input_size, output_size, sequence_length or 1)
                model.has_embedding = has_embedding
            else:
                model = NeuroSequential(*layer_list, output_size=output_size)
        
        # Store the model in the variables dictionary
        self.variables[var_name] = model
        return model

    def execute_layer(self, layer_type, params, input_size, sequence_length=None):
        if layer_type == 'Dense':
            units = None
            activation = 'relu'
            
            for param in params:
                if param[0] == 'named_param':
                    if param[1] == 'units':
                        units = int(self.execute(param[2]))
                    elif param[1] == 'activation':
                        activation = str(self.execute(param[2]))
            
            if units is None:
                raise ValueError("Dense layer requires units parameter")
            
            if activation == 'relu':
                return (torch.nn.Linear(input_size, units), torch.nn.ReLU())
            elif activation == 'sigmoid':
                return (torch.nn.Linear(input_size, units), torch.nn.Sigmoid())
            elif activation == 'tanh':
                return (torch.nn.Linear(input_size, units), torch.nn.Tanh())
            elif activation == 'leaky_relu':
                return (torch.nn.Linear(input_size, units), torch.nn.LeakyReLU())
            elif activation == 'elu':
                return (torch.nn.Linear(input_size, units), torch.nn.ELU())
            else:
                return torch.nn.Linear(input_size, units)
                
        elif layer_type == 'LSTM':
            hidden_size = None
            num_layers = 1
            bidirectional = False
            
            for param in params:
                if param[0] == 'named_param':
                    if param[1] == 'hidden_size':
                        hidden_size = int(self.execute(param[2]))
                    elif param[1] == 'num_layers':
                        num_layers = int(self.execute(param[2]))
                    elif param[1] == 'bidirectional':
                        bidirectional = bool(self.execute(param[2]))
            
            if hidden_size is None:
                raise ValueError("LSTM layer requires hidden_size parameter")
                
            return torch.nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                batch_first=True
            )
            
        elif layer_type == 'GRU':
            hidden_size = None
            num_layers = 1
            bidirectional = False
            
            for param in params:
                if param[0] == 'named_param':
                    if param[1] == 'hidden_size':
                        hidden_size = int(self.execute(param[2]))
                    elif param[1] == 'num_layers':
                        num_layers = int(self.execute(param[2]))
                    elif param[1] == 'bidirectional':
                        bidirectional = bool(self.execute(param[2]))
            
            if hidden_size is None:
                raise ValueError("GRU layer requires hidden_size parameter")
                
            return torch.nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                batch_first=True
            )
            
        elif layer_type == 'Attention':
            num_heads = 8
            dropout = 0.1
            
            for param in params:
                if param[0] == 'named_param':
                    if param[1] == 'num_heads':
                        num_heads = int(self.execute(param[2]))
                    elif param[1] == 'dropout':
                        dropout = float(self.execute(param[2]))
            
            return SelfAttention(
                embed_dim=input_size,
                num_heads=num_heads,
                dropout=dropout
            )
            
        elif layer_type == 'Embedding':
            vocab_size = None
            embedding_dim = None
            
            for param in params:
                if param[0] == 'named_param':
                    if param[1] == 'vocab_size':
                        vocab_size = int(self.execute(param[2]))
                    elif param[1] == 'embedding_dim':
                        embedding_dim = int(self.execute(param[2]))
            
            if vocab_size is None or embedding_dim is None:
                raise ValueError("Embedding layer requires vocab_size and embedding_dim parameters")
                
            return torch.nn.Embedding(vocab_size, embedding_dim)
            
        elif layer_type == 'Conv2D':
            in_channels = None
            out_channels = None
            kernel_size = 3
            
            for param in params:
                if param[0] == 'named_param':
                    if param[1] == 'in_channels':
                        in_channels = int(self.execute(param[2]))
                    elif param[1] == 'out_channels':
                        out_channels = int(self.execute(param[2]))
                    elif param[1] == 'kernel_size':
                        kernel_size = int(self.execute(param[2]))
            
            if in_channels is None or out_channels is None:
                raise ValueError("Conv2D layer requires in_channels and out_channels")
            
            return torch.nn.Conv2d(in_channels, out_channels, kernel_size)
            
        elif layer_type == 'MaxPool':
            kernel_size = 2
            for param in params:
                if param[0] == 'named_param' and param[1] == 'kernel_size':
                    kernel_size = int(self.execute(param[2]))
            return torch.nn.MaxPool2d(kernel_size)
            
        elif layer_type == 'Dropout':
            rate = 0.5
            for param in params:
                if param[0] == 'named_param' and param[1] == 'rate':
                    rate = float(self.execute(param[2]))
            return torch.nn.Dropout(rate)
            
        elif layer_type == 'Flatten':
            return torch.nn.Flatten()
            
        elif layer_type == 'normalize':
            # Get the number of features from the previous layer
            num_features = None
            for param in params:
                if param[0] == 'named_param' and param[1] == 'features':
                    num_features = int(self.execute(param[2]))
            if num_features is None:
                num_features = input_size  # Use input size as number of features
            # Use LayerNorm instead of BatchNorm for better handling of small batches
            return torch.nn.LayerNorm(num_features)
            
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

    def get_loss_function(self, params):
        loss_type = None
        for param in params:
            if param[0] == 'named_param' and param[1] == 'type':
                loss_type = str(self.execute(param[2]))
        
        if loss_type == 'mse':
            return torch.nn.MSELoss()
        elif loss_type == 'cross_entropy':
            return torch.nn.CrossEntropyLoss(ignore_index=0)
        elif loss_type == 'bce':
            return torch.nn.BCEWithLogitsLoss()  # Using BCEWithLogitsLoss for better numerical stability
        else:
            return torch.nn.CrossEntropyLoss(ignore_index=0)

    def get_optimizer(self, model, params):
        optimizer_type = None
        learning_rate = 0.01
        
        for param in params:
            if param[0] == 'named_param':
                if param[1] == 'type':
                    optimizer_type = str(self.execute(param[2]))
                elif param[1] == 'learning_rate':
                    learning_rate = float(self.execute(param[2]))
        
        if optimizer_type == 'adam':
            return torch.optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_type == 'sgd':
            return torch.optim.SGD(model.parameters(), lr=learning_rate)
        elif optimizer_type == 'rmsprop':
            return torch.optim.RMSprop(model.parameters(), lr=learning_rate)
        else:
            return torch.optim.Adam(model.parameters(), lr=learning_rate)  # Default optimizer

    def execute_method_call(self, obj_expr, method_name, params):
        obj = self.execute(obj_expr)
        if obj is None:
            raise ValueError(f"Object not found")
        
        # Convert params list to dictionary
        param_dict = {}
        for param in params:
            if param[0] == 'param':
                # Handle positional parameters
                if not hasattr(param_dict, 'positional'):
                    param_dict['positional'] = []
                param_dict['positional'].append(self.execute(param[1]))
            elif param[0] == 'named_param':
                # Handle named parameters
                param_dict[param[1]] = self.execute(param[2])
        
        if method_name == 'train':
            # Get training parameters with defaults
            epochs = int(param_dict.get('epochs', 10))
            learning_rate = float(param_dict.get('learning_rate', 0.001))
            teacher_forcing_ratio = float(param_dict.get('teacher_forcing_ratio', 0.5))
            scheduler = param_dict.get('scheduler', None)
            warmup_steps = int(param_dict.get('warmup_steps', 10))
            data = param_dict['positional'][0] if 'positional' in param_dict else None
            
            if not isinstance(data, NeuroMatrix):
                raise ValueError("Training data must be a NeuroMatrix object")
            
            # Convert data to PyTorch dataset
            dataset = data.to_torch_dataset()
            
            # Initialize optimizer and scheduler
            optimizer = torch.optim.Adam(obj.parameters(), lr=learning_rate, weight_decay=0.01)
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
        result = self.execute(value)
        self.variables[var_name] = result
        return result

    def execute_binop(self, op, left, right):
        left_val = self.execute(left)
        right_val = self.execute(right)
        
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
        value = self.execute(expr)
        print(value)
        return value

    def execute_print_formatted(self, format_str, expr):
        value = self.execute(expr)
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