# NEURO Abstract Syntax Tree (AST) Node Definitions

from abc import ABC, abstractmethod

class ASTNode(ABC):
    """Base class for all AST nodes."""
    # Optionally add line/column info here if needed for error reporting
    # def __init__(self, line=None, column=None):
    #     self.line = line
    #     self.column = column
    pass

# --- Program Structure ---

class Program(ASTNode):
    """Root node for a NEURO program."""
    def __init__(self, body):
        self.body = body  # List of statements/definitions

# --- Statements/Definitions ---

class ModelDefinition(ASTNode):
    """Node representing 'model = NeuralNetwork(...) { ... }'"""
    def __init__(self, name, params, layers):
        self.name = name      # Identifier (string)
        self.params = params  # Dictionary of parameters (e.g., input_size)
        self.layers = layers  # List of Layer nodes

class Layer(ASTNode):
    """Node representing a layer definition inside a model block."""
    def __init__(self, layer_type, params):
        self.layer_type = layer_type # String (e.g., "Dense", "Conv2D")
        self.params = params         # Dictionary of layer parameters (e.g., units, activation)

class LossDefinition(ASTNode):
    """Node representing 'loss(...)'"""
    def __init__(self, params):
        self.params = params # Dictionary of loss parameters (e.g., type, smoothing)

class OptimizerDefinition(ASTNode):
    """Node representing 'optimizer(...)'"""
    def __init__(self, params):
        self.params = params # Dictionary of optimizer parameters (e.g., type, learning_rate)

class TrainStatement(ASTNode):
    """Node representing 'model.train(...)'"""
    def __init__(self, model_name, data_source, params):
        self.model_name = model_name    # Identifier (string)
        self.data_source = data_source # Identifier or Literal (string path?)
        self.params = params           # Dictionary of training parameters (e.g., epochs)

class EvaluateStatement(ASTNode):
    """Node representing 'model.evaluate(...)'"""
    def __init__(self, model_name, data_source):
        self.model_name = model_name    # Identifier (string)
        self.data_source = data_source # Identifier or Literal

class AssignmentStatement(ASTNode):
    """Node representing 'target = value'"""
    def __init__(self, target, value):
        self.target = target # Identifier node
        self.value = value   # Expression or Call node

# --- Expressions (if needed later) ---
class Identifier(ASTNode):
    """Node representing an identifier (variable name)."""
    def __init__(self, name):
        self.name = name

class NumberLiteral(ASTNode):
    """Node representing a numeric literal."""
    def __init__(self, value):
        self.value = value

class StringLiteral(ASTNode):
    """Node representing a string literal."""
    def __init__(self, value):
        self.value = value

# Add more node types as the language grammar expands (e.g., assignments, function calls, etc.) 