# NEURO Abstract Syntax Tree (AST) Node Definitions

from abc import ABC, abstractmethod
from typing import Optional

# Import the NeuroType base class for type hinting the attribute
# Assuming neuro_types.py is in the same directory
# Use a forward reference string if necessary to avoid circular imports
from .neuro_types import NeuroType

class ASTNode(ABC):
    """Base class for all AST nodes."""
    # Optionally add line/column info here if needed for error reporting
    # def __init__(self, line=None, column=None):
    #     self.line = line
    #     self.column = column
    # Add a common attribute for holding type information inferred/checked later
    type_info: Optional['NeuroType'] = None
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
        self.type_info: Optional['NeuroType'] = None # Models have types

class Layer(ASTNode):
    """Node representing a layer definition inside a model block."""
    def __init__(self, layer_type, params):
        self.layer_type = layer_type # String (e.g., "Dense", "Conv2D")
        self.params = params         # Dictionary of layer parameters (e.g., units, activation)
        self.type_info: Optional['NeuroType'] = None # Layers have types (LayerType)

class LossDefinition(ASTNode):
    """Node representing 'loss(...)'"""
    def __init__(self, params):
        self.params = params # Dictionary of loss parameters (e.g., type, smoothing)
        self.type_info: Optional['NeuroType'] = None # Losses have types (LossType)

class OptimizerDefinition(ASTNode):
    """Node representing 'optimizer(...)'"""
    def __init__(self, params):
        self.params = params # Dictionary of optimizer parameters (e.g., type, learning_rate)
        self.type_info: Optional['NeuroType'] = None # Optimizers have types (OptimizerType)

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
        self.data_source = data_source # Identifier or Literal (string path?)

class MethodCall(ASTNode):
    """Node representing a general method call like 'object.method(...)' (used as expression)"""
    def __init__(self, object_name, method_name, args):
        self.object_name = object_name # Identifier (string)
        self.method_name = method_name # Identifier (string)
        self.args = args             # Dictionary of arguments (parsed key-value pairs)
        self.type_info: Optional['NeuroType'] = None # Method calls return a value with a type

class SaveStatement(ASTNode):
    """Node representing 'model.save("filepath")'"""
    def __init__(self, model_name, filepath):
        self.model_name = model_name # Identifier (string)
        self.filepath = filepath   # StringLiteral node

class AssignmentStatement(ASTNode):
    """Node representing 'target = value' or 'target1, target2 = value'"""
    def __init__(self, target, value):
        # self.target = target # Identifier node
        # Target can be a single Identifier or a list of Identifiers for tuple assignment
        self.target: Identifier | list[Identifier] = target
        self.value = value   # Expression or Call node
        # Assignment itself might be considered VoidType, but the target(s) and value have types
        # The type checker will handle assigning types to the target identifiers
        self.type_info: Optional['NeuroType'] = None # Typically VoidType for the statement itself

# --- Expressions (if needed later) ---
class Identifier(ASTNode):
    """Node representing an identifier (variable name)."""
    def __init__(self, name):
        self.name = name
        self.type_info: Optional['NeuroType'] = None # Identifiers refer to values with types

class NumberLiteral(ASTNode):
    """Node representing a numeric literal."""
    def __init__(self, value):
        self.value = value
        self.type_info: Optional['NeuroType'] = None # Will be IntType or FloatType

class StringLiteral(ASTNode):
    """Node representing a string literal."""
    def __init__(self, value):
        self.value = value
        self.type_info: Optional['NeuroType'] = None # Will be StringType

class FunctionCall(ASTNode):
    """Node representing a function call like 'func_name(arg1, arg2, ...)'."""
    def __init__(self, func_name, args):
        self.func_name = func_name # Identifier node (or just the string name?)
        self.args = args         # List of argument nodes (expressions)
        self.type_info: Optional['NeuroType'] = None # Function calls return a value with a type

# Add more node types as the language grammar expands (e.g., assignments, function calls, etc.) 