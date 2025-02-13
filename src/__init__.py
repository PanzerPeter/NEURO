"""
NEURO Language Package
"""

from .lexer import NeuroLexer
from .parser import NeuroParser
from .interpreter import (
    NeuroInterpreter, 
    CustomLayer, 
    ResidualBlock, 
    LabelSmoothing, 
    SelfAttention,
    SequenceModel,
    BranchingModule,
    PretrainedBackbone,
    TrainingCallback,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoardLogger
)
from .errors import NeuroError, ValidationError, ConfigurationError, get_context
from .matrix import NeuroMatrix
from .memory import MemoryManager, MemoryStats, GradientCheckpoint
from .debugger import ModelDebugger, DebugContext
from .repl import NeuroREPL

__version__ = '0.1.0-alpha'
__author__ = 'NEURO Development Team'

__all__ = [
    # Core components
    'NeuroLexer',
    'NeuroParser',
    'NeuroInterpreter',
    'NeuroMatrix',
    'NeuroREPL',
    
    # Model components
    'CustomLayer',
    'ResidualBlock',
    'LabelSmoothing',
    'SelfAttention',
    'SequenceModel',
    'BranchingModule',
    'PretrainedBackbone',
    
    # Training utilities
    'TrainingCallback',
    'EarlyStopping',
    'ModelCheckpoint',
    'TensorBoardLogger',
    
    # Memory management
    'MemoryManager',
    'MemoryStats',
    'GradientCheckpoint',
    
    # Debugging
    'ModelDebugger',
    'DebugContext',
    
    # Error handling
    'NeuroError',
    'ValidationError',
    'ConfigurationError',
    'get_context'
] 