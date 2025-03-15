"""
NEURO Error Handling

This module defines the error handling system for the NEURO language.
It provides detailed error messages and debugging information.

Key responsibilities:
- Define custom exception classes
- Implement error reporting system
- Track source locations for errors
- Provide helpful error messages
- Handle runtime and compile-time errors
- Support for debugging information
- Stack trace management

This module is crucial for developer experience and debugging.
"""

from typing import Optional, Any, Tuple

class NeuroError(Exception):
    """Base class for all NEURO language errors."""
    def __init__(self, message: str, line: Optional[int] = None, column: Optional[int] = None,
                 context: Optional[str] = None, source_line: Optional[str] = None, 
                 details: Optional[str] = None, token_length: Optional[int] = None):
        self.message = message
        self.line = line
        self.column = column
        self.context = context
        self.source_line = source_line
        self.details = details
        self.token_length = token_length
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the error message with available context."""
        parts = [self.message]
        if self.context:
            parts.append(f" ({self.context})")
        if self.line is not None:
            parts.append(f" at line {self.line}")
            if self.column is not None:
                parts.append(f", column {self.column}")
        if self.source_line:
            parts.append(f"\n{self.source_line}")
            if self.column is not None and self.token_length:
                parts.append("\n" + " " * self.column + "^" * self.token_length)
        if self.details:
            parts.append(f"\nDetails: {self.details}")
        return "".join(parts)

class NeuroSyntaxError(NeuroError):
    """Error raised for syntax errors."""
    pass

class NeuroRuntimeError(NeuroError):
    """Error raised for runtime errors."""
    pass

class NeuroNameError(NeuroError):
    """Error raised for undefined names."""
    def __init__(self, name: str, message: Optional[str] = None, **kwargs):
        self.name = name
        # Include the name in the message if not already included
        if message and name not in message:
            message_text = f"{message} '{name}'"
        else:
            message_text = message or f"Variable '{name}' is not defined"
        super().__init__(message_text, **kwargs)

class NeuroTypeError(NeuroError):
    """Error raised for type errors."""
    def __init__(self, message: str, expected_type: Optional[str] = None, 
                 actual_type: Optional[str] = None, **kwargs):
        details = None
        if expected_type and actual_type:
            details = f"Expected type: {expected_type}, got: {actual_type}"
        super().__init__(message, details=details, **kwargs)

class NeuroValueError(NeuroError):
    """Error raised for invalid values."""
    def __init__(self, message: str, value: Optional[Any] = None, **kwargs):
        # Check if details is already provided in kwargs
        if value is not None and 'details' not in kwargs:
            kwargs['details'] = f"Invalid value: {value}"
        super().__init__(message, **kwargs)

class NeuroShapeError(NeuroError):
    """Error raised for tensor shape mismatches."""
    def __init__(self, message: str, expected_shape: Tuple[int, ...], 
                 actual_shape: Tuple[int, ...], **kwargs):
        # Format the details to match test expectations
        details = f"expected {expected_shape}, got {actual_shape}"
        super().__init__(message, details=details, **kwargs)

class NeuroLayerError(NeuroError):
    """Error raised for layer-related errors."""
    def __init__(self, message: str, layer_type: Optional[str] = None, **kwargs):
        details = f"In layer type: {layer_type}" if layer_type else None
        super().__init__(message, details=details, **kwargs)

class NeuroAttributeError(NeuroError):
    """Error raised for attribute errors."""
    def __init__(self, obj: str, attr: str, **kwargs):
        super().__init__(
            f"Object '{obj}' has no attribute '{attr}'",
            **kwargs
        )

class NeuroImportError(NeuroError):
    """Raised when an import fails."""
    def __init__(self, message: str, line: Optional[int] = None, column: Optional[int] = None, module: Optional[str] = None):
        super().__init__(message, line, column)
        self.module = module

def format_error_location(code: str, line: int, column: int, context_lines: int = 2) -> str:
    """Format error location with context lines and pointer."""
    lines = code.split('\n')
    start = max(0, line - context_lines - 1)
    end = min(len(lines), line + context_lines)
    
    # Build the error message with line numbers and context
    result = []
    for i in range(start, end):
        line_num = i + 1
        if line_num == line:
            # Add the error line with pointer
            result.append(f"{line_num:4d} | {lines[i]}")
            result.append("     | " + " " * (column - 1) + "^")
        else:
            result.append(f"{line_num:4d} | {lines[i]}")
    
    return "\n".join(result)

def create_error_message(error: NeuroError, code: Optional[str] = None) -> str:
    """Create a detailed error message with context if code is provided."""
    message = str(error)
    
    if code and error.line is not None and error.column is not None:
        message += "\n\n" + format_error_location(
            code, error.line, error.column
        )
    
    return message 