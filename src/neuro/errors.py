"""
Error Handling Slice
Handles all error types and formatting for the NEURO compiler.
"""

from typing import Optional, List, Any
from dataclasses import dataclass


@dataclass
class SourceLocation:
    """Represents a location in source code."""
    filename: str
    line: int
    column: int
    
    def __str__(self) -> str:
        return f"{self.filename}:{self.line}:{self.column}"


class NeuroError(Exception):
    """Base class for all NEURO compiler errors."""
    
    def __init__(self, message: str, location: Optional[SourceLocation] = None):
        self.message = message
        self.location = location
        super().__init__(self._format_error())
    
    def _format_error(self) -> str:
        if self.location:
            return f"{self.location}: error: {self.message}"
        return f"error: {self.message}"


class NeuroSyntaxError(NeuroError):
    """Syntax error in NEURO source code."""
    
    def __init__(self, message: str, location: Optional[SourceLocation] = None, 
                 expected: Optional[str] = None, found: Optional[str] = None):
        self.expected = expected
        self.found = found
        
        if expected and found:
            full_message = f"{message}. Expected {expected}, found {found}"
        else:
            full_message = message
            
        super().__init__(full_message, location)


class NeuroTypeError(NeuroError):
    """Type error in NEURO source code."""
    
    def __init__(self, message: str, location: Optional[SourceLocation] = None,
                 expected_type: Optional[str] = None, actual_type: Optional[str] = None):
        self.expected_type = expected_type
        self.actual_type = actual_type
        
        if expected_type and actual_type:
            full_message = f"{message}. Expected type '{expected_type}', found '{actual_type}'"
        else:
            full_message = message
            
        super().__init__(full_message, location)


class NeuroSemanticError(NeuroError):
    """Semantic error in NEURO source code."""
    pass


class NeuroRuntimeError(NeuroError):
    """Runtime error during NEURO program execution."""
    pass


class ErrorReporter:
    """Collects and reports compilation errors."""
    
    def __init__(self):
        self.errors: List[NeuroError] = []
        self.warnings: List[str] = []
    
    def error(self, message: str, location: Optional[SourceLocation] = None) -> None:
        """Report a general error."""
        self.errors.append(NeuroError(message, location))
    
    def syntax_error(self, message: str, location: Optional[SourceLocation] = None,
                     expected: Optional[str] = None, found: Optional[str] = None) -> None:
        """Report a syntax error."""
        self.errors.append(NeuroSyntaxError(message, location, expected, found))
    
    def type_error(self, message: str, location: Optional[SourceLocation] = None,
                   expected_type: Optional[str] = None, actual_type: Optional[str] = None) -> None:
        """Report a type error."""
        self.errors.append(NeuroTypeError(message, location, expected_type, actual_type))
    
    def semantic_error(self, message: str, location: Optional[SourceLocation] = None) -> None:
        """Report a semantic error."""
        self.errors.append(NeuroSemanticError(message, location))
    
    def warning(self, message: str, location: Optional[SourceLocation] = None) -> None:
        """Report a warning."""
        if location:
            self.warnings.append(f"{location}: warning: {message}")
        else:
            self.warnings.append(f"warning: {message}")
    
    def has_errors(self) -> bool:
        """Check if any errors have been reported."""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if any warnings have been reported."""
        return len(self.warnings) > 0
    
    def get_error_count(self) -> int:
        """Get the number of errors."""
        return len(self.errors)
    
    def get_warning_count(self) -> int:
        """Get the number of warnings."""
        return len(self.warnings)
    
    def print_errors(self) -> None:
        """Print all errors to stderr."""
        for error in self.errors:
            print(f"{error}", file=__import__('sys').stderr)
    
    def print_warnings(self) -> None:
        """Print all warnings to stderr."""
        for warning in self.warnings:
            print(f"{warning}", file=__import__('sys').stderr)
    
    def print_summary(self) -> None:
        """Print error and warning summary."""
        if self.has_errors():
            error_count = self.get_error_count()
            error_text = "error" if error_count == 1 else "errors"
            print(f"{error_count} {error_text} generated.", file=__import__('sys').stderr)
        
        if self.has_warnings():
            warning_count = self.get_warning_count()
            warning_text = "warning" if warning_count == 1 else "warnings"
            print(f"{warning_count} {warning_text} generated.", file=__import__('sys').stderr)
    
    def clear(self) -> None:
        """Clear all errors and warnings."""
        self.errors.clear()
        self.warnings.clear() 