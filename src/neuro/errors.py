"""
NEURO Error Handling System

Comprehensive error types and error reporting for the NEURO compiler.
Provides detailed error messages with source location information.
"""

from typing import Optional, List, Any
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceLocation:
    """Represents a location in source code for error reporting."""
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
        super().__init__(self.format_error())
    
    def format_error(self) -> str:
        """Format the error message with location information."""
        if self.location:
            return f"{self.location}: {self.message}"
        return self.message


class LexerError(NeuroError):
    """Error during lexical analysis (tokenization)."""
    
    def __init__(self, message: str, location: SourceLocation, char: str = ""):
        self.char = char
        if char:
            message = f"{message} (found '{char}')"
        super().__init__(message, location)


class ParseError(NeuroError):
    """Error during parsing (AST construction)."""
    
    def __init__(self, message: str, location: Optional[SourceLocation] = None, 
                 expected: Optional[str] = None, found: Optional[str] = None):
        self.expected = expected
        self.found = found
        
        if expected and found:
            message = f"{message}: expected {expected}, found {found}"
        elif expected:
            message = f"{message}: expected {expected}"
        elif found:
            message = f"{message}: unexpected {found}"
            
        super().__init__(message, location)


class TypeError(NeuroError):
    """Error during type checking and inference."""
    
    def __init__(self, message: str, location: Optional[SourceLocation] = None,
                 expected_type: Optional[str] = None, actual_type: Optional[str] = None):
        self.expected_type = expected_type
        self.actual_type = actual_type
        
        if expected_type and actual_type:
            message = f"{message}: expected type '{expected_type}', got '{actual_type}'"
        elif expected_type:
            message = f"{message}: expected type '{expected_type}'"
        elif actual_type:
            message = f"{message}: cannot use type '{actual_type}'"
            
        super().__init__(message, location)


class NameError(NeuroError):
    """Error with variable/function names and scoping."""
    
    def __init__(self, message: str, location: Optional[SourceLocation] = None,
                 name: Optional[str] = None):
        self.name = name
        
        if name:
            message = f"{message}: '{name}'"
            
        super().__init__(message, location)


class CompilerError(NeuroError):
    """Error during code generation or optimization."""
    
    def __init__(self, message: str, location: Optional[SourceLocation] = None,
                 phase: Optional[str] = None):
        self.phase = phase
        
        if phase:
            message = f"{phase}: {message}"
            
        super().__init__(message, location)


class InternalError(NeuroError):
    """Internal compiler error - should not happen in normal operation."""
    
    def __init__(self, message: str, location: Optional[SourceLocation] = None):
        message = f"Internal compiler error: {message}"
        super().__init__(message, location)


def error_with_context(error: NeuroError, source_lines: List[str]) -> str:
    """
    Format an error with source code context for better debugging.
    
    Args:
        error: The error to format
        source_lines: Lines of source code for context
        
    Returns:
        Formatted error message with source context
    """
    if not error.location or not source_lines:
        return str(error)
    
    location = error.location
    line_num = location.line
    col_num = location.column
    
    # Build context window (3 lines before and after)
    start_line = max(0, line_num - 4)
    end_line = min(len(source_lines), line_num + 3)
    
    lines = []
    lines.append(str(error))
    lines.append("")
    
    for i in range(start_line, end_line):
        line_marker = ">>>" if i == line_num - 1 else "   "
        line_number = f"{i + 1:3d}"
        line_content = source_lines[i].rstrip()
        lines.append(f"{line_marker} {line_number} | {line_content}")
        
        # Add pointer to error column
        if i == line_num - 1 and col_num > 0:
            pointer = " " * (8 + col_num - 1) + "^"
            lines.append(pointer)
    
    return "\n".join(lines)


class ErrorReporter:
    """Collects and reports multiple errors during compilation."""
    
    def __init__(self):
        self.errors: List[NeuroError] = []
        self.warnings: List[NeuroError] = []
    
    def error(self, error: NeuroError) -> None:
        """Add an error to the collection."""
        self.errors.append(error)
    
    def warning(self, warning: NeuroError) -> None:
        """Add a warning to the collection."""
        self.warnings.append(warning)
    
    def has_errors(self) -> bool:
        """Check if any errors have been reported."""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if any warnings have been reported."""
        return len(self.warnings) > 0
    
    def report(self, source_lines: Optional[List[str]] = None) -> str:
        """
        Generate a formatted report of all errors and warnings.
        
        Args:
            source_lines: Source code lines for context
            
        Returns:
            Formatted error report
        """
        lines = []
        
        if self.errors:
            lines.append(f"Found {len(self.errors)} error(s):")
            lines.append("")
            
            for error in self.errors:
                if source_lines:
                    lines.append(error_with_context(error, source_lines))
                else:
                    lines.append(str(error))
                lines.append("")
        
        if self.warnings:
            lines.append(f"Found {len(self.warnings)} warning(s):")
            lines.append("")
            
            for warning in self.warnings:
                if source_lines:
                    lines.append(error_with_context(warning, source_lines))
                else:
                    lines.append(str(warning))
                lines.append("")
        
        return "\n".join(lines)
    
    def clear(self) -> None:
        """Clear all errors and warnings."""
        self.errors.clear()
        self.warnings.clear() 