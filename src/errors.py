"""
NEURO Error Handling Module
Provides custom error classes and utilities for better error reporting and debugging.
"""

import inspect
import traceback
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

@dataclass
class CodeContext:
    """Contains contextual information about where an error occurred."""
    line_no: int
    code_snippet: str
    function_name: Optional[str] = None
    file_name: Optional[str] = None

class NeuroError(Exception):
    """Base class for all NEURO-specific errors."""
    
    def __init__(
        self,
        message: str,
        context: Optional[CodeContext] = None,
        suggestions: Optional[List[str]] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.context = context
        self.suggestions = suggestions or []
        self.details = details or {}
        
        # Build the full error message
        full_message = self._build_error_message()
        super().__init__(full_message)
    
    def _build_error_message(self) -> str:
        """Builds a detailed error message with context and suggestions."""
        parts = [f"\n{'='*80}\nNEURO Error: {self.message}\n{'='*80}\n"]
        
        if self.context:
            parts.append(f"\nLocation:")
            if self.context.file_name:
                parts.append(f"  File: {self.context.file_name}")
            if self.context.function_name:
                parts.append(f"  Function: {self.context.function_name}")
            parts.append(f"  Line {self.context.line_no}:")
            parts.append(f"\nCode:")
            parts.append(f"  {self.context.code_snippet}")
        
        if self.details:
            parts.append("\nDetails:")
            for key, value in self.details.items():
                parts.append(f"  {key}: {value}")
        
        if self.suggestions:
            parts.append("\nSuggestions:")
            for suggestion in self.suggestions:
                parts.append(f"  • {suggestion}")
        
        return "\n".join(parts)

class MemoryError(NeuroError):
    """Raised when memory-related issues occur."""
    pass

class GradientError(NeuroError):
    """Raised when gradient-related issues occur."""
    pass

class ValidationError(NeuroError):
    """Raised when validation of models or data fails."""
    pass

class ConfigurationError(NeuroError):
    """Raised when there are issues with model or training configuration."""
    pass

def get_context(frames_back: int = 1) -> CodeContext:
    """
    Extracts context information from the current call stack.
    
    Args:
        frames_back: How many frames to go back in the stack to find context
    
    Returns:
        CodeContext object containing error location information
    """
    frame = inspect.currentframe()
    try:
        for _ in range(frames_back + 1):
            if frame.f_back is None:
                break
            frame = frame.f_back
            
        if frame is None:
            return None
            
        filename = frame.f_code.co_filename
        function = frame.f_code.co_name
        lineno = frame.f_lineno
        
        # Get the code snippet
        lines, _ = inspect.findsource(frame)
        start = max(0, lineno - 2)
        end = min(len(lines), lineno + 3)
        snippet = ''.join(lines[start:end])
        
        return CodeContext(
            line_no=lineno,
            code_snippet=snippet.strip(),
            function_name=function,
            file_name=filename
        )
    finally:
        del frame  # Avoid reference cycles

def format_memory_size(size_bytes: int) -> str:
    """
    Formats a memory size in bytes to a human-readable string.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.23 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB" 