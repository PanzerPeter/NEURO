# NEURO Custom Error Types

class NeuroError(Exception):
    """Base class for all NEURO-specific errors."""
    def __init__(self, message, line=None, column=None, source_line=None, cause=None):
        super().__init__(message)
        self.message = message
        self.line = line
        self.column = column
        self.source_line = source_line # Store the source line content
        # Store the original exception if provided, for chaining
        if cause:
             self.__cause__ = cause

    def __str__(self):
        location = ""
        if self.line is not None:
            location += f"[Line {self.line}"
            if self.column is not None:
                location += f", Col {self.column}"
            location += "] "
        
        context = ""
        if self.source_line:
            context += f"\n> {self.source_line.strip()}"
            if self.column is not None:
                 # Add a simple pointer to the column (adjusting for 0-based column and '> ' prefix)
                 pointer_offset = self.column - 1 + 2 
                 context += f"\n  {' ' * pointer_offset}^"

        return f"{location}{type(self).__name__}: {self.message}{context}"

class NeuroSyntaxError(NeuroError):
    """Error raised during lexing or parsing."""
    pass

class NeuroNameError(NeuroError):
    """Error raised when an identifier (variable, function) is not found."""
    pass

class NeuroTypeError(NeuroError):
    """Custom exception for type mismatch errors during interpretation or processing."""
    pass

class NeuroArgumentError(NeuroTypeError):
    """Error raised for invalid arguments passed to functions or methods (e.g., wrong number, wrong type)."""
    pass

class NeuroInterpreterError(NeuroError):
    """Generic error raised during interpretation/execution."""
    pass

class NeuroDataError(NeuroError):
    """Custom exception for NeuroMatrix data loading/validation errors."""
    pass

# Example Usage (how you might raise an error):
# if some_condition_fails:
#     raise NeuroSyntaxError("Invalid character encountered", line=5, column=10) 