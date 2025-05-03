# NEURO Custom Error Types

class NeuroError(Exception):
    """Base class for all NEURO-specific errors."""
    def __init__(self, message, line=None, column=None):
        super().__init__(message)
        self.message = message
        self.line = line
        self.column = column

    def __str__(self):
        location = ""
        if self.line is not None:
            location += f"[Line {self.line}" 
            if self.column is not None:
                location += f", Col {self.column}" 
            location += "] "
        return f"{location}{type(self).__name__}: {self.message}"

class NeuroSyntaxError(NeuroError):
    """Error raised during lexing or parsing."""
    pass

class NeuroNameError(NeuroError):
    """Error raised when an identifier (variable, function) is not found."""
    pass

class NeuroTypeError(NeuroError):
    """Custom exception for type mismatch errors during interpretation or processing."""
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