from pygments.lexer import RegexLexer, words, include
from pygments.token import *
from pygments import highlight
from pygments.formatters import Terminal256Formatter
from pygments.style import Style
from pygments.lexers import Python3Lexer

class NeuroStyle(Style):
    """Custom color scheme for NEURO language"""
    default_style = ""
    styles = {
        Comment.Single:      'italic #888888',  # Gray comments
        Keyword:            'bold #6699cc',     # Blue keywords
        Name.Function:      'bold #ffcc66',     # Yellow functions
        String:             '#cc99cc',          # Purple strings
        Number:             '#99cc99',          # Green numbers
        Operator:           'bold #f2777a',     # Red operators
        Name.Variable:      '#66cccc',          # Cyan variables
        Text:              '#ffffff',           # White text
    }

class NeuroLexer(RegexLexer):
    name = 'NEURO'
    aliases = ['neuro']
    filenames = ['*.nr']

    # Define keywords with different colors
    keywords = (
        'NeuralNetwork',
        'Dense',
        'Conv2D',
        'MaxPool',
        'Dropout',
        'Flatten',
        'normalize',
        'print'
    )

    functions = (
        'load_matrix',
        'save_model',
        'load_model',
        'train',
        'evaluate'
    )

    tokens = {
        'root': [
            # Comments in gray italic
            (r'#.*$', Comment.Single),
            
            # Keywords in bold blue
            (words(keywords, prefix=r'(?<!\.)', suffix=r'\b'), Keyword),
            
            # Functions in bold yellow
            (words(functions, prefix=r'(?<!\.)', suffix=r'\b'), Name.Function),
            
            # Strings in purple
            (r'"[^"]*"', String),
            
            # Numbers in green
            (r'\d*\.?\d+', Number),
            
            # Operators in bold red
            (r'[=+\-*/]', Operator),
            
            # Delimiters in bold red
            (r'[(){}.,]', Operator),
            
            # Variables in cyan
            (r'[a-zA-Z_][a-zA-Z0-9_]*', Name.Variable),
            
            # Whitespace
            (r'\s+', Text),
        ]
    }

def highlight_code(code):
    """Highlight NEURO code with custom colors."""
    return highlight(code, NeuroLexer(), Terminal256Formatter(style=NeuroStyle)) 