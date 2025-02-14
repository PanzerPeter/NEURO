from pygments.lexer import RegexLexer, words, include, bygroups
from pygments.token import *
from pygments import highlight
from pygments.formatters import Terminal256Formatter
from pygments.style import Style
from pygments.lexers import Python3Lexer

class NeuroStyle(Style):
    """Custom color scheme for NEURO language with extended styling"""
    default_style = ""
    styles = {
        Comment.Single:      'italic #888888',      # Gray comments
        Keyword:            'bold #6699cc',         # Blue keywords
        Keyword.Type:       'bold #5588bb',         # Darker blue for types
        Keyword.Constant:   'bold #7799dd',         # Different blue for constants
        Name.Function:      'bold #ffcc66',         # Yellow functions
        Name.Class:         'bold #ff9966',         # Orange for classes
        Name.Decorator:     'bold #cc99ff',         # Light purple for decorators
        String:             '#cc99cc',              # Purple strings
        Number:             '#99cc99',              # Green numbers
        Number.Float:       '#88bb88',              # Darker green for floats
        Number.Matrix:      '#77aa77',              # Even darker green for matrices
        Operator:           'bold #f2777a',         # Red operators
        Operator.Word:      'bold #ff6666',         # Brighter red for word operators
        Name.Variable:      '#66cccc',              # Cyan variables
        Name.Variable.Class: '#55bbbb',             # Darker cyan for class variables
        Name.Builtin:       '#ffbb66',              # Orange-yellow for builtins
        Name.Exception:     '#ff7777',              # Bright red for exceptions
        Text:              '#ffffff',               # White text
        Punctuation:       '#dddddd',              # Light gray for punctuation
    }

class NeuroLexer(RegexLexer):
    name = 'NEURO'
    aliases = ['neuro']
    filenames = ['*.nr', '*.nrm']

    # Extended keyword lists
    keywords = (
        'NeuralNetwork', 'Dense', 'Conv2D', 'MaxPool', 'Dropout', 'Flatten',
        'normalize', 'print', 'return', 'if', 'else', 'for', 'while',
        'break', 'continue', 'import', 'from'
    )

    declarations = (
        'model', 'layer', 'function', 'class', 'def'
    )

    types = (
        'Matrix', 'Vector', 'Tensor', 'Number', 'String', 'Boolean'
    )

    builtins = (
        'load_matrix', 'save_model', 'load_model', 'train', 'evaluate',
        'predict', 'compile', 'fit', 'summary'
    )

    constants = (
        'true', 'false', 'null', 'None'
    )

    tokens = {
        'root': [
            # Comments
            (r'#.*$', Comment.Single),
            
            # Docstrings
            (r'"""(?:.|\n)*?"""', String.Doc),
            (r"'''(?:.|\n)*?'''", String.Doc),
            
            # Keywords with different categories
            (words(declarations, prefix=r'(?<!\.)', suffix=r'\b'), Keyword.Declaration),
            (words(keywords, prefix=r'(?<!\.)', suffix=r'\b'), Keyword),
            (words(types, prefix=r'(?<!\.)', suffix=r'\b'), Keyword.Type),
            (words(constants, prefix=r'(?<!\.)', suffix=r'\b'), Keyword.Constant),
            
            # Decorators
            (r'@[\w.]+', Name.Decorator),
            
            # Functions and methods
            (r'(def)(\s+)([a-zA-Z_][a-zA-Z0-9_]*)',
             bygroups(Keyword.Declaration, Text, Name.Function)),
            (words(builtins, prefix=r'(?<!\.)', suffix=r'\b'), Name.Builtin),
            
            # Matrix literals
            (r'\[\[.*?\]\]', Number.Matrix),
            
            # Strings with escape sequences
            (r'"(?:[^"\\]|\\.)*"', String.Double),
            (r"'(?:[^'\\]|\\.)*'", String.Single),
            
            # Numbers with different formats
            (r'\d+\.\d*([eE][+-]?\d+)?', Number.Float),
            (r'\d*\.\d+([eE][+-]?\d+)?', Number.Float),
            (r'\d+[eE][+-]?\d+', Number.Float),
            (r'\d+', Number.Integer),
            
            # Enhanced operator handling
            (r'(==|!=|<=|>=|<|>|\+|-|\*|/|%|\*\*|//|\+=|-=|\*=|/=|%=)', Operator),
            (words(('and', 'or', 'not', 'in', 'is'), prefix=r'(?<!\.)', suffix=r'\b'), Operator.Word),
            
            # Delimiters and punctuation
            (r'[\[\](){}.,:]', Punctuation),
            
            # Class definitions
            (r'(class)(\s+)([a-zA-Z_][a-zA-Z0-9_]*)',
             bygroups(Keyword.Declaration, Text, Name.Class)),
            
            # Variables with different scopes
            (r'self\.[a-zA-Z_][a-zA-Z0-9_]*', Name.Variable.Instance),
            (r'[A-Z][a-zA-Z0-9_]*', Name.Class),
            (r'[a-zA-Z_][a-zA-Z0-9_]*', Name.Variable),
            
            # Whitespace
            (r'\s+', Text),
        ]
    }

def highlight_code(code):
    """Highlight NEURO code with custom colors."""
    return highlight(code, NeuroLexer(), Terminal256Formatter(style=NeuroStyle)) 