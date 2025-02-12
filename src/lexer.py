from ply import lex

class NeuroLexer:
    # List of token names
    tokens = (
        # Keywords
        'NEURAL_NETWORK',
        'LOAD_MATRIX',
        'SAVE_MODEL',
        'LOAD_MODEL',
        'DENSE',
        'CONV2D',
        'MAXPOOL',
        'DROPOUT',
        'FLATTEN',
        'NORMALIZE',
        'LSTM',
        'GRU',
        'ATTENTION',
        'EMBEDDING',
        'LOSS',
        'OPTIMIZER',
        'PRINT',
        
        # Operators
        'EQUALS',
        'PLUS',
        'MINUS',
        'TIMES',
        'DIVIDE',
        
        # Delimiters
        'LPAREN',
        'RPAREN',
        'COMMA',
        'DOT',
        'LBRACE',
        'RBRACE',
        
        # Data types
        'NUMBER',
        'STRING',
        'ID',
    )

    # Regular expression rules for simple tokens
    t_EQUALS = r'='
    t_PLUS = r'\+'
    t_MINUS = r'-'
    t_TIMES = r'\*'
    t_DIVIDE = r'/'
    t_LPAREN = r'\('
    t_RPAREN = r'\)'
    t_COMMA = r','
    t_DOT = r'\.'
    t_LBRACE = r'\{'
    t_RBRACE = r'\}'

    # Keywords
    keywords = {
        'NeuralNetwork': 'NEURAL_NETWORK',
        'load_matrix': 'LOAD_MATRIX',
        'save_model': 'SAVE_MODEL',
        'load_model': 'LOAD_MODEL',
        'Dense': 'DENSE',
        'Conv2D': 'CONV2D',
        'MaxPool': 'MAXPOOL',
        'Dropout': 'DROPOUT',
        'Flatten': 'FLATTEN',
        'normalize': 'NORMALIZE',
        'LSTM': 'LSTM',
        'GRU': 'GRU',
        'Attention': 'ATTENTION',
        'Embedding': 'EMBEDDING',
        'loss': 'LOSS',
        'optimizer': 'OPTIMIZER',
        'print': 'PRINT',
    }

    def __init__(self):
        # Build the lexer
        self.lexer = None
        self.build()

    # Regular expression rule with action for numbers
    def t_NUMBER(self, t):
        r'-?\d*\.?\d+'
        t.value = float(t.value)
        return t

    # Regular expression rule with action for strings
    def t_STRING(self, t):
        r'\"([^\\\n]|(\\.))*?\"'
        t.value = t.value[1:-1]
        return t

    # Regular expression rule for identifiers
    def t_ID(self, t):
        r'[a-zA-Z_][a-zA-Z_0-9]*'
        t.type = self.keywords.get(t.value, 'ID')
        return t

    # Define a rule so we can track line numbers
    def t_newline(self, t):
        r'\n+'
        t.lexer.lineno += len(t.value)

    # A string containing ignored characters (spaces and tabs)
    t_ignore = ' \t'

    # Skip comments
    def t_COMMENT(self, t):
        r'\#.*'
        pass

    # Error handling rule
    def t_error(self, t):
        print(f"Illegal character '{t.value[0]}' at line {t.lexer.lineno}")
        t.lexer.skip(1)

    def build(self, **kwargs):
        self.lexer = lex.lex(module=self, **kwargs)

    def tokenize(self, data):
        self.lexer.input(data)
        tokens = []
        while True:
            tok = self.lexer.token()
            if not tok:
                break
            tokens.append(tok)
        return tokens

# Example usage
if __name__ == '__main__':
    # Test input
    test_input = '''
    model = NeuralNetwork(input_size=128, output_size=10)
    model.train(data, epochs=10)
    accuracy = model.evaluate(test_data)
    '''
    
    # Create lexer and tokenize
    lexer = NeuroLexer()
    lexer.build()
    tokens = lexer.tokenize(test_input)
    
    # Print tokens
    for token in tokens:
        print(token) 