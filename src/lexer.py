from ply import lex

class NeuroLexer:
    # List of token names
    tokens = [
        'ID', 'NUMBER', 'PLUS', 'MINUS', 'TIMES', 'DIVIDE',
        'LPAREN', 'RPAREN', 'LBRACE', 'RBRACE', 'EQUALS',
        'COMMA', 'DOT', 'STRING', 'AT', 'BRANCH',
        'DENSE', 'CONV2D', 'MAXPOOL', 'DROPOUT', 'FLATTEN',
        'NORMALIZE', 'LSTM', 'GRU', 'ATTENTION', 'EMBEDDING',
        'CUSTOM_LAYER', 'PRETRAINED', 'DEF',
        'NEURAL_NETWORK', 'LOAD_MATRIX', 'SAVE_MODEL', 'LOAD_MODEL',
        'LOSS', 'OPTIMIZER', 'PRINT',
        'COLON', 'LBRACKET', 'RBRACKET', 'SEMI',
        'RETURN', 'FOR', 'IN', 'RANGE'
    ]
    
    # Regular expression rules for simple tokens
    t_PLUS = r'\+'
    t_MINUS = r'-'
    t_TIMES = r'\*'
    t_DIVIDE = r'/'
    t_LPAREN = r'\('
    t_RPAREN = r'\)'
    t_LBRACE = r'\{'
    t_RBRACE = r'\}'
    t_EQUALS = r'='
    t_COMMA = r','
    t_DOT = r'\.'
    t_AT = r'@'
    t_COLON = r':'
    t_LBRACKET = r'\['
    t_RBRACKET = r'\]'
    t_SEMI = r';'
    
    # Keywords
    keywords = {
        'Dense': 'DENSE',
        'Conv2D': 'CONV2D',
        'MaxPool': 'MAXPOOL',
        'Dropout': 'DROPOUT',
        'Flatten': 'FLATTEN',
        'Normalize': 'NORMALIZE',
        'LSTM': 'LSTM',
        'GRU': 'GRU',
        'Attention': 'ATTENTION',
        'Embedding': 'EMBEDDING',
        'Branch': 'BRANCH',
        'custom_layer': 'CUSTOM_LAYER',
        'pretrained': 'PRETRAINED',
        'def': 'DEF',
        'NeuralNetwork': 'NEURAL_NETWORK',
        'load_matrix': 'LOAD_MATRIX',
        'save_model': 'SAVE_MODEL',
        'load_model': 'LOAD_MODEL',
        'loss': 'LOSS',
        'optimizer': 'OPTIMIZER',
        'print': 'PRINT',
        'return': 'RETURN',
        'for': 'FOR',
        'in': 'IN',
        'range': 'RANGE'
    }
    
    def t_ID(self, t):
        r'[a-zA-Z_][a-zA-Z_0-9]*'
        t.type = self.keywords.get(t.value, 'ID')
        return t
        
    def t_NUMBER(self, t):
        r'\d*\.\d+|\d+'
        t.value = float(t.value)
        return t
        
    def t_STRING(self, t):
        r'\"[^\"]*\"|\'[^\']*\''
        t.value = t.value[1:-1]  # Remove quotes
        return t
        
    # Track line numbers
    def t_newline(self, t):
        r'\n+'
        t.lexer.lineno += len(t.value)
        
    # Ignored characters (spaces and tabs)
    t_ignore = ' \t'
    
    def t_COMMENT(self, t):
        r'\#.*'
        pass  # No return value. Token is discarded
        
    def t_error(self, t):
        print(f"Illegal character '{t.value[0]}' at line {t.lineno}")
        t.lexer.skip(1)
        
    def build(self, **kwargs):
        self.lexer = lex.lex(module=self, **kwargs)
        
    def tokenize(self, data):
        self.lexer.input(data)
        return list(self.lexer)

    def t_BRANCH(self, t):
        r'Branch'
        return t

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