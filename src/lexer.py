from ply import lex

class NeuroLexer:
    # List of token names
    tokens = [
        'ID', 'NUMBER', 'STRING',
        'PLUS', 'MINUS', 'TIMES', 'DIVIDE', 'MODULO', 'POWER',
        'EQUALS', 'EQUALS_EQUALS', 'NOT_EQUALS',
        'LESS_THAN', 'GREATER_THAN', 'LESS_THAN_EQUALS', 'GREATER_THAN_EQUALS',
        'LPAREN', 'RPAREN', 'LBRACE', 'RBRACE', 'LBRACKET', 'RBRACKET',
        'COMMA', 'DOT', 'AT', 'SEMI',
        'AND', 'OR', 'NOT',
        'NEURAL_NETWORK', 'DENSE', 'CONV2D', 'MAXPOOL', 'DROPOUT',
        'FLATTEN', 'NORMALIZE', 'LSTM', 'GRU', 'ATTENTION', 'EMBEDDING',
        'SAVE_MATRIX', 'LOAD_MATRIX', 'SAVE_MODEL', 'LOAD_MODEL',
        'LOSS', 'OPTIMIZER', 'DEF', 'TRAIN', 'PREDICT', 'EVALUATE',
        'TRUE', 'FALSE', 'NONE', 'PRINT', 'PRETRAINED', 'CUSTOM_LAYER',
        'BRANCH', 'RETURN', 'FOR', 'IN', 'RANGE',
        'UMINUS'
    ]
    
    # Regular expression rules for simple tokens
    t_PLUS = r'\+'
    t_MINUS = r'-'
    t_TIMES = r'\*'
    t_DIVIDE = r'/'
    t_MODULO = r'%'
    t_POWER = r'\*\*'
    t_LPAREN = r'\('
    t_RPAREN = r'\)'
    t_LBRACE = r'\{'
    t_RBRACE = r'\}'
    t_EQUALS = r'='
    t_EQUALS_EQUALS = r'=='
    t_NOT_EQUALS = r'!='
    t_LESS_THAN = r'<'
    t_GREATER_THAN = r'>'
    t_LESS_THAN_EQUALS = r'<='
    t_GREATER_THAN_EQUALS = r'>='
    t_COMMA = r','
    t_DOT = r'\.'
    t_AT = r'@'
    t_SEMI = r';'
    
    # Keywords with case-sensitive matching
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
        'NeuralNetwork': 'NEURAL_NETWORK',
        'custom_layer': 'CUSTOM_LAYER',
        'pretrained': 'PRETRAINED',
        'def': 'DEF',
        'return': 'RETURN',
        'for': 'FOR',
        'in': 'IN',
        'range': 'RANGE',
        'True': 'TRUE',
        'False': 'FALSE',
        'None': 'NONE',
        'print': 'PRINT',
        'and': 'AND',
        'or': 'OR',
        'not': 'NOT'
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
        r'\"([^\"\\]|\\.)*\"|\'([^\'\\]|\\.)*\''
        t.value = t.value[1:-1]  # Remove quotes
        return t
        
    def t_COMMENT(self, t):
        r'\#[^\n]*'  # Match comments starting with # until end of line
        pass  # Discard comments
        
    def t_newline(self, t):
        r'\n+'
        t.lexer.lineno += len(t.value)
        
    t_ignore = ' \t'  # Ignore spaces and tabs
    
    def t_error(self, t):
        print(f"Illegal character '{t.value[0]}' at line {t.lexer.lineno}")
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