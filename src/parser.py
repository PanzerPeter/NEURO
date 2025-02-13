import ply.lex as lex
import ply.yacc as yacc
from .lexer import NeuroLexer

class NeuroParser:
    tokens = [
        'ID', 'NUMBER', 'STRING',
        'PLUS', 'MINUS', 'TIMES', 'DIVIDE', 'MODULO', 'POWER',
        'EQUALS', 'EQUALS_EQUALS', 'NOT_EQUALS',
        'LESS_THAN', 'GREATER_THAN', 'LESS_THAN_EQUALS', 'GREATER_THAN_EQUALS',
        'LPAREN', 'RPAREN', 'LBRACE', 'RBRACE', 'LBRACKET', 'RBRACKET',
        'SEMI', 'COMMA', 'DOT', 'AT',
        'AND', 'OR', 'NOT',
        'NEURAL_NETWORK', 'DENSE', 'CONV2D', 'MAXPOOL', 'DROPOUT',
        'FLATTEN', 'NORMALIZE', 'LSTM', 'GRU', 'ATTENTION', 'EMBEDDING',
        'SAVE_MATRIX', 'LOAD_MATRIX', 'SAVE_MODEL', 'LOAD_MODEL',
        'LOSS', 'OPTIMIZER', 'DEF', 'TRAIN', 'PREDICT', 'EVALUATE',
        'TRUE', 'FALSE', 'NONE', 'PRINT', 'PRETRAINED', 'CUSTOM_LAYER',
        'BRANCH', 'RETURN', 'FOR', 'IN', 'RANGE',
        'UMINUS'
    ]

    reserved = {
        'and': 'AND',
        'or': 'OR',
        'not': 'NOT',
        'neural_network': 'NEURAL_NETWORK',
        'dense': 'DENSE',
        'conv2d': 'CONV2D',
        'maxpool': 'MAXPOOL',
        'dropout': 'DROPOUT',
        'flatten': 'FLATTEN',
        'normalize': 'NORMALIZE',
        'lstm': 'LSTM',
        'gru': 'GRU',
        'attention': 'ATTENTION',
        'embedding': 'EMBEDDING',
        'save_matrix': 'SAVE_MATRIX',
        'load_matrix': 'LOAD_MATRIX',
        'save_model': 'SAVE_MODEL',
        'load_model': 'LOAD_MODEL',
        'loss': 'LOSS',
        'optimizer': 'OPTIMIZER',
        'def': 'DEF',
        'train': 'TRAIN',
        'predict': 'PREDICT',
        'evaluate': 'EVALUATE',
        'True': 'TRUE',
        'False': 'FALSE',
        'None': 'NONE',
        'print': 'PRINT',
        'pretrained': 'PRETRAINED',
        'custom_layer': 'CUSTOM_LAYER',
        'branch': 'BRANCH',
        'return': 'RETURN',
        'for': 'FOR',
        'in': 'IN',
        'range': 'RANGE'
    }

    t_PLUS = r'\+'
    t_MINUS = r'-'
    t_TIMES = r'\*'
    t_DIVIDE = r'/'
    t_MODULO = r'%'
    t_POWER = r'\*\*'
    t_EQUALS = r'='
    t_EQUALS_EQUALS = r'=='
    t_NOT_EQUALS = r'!='
    t_LESS_THAN = r'<'
    t_GREATER_THAN = r'>'
    t_LESS_THAN_EQUALS = r'<='
    t_GREATER_THAN_EQUALS = r'>='
    t_LPAREN = r'\('
    t_RPAREN = r'\)'
    t_LBRACE = r'\{'
    t_RBRACE = r'\}'
    t_LBRACKET = r'\['
    t_RBRACKET = r'\]'
    t_COMMA = r','
    t_DOT = r'\.'
    t_SEMI = r';'
    t_COLON = r':'
    t_AT = r'@'

    def t_ID(self, t):
        r'[a-zA-Z_][a-zA-Z_0-9]*'
        t.type = self.reserved.get(t.value, 'ID')
        return t

    def t_NUMBER(self, t):
        r'\d*\.\d+|\d+'
        t.value = float(t.value)
        return t

    def t_STRING(self, t):
        r'\"([^\"\\]|\\.)*\"|\'([^\'\\]|\\.)*\''
        t.value = t.value[1:-1]  # Remove quotes
        return t

    t_ignore = ' \t\n'

    def t_error(self, t):
        print(f"Illegal character '{t.value[0]}'")
        t.lexer.skip(1)

    precedence = (
        ('left', 'OR'),
        ('left', 'AND'),
        ('left', 'EQUALS_EQUALS', 'NOT_EQUALS'),
        ('left', 'LESS_THAN', 'GREATER_THAN', 'LESS_THAN_EQUALS', 'GREATER_THAN_EQUALS'),
        ('left', 'PLUS', 'MINUS'),
        ('left', 'TIMES', 'DIVIDE'),
        ('nonassoc', 'UMINUS'),
    )

    def __init__(self):
        self.lexer = NeuroLexer()
        self.lexer.build()
        self.parser = yacc.yacc(module=self)
        self.error = False  # Initialize error flag

    def parse(self, input_string):
        self.error = False  # Reset error flag
        result = self.parser.parse(input_string, lexer=self.lexer.lexer)
        if self.error:
            raise Exception("Parsing failed due to syntax errors")
        return result

    def p_program(self, p):
        '''program : statement_list
                  | empty'''
        p[0] = ('program', p[1] if p[1] else [])

    def p_statement_list(self, p):
        '''
        statement_list : statement
                      | statement_list statement
                      | empty
        '''
        if len(p) == 2:
            if p[1] is None:
                p[0] = []
            else:
                p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[2]]

    def p_statement(self, p):
        '''
        statement : expression SEMI
                 | assignment SEMI
                 | neural_network_stmt
                 | loss_stmt
                 | optimizer_stmt
                 | custom_layer_stmt
                 | save_model_stmt
                 | load_model_stmt
                 | save_matrix_stmt
                 | load_matrix_stmt
                 | train_stmt
                 | predict_stmt
                 | evaluate_stmt
                 | print_stmt
                 | return_stmt
                 | for_loop
                 | layer_def SEMI
                 | empty
        '''
        p[0] = p[1]

    def p_print_stmt(self, p):
        '''
        print_stmt : PRINT LPAREN expression RPAREN SEMI
        '''
        p[0] = ('print', p[3])

    def p_load_matrix_stmt(self, p):
        '''
        load_matrix_stmt : ID EQUALS LOAD_MATRIX STRING
        '''
        p[0] = ('load_matrix', p[1], p[4])

    def p_save_model_stmt(self, p):
        '''
        save_model_stmt : SAVE_MODEL ID STRING
        '''
        p[0] = ('save_model', p[2], p[3])

    def p_load_model_stmt(self, p):
        '''
        load_model_stmt : ID EQUALS LOAD_MODEL STRING
        '''
        p[0] = ('load_model', p[1], p[4])

    def p_save_matrix_stmt(self, p):
        '''
        save_matrix_stmt : SAVE_MATRIX ID STRING SEMI
        '''
        p[0] = ('save_matrix', p[2], p[3])

    def p_neural_network_stmt(self, p):
        '''
        neural_network_stmt : ID EQUALS NEURAL_NETWORK LPAREN parameter_list RPAREN neural_block
                          | ID EQUALS NEURAL_NETWORK LPAREN parameter_list RPAREN SEMI
        '''
        if len(p) == 8:
            layers = p[7] if isinstance(p[7], list) else []
            p[0] = ('assignment', p[1], ('neural_network', p[5], layers))
        else:
            p[0] = ('assignment', p[1], ('neural_network', p[5], []))

    def p_neural_block(self, p):
        '''
        neural_block : LBRACE layer_list RBRACE
                    | LBRACE RBRACE
        '''
        if len(p) == 4:
            p[0] = p[2] if p[2] else []
        else:
            p[0] = []

    def p_layer_list(self, p):
        '''
        layer_list : layer_def SEMI
                  | layer_list layer_def SEMI
                  | empty
        '''
        if len(p) == 2:  # Empty case
            p[0] = []
        elif len(p) == 3:  # Single layer with semicolon
            p[0] = [p[1]]
        else:  # Multiple layers
            p[0] = p[1] + [p[2]]

    def p_layer_def(self, p):
        '''
        layer_def : layer_type LPAREN parameter_list RPAREN
                 | layer_type LPAREN RPAREN
                 | ID LPAREN parameter_list RPAREN
                 | ID LPAREN RPAREN
        '''
        if len(p) == 5:  # With parameters
            layer_name = p[1].lower() if isinstance(p[1], str) else p[1]
            p[0] = ('layer', layer_name, p[3])
        else:  # Without parameters
            layer_name = p[1].lower() if isinstance(p[1], str) else p[1]
            p[0] = ('layer', layer_name, [])

    def p_layer_type(self, p):
        '''
        layer_type : DENSE
                  | CONV2D
                  | MAXPOOL
                  | DROPOUT
                  | FLATTEN
                  | NORMALIZE
                  | LSTM
                  | GRU
                  | ATTENTION
                  | EMBEDDING
        '''
        p[0] = p[1]

    def p_parameter_list(self, p):
        '''
        parameter_list : named_param
                      | parameter_list COMMA named_param
                      | expression
                      | parameter_list COMMA expression
                      | empty
        '''
        if len(p) == 2:
            if p[1] is None:  # Empty case
                p[0] = []
            else:  # Single parameter
                p[0] = [p[1]]
        else:  # Multiple parameters
            p[0] = p[1] + [p[3]]

    def p_named_param(self, p):
        '''
        named_param : ID EQUALS expression
                   | ID EQUALS STRING
                   | ID EQUALS NUMBER
        '''
        if isinstance(p[3], str) and p[3].startswith('"') and p[3].endswith('"'):
            # Remove quotes from string literals
            p[0] = ('named_param', p[1], ('string', p[3][1:-1]))
        elif isinstance(p[3], (int, float)):
            p[0] = ('named_param', p[1], ('number', float(p[3])))
        else:
            p[0] = ('named_param', p[1], p[3])

    def p_assignment(self, p):
        '''assignment : ID EQUALS expression'''
        p[0] = ('assignment', p[1], p[3])

    def p_empty(self, p):
        '''
        empty :
        '''
        p[0] = None

    def p_error(self, p):
        if p:
            print(f"Syntax error at '{p.value}', line {p.lineno}")
        else:
            print("Syntax error at EOF")
        self.error = True
        
        # Set error flag and try to recover
        if self.parser:
            while True:
                tok = self.parser.token()
                if not tok or tok.type in ['RPAREN', 'RBRACE', 'SEMI']:
                    break
            self.parser.restart()

    def p_config_list(self, p):
        '''
        config_list : config_stmt
                   | config_list config_stmt
        '''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[2]]

    def p_config_stmt(self, p):
        '''
        config_stmt : loss_stmt
                   | optimizer_stmt
        '''
        p[0] = p[1]

    def p_loss_stmt(self, p):
        '''
        loss_stmt : LOSS LPAREN parameter_list RPAREN SEMI
                 | LOSS LPAREN RPAREN SEMI
        '''
        if len(p) == 6:  # With parameters
            p[0] = ('loss', p[3])
        else:  # Without parameters
            p[0] = ('loss', [])

    def p_optimizer_stmt(self, p):
        '''
        optimizer_stmt : OPTIMIZER LPAREN parameter_list RPAREN SEMI
                     | OPTIMIZER LPAREN RPAREN SEMI
        '''
        if len(p) == 6:  # With parameters
            p[0] = ('optimizer', p[3])
        else:  # Without parameters
            p[0] = ('optimizer', [])

    def p_decorated_statement(self, p):
        '''
        decorated_statement : AT decorator_expr DEF ID LPAREN parameters RPAREN block
                          | AT decorator_expr DEF ID LPAREN RPAREN block
                          | AT decorator_expr DEF ID LPAREN ID RPAREN block
                          | AT decorator_expr branch_def
        '''
        if len(p) == 9:
            if isinstance(p[6], str):  # Single parameter case
                p[0] = ('custom_layer', p[4], p[6], p[8])
            else:  # Parameters list case
                p[0] = ('custom_layer', p[4], p[6], p[8])
        elif len(p) == 8:  # No parameters case
            p[0] = ('custom_layer', p[4], [], p[7])
        else:  # Branch case
            p[0] = ('decorated', p[2], p[4])

    def p_decorator_expr(self, p):
        '''
        decorator_expr : PRETRAINED LPAREN STRING RPAREN
                     | CUSTOM_LAYER
                     | ID
                     | ID LPAREN parameters RPAREN
                     | ID LPAREN RPAREN
        '''
        if len(p) == 5:
            if p[1] == 'pretrained':
                p[0] = ('decorator', p[1], p[3])
            else:
                p[0] = ('decorator', p[1], p[3])
        elif len(p) == 2:
            p[0] = ('decorator', p[1], None)
        elif len(p) == 4:
            p[0] = ('decorator', p[1], [])
        else:
            p[0] = ('decorator', p[1], p[3])

    def p_decorator_list(self, p):
        '''
        decorator_list : AT CUSTOM_LAYER
                      | AT PRETRAINED LPAREN STRING RPAREN
                      | AT PRETRAINED
        '''
        if len(p) == 3:  # @custom_layer or @pretrained
            p[0] = [('decorator', p[2].lower(), None)]
        elif len(p) == 6:  # @pretrained("model_name")
            p[0] = [('decorator', p[2].lower(), p[4])]
        else:
            p[0] = [('decorator', p[2].lower(), None)]

    def p_custom_layer_stmt(self, p):
        '''
        custom_layer_stmt : decorator_list function_def
        '''
        # Ensure p[1] is a list and contains at least one decorator
        decorators = p[1] if isinstance(p[1], list) else [p[1]]
        if not decorators or not isinstance(decorators[0], tuple):
            decorators = [decorators]
        # Return just the decorator tuple, not wrapped in a list
        p[0] = ('decorated', decorators[0], p[2])

    def p_function_def(self, p):
        '''
        function_def : DEF ID LPAREN parameter_list RPAREN neural_block
                    | DEF ID LPAREN RPAREN neural_block
                    | DEF ID LPAREN ID RPAREN neural_block
        '''
        if len(p) == 7:  # With parameters
            if isinstance(p[4], str):  # Single ID parameter
                p[0] = ('custom_layer', p[2], [p[4]], p[6])
            else:  # Parameter list
                p[0] = ('custom_layer', p[2], p[4], p[6])
        else:  # Without parameters
            p[0] = ('custom_layer', p[2], [], p[5])

    def p_branch_def(self, p):
        '''
        branch_def : BRANCH ID LPAREN parameter_list RPAREN neural_block
                  | BRANCH ID LPAREN RPAREN neural_block
                  | BRANCH ID neural_block
        '''
        if len(p) == 7:  # With parameters
            p[0] = ('branch', p[2], p[4], p[6])
        elif len(p) == 6:  # Empty parameters
            p[0] = ('branch', p[2], [], p[5])
        else:  # No parameters
            p[0] = ('branch', p[2], [], p[3])

    def p_return_stmt(self, p):
        '''
        return_stmt : RETURN expression SEMI
        '''
        p[0] = ('return', p[2])

    def p_for_loop(self, p):
        '''
        for_loop : FOR ID IN RANGE LPAREN expression RPAREN neural_block
        '''
        p[0] = ('for', p[2], p[6], p[8])

    def p_method_call(self, p):
        '''
        method_call : ID DOT ID LPAREN parameter_list RPAREN
                   | ID DOT ID LPAREN RPAREN
                   | ID LPAREN parameter_list RPAREN
                   | ID LPAREN RPAREN
        '''
        if len(p) == 7:  # object.method(params)
            p[0] = ('method_call', ('id', p[1]), p[3], p[5])
        elif len(p) == 6:  # object.method()
            p[0] = ('method_call', ('id', p[1]), p[3], [])
        elif len(p) == 5:  # function(params)
            p[0] = ('method_call', None, p[1], p[3])
        else:  # function()
            p[0] = ('method_call', None, p[1], [])

    def p_expression(self, p):
        '''
        expression : NUMBER
                   | STRING
                   | TRUE
                   | FALSE
                   | NONE
                   | ID
                   | method_call
                   | LPAREN expression RPAREN
                   | MINUS expression %prec UMINUS
                   | NOT expression
                   | expression PLUS expression
                   | expression MINUS expression
                   | expression TIMES expression
                   | expression DIVIDE expression
                   | expression MODULO expression
                   | expression POWER expression
                   | expression AND expression
                   | expression OR expression
                   | expression EQUALS_EQUALS expression
                   | expression NOT_EQUALS expression
                   | expression LESS_THAN expression
                   | expression GREATER_THAN expression
                   | expression LESS_THAN_EQUALS expression
                   | expression GREATER_THAN_EQUALS expression
        '''
        if len(p) == 2:
            if isinstance(p[1], float):
                p[0] = ('number', p[1])
            elif isinstance(p[1], str):
                if p[1] == 'True':
                    p[0] = ('bool', True)
                elif p[1] == 'False':
                    p[0] = ('bool', False)
                elif p[1] == 'None':
                    p[0] = ('none', None)
                else:
                    if p.slice[1].type == 'STRING':
                        p[0] = ('string', p[1])  # String value already without quotes
                    else:
                        p[0] = ('id', p[1])
            else:
                p[0] = p[1]  # For method_call
        elif len(p) == 3:
            if p[1] == 'not':
                p[0] = ('not', p[2])
            else:  # MINUS expression
                if p[2][0] == 'number':
                    p[0] = ('number', -p[2][1])
                else:
                    p[0] = ('neg', p[2])
        elif len(p) == 4:
            if p[1] == '(':
                p[0] = p[2]
            else:
                p[0] = ('binop', p[2], p[1], p[3])
        elif len(p) == 5:  # ID LPAREN RPAREN
            p[0] = ('method_call', ('id', p[1]), [])
        elif len(p) == 6:  # ID LPAREN parameter_list RPAREN
            p[0] = ('method_call', ('id', p[1]), p[3])
        else:  # ID DOT ID LPAREN parameter_list RPAREN
            p[0] = ('method_call', ('id', p[3]), p[5], ('id', p[1]))

    def p_term(self, p):
        '''
        term : factor
             | MINUS NUMBER
             | MINUS ID
             | NOT factor
        '''
        if len(p) == 2:
            p[0] = p[1]
        elif p[1] == '-':
            if isinstance(p[2], (int, float)):
                p[0] = ('number', -p[2])
            else:
                p[0] = ('unary_op', p[1], ('id', p[2]))
        else:
            p[0] = ('unary_op', p[1], p[2])

    def p_factor(self, p):
        '''
        factor : NUMBER
               | STRING
               | TRUE
               | FALSE
               | NONE
               | ID
               | LPAREN expression RPAREN
               | LBRACKET list_items RBRACKET
               | LBRACKET RBRACKET
               | method_call
               | ID DOT ID
        '''
        if len(p) == 2:
            if isinstance(p[1], (int, float)):
                p[0] = ('number', p[1])
            elif isinstance(p[1], str):
                if p[1] == 'True':
                    p[0] = ('boolean', True)
                elif p[1] == 'False':
                    p[0] = ('boolean', False)
                elif p[1] == 'None':
                    p[0] = ('none',)
                elif p[1].startswith('"') or p[1].startswith("'"):
                    p[0] = ('string', p[1][1:-1])  # Remove quotes
                else:
                    p[0] = ('id', p[1])
        elif len(p) == 4:
            if p[1] == '(':
                p[0] = p[2]
            elif p[1] == '[':
                p[0] = ('list', p[2])
            else:
                p[0] = ('attribute', p[1], p[3])
        elif len(p) == 3:
            p[0] = ('list', [])

    def p_list_items(self, p):
        '''
        list_items : expression
                  | list_items COMMA expression
                  | empty
        '''
        if len(p) == 2:
            if p[1] is None:
                p[0] = []
            else:
                p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[3]]

    def p_parameters(self, p):
        '''
        parameters : parameter_list
                  | empty
        '''
        p[0] = p[1] if p[1] else []

    def p_block(self, p):
        '''
        block : LBRACE statement_list RBRACE
             | LBRACE RBRACE
        '''
        if len(p) == 4:
            statements = []
            for stmt in (p[2] if isinstance(p[2], list) else [p[2]]):
                if stmt is not None:
                    statements.append(stmt)
            p[0] = statements
        else:
            p[0] = []

    def p_train_stmt(self, p):
        '''
        train_stmt : ID DOT TRAIN LPAREN parameter_list RPAREN SEMI
                  | ID DOT TRAIN LPAREN RPAREN SEMI
        '''
        if len(p) == 8:  # With parameters
            p[0] = ('train', p[1], p[5])
        else:  # Without parameters
            p[0] = ('train', p[1], [])

    def p_predict_stmt(self, p):
        '''
        predict_stmt : ID DOT PREDICT LPAREN parameter_list RPAREN SEMI
                    | ID DOT PREDICT LPAREN RPAREN SEMI
        '''
        if len(p) == 8:  # With parameters
            p[0] = ('predict', p[1], p[5])
        else:  # Without parameters
            p[0] = ('predict', p[1], [])

    def p_evaluate_stmt(self, p):
        '''
        evaluate_stmt : ID DOT EVALUATE LPAREN parameter_list RPAREN SEMI
                     | ID DOT EVALUATE LPAREN RPAREN SEMI
        '''
        if len(p) == 8:  # With parameters
            p[0] = ('evaluate', p[1], p[5])
        else:  # Without parameters
            p[0] = ('evaluate', p[1], [])

    def p_expression_list(self, p):
        '''
        expression_list : expression
                      | expression_list COMMA expression
        '''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[3]]


# Example usage
if __name__ == '__main__':
    # Test input
    test_input = '''
    model = NeuralNetwork(input_size=128, output_size=10);
    model.train(data, epochs=10);
    accuracy = model.evaluate(test_data);
    '''
    
    # Parse input
    parser = NeuroParser()
    ast = parser.parse(test_input)
    print("Abstract Syntax Tree:", ast) 