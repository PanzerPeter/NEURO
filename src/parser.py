from ply import yacc
from .lexer import NeuroLexer

class NeuroParser:
    def __init__(self):
        self.lexer = NeuroLexer()
        self.tokens = self.lexer.tokens
        
        # Operator precedence to resolve shift/reduce conflicts
        self.precedence = (
            ('right', 'UMINUS'),  # Add unary minus precedence
            ('left', 'PLUS', 'MINUS'),
            ('left', 'TIMES', 'DIVIDE'),
            ('left', 'DOT'),  # Add precedence for method calls
        )
        
        self.parser = yacc.yacc(module=self)

    def p_program(self, p):
        '''program : statement_list'''
        p[0] = ('program', p[1])

    def p_statement_list(self, p):
        '''statement_list : statement
                        | statement_list statement'''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[2]]

    def p_statement(self, p):
        '''statement : assignment
                    | expression
                    | neural_network_stmt
                    | print_stmt
                    | load_matrix_stmt
                    | save_model_stmt
                    | load_model_stmt
                    | layer_stmt
                    | loss_stmt
                    | optimizer_stmt'''
        p[0] = p[1]

    def p_print_stmt(self, p):
        '''print_stmt : PRINT LPAREN expression RPAREN
                     | PRINT LPAREN STRING COMMA expression RPAREN'''
        if len(p) == 5:
            p[0] = ('print', p[3])
        else:
            p[0] = ('print_formatted', p[3], p[5])

    def p_load_matrix_stmt(self, p):
        '''load_matrix_stmt : ID EQUALS LOAD_MATRIX LPAREN STRING RPAREN'''
        p[0] = ('assignment', p[1], ('load_matrix', p[5]))

    def p_save_model_stmt(self, p):
        '''save_model_stmt : SAVE_MODEL LPAREN ID COMMA STRING RPAREN'''
        p[0] = ('save_model', p[3], p[5])

    def p_load_model_stmt(self, p):
        '''load_model_stmt : ID EQUALS LOAD_MODEL LPAREN STRING RPAREN'''
        p[0] = ('assignment', p[1], ('load_model', p[5]))

    def p_neural_network_stmt(self, p):
        '''neural_network_stmt : ID EQUALS NEURAL_NETWORK LPAREN parameters RPAREN
                              | ID EQUALS NEURAL_NETWORK LPAREN parameters RPAREN LBRACE layer_list RBRACE
                              | ID EQUALS NEURAL_NETWORK LPAREN parameters RPAREN LBRACE layer_list config_list RBRACE'''
        if len(p) == 7:
            p[0] = ('neural_network', p[1], p[5], None, None)
        elif len(p) == 10:
            p[0] = ('neural_network', p[1], p[5], p[8], None)
        else:
            p[0] = ('neural_network', p[1], p[5], p[8], p[9])

    def p_layer_list(self, p):
        '''layer_list : layer_stmt
                     | layer_list layer_stmt'''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[2]]

    def p_layer_stmt(self, p):
        '''layer_stmt : DENSE LPAREN parameters RPAREN
                     | CONV2D LPAREN parameters RPAREN
                     | MAXPOOL LPAREN parameters RPAREN
                     | DROPOUT LPAREN parameters RPAREN
                     | FLATTEN LPAREN RPAREN
                     | NORMALIZE LPAREN parameters RPAREN
                     | LSTM LPAREN parameters RPAREN
                     | GRU LPAREN parameters RPAREN
                     | ATTENTION LPAREN parameters RPAREN
                     | EMBEDDING LPAREN parameters RPAREN'''
        if len(p) == 4:
            p[0] = ('layer', p[1], [])
        else:
            p[0] = ('layer', p[1], p[3])

    def p_expression(self, p):
        '''expression : expression DOT ID LPAREN parameters RPAREN
                     | term'''
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = ('method_call', p[1], p[3], p[5])

    def p_term(self, p):
        '''term : NUMBER
                | STRING
                | ID
                | MINUS term %prec UMINUS
                | expression PLUS expression
                | expression MINUS expression
                | expression TIMES expression
                | expression DIVIDE expression'''
        if len(p) == 2:
            if isinstance(p[1], (int, float)):
                p[0] = ('number', p[1])
            elif p.slice[1].type == 'STRING':  # Check token type directly
                p[0] = ('string', p[1])
            else:
                p[0] = ('id', p[1])
        elif len(p) == 3:  # Unary minus
            if p[1] == '-':
                if p[2][0] == 'number':
                    p[0] = ('number', -p[2][1])
                else:
                    p[0] = ('binop', '*', ('number', -1), p[2])
        else:
            p[0] = ('binop', p[2], p[1], p[3])

    def p_parameters(self, p):
        '''parameters : parameter_list
                     | empty'''
        p[0] = p[1] if p[1] is not None else []

    def p_parameter_list(self, p):
        '''parameter_list : parameter
                        | parameter_list COMMA parameter'''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[3]]

    def p_parameter(self, p):
        '''parameter : ID EQUALS expression
                    | expression'''
        if len(p) == 4:
            p[0] = ('named_param', p[1], p[3])
        else:
            p[0] = ('param', p[1])

    def p_assignment(self, p):
        '''assignment : ID EQUALS expression'''
        p[0] = ('assignment', p[1], p[3])

    def p_empty(self, p):
        '''empty :'''
        pass

    def p_error(self, p):
        if p:
            print(f"Syntax error at '{p.value}', line {p.lineno}")
        else:
            print("Syntax error at EOF")

    def parse(self, input_string):
        return self.parser.parse(input_string, lexer=self.lexer.lexer)

    def p_config_list(self, p):
        '''config_list : config_stmt
                      | config_list config_stmt'''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[2]]

    def p_config_stmt(self, p):
        '''config_stmt : LOSS LPAREN parameters RPAREN
                      | OPTIMIZER LPAREN parameters RPAREN'''
        p[0] = ('config', p[1], p[3])

    def p_loss_stmt(self, p):
        '''loss_stmt : LOSS LPAREN parameters RPAREN'''
        p[0] = ('config', 'loss', p[3])

    def p_optimizer_stmt(self, p):
        '''optimizer_stmt : OPTIMIZER LPAREN parameters RPAREN'''
        p[0] = ('config', 'optimizer', p[3])


# Example usage
if __name__ == '__main__':
    # Test input
    test_input = '''
    model = NeuralNetwork(input_size=128, output_size=10)
    model.train(data, epochs=10)
    accuracy = model.evaluate(test_data)
    '''
    
    # Parse input
    parser = NeuroParser()
    ast = parser.parse(test_input)
    print("Abstract Syntax Tree:", ast) 