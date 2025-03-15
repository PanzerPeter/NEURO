"""
NEURO Language Main Entry Point

This is the main entry point for the NEURO language.
It provides the command-line interface and high-level language functionality.

Key responsibilities:
- Command-line argument parsing
- File loading and execution
- REPL implementation
- Error handling and reporting
- Version and help information
- Environment setup
- Integration of all language components

This is the main interface for users of the NEURO language.
"""

import sys
import os
import click
from typing import Optional
from .lexer import NeuroLexer
from .parser import NeuroParser
from .interpreter import NeuroInterpreter
from .errors import NeuroError, create_error_message

class Neuro:
    """Main NEURO language class."""
    
    def __init__(self):
        self.lexer = NeuroLexer()
        self.parser = NeuroParser()
        self.interpreter = NeuroInterpreter()
    
    def run_file(self, file_path: str) -> None:
        """Run a NEURO program from a file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not file_path.endswith('.nr'):
            raise ValueError("File must have .nr extension")
        
        with open(file_path, 'r') as f:
            source = f.read()
        
        try:
            # Parse and execute
            ast = self.parser.parse(source)
            self.interpreter.interpret(ast)
        except NeuroError as e:
            print(create_error_message(e, source))
            sys.exit(1)
        except Exception as e:
            print(f"Internal error: {str(e)}")
            sys.exit(1)
    
    def run_repl(self) -> None:
        """Run the NEURO REPL (interactive shell)."""
        print(f"NEURO Language v{self.version()} REPL")
        print("Type 'exit' to exit, 'help' for help.")
        
        while True:
            try:
                # Get input
                source = input('>>> ')
                
                # Handle special commands
                if source.lower() == 'exit':
                    break
                elif source.lower() == 'help':
                    self.print_help()
                    continue
                
                # Parse and execute
                ast = self.parser.parse(source)
                result = self.interpreter.interpret(ast)
                
                # Print result if not None
                if result is not None:
                    print(result)
            
            except NeuroError as e:
                print(create_error_message(e, source))
            except Exception as e:
                print(f"Internal error: {str(e)}")
    
    def version(self) -> str:
        """Get NEURO version."""
        from . import __version__
        return __version__
    
    def print_help(self) -> None:
        """Print REPL help information."""
        help_text = """
NEURO Language Help
------------------
Commands:
  exit          Exit the REPL
  help          Show this help message

Basic Syntax:
  # Neural Network Definition
  model = NeuralNetwork(input_size=784, output_size=10) {
      Dense(units=128, activation="relu");
      Dropout(rate=0.2);
      Dense(units=10, activation="softmax");
  }

  # Data Loading
  data = load_matrix("data.nrm");
  train_data, val_data = data.split(0.8);

  # Training
  model.train(
      data=train_data,
      validation_data=val_data,
      epochs=10
  );
"""
        print(help_text)

# Command Line Interface
@click.group()
def cli():
    """NEURO programming language CLI."""
    pass

@cli.command()
@click.argument('file', type=click.Path(exists=True))
def run(file: str):
    """Run a NEURO program."""
    neuro = Neuro()
    neuro.run_file(file)

@cli.command()
def repl():
    """Start the NEURO REPL."""
    neuro = Neuro()
    neuro.run_repl()

@cli.command()
def version():
    """Show NEURO version."""
    neuro = Neuro()
    print(f"NEURO v{neuro.version()}")

def main():
    """Main entry point."""
    cli() 