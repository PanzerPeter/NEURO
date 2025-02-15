"""
NEURO REPL Module
Provides interactive development environment for NEURO.
"""

import cmd
import sys
import torch
from typing import Optional, Dict, Any, List
from .interpreter import NeuroInterpreter
from .errors import NeuroError
from tabulate import tabulate
import torch.nn as nn
import graphviz
import os

class NeuroREPL(cmd.Cmd):
    intro = 'Welcome to NEURO REPL. Type help or ? to list commands.'
    prompt = '>>> '

    def __init__(self):
        super().__init__()
        self.interpreter = NeuroInterpreter()
        self.variables = {}
        self.history = []
        self.last_result = None
        self.debug_mode = False
        self.watch_vars = set()
        self.profiling = False
        self.completions = {
            'model.': ['train', 'evaluate', 'predict', 'save', 'load'],
            'Dense': ['units', 'activation', 'input_shape'],
            'Conv2D': ['filters', 'kernel_size', 'strides', 'padding'],
            'LSTM': ['units', 'return_sequences', 'activation'],
            'optimizer.': ['learning_rate', 'momentum', 'decay'],
            'loss.': ['categorical_crossentropy', 'binary_crossentropy', 'mse']
        }

    def get_completions(self, text: str) -> List[str]:
        """Get code completions for the given text."""
        if '.' in text:
            obj, _ = text.split('.')
            return self.completions.get(obj + '.', [])
        return [k[:-1] for k in self.completions.keys() if k.startswith(text)]

    def add_to_history(self, cmd: str) -> None:
        """Add a command to history."""
        if cmd.strip():
            self.history.append(cmd)

    def handle_input(self, code: str) -> Any:
        """Handle user input."""
        try:
            self.history.append(code)
            result = self.interpreter.interpret(code)
            
            # Update state
            if result is not None:
                self.last_result = result
                self.variables['_'] = result
                
                # If the input was an assignment, store the variable
                if isinstance(result, tuple) and result[0] == 'assignment':
                    var_name = result[1]
                    var_value = result[2]
                    self.variables[var_name] = var_value
                    return f"Created {type(var_value).__name__}"
                
            return result
        except Exception as e:
            print(f"Error:\n{str(e)}")
            return None

    def handle_command(self, command: str) -> str:
        """Handle REPL commands."""
        output = []
        if command == "help":
            output.extend([
                "\nAvailable commands:",
                "------------------",
                "help          : Show this help message",
                "help <topic>  : Show help for a specific topic",
                "clear         : Clear the screen and reset environment",
                "vars          : List all variables",
                "reset         : Reset the environment",
                "exit          : Exit the REPL",
                "\nCode Execution:",
                "--------------",
                "- Type NEURO code directly to execute",
                "- Multi-line code blocks are supported",
                "- Use '=' for assignment",
                "- Access previous results with '_'",
                "\nDebugging:",
                "---------",
                "debug        : Toggle debug mode",
                "watch        : Add variables to watch",
                "profile     : Toggle profiling"
            ])
        elif command.startswith("help "):
            topic = command[5:].strip()
            help_text = self.show_topic_help(topic)
            output.extend(help_text if help_text else [f"No help available for '{topic}'"])
        elif command == "vars":
            vars_list = [f"{name}: {type(value).__name__}" for name, value in self.variables.items()]
            output.extend(["Variables:"] + vars_list if vars_list else ["No variables defined"])
        elif command == "clear":
            self.clear_environment()
            output.append("Environment cleared.")
        elif command == "reset":
            self.reset_environment()
            output.append("Environment reset.")
        elif command == "debug":
            self.debug_mode = not self.debug_mode
            output.append(f"Debug mode {'enabled' if self.debug_mode else 'disabled'}")
        elif command.startswith('watch '):
            var = command[6:].strip()
            self.watch_vars.add(var)
            output.append(f"Watching variable: {var}")
        elif command == 'profile':
            self.profiling = not self.profiling
            output.append(f"Profiling {'enabled' if self.profiling else 'disabled'}")
        elif command.startswith('summary '):
            model_name = command[8:].strip()
            if model_name in self.variables:
                model = self.variables[model_name]
                output.append(str(model))
                output.append("\nModel Summary")
                output.append("=" * 50)
                output.append(self.get_model_summary(model))
            else:
                output.append(f"Model '{model_name}' not found")
        else:
            output.append(f"Unknown command: {command}")
        
        return "\n".join(output)

    def show_variables(self) -> List[str]:
        """Display all variables in the environment."""
        headers = ["Name", "Type", "Value"]
        rows = []
        for name, value in self.variables.items():
            rows.append([name, type(value).__name__, value])
        table = tabulate(rows, headers=headers, tablefmt="grid")
        return [table]

    def show_topic_help(self, topic: str) -> List[str]:
        """Show help for a specific topic."""
        help_texts = {
            'Dense': [
                "Dense Layer",
                "-----------",
                "A fully connected neural network layer.",
                "",
                "Parameters:",
                "  units: Number of output units",
                "  activation: Activation function (e.g., 'relu', 'sigmoid')",
                "  use_bias: Whether to include a bias term (default: True)"
            ],
            'NeuralNetwork': [
                "Neural Network",
                "-------------",
                "Creates a new neural network model.",
                "",
                "Parameters:",
                "  input_size: Size of input features",
                "  output_size: Size of output",
                "  name: Optional name for the model"
            ]
        }
        return help_texts.get(topic, [])

    def clear_environment(self) -> None:
        """Clear the screen and reset environment."""
        os.system('cls' if os.name == 'nt' else 'clear')
        self.reset_environment()

    def reset_environment(self) -> None:
        """Reset the REPL environment."""
        self.interpreter = NeuroInterpreter()
        self.variables = {}
        self.last_result = None
        self.debug_mode = False
        self.watch_vars = set()
        print("Environment reset.")

    def default(self, line: str) -> bool:
        """Handle NEURO code execution."""
        try:
            self.last_result = self.interpreter.interpret(line)
            if self.last_result is not None:
                print(self.last_result)
            return False
        except NeuroError as e:
            print(f"Error: {str(e)}")
            return False

    def get_model_summary(self, model) -> str:
        """Generate a summary of the model architecture."""
        if not hasattr(model, 'parameters'):
            return "Not a valid neural network model"
        
        total_params = sum(p.numel() for p in model.parameters())
        layers = []
        
        for idx, (name, module) in enumerate(model.named_children()):
            params = sum(p.numel() for p in module.parameters())
            shape = "?" # We would need forward pass to determine this
            layers.append(f"{idx:<8} {type(module).__name__:<12} {shape:<16} {params:>9}")
        
        summary = [
            "+---------+------------+----------------+-----------+",
            "|   Layer | Type       | Output Shape   |   Param # |",
            "+=========+============+================+===========+",
        ]
        
        for layer in layers:
            summary.append(f"| {layer} |")
            summary.append("+---------+------------+----------------+-----------+")
            
        summary.append(f"Total parameters: {total_params}")
        return "\n".join(summary)

    def run(self) -> None:
        """Run the REPL."""
        print("NEURO REPL v1.0")
        print('Type "help" for more information.')
        
        while True:
            try:
                code = input(">>> ")
                if code.strip() == "exit":
                    break
                elif code.startswith(":"):
                    self.handle_command(code[1:].strip())
                else:
                    result = self.handle_input(code)
                    if result:
                        print(result)
            except KeyboardInterrupt:
                print("\nKeyboardInterrupt")
            except EOFError:
                break
        
        print("\nGoodbye!")

def main():
    """Start the NEURO REPL."""
    try:
        NeuroREPL().cmdloop()
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)

if __name__ == '__main__':
    main() 