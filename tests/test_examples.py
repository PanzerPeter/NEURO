import unittest
import subprocess
import sys
import os
from pathlib import Path
import shutil

# Determine project root and other paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # tests/../ -> Neuro1.0/
EXAMPLES_DIR = PROJECT_ROOT / "examples"
TEST_OUTPUT_DIR = PROJECT_ROOT / "tests" / "test_outputs"
COMPILER_MODULE = "neuro.main"  # To be run as 'python -m neuro.main'

class TestExamples(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        if TEST_OUTPUT_DIR.exists():
            shutil.rmtree(TEST_OUTPUT_DIR)

    def test_compile_and_run_all_examples(self):
        example_files = list(EXAMPLES_DIR.glob("*.nr"))
        if not example_files:
            self.skipTest(f"No example files (.nr) found in {EXAMPLES_DIR}")

        for example_file in example_files:
            with self.subTest(example=example_file.name):
                executable_name_base = example_file.stem
                executable_path = TEST_OUTPUT_DIR / executable_name_base
                
                # On Windows, the primary target is .exe, but keep original for scripts
                if sys.platform == "win32":
                    executable_path_with_ext = executable_path.with_suffix('.exe')
                else:
                    executable_path_with_ext = executable_path # No specific extension for Linux/macOS

                # Compile
                compile_cmd = [
                    sys.executable,  # Use the current python interpreter
                    "-m", COMPILER_MODULE,
                    str(example_file),
                    "-o", str(executable_path_with_ext) # Target specific extension for clang/gcc
                ]
                
                # Add debug or optimization flags
                if "CI" in os.environ:  # Less verbose on CI
                    compile_cmd.append("-O2")
                else:  # More debuggable locally
                    compile_cmd.append("--debug")

                try:
                    env = os.environ.copy()
                    # Ensure PYTHONPATH includes the project root for 'neuro.main' module discovery
                    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
                    env["PYTHONUTF8"] = "1" # Force Python in subprocess to use UTF-8

                    completed_compile = subprocess.run(
                        compile_cmd,
                        capture_output=True,
                        text=True,
                        encoding='utf-8', # Specify UTF-8 for decoding stdout/stderr
                        check=False, # Check manually to provide better error messages
                        cwd=PROJECT_ROOT,
                        env=env,
                        timeout=60 # 60 second timeout for compilation
                    )

                    if completed_compile.returncode != 0:
                        self.fail(
                            f"""Compilation failed for {example_file.name} with exit code {completed_compile.returncode}.
Command: {' '.join(compile_cmd)}
Stdout:
{completed_compile.stdout}
Stderr:
{completed_compile.stderr}"""
                        )
                    
                    # The compiler might print success to stdout or stderr depending on verbosity
                    # We are checking for "Compilation successful" which is printed by main.py
                    success_message_found = "Compilation successful" in completed_compile.stdout or \
                                            "Compilation successful" in completed_compile.stderr
                    
                    # If --emit-llvm-ir is used with --debug, it might not print "Compilation successful"
                    # but still exit 0. For this test, we expect actual compilation.
                    if "--emit-llvm-ir" not in compile_cmd and not success_message_found:
                         self.fail(
                            f"""Compiler did not report explicit success for {example_file.name}.
Command: {' '.join(compile_cmd)}
Stdout:
{completed_compile.stdout}
Stderr:
{completed_compile.stderr}"""
                        )

                except subprocess.TimeoutExpired as e:
                    self.fail(
                        f"""Compilation timed out for {example_file.name}.
Command: {' '.join(e.cmd)}
Stdout:
{e.stdout}
Stderr:
{e.stderr}"""
                    )
                except FileNotFoundError:
                     self.fail(f"Compiler command '{compile_cmd[0]}' not found. Ensure Python and the project structure are correct.")


                # Determine which executable/script was actually created
                final_executable_to_run = None
                potential_paths = [executable_path_with_ext] # Main target
                
                # Add script fallbacks based on original stem (e.g. examples/hello_world -> test_outputs/hello_world.bat)
                script_fallback_base = TEST_OUTPUT_DIR / example_file.stem
                if sys.platform == "win32":
                    potential_paths.append(script_fallback_base.with_suffix('.bat'))
                potential_paths.append(script_fallback_base.with_suffix('.sh'))
                
                # In case the output was just the name without extension (e.g. for Unix)
                if sys.platform != "win32" and not executable_path_with_ext.suffix: # e.g. 'output_path' was 'foo'
                     pass # executable_path_with_ext is already 'foo'
                elif executable_path_with_ext != executable_path : # e.g. output_path_with_ext was foo.exe but foo is also possible
                     potential_paths.append(executable_path)


                for p_path in potential_paths:
                    if p_path.exists():
                        final_executable_to_run = p_path
                        break
                
                if not final_executable_to_run:
                    self.fail(f"Compilation reported success (or did not error), but no executable or script found for {example_file.name} at expected paths: {potential_paths}")

                # Run the compiled output
                run_cmd_exec = [str(final_executable_to_run)]
                try:
                    completed_run = subprocess.run(
                        run_cmd_exec,
                        capture_output=True,
                        text=True,
                        encoding='utf-8', # Specify UTF-8 for decoding stdout/stderr
                        check=True, # Will raise CalledProcessError for non-zero exit
                        timeout=10  # 10 second timeout for execution
                    )
                except subprocess.CalledProcessError as e:
                    self.fail(
                        f"""Execution failed for {final_executable_to_run} (from {example_file.name}) with exit code {e.returncode}.
Stdout:
{e.stdout}
Stderr:
{e.stderr}"""
                    )
                except subprocess.TimeoutExpired:
                    self.fail(f"Execution timed out for {final_executable_to_run} (from {example_file.name}).")
                finally:
                    # Clean up individual executables/scripts after their test run
                    # tearDownClass will get any missed ones, but this is cleaner.
                    for p_path in potential_paths:
                         if p_path.exists():
                              p_path.unlink()

if __name__ == "__main__":
    unittest.main() 