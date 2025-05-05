# NEURO Best Practices (Alpha)

As NEURO is in an early stage (Alpha), these are initial suggestions rather than strict rules. Best practices will evolve as the language matures.

- **Use Clear Variable Names:** Choose descriptive names for models, data matrices, and configurations.
- **Comment Your Code:** Especially for complex model architectures or non-obvious processing steps.
- **Configure Before Training:** Define your `Loss(...)` and `Optimizer(...)` configurations clearly *before* the corresponding `model.train(...)` call.
- **Leverage the Type Checker:** Run scripts with the `--use-real` flag (`python src/neuro.py run your_script.nr --use-real`) to enable static type checking. Address any reported type errors, as they often indicate potential runtime issues like incompatible layer shapes or incorrect function usage.
- **Organize Data:** Structure your `.nrm` data files logically with appropriate metadata.
- **Check Documentation:** Refer to `README.md` and `language_guide.md` for the latest syntax and supported features.
- **Provide Feedback:** Since it's Alpha, report issues or suggestions on the project's issue tracker.

