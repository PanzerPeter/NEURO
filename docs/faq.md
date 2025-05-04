# Frequently Asked Questions (FAQ)

**Q: How do I install NEURO?**

A: Clone the repository, create a virtual environment, activate it, and then run `pip install -e .` from the project root directory. See `docs/getting_started.md` or `README.md` for details.

**Q: How do I run a NEURO script?**

A: Once installed, use the command `neuro run path/to/your_script.nr` in your terminal.

**Q: What features are currently implemented?**

A: NEURO is in Alpha. Core features include defining models (Dense, Conv2D, BatchNorm, Flatten layers), loading data (`load_matrix`), configuring loss/optimizers (`Loss`, `Optimizer`), and basic training (`model.train`). See the "Features" section in `README.md` for a more detailed list.

**Q: Why doesn't `model.load("file.pt")` automatically rebuild my model?**

A: Currently, loading only restores the model *weights* (state dictionary). It cannot yet reconstruct the model architecture from the saved file. You must define the exact same model architecture in your `.nr` script *before* attempting to load weights into it. Full model reconstruction is planned for the future.

**Q: How are loss functions and optimizers configured?**

A: You assign the results of `Loss(...)` and `Optimizer(...)` calls to variables (e.g., `loss_cfg = Loss(type="bce");`). The interpreter uses the *most recently defined* loss and optimizer configuration when `model.train()` is executed.

**Q: Is NEURO ready for production use?**

A: No. NEURO is currently in an Alpha stage. The API may change, features are incomplete, and bugs are expected. Use it for experimentation at your own risk.

**Q: Where can I find examples?**

A: Check the `examples/` directory in the repository, the Quick Start section in `README.md`, and the examples within `docs/language_guide.md`.

