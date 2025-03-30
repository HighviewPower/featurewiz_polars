# ./CONTRIBUTING.md (Updated)

# Contributing to featurewiz_polars

First off, thank you for considering contributing! We welcome contributions from everyone. Whether it's reporting a bug, proposing a feature, or writing code, your help is appreciated.

## How Can I Contribute?

### Reporting Bugs
*   Ensure the bug was not already reported by searching on GitHub under [Issues](https://github.com/AutoViML/featurewiz_polars/issues).
*   If you're unable to find an open issue addressing the problem, [open a new one](https://github.com/AutoViML/featurewiz_polars/issues/new). Be sure to include a **title and clear description**, as much relevant information as possible, and a **code sample** or an **executable test case** demonstrating the expected behavior that is not occurring.

### Suggesting Enhancements
*   Open a new issue to discuss your enhancement suggestion. Clearly describe the proposed feature, why it's needed, and provide examples if possible.

### Pull Requests
We actively welcome your pull requests!

1.  **Fork the repo** and create your branch from `main`.
2.  **Set up your development environment:**
    ```bash
    git clone <your-fork-url>
    cd featurewiz_polars
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -e ".[dev]" # Install in editable mode with dev dependencies
    pre-commit install # Optional, but recommended: install pre-commit hooks
    ```
3.  **Make your changes.** Add features or fix bugs.
4.  **Add Tests:** If you've added code that should be tested, add tests to the `tests/` directory.
5.  **Ensure Code Quality:**
    *   Format your code using Ruff/Black: `ruff format .` or `black .`
    *   Lint your code using Ruff: `ruff check .`
    *   Run type checks using MyPy: `mypy featurewiz_polars`
    *   Run the test suite using Pytest: `pytest tests/ --cov=featurewiz_polars`
    *   Ensure tests pass and coverage meets requirements (aim for high coverage).
6.  **Update Documentation:** If your changes affect documentation (docstrings, README, etc.), please update them accordingly.
7.  **Commit your changes** using a clear commit message.
8.  **Push** to your fork and submit a **Pull Request (PR)** to the `main` branch of the `AutoViML/featurewiz_polars` repository.
9.  **Link the PR to an issue** if it resolves one (e.g., "Closes #123").
10. **Wait for review.** Address any comments or feedback from the maintainers.

## Coding Standards
*   Follow PEP 8 style guidelines.
*   Use `ruff` for linting and `black` or `ruff format` for code formatting (configuration is in `pyproject.toml` or `.ruff.toml` if added).
*   Write clear, understandable code with meaningful variable names.
*   Add type hints to function signatures.
*   Write comprehensive docstrings for public modules, classes, and functions (NumPy or Google style).

Thank you for contributing!