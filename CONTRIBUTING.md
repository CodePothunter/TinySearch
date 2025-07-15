# Contributing to TinySearch

Thank you for considering contributing to TinySearch! This document provides guidelines and instructions for contributing to the project.

## Development Environment

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/yourusername/tinysearch.git
   cd tinysearch
   ```

2. Create a virtual environment and install development dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   pip install -e ".[dev]"
   ```

3. Run tests to ensure everything is working:
   ```bash
   pytest
   ```

## Code Style

This project uses:
- [Black](https://black.readthedocs.io/) for code formatting
- [isort](https://pycqa.github.io/isort/) for import sorting
- [Flake8](https://flake8.pycqa.org/) for linting
- [mypy](https://mypy.readthedocs.io/) for type checking

You can run all style checks with:

```bash
black tinysearch tests
isort tinysearch tests
flake8 tinysearch tests
mypy tinysearch
```

## Creating a New Component

TinySearch is designed to be extensible. Here's how to create a new component:

1. Identify which interface to implement:
   - `DataAdapter` for new data sources
   - `TextSplitter` for new text splitting strategies
   - `Embedder` for new embedding models
   - `VectorIndexer` for new vector indexing methods
   - `QueryEngine` for new query processing techniques

2. Create a new file in the appropriate module directory.

3. Implement the required interface methods.

4. Add tests for your component.

5. Update documentation to reflect your new component.

## Pull Request Process

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, add tests, and ensure all tests pass.

3. Update documentation if necessary.

4. Push your changes to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

5. Create a pull request to the main repository's `main` branch.

6. Describe your changes in detail, including the motivation for the change.

7. Wait for review and address any feedback.

## Adding Dependencies

If your contribution requires new dependencies:

1. Add them to the appropriate section in `setup.py`.
2. Document why the dependency is necessary.
3. Consider making it optional by adding it to `extras_require` if it's not core functionality.

## Documentation

All new features should include:

1. Docstrings for all public methods and classes.
2. Updates to the relevant README sections or documentation files.
3. Example usage if applicable.

## Testing

All new features should include tests:

1. Unit tests for individual components.
2. Integration tests if the feature interacts with other components.
3. Ensure all tests pass before submitting a pull request.

## Code of Conduct

Please be respectful and inclusive in all interactions within the project community. We aim to maintain a welcoming environment for everyone, regardless of background or experience level.

## Questions?

If you have questions about contributing, please open an issue or contact the project maintainers.

Thank you for contributing to TinySearch! 