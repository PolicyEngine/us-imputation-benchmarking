# US Imputation Benchmarking - Developer Guide

## Build & Test Commands
```bash
# Install package in development mode
pip install -e .

# Run all tests
python us_imputation_benchmarking/tests.py

# Run specific model test (example)
python -c "from us_imputation_benchmarking import tests; tests.test_qrf()"

# Install development dependencies
pip install black isort mypy pytest
```

## Code Style Guidelines

### Formatting & Organization
- Use 4 spaces for indentation
- Maximum line length: 88 characters (Black default)
- Format code with Black: `black us_imputation_benchmarking/`
- Sort imports with isort: `isort us_imputation_benchmarking/`

### Naming & Types
- Use snake_case for variables, functions, and modules
- Use CamelCase for classes
- Constants should be UPPERCASE
- Add type hints to all function parameters and return values
- Document functions with ReStructuredText-style docstrings

### Imports
- Group imports: standard library, third-party, local modules
- Import specific functions/classes rather than entire modules when practical

### Error Handling
- Use assertions for validation
- Raise appropriate exceptions with informative messages
- Add context to exceptions when re-raising