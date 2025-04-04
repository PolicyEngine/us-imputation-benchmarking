# MicroImpute - Developer Guide

## Code Style Guidelines

### Formatting & Organization
- Use 4 spaces for indentation
- Maximum line length: 79 characters (Black default)
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

## Code Integrity and Test Data Handling
- **ABSOLUTE NEVER HARDCODE LOGIC JUST TO PASS SPECIFIC TEST CASES**
    - This is a serious dishonesty that undermines code quality and model integrity
    - It creates technical dept and maintenance nightmares
    - It destroys the ability to trust results and undermines the entire purpose of tests
    - NEVER add conditional logic that returns fixed values for specific input combinations