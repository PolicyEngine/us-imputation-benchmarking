# The Imputer Class

The `Imputer` class is an abstract base class that defines the common interface for all imputation models. It provides the basic structure with methods for data validation, model fitting and prediction. All other models inherit from this class and implement the required methods.

## Key Features
- Defines a consistent API with `fit()` and `predict()` methods
- Ensures no model can call `predict()` without fitting the model to the data first
- Handles validation of parameters and input data
