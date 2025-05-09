# Example: imputing wealth on the CPS data set

This example demonstrates how to use the microimpute package to impute wealth variables from the Survey of Consumer Finances (SCF) onto the Current Population Survey (CPS). 

The Survey of Consumer Finances (SCF) contains detailed information about household wealth, but the Current Population Survey (CPS) does not. By using the microimpute package's `autoimpute` function, we can leverage the relationships between common variables in both surveys to impute wealth variables onto the CPS dataset.
