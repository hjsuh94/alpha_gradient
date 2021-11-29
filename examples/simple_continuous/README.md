# Simple Continuous Functions
The goal of the functions in this folder is to analyze bias and variance of the first and zero order batched gradients in the setting of continuous functions.

We are interested in the following questions:
    - How does the variance of the two objects scale per the dimension of the objective function?
    - How are sample variance affected as sample size `n` increases?
    - What effect does it have on the convergence rate of the optimization algorithm?

We provide two cases of functions that are general to dimensionality:
    - Linear functions.
    - Polynomial functions.
