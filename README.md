# Ecotracer Analytics

A Python library for the analytical analysis of Ecopath with Ecosim (EwE) Ecotracer models. This tool provides methods to represent contaminant dynamics as a linear system, enabling direct calculation of equilibrium states and reverse parameter estimation.

## Features

- **Linear System Extraction:** Converts Ecotracer dynamics into matrix form ($dC/dt = b - MC$).
- **Analytical Equilibrium:** Solves for steady-state concentrations using matrix inversion.
- **Reverse Analysis:** Estimates parameters from observed concentration data using constrained optimization.
- **Uncertainty Quantification:** Includes nullspace sampling and nested sampling workflows to propagate parameter and observation uncertainty.

## Documentation

Example workflows are provided in the `pyewe_sandbox` directory:

1. `01_linear_system_basics.ipynb`: Forward dynamics and analytical equilibrium.
2. `02_reverse_parameter_estimation.ipynb`: Parameter estimation and nullspace exploration.
3. `03_nested_sampling_missing_data.ipynb`: Handling unobserved groups via nested sampling.

Detailed documentation is available in the `docs` directory.
