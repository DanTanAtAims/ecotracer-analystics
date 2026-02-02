# Ecotracer Analytics

**Ecotracer Analytics** is a Python library for advanced analysis of Ecopath with Ecosim (EwE) Ecotracer models. It provides tools to extract linear system representations of contaminant flows, calculate equilibrium states analytically, and perform "reverse" parameter estimation.

## Features

*   **Linear System Extraction**: Convert Ecotracer dynamics into Matrix form. $$ \frac{dC}{dt} = b - M \cdot C $$.
*   **Simulate ODE**: Simulate the ODE overtime without the need to solve for a stable ecosim parameterisation.
*   **Analytic Equilibrium**: Solve for steady-state concentrations without time-stepping simulations. $$ C = M^{-1} \cdot b $$
*   **Parameter Analysis**: Estimate Ecotracer parameters (direct absorption rates, decay, etc.) that reproduce a specific observed equilibrium.
*   **Uncertainty Propagation**: Calculate analytic confidence intervals for predictions based on parameter uncertainty.

## Installation

```bash
pip install ecotracer-analytics
```

## Quick Start

```python
import pyewe as ewe
from ecotracer_analytics import EcopathSource, EcotracerSource, calculate_equilibrium

# 1. Load your EwE Model
model = ewe.EwEScenarioInterface("MyModel.eiixml")
core = model._core_instance

# 2. Wrap sources
path_source = EcopathSource(core)
tracer_source = EcotracerSource(core)

# 3. Calculate Equilibrium
C_eq = calculate_equilibrium(path_source, tracer_source)
print("Equilibrium Concentrations:", C_eq)
```
