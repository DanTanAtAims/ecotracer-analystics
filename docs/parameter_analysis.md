# Parameter Analysis Workflow

Estimating Ecotracer parameters are difficult. It is often easier to measure observed
contaminant concentrations in fisheries then to estimate metabolic decay rates,
assimilation efficiencies, direct absorption rates etc. In many circumstances, modellers
will be more confident about contaminant concentration observations that the model
parameterisation and are forced to work backwards tinkering, with parameters to reproduce
observations.

The parameter analysis workflows allows users to explore the parameter space of Ecotracer
constrained by the observed contaminant concentrations, and sample from parameter space that
is guaranteed to reproduce observations. 

**It is important to note that users will still need a understanding of reasonable parameters given that Ecotracer is unidentifiable when fitting
to a single timestep of observed concentrations.**

This is useful for:
*   **Calibration**: Fitting a model to field data.
*   **Exploration**: Understanding the set of parameters that yield the same result.
*   **Uncertainty Quantification**: Generating ensembles of valid parameters to propagate uncertainty in non-equilibrium conditions.

## Mathematical Theory

The Ecotracer system, assuming the underlying fishery dynamics are constant, are governed
by a simple ODE,

$$ \frac{dC}{dt}= b - M \cdot C$$

where the $M \in \mathbb{R}^{(n \times n)}$ is constructed from Ecopath and Ecotracer parameters and governs gains and
losses between functional groups that are directly proportional to contaminant
concentration. The vector $b \in \mathbb{R}^n$ describes forcing flows such as base inflow
and immigration. This formulation gives us functionality beyond the Ecotracer module in EwE,
to examine how different forcing, beyond the environmental inflows, drive change.

To explore the parameter space we can re-express the linear system in terms of the parameters,

$$ K \cdot \theta = r $$

where $K \in \mathbb{R}^{n \times P}$ and is a function of Ecopath state and Concentration
but independent of the parameters $\theta \in \mathbb{R}^P$.
 
### 1. Optimization (The "Best" Estimate)
Given that the new system does not yield a unique solution for $\theta$, if there is a
solution, there will be infinitely many. Before we begin sampling from possible solutions we
want to inform sampling to be centered around realistic values, but values that are
consistent with the observed equilibrium.

With a best initial guess $\theta_0$, we can formulate a center to inform sampling using a
quadratic programming problem.

$$ 
\begin{align*}
    &\min_\theta || \theta - \theta_{0} ||^2 \\
    &\text{s.t.} \quad \mathbf{K} \theta = \mathbf{r} \\
    &\theta_{min} \le \theta \le \theta_{max} 
\end{align*}
$$

### 2. Nullspace Sampling (The Ensemble)
Usually, the system is underdetermined (more parameters than equations), there is a subspace of valid solutions.

We explore this space using a **Hit-and-Run** sampler.
Any valid solution $\theta$ can be expressed as:
$$ \theta = \theta_{particular} + \mathbf{N} \cdot z $$

Where:
*   $\theta_{particular}$: A valid solution (e.g., $\theta^*$).
*   $\mathbf{N}$: The matrix of nullspace basis vectors ($\mathbf{K} \cdot \mathbf{N} = 0$).
*   $z$: A vector of coefficients.

The sampler randomly moves in the $z$-space, ensuring the resulting $\theta$ remains within the physical bounds $[\theta_{min}, \theta_{max}]$.

## Usage

### 1. Setup and Build Constraints
```python
from ecotracer_analytics import (
    EcopathSource, EcotracerSource, calculate_equilibrium,
    ParameterDefinition, LinearConstraintBuilder, ConstrainedSolver
)

# Load Data
path_source = EcopathSource(core)
tracer_source = EcotracerSource(core)
C_obs = calculate_equilibrium(path_source, tracer_source) # Or provide your own

# Define Parameters
param_def = ParameterDefinition(path_source, tracer_source)

# Build K and r
builder = LinearConstraintBuilder(path_source, tracer_source, param_def)
K, r = builder.build(C_obs)
```

### 2. Solve for Optimal Parameters
```python
solver = ConstrainedSolver()
theta_opt, active_mask = solver.solve(K, r, param_def)
```

### 3. Sample the Nullspace
```python
samples = solver.sample(
    K, r, param_def,
    n_samples=1000,
    x_center=theta_opt
)
# samples shape: (1000, n_params)
```

## Parameter Reference

The reverse analysis optimizes the following parameters for each functional group:

| Parameter Name | Description | Contribution |
| :--- | :--- | :--- |
| `Base Inflow` | Environmental Inflow | Source for Env |
| `Uptake_{i}` | Direct Absorption Rate | Gain for Group $i$, Loss for Env |
| `MetaDecay_{i}` | Metabolic Decay Rate | Loss for Group $i$, Gain for Env |
| `PhysDecay_{i}` | Physical Decay Rate | Loss for Group $i$ (leaves system) |
| `AE_{i}` | Assimilation Efficiency | Scales gain from diet |
| `ImmigConc_{i}` | Immigration Concentration | Source for Group $i$ |
