# API Reference

## Linear System

### `calculate_equilibrium`
```python
def calculate_equilibrium(path_source: EcopathSource, tracer_source: EcotracerSource) -> np.ndarray
```
Solves the linear system $M \cdot C = b$ to find the steady-state concentrations.

### `calculate_coefficient`
```python
def calculate_coefficient(path_source: EcopathSource, tracer_source: EcotracerSource) -> np.ndarray
```
Constructs the coefficient matrix $M$ (size $(N+1) \times (N+1)$).

### `calculate_intercept`
```python
def calculate_intercept(path_source: EcopathSource, tracer_source: EcotracerSource) -> np.ndarray
```
Constructs the intercept vector $b$ (size $N+1$). 

---

## Dynamics

### `simulate_dynamics`
```python
def simulate_dynamics(
    path_source: EcopathSource,
    tracer_source: EcotracerSource,
    y0: np.ndarray,
    t_span: Tuple[float, float],
    t_eval: Optional[np.ndarray] = None,
    method: str = "RK45",
    **kwargs: Any
) -> scipy.integrate.OdeResult
```
Simulates the system forward in time ($\frac{dC}{dt} = b - M \cdot C$).

### `build_state_vector`
```python
def build_state_vector(
    path_source: EcopathSource,
    env_conc: float,
    group_concs: np.ndarray
) -> np.ndarray
```
Helper to construct the state vector $y = [C_{env}, C_1, \dots, C_N]$.

---

## Reverse Analysis

### `ParameterDefinition`
```python
class ParameterDefinition:
    def __init__(self, path_source: EcopathSource, tracer_source: EcotracerSource)
```
Defines the mapping between the flat parameter vector $\theta$ and the Ecotracer model structure. Handles priors, bounds, and fixed statuses.

### `LinearConstraintBuilder`
```python
class LinearConstraintBuilder:
    def __init__(self, path_source: EcopathSource, tracer_source: EcotracerSource, param_def: ParameterDefinition)
    
    def build(self, C_obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]
```
Constructs the linear constraint system $\mathbf{K} \cdot \theta = \mathbf{r}$.

### `ConstrainedSolver`
```python
class ConstrainedSolver:
    def solve(
        self, 
        K: np.ndarray, 
        r: np.ndarray, 
        param_def: ParameterDefinition, 
        x0: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]
```
Finds the optimal parameter set $\theta^*$ that satisfies $\mathbf{K}\theta = \mathbf{r}$ and bounds, minimizing the distance to the priors.

### `ConstrainedSolver.sample`
```python
    def sample(
        self,
        K: np.ndarray,
        r: np.ndarray,
        param_def: ParameterDefinition,
        n_samples: int = 1000,
        sigma: float = 0.05,
        x_center: Optional[np.ndarray] = None,
        x_anchor: Optional[np.ndarray] = None
    ) -> np.ndarray
```
Samples feasible parameter sets from the nullspace of the constraints using a Hit-and-Run algorithm with truncated Gaussian steps.
*   `sigma`: Standard deviation of the sampling step size.
*   `x_center`: The center of the distribution (defaults to priors).
*   `x_anchor`: A valid starting point on the manifold (defaults to result of `solve`).

```