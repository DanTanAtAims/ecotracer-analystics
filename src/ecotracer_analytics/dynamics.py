import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple, Any, Optional

from .core import EcopathSource, EcotracerSource
from .linear_system import calculate_coefficient, calculate_intercept

def build_state_vector(
    path_source: EcopathSource, 
    env_conc: float, 
    group_concs: np.ndarray
) -> np.ndarray:
    """
    Helper to build the state vector x = [C_env, A_1, ..., A_N] from concentrations.
    
    Parameters
    ----------
    path_source : EcopathSource
        Used to retrieve biomass for converting concentration to amount (A_i = C_i * B_i).
    env_conc : float
        Initial concentration in the environment.
    group_concs : np.ndarray
        Initial concentrations for the N functional groups.
    
    Returns
    -------
    np.ndarray
        State vector of dimension N+1.
    """
    num_groups = len(path_source.biomass)
    if len(group_concs) != num_groups:
        raise ValueError(f"Expected {num_groups} group concentrations, got {len(group_concs)}")
    
    y = np.zeros(num_groups + 1)
    y[0] = env_conc
    y[1:] = group_concs * path_source.biomass
    return y

def simulate_dynamics(
    path_source: EcopathSource,
    tracer_source: EcotracerSource,
    y0: np.ndarray,
    t_span: Tuple[float, float],
    t_eval: Optional[np.ndarray] = None,
    method: str = "RK45",
    **kwargs: Any
):
    """
    Simulates the Ecotracer ODE system over time.
    dy/dt = b - M * y

    Parameters
    ----------
    path_source : EcopathSource
        Ecopath parameters wrapper.
    tracer_source : EcotracerSource
        Ecotracer parameters wrapper.
    y0 : np.ndarray
        Initial state vector [C_env, A_1, ..., A_N].
        Use build_state_vector() to construct this from concentrations.
    t_span : tuple
        (t_start, t_end) for the simulation.
    t_eval : np.ndarray, optional
        Times at which to store the computed solution.
    method : str, default "RK45"
        Integration method to use.
    **kwargs
        Additional arguments passed to scipy.integrate.solve_ivp.

    Returns
    -------
    scipy.integrate._ivp.ivp.OdeResult
        The result object from solve_ivp.
    """
    
    # 1. Build System Matrices
    # M (Coefficient Matrix): diagonal = loss rates, off-diagonal = -transfer rates
    # b (Intercept Vector): independent inflows (environment base inflow, immigration)
    M = calculate_coefficient(path_source, tracer_source)
    b = calculate_intercept(path_source, tracer_source)

    # 2. Define Derivative Function
    def system_derivative(t, y):
        # dy/dt = b - M @ y
        # Note: M is built such that M*y accounts for both losses (positive diagonal)
        # and gains from other groups (negative off-diagonals).
        return b - M @ y

    # 3. Solve the IVP
    return solve_ivp(
        system_derivative, 
        t_span, 
        y0, 
        t_eval=t_eval, 
        method=method, 
        **kwargs
    )