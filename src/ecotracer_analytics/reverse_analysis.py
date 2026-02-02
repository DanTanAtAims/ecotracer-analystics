import numpy as np
from scipy.optimize import minimize
from scipy.linalg import null_space
from typing import Tuple, List, Dict, Optional, NamedTuple
from dataclasses import dataclass

from scipy.sparse.linalg import lsqr


from ecotracer_analytics.core import EcopathSource, EcotracerSource
from ecotracer_analytics.linear_system import calculate_coefficient, calculate_intercept

# --- 1. Parameter Definitions ---

class ParameterType:
    UPTAKE = "uptake"
    METABOLIC_DECAY = "metabolic_decay"
    PHYSICAL_DECAY = "physical_decay"
    ASSIMILATION_EFFICIENCY = "assim_eff"
    BASE_INFLOW = "base_inflow"
    IMMIGRATION_CONC = "immigration_conc"
    ENV_VOLUME_EXCHANGE = "env_volume_exchange"

@dataclass
class ParameterInfo:
    index: int
    name: str
    param_type: str
    group_index: Optional[int]
    prior: float
    lb: float
    ub: float
    is_fixed: bool = False

class ParameterDefinition:
    def __init__(self, path_source: EcopathSource, tracer_source: EcotracerSource):
        self.params: List[ParameterInfo] = []
        self._build_parameters(path_source, tracer_source)

    def _build_parameters(self, path_source: EcopathSource, tracer_source: EcotracerSource):
        idx = 0
        num_groups = len(path_source.biomass)

        # 1. Base Inflow (Env)
        self.params.append(ParameterInfo(idx, "Base Inflow", ParameterType.BASE_INFLOW, None, tracer_source.base_inflow, 0.0, np.inf))
        idx += 1

        # 2. Uptake Rates (Groups)
        for i in range(num_groups):
            val = tracer_source.dir_abs_r[i]
            self.params.append(ParameterInfo(idx, f"Uptake_{i}", ParameterType.UPTAKE, i, val, 0.0, np.inf))
            idx += 1

        # 3. Metabolic Decay (Groups)
        for i in range(num_groups):
            val = tracer_source.meta_dec_r[i]
            self.params.append(ParameterInfo(idx, f"MetaDecay_{i}", ParameterType.METABOLIC_DECAY, i, val, 0.0, np.inf))
            idx += 1

        # 4. Physical Decay (Env + Groups)
        # Env (index 0 effectively in code, but stored separately in source)
        val_env = tracer_source.env_decay
        self.params.append(ParameterInfo(idx, "PhysDecay_Env", ParameterType.PHYSICAL_DECAY, -1, val_env, 0.0, np.inf)) # -1 for Env
        idx += 1

        # 4b. Environment Volume Exchange Loss (Env)
        val_vol = tracer_source.env_volume_exchange_loss
        self.params.append(ParameterInfo(idx, "EnvVolExchange", ParameterType.ENV_VOLUME_EXCHANGE, -1, val_vol, 0.0, np.inf))
        idx += 1

        for i in range(num_groups):
            val = tracer_source.phys_dec_r[i]
            self.params.append(ParameterInfo(idx, f"PhysDecay_{i}", ParameterType.PHYSICAL_DECAY, i, val, 0.0, np.inf))
            idx += 1

        # 5. Assimilation Efficiency (Groups)
        for i in range(num_groups):
            val = tracer_source.assim_eff[i]
            # AE is typically 0 to 1
            self.params.append(ParameterInfo(idx, f"AE_{i}", ParameterType.ASSIMILATION_EFFICIENCY, i, val, 0.0, 1.0))
            idx += 1

        # 6. Immigration Concentration (Groups)
        for i in range(num_groups):
            val = tracer_source.immigration_c[i]
            self.params.append(ParameterInfo(idx, f"ImmigConc_{i}", ParameterType.IMMIGRATION_CONC, i, val, 0.0, np.inf))
            idx += 1

    @property
    def num_params(self):
        return len(self.params)

    @property
    def priors(self):
        return np.array([p.prior for p in self.params])

    @property
    def bounds(self):
        return [(p.lb, p.ub) for p in self.params]

def fill_uptake_coefficients(
    K: np.ndarray,
    path_source: EcopathSource,
    equilibrium_c: np.ndarray,
    param_def: ParameterDefinition
) -> None:
    """
    Analytically calculates K matrix entries for Uptake parameters (u_i)
    for the Biota mass balance equations (Rows 1..N).
    Contribution to Group i Mass Balance: + u_i * B_i * C_env
    """
    C_env = equilibrium_c[0]
    for param in param_def.params:
        if param.param_type == ParameterType.UPTAKE:
            g_idx = param.group_index
            col_idx = param.index
            B_i = path_source.biomass[g_idx]
            coeff = B_i * C_env
            # Row: g_idx + 1 (because Row 0 is Environment)
            K[g_idx + 1, col_idx] = coeff
            # Environment Loss
            K[0, col_idx] = -coeff

def fill_metabolic_decay_coefficients_biota(
    K: np.ndarray,
    equilibrium_c: np.ndarray,
    param_def: ParameterDefinition
) -> None:
    """
    Analytically calculates K matrix entries for Metabolic Decay parameters (k_meta_i)
    for the Biota mass balance equations (Rows 1..N).
    Contribution to Group i Mass Balance (Row i+1): - k_meta_i * Mass_i
    """
    for param in param_def.params:
        if param.param_type == ParameterType.METABOLIC_DECAY:
            g_idx = param.group_index
            col_idx = param.index

            c_i = equilibrium_c[g_idx + 1]

            # Group loss
            K[g_idx + 1, col_idx] = -c_i
            # Env Gaini
            K[0, col_idx] = c_i

def fill_assimilation_efficiency_coefficients_biota(
    K: np.ndarray,
    path_source: EcopathSource,
    equilibrium_c: np.ndarray,
    param_def: ParameterDefinition
) -> None:
    """
    Analytically calculates K matrix entries for Assimilation Efficiency parameters (AE_i)
    for the Biota mass balance equations (Rows 1..N).

    Contribution to Group i Mass Balance (Row i+1): + AE_i * Intake_i
    Where Intake_i = Sum_j(Consumption_ji * Mass_j / B_j)
    """
    biomass = path_source.biomass
    inv_biomass = np.zeros_like(biomass)
    mask_b = biomass > 0
    inv_biomass[mask_b] = 1.0 / biomass[mask_b]

    prey_specific_loss = path_source.consumption * inv_biomass[:, np.newaxis] # Row j scaled by 1/B_j
    total_a_in_diet = equilibrium_c[1:] @ prey_specific_loss

    det_indices = path_source.is_detritus
    total_a_in_diet[det_indices] = 0

    for param in param_def.params:
       if param.param_type == ParameterType.ASSIMILATION_EFFICIENCY:
            g_idx = param.group_index
            col_idx = param.index

            # Row g_idx + 1: The Group Gain
            flux = total_a_in_diet[g_idx]
            K[g_idx + 1, col_idx] = flux
            K[0, col_idx] = -flux

def fill_physical_decay_coefficients(
    K: np.ndarray,
    equilibrium_c: np.ndarray,
    param_def: ParameterDefinition
) -> None:
    """
    Analytically calculates K matrix entries for Physical Decay parameters (k_phys_i).
    Includes both the Environment and Biota mass balance equations.

    Contribution to Environment Eq (Row 0): - k_phys_env * C_env
    Contribution to Group i Eq (Row i+1):   - k_phys_i   * Mass_i
    """
    for param in param_def.params:
        if param.param_type == ParameterType.PHYSICAL_DECAY:
            col_idx = param.index
            g_idx = param.group_index

            K[g_idx + 1, col_idx] = -equilibrium_c[g_idx + 1]

def fill_immigration_coefficients(
    K: np.ndarray,
    path_source: EcopathSource,
    param_def: ParameterDefinition
) -> None:
    """
    Analytically calculates K matrix entries for Immigration Concentration parameters (C_mig_i).

    Contribution to Group i Mass Balance (Row i+1): + ImmigRate_i * C_mig_i
    """
    for param in param_def.params:
        if param.param_type == ParameterType.IMMIGRATION_CONC:
            g_idx = param.group_index
            col_idx = param.index

            immig_rate_i = path_source.immigration_rate[g_idx]
            K[g_idx + 1, col_idx] = immig_rate_i

def fill_base_inflow_coefficients(
    K: np.ndarray,
    param_def: ParameterDefinition
) -> None:
    """
    Analytically calculates K matrix for Base Inflow.
    Contribution to Env (Row 0): +1
    """
    for param in param_def.params:
        if param.param_type == ParameterType.BASE_INFLOW:
            K[0, param.index] = 1.0

def fill_env_volume_exchange_coefficients(
    K: np.ndarray,
    equilibrium_c: np.ndarray,
    param_def: ParameterDefinition
) -> None:
    """
    Analytically calculates K matrix for Environment Volume Exchange Loss.
    Contribution to Env (Row 0): - VolExchange * C_env
    """
    for param in param_def.params:
        if param.param_type == ParameterType.ENV_VOLUME_EXCHANGE:

            K[0, param.index] = -equilibrium_c[0]


# --- 2. Linear Constraint Builder ---

class LinearConstraintBuilder:
    """
    Constructs K and r such that K * theta = r represents M(theta)*C_obs = b(theta).
    """
    def __init__(self, path_source: EcopathSource, tracer_source: EcotracerSource, param_def: ParameterDefinition):
        self.path_source = path_source
        self.tracer_source = tracer_source # Used only for structure, not values (values come from theta)
        self.param_def = param_def
        self.N = len(path_source.biomass)

    def build(self, C_obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Derives K (matrix) and r (vector) analytically.
        K * theta = r

        Parameters
        ----------
        C_obs : np.ndarray
            Full vector of observed concentrations (Env + N Groups).
        """
        if len(C_obs) != self.N + 1:
            raise ValueError(f"C_obs must have length {self.N + 1}")

        P = self.param_def.num_params
        K = np.zeros((self.N + 1, P))

        # 1. Fill K using analytic helpers
        fill_base_inflow_coefficients(K, self.param_def)
        fill_uptake_coefficients(K, self.path_source, C_obs, self.param_def)
        fill_metabolic_decay_coefficients_biota(K, C_obs, self.param_def)
        fill_physical_decay_coefficients(K, C_obs, self.param_def)
        fill_env_volume_exchange_coefficients(K, C_obs, self.param_def)
        fill_assimilation_efficiency_coefficients_biota(K, self.path_source, C_obs, self.param_def)
        fill_immigration_coefficients(K, self.path_source, self.param_def)

        # 2. Calculate r (Base Imbalance)
        # r = M_const * C - b_const
        zero_tracer = self._create_zero_tracer_source()
        M_const = calculate_coefficient(self.path_source, zero_tracer)
        b_const = calculate_intercept(self.path_source, zero_tracer)

        r = M_const @ C_obs - b_const

        return K, r

    def _create_zero_tracer_source(self) -> EcotracerSource:
        """Creates a dummy source with all optimizable parameters set to zero."""
        class DummyTracerSource:
            pass
        dummy = DummyTracerSource()
        dummy.base_inflow = 0.0
        dummy.dir_abs_r = np.zeros(self.N)
        dummy.meta_dec_r = np.zeros(self.N)
        dummy.env_decay = 0.0
        dummy.phys_dec_r = np.zeros(self.N)
        dummy.assim_eff = np.zeros(self.N)
        dummy.immigration_c = np.zeros(self.N)
        dummy.env_volume_exchange_loss = 0.0
        return dummy

    def _create_tracer_source_with_params(self, theta: np.ndarray) -> EcotracerSource:
        """Helper to inject a parameter vector into a dummy EcotracerSource."""
        # Create a copy or new instance.
        # Since EcotracerSource reads from core, we need to manually overwrite attributes.
        # We can just create a dummy object that mimics EcotracerSource interface expected by linear_system.

        class DummyTracerSource:
            pass

        dummy = DummyTracerSource()

        # Initialize arrays with zeros
        dummy.base_inflow = 0.0
        dummy.dir_abs_r = np.zeros(self.N)
        dummy.meta_dec_r = np.zeros(self.N)
        dummy.env_decay = 0.0
        dummy.phys_dec_r = np.zeros(self.N)
        dummy.assim_eff = np.zeros(self.N)
        dummy.immigration_c = np.zeros(self.N)

        # Constant that is NOT in parameters (assume fixed from original source)
        dummy.env_volume_exchange_loss = 0.0

        # Map theta to attributes
        for k, val in enumerate(theta):
            p_info = self.param_def.params[k]
            idx = p_info.group_index
            ptype = p_info.param_type

            if ptype == ParameterType.BASE_INFLOW:
                dummy.base_inflow = val
            elif ptype == ParameterType.ENV_VOLUME_EXCHANGE:
                dummy.env_volume_exchange_loss = val
            elif ptype == ParameterType.UPTAKE:
                dummy.dir_abs_r[idx] = val
            elif ptype == ParameterType.METABOLIC_DECAY:
                dummy.meta_dec_r[idx] = val
            elif ptype == ParameterType.PHYSICAL_DECAY:
                if idx == -1: dummy.env_decay = val
                else: dummy.phys_dec_r[idx] = val
            elif ptype == ParameterType.ASSIMILATION_EFFICIENCY:
                dummy.assim_eff[idx] = val
            elif ptype == ParameterType.IMMIGRATION_CONC:
                dummy.immigration_c[idx] = val

        return dummy


import numpy as np
from typing import Optional, Tuple, Dict
from scipy.optimize import minimize, Bounds, LinearConstraint

def solve_qp_closest_to_xbar_with_fixed_scipy(
    K: np.ndarray,
    r: np.ndarray,
    xbar: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    fixed_mask: np.ndarray,
    fixed_values: Optional[np.ndarray] = None,
    trust_tol: float = 1e-12,
    max_iter: int = 10_000,
    verbose: bool = False,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Solve: minimize 0.5*||x - xbar||^2 s.t. Kx=r, lb <= x <= ub, and x[fixed_mask]=fixed_values.
    Uses SciPy's trust-constr solver with an exact quadratic objective (I, -xbar).
    Reduces to the free variables first to keep the equality constraints well-conditioned.

    Returns (x_opt, info).
    """
    # --- Validate inputs ---
    K = np.asarray(K, dtype=float); r = np.asarray(r, dtype=float).reshape(-1)
    xbar = np.asarray(xbar, dtype=float).reshape(-1)
    lb = np.asarray(lb, dtype=float).reshape(-1); ub = np.asarray(ub, dtype=float).reshape(-1)
    fixed_mask = np.asarray(fixed_mask, dtype=bool).reshape(-1)

    m, n = K.shape
    if r.shape != (m,): raise ValueError(f"r must be shape ({m},), got {r.shape}")
    if xbar.shape != (n,): raise ValueError(f"xbar must be shape ({n},), got {xbar.shape}")
    if lb.shape != (n,) or ub.shape != (n,): raise ValueError("lb, ub must be shape (n,)")
    if np.any(lb > ub):
        bad = np.where(lb > ub)[0]; raise ValueError(f"Infeasible bounds at indices {bad}")

    # Merge hard-equality bounds into fixed set
    eq_fix = np.isclose(lb, ub, rtol=0.0, atol=0.0)
    merged_fixed_mask = fixed_mask | eq_fix

    # Fixed values array
    if fixed_values is None:
        fixed_values = xbar[fixed_mask].copy()
    else:
        fixed_values = np.asarray(fixed_values, dtype=float).reshape(-1)
        if fixed_values.shape != (fixed_mask.sum(),):
            raise ValueError("fixed_values length must match fixed_mask True count.")

    # Compose the consistent fixed vector in the merged set
    all_idx = np.arange(n)
    fixed_idx_explicit = all_idx[fixed_mask]
    fixed_idx_bounds = all_idx[eq_fix & ~fixed_mask]
    merged_fixed_idx = np.concatenate([fixed_idx_explicit, fixed_idx_bounds])
    merged_fixed_vals = np.concatenate([fixed_values, lb[fixed_idx_bounds]])  # lb==ub here

    # Bounds sanity for fixed
    if np.any(merged_fixed_vals < lb[merged_fixed_idx] - 1e-15) or np.any(merged_fixed_vals > ub[merged_fixed_idx] + 1e-15):
        raise ValueError("Some fixed values violate their bounds.")

    # Free index set
    free_mask = ~merged_fixed_mask
    free_idx = all_idx[free_mask]
    nv = free_idx.size

    # Trivial case: all fixed — validate feasibility and return
    if nv == 0:
        x_full = np.zeros(n)
        x_full[merged_fixed_idx] = merged_fixed_vals
        if np.any(x_full < lb - 1e-15) or np.any(x_full > ub + 1e-15):
            raise ValueError("All variables fixed but violate bounds.")
        aff_res = float(np.linalg.norm(K @ x_full - r))
        if aff_res > trust_tol:
            raise ValueError("All variables fixed but Kx=r infeasible with given fixed values.")
        return x_full, dict(status="all_fixed", iters=0, affine_residual=aff_res, feasible_bounds=True)

    # Reduced system: Kv x_v = r' := r - Kf c_f
    Kf = K[:, merged_fixed_idx] if merged_fixed_idx.size else np.zeros((m, 0))
    Kv = K[:, free_idx]
    r_prime = r - (Kf @ merged_fixed_vals if merged_fixed_idx.size else 0.0)

    # Reduced bounds and prior
    lb_v = lb[free_idx]; ub_v = ub[free_idx]; xbar_v = xbar[free_idx]

    # Quick feasibility check for equality (consistency)
    # If inconsistent, no solution exists
    if m > 0:
        sol_ls, *_ = np.linalg.lstsq(Kv, r_prime, rcond=None)
        res_norm = np.linalg.norm(Kv @ sol_ls - r_prime)
        if res_norm > 1e-10:
            raise ValueError(f"Reduced equality inconsistent: ||Kv x - r'||={res_norm:.3e}")

    # QP in free variables:
    # minimize f(xv) = 0.5 * ||xv - xbar_v||^2
    # s.t. Kv xv = r', lb_v <= xv <= ub_v
    # Gradient: g(xv) = xv - xbar_v
    # Hessian: H = I (positive definite)

# --- 3. Constrained Solver ---

import numpy as np
from typing import Optional, Tuple, Dict

def project_to_affine_box_with_fixed(
    K: np.ndarray,
    r: np.ndarray,
    xbar: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    fixed_mask: np.ndarray,
    fixed_values: Optional[np.ndarray] = None,
    max_iters: int = 10_000,
    tol: float = 1e-8,
    tol_consistency: float = 1e-10,
    verbose: bool = False,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Project xbar onto the set { x : K x = r, lb <= x <= ub, x[fixed_mask] = fixed_values }
    by reducing to the free variables and using Dykstra's algorithm between:
        - affine set S_v = { x_v : K_v x_v = r' }
        - box B_v = [lb_v, ub_v]
    Returns the full-length vector x* and diagnostic info.

    Parameters
    ----------
    K : (m, n) ndarray
        Constraint matrix in K x = r.
    r : (m,) ndarray
        Right-hand side.
    xbar : (n,) ndarray
        Prior / target vector to be as close as possible to (Euclidean norm).
    lb, ub : (n,) ndarray
        Elementwise lower/upper bounds. Must satisfy lb[i] <= ub[i].
        You can encode fixed variables either via (lb==ub) and/or via fixed_mask+fixed_values.
    fixed_mask : (n,) boolean ndarray
        True for indices to be fixed.
    fixed_values : (k,) ndarray, optional
        Values for the fixed indices (k = fixed_mask.sum()).
        If None, will use xbar[fixed_mask] (after validating within bounds).
    max_iters : int
        Maximum iterations for Dykstra's algorithm.
    tol : float
        Convergence tolerance on successive iterate change (Euclidean norm).
    tol_consistency : float
        Tolerance used to check consistency of the reduced affine system K_v x_v = r'.
    verbose : bool
        If True, prints brief convergence info.

    Returns
    -------
    x_opt : (n,) ndarray
        The projected vector (solution).
    info : dict
        Diagnostics: {'iters', 'affine_residual', 'changed_norm', 'feasible_affine', 'feasible_bounds'}

    Notes
    -----
    - If there are no equality constraints (K has zero rows), the solution is simply clipping xbar
      on free variables with fixed variables enforced.
    - If all variables are fixed, the function validates feasibility and returns the fixed vector.
    - This uses only NumPy (no SciPy) via least-squares solves for the affine projection.

    Example
    -------
    >>> n = 5
    >>> K = np.array([[1, 1, 0, 0, 0],
    ...               [0, 1, 1, 0, 0]], dtype=float)
    >>> r = np.array([1.0, 0.5])
    >>> xbar = np.zeros(n)
    >>> lb = -np.ones(n)
    >>> ub =  np.ones(n)
    >>> fixed_mask = np.array([False, False, False, True, True])
    >>> fixed_values = np.array([0.2, -0.1])  # for indices 3 and 4
    >>> x_opt, info = project_to_affine_box_with_fixed(K, r, xbar, lb, ub, fixed_mask, fixed_values)
    >>> print(x_opt, info['affine_residual'])
    """
    # ---------------------
    # Validations and setup
    # ---------------------
    K = np.atleast_2d(np.asarray(K, dtype=float))
    r = np.atleast_1d(np.asarray(r, dtype=float))
    xbar = np.atleast_1d(np.asarray(xbar, dtype=float))
    lb = np.atleast_1d(np.asarray(lb, dtype=float))
    ub = np.atleast_1d(np.asarray(ub, dtype=float))
    fixed_mask = np.asarray(fixed_mask, dtype=bool)

    m, n = K.shape
    if r.shape != (m,):
        raise ValueError(f"r must have shape ({m},), got {r.shape}")
    if xbar.shape != (n,):
        raise ValueError(f"xbar must have shape ({n},), got {xbar.shape}")
    if lb.shape != (n,) or ub.shape != (n,):
        raise ValueError(f"lb, ub must have shape ({n},), got {lb.shape}, {ub.shape}")
    if np.any(lb > ub + 1e-14):
        bad = np.where(lb > ub)[0]
        raise ValueError(f"Found lb[i] > ub[i] at indices: {bad}")

    if fixed_values is None:
        # Default to xbar at fixed indices, but ensure within bounds
        fixed_values = xbar[fixed_mask].copy()
    else:
        fixed_values = np.atleast_1d(np.asarray(fixed_values, dtype=float))
        if fixed_values.shape != (fixed_mask.sum(),):
            raise ValueError(
                f"fixed_values must have shape ({fixed_mask.sum()},), got {fixed_values.shape}"
            )

    # Enforce fixed values must be within bounds
    if np.any(fixed_values < lb[fixed_mask] - 1e-14) or np.any(fixed_values > ub[fixed_mask] + 1e-14):
        idx = np.where((fixed_values < lb[fixed_mask]) | (fixed_values > ub[fixed_mask]))[0]
        raise ValueError(f"Fixed values violate bounds at fixed indices (relative positions): {idx}")

    # If lb == ub at some indices, they are also effectively fixed; merge with fixed_mask
    tie_fixed = np.isclose(lb, ub, atol=0.0, rtol=0.0)
    if np.any(tie_fixed & ~fixed_mask):
        # augment fixed set and fixed values
        extra_mask = tie_fixed & ~fixed_mask
        fixed_mask = fixed_mask | extra_mask
        fixed_values = np.concatenate([fixed_values, lb[extra_mask]])
        # Keep order consistent by reconstructing with index map below

    # Build consistent index maps for fixed and free
    all_idx = np.arange(n)
    fixed_idx = all_idx[fixed_mask]
    free_idx = all_idx[~fixed_mask]
    k = fixed_idx.size
    nv = free_idx.size

    # Trivial cases
    if nv == 0:
        # All fixed: check feasibility Kx=r
        x_full = np.zeros(n)
        x_full[fixed_idx] = fixed_values
        # clip to [lb, ub] (should already satisfy)
        x_full = np.clip(x_full, lb, ub)
        affine_residual = np.linalg.norm(K @ x_full - r)
        feasible_affine = affine_residual <= tol_consistency
        feasible_bounds = np.all(x_full >= lb - 1e-12) and np.all(x_full <= ub + 1e-12)
        if not feasible_affine:
            raise ValueError("All variables are fixed but K x = r is infeasible with given fixed values.")
        return x_full, {
            "iters": 0,
            "affine_residual": float(affine_residual),
            "changed_norm": 0.0,
            "feasible_affine": bool(feasible_affine),
            "feasible_bounds": bool(feasible_bounds),
        }

    # ---------------------
    # Reduce to free variables
    # ---------------------
    Kv = K[:, free_idx]          # (m, nv)
    Kf = K[:, fixed_idx] if k > 0 else np.zeros((m, 0))
    r_prime = r - (Kf @ fixed_values if k > 0 else 0.0)

    xbar_v = xbar[free_idx]
    lb_v = lb[free_idx]
    ub_v = ub[free_idx]

    # Consistency check for the reduced affine set: does there exist x_v s.t. Kv x_v = r'?
    # We test via least-squares residual
    if m > 0:
        sol_ls, *_ = np.linalg.lstsq(Kv, r_prime, rcond=None)
        res_norm = np.linalg.norm(Kv @ sol_ls - r_prime)
        if res_norm > 1e-8 and verbose:
            # Not necessarily fatal here—projection onto the *intersection* requires affine feasibility,
            # but Dykstra needs the affine set to be non-empty. If inconsistent, raise a clear error.
            pass
        if res_norm > tol_consistency:
            raise ValueError(
                "Inconsistent constraints after applying fixed values: K_v x_v = r' has no solution "
                f"(least-squares residual {res_norm:.3e} > tol_consistency={tol_consistency})."
            )

    # ---------------------
    # Projections (helpers)
    # ---------------------
    def proj_box(y: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(y, lb_v), ub_v)

    if m == 0:
        # No equality constraints: solution is just clipping the target on free vars
        x_v = proj_box(xbar_v)
        x_full = np.zeros(n)
        x_full[free_idx] = x_v
        x_full[fixed_idx] = fixed_values
        affine_residual = 0.0
        return x_full, {
            "iters": 1,
            "affine_residual": float(affine_residual),
            "changed_norm": float(np.linalg.norm(x_v - xbar_v)),
            "feasible_affine": True,
            "feasible_bounds": True,
        }

    # Projection onto the affine set S_v = {x_v : Kv x_v = r'}
    # Given y, find y' = argmin ||y' - y|| s.t. Kv y' = r'
    # y' = y - Kv^T * lambda, where (Kv Kv^T) lambda = Kv y - r'
    # We compute lambda by least squares to handle rank-deficiency robustly.
    KvKvT = Kv @ Kv.T  # (m, m)
    def proj_affine(y: np.ndarray) -> np.ndarray:
        rhs = Kv @ y - r_prime
        # Solve (KvKvT) lambda ≈ rhs
        lam, *_ = np.linalg.lstsq(KvKvT, rhs, rcond=None)
        return y - Kv.T @ lam

    # ---------------------
    # Initialization
    # ---------------------
    # Reasonable start: project xbar_v onto the affine set first, then clip.
    x = proj_affine(xbar_v)
    x = proj_box(x)

    # Dykstra's algorithm between affine and box:
    #   y = x + p ; xA = Proj_affine(y); p = y - xA
    #   y = xA + q; x  = Proj_box(y);   q = y - x
    p = np.zeros_like(x)  # residual for affine
    q = np.zeros_like(x)  # residual for box

    changed = np.inf
    it = 0
    for it in range(1, max_iters + 1):
        x_prev = x.copy()

        # Affine step
        y = x + p
        xA = proj_affine(y)
        p = y - xA

        # Box step
        y = xA + q
        x = proj_box(y)
        q = y - x

        changed = np.linalg.norm(x - x_prev)
        if verbose and (it % 100 == 0 or changed < tol):
            # Compute affine residual for monitoring
            aff_res = np.linalg.norm(Kv @ x - r_prime)
            print(f"[Iter {it}] Δ={changed:.3e}, affine_res={aff_res:.3e}")

        if changed < tol:
            break

    # ---------------------
    # Reassemble full solution & diagnostics
    # ---------------------
    x_full = np.zeros(n)
    x_full[free_idx] = x
    x_full[fixed_idx] = fixed_values

    affine_residual = float(np.linalg.norm(K @ x_full - r))
    feasible_affine = affine_residual <= 1e2 * tol  # looser at final iterate
    feasible_bounds = bool(np.all(x_full >= lb - 1e-10) and np.all(x_full <= ub + 1e-10))

    info = {
        "iters": it,
        "affine_residual": affine_residual,
        "changed_norm": float(changed),
        "feasible_affine": feasible_affine,
        "feasible_bounds": feasible_bounds,
    }
    return x_full, info


import numpy as np
from typing import Optional, Tuple, Dict
from scipy.optimize import minimize, Bounds, LinearConstraint

def solve_qp_closest_to_xbar_with_fixed_scipy(
    K: np.ndarray,
    r: np.ndarray,
    xbar: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    fixed_mask: np.ndarray,
    fixed_values: Optional[np.ndarray] = None,
    trust_tol: float = 1e-12,
    max_iter: int = 10_000,
    verbose: bool = False,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Solve: minimize 0.5*||x - xbar||^2 s.t. Kx=r, lb <= x <= ub, and x[fixed_mask]=fixed_values.
    Uses SciPy's trust-constr solver with an exact quadratic objective (I, -xbar).
    Reduces to the free variables first to keep the equality constraints well-conditioned.

    Returns (x_opt, info).
    """
    # --- Validate inputs ---
    K = np.asarray(K, dtype=float); r = np.asarray(r, dtype=float).reshape(-1)
    xbar = np.asarray(xbar, dtype=float).reshape(-1)
    lb = np.asarray(lb, dtype=float).reshape(-1); ub = np.asarray(ub, dtype=float).reshape(-1)
    fixed_mask = np.asarray(fixed_mask, dtype=bool).reshape(-1)

    m, n = K.shape
    if r.shape != (m,): raise ValueError(f"r must be shape ({m},), got {r.shape}")
    if xbar.shape != (n,): raise ValueError(f"xbar must be shape ({n},), got {xbar.shape}")
    if lb.shape != (n,) or ub.shape != (n,): raise ValueError("lb, ub must be shape (n,)")
    if np.any(lb > ub):
        bad = np.where(lb > ub)[0]; raise ValueError(f"Infeasible bounds at indices {bad}")

    # Merge hard-equality bounds into fixed set
    eq_fix = np.isclose(lb, ub, rtol=0.0, atol=0.0)
    merged_fixed_mask = fixed_mask | eq_fix

    # Fixed values array
    if fixed_values is None:
        fixed_values = xbar[fixed_mask].copy()
    else:
        fixed_values = np.asarray(fixed_values, dtype=float).reshape(-1)
        if fixed_values.shape != (fixed_mask.sum(),):
            raise ValueError("fixed_values length must match fixed_mask True count.")

    # Compose the consistent fixed vector in the merged set
    all_idx = np.arange(n)
    fixed_idx_explicit = all_idx[fixed_mask]
    fixed_idx_bounds = all_idx[eq_fix & ~fixed_mask]
    merged_fixed_idx = np.concatenate([fixed_idx_explicit, fixed_idx_bounds])
    merged_fixed_vals = np.concatenate([fixed_values, lb[fixed_idx_bounds]])  # lb==ub here

    # Bounds sanity for fixed
    if np.any(merged_fixed_vals < lb[merged_fixed_idx] - 1e-15) or np.any(merged_fixed_vals > ub[merged_fixed_idx] + 1e-15):
        raise ValueError("Some fixed values violate their bounds.")

    # Free index set
    free_mask = ~merged_fixed_mask
    free_idx = all_idx[free_mask]
    nv = free_idx.size

    # Trivial case: all fixed — validate feasibility and return
    if nv == 0:
        x_full = np.zeros(n)
        x_full[merged_fixed_idx] = merged_fixed_vals
        if np.any(x_full < lb - 1e-15) or np.any(x_full > ub + 1e-15):
            raise ValueError("All variables fixed but violate bounds.")
        aff_res = float(np.linalg.norm(K @ x_full - r))
        if aff_res > trust_tol:
            raise ValueError("All variables fixed but Kx=r infeasible with given fixed values.")
        return x_full, dict(status="all_fixed", iters=0, affine_residual=aff_res, feasible_bounds=True)

    # Reduced system: Kv x_v = r' := r - Kf c_f
    Kf = K[:, merged_fixed_idx] if merged_fixed_idx.size else np.zeros((m, 0))
    Kv = K[:, free_idx]
    r_prime = r - (Kf @ merged_fixed_vals if merged_fixed_idx.size else 0.0)

    # Reduced bounds and prior
    lb_v = lb[free_idx]; ub_v = ub[free_idx]; xbar_v = xbar[free_idx]

    # Quick feasibility check for equality (consistency)
    # If inconsistent, no solution exists
    if m > 0:
        sol_ls, *_ = np.linalg.lstsq(Kv, r_prime, rcond=None)
        res_norm = np.linalg.norm(Kv @ sol_ls - r_prime)
        if res_norm > 1e-10:
            raise ValueError(f"Reduced equality inconsistent: ||Kv x - r'||={res_norm:.3e}")

    # QP in free variables:
    # minimize f(xv) = 0.5 * ||xv - xbar_v||^2
    # s.t. Kv xv = r', lb_v <= xv <= ub_v
    # Gradient: g(xv) = xv - xbar_v
    # Hessian: H = I (positive definite)

    def fun(xv):
        d = xv - xbar_v
        return 0.5 * np.dot(d, d)

    def jac(xv):
        return xv - xbar_v

    def hess(xv):
        # Identity Hessian
        return np.eye(nv)

    # Constraints & bounds
    lin_con = LinearConstraint(Kv, r_prime, r_prime) if m > 0 else None
    bnds = Bounds(lb_v, ub_v)

    # Warm start: projection of xbar_v onto equality via pseudoinverse (then clip to bounds)
    if m > 0:
        # x0 = argmin ||x - xbar_v|| s.t. Kv x = r' has closed form:
        # x0 = xbar_v - Kv^T (Kv Kv^T)^{+} (Kv xbar_v - r')
        U, S, Vt = np.linalg.svd(Kv, full_matrices=False)
        rank = (S > 1e-12 * (S[0] if S.size else 1.0)).sum()
        if rank > 0:
            S_inv = np.zeros_like(S); S_inv[:rank] = 1.0 / S[:rank]
            # projector onto row-space: P_row = U_r U_r^T
            resid = U[:, :rank].T @ (Kv @ xbar_v - r_prime)
            corr = Vt[:rank, :].T @ (S_inv[:rank] * resid)
            x0 = xbar_v - corr
        else:
            # Kv ~ 0 ⇒ must have r' ~ 0; start at xbar_v
            x0 = xbar_v.copy()
    else:
        x0 = xbar_v.copy()
    x0 = np.clip(x0, lb_v, ub_v)

    options = dict(verbose=3 if verbose else 0, maxiter=max_iter, gtol=trust_tol, xtol=trust_tol, barrier_tol=trust_tol)
    res = minimize(
        fun, x0, method="trust-constr",
        jac=jac, hess=hess,
        constraints=() if lin_con is None else (lin_con,),
        bounds=bnds,
        options=options,
    )

    if not res.success and verbose:
        print(f"[trust-constr] status={res.status}, message={res.message}")

    x_v_opt = res.x
    # Reassemble
    x_full = np.zeros(n)
    x_full[free_idx] = x_v_opt
    x_full[merged_fixed_idx] = merged_fixed_vals

    # Diagnostics
    aff_res = float(np.linalg.norm(K @ x_full - r)) if m > 0 else 0.0
    feas_bounds = bool(np.all(x_full >= lb - 1e-10) and np.all(x_full <= ub + 1e-10))

    info = dict(
        status=res.status,
        message=res.message,
        success=bool(res.success),
        iters=res.niter,
        fun=float(res.fun),
        affine_residual=aff_res,
        feasible_bounds=feas_bounds,
        optimality=float(res.optimality) if hasattr(res, "optimality") else np.nan,
    )
    return x_full, info

import numpy as np
import scipy.sparse as sp
import cvxpy as cp
from numpy.linalg import norm

def closest_feasible_via_nullspace_cvxpy(
    K_sparse, r, xbar, lb, ub,
    fixed_mask=None, fixed_values=None,
    svd_rtol=1e-12, solver="OSQP",
    x_feasible=None,            # <<< optional known feasible full-length x
    verbose=False
):
    """
    Project xbar onto {x: Kx=r, lb <= x <= ub, and fixed vars}, by parameterizing
    x = x_v0 + N z with KN=0, K_v x_v0 = r'. Equality is enforced by construction.

    Notes:
      - If you pass x_feasible (a known equality-feasible vector that also satisfies bounds),
        we anchor at it so z=0 is feasible. Otherwise we anchor at the SVD pseudoinverse x_v0
        (still equality-feasible), and the solver will move in the nullspace to satisfy bounds.
    """
    # --- array hygiene ---
    K = sp.csr_matrix(K_sparse)
    r = np.asarray(r, float).reshape(-1)
    xbar = np.asarray(xbar, float).reshape(-1)
    lb = np.asarray(lb, float).reshape(-1)
    ub = np.asarray(ub, float).reshape(-1)
    m, n = K.shape
    assert r.shape == (m,)
    assert xbar.shape == (n,)
    assert lb.shape == (n,) and ub.shape == (n,)
    if fixed_mask is None:
        fixed_mask = np.zeros(n, dtype=bool)
    else:
        fixed_mask = np.asarray(fixed_mask, bool).reshape(-1)

    # Fixed values: IMPORTANT — do not silently take from xbar unless you intend to.
    # Best practice: pass fixed_values explicitly or keep xbar unchanged on fixed indices.
    if fixed_values is None:
        fixed_values = xbar[fixed_mask].copy()
    else:
        fixed_values = np.asarray(fixed_values, float).reshape(-1)
        assert fixed_values.shape == (fixed_mask.sum(),)

    # Merge equality-fixed from bounds
    eq_fix = np.isclose(lb, ub, rtol=0.0, atol=0.0)
    merged_fixed_mask = fixed_mask | eq_fix

    # Build fixed indices/values in one consistent vector
    all_idx = np.arange(n)
    fixed_idx_explicit = all_idx[fixed_mask]
    fixed_idx_bounds   = all_idx[eq_fix & ~fixed_mask]
    merged_fixed_idx   = np.concatenate([fixed_idx_explicit, fixed_idx_bounds])
    merged_fixed_vals  = np.concatenate([fixed_values, lb[fixed_idx_bounds]])  # lb==ub here

    free_mask = ~merged_fixed_mask
    free_idx  = all_idx[free_mask]
    nf = free_idx.size

    if nf == 0:
        # All fixed — verify feasibility quickly
        x_full = np.zeros(n)
        x_full[merged_fixed_idx] = merged_fixed_vals
        if not (np.all(x_full >= lb - 1e-12) and np.all(x_full <= ub + 1e-12)):
            raise ValueError("All variables fixed but violate box bounds.")
        aff_res = float(norm(K @ x_full - r))
        if aff_res > 1e-12:
            raise ValueError("All variables fixed but Kx=r infeasible with given fixed values.")
        return x_full, dict(affine_residual=aff_res, status="all_fixed")

    # Reduce Kx=r to free variables: K_v x_v = r' := r - K_f c_f
    Kv = K[:, free_idx]
    if merged_fixed_idx.size:
        r_prime = r - (K[:, merged_fixed_idx] @ merged_fixed_vals)
    else:
        r_prime = r.copy()

    xbar_v = xbar[free_idx]
    lb_v   = lb[free_idx]
    ub_v   = ub[free_idx]

    # --- Single FULL SVD to get particular solution and FULL nullspace ---
    Kv_dense = Kv.toarray()
    U, S, Vt_full = np.linalg.svd(Kv_dense, full_matrices=True)
    smax = S[0] if S.size else 0.0
    rank = int(np.sum(S > svd_rtol * max(1.0, smax)))

    # Particular solution x_v0 (equality-feasible)
    if rank > 0:
        x_v0 = (Vt_full[:rank, :].T * (1.0 / S[:rank])) @ (U[:, :rank].T @ r_prime)
    else:
        if norm(r_prime) > 1e-12:
            raise ValueError("Reduced equality inconsistent: Kv≈0 but r'≠0.")
        x_v0 = np.zeros(nf)

    # FULL nullspace basis
    N = Vt_full[rank:, :].T  # shape: (n_free, n_free - rank)
    nullity = N.shape[1]
    if verbose:
        print(f"[SVD] rank={rank}, nullity={nullity}, N.shape={N.shape}")

    # --- Choose the anchor for the QP ---
    # If you have a known feasible full x (Kx=r & in-bounds), anchor at its free slice to make z=0 feasible.
    if x_feasible is not None:
        x_feasible = np.asarray(x_feasible, float).reshape(-1)
        assert x_feasible.shape == (n,)
        # Also ensure it respects *these* fixed values (merged set)
        if merged_fixed_idx.size:
            if not np.allclose(x_feasible[merged_fixed_idx], merged_fixed_vals, atol=1e-10):
                raise ValueError("x_feasible disagrees with merged fixed values.")
        x0_free = x_feasible[free_idx].copy()
        # Check equality (should be tiny)
        if verbose:
            print("[Anchor] ||Kv x0_free - r'|| =", norm(Kv @ x0_free - r_prime))
        x_v_anchor = x0_free  # <<< equality-feasible & (should be) within bounds
    else:
        x_v_anchor = x_v0     # <<< equality-feasible particular solution

    # --- Pre-solve feasibility guards ---

    # 1) Check z=0 feasibility with current anchor
    l_shift = lb_v - x_v_anchor
    u_shift = ub_v - x_v_anchor
    z0_lb_viol = np.sum(l_shift > 1e-12)
    z0_ub_viol = np.sum(u_shift < -1e-12)
    if verbose:
        print(f"[z=0] lower violations={z0_lb_viol}, upper violations={z0_ub_viol}")

    # 2) Pinned rows: if a row of N is ~0, that coordinate cannot move; it must already be within bounds.
    if nullity > 0:
        row_norms = np.linalg.norm(N, axis=1)
        pinned = row_norms < 1e-14
        pinned_viol = np.where(pinned & ((x_v_anchor < lb_v - 1e-12) | (x_v_anchor > ub_v + 1e-12)))[0]
        if pinned_viol.size and verbose:
            print("[Pinned] violating pinned indices (first 10):", pinned_viol[:10])
        if pinned_viol.size:
            raise RuntimeError("Bounds exclude the equality manifold (pinned coords out of bounds).")
    else:
        # No nullspace: equality pins all free coords exactly; they must already satisfy bounds.
        if not (np.all(x_v_anchor >= lb_v - 1e-12) and np.all(x_v_anchor <= ub_v + 1e-12)):
            raise RuntimeError("No nullspace and equality solution violates bounds — infeasible.")

    # 3) Direct contradictions on each coordinate (l_shift > u_shift)
    bad = np.where(l_shift > u_shift + 1e-14)[0]
    if bad.size and verbose:
        print("[Contradictions] count:", bad.size, " sample:", bad[:10])
    if bad.size:
        raise RuntimeError("Found l_shift > u_shift for some rows — infeasible bounds for this anchor ordering.")

    # --- Build and solve the QP in z ---
    z = cp.Variable(nullity)
    expr = x_v_anchor + N @ z
    d    = xbar_v - x_v_anchor

    # Only impose constraints on finite bounds (avoid passing +/-inf rows)
    cons = []
    mask_lb = np.isfinite(lb_v)
    mask_ub = np.isfinite(ub_v)
    if mask_lb.any():
        cons.append(expr[mask_lb] >= lb_v[mask_lb])
    if mask_ub.any():
        cons.append(expr[mask_ub] <= ub_v[mask_ub])

    prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(N @ z - d)), cons)

    # Try OSQP first, then ECOS if it returns infeasible
    try_solvers = [solver] if solver else ["ECOS"]
    if "OSQP" not in try_solvers:
        try_solvers.append("OSQP")

    last_status = None
    for s in try_solvers:
        try:
            prob.solve(solver=s, eps_abs=1e-10, eps_rel=1e-15, verbose=verbose)
            last_status = prob.status
            if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                break
        except Exception as e:
            last_status = f"exception: {e}"

    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"QP solver failed: status={last_status}")

    z_star = np.zeros(nullity) if nullity == 0 else np.asarray(z.value, float).reshape(-1)
    x_v = x_v_anchor + (N @ z_star if nullity > 0 else 0.0)

    # Reassemble full x
    x_full = np.zeros(n)
    x_full[free_idx] = x_v
    x_full[merged_fixed_idx] = merged_fixed_vals

    info = {
        "status": prob.status,
        "rank": rank,
        "nullity": int(nullity),
        "affine_residual": float(norm(K @ x_full - r)),
        "bounds_ok": bool(np.all(x_full >= lb - 1e-9) and np.all(x_full <= ub + 1e-9)),
        "obj": float(0.5 * np.sum((x_full - xbar) ** 2)),
        "null_basis": N,
        "free_idx": free_idx,
    }
    return x_full, info

class ConstrainedSolver:
    def solve(self, K: np.ndarray, r: np.ndarray, param_def: ParameterDefinition, x0:
              np.ndarray = None, verbose = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solves: min ||theta - prior||^2
        Subject to:
            1. K * theta = r (Mass Balance)
            2. LB <= theta <= UB (Parameter Bounds)
            3. Fixed Parameters
        """
        prior = param_def.priors
        bounds = param_def.bounds

        target = x0 if x0 is not None else prior

        fixed_mask = np.array([p.is_fixed for p in param_def.params], dtype=bool)
        fixed_values = param_def.priors[fixed_mask]
        lb = np.array([b[0] for b in bounds])
        ub = np.array([b[1] for b in bounds])

        # Use the robust projection utility
        theta_opt, info = closest_feasible_via_nullspace_cvxpy(
            K, r, target, lb, ub, fixed_mask, fixed_values=fixed_values, verbose=verbose#, max_iter = 100, verbose=True
        )


        # Identify Active Set
        active_mask = fixed_mask.copy()
        tol = 1e-6
        for i, (l, u) in enumerate(bounds):
            if not fixed_mask[i]:
                if theta_opt[i] <= l + tol or theta_opt[i] >= u - tol:
                    active_mask[i] = True

        return theta_opt, active_mask

    def sample(
        self,
        K: np.ndarray,
        r: np.ndarray,
        param_def: ParameterDefinition,
        n_samples: int = 1000,
        sigma: float = 0.05,
        x_center: Optional[np.ndarray] = None,
        x_anchor: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Samples from the nullspace of the constraints K*theta=r around a center point.

        Args:
            K: Constraint matrix
            r: Constraint vector
            param_def: Parameter definitions
            n_samples: Number of samples to generate
            sigma: Standard deviation for the Gaussian sampling in nullspace
            x_center: Center point for the Gaussian distribution (in full parameter space).
                      If None, uses param_def.priors.
            x_anchor: A feasible point satisfying K*theta=r. If None, one will be computed.
        """
        prior = param_def.priors
        bounds = param_def.bounds

        fixed_mask = np.array([p.is_fixed for p in param_def.params], dtype=bool)
        fixed_values = param_def.priors[fixed_mask]
        lb = np.array([b[0] for b in bounds])
        ub = np.array([b[1] for b in bounds])

        target = x_center if x_center is not None else prior

        x_full, info = closest_feasible_via_nullspace_cvxpy(
            K, r, target, lb, ub, fixed_mask, fixed_values=fixed_values,
            x_feasible=x_anchor, verbose=False
        )

        #if not info['bounds_ok'] or info['affine_residual'] > 1e-6:
        #     print(f"Warning: Starting point for sampling might be infeasible. Bounds OK: {info['bounds_ok']}, Residual: {info['affine_residual']}")

        if x_anchor is None:
            x_anchor = x_full

        N = info["null_basis"]
        free_idx = info["free_idx"]

        if x_center is None:
            x_center = prior

        x_center_free = x_center[free_idx]
        x_anchor_free = x_anchor[free_idx]

        lb_v = lb[free_idx]
        ub_v = ub[free_idx]

        samples_free = sample_nullspace_gaussian(
            Q=N,
            x_anchor=x_anchor_free,
            lb_v=lb_v,
            ub_v=ub_v,
            x_center=x_center_free,
            sigma=sigma,
            n_samples=n_samples
        )

        if len(samples_free) == 0:
             print("Error: sample_nullspace_gaussian returned 0 samples.")

        n_params = len(prior)
        full_samples = np.zeros((n_samples, n_params))
        full_samples[:] = x_anchor
        full_samples[:, free_idx] = samples_free

        return full_samples

def _sample_truncnorm(mean, sd, a, b, rng):
    """Sample from N(mean, sd^2) truncated to [a, b]. Uses SciPy if available, else crude fallback."""
    try:
        from scipy.stats import truncnorm
        alpha = (a - mean) / sd
        beta  = (b - mean) / sd
        return truncnorm.rvs(alpha, beta, loc=mean, scale=sd, random_state=rng)
    except Exception:
        # Fallback: simple rejection (works well enough if truncation isn't extreme)
        # Or if sd is very small/large or bounds are weird
        for _ in range(20_000):
            t = rng.normal(mean, sd)
            if a <= t <= b:
                return t
        return 0.5 * (a + b)  # last-resort midpoint

def _feasible_interval(Q, y, x_anchor, lb_v, ub_v, u):
    """
    Compute [tmin, tmax] s.t. lb_v <= x_anchor + Q(y + t u) <= ub_v.
    Returns (tmin, tmax, feasible_bool).
    """
    # w is direction in x-space along which x moves when we change t
    w = Q @ u              # shape (n_free,)
    x_curr = x_anchor + Q @ y
    tmin, tmax = -np.inf, np.inf

    for i in range(w.shape[0]):
        wi = w[i]
        if np.abs(wi) < 1e-15:
            # No motion in this coordinate; must already be within bounds
            if not (lb_v[i] - 1e-9 <= x_curr[i] <= ub_v[i] + 1e-9):
                return None, None, False
            continue
        # From lb: x_curr[i] + wi * t >= lb_v[i]  => t >= (lb_v[i] - x_curr[i]) / wi  if wi>0
        # From ub: x_curr[i] + wi * t <= ub_v[i]  => t <= (ub_v[i] - x_curr[i]) / wi  if wi>0
        t1 = (lb_v[i] - x_curr[i]) / wi
        t2 = (ub_v[i] - x_curr[i]) / wi
        if wi > 0:
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
        else:
            tmin = max(tmin, t2)
            tmax = min(tmax, t1)
        if tmin > tmax + 1e-9: # Tolerance
            return None, None, False
        if tmin > tmax:
             tmin = tmax = (tmin + tmax) / 2.0

    return tmin, tmax, True

def sample_nullspace_gaussian(
    Q,                      # (n_free, k) orthonormal nullspace basis
    x_anchor,               # (n_free,) equality-feasible anchor (e.g., x_opt[free])
    lb_v, ub_v,             # (n_free,) bounds for free variables
    x_center=None,          # (n_free,) center in x-space (e.g., prior or x_opt); default = x_anchor
    sigma=0.05,             # std in y-space; controls spread around center
    n_samples=1_000,
    burn_in=200,
    thinning=1,
    rng=None
):
    """
    Truncated-Gaussian hit-and-run sampler in nullspace coordinates.
    Returns an array X of shape (n_samples, n_free) with feasible equality solutions.
    """
    rng = np.random.default_rng() if rng is None else rng
    n_free, k = Q.shape
    assert x_anchor.shape == (n_free,)
    assert lb_v.shape == (n_free,) and ub_v.shape == (n_free,)

    # Center in y-space
    if x_center is None:
        x_center = x_anchor.copy()
    mu_y = Q.T @ (x_center - x_anchor)  # target mean in y-space

    # Initialize at y=0 -> x = x_anchor (feasible)
    y = np.zeros(k)
    samples = []

    total_iters = burn_in + n_samples * thinning

    # Debug counters
    n_interval_failures = 0
    n_t_infinite = 0

    for it in range(total_iters):
        # Random direction on unit sphere in y-space
        u = rng.normal(size=k)
        u_norm = np.linalg.norm(u)
        if u_norm == 0.0:
            continue
        u /= u_norm

        # Feasible interval along this direction
        tmin, tmax, ok = _feasible_interval(Q, y, x_anchor, lb_v, ub_v, u)

        if not ok:
            n_interval_failures += 1
            # Retry direction
            continue

        if not np.isfinite(tmin) or not np.isfinite(tmax):
            n_t_infinite += 1
            continue

        # Conditional along the line: t ~ N(m, sigma^2) truncated to [tmin, tmax]
        m = u @ (mu_y - y)
        t = _sample_truncnorm(m, sigma, tmin, tmax, rng)

        # Move
        y = y + t * u

        # Collect after burn-in and thinning
        if it >= burn_in and ((it - burn_in) % thinning == 0):
            x = x_anchor + Q @ y
            samples.append(x.copy())

    if len(samples) == 0:
        print(f"Debug: Sampling failed. Total Iters: {total_iters}. Interval Failures: {n_interval_failures}. Infinite T: {n_t_infinite}")

    return np.array(samples)
