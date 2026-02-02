import numpy as np

from .core import EcopathSource, EcotracerSource

def calculate_intercept(
    path_source: EcopathSource, tracer_source: EcotracerSource
) -> np.ndarray:
    """
    Calculates the intercept vector b for the Concentration-based system.
    Matches the 'independent' alpha terms from equilibrium.ipynb.
    """
    num_groups = len(path_source.biomass)
    b = np.zeros(num_groups + 1)

    b[0] = tracer_source.base_inflow

    immig_conc = tracer_source.immigration_c
    immig_rate = np.array(path_source.immigration_rate)
    b[1:] = immig_conc * immig_rate

    return b

def calculate_coefficient(
    path_source: EcopathSource, tracer_source: EcotracerSource
) -> np.ndarray:
    """
    Assembles the full coefficient matrix M for the Concentration-based system.
    System: dC/dt = Alpha - Beta * C = 0  =>  Beta * C - Alpha_dependent = Alpha_independent
    Rearranged for Mx = b:
    (Beta_diagonal - Dependent_Alpha_Coeffs) * x = b
    """
    N = len(path_source.biomass)
    M = np.zeros((N + 1, N + 1))

    biomass = path_source.biomass

    beta_env = (
        tracer_source.env_decay +
        tracer_source.env_volume_exchange_loss +
        np.sum(tracer_source.dir_abs_r * biomass)
    )
    M[0, 0] = beta_env

    z_mortality = (
        path_source.fishing_mort_rate +
        path_source.predation_mort_rate +
        path_source.other_mort_rate +
        path_source.emigration_rate
    )
    beta_biota = z_mortality + tracer_source.meta_dec_r + tracer_source.phys_dec_r

    det_indices = path_source.is_detritus
    bio_indices = path_source.not_detritus

    M[np.arange(1, N+1), np.arange(1, N+1)] = beta_biota

    cons = path_source.consumption
    det_flow = np.sum(cons[det_indices, :], axis=1)

    det_biomass = biomass[det_indices]
    det_loss_rate = np.zeros_like(det_biomass)
    mask = det_biomass > 0
    det_loss_rate[mask] = det_flow[mask] / det_biomass[mask]

    beta_det = (
        det_loss_rate +
        tracer_source.phys_dec_r[det_indices] +
        tracer_source.meta_dec_r[det_indices] +
        path_source.detritus_out_rates[det_indices]
    )

    M[det_indices + 1, det_indices + 1] = beta_det

    M[0, 1:] -= tracer_source.meta_dec_r
    unassim_total_flux = np.sum(
        cons[:, bio_indices] * (1 - tracer_source.assim_eff[np.newaxis, bio_indices]),
        axis=1
    )
    inv_biomass = np.zeros_like(biomass)
    mask_b = biomass > 0
    inv_biomass[mask_b] = 1.0 / biomass[mask_b]

    unassim_coeffs = unassim_total_flux * inv_biomass
    M[0, 1:] -= unassim_coeffs

    det_out_coeffs = np.zeros(N)
    det_out_coeffs[det_indices] = path_source.detritus_out_rates[det_indices]
    M[0, 1:] -= det_out_coeffs


    M[1:, 0] -= tracer_source.dir_abs_r * biomass

    prey_specific_loss = cons * inv_biomass[:, np.newaxis] # Row j scaled by 1/B_j
    pred_gain_matrix = prey_specific_loss.T # i x j

    diet_coeffs = pred_gain_matrix * tracer_source.assim_eff[:, np.newaxis]

    diet_coeffs[det_indices, :] = 0
    M[1:, 1:] -= diet_coeffs

    raw_discards = path_source.discards
    dis_mort = path_source.discard_mortality
    b_all = biomass

    disc_loss_rate = np.zeros_like(raw_discards)
    mask_b = b_all > 0
    disc_loss_rate[:, mask_b] = (raw_discards[:, mask_b] / b_all[mask_b]) * dis_mort[:, mask_b]

    discard_fate = path_source.discard_fate

    det_gain_from_discards = discard_fate.T @ disc_loss_rate

    other_mort = path_source.other_mort_rate[bio_indices] # Living
    det_fate = path_source.det_fate # Living x Detritus

    det_gain_from_m0 = (det_fate * other_mort[:, np.newaxis]).T

    rows = det_indices + 1
    cols_all = np.arange(1, N+1)
    M[np.ix_(rows, cols_all)] -= det_gain_from_discards

    cols_bio = bio_indices + 1
    M[np.ix_(rows, cols_bio)] -= det_gain_from_m0

    return M

def calculate_equilibrium(
    path_source: EcopathSource, tracer_source: EcotracerSource
) -> np.ndarray:
    """
    Calculates the equilibrium state vector C (Concentration)
    by solving M * C = b.
    """
    b = calculate_intercept(path_source, tracer_source)
    M = calculate_coefficient(path_source, tracer_source)

    try:
        return np.linalg.solve(M, b)
    except np.linalg.LinAlgError:
        print("Least Squares")
        return np.linalg.lstsq(M, b, rcond=None)[0]

def calculate_beta(
    path_source: EcopathSource, tracer_source: EcotracerSource
) -> np.ndarray:
    """
    Extracts the Beta parameters (Total Loss Rates) for all groups (Env + Biota).
    Returns a 1D array of size N+1.
    """
    # Beta is exactly the diagonal of the coefficient matrix M in this formulation.
    M = calculate_coefficient(path_source, tracer_source)
    return np.diag(M).copy()

def calculate_alpha(
    path_source: EcopathSource, tracer_source: EcotracerSource, x_eq: np.ndarray = None
) -> np.ndarray:
    """
    Calculates the Alpha parameters (Total Gains) for all groups (Env + Biota).
    Alpha = Independent_Inflow + Gains_from_Other_Groups + Cannibalism_Gain

    If x_eq is None, it is calculated automatically.
    """
    if x_eq is None:
        x_eq = calculate_equilibrium(path_source, tracer_source)

    M = calculate_coefficient(path_source, tracer_source)
    b = calculate_intercept(path_source, tracer_source)
    beta = np.diag(M)

    N = len(path_source.biomass)
    beta_pure = np.zeros(N + 1)

    beta_pure[0] = (
        tracer_source.env_decay +
        tracer_source.env_volume_exchange_loss +
        np.sum(tracer_source.dir_abs_r * path_source.biomass)
    )

    z_mortality = (
        path_source.fishing_mort_rate +
        path_source.predation_mort_rate +
        path_source.other_mort_rate +
        path_source.emigration_rate
    )
    beta_pure[1:] = z_mortality + tracer_source.meta_dec_r + tracer_source.phys_dec_r

    det_indices = path_source.is_detritus
    cons = path_source.consumption
    det_flow = np.sum(cons[det_indices, :], axis=1)
    det_biomass = path_source.biomass[det_indices]

    det_loss_rate = np.zeros_like(det_biomass)
    mask = det_biomass > 0
    det_loss_rate[mask] = det_flow[mask] / det_biomass[mask]

    beta_det = (
        det_loss_rate +
        tracer_source.phys_dec_r[det_indices] +
        tracer_source.meta_dec_r[det_indices] +
        path_source.detritus_out_rates[det_indices]
    )
    beta_pure[det_indices + 1] = beta_det

    Gains_Matrix = np.diag(beta_pure) - M

    alpha = b + Gains_Matrix @ x_eq

    return alpha

def calculate_beta_pure(
    path_source: EcopathSource, tracer_source: EcotracerSource
) -> np.ndarray:
    """
    Returns the pure Beta (Total Loss Rate) vector, uncorrected for cannibalism.
    Matches 'beta' in equilibrium.ipynb.
    """
    # Reusing the logic from calculate_alpha
    N = len(path_source.biomass)
    beta_pure = np.zeros(N + 1)

    # Env
    beta_pure[0] = (
        tracer_source.env_decay +
        tracer_source.env_volume_exchange_loss +
        np.sum(tracer_source.dir_abs_r * path_source.biomass)
    )

    # Biota
    z_mortality = (
        path_source.fishing_mort_rate +
        path_source.predation_mort_rate +
        path_source.other_mort_rate +
        path_source.emigration_rate
    )
    beta_pure[1:] = z_mortality + tracer_source.meta_dec_r + tracer_source.phys_dec_r

    # Detritus
    det_indices = path_source.is_detritus
    bio_indices = path_source.not_detritus
    cons = path_source.consumption

    # Sum only over Living predators (prey_idx is det_indices, predator_idx is bio_indices)
    # consumption is (prey x predator)
    det_flow = np.sum(cons[np.ix_(det_indices, bio_indices)], axis=1)

    det_biomass = path_source.biomass[det_indices]

    det_loss_rate = np.zeros_like(det_biomass)
    mask = det_biomass > 0
    det_loss_rate[mask] = det_flow[mask] / det_biomass[mask]

    beta_det = (
        det_loss_rate +
        tracer_source.phys_dec_r[det_indices] +
        tracer_source.meta_dec_r[det_indices] +
        path_source.detritus_out_rates[det_indices]
    )
    beta_pure[det_indices + 1] = beta_det

    return beta_pure
