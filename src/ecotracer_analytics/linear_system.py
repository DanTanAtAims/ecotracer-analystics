import numpy as np

from .core import EcopathSource, EcotracerSource

def calculate_intercept(
    path_source: EcopathSource, tracer_source: EcotracerSource
) -> np.ndarray:
    """
    Calculates the intercept vector b.
    b = [Env_Inflow, Immig_1, ..., Immig_N]
    """
    num_groups = len(path_source.biomass)
    b = np.zeros(num_groups + 1)

    # Environment
    b[0] = tracer_source.base_inflow

    # Biota/Detritus (Immigration)
    # Gain = Immig_Conc * Immig_Biomass_Rate
    # Ensure inputs are numpy arrays
    immig_conc = tracer_source.immigration_c
    immig_rate = np.array(path_source.immigration_rate)
    b[1:] = immig_conc * immig_rate

    return b

def calculate_env_coefficient(
    path_source: EcopathSource, tracer_source: EcotracerSource
):
    """
    Calculates the environment diagonal coefficient (M_0,0).
    M_0,0 = Decay + Vol_Exchange + Sum(Direct_Uptake)
    """
    # Direct Uptake Rate (from Env) = u_i * B_i
    total_uptake = np.sum(tracer_source.dir_abs_r * path_source.biomass)

    return tracer_source.env_decay + tracer_source.env_volume_exchange_loss + total_uptake

def calculate_detritus_coefficient(
    path_source: EcopathSource, tracer_source: EcotracerSource
):
    """
    Calculates diagonal coefficients (beta) for detritus groups.
    beta_k = (Sum_Pred Q_pk / B_k) + Phys_Decay + Meta_Decay + Det_Out
    """
    det_indices = path_source.is_detritus
    bio_indices = path_source.not_detritus
    biomass = path_source.biomass[det_indices]

    # Consumption of detritus k by all predators p: Sum over columns for row k
    # consumption is Prey x Predator
    total_predation = np.sum(path_source.consumption[np.ix_(det_indices, bio_indices)], axis=1)

    # Rates
    phys_dec = tracer_source.phys_dec_r[det_indices]
    meta_dec = tracer_source.meta_dec_r[det_indices]
    det_out = path_source.detritus_out_rates[det_indices]

    # Avoid division by zero if biomass is 0
    predation_rate = np.zeros_like(biomass)
    mask = biomass > 0
    predation_rate[mask] = total_predation[mask] / biomass[mask]

    return predation_rate + phys_dec + meta_dec + det_out

def calculate_biota_coefficient(
    path_source: EcopathSource, tracer_source: EcotracerSource
):
    """
    Calculates diagonal coefficients (beta) for living groups.
    beta_i = Z_i + Meta + Phys - Cannibalism_Correction
    """
    bio_indices = path_source.not_detritus

    # Total Mortality Z
    fishing = path_source.fishing_mort_rate[bio_indices]
    predation = path_source.predation_mort_rate[bio_indices]
    other = path_source.other_mort_rate[bio_indices]
    emigration = path_source.emigration_rate[bio_indices]

    z_mort = fishing + predation + other + emigration

    # Rates
    meta = tracer_source.meta_dec_r[bio_indices]
    phys = tracer_source.phys_dec_r[bio_indices]

    # Cannibalism Correction: AE_i * Q_ii / B_i
    biomass = path_source.biomass[bio_indices]
    ae = tracer_source.assim_eff[bio_indices]

    cannibalism_term = np.zeros_like(biomass)

    # efficient extraction of diagonals for bio_indices from full matrix
    full_cons = path_source.consumption
    q_ii = full_cons[bio_indices, bio_indices]

    mask = biomass > 0
    cannibalism_term[mask] = ae[mask] * q_ii[mask] / biomass[mask]

    return z_mort + meta + phys - cannibalism_term

def calculate_coefficient(
    path_source: EcopathSource, tracer_source: EcotracerSource
) -> np.ndarray:
    """
    Assembles the full coefficient matrix M.
    """
    N = len(path_source.biomass)
    M = np.zeros((N + 1, N + 1))

    # --- Diagonals ---
    M[0, 0] = calculate_env_coefficient(path_source, tracer_source)

    bio_indices = path_source.not_detritus
    M[bio_indices + 1, bio_indices + 1] = calculate_biota_coefficient(path_source, tracer_source)

    det_indices = path_source.is_detritus
    M[det_indices + 1, det_indices + 1] = calculate_detritus_coefficient(path_source, tracer_source)
    # --- Off-Diagonals ---
    biomass = path_source.biomass
    cons = path_source.consumption # Prey x Pred

    # 1. Environment Row (0, j): Gains from Biota/Detritus
    # -(Meta + Det_Out + Unassimilated)
    gains_direct = tracer_source.meta_dec_r + path_source.detritus_out_rates
    M[0, 1:] -= gains_direct

    # Unassimilated: sum_pred (1-AE_p) * Q_pj / B_j
    unassim_flow = np.sum(
        cons[:, bio_indices] / biomass[:, np.newaxis]
        * (1 - tracer_source.assim_eff[np.newaxis, bio_indices])
    , axis = 1)

    M[0, 1:] -= unassim_flow

    # 2. Biota/Detritus Rows (i, 0): Direct Uptake from Env
    uptake = tracer_source.dir_abs_r * biomass
    M[1:, 0] = -uptake

    # 3. Biota Rows (i, j): Dietary Uptake
    # M[i, j] = - AE_i * Q_ij / B_j
    inv_biomass = np.zeros_like(biomass)
    inv_biomass = 1.0 / biomass

    # diet_matrix[i, j] (Pred i, Prey j)
    # Transpose cons to get Pred x Prey
    diet_matrix = (cons.T * inv_biomass[np.newaxis, :]) * tracer_source.assim_eff[:, np.newaxis]

    # Zero out diagonal to avoid double counting cannibalism (handled in M_ii)
    np.fill_diagonal(diet_matrix, 0)
    diet_matrix[det_indices, :] = 0
    M[1:, 1:] -= diet_matrix

    # 4. Detritus Rows (k, j): Gains from Other Mort + Discards
    m0 = path_source.other_mort_rate[bio_indices]
    fate = path_source.det_fate # Living x Detritus

    # Detritus Consumption

    # M0 term: (m0 * fate).T -> Detritus x Living
    m0_term = (fate * m0[:, None]).T

    # Discards term
    discards = path_source.discards[:, bio_indices] # Fleet x Living
    dis_mort = path_source.discard_mortality[:, bio_indices]
    b_living = biomass[bio_indices]

    d_rate = np.zeros_like(discards)
    mask_l = b_living > 0
    d_rate[:, mask_l] = (discards[:, mask_l] / b_living[mask_l]) * dis_mort[:, mask_l]

    discard_fate = path_source.discard_fate # Fleet x Detritus

    # (Det x Fleet) @ (Fleet x Living) -> Det x Living
    discard_term = discard_fate.T @ d_rate

    total_det_gain = m0_term + discard_term

    # Update block M[det_row, bio_col]
    rows = det_indices + 1
    cols = bio_indices + 1
    M[np.ix_(rows, cols)] -= total_det_gain

    return M

def calculate_equilibrium(
    path_source: EcopathSource, tracer_source: EcotracerSource
) -> np.ndarray:
    """
    Calculates the equilibrium state vector x by solving M * x = b.
    Returns:
        np.ndarray: [C_env, A_1, ..., A_N]
    """
    b = calculate_intercept(path_source, tracer_source)
    M = calculate_coefficient(path_source, tracer_source)

    try:
        # Standard solver for square, non-singular matrices
        return np.linalg.solve(M, b)
    except np.linalg.LinAlgError:
        # Fallback to least-squares solver if the matrix is singular or near-singular
        print("Least Squares")
        return np.linalg.lstsq(M, b, rcond=None)[0]
