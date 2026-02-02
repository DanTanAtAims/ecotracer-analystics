import pyewe as ewe
import numpy as np
import pandas as pd
import ecotracer_analytics as etca
import os

def test_dynamics():
    # Load the EwE model
    model_path = "/home/dtan/repos/pyewe_sandbox/East Bass Strait.eiixml"
    
    print(f"Loading model from {model_path}...")
    scen_int = ewe.EwEScenarioInterface(
        model_path, 
        ecosim_scenario="constant fishing", 
        ecotracer_scenario="smaller_equi"
    )

    # Initialize sources
    path_source = etca.EcopathSource(scen_int._core_instance)
    tracer_source = etca.EcotracerSource(scen_int._core_instance)

    # 1. Calculate Analytical Equilibrium (Now Concentrations)
    x_eq = etca.calculate_equilibrium(path_source, tracer_source)
    print("Analytical equilibrium calculated (Concentrations).")

    # Check derivative at equilibrium
    M = etca.calculate_coefficient(path_source, tracer_source)
    b = etca.calculate_intercept(path_source, tracer_source)
    print(f"Matrix M condition number: {np.linalg.cond(M):.2e}")
    deriv_at_eq = b - M @ x_eq
    print(f"Max absolute derivative at analytical equilibrium: {np.max(np.abs(deriv_at_eq)):.2e}")

    # 2. Setup simulation
    # Initial concentrations: let's start at 0.5 * equilibrium
    # x_eq is [C_env, C_1, ..., C_N]
    
    y0 = x_eq * 0.5

    t_span = (0, 1000) # Simulate for 1000 years
    t_eval = np.linspace(0, 1000, 11)

    print(f"Starting simulation for t={t_span}...")
    sol = etca.simulate_dynamics(path_source, tracer_source, y0, t_span, t_eval=t_eval, method="LSODA")

    if sol.success:
        print("Simulation successful.")
        # Check trajectory for index 58
        idx_worst = 58
        print(f"Trajectory for index {idx_worst}:")
        for t, val in zip(sol.t, sol.y[idx_worst, :]):
            print(f"  t={t:4.0f}, val={val:.2e}")
        
        # Check if the final state is close to equilibrium
        y_final = sol.y[:, -1]
        
        # We expect convergence towards x_eq
        # Handle zero or near-zero denominators
        denom = np.where(np.abs(x_eq) > 1e-15, x_eq, 1e-15)
        rel_diff = np.abs((y_final - x_eq) / denom)
        
        # Focus on groups where concentration is significant
        mask = x_eq > 1e-10
        if np.any(mask):
            max_rel_diff_sig = np.max(rel_diff[mask])
            print(f"Max relative difference for significant groups: {max_rel_diff_sig:.2e}")
        
        max_rel_diff = np.max(rel_diff)
        print(f"Max relative difference (all): {max_rel_diff:.2e}")
        
        if np.any(mask) and max_rel_diff_sig < 1e-2:
            print("SUCCESS: System converged towards analytical equilibrium.")
        else:
            print("WARNING: System has not fully converged.")
            # Print indices of worst offenders
            worst_idx = np.argmax(rel_diff)
            print(f"Worst offender index: {worst_idx}, value: {y_final[worst_idx]:.2e}, expected: {x_eq[worst_idx]:.2e}")

    else:
        print(f"ERROR: Simulation failed. Message: {sol.message}")

if __name__ == "__main__":
    test_dynamics()