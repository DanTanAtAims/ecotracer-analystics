import sys
import os
import numpy as np
import pandas as pd

# Add the src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

try:
    import pyewe as ewe
    from ecotracer_analytics.core import EcoTracerAnalytics
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def main():
    # Path to model in pyewe_sandbox
    model_path = "/home/dtan/repos/pyewe_sandbox/East Bass Strait.eiixml"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    print("Initializing Core...")
    try:
        scen_int = ewe.EwEScenarioInterface(model_path, ecosim_scenario="constant fishing", ecotracer_scenario="smaller_equi")
        # Initialize ecosim to ensure data structures are populated (like DetritusOut)
        # We don't need to run it, just ensure initialization.
    except Exception as e:
        print(f"Failed to initialize pyewe: {e}")
        return

    analytics = EcoTracerAnalytics(scen_int)
    
    print("\nCalculating Analytical Equilibrium...")
    equi_vector = analytics.predict_equilibrium()
    
    print("\nComparing with Model Initial Amounts:")
    model_amounts = np.concatenate(([analytics.data['init_env_conc']], analytics.data['init_amount']))
    
    diff = equi_vector - model_amounts
    rel_diff = np.abs(diff) / (model_amounts + 1e-15)
    
    print(f"Max Absolute Difference: {np.max(np.abs(diff)):.2e}")
    print(f"Max Relative Difference: {np.max(rel_diff):.2e}")
    print(f"Mean Relative Difference: {np.mean(rel_diff):.2e}")

    if np.max(rel_diff) < 1e-5:
        print("\nSUCCESS: Analytical calculation matches model initial state!")
    else:
        print("\nWARNING: Significant differences detected.")
        # Print top 5 differences
        indices = np.argsort(rel_diff)[::-1][:5]
        for i in indices:
            name = "Env" if i == 0 else f"Group {i}"
            print(f"{name}: Analytical={equi_vector[i]:.2e}, Model={model_amounts[i]:.2e}, RelDiff={rel_diff[i]:.2e}")

    print("\nTesting ODE Simulation (small step)...")
    sol = analytics.run_ode((0, 0.1))
    print(f"ODE Simulation completed. Status: {sol.message}")

if __name__ == "__main__":
    main()
