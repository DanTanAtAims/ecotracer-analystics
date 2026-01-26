import pyewe as ewe
import numpy as np
import ecotracer_analytics as etca

def check_stability():
    model_path = "/home/dtan/repos/pyewe_sandbox/East Bass Strait.eiixml"
    scen_int = ewe.EwEScenarioInterface(
        model_path, 
        ecosim_scenario="constant fishing", 
        ecotracer_scenario="smaller_equi"
    )

    path_source = etca.EcopathSource(scen_int._core_instance)
    tracer_source = etca.EcotracerSource(scen_int._core_instance)

    M = etca.calculate_coefficient(path_source, tracer_source)
    eigenvalues = np.linalg.eigvals(M)

    # For stability in dy/dt = b - My, all eigenvalues of M should have positive real parts.
    # (Because it is equivalent to dy/dt = -M(y - y_eq))
    min_real = np.min(eigenvalues.real)
    max_real = np.max(eigenvalues.real)
    
    print(f"Eigenvalues real parts: min={min_real:.2e}, max={max_real:.2e}")
    
    if min_real < 0:
        print("SYSTEM IS UNSTABLE: Found eigenvalue with negative real part.")
        unstable_indices = np.where(eigenvalues.real < 0)[0]
        for idx in unstable_indices:
             print(f"Unstable eigenvalue: {eigenvalues[idx]}")
    else:
        print("System is theoretically stable (all eigenvalues have positive real parts).")

if __name__ == "__main__":
    check_stability()
