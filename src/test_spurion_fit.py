"""
Test spurion fitting to CKM observables
"""

import numpy as np
from flavor_spurion import (
    FlavorSpurion, U1Charges, ClebschCoefficients,
    fit_spurion_to_CKM
)

# Quark masses (normalized ratios)
m_up = np.array([1.0, 577.0, 78636.0])
m_down = np.array([1.0, 20.3, 890.0])

# Target CKM observables
target = {
    'sin2_12': 0.0510,     # Cabibbo angle
    'sin2_23': 0.00157,    # V_cb
    'sin2_13': 0.000128,   # V_ub
    'delta_CP': 1.22,      # CP phase (radians)
    'J_CP': 3.0e-5         # Jarlskog invariant
}

print("="*60)
print("FITTING SINGLE SPURION TO CKM OBSERVABLES")
print("="*60)
print()
print("Replacing 12 free parameters with 2 spurion + 6 Clebsch (order 1)...")
print()

# Fit spurion with Clebsch optimization
spurion, clebsch_opt, max_error = fit_spurion_to_CKM(
    m_up, m_down, target,
    charges=None,  # Use default FN_standard
    clebsch=None,  # Use default A4 as starting point
    optimize_clebsch=True,
    verbose=True
)

print()
print(f"Result: {spurion}")
print(f"Max error: {max_error*100:.4f}%")
print(f"Maximum error: {max_error*100:.4f}%")
print()
print("Parameter reduction: 12 complex → 1 complex spurion")
print("  (magnitude + phase) + discrete charges + Clebsch coefficients")
