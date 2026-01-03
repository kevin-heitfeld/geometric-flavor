"""
Scan over different charge assignments and Clebsch structures
to find best fit to CKM observables
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

print("="*70)
print("SCANNING CHARGE ASSIGNMENTS FOR BEST CKM FIT")
print("="*70)
print()

# Different charge patterns to try
charge_patterns = [
    ([3, 2, 0], [3, 2, 0], "FN standard (3,2,0)"),
    ([4, 2, 0], [4, 2, 0], "Extended FN (4,2,0)"),
    ([3, 1, 0], [3, 1, 0], "Compressed FN (3,1,0)"),
    ([5, 3, 0], [5, 3, 0], "Large FN (5,3,0)"),
    ([3, 2, 0], [4, 2, 0], "Mixed up(3,2,0) down(4,2,0)"),
    ([4, 2, 0], [3, 2, 0], "Mixed up(4,2,0) down(3,2,0)"),
    ([3, 2, 1], [3, 2, 1], "Small third gen (3,2,1)"),
]

best_error = 1e10
best_config = None
best_spurion = None
best_clebsch = None

for q_up, q_down, desc in charge_patterns:
    print(f"Trying {desc}...")

    # Create custom charges
    charges = U1Charges('FN_standard')
    charges.q_up = np.array(q_up)
    charges.q_down = np.array(q_down)

    try:
        spurion, clebsch_opt, max_error = fit_spurion_to_CKM(
            m_up, m_down, target,
            charges=charges,
            optimize_clebsch=True,
            verbose=False
        )

        print(f"  → Error: {max_error*100:.4f}%")
        print(f"     |Z| = {spurion.magnitude:.4f}, arg(Z) = {spurion.phase:.4f} rad")

        if max_error < best_error:
            best_error = max_error
            best_config = desc
            best_spurion = spurion
            best_clebsch = clebsch_opt
            best_charges = charges

    except Exception as e:
        print(f"  → Failed: {e}")

    print()

print("="*70)
print("BEST CONFIGURATION")
print("="*70)
print(f"Charge pattern: {best_config}")
print(f"q_up:   {best_charges.q_up}")
print(f"q_down: {best_charges.q_down}")
print(f"Spurion: |Z| = {best_spurion.magnitude:.6f}, arg(Z) = {best_spurion.phase:.6f} rad")
print(f"Maximum error: {best_error*100:.4f}%")
print()
print(f"Clebsch (up):")
print(f"  [0,1]={best_clebsch.C_up[0,1]:.4f}, [0,2]={best_clebsch.C_up[0,2]:.4f}, [1,2]={best_clebsch.C_up[1,2]:.4f}")
print(f"Clebsch (down):")
print(f"  [0,1]={best_clebsch.C_down[0,1]:.4f}, [0,2]={best_clebsch.C_down[0,2]:.4f}, [1,2]={best_clebsch.C_down[1,2]:.4f}")
