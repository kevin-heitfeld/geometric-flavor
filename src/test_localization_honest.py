"""
HONEST TEST: Geometric localization WITHOUT calibration
"""

import numpy as np
from localization_from_geometry import compute_localization_parameters

# Fitted values
g_lep_fit = np.array([1.00, 1.10599770, 1.00816488])
g_up_fit = np.array([1.00, 1.12996338, 1.01908896])
g_down_fit = np.array([1.00, 0.96185547, 1.00057316])

A_lep_fit = np.array([0.00, -0.72084622, -0.92315966])
A_up_fit = np.array([0.00, -0.87974875, -1.48332060])
A_down_fit = np.array([0.00, -0.33329575, -0.88288836])

print("="*70)
print("HONEST TEST: NO CALIBRATION")
print("="*70)
print()

# Compute WITHOUT calibration - pure geometry
g_lep, g_up, g_down, A_lep, A_up, A_down = compute_localization_parameters(
    charges_lep=np.array([3, 2, 0]),
    charges_up=np.array([3, 2, 0]),
    charges_down=np.array([3, 2, 0]),
    calibrate=False,  # NO CALIBRATION
    verbose=True
)

print()
print("="*70)
print("COMPARISON: Pure Geometry vs. Fitted")
print("="*70)

def compare(name, geom, fit):
    print(f"\n{name}:")
    for i in range(3):
        if fit[i] != 0:
            err = abs(geom[i] - fit[i]) / abs(fit[i]) * 100
        else:
            err = 0 if geom[i] == 0 else 100
        print(f"  [{i}]  Geom: {geom[i]:8.5f}  Fit: {fit[i]:8.5f}  Error: {err:6.2f}%")

compare("g_lep", g_lep, g_lep_fit)
compare("g_up", g_up, g_up_fit)
compare("g_down", g_down, g_down_fit)
compare("A_lep", A_lep, A_lep_fit)
compare("A_up", A_up, A_up_fit)
compare("A_down", A_down, A_down_fit)

print()
print("="*70)
print("VERDICT:")
print("="*70)
print("If errors are >10%, we are NOT deriving from geometry.")
print("We are just fitting with extra steps.")
print()
print("True geometric derivation would give <5% error WITHOUT calibration.")
print("="*70)
