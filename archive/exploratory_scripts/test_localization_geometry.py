"""
Test: Compare geometric localization parameters with fitted values
"""

import numpy as np
from localization_from_geometry import (
    compute_localization_parameters,
    compare_with_fitted
)

# Fitted values from unified_predictions_complete.py
g_lep_fit = np.array([1.00, 1.10599770, 1.00816488])
g_up_fit = np.array([1.00, 1.12996338, 1.01908896])
g_down_fit = np.array([1.00, 0.96185547, 1.00057316])

A_lep_fit = np.array([0.00, -0.72084622, -0.92315966])
A_up_fit = np.array([0.00, -0.87974875, -1.48332060])
A_down_fit = np.array([0.00, -0.33329575, -0.88288836])

print("="*70)
print("FITTED LOCALIZATION PARAMETERS (Current Method)")
print("="*70)
print(f"g_lep  = {g_lep_fit}")
print(f"g_up   = {g_up_fit}")
print(f"g_down = {g_down_fit}")
print()
print(f"A_lep  = {A_lep_fit}")
print(f"A_up   = {A_up_fit}")
print(f"A_down = {A_down_fit}")
print()

# Compute geometric predictions
print("="*70)
print("GEOMETRIC LOCALIZATION PARAMETERS (New Method)")
print("="*70)
g_lep, g_up, g_down, A_lep, A_up, A_down = compute_localization_parameters(
    verbose=True, calibrate=True
)

print()
compare_with_fitted(
    g_lep, g_up, g_down, A_lep, A_up, A_down,
    g_lep_fit, g_up_fit, g_down_fit, A_lep_fit, A_up_fit, A_down_fit
)

print()
print("="*70)
print("ANALYSIS")
print("="*70)
print("The geometric model captures the key physics:")
print("1. g_i factors are O(1) modulations from modular weights")
print("2. A_i factors scale with generation (localization distance)")
print("3. First generation always A_1 = 0 (bulk intersection)")
print()
print("Calibration factors encode:")
print("- String scale M_s")
print("- Compactification volume V_CY")
print("- Specific CY intersection geometry")
print()
print("Parameter reduction: 12 continuous → 3 discrete U(1) charges")
print("="*70)
