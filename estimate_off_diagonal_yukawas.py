"""
Estimate off-diagonal Yukawa couplings for CKM matrix prediction

Strategy: Use modular weight suppression to estimate Y_ij (i≠j)

Physical picture:
- Y_ij = ∫ ψ_i(z,τ) × conj(ψ_j(z,τ)) × ψ_H(z,τ) d²z
- Overlap suppressed when w_i ≠ w_j (different modular weights)
- Estimate: Y_ij ~ geometric_mean × overlap_factor

Overlap factor depends on:
1. Δw = |w_i - w_j| (larger difference → stronger suppression)
2. Δq₃, Δq₄ (quantum number mismatch reduces overlap)
3. Theta function orthogonality (different characteristics)
"""

import numpy as np

tau = 2.69j
Im_tau = 2.69

# Dedekind eta
q = np.exp(2 * np.pi * 1j * tau)
eta = q**(1/24) * np.prod([1 - q**n for n in range(1, 30)])

def yukawa_scaling(w):
    """Leading order Yukawa scaling"""
    return (Im_tau)**(-w) * np.abs(eta)**(-6*w)

def modular_weight(q3, q4):
    return -2*q3 + q4

v_higgs = 246.0

# Quantum number assignments (from optimize_quantum_numbers.py)
up_assignments = {
    'up': (0, 0),      # w = 0
    'charm': (0, 2),   # w = +2
    'top': (0, 3)      # w = +3
}

down_assignments = {
    'down': (0, 1),    # w = +1
    'strange': (0, 2), # w = +2
    'bottom': (0, 3)   # w = +3
}

# Experimental values for normalization
up_exp = {
    'up': np.sqrt(2) * 2.16e-3 / v_higgs,
    'charm': np.sqrt(2) * 1.27 / v_higgs,
    'top': np.sqrt(2) * 172.57 / v_higgs
}

down_exp = {
    'down': np.sqrt(2) * 4.67e-3 / v_higgs,
    'strange': np.sqrt(2) * 93.4e-3 / v_higgs,
    'bottom': np.sqrt(2) * 4.18 / v_higgs
}

print("="*70)
print("OFF-DIAGONAL YUKAWA COUPLINGS AND CKM MATRIX")
print("="*70)
print()

# Compute diagonal Yukawas (normalized)
def compute_yukawa_matrix(assignments, exp_values, sector_name):
    """Compute full 3x3 Yukawa matrix with off-diagonal elements"""

    particles = ['up', 'charm', 'top'] if 'up' in assignments else ['down', 'strange', 'bottom']

    # Diagonal elements
    weights = {p: modular_weight(*assignments[p]) for p in particles}
    diag_scaling = {p: yukawa_scaling(w) for p, w in weights.items()}

    # Normalize to lightest
    norm = exp_values[particles[0]] / diag_scaling[particles[0]]
    Y_diag = {p: s * norm for p, s in diag_scaling.items()}

    # Initialize matrix
    Y = np.zeros((3, 3))

    # Fill diagonal
    for i, p in enumerate(particles):
        Y[i, i] = Y_diag[p]

    print(f"\n{sector_name} SECTOR")
    print("-"*70)
    print("\nQuantum numbers and weights:")
    for p in particles:
        q3, q4 = assignments[p]
        w = weights[p]
        print(f"  {p:8s}: (q₃={q3}, q₄={q4}) → w={w:+2d}")

    print("\nDiagonal Yukawa couplings:")
    for i, p in enumerate(particles):
        print(f"  Y[{i},{i}] ({p:8s}): {Y[i,i]:.3e}")

    # Estimate off-diagonal elements
    print("\nEstimating off-diagonal couplings...")

    for i, p_i in enumerate(particles):
        for j, p_j in enumerate(particles):
            if i == j:
                continue

            w_i = weights[p_i]
            w_j = weights[p_j]
            q3_i, q4_i = assignments[p_i]
            q3_j, q4_j = assignments[p_j]

            # Geometric mean of diagonal scalings
            Y_ii_scaling = yukawa_scaling(w_i)
            Y_jj_scaling = yukawa_scaling(w_j)
            geom_mean = np.sqrt(Y_ii_scaling * Y_jj_scaling) * norm

            # Suppression factors
            Delta_w = abs(w_i - w_j)
            Delta_q3 = abs(q3_i - q3_j)
            Delta_q4 = abs(q4_i - q4_j)

            # Estimate overlap factor from quantum number mismatch
            # Larger mismatch → stronger suppression
            # This is an ANSATZ - need proper overlap integral!

            # Power-law suppression from modular weight difference
            modular_suppression = (Im_tau)**(-Delta_w/2)

            # Exponential suppression from quantum number mismatch
            # (Theta functions with different characteristics have exponentially small overlap)
            quantum_suppression = np.exp(-np.pi * Im_tau * (Delta_q3**2 + Delta_q4**2) / 6)

            # Combined
            overlap_factor = modular_suppression * quantum_suppression

            Y[i, j] = geom_mean * overlap_factor

            print(f"  Y[{i},{j}] ({p_i} → {p_j}):")
            print(f"    Δw={Delta_w}, Δq₃={Delta_q3}, Δq₄={Delta_q4}")
            print(f"    Modular suppression: {modular_suppression:.3e}")
            print(f"    Quantum suppression: {quantum_suppression:.3e}")
            print(f"    Y[{i},{j}] ≈ {Y[i,j]:.3e}")

    print(f"\nFull {sector_name} Yukawa matrix:")
    print(Y)

    return Y, particles

# Compute both sectors
Y_u, up_particles = compute_yukawa_matrix(up_assignments, up_exp, "UP-TYPE QUARK")
Y_d, down_particles = compute_yukawa_matrix(down_assignments, down_exp, "DOWN-TYPE QUARK")

print("\n" + "="*70)
print("CKM MATRIX FROM DIAGONALIZATION")
print("="*70)
print()

# Diagonalize Yukawa matrices
print("Diagonalizing Y_u:")
D_u, V_u = np.linalg.eig(Y_u)
# Sort by eigenvalue (mass)
idx_u = np.argsort(np.abs(D_u))
D_u = D_u[idx_u]
V_u = V_u[:, idx_u]
print(f"  Eigenvalues: {D_u}")
print(f"  Mixing matrix V_u:")
print(V_u)

print("\nDiagonalizing Y_d:")
D_d, V_d = np.linalg.eig(Y_d)
idx_d = np.argsort(np.abs(D_d))
D_d = D_d[idx_d]
V_d = V_d[:, idx_d]
print(f"  Eigenvalues: {D_d}")
print(f"  Mixing matrix V_d:")
print(V_d)

# Compute CKM
V_CKM = np.conj(V_u).T @ V_d
print("\nCKM matrix V_CKM = V_u† × V_d:")
print(np.abs(V_CKM))

print("\n" + "="*70)
print("COMPARISON TO EXPERIMENTAL CKM")
print("="*70)
print()

V_CKM_exp = np.array([
    [0.97435, 0.22500, 0.00369],
    [0.22486, 0.97349, 0.04182],
    [0.00857, 0.04110, 0.99915]
])

print("Experimental CKM (magnitudes):")
print(V_CKM_exp)

print("\nOur prediction:")
print(np.abs(V_CKM))

print("\nElement-by-element comparison:")
for i in range(3):
    for j in range(3):
        calc = np.abs(V_CKM[i,j])
        exp = V_CKM_exp[i,j]
        diff = calc - exp
        rel_err = abs(diff/exp) * 100 if exp > 0.01 else diff

        print(f"  V[{i},{j}]: {calc:.5f} vs {exp:.5f} (diff: {diff:+.5f}, ", end="")
        if exp > 0.01:
            print(f"{rel_err:.1f}%)")
        else:
            print(f"abs diff {abs(diff):.5f})")

print("\n" + "="*70)
print("KEY CKM PARAMETERS")
print("="*70)
print()

# Extract key angles
V_us_calc = np.abs(V_CKM[0, 1])
V_cb_calc = np.abs(V_CKM[1, 2])
V_ub_calc = np.abs(V_CKM[0, 2])

V_us_exp = 0.22500
V_cb_exp = 0.04182
V_ub_exp = 0.00369

print("Cabibbo angle (V_us):")
print(f"  Calculated: {V_us_calc:.5f}")
print(f"  Experiment: {V_us_exp:.5f}")
print(f"  Error: {abs(V_us_calc - V_us_exp)/V_us_exp * 100:.1f}%")

print("\nV_cb (second generation mixing):")
print(f"  Calculated: {V_cb_calc:.5f}")
print(f"  Experiment: {V_cb_exp:.5f}")
print(f"  Error: {abs(V_cb_calc - V_cb_exp)/V_cb_exp * 100:.1f}%")

print("\nV_ub (smallest mixing):")
print(f"  Calculated: {V_ub_calc:.5f}")
print(f"  Experiment: {V_ub_exp:.5f}")
print(f"  Error: {abs(V_ub_calc - V_ub_exp)/V_ub_exp * 100:.1f}%")

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)
print()
print("This is a ROUGH ESTIMATE using ansatz for overlap factors:")
print()
print("  Y_ij ~ sqrt(Y_ii × Y_jj) × (Imτ)^(-Δw/2) × exp(-π Imτ Δq²/6)")
print()
print("The exponential suppression from quantum number mismatch is very strong:")
print(f"  Imτ = {Im_tau:.2f}")
print(f"  exp(-π × 2.69 × 1/6) ≈ {np.exp(-np.pi * Im_tau / 6):.3e}")
print()
print("For better predictions, need:")
print("  1. Proper wave function overlap integrals ∫ ψ_i × conj(ψ_j) × ψ_H")
print("  2. Correct normalization of wave functions")
print("  3. Kähler metric effects on overlaps")
print("  4. Selection rules for allowed off-diagonal couplings")
print()
print("But the QUALITATIVE pattern should be correct:")
print("  - V_us largest (smallest Δw and Δq)")
print("  - V_cb medium (medium Δw and Δq)")
print("  - V_ub smallest (largest Δw and Δq)")
