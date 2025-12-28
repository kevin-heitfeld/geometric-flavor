"""
Simplified CKM estimate: Power-law suppression only

Previous attempt used exponential suppression exp(-π Imτ Δq²) which was TOO STRONG.
Try simpler ansatz: Y_ij ~ sqrt(Y_ii × Y_jj) × (Imτ)^(-|Δw|/α)
where α is an unknown parameter we can constrain from V_us.
"""

import numpy as np

tau = 2.69j
Im_tau = 2.69

q = np.exp(2 * np.pi * 1j * tau)
eta = q**(1/24) * np.prod([1 - q**n for n in range(1, 30)])

def yukawa_scaling(w):
    return (Im_tau)**(-w) * np.abs(eta)**(-6*w)

def modular_weight(q3, q4):
    return -2*q3 + q4

v_higgs = 246.0

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

def compute_yukawa_matrix_powerlaw(assignments, exp_values, alpha):
    """
    Compute Yukawa matrix with power-law suppression.
    
    Y_ij ~ sqrt(Y_ii × Y_jj) × (Imτ)^(-|Δw|/α)
    
    α controls strength of suppression:
    - α = 1: strongest suppression (Imτ)^(-|Δw|)
    - α = 2: weaker suppression (Imτ)^(-|Δw|/2)
    - α large: very weak suppression
    """
    particles = ['up', 'charm', 'top'] if 'up' in assignments else ['down', 'strange', 'bottom']
    
    weights = {p: modular_weight(*assignments[p]) for p in particles}
    diag_scaling = {p: yukawa_scaling(w) for p, w in weights.items()}
    
    norm = exp_values[particles[0]] / diag_scaling[particles[0]]
    Y_diag = {p: s * norm for p, s in diag_scaling.items()}
    
    Y = np.zeros((3, 3))
    
    for i, p_i in enumerate(particles):
        for j, p_j in enumerate(particles):
            if i == j:
                Y[i, j] = Y_diag[p_i]
            else:
                w_i = weights[p_i]
                w_j = weights[p_j]
                Delta_w = abs(w_i - w_j)
                
                Y_ii_scaling = yukawa_scaling(w_i)
                Y_jj_scaling = yukawa_scaling(w_j)
                geom_mean = np.sqrt(Y_ii_scaling * Y_jj_scaling) * norm
                
                suppression = (Im_tau)**(-Delta_w / alpha)
                Y[i, j] = geom_mean * suppression
    
    return Y

print("="*70)
print("POWER-LAW SUPPRESSION: SCAN OVER α")
print("="*70)
print()
print("Testing Y_ij ~ sqrt(Y_ii × Y_jj) × (Imτ)^(-|Δw|/α)")
print()

# Target: V_us ≈ 0.225
V_us_exp = 0.225
V_cb_exp = 0.04182
V_ub_exp = 0.00369

V_CKM_exp = np.array([
    [0.97435, 0.22500, 0.00369],
    [0.22486, 0.97349, 0.04182],
    [0.00857, 0.04110, 0.99915]
])

print("Scanning α from 0.5 to 10.0...\n")

best_alpha = None
best_error = float('inf')

for alpha in np.linspace(0.5, 10.0, 20):
    Y_u = compute_yukawa_matrix_powerlaw(up_assignments, up_exp, alpha)
    Y_d = compute_yukawa_matrix_powerlaw(down_assignments, down_exp, alpha)
    
    D_u, V_u = np.linalg.eig(Y_u)
    idx_u = np.argsort(np.abs(D_u))
    V_u = V_u[:, idx_u]
    
    D_d, V_d = np.linalg.eig(Y_d)
    idx_d = np.argsort(np.abs(D_d))
    V_d = V_d[:, idx_d]
    
    V_CKM = np.conj(V_u).T @ V_d
    
    V_us_calc = np.abs(V_CKM[0, 1])
    V_cb_calc = np.abs(V_CKM[1, 2])
    V_ub_calc = np.abs(V_CKM[0, 2])
    
    # Error on key elements
    error = (abs(V_us_calc - V_us_exp)/V_us_exp +
             abs(V_cb_calc - V_cb_exp)/V_cb_exp +
             abs(V_ub_calc - V_ub_exp)/V_ub_exp) / 3
    
    if error < best_error:
        best_error = error
        best_alpha = alpha
        best_V_CKM = V_CKM
        best_V_us = V_us_calc
        best_V_cb = V_cb_calc
        best_V_ub = V_ub_calc
    
    if alpha in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
        print(f"α = {alpha:.1f}:")
        print(f"  V_us = {V_us_calc:.5f} (exp: {V_us_exp:.5f})")
        print(f"  V_cb = {V_cb_calc:.5f} (exp: {V_cb_exp:.5f})")
        print(f"  V_ub = {V_ub_calc:.5f} (exp: {V_ub_exp:.5f})")
        print(f"  Avg error: {error*100:.1f}%")
        print()

print("="*70)
print(f"BEST FIT: α = {best_alpha:.2f}")
print("="*70)
print()

# Recompute with best alpha
Y_u = compute_yukawa_matrix_powerlaw(up_assignments, up_exp, best_alpha)
Y_d = compute_yukawa_matrix_powerlaw(down_assignments, down_exp, best_alpha)

print("Up-type Yukawa matrix:")
print(Y_u)
print()

print("Down-type Yukawa matrix:")
print(Y_d)
print()

D_u, V_u = np.linalg.eig(Y_u)
idx_u = np.argsort(np.abs(D_u))
V_u = V_u[:, idx_u]

D_d, V_d = np.linalg.eig(Y_d)
idx_d = np.argsort(np.abs(D_d))
V_d = V_d[:, idx_d]

V_CKM = np.conj(V_u).T @ V_d

print("Predicted CKM matrix:")
print(np.abs(V_CKM))
print()

print("Experimental CKM matrix:")
print(V_CKM_exp)
print()

print("Element-by-element comparison:")
for i in range(3):
    for j in range(3):
        calc = np.abs(V_CKM[i,j])
        exp = V_CKM_exp[i,j]
        error = abs(calc - exp) / exp * 100 if exp > 0.01 else abs(calc - exp)
        print(f"  V[{i},{j}]: {calc:.5f} vs {exp:.5f} ", end="")
        if exp > 0.01:
            print(f"({error:6.1f}%)")
        else:
            print(f"(Δ={calc-exp:+.5f})")

print()
print("="*70)
print("KEY RESULTS")
print("="*70)
print()

print(f"Optimal suppression: α = {best_alpha:.2f}")
print(f"  → Y_ij ~ sqrt(Y_ii × Y_jj) × (Imτ)^(-|Δw|/{best_alpha:.2f})")
print()

print("Cabibbo angle (V_us):")
print(f"  Predicted: {best_V_us:.5f}")
print(f"  Experiment: {V_us_exp:.5f}")
print(f"  Error: {abs(best_V_us - V_us_exp)/V_us_exp * 100:.1f}%")

print("\nV_cb:")
print(f"  Predicted: {best_V_cb:.5f}")
print(f"  Experiment: {V_cb_exp:.5f}")
print(f"  Error: {abs(best_V_cb - V_cb_exp)/V_cb_exp * 100:.1f}%")

print("\nV_ub:")
print(f"  Predicted: {best_V_ub:.5f}")
print(f"  Experiment: {V_ub_exp:.5f}")
print(f"  Error: {abs(best_V_ub - V_ub_exp)/V_ub_exp * 100:.1f}%")

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)
print()
print(f"The best fit α ≈ {best_alpha:.2f} tells us:")
print()
print("  - Off-diagonal suppression follows power law (Imτ)^(-Δw/α)")
print(f"  - At Imτ = 2.69, Δw = 1 gives suppression ~ 2.69^(-1/{best_alpha:.2f}) ≈ {Im_tau**(-1/best_alpha):.3f}")
print(f"  - At Imτ = 2.69, Δw = 2 gives suppression ~ 2.69^(-2/{best_alpha:.2f}) ≈ {Im_tau**(-2/best_alpha):.3f}")
print()

if best_alpha < 2.0:
    print("  → Strong suppression (α < 2): Overlaps exponentially suppressed")
elif best_alpha < 5.0:
    print("  → Moderate suppression (2 < α < 5): Overlaps power-law suppressed")
else:
    print("  → Weak suppression (α > 5): Overlaps only mildly suppressed")

print()
print("Next steps:")
print("  1. Derive α from first principles (wave function overlap integrals)")
print("  2. Check if α depends on sector or is universal")
print("  3. Include CP-violating phases (Jarlskog invariant)")
print("  4. Compute full 3×3 unitarity test")
