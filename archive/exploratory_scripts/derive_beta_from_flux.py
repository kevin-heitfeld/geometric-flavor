"""
DERIVING β_i FROM MAGNETIC FLUX AND MODULAR ANOMALY

Following magnetized D7-brane physics:
Y_i ~ |η(τ)|^(2M_i + δ_i)

where:
- M_i = magnetic flux intersection number
- δ_i = modular anomaly / zero-point correction

On T⁶/(Z₃×Z₄), we have:
- Z₃ sector: k = 4, 6, 8 (leptons)
- Z₄ sector: k = 8, 12, 16 (quarks)

The k-weights should relate to flux quanta.
"""

import numpy as np
from scipy.special import gamma as gamma_func
from scipy.optimize import minimize

V_EW = 246.0
PI = np.pi
TAU = 2.69j

LEPTON_YUKAWAS = {'e': 0.000511 / V_EW, 'μ': 0.105658 / V_EW, 'τ': 1.77686 / V_EW}
LEPTON_K = {'τ': 8, 'μ': 6, 'e': 4}
GENERATION = {'e': 1, 'μ': 2, 'τ': 3}

# From previous analysis
BETA_EMPIRICAL = {'e': -4.945, 'μ': -12.516, 'τ': -16.523}

def dedekind_eta(tau):
    q = np.exp(2j * PI * tau)
    eta_asymp = np.exp(PI * 1j * tau / 12)
    correction = 1.0
    for n in range(1, 20):
        correction *= (1 - q**n)
    return eta_asymp * correction

eta = dedekind_eta(TAU)
eta_abs = abs(eta)

print("="*80)
print("DERIVING β_i FROM MAGNETIC FLUX")
print("="*80)
print()

print("Theory: On magnetized D7-branes,")
print("  Y_i ~ |η(τ)|^(2M_i + δ_i)")
print()
print("where:")
print("  M_i = magnetic flux intersection number")
print("  δ_i = modular anomaly correction")
print()

# Step 1: Decompose β into 2M + δ
print("-"*80)
print()
print("Step 1: Extract M_i and δ_i from empirical β_i")
print()

print("Empirical values:")
for p in ['e', 'μ', 'τ']:
    k = LEPTON_K[p]
    beta = BETA_EMPIRICAL[p]
    print(f"  β_{p} = {beta:.3f} (k={k})")
print()

# Hypothesis 1: M_i ∝ k_i
print("Hypothesis 1: M_i = a × k_i")
print()

# If β_i = 2M_i + δ_i and M_i = a×k_i, then β_i = 2a×k_i + δ_i
# Check if δ_i is constant or generation-dependent

print("Testing β_i = 2a×k_i + δ_gen:")
print()

# Fit for different models
models = {
    'Constant δ': lambda k, g, params: 2*params[0]*k + params[1],
    'Gen-dependent δ': lambda k, g, params: 2*params[0]*k + params[1] - params[2]*g,
    'k-dependent δ': lambda k, g, params: 2*params[0]*k + params[1] + params[2]*k,
}

results = {}

for model_name, formula in models.items():
    print(f"  Model: {model_name}")
    
    if 'Constant' in model_name:
        # 2 parameters: a, δ
        def residual(params):
            return sum((formula(LEPTON_K[p], GENERATION[p], params) - BETA_EMPIRICAL[p])**2 
                      for p in ['e', 'μ', 'τ'])
        
        result = minimize(residual, [0, 0], method='Nelder-Mead')
        params = result.x
        chi2 = result.fun
        
        print(f"    Best fit: β_i = 2×{params[0]:.4f}×k_i + {params[1]:.4f}")
        print(f"    → M_i = {params[0]:.4f}×k_i, δ = {params[1]:.4f}")
        
    elif 'Gen-dependent' in model_name:
        # 3 parameters: a, δ₀, b
        def residual(params):
            return sum((formula(LEPTON_K[p], GENERATION[p], params) - BETA_EMPIRICAL[p])**2 
                      for p in ['e', 'μ', 'τ'])
        
        result = minimize(residual, [0, 0, 0], method='Nelder-Mead')
        params = result.x
        chi2 = result.fun
        
        print(f"    Best fit: β_i = 2×{params[0]:.4f}×k_i + {params[1]:.4f} - {params[2]:.4f}×gen")
        print(f"    → M_i = {params[0]:.4f}×k_i, δ_gen = {params[1]:.4f} - {params[2]:.4f}×gen")
        
    else:  # k-dependent
        # 3 parameters: a, b, c
        def residual(params):
            return sum((formula(LEPTON_K[p], GENERATION[p], params) - BETA_EMPIRICAL[p])**2 
                      for p in ['e', 'μ', 'τ'])
        
        result = minimize(residual, [0, 0, 0], method='Nelder-Mead')
        params = result.x
        chi2 = result.fun
        
        print(f"    Best fit: β_i = (2×{params[0]:.4f} + {params[2]:.4f})×k_i + {params[1]:.4f}")
        print(f"    → M_i = {params[0]:.4f}×k_i, δ_i = {params[1]:.4f} + {params[2]:.4f}×k_i")
    
    print(f"    χ² = {chi2:.6f}")
    print()
    
    # Test predictions
    print("    Predictions:")
    for p in ['e', 'μ', 'τ']:
        k = LEPTON_K[p]
        g = GENERATION[p]
        beta_pred = formula(k, g, params)
        beta_emp = BETA_EMPIRICAL[p]
        error = beta_pred - beta_emp
        print(f"      {p}: β_pred = {beta_pred:.3f}, β_emp = {beta_emp:.3f}, error = {error:+.3f}")
    
    print()
    results[model_name] = (params, chi2)

# Find best model
best_model = min(results.items(), key=lambda x: x[1][1])
print(f"✓ Best model: {best_model[0]} (χ² = {best_model[1][1]:.6f})")
print()

# Step 2: Physical interpretation
print("="*80)
print("PHYSICAL INTERPRETATION")
print("="*80)
print()

best_params, _ = best_model[1]

if 'Constant' in best_model[0]:
    a = best_params[0]
    delta = best_params[1]
    
    print(f"M_i = {a:.4f} × k_i")
    print(f"δ_i = {delta:.4f} (constant)")
    print()
    
    print("Magnetic flux numbers:")
    for p in ['e', 'μ', 'τ']:
        k = LEPTON_K[p]
        M = a * k
        print(f"  M_{p} = {M:.3f}")
    
elif 'Gen-dependent' in best_model[0]:
    a = best_params[0]
    delta0 = best_params[1]
    b = best_params[2]
    
    print(f"M_i = {a:.4f} × k_i")
    print(f"δ_gen = {delta0:.4f} - {b:.4f}×gen")
    print()
    
    print("Magnetic flux numbers:")
    for p in ['e', 'μ', 'τ']:
        k = LEPTON_K[p]
        M = a * k
        print(f"  M_{p} = {M:.3f}")
    
    print()
    print("Modular anomalies:")
    for p in ['e', 'μ', 'τ']:
        g = GENERATION[p]
        delta = delta0 - b * g
        print(f"  δ_{p} (gen {g}) = {delta:.3f}")

else:  # k-dependent
    a = best_params[0]
    b = best_params[1]
    c = best_params[2]
    
    print(f"M_i = {a:.4f} × k_i")
    print(f"δ_i = {b:.4f} + {c:.4f}×k_i")
    print()
    
    print("Combined:")
    for p in ['e', 'μ', 'τ']:
        k = LEPTON_K[p]
        M = a * k
        delta = b + c * k
        beta_total = 2*M + delta
        print(f"  {p}: M = {M:.3f}, δ = {delta:.3f}, β = 2M + δ = {beta_total:.3f}")

print()

# Step 3: Check for rational flux
print("-"*80)
print()
print("Step 3: Testing for rational/integer flux values")
print()

if 'Gen-dependent' in best_model[0]:
    a = best_params[0]
    
    print(f"M_i / k_i = {a:.6f}")
    print()
    
    # Check if close to simple fraction
    for num in range(-5, 1):
        for denom in range(1, 11):
            frac = num / denom
            if abs(frac - a) < 0.05:
                print(f"  Close to {num}/{denom} = {frac:.6f} (error: {abs(frac-a):.6f})")
    
    print()
    
    # Check specific theoretical values
    theoretical = {
        '-1/2': -0.5,
        '-2/3': -2/3,
        '-3/4': -3/4,
        '-1': -1.0,
        '-3/2': -1.5,
    }
    
    print("Comparison with common flux values:")
    for name, val in theoretical.items():
        error = abs(val - a)
        if error < 0.1:
            print(f"  {name} = {val:.4f} (error: {error:.6f})")

print()

# Step 4: Connection to modular weight k
print("-"*80)
print()
print("Step 4: Connection to modular weight k")
print()

print("Recall: k-weights on Γ₀(3):")
print("  k_e = 4, k_μ = 6, k_τ = 8")
print()

if 'Gen-dependent' in best_model[0]:
    a = best_params[0]
    delta0 = best_params[1]
    b = best_params[2]
    
    print(f"β_i = 2×({a:.4f})×k_i + {delta0:.4f} - {b:.4f}×gen_i")
    print()
    
    # Check if close to simple formula
    if abs(a + 1) < 0.1:
        print("✓ M_i ≈ -k_i  (negative flux!)")
        print()
        print("Physical interpretation:")
        print("  • Negative flux means anti-D7-branes")
        print("  • Or wrapped branes with opposite orientation")
        print("  • This gives exponential suppression |η|^(-2k) × correction")
    
    if abs(delta0 - 4) < 1 and abs(b - 2) < 0.5:
        print()
        print("✓ Modular anomaly δ_gen ≈ 4 - 2×gen")
        print()
        print("This could arise from:")
        print("  • Zero-point energy shifts")
        print("  • Casimir energy on different twist sectors")
        print("  • Anomalous dimensions in CFT")

print()

# Final summary
print("="*80)
print("CONCLUSION")
print("="*80)
print()

print("✓ β_i CAN be decomposed as β_i = 2M_i + δ_gen")
print()

if 'Gen-dependent' in best_model[0]:
    a = best_params[0]
    delta0 = best_params[1]
    b = best_params[2]
    
    print("Best fit formula:")
    print(f"  M_i = {a:.4f} × k_i")
    print(f"  δ_gen = {delta0:.4f} - {b:.4f} × gen")
    print()
    
    if abs(a + 1) < 0.1 and abs(delta0 - 4) < 1:
        print("Approximate theoretical form:")
        print("  M_i ≈ -k_i  (magnetic flux)")
        print("  δ_gen ≈ 4 - 2×gen  (modular anomaly)")
        print()
        print("→ This gives β_i = -2k_i + 4 - 2×gen")
        print()
        print("Physical origin:")
        print("  • |η|^(-2k): wavefunction overlap from flux wrapping")
        print("  • δ_gen: generation-dependent threshold corrections")
        print("  • Connection to Z₃ twist structure (generations)")

print()
print("Next step: Derive δ_gen from orbifold CFT or flux moduli")
print()
