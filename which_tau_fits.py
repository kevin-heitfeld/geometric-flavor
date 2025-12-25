"""
CRITICAL TEST: Which τ value actually fits the Standard Model data?

We have conflicting τ values:
- Manuscript: τ = 1.2 + 0.8i
- Theory #14: τ = 2.69i

These are NOT SL(2,ℤ) equivalent (j-invariants differ by 10^7).
They give DIFFERENT modular forms → DIFFERENT predictions.

Let's check which one actually fits the data!
"""

import numpy as np
from scipy.optimize import minimize

# Experimental values at M_Z
# Masses in GeV
m_e_exp = 0.000510998950
m_mu_exp = 0.1056583755
m_tau_exp = 1.77686

m_u_exp = 0.00216  # 2.16 MeV
m_c_exp = 1.27
m_t_exp = 172.69

m_d_exp = 0.00467  # 4.67 MeV
m_s_exp = 0.093
m_b_exp = 4.18

# CKM angles (radians)
theta12_ckm_exp = 0.22736  # 13.04°
theta23_ckm_exp = 0.04214  # 2.41°
theta13_ckm_exp = 0.00359  # 0.206°

# Simple modular form model (weight k)
def modular_forms(tau, k_sector):
    """
    Compute modular forms at given τ

    For weight k, the Yukawa ratio structure is:
    Y_i ~ |η(τ)|^(2k_i) where η is Dedekind eta function
    """
    q = np.exp(2j * np.pi * tau)

    # Dedekind eta (using first 50 terms)
    eta = q**(1/24)
    for n in range(1, 50):
        eta *= (1 - q**n)

    # For different generations (different modular weights)
    # Generation i has weight k_i → suppression |η|^(2k_i)
    weights = np.array([k_sector, k_sector-2, k_sector-4])  # 3rd, 2nd, 1st generation

    Y_ratios = np.abs(eta)**(2 * weights)

    # Normalize to 3rd generation
    Y_ratios = Y_ratios / Y_ratios[0]

    return Y_ratios

def compute_masses(tau, k_lep, k_up, k_down, Y0_lep, Y0_up, Y0_down):
    """
    Compute fermion masses from modular parameter τ
    """
    # Get Yukawa ratios
    Y_lep = modular_forms(tau, k_lep)
    Y_up = modular_forms(tau, k_up)
    Y_down = modular_forms(tau, k_down)

    # Masses (Y0 sets overall scale)
    m_tau = Y0_lep * Y_lep[0]
    m_mu = Y0_lep * Y_lep[1]
    m_e = Y0_lep * Y_lep[2]

    m_t = Y0_up * Y_up[0]
    m_c = Y0_up * Y_up[1]
    m_u = Y0_up * Y_up[2]

    m_b = Y0_down * Y_down[0]
    m_s = Y0_down * Y_down[1]
    m_d = Y0_down * Y_down[2]

    return (m_e, m_mu, m_tau, m_u, m_c, m_t, m_d, m_s, m_b)

def chi_squared(tau, k_lep=8, k_up=6, k_down=4):
    """
    Compute χ² for a given τ (with fixed k values)

    Free parameters: overall Yukawa scales Y0
    """
    # Optimize Y0 values to minimize χ²
    def objective(Y0_log):
        Y0_lep = np.exp(Y0_log[0])
        Y0_up = np.exp(Y0_log[1])
        Y0_down = np.exp(Y0_log[2])

        masses = compute_masses(tau, k_lep, k_up, k_down, Y0_lep, Y0_up, Y0_down)
        m_e, m_mu, m_tau, m_u, m_c, m_t, m_d, m_s, m_b = masses

        # χ² (9 observables)
        chi2 = 0.0
        chi2 += ((m_e - m_e_exp) / m_e_exp)**2
        chi2 += ((m_mu - m_mu_exp) / m_mu_exp)**2
        chi2 += ((m_tau - m_tau_exp) / m_tau_exp)**2
        chi2 += ((m_u - m_u_exp) / m_u_exp)**2
        chi2 += ((m_c - m_c_exp) / m_c_exp)**2
        chi2 += ((m_t - m_t_exp) / m_t_exp)**2
        chi2 += ((m_d - m_d_exp) / m_d_exp)**2
        chi2 += ((m_s - m_s_exp) / m_s_exp)**2
        chi2 += ((m_b - m_b_exp) / m_b_exp)**2

        return chi2

    # Initial guess for Y0 (log scale)
    Y0_init = np.log([1.0, 100.0, 1.0])

    # Optimize
    result = minimize(objective, Y0_init, method='Nelder-Mead')

    return result.fun, result.x

# Test the three candidate τ values
tau_candidates = {
    'manuscript (1.2 + 0.8i)': 1.2 + 0.8j,
    'orbifold (0.5 + 1.6i)': 0.5 + 1.6j,
    'theory14 (2.69i)': 2.69j,
}

print("="*70)
print("CRITICAL TEST: Which τ fits the Standard Model data?")
print("="*70)
print("\nComparing three candidate values:")
print("  1. τ = 1.2 + 0.8i (manuscript baseline)")
print("  2. τ = 0.5 + 1.6i (orbifold fixed point)")
print("  3. τ = 2.69i (Theory #14 fit)")
print("\nModel: Simple modular weights (k_lep=8, k_up=6, k_down=4)")
print("Free parameters: Three overall Yukawa scales Y0")
print("Observables: 9 fermion masses")
print("="*70)

results = {}
for name, tau in tau_candidates.items():
    print(f"\nTesting τ = {tau:.3f} ({name})...")

    chi2, Y0_opt = chi_squared(tau, k_lep=8, k_up=6, k_down=4)
    chi2_dof = chi2 / (9 - 3)  # 9 observables, 3 parameters

    Y0_lep, Y0_up, Y0_down = np.exp(Y0_opt)

    # Compute final masses
    masses = compute_masses(tau, 8, 6, 4, Y0_lep, Y0_up, Y0_down)
    m_e, m_mu, m_tau, m_u, m_c, m_t, m_d, m_s, m_b = masses

    print(f"  χ² = {chi2:.2f}")
    print(f"  χ²/dof = {chi2_dof:.2f}")
    print(f"\n  Optimal scales:")
    print(f"    Y₀(lepton) = {Y0_lep:.4f}")
    print(f"    Y₀(up)     = {Y0_up:.2f}")
    print(f"    Y₀(down)   = {Y0_down:.4f}")

    print(f"\n  Mass predictions vs data:")
    print(f"    m_e: {m_e*1e3:.4f} MeV vs {m_e_exp*1e3:.4f} MeV (Δ = {abs(m_e-m_e_exp)/m_e_exp*100:.1f}%)")
    print(f"    m_μ: {m_mu:.4f} GeV vs {m_mu_exp:.4f} GeV (Δ = {abs(m_mu-m_mu_exp)/m_mu_exp*100:.1f}%)")
    print(f"    m_τ: {m_tau:.3f} GeV vs {m_tau_exp:.3f} GeV (Δ = {abs(m_tau-m_tau_exp)/m_tau_exp*100:.1f}%)")
    print(f"    m_u: {m_u*1e3:.3f} MeV vs {m_u_exp*1e3:.3f} MeV (Δ = {abs(m_u-m_u_exp)/m_u_exp*100:.1f}%)")
    print(f"    m_c: {m_c:.3f} GeV vs {m_c_exp:.3f} GeV (Δ = {abs(m_c-m_c_exp)/m_c_exp*100:.1f}%)")
    print(f"    m_t: {m_t:.2f} GeV vs {m_t_exp:.2f} GeV (Δ = {abs(m_t-m_t_exp)/m_t_exp*100:.1f}%)")
    print(f"    m_d: {m_d*1e3:.3f} MeV vs {m_d_exp*1e3:.3f} MeV (Δ = {abs(m_d-m_d_exp)/m_d_exp*100:.1f}%)")
    print(f"    m_s: {m_s*1e3:.1f} MeV vs {m_s_exp*1e3:.1f} MeV (Δ = {abs(m_s-m_s_exp)/m_s_exp*100:.1f}%)")
    print(f"    m_b: {m_b:.3f} GeV vs {m_b_exp:.3f} GeV (Δ = {abs(m_b-m_b_exp)/m_b_exp*100:.1f}%)")

    # Store results
    results[name] = {
        'tau': tau,
        'chi2': chi2,
        'chi2_dof': chi2_dof,
        'masses': masses
    }

print("\n" + "="*70)
print("SUMMARY: Which τ fits best?")
print("="*70)

# Sort by χ²/dof
sorted_results = sorted(results.items(), key=lambda x: x[1]['chi2_dof'])

print("\nRanking by fit quality:")
for i, (name, res) in enumerate(sorted_results, 1):
    chi2_dof = res['chi2_dof']
    status = "EXCELLENT" if chi2_dof < 2 else "ACCEPTABLE" if chi2_dof < 5 else "POOR"
    print(f"  {i}. {name}: χ²/dof = {chi2_dof:.2f} ({status})")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

best_name, best_res = sorted_results[0]
print(f"\n✓ BEST FIT: {best_name}")
print(f"  χ²/dof = {best_res['chi2_dof']:.2f}")

if 'theory14' in best_name:
    print("\n→ Theory #14 value (τ = 2.69i) is CORRECT")
    print("→ Manuscript value (τ = 1.2 + 0.8i) is WRONG")
    print("\n**ACTION REQUIRED**: Update manuscript to use τ = 2.69i throughout!")
elif 'manuscript' in best_name:
    print("\n→ Manuscript value (τ = 1.2 + 0.8i) is CORRECT")
    print("→ Theory #14 value (τ = 2.69i) is WRONG")
    print("\n**ACTION REQUIRED**: Re-run Theory #14 and update cosmology!")
elif 'orbifold' in best_name:
    print("\n→ Orbifold value (τ = 0.5 + 1.6i) is CORRECT")
    print("→ Both manuscript and Theory #14 are WRONG")
    print("\n**ACTION REQUIRED**: Update everything to use τ = 0.5 + 1.6i!")

print("\n" + "="*70)
