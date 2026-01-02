"""
Optimize Gauge Couplings
=========================

Optimize Kac-Moody levels k = [k₁, k₂, k₃] and string coupling g_s
to match observed gauge couplings at M_Z.

Current errors: α₁ (111%), α₂ (10%), α₃ (87%)
Target: <20% error on all three

Strategy:
- k_i must be positive integers (from string theory)
- g_s is continuous, typically 0.1-1.0
- Use differential evolution for global search
- Then round k values to nearest integers
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution


# Beta functions for RG running (SM + thresholds)
BETA_U1 = {'b1': 41.0/10.0, 'b2': 199.0/50.0}
BETA_SU2 = {'b1': -19.0/6.0, 'b2': 35.0/6.0}
BETA_SU3 = {'b1': -7.0, 'b2': -26.0}


def dedekind_eta(tau, n_terms=50):
    """Dedekind eta function η(τ) = q^(1/24) Π(1-q^n)"""
    q = np.exp(2j * np.pi * tau)
    eta = q**(1.0/24.0)
    for n in range(1, n_terms):
        eta *= (1.0 - q**n)
    return eta


def run_gauge_twoloop(alpha_in, b1, b2, M_in, M_out):
    """
    Run gauge coupling from M_in to M_out using 2-loop RG

    dα/dt = (b1/2π) α² + (b2/8π²) α³

    where t = ln(M)
    """
    t = np.log(M_out / M_in)

    # 1-loop running (dominant)
    alpha_inv_out = 1.0/alpha_in - (b1 / (2*np.pi)) * t
    alpha_1loop = 1.0 / alpha_inv_out

    # 2-loop correction
    correction = (b2 / (8*np.pi**2)) * t * alpha_in
    alpha_out = alpha_1loop * (1 + correction * alpha_1loop)

    return alpha_out


def gauge_coupling_at_MZ(k_i, g_s, tau, beta_1, beta_2):
    """Predict gauge coupling at M_Z from GUT scale"""
    M_GUT = 2e16
    M_Z = 91.2

    # GUT scale: α(M_GUT) = g_s²/k_i
    alpha_GUT = g_s**2 / k_i

    # String threshold from η(τ)
    eta = dedekind_eta(tau)
    threshold = np.real(np.log(eta)) * (k_i / 12.0)
    alpha_inv_GUT = 1.0 / alpha_GUT + threshold

    # 2-loop RG running
    alpha_GUT_eff = 1.0 / alpha_inv_GUT
    alpha_MZ = run_gauge_twoloop(alpha_GUT_eff, beta_1, beta_2, M_GUT, M_Z)

    return alpha_MZ


# Modular parameter
tau = 1j  # Pure imaginary (Re[τ]=0)

# Observed values
alpha_1_obs = (5.0/3.0) / 127.9  # GUT normalized U(1)
alpha_2_obs = 1.0 / 29.6          # SU(2)
alpha_3_obs = 0.1184              # SU(3)


def objective(params):
    """
    Minimize maximum relative error over all 3 gauge couplings.

    Parameters
    ----------
    params : array [4]
        [k₁, k₂, k₃, g_s]
    """
    try:
        k_1, k_2, k_3, g_s = params

        # Predict couplings
        alpha_1_pred = gauge_coupling_at_MZ(k_1, g_s, tau,
                                           BETA_U1['b1'], BETA_U1['b2'])
        alpha_2_pred = gauge_coupling_at_MZ(k_2, g_s, tau,
                                           BETA_SU2['b1'], BETA_SU2['b2'])
        alpha_3_pred = gauge_coupling_at_MZ(k_3, g_s, tau,
                                           BETA_SU3['b1'], BETA_SU3['b2'])

        # Relative errors
        err_1 = abs(alpha_1_pred - alpha_1_obs) / alpha_1_obs
        err_2 = abs(alpha_2_pred - alpha_2_obs) / alpha_2_obs
        err_3 = abs(alpha_3_pred - alpha_3_obs) / alpha_3_obs

        # Minimax
        return max(err_1, err_2, err_3)

    except:
        return 1e10


print("="*80)
print("OPTIMIZING GAUGE COUPLINGS")
print("="*80)
print()
print(f"Target gauge couplings at M_Z:")
print(f"  α₁ = {alpha_1_obs:.4f} = (5/3)/127.9 (GUT normalized)")
print(f"  α₂ = {alpha_2_obs:.4f} = 1/29.6")
print(f"  α₃ = {alpha_3_obs:.4f} = 0.1184")
print()

# Initial guess (current values)
x0 = [8.0, 6.0, 4.0, 0.5]

# Bounds: k_i should be positive integers, g_s in reasonable range
# For continuous optimization, allow k to vary continuously, will round later
bounds = [(1, 15), (1, 15), (1, 15), (0.1, 2.0)]

print("Phase 1: Differential evolution (continuous k values)...")
result_de = differential_evolution(objective, bounds, maxiter=3000,
                                   seed=42, atol=1e-10, tol=1e-10)
print(f"DE complete: max error = {result_de.fun*100:.2f}%")
print(f"  k₁ = {result_de.x[0]:.2f}, k₂ = {result_de.x[1]:.2f}, k₃ = {result_de.x[2]:.2f}, g_s = {result_de.x[3]:.4f}")
print()

print("Phase 2: L-BFGS-B refinement...")
result = minimize(objective, result_de.x, method='L-BFGS-B', bounds=bounds,
                 options={'maxiter': 10000, 'ftol': 1e-14})
print(f"L-BFGS-B complete: max error = {result.fun*100:.2f}%")
print(f"  k₁ = {result.x[0]:.3f}, k₂ = {result.x[1]:.3f}, k₃ = {result.x[2]:.3f}, g_s = {result.x[3]:.4f}")
print()

# Extract continuous results
k_1_cont, k_2_cont, k_3_cont, g_s_opt = result.x

print("="*80)
print("OPTIMIZED PARAMETERS (CONTINUOUS)")
print("="*80)
print()
print(f"Kac-Moody levels (continuous):")
print(f"  k₁ = {k_1_cont:.3f}")
print(f"  k₂ = {k_2_cont:.3f}")
print(f"  k₃ = {k_3_cont:.3f}")
print(f"String coupling:")
print(f"  g_s = {g_s_opt:.4f}")
print()

# Predict with continuous values
alpha_1_cont = gauge_coupling_at_MZ(k_1_cont, g_s_opt, tau,
                                    BETA_U1['b1'], BETA_U1['b2'])
alpha_2_cont = gauge_coupling_at_MZ(k_2_cont, g_s_opt, tau,
                                    BETA_SU2['b1'], BETA_SU2['b2'])
alpha_3_cont = gauge_coupling_at_MZ(k_3_cont, g_s_opt, tau,
                                    BETA_SU3['b1'], BETA_SU3['b2'])

err_1_cont = abs(alpha_1_cont - alpha_1_obs) / alpha_1_obs * 100
err_2_cont = abs(alpha_2_cont - alpha_2_obs) / alpha_2_obs * 100
err_3_cont = abs(alpha_3_cont - alpha_3_obs) / alpha_3_obs * 100

print("PREDICTIONS (CONTINUOUS):")
print(f"  α₁: {alpha_1_cont:.4f} (obs: {alpha_1_obs:.4f}) - error: {err_1_cont:.2f}%")
print(f"  α₂: {alpha_2_cont:.4f} (obs: {alpha_2_obs:.4f}) - error: {err_2_cont:.2f}%")
print(f"  α₃: {alpha_3_cont:.4f} (obs: {alpha_3_obs:.4f}) - error: {err_3_cont:.2f}%")
print(f"  Maximum error: {max(err_1_cont, err_2_cont, err_3_cont):.2f}%")
print()

# Now try rounding k values to integers
print("="*80)
print("ROUNDING TO INTEGER KAC-MOODY LEVELS")
print("="*80)
print()

# Try different rounding combinations
best_int_err = 1e10
best_int_params = None

for dk1 in [-1, 0, 1]:
    for dk2 in [-1, 0, 1]:
        for dk3 in [-1, 0, 1]:
            k_1_int = max(1, int(np.round(k_1_cont)) + dk1)
            k_2_int = max(1, int(np.round(k_2_cont)) + dk2)
            k_3_int = max(1, int(np.round(k_3_cont)) + dk3)

            # Optimize g_s for these integer k values
            def obj_gs(gs):
                alpha_1 = gauge_coupling_at_MZ(k_1_int, gs, tau,
                                              BETA_U1['b1'], BETA_U1['b2'])
                alpha_2 = gauge_coupling_at_MZ(k_2_int, gs, tau,
                                              BETA_SU2['b1'], BETA_SU2['b2'])
                alpha_3 = gauge_coupling_at_MZ(k_3_int, gs, tau,
                                              BETA_SU3['b1'], BETA_SU3['b2'])

                err_1 = abs(alpha_1 - alpha_1_obs) / alpha_1_obs
                err_2 = abs(alpha_2 - alpha_2_obs) / alpha_2_obs
                err_3 = abs(alpha_3 - alpha_3_obs) / alpha_3_obs

                return max(err_1, err_2, err_3)

            result_gs = minimize(obj_gs, [g_s_opt], bounds=[(0.1, 2.0)],
                               method='L-BFGS-B')

            if result_gs.fun < best_int_err:
                best_int_err = result_gs.fun
                best_int_params = (k_1_int, k_2_int, k_3_int, result_gs.x[0])

k_1_int, k_2_int, k_3_int, g_s_int = best_int_params

print(f"Best integer Kac-Moody levels:")
print(f"  k₁ = {k_1_int}")
print(f"  k₂ = {k_2_int}")
print(f"  k₃ = {k_3_int}")
print(f"String coupling:")
print(f"  g_s = {g_s_int:.4f}")
print()

# Predict with integer values
alpha_1_int = gauge_coupling_at_MZ(k_1_int, g_s_int, tau,
                                   BETA_U1['b1'], BETA_U1['b2'])
alpha_2_int = gauge_coupling_at_MZ(k_2_int, g_s_int, tau,
                                   BETA_SU2['b1'], BETA_SU2['b2'])
alpha_3_int = gauge_coupling_at_MZ(k_3_int, g_s_int, tau,
                                   BETA_SU3['b1'], BETA_SU3['b2'])

err_1_int = abs(alpha_1_int - alpha_1_obs) / alpha_1_obs * 100
err_2_int = abs(alpha_2_int - alpha_2_obs) / alpha_2_obs * 100
err_3_int = abs(alpha_3_int - alpha_3_obs) / alpha_3_obs * 100

print("PREDICTIONS (INTEGER k):")
print(f"  α₁: {alpha_1_int:.4f} (obs: {alpha_1_obs:.4f}) - error: {err_1_int:.2f}%")
print(f"  α₂: {alpha_2_int:.4f} (obs: {alpha_2_obs:.4f}) - error: {err_2_int:.2f}%")
print(f"  α₃: {alpha_3_int:.4f} (obs: {alpha_3_obs:.4f}) - error: {err_3_int:.2f}%")
print(f"  Maximum error: {max(err_1_int, err_2_int, err_3_int):.2f}%")
print()

print("="*80)
print("COMPARISON")
print("="*80)
print()
print(f"Current (k=[8,6,4], g_s=0.5):")
print(f"  Errors: α₁ 111%, α₂ 10%, α₃ 87%")
print()
print(f"Continuous optimization:")
print(f"  k = [{k_1_cont:.2f}, {k_2_cont:.2f}, {k_3_cont:.2f}], g_s = {g_s_opt:.4f}")
print(f"  Errors: α₁ {err_1_cont:.1f}%, α₂ {err_2_cont:.1f}%, α₃ {err_3_cont:.1f}%")
print()
print(f"Integer Kac-Moody levels (string theory constraint):")
print(f"  k = [{k_1_int}, {k_2_int}, {k_3_int}], g_s = {g_s_int:.4f}")
print(f"  Errors: α₁ {err_1_int:.1f}%, α₂ {err_2_int:.1f}%, α₃ {err_3_int:.1f}%")
print()

print("="*80)
print("Parameters for unified_predictions_complete.py:")
print("="*80)
print(f"k_CKM = np.array([{k_1_int}, {k_2_int}, {k_3_int}])")
print(f"g_s = {g_s_int:.6f}")
