import numpy as np

# Test gauge coupling calculation
k_gauge = np.array([7, 6, 6])
g_s = 0.361890
tau = 1j

def dedekind_eta(tau, n_terms=50):
    q = np.exp(2j * np.pi * tau)
    eta = q**(1.0/24.0)
    for n in range(1, n_terms):
        eta *= (1.0 - q**n)
    return eta

def run_gauge_twoloop(alpha_in, b1, b2, M_in, M_out):
    t = np.log(M_out / M_in)
    alpha_inv_out = 1.0/alpha_in - (b1 / (2*np.pi)) * t
    alpha_1loop = 1.0 / alpha_inv_out
    correction = (b2 / (8*np.pi**2)) * t * alpha_in
    alpha_out = alpha_1loop * (1 + correction * alpha_1loop)
    return alpha_out

BETA_U1 = {'b1': 41.0/10.0, 'b2': 199.0/50.0}
BETA_SU2 = {'b1': -19.0/6.0, 'b2': 35.0/6.0}
BETA_SU3 = {'b1': -7.0, 'b2': -26.0}

M_GUT = 2e16
M_Z = 91.2

for i, name in enumerate(['U(1)', 'SU(2)', 'SU(3)']):
    k_i = k_gauge[i]
    alpha_GUT = g_s**2 / k_i

    eta = dedekind_eta(tau)
    threshold = np.real(np.log(eta)) * (k_i / 12.0)
    alpha_inv_GUT = 1.0 / alpha_GUT + threshold
    alpha_GUT_eff = 1.0 / alpha_inv_GUT

    if i == 0:
        beta = BETA_U1
    elif i == 1:
        beta = BETA_SU2
    else:
        beta = BETA_SU3

    alpha_MZ = run_gauge_twoloop(alpha_GUT_eff, beta['b1'], beta['b2'], M_GUT, M_Z)

    print(f"{name}: k={k_i}, alpha_GUT={alpha_GUT:.6f}, threshold={threshold:.6f}, alpha_MZ={alpha_MZ:.6f}")

print()
print("Observed:")
print(f"  U(1): {(5.0/3.0)/127.9:.6f}")
print(f"  SU(2): {1.0/29.6:.6f}")
print(f"  SU(3): 0.118400")
