import numpy as np

def dedekind_eta(tau):
    """Dedekind eta function η(τ)"""
    q = np.exp(2j * np.pi * tau)
    if np.abs(q) > 0.99:
        return np.nan
    product = np.prod([1 - q**n for n in range(1, 50)])
    return q**(1/24) * product

def mass_with_localization(k_i, tau, A_i, g_s, eta_func):
    """Mass calculation with modular form + localization + corrections"""
    eta = eta_func(tau)

    # Modular form contribution
    modular = np.abs(eta ** (k_i / 2.0))**2

    # Localization from D-brane separation
    localization = np.exp(-2.0 * A_i * np.imag(tau))

    # Loop corrections
    q = np.exp(2j * np.pi * tau)
    terms = [n * q**(n-1) / (1 - q**n) for n in range(1, 50)]
    eta_deriv = eta * 2j * np.pi * (1/24 - np.sum(terms))
    loop_corr = g_s**2 * (k_i**2 / (4 * np.pi)) * np.abs(eta_deriv / eta)**2

    # RG running
    M_string = 5e17
    M_Z = 91.2
    gamma_anom = k_i / (16 * np.pi**2)
    rg_factor = (M_Z / M_string)**(gamma_anom)

    return modular * localization * (1.0 + loop_corr) * rg_factor

# Parameters from optimization
tau_0 = 2.7j
c_lep = 13/14
g_factors = np.array([1.0, 1.0612, 1.0139])
tau_lep = tau_0 * c_lep * g_factors

A_lep = np.array([0.0, -0.79570079, -1.06610462])
g_s = np.exp(-np.log(np.imag(tau_0)))  # Fixed: negative log!
k_mass = np.array([8, 6, 4])

print("Test mass calculation:")
print(f"tau_lep = {tau_lep}")
print(f"A_lep = {A_lep}")
print(f"g_s = {g_s}")
print(f"k_mass = {k_mass}")
print()

m_lep = np.array([mass_with_localization(k_mass[i], tau_lep[i], A_lep[i], g_s, dedekind_eta)
                  for i in range(3)])

print(f"m_lep = {m_lep}")
print(f"Ratios: {m_lep[1]/m_lep[0]:.1f}, {m_lep[2]/m_lep[0]:.1f}")
print(f"Observed: 206.8, 3477")
print(f"Errors: {100*abs(m_lep[1]/m_lep[0] - 206.8)/206.8:.1f}%, {100*abs(m_lep[2]/m_lep[0] - 3477)/3477:.1f}%")
