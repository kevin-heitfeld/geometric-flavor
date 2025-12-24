"""
Simple RG Running Test: M_Z → 1 TeV
====================================

Following Grok's advice, but with practical scale to avoid numerical issues.
Test if RG running affects CKM elements at collider scales.
"""

import numpy as np
from scipy.integrate import solve_ivp

# Constants
M_Z = 91.1876  # GeV
v_higgs = 246.22  # GeV

# Quark masses at M_Z (GeV, MS-bar)
MASSES = {
    'u': 0.00216,
    'c': 1.27,
    't': 172.76,
    'd': 0.00467,
    's': 0.0933333,
    'b': 4.18,
}


def ckm_matrix(theta12, theta23, theta13, delta):
    """Standard parameterization of CKM matrix."""
    c12, s12 = np.cos(theta12), np.sin(theta12)
    c23, s23 = np.cos(theta23), np.sin(theta23)
    c13, s13 = np.cos(theta13), np.sin(theta13)

    return np.array([
        [c12*c13, s12*c13, s13*np.exp(-1j*delta)],
        [-s12*c23 - c12*s23*s13*np.exp(1j*delta),
          c12*c23 - s12*s23*s13*np.exp(1j*delta),
          s23*c13],
        [s12*s23 - c12*c23*s13*np.exp(1j*delta),
         -c12*s23 - s12*c23*s13*np.exp(1j*delta),
          c23*c13]
    ])


def initial_yukawas():
    """Construct Yukawa matrices at M_Z with CKM in down sector."""

    # Yukawa couplings from masses
    y_u = np.array([MASSES['u'], MASSES['c'], MASSES['t']]) * np.sqrt(2) / v_higgs
    y_d = np.array([MASSES['d'], MASSES['s'], MASSES['b']]) * np.sqrt(2) / v_higgs

    # Our predicted CKM
    V = ckm_matrix(
        theta12=np.radians(12.9 - 1.55),
        theta23=np.radians(2.40),
        theta13=np.radians(0.21),
        delta=np.radians(66.5)
    )

    # Y_u diagonal, Y_d = V† diag(y_d)
    Y_u = np.diag(y_u)
    Y_d = V.conj().T @ np.diag(y_d)

    return Y_u, Y_d, V


def beta_functions(t, y, trace_Y_u_sq, trace_Y_d_sq):
    """
    One-loop beta functions for gauge + Yukawa couplings.
    Simplified: only keep dominant terms.
    """
    # Extract values
    g1, g2, g3 = y[0:3]
    Y_u = (y[3:12] + 1j*y[12:21]).reshape(3,3)
    Y_d = (y[21:30] + 1j*y[30:39]).reshape(3,3)

    # Gauge beta functions (SM one-loop)
    beta_g1 = (16*np.pi**2)**(-1) * g1**3 * 41/10
    beta_g2 = (16*np.pi**2)**(-1) * g2**3 * (-19/6)
    beta_g3 = (16*np.pi**2)**(-1) * g3**3 * (-7)

    # Yukawa beta functions (dominant terms)
    # β_Y_u = Y_u × [3/2 Tr(Y_u† Y_u + Y_d† Y_d) + (Y_u† Y_u) - g terms]
    # β_Y_d = Y_d × [3/2 Tr(Y_u† Y_u + Y_d† Y_d) + (Y_d† Y_d) - g terms]

    T_u = Y_u.conj().T @ Y_u
    T_d = Y_d.conj().T @ Y_d

    trace_term = 3/2 * (np.trace(T_u) + np.trace(T_d))

    gauge_u = 17/20*g1**2 + 9/4*g2**2 + 8*g3**2
    gauge_d = 1/4*g1**2 + 9/4*g2**2 + 8*g3**2

    beta_Y_u = (16*np.pi**2)**(-1) * Y_u @ (trace_term*np.eye(3) + 3*T_u - gauge_u*np.eye(3))
    beta_Y_d = (16*np.pi**2)**(-1) * Y_d @ (trace_term*np.eye(3) + 3*T_d - gauge_d*np.eye(3))

    # Pack output
    return np.concatenate([
        [beta_g1, beta_g2, beta_g3],
        beta_Y_u.real.flatten(),
        beta_Y_u.imag.flatten(),
        beta_Y_d.real.flatten(),
        beta_Y_d.imag.flatten(),
    ])


def run_to_scale(M_final=1000):
    """Run from M_Z to M_final GeV."""

    print(f"\n{'='*70}")
    print(f"RG RUNNING: M_Z ({M_Z:.1f} GeV) → {M_final:.1f} GeV")
    print(f"{'='*70}\n")

    # Initial conditions
    Y_u, Y_d, V_initial = initial_yukawas()

    # Gauge couplings at M_Z
    alpha_em = 1/127.95
    alpha_s = 0.1179
    sin2_w = 0.23122

    g1 = np.sqrt(60*np.pi*alpha_em / (3*(3-4*sin2_w)))
    g2 = np.sqrt(4*np.pi*alpha_em / sin2_w)
    g3 = np.sqrt(4*np.pi*alpha_s)

    print(f"Initial state at M_Z:")
    print(f"  |V_cd| = {abs(V_initial[1,0]):.5f}")
    print(f"  y_t = {abs(np.diag(Y_u)[2]):.4f}")
    print(f"  g1, g2, g3 = {g1:.4f}, {g2:.4f}, {g3:.4f}\n")

    # Pack initial state
    y0 = np.concatenate([
        [g1, g2, g3],
        Y_u.real.flatten(),
        Y_u.imag.flatten(),
        Y_d.real.flatten(),
        Y_d.imag.flatten(),
    ])

    # Integration
    t0 = 0
    t1 = np.log(M_final / M_Z)

    # Precompute traces for beta function
    trace_Y_u_sq = np.trace(Y_u.conj().T @ Y_u)
    trace_Y_d_sq = np.trace(Y_d.conj().T @ Y_d)

    sol = solve_ivp(
        lambda t, y: beta_functions(t, y, trace_Y_u_sq, trace_Y_d_sq),
        (t0, t1),
        y0,
        method='RK45',
        rtol=1e-6,
        atol=1e-8,
        dense_output=True,
    )

    if not sol.success:
        print(f"⚠️ Integration failed: {sol.message}")
        return None

    # Final state
    y_final = sol.y[:, -1]
    Y_u_final = (y_final[3:12] + 1j*y_final[12:21]).reshape(3,3)
    Y_d_final = (y_final[21:30] + 1j*y_final[30:39]).reshape(3,3)

    # Extract CKM via SVD
    from scipy.linalg import svd
    U_u, _, _ = svd(Y_u_final)
    U_d, _, _ = svd(Y_d_final)
    V_final = U_u.conj().T @ U_d

    print(f"Final state at {M_final:.1f} GeV:")
    print(f"  |V_cd| = {abs(V_final[1,0]):.5f}")
    print(f"  y_t = {abs(np.linalg.eigvals(Y_u_final.conj().T @ Y_u_final)[2]**0.5):.4f}")
    print(f"  g1, g2, g3 = {y_final[0]:.4f}, {y_final[1]:.4f}, {y_final[2]:.4f}\n")

    # Compare
    shift = (abs(V_final[1,0]) - abs(V_initial[1,0])) / abs(V_initial[1,0]) * 100
    print(f"V_cd shift: {abs(V_initial[1,0]):.5f} → {abs(V_final[1,0]):.5f} ({shift:+.2f}%)")

    # Check against observation
    V_cd_obs = 0.221
    V_cd_err = 0.001
    dev_initial = (abs(V_initial[1,0]) - V_cd_obs) / V_cd_err
    dev_final = (abs(V_final[1,0]) - V_cd_obs) / V_cd_err

    print(f"\nComparison with PDG (V_cd = {V_cd_obs} ± {V_cd_err}):")
    print(f"  At M_Z:       {dev_initial:+.1f}σ")
    print(f"  At {M_final:.0f} GeV:  {dev_final:+.1f}σ")

    if abs(dev_final) < abs(dev_initial):
        print(f"  ✓ RG running IMPROVES agreement!")
    else:
        print(f"  ✗ RG running WORSENS agreement")

    return sol


if __name__ == "__main__":
    # Test at 1 TeV
    run_to_scale(1000)

    # Test at 10 TeV
    print("\n" + "="*70 + "\n")
    run_to_scale(10000)
