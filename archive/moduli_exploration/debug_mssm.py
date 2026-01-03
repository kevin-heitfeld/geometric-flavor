"""
Debug MSSM Running - Quick Test

Testing if MSSM actually helps with sin²θ_W
"""

import numpy as np
from scipy.integrate import solve_ivp

M_Z = 91.1876
M_GUT = 2e16

# Experimental
sin2_theta_W_exp = 0.23122
alpha_s_exp = 0.1179

# Beta functions
b1_MSSM, b2_MSSM, b3_MSSM = 33/5, 1, -3
b1_SM, b2_SM, b3_SM = 41/10, -19/6, -7

def run_MSSM(alpha_GUT, M_SUSY):
    """MSSM from M_GUT to M_SUSY, then SM to M_Z"""

    # Phase 1: MSSM
    def rg_mssm(t, a):
        return np.array([
            (b1_MSSM/(2*np.pi)) * a[0]**2,
            (b2_MSSM/(2*np.pi)) * a[1]**2,
            (b3_MSSM/(2*np.pi)) * a[2]**2
        ])

    sol1 = solve_ivp(rg_mssm, (np.log(M_GUT), np.log(M_SUSY)),
                     [alpha_GUT]*3, rtol=1e-8)
    a_susy = sol1.y[:, -1]

    # Phase 2: SM
    def rg_sm(t, a):
        return np.array([
            (b1_SM/(2*np.pi)) * a[0]**2,
            (b2_SM/(2*np.pi)) * a[1]**2,
            (b3_SM/(2*np.pi)) * a[2]**2
        ])

    sol2 = solve_ivp(rg_sm, (np.log(M_SUSY), np.log(M_Z)),
                     a_susy, rtol=1e-8)
    return sol2.y[:, -1]

def run_SM_only(alpha_GUT):
    """Pure SM from M_GUT to M_Z"""
    def rg_sm(t, a):
        return np.array([
            (b1_SM/(2*np.pi)) * a[0]**2,
            (b2_SM/(2*np.pi)) * a[1]**2,
            (b3_SM/(2*np.pi)) * a[2]**2
        ])

    sol = solve_ivp(rg_sm, (np.log(M_GUT), np.log(M_Z)),
                    [alpha_GUT]*3, rtol=1e-8)
    return sol.y[:, -1]

print("Testing α_GUT values to match α_s(M_Z) = 0.118:")
print()

for alpha_GUT in [0.020, 0.025, 0.030, 0.035, 0.040]:
    # Pure SM
    a_sm = run_SM_only(alpha_GUT)
    sin2_sm = a_sm[0]/(a_sm[0]+a_sm[1])

    # MSSM at 3 TeV
    a_mssm = run_MSSM(alpha_GUT, 3e3)
    sin2_mssm = a_mssm[0]/(a_mssm[0]+a_mssm[1])

    print(f"α_GUT = {alpha_GUT:.3f}")
    print(f"  Pure SM:  α₃={a_sm[2]:.4f}, sin²θ_W={sin2_sm:.4f}")
    print(f"  MSSM(3T): α₃={a_mssm[2]:.4f}, sin²θ_W={sin2_mssm:.4f}")
    print()
