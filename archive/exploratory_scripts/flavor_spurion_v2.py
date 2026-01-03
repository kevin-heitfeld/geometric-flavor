"""
Flavor Spurion Mechanism v2 - Hierarchical Structure

Instead of strict FN formula ε_ij = C_ij × Z^{|qi-qj|}, use:
  ε_ij = α_ij × Z^{n_ij}

Where:
- Z is the single complex spurion (2 parameters: |Z|, arg(Z))
- n_ij are integer powers from charge differences (discrete, fixed)
- α_ij are order-1 prefactors (constrained to ~0.1-3 range)

This gives:
- CKM: 2 spurion + 3 up powers + 3 down powers + 6 prefactors = 14 params
- But: Powers n_ij are integers → only ~3-4 choices each
- So effective freedom is much less than original 24 real parameters

Key improvement over v1:
- Allows α_ij to vary within physical range (0.1-3)
- Instead of demanding exact FN formula
- Still maintains hierarchical structure from spurion powers
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class SpurionConfig:
    """Single complex spurion configuration"""
    magnitude: float  # |Z| ~ λ_C ~ 0.22
    phase: float      # arg(Z) = source of CP violation

    @property
    def value(self) -> complex:
        return self.magnitude * np.exp(1j * self.phase)

    def __repr__(self):
        return f"Spurion(|Z|={self.magnitude:.4f}, arg(Z)={self.phase:.4f} rad)"

@dataclass
class HierarchyConfig:
    """Hierarchical structure from powers and prefactors"""
    powers_up: np.ndarray      # n_ij for up sector (3 values for upper triangle)
    powers_down: np.ndarray    # n_ij for down sector
    prefactors_up: np.ndarray  # α_ij for up sector (order 1)
    prefactors_down: np.ndarray  # α_ij for down sector (order 1)

    def __repr__(self):
        return (f"Hierarchy(n_up={self.powers_up}, n_down={self.powers_down}, "
                f"α_up={self.prefactors_up}, α_down={self.prefactors_down})")


def generate_mixing_v2(spurion: SpurionConfig, hierarchy: HierarchyConfig,
                       sector: str = 'up') -> np.ndarray:
    """
    Generate 3×3 mixing matrix using hierarchical spurion structure.

    Formula: ε_ij = α_ij × Z^{n_ij}

    Parameters:
    -----------
    spurion : SpurionConfig
        Single complex VEV
    hierarchy : HierarchyConfig
        Powers and prefactors
    sector : str
        'up' or 'down'

    Returns:
    --------
    epsilon : ndarray (3×3)
        Off-diagonal mixing matrix
    """
    Z = spurion.value

    if sector == 'up':
        powers = hierarchy.powers_up
        prefactors = hierarchy.prefactors_up
    else:
        powers = hierarchy.powers_down
        prefactors = hierarchy.prefactors_down

    epsilon = np.zeros((3, 3), dtype=complex)

    # Upper triangle
    epsilon[0, 1] = prefactors[0] * Z**powers[0]
    epsilon[0, 2] = prefactors[1] * Z**powers[1]
    epsilon[1, 2] = prefactors[2] * Z**powers[2]

    # Symmetric
    epsilon[1, 0] = epsilon[0, 1]
    epsilon[2, 0] = epsilon[0, 2]
    epsilon[2, 1] = epsilon[1, 2]

    return epsilon


def compute_CKM_v2(spurion: SpurionConfig, hierarchy: HierarchyConfig,
                   mass_up: np.ndarray, mass_down: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Compute CKM matrix from hierarchical spurion structure.

    Returns:
    --------
    V_CKM : ndarray (3×3)
        CKM matrix
    observables : dict
        sin²θ₁₂, sin²θ₂₃, sin²θ₁₃, δ_CP, J_CP
    """
    # Generate mixing matrices
    eps_up = generate_mixing_v2(spurion, hierarchy, 'up')
    eps_down = generate_mixing_v2(spurion, hierarchy, 'down')

    # Build Yukawa matrices
    Y_up = np.diag(mass_up).astype(complex) + eps_up
    Y_down = np.diag(mass_down).astype(complex) + eps_down

    # Diagonalize
    U_uL, _, _ = np.linalg.svd(Y_up)
    U_dL, _, _ = np.linalg.svd(Y_down)

    # CKM matrix
    V_CKM = U_uL @ U_dL.conj().T

    # Extract observables
    s12_sq = abs(V_CKM[0, 1])**2
    s23_sq = abs(V_CKM[1, 2])**2
    s13_sq = abs(V_CKM[0, 2])**2

    # CP phase (Jarlskog)
    J_CP = np.imag(V_CKM[0, 0] * V_CKM[1, 1] * V_CKM[0, 1].conj() * V_CKM[1, 0].conj())

    # Extract δ_CP (approximate)
    delta_CP = np.angle(V_CKM[0, 2])

    observables = {
        'sin2_12': s12_sq,
        'sin2_23': s23_sq,
        'sin2_13': s13_sq,
        'delta_CP': delta_CP,
        'J_CP': abs(J_CP)
    }

    return V_CKM, observables


def fit_spurion_v2(mass_up: np.ndarray, mass_down: np.ndarray,
                   target: Dict[str, float], verbose: bool = True) -> Tuple[SpurionConfig, HierarchyConfig, float]:
    """
    Fit hierarchical spurion structure to CKM observables.

    Parameters:
    -----------
    mass_up, mass_down : ndarray
        Quark mass ratios
    target : dict
        Target CKM observables

    Returns:
    --------
    spurion : SpurionConfig
        Optimized spurion
    hierarchy : HierarchyConfig
        Optimized hierarchy structure
    max_error : float
        Maximum relative error
    """
    from scipy.optimize import differential_evolution, minimize

    def objective(params):
        """Minimize maximum error"""
        # params = [|Z|, arg(Z),
        #           n_up[3], n_down[3],
        #           α_up[3], α_down[3]]

        spurion = SpurionConfig(magnitude=params[0], phase=params[1])

        hierarchy = HierarchyConfig(
            powers_up=np.round(params[2:5]).astype(int),      # Round to integers
            powers_down=np.round(params[5:8]).astype(int),
            prefactors_up=params[8:11],
            prefactors_down=params[11:14]
        )

        try:
            _, obs = compute_CKM_v2(spurion, hierarchy, mass_up, mass_down)

            errors = []
            for key in ['sin2_12', 'sin2_23', 'sin2_13', 'delta_CP', 'J_CP']:
                err = abs(obs[key] - target[key]) / abs(target[key])
                errors.append(err)

            return max(errors)
        except:
            return 1e10

    # Bounds
    bounds = [
        (0.1, 0.5),     # |Z|
        (0, 2*np.pi),   # arg(Z)
        (1, 4), (1, 4), (1, 4),  # n_up (integer powers)
        (1, 4), (1, 4), (1, 4),  # n_down
        (0.1, 3.0), (0.1, 3.0), (0.1, 3.0),  # α_up (order 1)
        (0.1, 3.0), (0.1, 3.0), (0.1, 3.0),  # α_down
    ]

    # Optimize
    result = differential_evolution(objective, bounds, seed=42, maxiter=3000,
                                   atol=1e-10, tol=1e-10, workers=1)
    result = minimize(objective, result.x, method='Nelder-Mead',
                     options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14})

    # Extract results
    spurion_opt = SpurionConfig(magnitude=result.x[0], phase=result.x[1])
    hierarchy_opt = HierarchyConfig(
        powers_up=np.round(result.x[2:5]).astype(int),
        powers_down=np.round(result.x[5:8]).astype(int),
        prefactors_up=result.x[8:11],
        prefactors_down=result.x[11:14]
    )

    max_error = result.fun

    if verbose:
        print(f"Optimized spurion: |Z| = {spurion_opt.magnitude:.6f}, arg(Z) = {spurion_opt.phase:.6f} rad")
        print(f"Powers (up):     {hierarchy_opt.powers_up}")
        print(f"Powers (down):   {hierarchy_opt.powers_down}")
        print(f"Prefactors (up):   {hierarchy_opt.prefactors_up}")
        print(f"Prefactors (down): {hierarchy_opt.prefactors_down}")
        print(f"Maximum error: {max_error*100:.4f}%")

    return spurion_opt, hierarchy_opt, max_error


if __name__ == "__main__":
    # Test
    m_up = np.array([1.0, 577.0, 78636.0])
    m_down = np.array([1.0, 20.3, 890.0])

    target = {
        'sin2_12': 0.0510,
        'sin2_23': 0.00157,
        'sin2_13': 0.000128,
        'delta_CP': 1.22,
        'J_CP': 3.0e-5
    }

    print("Fitting hierarchical spurion structure to CKM...")
    spurion, hierarchy, error = fit_spurion_v2(m_up, m_down, target)

    print(f"\nResult:")
    print(spurion)
    print(hierarchy)
    print(f"Max error: {error*100:.4f}%")
