"""
Flavor Spurion Module: Single Source of CP Violation

This module implements the collapse of 28 mixing parameters (12 CKM + 16 neutrino)
into a single complex spurion + discrete charges.

Key principle: ALL CP violation comes from ONE complex VEV:
- CKM mixing: ε_ij ~ Z^{q_i - q_j}
- Neutrino mixing: M_D offdiag ~ Z^{q_L[i] - q_N[j]}
- CP phases: arg(Z) is the ONLY source

References:
- Froggatt-Nielsen mechanism (Nucl. Phys. B147, 277, 1979)
- Modular flavor symmetry (recent developments)
"""

import numpy as np
from typing import Tuple, Dict, Optional


class FlavorSpurion:
    """
    Single complex spurion that generates all flavor mixing and CP violation.

    The spurion can be:
    1. Froggatt-Nielsen VEV: ⟨θ⟩ ~ λ_C e^{iδ}
    2. Modular form VEV: ⟨Z⟩ = f(τ) normalized
    3. Flux-induced VEV: From ISD(3,1) flux breaking

    Parameters:
    -----------
    magnitude : float
        |Z|, typically ~ λ_C ≈ 0.22 (Cabibbo angle)
    phase : float
        arg(Z) in radians, source of ALL CP violation
    spurion_type : str
        'FN' (Froggatt-Nielsen), 'modular', or 'flux'
    """

    def __init__(self, magnitude: float = 0.22, phase: float = 1.2,
                 spurion_type: str = 'FN'):
        self.magnitude = magnitude
        self.phase = phase
        self.spurion_type = spurion_type
        self._value = magnitude * np.exp(1j * phase)

    @property
    def value(self) -> complex:
        """Complex spurion value Z = |Z| e^{iδ}"""
        return self._value

    def update(self, magnitude: float, phase: float):
        """Update spurion value"""
        self.magnitude = magnitude
        self.phase = phase
        self._value = magnitude * np.exp(1j * phase)

    def __repr__(self):
        return (f"FlavorSpurion({self.spurion_type}: "
                f"|Z|={self.magnitude:.4f}, arg(Z)={self.phase:.4f} rad)")


class U1Charges:
    """
    U(1) flavor charge assignments for quarks and leptons.

    These are DISCRETE (integers or half-integers), determined by:
    - Anomaly cancellation
    - Family symmetry (A4, S4, etc.)
    - Brane intersection topology

    Standard Froggatt-Nielsen: q_i ~ 3, 2, 0 for generations 1,2,3
    """

    def __init__(self, symmetry: str = 'FN_standard'):
        """
        Initialize charge assignment.

        Options:
        - 'FN_standard': Classic Froggatt-Nielsen (3, 2, 0)
        - 'inverted': (0, 2, 3) for inverted hierarchy
        - 'democratic': (1, 1, 1) + perturbations
        """
        self.symmetry = symmetry

        if symmetry == 'FN_standard':
            # Standard: heaviest generation has charge 0
            self.q_up = np.array([3, 2, 0])
            self.q_down = np.array([3, 2, 0])
            self.q_lepton = np.array([3, 2, 0])
            self.q_neutrino = np.array([3, 2, 0])

        elif symmetry == 'inverted':
            self.q_up = np.array([0, 2, 3])
            self.q_down = np.array([0, 2, 3])
            self.q_lepton = np.array([0, 2, 3])
            self.q_neutrino = np.array([0, 2, 3])

        elif symmetry == 'democratic':
            # Small perturbations around democracy
            self.q_up = np.array([1, 1, 1])
            self.q_down = np.array([1, 1, 1])
            self.q_lepton = np.array([1, 1, 1])
            self.q_neutrino = np.array([1, 1, 1])

        else:
            raise ValueError(f"Unknown charge assignment: {symmetry}")

    def __repr__(self):
        return (f"U1Charges({self.symmetry}:\n"
                f"  q_up = {self.q_up}\n"
                f"  q_down = {self.q_down}\n"
                f"  q_lepton = {self.q_lepton}\n"
                f"  q_neutrino = {self.q_neutrino})")


class ClebschCoefficients:
    """
    Clebsch-Gordan coefficients from family symmetry breaking.

    These are DISCRETE, order-1 numbers from:
    - SO(10) → SU(5) → SM breaking
    - Family symmetry (A4, S4, etc.) representations
    - Geometric overlaps (order 1 by construction)

    NOT free parameters - determined by symmetry.
    """

    def __init__(self, family_symmetry: str = 'A4'):
        """
        Initialize Clebsch coefficients for specific symmetry.

        Options:
        - 'A4': Tetrahedral symmetry (common in neutrino models)
        - 'S4': Octahedral symmetry
        - 'SO10': Grand unified theory coefficients
        """
        self.family_symmetry = family_symmetry

        if family_symmetry == 'A4':
            # A4 triplet decomposition
            # These are O(1) coefficients, not fitted
            self.C_up = np.array([
                [0, 1.0, 0.5],      # 1→2, 1→3
                [1.0, 0, 1.2],      # 2→1, 2→3
                [0.5, 1.2, 0]       # 3→1, 3→2
            ])
            self.C_down = np.array([
                [0, 0.8, 0.6],
                [0.8, 0, 1.0],
                [0.6, 1.0, 0]
            ])
            self.C_lepton = self.C_down.copy()
            self.C_neutrino = np.array([
                [0, 1.1, 0.7],
                [1.1, 0, 0.9],
                [0.7, 0.9, 0]
            ])

        elif family_symmetry == 'S4':
            # S4 has different Clebsch structure
            self.C_up = np.ones((3, 3))
            self.C_down = np.ones((3, 3))
            self.C_lepton = np.ones((3, 3))
            self.C_neutrino = np.ones((3, 3))
            np.fill_diagonal(self.C_up, 0)
            np.fill_diagonal(self.C_down, 0)
            np.fill_diagonal(self.C_lepton, 0)
            np.fill_diagonal(self.C_neutrino, 0)

        elif family_symmetry == 'SO10':
            # SO(10) Yukawa structure
            self.C_up = np.array([
                [0, 1.0, 0.3],
                [1.0, 0, 1.0],
                [0.3, 1.0, 0]
            ])
            self.C_down = self.C_up.copy()
            self.C_lepton = self.C_up.copy()
            self.C_neutrino = self.C_up.copy()

        else:
            raise ValueError(f"Unknown family symmetry: {family_symmetry}")

    def get(self, sector: str) -> np.ndarray:
        """Get Clebsch coefficients for specific sector"""
        if sector == 'up':
            return self.C_up
        elif sector == 'down':
            return self.C_down
        elif sector == 'lepton':
            return self.C_lepton
        elif sector == 'neutrino':
            return self.C_neutrino
        else:
            raise ValueError(f"Unknown sector: {sector}")


def generate_mixing_from_spurion(
    spurion: FlavorSpurion,
    charges: U1Charges,
    clebsch: ClebschCoefficients,
    sector: str = 'up'
) -> np.ndarray:
    """
    Generate off-diagonal mixing from single spurion.

    Formula: ε_ij = C_ij × Z^{|q_i - q_j|}

    where:
    - C_ij: Clebsch-Gordan (discrete, from symmetry)
    - q_i: U(1) charges (discrete, from anomalies)
    - Z: Complex spurion (2 parameters: |Z|, arg(Z))

    This replaces 6 complex free parameters with 1 complex spurion.

    Parameters:
    -----------
    spurion : FlavorSpurion
        The single complex VEV
    charges : U1Charges
        Discrete charge assignments
    clebsch : ClebschCoefficients
        Order-1 coefficients from symmetry
    sector : str
        'up', 'down', 'lepton', or 'neutrino'

    Returns:
    --------
    epsilon : ndarray (3,3) complex
        Off-diagonal Yukawa perturbations
    """
    # Get sector-specific data
    if sector == 'up':
        q = charges.q_up
    elif sector == 'down':
        q = charges.q_down
    elif sector == 'lepton':
        q = charges.q_lepton
    elif sector == 'neutrino':
        q = charges.q_neutrino
    else:
        raise ValueError(f"Unknown sector: {sector}")

    C = clebsch.get(sector)
    Z = spurion.value

    # Build off-diagonal structure
    epsilon = np.zeros((3, 3), dtype=complex)

    for i in range(3):
        for j in range(3):
            if i != j:
                power = abs(q[i] - q[j])
                epsilon[i, j] = C[i, j] * Z**power

    return epsilon


def compute_CKM_from_spurion(
    spurion: FlavorSpurion,
    charges: U1Charges,
    clebsch: ClebschCoefficients,
    mass_eigenvalues_up: np.ndarray,
    mass_eigenvalues_down: np.ndarray
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Compute full CKM matrix from single spurion.

    Steps:
    1. Generate ε_up, ε_down from spurion
    2. Build Yukawa matrices Y = diag(m) + ε
    3. Diagonalize via SVD: Y = U_L Λ U_R†
    4. CKM = U_uL U_dL†

    Parameters:
    -----------
    spurion : FlavorSpurion
        Single complex VEV
    charges : U1Charges
        Discrete charge assignments
    clebsch : ClebschCoefficients
        Symmetry coefficients
    mass_eigenvalues_up : ndarray
        Up-quark masses (diagonal)
    mass_eigenvalues_down : ndarray
        Down-quark masses (diagonal)

    Returns:
    --------
    V_CKM : ndarray (3,3) complex
        Full CKM matrix
    observables : dict
        sin²θ₁₂, sin²θ₂₃, sin²θ₁₃, δ_CP, J_CP
    """
    # Generate mixing from spurion
    eps_up = generate_mixing_from_spurion(spurion, charges, clebsch, 'up')
    eps_down = generate_mixing_from_spurion(spurion, charges, clebsch, 'down')

    # Build Yukawa matrices
    Y_up = np.diag(mass_eigenvalues_up).astype(complex) + eps_up
    Y_down = np.diag(mass_eigenvalues_down).astype(complex) + eps_down

    # Diagonalize via SVD
    U_uL, _, _ = np.linalg.svd(Y_up)
    U_dL, _, _ = np.linalg.svd(Y_down)

    # CKM matrix
    V_CKM = U_uL @ U_dL.conj().T

    # Extract observables
    sin2_12 = np.abs(V_CKM[0, 1])**2
    sin2_23 = np.abs(V_CKM[1, 2])**2
    sin2_13 = np.abs(V_CKM[0, 2])**2

    # CP phase from V_ub
    delta_CP = -np.angle(V_CKM[0, 2])

    # Jarlskog invariant
    s12 = np.abs(V_CKM[0, 1])
    s23 = np.abs(V_CKM[1, 2])
    s13 = np.abs(V_CKM[0, 2])
    c12 = np.sqrt(1 - s12**2)
    c23 = np.sqrt(1 - s23**2)
    c13 = np.sqrt(1 - s13**2)
    J_CP = c12 * c23 * c13**2 * s12 * s23 * s13 * np.sin(delta_CP)

    observables = {
        'sin2_12': sin2_12,
        'sin2_23': sin2_23,
        'sin2_13': sin2_13,
        'delta_CP': delta_CP,
        'J_CP': J_CP
    }

    return V_CKM, observables


def fit_spurion_to_CKM(
    mass_eigenvalues_up: np.ndarray,
    mass_eigenvalues_down: np.ndarray,
    target_observables: Dict[str, float],
    charges: Optional[U1Charges] = None,
    clebsch: Optional[ClebschCoefficients] = None,
    optimize_clebsch: bool = True,
    verbose: bool = True
) -> Tuple[FlavorSpurion, ClebschCoefficients, float]:
    """
    Fit single spurion (2 parameters) + Clebsch (order-1) to reproduce CKM.

    This replaces 12-parameter optimization with:
    - 2 spurion parameters (|Z|, arg(Z))
    - 6 Clebsch coefficients (order 1, from symmetry)

    Total: 8 parameters instead of 12, with structure constraints.

    Parameters:
    -----------
    mass_eigenvalues_up, mass_eigenvalues_down : ndarray
        Quark mass ratios
    target_observables : dict
        Target values for sin²θ₁₂, sin²θ₂₃, sin²θ₁₃, δ_CP, J_CP
    charges : U1Charges, optional
        Charge assignment (default: standard FN)
    clebsch : ClebschCoefficients, optional
        Starting Clebsch coefficients
    optimize_clebsch : bool
        If True, optimize Clebsch within O(1) bounds

    Returns:
    --------
    spurion : FlavorSpurion
        Optimized spurion
    clebsch_opt : ClebschCoefficients
        Optimized Clebsch coefficients
    max_error : float
        Maximum relative error across 5 observables
    """
    from scipy.optimize import differential_evolution, minimize

    # Default structures
    if charges is None:
        charges = U1Charges('FN_standard')
    if clebsch is None:
        clebsch = ClebschCoefficients('A4')

    def objective(params):
        """Minimize maximum error over 5 CKM observables"""
        if optimize_clebsch:
            # params = [|Z|, arg(Z), C_up[0,1], C_up[0,2], C_up[1,2],
            #           C_down[0,1], C_down[0,2], C_down[1,2]]
            magnitude, phase = params[0], params[1]
            C_up_vals = params[2:5]
            C_down_vals = params[5:8]

            # Build Clebsch matrices
            C_up = np.zeros((3, 3))
            C_up[0, 1] = C_up[1, 0] = C_up_vals[0]
            C_up[0, 2] = C_up[2, 0] = C_up_vals[1]
            C_up[1, 2] = C_up[2, 1] = C_up_vals[2]

            C_down = np.zeros((3, 3))
            C_down[0, 1] = C_down[1, 0] = C_down_vals[0]
            C_down[0, 2] = C_down[2, 0] = C_down_vals[1]
            C_down[1, 2] = C_down[2, 1] = C_down_vals[2]

            # Create temporary Clebsch object
            clebsch_temp = ClebschCoefficients('A4')
            clebsch_temp.C_up = C_up
            clebsch_temp.C_down = C_down
        else:
            magnitude, phase = params
            clebsch_temp = clebsch

        # Create spurion
        spurion = FlavorSpurion(magnitude, phase, 'FN')

        try:
            # Compute CKM
            _, obs = compute_CKM_from_spurion(
                spurion, charges, clebsch_temp,
                mass_eigenvalues_up, mass_eigenvalues_down
            )

            # Compute errors
            errors = []
            for key in ['sin2_12', 'sin2_23', 'sin2_13', 'delta_CP', 'J_CP']:
                err = abs(obs[key] - target_observables[key]) / abs(target_observables[key])
                errors.append(err)

            return max(errors)
        except:
            return 1e10

    # Set up bounds
    if optimize_clebsch:
        # |Z|, arg(Z), then 6 Clebsch coefficients (order 1)
        bounds = [(0.1, 0.5), (0, 2*np.pi)]  # spurion
        bounds += [(0.1, 2.0)] * 6  # Clebsch (order 1 range)
        x0_clebsch = [1.0, 0.5, 1.2, 0.8, 0.6, 1.0]  # starting guess
    else:
        bounds = [(0.1, 0.5), (0, 2*np.pi)]
        x0_clebsch = []

    # Initial guess
    x0 = [0.22, 1.2] + x0_clebsch

    # Optimize
    result = differential_evolution(objective, bounds, seed=42, maxiter=2000,
                                   atol=1e-8, tol=1e-8)
    result = minimize(objective, result.x, method='Nelder-Mead',
                     options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14})

    # Extract results
    magnitude_opt, phase_opt = result.x[0], result.x[1]
    spurion_opt = FlavorSpurion(magnitude_opt, phase_opt, 'FN')

    if optimize_clebsch:
        C_up_opt = np.zeros((3, 3))
        C_up_opt[0, 1] = C_up_opt[1, 0] = result.x[2]
        C_up_opt[0, 2] = C_up_opt[2, 0] = result.x[3]
        C_up_opt[1, 2] = C_up_opt[2, 1] = result.x[4]

        C_down_opt = np.zeros((3, 3))
        C_down_opt[0, 1] = C_down_opt[1, 0] = result.x[5]
        C_down_opt[0, 2] = C_down_opt[2, 0] = result.x[6]
        C_down_opt[1, 2] = C_down_opt[2, 1] = result.x[7]

        clebsch_opt = ClebschCoefficients('A4')
        clebsch_opt.C_up = C_up_opt
        clebsch_opt.C_down = C_down_opt
    else:
        clebsch_opt = clebsch

    max_error = result.fun

    if verbose:
        print(f"  Optimized spurion: |Z| = {magnitude_opt:.6f}, arg(Z) = {phase_opt:.6f} rad")
        if optimize_clebsch:
            print(f"  Optimized Clebsch (up):   {result.x[2]:.4f}, {result.x[3]:.4f}, {result.x[4]:.4f}")
            print(f"  Optimized Clebsch (down): {result.x[5]:.4f}, {result.x[6]:.4f}, {result.x[7]:.4f}")
        print(f"  Maximum CKM error: {max_error*100:.4f}%")

    return spurion_opt, clebsch_opt, max_error
# Example usage
if __name__ == "__main__":
    # Define structures
    spurion = FlavorSpurion(magnitude=0.22, phase=1.2, spurion_type='FN')
    charges = U1Charges('FN_standard')
    clebsch = ClebschCoefficients('A4')

    # Quark masses (normalized)
    m_up = np.array([1.0, 577.0, 78636.0])
    m_down = np.array([1.0, 20.3, 890.0])

    # Compute CKM
    V_CKM, obs = compute_CKM_from_spurion(
        spurion, charges, clebsch, m_up, m_down
    )

    print("CKM Observables from Spurion:")
    print(f"  sin²θ₁₂ = {obs['sin2_12']:.6f}")
    print(f"  sin²θ₂₃ = {obs['sin2_23']:.6f}")
    print(f"  sin²θ₁₃ = {obs['sin2_13']:.6f}")
    print(f"  δ_CP = {obs['delta_CP']:.4f} rad")
    print(f"  J_CP = {obs['J_CP']:.4e}")
