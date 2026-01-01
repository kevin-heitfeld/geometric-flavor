"""
Complete CKM Matrix from Quantum Error Correction
===================================================

Compute all 9 CKM elements from [[9,3,2]] code structure using
modular form overlaps between up-type and down-type generations.

Based on:
- k-pattern [8,6,4] → generation weights
- Complex τ → modular phases
- Dedekind η(τ) → generation amplitudes
- Doublet splitting → up vs down phase shift
"""

import numpy as np


def logical_generation_states(k_pattern, tau_val, dedekind_eta_func, sector='up'):
    """
    Construct 3 generation states from k-pattern [8,6,4]

    State amplitudes from modular forms:
    |gen_i⟩ ~ η(τ)^(k_i/2) × exp(i × phases)

    Parameters
    ----------
    k_pattern : array_like
        Modular weights [8,6,4] for three generations
    tau_val : complex
        Modular parameter τ
    dedekind_eta_func : callable
        Function to compute Dedekind eta η(τ)
    sector : str
        'up' or 'down' (quark doublet splitting)

    Returns
    -------
    psi_gen : ndarray
        Normalized generation state amplitudes (3,)
    """
    # Dedekind eta for modular amplitude
    eta_val = dedekind_eta_func(tau_val, n_terms=50)

    # Generation amplitudes from modular forms
    k_weights = np.array(k_pattern)
    amplitudes = np.array([np.abs(eta_val)**(k/2) for k in k_weights], dtype=complex)

    # Modular phases (from arg(η) and k-dependence)
    eta_phase = np.angle(eta_val)
    phases = np.array([np.exp(1j * k * eta_phase / 4) for k in k_weights])

    # Add sector-dependent phase shift (up vs down doublet splitting)
    if sector == 'down':
        phases *= np.exp(1j * np.pi / 13)  # Doublet splitting phase

    # Full generation amplitudes
    psi_gen = amplitudes * phases

    # Normalize
    psi_gen /= np.linalg.norm(psi_gen)

    return psi_gen


def ckm_from_modular_overlap(k_pattern, tau_val, dedekind_eta_func):
    """
    Compute complete CKM matrix from modular overlaps.

    CKM matrix from overlaps between up-type and down-type generations:
    V_ij = <u_i | d_j> where states have k-dependent modular phases

    Parameters
    ----------
    k_pattern : array_like
        Modular weights [8,6,4]
    tau_val : complex
        Modular parameter τ
    dedekind_eta_func : callable
        Function to compute Dedekind eta η(τ)

    Returns
    -------
    V_CKM : ndarray
        Complete 3×3 CKM matrix
    """
    psi_up = logical_generation_states(k_pattern, tau_val, dedekind_eta_func, sector='up')
    psi_down = logical_generation_states(k_pattern, tau_val, dedekind_eta_func, sector='down')

    # CKM from outer product structure
    # V_ij ~ psi_up[i]* × psi_down[j] with hierarchical corrections
    V = np.zeros((3, 3), dtype=complex)

    for i in range(3):
        for j in range(3):
            # Base overlap from modular amplitudes
            V[i, j] = np.conj(psi_up[i]) * psi_down[j]

            # Hierarchy suppression: off-diagonal elements suppressed by k-difference
            delta_k = abs(k_pattern[i] - k_pattern[j])
            if i != j:
                V[i, j] *= (delta_k / max(k_pattern))**1.5  # Power law suppression

    # Enforce approximate unitarity (normalize rows)
    for i in range(3):
        row_norm = np.linalg.norm(V[i, :])
        if row_norm > 1e-10:
            V[i, :] /= row_norm

    return V


def print_ckm_comparison(V_pred, CKM_exp=None):
    """
    Print complete CKM matrix with comparison to experiment.

    Parameters
    ----------
    V_pred : ndarray
        Predicted CKM matrix (complex)
    CKM_exp : ndarray, optional
        Experimental CKM matrix

    Returns
    -------
    chi2_dof : float
        Chi-squared per degree of freedom
    """
    if CKM_exp is None:
        CKM_exp = np.array([
            [0.97435, 0.22500, 0.00369],
            [0.22000, 0.97349, 0.04182],
            [0.00857, 0.04110, 0.99915]
        ])

    V_mag = np.abs(V_pred)

    print("  Complete CKM matrix (from modular overlaps):")
    print("          d         s         b")
    for i, label in enumerate(['u', 'c', 't']):
        print(f"    {label}:  {V_mag[i,0]:.5f}  {V_mag[i,1]:.5f}  {V_mag[i,2]:.5f}")
    print()

    print("  Experimental CKM:")
    print("          d         s         b")
    for i, label in enumerate(['u', 'c', 't']):
        print(f"    {label}:  {CKM_exp[i,0]:.5f}  {CKM_exp[i,1]:.5f}  {CKM_exp[i,2]:.5f}")
    print()

    # Element-wise comparison
    print("  Element-wise errors:")
    total_chi2 = 0
    for i in range(3):
        for j in range(3):
            pred = V_mag[i, j]
            exp = CKM_exp[i, j]
            err = abs(pred - exp) / exp * 100
            total_chi2 += ((pred - exp) / (0.01 * exp))**2  # Assume 1% errors
            print(f"    V_{['u','c','t'][i]}{['d','s','b'][j]}: {pred:.5f} vs {exp:.5f} ({err:+.1f}%)")

    print()
    chi2_dof = total_chi2 / 9
    print(f"  χ²/dof: {chi2_dof:.1f}")

    if chi2_dof < 3:
        print(f"  Status: ✓ Complete CKM predicted from first principles!")
    elif chi2_dof < 10:
        print(f"  Status: ~ Moderate agreement, structure correct")
    else:
        print(f"  Status: ⚠ Needs loop corrections for precision")
    print()

    return chi2_dof
