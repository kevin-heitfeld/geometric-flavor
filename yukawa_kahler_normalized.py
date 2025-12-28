"""
Yukawa Matrix with Kähler Normalization and Diagonalization Analysis

This script implements the improvements suggested by AI feedback:
1. Kähler metric normalization: Y^phys = K_L^(-1/2) Y K_R^(-1/2)
2. Full diagonalization to extract mass eigenvalues and mixing angles
3. Comparison with experimental charged lepton sector

Week 2, Day 14: Refinement based on ChatGPT/Gemini/Grok/Kimi feedback

Author: Enhanced Yukawa calculation
Date: December 28, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from compute_yukawa_matrix_full import (
    compute_yukawa_matrix, compare_with_phenomenology
)

# =============================================================================
# PART 1: KÄHLER METRIC NORMALIZATION
# =============================================================================

def kahler_metric_diagonal(tau, generation_weights):
    """
    Compute diagonal Kähler metric for matter fields.
    
    In Type IIB/F-theory, the Kähler metric depends on moduli:
    
    K_i = (Imτ)^(-w_i) × (volume factors)
    
    For diagonal approximation:
    K = diag(K_e, K_μ, K_τ)
    
    Parameters
    ----------
    tau : complex
        Complex structure modulus
    generation_weights : list of float
        Modular weights [w_e, w_μ, w_τ]
        
    Returns
    -------
    K : ndarray (3,3)
        Kähler metric (diagonal)
    """
    Im_tau = np.imag(tau)
    
    K = np.zeros((3, 3), dtype=float)
    
    for i, w in enumerate(generation_weights):
        # Kähler metric scaling
        # For w < 0: Stronger suppression (smaller field norm)
        # For w > 0: Enhancement (larger field norm)
        K[i, i] = (Im_tau)**(-w)
    
    return K


def normalize_yukawa_kahler(Y_raw, K_L, K_R=None):
    """
    Apply Kähler normalization to Yukawa matrix.
    
    Physical Yukawa coupling:
    Y^phys = K_L^(-1/2) × Y^raw × K_R^(-1/2)
    
    For leptons (assuming same Kähler for left and right):
    Y^phys = K^(-1/2) × Y^raw × K^(-1/2)
    
    Parameters
    ----------
    Y_raw : ndarray (3,3)
        Raw Yukawa matrix from overlap calculation
    K_L : ndarray (3,3)
        Kähler metric for left-handed fields
    K_R : ndarray (3,3), optional
        Kähler metric for right-handed fields (default: same as K_L)
        
    Returns
    -------
    Y_phys : ndarray (3,3)
        Physical Yukawa matrix
    """
    if K_R is None:
        K_R = K_L
    
    # Compute K^(-1/2) by diagonalizing
    K_L_sqrt_inv = np.diag(1.0 / np.sqrt(np.diag(K_L)))
    K_R_sqrt_inv = np.diag(1.0 / np.sqrt(np.diag(K_R)))
    
    # Apply normalization
    Y_phys = K_L_sqrt_inv @ Y_raw @ K_R_sqrt_inv
    
    return Y_phys


# =============================================================================
# PART 2: DIAGONALIZATION AND MASS EIGENVALUES
# =============================================================================

def diagonalize_yukawa(Y):
    """
    Diagonalize Yukawa matrix to extract mass eigenvalues and mixing.
    
    For a general complex matrix:
    Y = U_L^† Y_diag U_R
    
    For hermitian approximation (small off-diagonal):
    Y ≈ U^† Y_diag U
    
    Parameters
    ----------
    Y : ndarray (3,3)
        Yukawa matrix
        
    Returns
    -------
    eigenvalues : ndarray (3,)
        Yukawa eigenvalues (related to masses: m_i = y_i v / sqrt(2))
    U_L : ndarray (3,3)
        Left mixing matrix
    U_R : ndarray (3,3)
        Right mixing matrix
    """
    # For non-hermitian matrix, use SVD
    U_L, y_diag, U_R_dagger = np.linalg.svd(Y)
    U_R = U_R_dagger.T.conj()
    
    # Sort by magnitude (descending: tau, muon, electron)
    idx = np.argsort(y_diag)[::-1]
    eigenvalues = y_diag[idx]
    U_L = U_L[:, idx]
    U_R = U_R[:, idx]
    
    return eigenvalues, U_L, U_R


def mixing_angles_from_unitary(U):
    """
    Extract mixing angles from unitary matrix.
    
    Standard parameterization:
    U = R_23(θ_23) × R_13(θ_13, δ) × R_12(θ_12)
    
    For small mixing (diagonal-dominated):
    θ_ij ≈ |U_ij| for i ≠ j
    
    Parameters
    ----------
    U : ndarray (3,3)
        Unitary mixing matrix
        
    Returns
    -------
    angles : dict
        Mixing angles in degrees
    """
    # Extract angles from matrix elements
    # For small mixing, off-diagonal elements ≈ sin(θ)
    
    theta_12 = np.arcsin(min(abs(U[0, 1]), 1.0))  # e-μ mixing
    theta_13 = np.arcsin(min(abs(U[0, 2]), 1.0))  # e-τ mixing
    theta_23 = np.arcsin(min(abs(U[1, 2]), 1.0))  # μ-τ mixing
    
    angles = {
        'theta_12': np.degrees(theta_12),
        'theta_13': np.degrees(theta_13),
        'theta_23': np.degrees(theta_23)
    }
    
    return angles


# =============================================================================
# PART 3: COMPREHENSIVE VALIDATION
# =============================================================================

def validate_with_kahler_normalization(tau, use_kahler=True, kahler_strength=1.0):
    """
    Full validation with Kähler normalization.
    
    Parameters
    ----------
    tau : complex
        Complex structure modulus
    use_kahler : bool
        Whether to apply Kähler normalization
    kahler_strength : float
        Scaling parameter for Kähler metric (1.0 = standard)
        
    Returns
    -------
    results : dict
        Complete analysis results
    """
    print("\n")
    print("=" * 70)
    print("YUKAWA MATRIX WITH KÄHLER NORMALIZATION AND DIAGONALIZATION")
    print("=" * 70)
    print("\n")
    
    # Step 1: Compute raw Yukawa matrix
    print("STEP 1: Computing raw Yukawa matrix from wave function overlaps")
    print("-" * 70)
    Y_raw = compute_yukawa_matrix(tau, use_numerical=False)
    
    # Step 2: Apply Kähler normalization
    print("\n")
    print("STEP 2: Applying Kähler metric normalization")
    print("-" * 70)
    
    generation_weights = [-2.0, 0.0, 1.0]  # w_e, w_μ, w_τ
    
    if use_kahler:
        K = kahler_metric_diagonal(tau, generation_weights)
        
        # Scale Kähler metric
        K_scaled = K ** kahler_strength
        
        print(f"Kähler metric (strength={kahler_strength}):")
        for i, gen in enumerate(['Electron', 'Muon', 'Tau']):
            print(f"  K_{gen[0]} = {K_scaled[i,i]:.4f} (w={generation_weights[i]})")
        print()
        
        Y_phys = normalize_yukawa_kahler(Y_raw, K_scaled)
        
        print("Effect of Kähler normalization:")
        for i, gen in enumerate(['Electron', 'Muon', 'Tau']):
            correction = np.abs(Y_phys[i,i]) / np.abs(Y_raw[i,i])
            print(f"  Y_{gen[0]}: {np.abs(Y_raw[i,i]):.4e} → {np.abs(Y_phys[i,i]):.4e} "
                  f"(×{correction:.3f})")
    else:
        Y_phys = Y_raw
        print("Kähler normalization: DISABLED")
    
    # Step 3: Diagonalize to get mass eigenvalues
    print("\n")
    print("STEP 3: Diagonalizing Yukawa matrix")
    print("-" * 70)
    
    eigenvalues, U_L, U_R = diagonalize_yukawa(Y_phys)
    
    print("Mass eigenvalues (Yukawa couplings):")
    exp_values = [2.80e-6, 6.09e-4, 1.04e-2]  # e, μ, τ
    labels = ['Tau', 'Muon', 'Electron']  # Sorted by magnitude
    
    for i in range(3):
        exp_idx = 2 - i  # Reverse order for comparison
        error = abs(eigenvalues[i] - exp_values[exp_idx]) / exp_values[exp_idx] * 100
        print(f"  y_{labels[i][0].lower()} = {eigenvalues[i]:.4e} "
              f"(exp: {exp_values[exp_idx]:.4e}, error: {error:6.2f}%)")
    
    # Step 4: Analyze mixing angles
    print("\n")
    print("STEP 4: Charged lepton mixing angles")
    print("-" * 70)
    
    angles_L = mixing_angles_from_unitary(U_L)
    
    print("Left-handed mixing angles:")
    for key, val in angles_L.items():
        print(f"  {key} = {val:.2f}°")
    
    print()
    print("Physical interpretation:")
    if max(angles_L.values()) < 5.0:
        print("  ✅ Charged lepton mixing is SMALL (<5°)")
        print("  → PMNS angles dominated by neutrino sector (as expected)")
    else:
        print("  ⚠️ Charged lepton mixing is SIGNIFICANT (>5°)")
        print("  → May contribute to PMNS mixing")
    
    # Step 5: Compare with experiment
    print("\n")
    print("STEP 5: Comparison with experimental data")
    print("-" * 70)
    
    hierarchy_ok = compare_with_phenomenology(Y_phys)
    
    # Step 6: Summary statistics
    print("\n")
    print("SUMMARY STATISTICS")
    print("-" * 70)
    
    errors = []
    for i in range(3):
        exp_idx = 2 - i
        error = abs(eigenvalues[i] - exp_values[exp_idx]) / exp_values[exp_idx] * 100
        errors.append(error)
    
    avg_error = np.mean(errors)
    max_error = np.max(errors)
    
    print(f"Average error:  {avg_error:.2f}%")
    print(f"Maximum error:  {max_error:.2f}%")
    print(f"Hierarchy:      {'CORRECT ✓' if hierarchy_ok else 'INCORRECT ✗'}")
    print(f"Max mixing:     {max(angles_L.values()):.2f}°")
    print()
    
    # Return results
    results = {
        'Y_raw': Y_raw,
        'Y_phys': Y_phys,
        'eigenvalues': eigenvalues,
        'U_L': U_L,
        'U_R': U_R,
        'mixing_angles': angles_L,
        'errors': errors,
        'avg_error': avg_error,
        'hierarchy_ok': hierarchy_ok
    }
    
    return results


# =============================================================================
# PART 4: VISUALIZATION
# =============================================================================

def visualize_kahler_effect(tau):
    """
    Visualize the effect of Kähler normalization on Yukawa eigenvalues.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Test different Kähler strengths
    strengths = np.linspace(0.5, 1.5, 11)
    
    eigenvalues_all = []
    errors_all = []
    
    for strength in strengths:
        results = validate_with_kahler_normalization(tau, use_kahler=True, 
                                                      kahler_strength=strength)
        eigenvalues_all.append(results['eigenvalues'])
        errors_all.append(results['avg_error'])
    
    eigenvalues_all = np.array(eigenvalues_all)
    
    # Plot 1: Eigenvalue evolution
    ax1 = axes[0]
    
    exp_values = [1.04e-2, 6.09e-4, 2.80e-6]  # τ, μ, e (sorted)
    labels = ['Tau', 'Muon', 'Electron']
    colors = ['tab:orange', 'tab:blue', 'tab:green']
    
    for i in range(3):
        ax1.plot(strengths, eigenvalues_all[:, i], 'o-', label=labels[i], 
                color=colors[i], linewidth=2, markersize=6)
        ax1.axhline(exp_values[i], color=colors[i], linestyle='--', alpha=0.5,
                   label=f'{labels[i]} (exp)')
    
    ax1.set_xlabel('Kähler Strength Parameter', fontsize=12)
    ax1.set_ylabel('Yukawa Coupling', fontsize=12)
    ax1.set_yscale('log')
    ax1.set_title('Effect of Kähler Normalization on Yukawas', fontsize=14, 
                  fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Average error vs Kähler strength
    ax2 = axes[1]
    
    ax2.plot(strengths, errors_all, 'o-', color='purple', linewidth=2, 
            markersize=8)
    ax2.axhline(100, color='red', linestyle='--', alpha=0.5, 
               label='Factor 2 error')
    ax2.axhline(50, color='orange', linestyle='--', alpha=0.5,
               label='50% error')
    
    # Mark minimum
    min_idx = np.argmin(errors_all)
    ax2.plot(strengths[min_idx], errors_all[min_idx], 'r*', markersize=20,
            label=f'Minimum at {strengths[min_idx]:.2f}')
    
    ax2.set_xlabel('Kähler Strength Parameter', fontsize=12)
    ax2.set_ylabel('Average Error (%)', fontsize=12)
    ax2.set_title('Optimal Kähler Normalization', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('yukawa_kahler_optimization.png', dpi=300, bbox_inches='tight')
    print("📊 Visualization saved: yukawa_kahler_optimization.png")
    print()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n")
    print("=" * 70)
    print("YUKAWA MATRIX REFINEMENT: KÄHLER NORMALIZATION + DIAGONALIZATION")
    print("Addressing feedback from ChatGPT, Gemini, Grok, and Kimi")
    print("=" * 70)
    print("\n")
    
    tau = 2.69j
    
    # Part 1: Without Kähler normalization (baseline)
    print("\n" + "=" * 70)
    print("BASELINE: Raw Yukawa matrix (no Kähler correction)")
    print("=" * 70)
    results_raw = validate_with_kahler_normalization(tau, use_kahler=False)
    
    # Part 2: With Kähler normalization (standard)
    print("\n" + "=" * 70)
    print("REFINED: With Kähler normalization (standard strength)")
    print("=" * 70)
    results_kahler = validate_with_kahler_normalization(tau, use_kahler=True, 
                                                         kahler_strength=1.0)
    
    # Part 3: Optimize Kähler strength
    print("\n" + "=" * 70)
    print("OPTIMIZATION: Finding optimal Kähler strength")
    print("=" * 70)
    visualize_kahler_effect(tau)
    
    # Part 4: Final summary
    print("=" * 70)
    print("FINAL ASSESSMENT")
    print("=" * 70)
    print()
    
    improvement = results_raw['avg_error'] - results_kahler['avg_error']
    
    print(f"Without Kähler normalization:")
    print(f"  Average error: {results_raw['avg_error']:.2f}%")
    print(f"  Max mixing:    {max(results_raw['mixing_angles'].values()):.2f}°")
    print()
    
    print(f"With Kähler normalization:")
    print(f"  Average error: {results_kahler['avg_error']:.2f}%")
    print(f"  Max mixing:    {max(results_kahler['mixing_angles'].values()):.2f}°")
    print(f"  Improvement:   {improvement:.2f}% (error reduction)")
    print()
    
    if results_kahler['avg_error'] < 100:
        print("✅ SUCCESS: Average error within factor 2 (acceptable for LO + Kähler)")
    elif results_kahler['avg_error'] < 200:
        print("⚠️ PARTIAL: Still factors of 2-3 off, but Kähler helps")
    else:
        print("❌ ISSUE: Errors still large, need numerical overlaps or different approach")
    
    print()
    
    if max(results_kahler['mixing_angles'].values()) < 5.0:
        print("✅ MIXING: Charged lepton mixing small → PMNS from neutrinos ✓")
    
    print()
    print("=" * 70)
    print("Kähler normalization analysis complete!")
    print("=" * 70)
    print()
