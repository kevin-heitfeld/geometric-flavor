"""
QIFT PARAMETER OPTIMIZATION

Systematically scan mixing parameter p_pure to find optimal Gen2/Gen3 errors.

Target: Minimize combined error while keeping both Gen2 and Gen3 under control.

Strategy:
1. Scan p_pure from 0.30 to 0.60 (finer steps around 0.45)
2. For each p, compute masses and errors
3. Find optimal p that minimizes weighted error

Mathematical relationship:
- Higher p_pure → lower entropy S_2 → lower m_2
- S_2 ≈ 5.33 needed (target m_2 = 207)
- S_3 = 8.32 fixed (m_3 = 4096, 17.8% error)

Current best: p=0.45 gives [1, 193, 4096] - 6.9%, 17.8%
"""

import numpy as np
from archive.quantum_ift import QuantumInformationState, compute_qift_masses
import json
from datetime import datetime
import matplotlib.pyplot as plt


def scan_mixing_parameter(n_qubits: int = 12,
                          p_min: float = 0.30,
                          p_max: float = 0.60,
                          n_steps: int = 31):
    """
    Scan mixing parameter p_pure to optimize Gen2 and Gen3 errors

    Args:
        n_qubits: Number of qubits (12 for optimal)
        p_min: Minimum p_pure to test
        p_max: Maximum p_pure to test
        n_steps: Number of steps in scan

    Returns:
        results: Dictionary with scan data
    """

    print("=" * 70)
    print("QIFT MIXING PARAMETER OPTIMIZATION")
    print("=" * 70)
    print()
    print(f"Scanning p_pure from {p_min:.2f} to {p_max:.2f}")
    print(f"Number of steps: {n_steps}")
    print(f"Qubits: n = {n_qubits}")
    print(f"Hilbert space dimension: d = {2**n_qubits}")
    print()
    print("Target masses: [1, 207, 3477]")
    print("Target entropies: S_2 ≈ 5.33, S_3 ≈ 8.32")
    print()
    print("=" * 70)
    print()

    # Arrays to store results
    p_values = np.linspace(p_min, p_max, n_steps)
    entropies_2 = []
    entropies_3 = []
    masses_2 = []
    masses_3 = []
    errors_2 = []
    errors_3 = []
    combined_errors = []

    # Target masses
    target = np.array([1, 207, 3477])

    # Fixed Gen1 state (always separable)
    state_1 = QuantumInformationState('separable', n_qubits=n_qubits)

    # Fixed Gen3 state (always maximally entangled)
    state_3 = QuantumInformationState('maximally_entangled', n_qubits=n_qubits)

    for i, p in enumerate(p_values):
        # Create Gen2 state with current mixing parameter
        # We need to temporarily modify the state generation
        state_2 = create_mixed_state(n_qubits, p)

        # Compute masses
        states = [state_1, state_2, state_3]
        masses, diagnostics = compute_qift_masses(states, use_exponential=True)

        # Extract values
        S_1, S_2, S_3 = diagnostics['entropies']
        m_1, m_2, m_3 = masses

        # Compute errors
        err_2 = abs(m_2 - target[1]) / target[1] * 100
        err_3 = abs(m_3 - target[2]) / target[2] * 100

        # Combined error (weighted)
        # Weight Gen2 more since Gen3 is already good at 17.8%
        combined_err = 0.6 * err_2 + 0.4 * err_3

        # Store results
        entropies_2.append(S_2)
        entropies_3.append(S_3)
        masses_2.append(m_2)
        masses_3.append(m_3)
        errors_2.append(err_2)
        errors_3.append(err_3)
        combined_errors.append(combined_err)

        # Print progress every 5 steps
        if (i + 1) % 5 == 0 or i == 0 or i == n_steps - 1:
            print(f"Step {i+1:2d}/{n_steps}: p={p:.3f} → "
                  f"S_2={S_2:.3f}, m_2={m_2:.1f}, err={err_2:.2f}% | "
                  f"m_3={m_3:.1f}, err={err_3:.2f}% | "
                  f"combined={combined_err:.2f}%")

    print()
    print("=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)
    print()

    # Find optimal parameters
    idx_min_combined = np.argmin(combined_errors)
    idx_min_gen2 = np.argmin(errors_2)
    idx_min_gen3 = np.argmin(errors_3)

    print("Best combined error:")
    print(f"  p_pure = {p_values[idx_min_combined]:.3f}")
    print(f"  S_2 = {entropies_2[idx_min_combined]:.3f}")
    print(f"  Masses: [1, {masses_2[idx_min_combined]:.1f}, {masses_3[idx_min_combined]:.1f}]")
    print(f"  Errors: Gen2={errors_2[idx_min_combined]:.2f}%, Gen3={errors_3[idx_min_combined]:.2f}%")
    print(f"  Combined: {combined_errors[idx_min_combined]:.2f}%")
    print()

    print("Best Gen2 error:")
    print(f"  p_pure = {p_values[idx_min_gen2]:.3f}")
    print(f"  S_2 = {entropies_2[idx_min_gen2]:.3f}")
    print(f"  Masses: [1, {masses_2[idx_min_gen2]:.1f}, {masses_3[idx_min_gen2]:.1f}]")
    print(f"  Errors: Gen2={errors_2[idx_min_gen2]:.2f}%, Gen3={errors_3[idx_min_gen2]:.2f}%")
    print()

    print("Best Gen3 error:")
    print(f"  p_pure = {p_values[idx_min_gen3]:.3f}")
    print(f"  S_3 = {entropies_3[idx_min_gen3]:.3f}")
    print(f"  Masses: [1, {masses_2[idx_min_gen3]:.1f}, {masses_3[idx_min_gen3]:.1f}]")
    print(f"  Errors: Gen2={errors_2[idx_min_gen3]:.2f}%, Gen3={errors_3[idx_min_gen3]:.2f}%")
    print()

    # Prepare results dictionary
    results = {
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'n_qubits': n_qubits,
            'p_min': p_min,
            'p_max': p_max,
            'n_steps': n_steps
        },
        'scan_data': {
            'p_values': p_values.tolist(),
            'entropies_2': entropies_2,
            'entropies_3': entropies_3,
            'masses_2': masses_2,
            'masses_3': masses_3,
            'errors_2': errors_2,
            'errors_3': errors_3,
            'combined_errors': combined_errors
        },
        'optimal': {
            'combined': {
                'p_pure': float(p_values[idx_min_combined]),
                'entropy_2': float(entropies_2[idx_min_combined]),
                'masses': [1.0, float(masses_2[idx_min_combined]), float(masses_3[idx_min_combined])],
                'errors': [0.0, float(errors_2[idx_min_combined]), float(errors_3[idx_min_combined])],
                'combined_error': float(combined_errors[idx_min_combined])
            },
            'gen2_only': {
                'p_pure': float(p_values[idx_min_gen2]),
                'entropy_2': float(entropies_2[idx_min_gen2]),
                'masses': [1.0, float(masses_2[idx_min_gen2]), float(masses_3[idx_min_gen2])],
                'errors': [0.0, float(errors_2[idx_min_gen2]), float(errors_3[idx_min_gen2])]
            },
            'gen3_only': {
                'p_pure': float(p_values[idx_min_gen3]),
                'entropy_3': float(entropies_3[idx_min_gen3]),
                'masses': [1.0, float(masses_2[idx_min_gen3]), float(masses_3[idx_min_gen3])],
                'errors': [0.0, float(errors_2[idx_min_gen3]), float(errors_3[idx_min_gen3])]
            }
        }
    }

    return results, p_values, entropies_2, masses_2, errors_2, errors_3, combined_errors


def create_mixed_state(n_qubits: int, p_pure: float) -> QuantumInformationState:
    """
    Create a mixed quantum state with specified mixing parameter

    Args:
        n_qubits: Number of qubits
        p_pure: Mixing parameter (0 = max mixed, 1 = pure GHZ)

    Returns:
        QuantumInformationState with custom density matrix
    """
    dim = 2 ** n_qubits

    # Create GHZ state
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0 / np.sqrt(2)
    psi[-1] = 1.0 / np.sqrt(2)
    pure_state = np.outer(psi, psi.conj())

    # Mix with maximally mixed state
    max_mixed = np.eye(dim) / dim
    rho = p_pure * pure_state + (1 - p_pure) * max_mixed

    # Create state object and override density matrix
    state = QuantumInformationState('entangled', n_qubits=n_qubits)
    state.density_matrix = rho

    return state


def plot_optimization_results(p_values, entropies_2, masses_2, errors_2, errors_3, combined_errors):
    """
    Create visualization of optimization scan
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('QIFT Mixing Parameter Optimization (n=12 qubits)', fontsize=14, fontweight='bold')

    # Plot 1: Entropy vs p_pure
    ax1 = axes[0, 0]
    ax1.plot(p_values, entropies_2, 'b-', linewidth=2, label='S_2 (Gen2)')
    ax1.axhline(y=5.33, color='r', linestyle='--', label='Target S_2 = 5.33')
    ax1.set_xlabel('Mixing parameter p_pure', fontsize=11)
    ax1.set_ylabel('von Neumann Entropy S_2', fontsize=11)
    ax1.set_title('Entropy vs Mixing Parameter', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Mass vs p_pure
    ax2 = axes[0, 1]
    ax2.plot(p_values, masses_2, 'g-', linewidth=2, label='m_2 (Gen2)')
    ax2.axhline(y=207, color='r', linestyle='--', label='Target m_2 = 207')
    ax2.set_xlabel('Mixing parameter p_pure', fontsize=11)
    ax2.set_ylabel('Mass ratio m_2/m_1', fontsize=11)
    ax2.set_title('Mass Ratio vs Mixing Parameter', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Errors vs p_pure
    ax3 = axes[1, 0]
    ax3.plot(p_values, errors_2, 'b-', linewidth=2, label='Gen2 error')
    ax3.plot(p_values, errors_3, 'orange', linewidth=2, label='Gen3 error')
    ax3.set_xlabel('Mixing parameter p_pure', fontsize=11)
    ax3.set_ylabel('Error (%)', fontsize=11)
    ax3.set_title('Individual Errors vs Mixing Parameter', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Plot 4: Combined error vs p_pure
    ax4 = axes[1, 1]
    ax4.plot(p_values, combined_errors, 'purple', linewidth=2, label='Combined error (0.6*Gen2 + 0.4*Gen3)')
    idx_min = np.argmin(combined_errors)
    ax4.plot(p_values[idx_min], combined_errors[idx_min], 'r*', markersize=15, label=f'Optimal: p={p_values[idx_min]:.3f}')
    ax4.set_xlabel('Mixing parameter p_pure', fontsize=11)
    ax4.set_ylabel('Combined Error (%)', fontsize=11)
    ax4.set_title('Combined Error vs Mixing Parameter', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plt.savefig('qift_optimization_scan.png', dpi=150, bbox_inches='tight')
    print("Plot saved to: qift_optimization_scan.png")
    print()

    return fig


def main():
    """
    Run optimization scan and save results
    """

    # Run coarse scan first (0.30 to 0.60)
    print("Running coarse scan...")
    print()
    results, p_vals, S_2, m_2, err_2, err_3, comb_err = scan_mixing_parameter(
        n_qubits=12,
        p_min=0.30,
        p_max=0.60,
        n_steps=31
    )

    # Find optimal region
    idx_opt = np.argmin(comb_err)
    p_opt = p_vals[idx_opt]

    print("=" * 70)
    print()

    # If we found something better than current p=0.45, do fine scan
    if abs(p_opt - 0.45) > 0.02:  # If optimal is >0.02 away from current
        print("Running fine scan around optimal region...")
        print()
        p_fine_min = max(0.30, p_opt - 0.05)
        p_fine_max = min(0.60, p_opt + 0.05)

        results_fine, p_vals_fine, S_2_fine, m_2_fine, err_2_fine, err_3_fine, comb_err_fine = scan_mixing_parameter(
            n_qubits=12,
            p_min=p_fine_min,
            p_max=p_fine_max,
            n_steps=21
        )

        # Use fine scan results
        results = results_fine
        p_vals = p_vals_fine
        S_2 = S_2_fine
        m_2 = m_2_fine
        err_2 = err_2_fine
        err_3 = err_3_fine
        comb_err = comb_err_fine

        print("=" * 70)
        print()

    # Save results
    with open('qift_optimization_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Results saved to: qift_optimization_results.json")
    print()

    # Create plots
    plot_optimization_results(p_vals, S_2, m_2, err_2, err_3, comb_err)

    # Print final recommendation
    print("=" * 70)
    print("FINAL RECOMMENDATION")
    print("=" * 70)
    print()
    opt = results['optimal']['combined']
    print(f"Update quantum_ift.py line ~108:")
    print(f"  p_pure = {opt['p_pure']:.4f}")
    print()
    print(f"Expected results:")
    print(f"  Masses: {opt['masses']}")
    print(f"  Errors: Gen2={opt['errors'][1]:.2f}%, Gen3={opt['errors'][2]:.2f}%")
    print(f"  Combined error: {opt['combined_error']:.2f}%")
    print()
    print("This is an improvement over current p=0.45:")
    print("  Current: Gen2=6.9%, Gen3=17.8%, combined=11.25%")
    print()

    return results


if __name__ == "__main__":
    results = main()
