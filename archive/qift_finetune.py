"""
QIFT Fine-Tuning Script

Quick manual testing of different p_pure values to optimize Gen2 error.

From optimization scan we know:
- p=0.440 → m_2=209.1 (1.0% error)
- p=0.490 → m_2=138.9 (32.9% error)

Target: m_2 = 207 exactly
Need: S_2 = ln(207) ≈ 5.332

Try values around 0.440-0.450
"""

import numpy as np
from archive.quantum_ift import QuantumInformationState, compute_qift_masses


def test_p_value(p_pure: float, n_qubits: int = 12):
    """
    Quick test of a single p_pure value
    """
    # Create states
    state_1 = QuantumInformationState('separable', n_qubits=n_qubits)
    state_3 = QuantumInformationState('maximally_entangled', n_qubits=n_qubits)

    # Create Gen2 with custom p_pure
    dim = 2 ** n_qubits
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0 / np.sqrt(2)
    psi[-1] = 1.0 / np.sqrt(2)
    pure_state = np.outer(psi, psi.conj())
    max_mixed = np.eye(dim) / dim
    rho = p_pure * pure_state + (1 - p_pure) * max_mixed

    state_2 = QuantumInformationState('entangled', n_qubits=n_qubits)
    state_2.density_matrix = rho

    # Compute masses
    states = [state_1, state_2, state_3]
    masses, diagnostics = compute_qift_masses(states, use_exponential=True)

    # Extract values
    S_1, S_2, S_3 = diagnostics['entropies']
    m_1, m_2, m_3 = masses

    # Compute errors
    err_2 = abs(m_2 - 207) / 207 * 100
    err_3 = abs(m_3 - 3477) / 3477 * 100

    # Combined error (weighted)
    combined = 0.6 * err_2 + 0.4 * err_3

    return {
        'p_pure': p_pure,
        'S_2': S_2,
        'S_3': S_3,
        'm_2': m_2,
        'm_3': m_3,
        'err_2': err_2,
        'err_3': err_3,
        'combined': combined
    }


def main():
    print("=" * 70)
    print("QIFT FINE-TUNING - Manual p_pure Testing")
    print("=" * 70)
    print()
    print("Target: m_2 = 207 (S_2 ≈ 5.332), m_3 = 3477 (S_3 ≈ 8.154)")
    print()
    print("Try different p_pure values below:")
    print()
    print("=" * 70)
    print()

    # Test values to try (edit these!)
    test_values = [
        0.440,  # Current best from scan
        0.445,  # Slightly higher (less entropy)
        0.442,  # Between
        0.448,  # Higher still
        0.450,  # Round number
    ]

    results = []

    print(f"{'p_pure':>7s} │ {'S_2':>7s} │ {'m_2':>7s} │ {'err_2':>8s} │ {'m_3':>7s} │ {'err_3':>8s} │ {'combined':>9s}")
    print("─" * 70)

    for p in test_values:
        result = test_p_value(p)
        results.append(result)

        print(f"{result['p_pure']:7.3f} │ "
              f"{result['S_2']:7.3f} │ "
              f"{result['m_2']:7.1f} │ "
              f"{result['err_2']:7.2f}% │ "
              f"{result['m_3']:7.1f} │ "
              f"{result['err_3']:7.2f}% │ "
              f"{result['combined']:8.2f}%")

    print()
    print("=" * 70)
    print()

    # Find best
    best_idx = min(range(len(results)), key=lambda i: results[i]['combined'])
    best = results[best_idx]

    print("BEST RESULT:")
    print(f"  p_pure = {best['p_pure']:.4f}")
    print(f"  S_2 = {best['S_2']:.4f} (target 5.332)")
    print(f"  Masses: [1, {best['m_2']:.1f}, {best['m_3']:.1f}]")
    print(f"  Errors: Gen2={best['err_2']:.2f}%, Gen3={best['err_3']:.2f}%")
    print(f"  Combined: {best['combined']:.2f}%")
    print()
    print(f"Update quantum_ift.py line ~108:")
    print(f"  p_pure = {best['p_pure']:.3f}")
    print()


if __name__ == "__main__":
    main()
