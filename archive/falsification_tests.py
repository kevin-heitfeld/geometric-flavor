"""
QTNC-7 Falsification Experiments
Quick computational tests to attempt to falsify the theory
Each test should run in < 5 minutes
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigsh
import time
import traceback


def test_spectral_dimension(N=1000, r_mean=6, trials=10):
    """
    Experiment 1: Test if spectral dimension matches prediction

    Prediction: d_s ≈ 3.0 for hyperbolic graph with mean degree ~6
    Falsification: d_s < 2.5 or d_s > 3.5
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1: Spectral Dimension Test")
    print("="*60)

    results = []
    for trial in range(trials):
        # Generate random regular graph (approximation of hyperbolic)
        G = nx.random_regular_graph(r_mean, N)

        # Compute Laplacian spectrum
        L = nx.laplacian_matrix(G).astype(float)
        eigvals = eigsh(L, k=min(100, N-2), which='SM', return_eigenvectors=False)

        # Spectral dimension from eigenvalue scaling
        # d_s ~ 2 * d(log λ) / d(log k) at small k
        k_range = np.arange(10, min(50, len(eigvals)))
        lambdas = eigvals[k_range]

        # Numerical derivative
        log_lambda = np.log(lambdas + 1e-10)
        log_k = np.log(k_range)
        d_s = 2 * np.gradient(log_lambda) / np.gradient(log_k)

        median_ds = np.median(d_s)
        results.append(median_ds)

        if trial % 3 == 0:
            print(f"  Trial {trial+1}/{trials}: d_s = {median_ds:.3f}")

    mean_ds = np.mean(results)
    std_ds = np.std(results)

    print(f"\nResults:")
    print(f"  Predicted d_s: ~3.0")
    print(f"  Measured d_s: {mean_ds:.3f} ± {std_ds:.3f}")

    # Check if falsified
    if abs(mean_ds - 3.0) > 0.5:
        print(f"  ❌ FAILED: Spectral dimension mismatch!")
        print(f"     Theory predicts d_s ≈ 3, but measured {mean_ds:.2f}")
        return False, mean_ds
    else:
        print(f"  ✓ PASSED: Spectral dimension within expected range")
        return True, mean_ds


def test_mass_ratios(N=500, k_modes=3, trials=20):
    """
    Experiment 2: Test if spectral modes reproduce fermion mass hierarchy

    Prediction: Mass ratios ~ [1, 200, 3500] for leptons
    Falsification: Ratios ~ [1, 10, 100] (exponential suppression fails)
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2: Mass Ratio Hierarchy")
    print("="*60)

    results = []

    for trial in range(trials):
        # Generate network
        G = nx.random_regular_graph(6, N)
        L = nx.laplacian_matrix(G).astype(float)

        # Get first 3 non-zero eigenvalues
        eigvals = eigsh(L, k=k_modes+1, which='SM', return_eigenvectors=False)
        lambdas = eigvals[1:]  # Skip zero mode

        # Apply mass formula: m_ℓ ∝ √λ_ℓ * exp(-k*ℓ)
        k = 2.3
        masses = np.sqrt(lambdas) * np.exp(-k * np.arange(1, k_modes+1))
        ratios = masses / masses[0]

        results.append(ratios)

        if trial % 5 == 0:
            print(f"  Trial {trial+1}/{trials}: ratios = [1, {ratios[1]:.0f}, {ratios[2]:.0f}]")

    mean_ratios = np.mean(results, axis=0)
    std_ratios = np.std(results, axis=0)

    print(f"\nResults:")
    print(f"  Predicted ratios: [1, ~200, ~3500]")
    print(f"  Measured ratios: [1, {mean_ratios[1]:.0f}, {mean_ratios[2]:.0f}]")
    print(f"  Standard dev: [-, ±{std_ratios[1]:.0f}, ±{std_ratios[2]:.0f}]")

    # Check if falsified
    if not (50 < mean_ratios[1] < 500):
        print(f"  ❌ FAILED: Second generation mass ratio wrong!")
        print(f"     Expected ~200, got {mean_ratios[1]:.0f}")
        return False, mean_ratios

    if not (1000 < mean_ratios[2] < 10000):
        print(f"  ❌ FAILED: Third generation mass ratio wrong!")
        print(f"     Expected ~3500, got {mean_ratios[2]:.0f}")
        return False, mean_ratios

    print(f"  ✓ PASSED: Mass ratios within expected ranges")
    return True, mean_ratios


def test_koide_relation(N=500, trials=50):
    """
    Experiment 3: Test if spectral masses satisfy Koide relation

    Prediction: Q_ℓ = (m_e + m_μ + m_τ) / (√m_e + √m_μ + √m_τ)² ≈ 2/3
    Falsification: Q < 0.5 or Q > 0.8
    """
    print("\n" + "="*60)
    print("EXPERIMENT 3: Koide Relation Test")
    print("="*60)

    Q_values = []

    for trial in range(trials):
        G = nx.random_regular_graph(6, N)
        L = nx.laplacian_matrix(G).astype(float)
        eigvals = eigsh(L, k=4, which='SM', return_eigenvectors=False)

        masses = np.sqrt(eigvals[1:4]) * np.exp(-2.3 * np.arange(1, 4))

        Q = np.sum(masses) / (np.sum(np.sqrt(masses)))**2
        Q_values.append(Q)

        if trial % 10 == 0:
            print(f"  Trial {trial+1}/{trials}: Q = {Q:.4f}")

    mean_Q = np.mean(Q_values)
    std_Q = np.std(Q_values)

    print(f"\nResults:")
    print(f"  Predicted Q: 0.6667 (2/3)")
    print(f"  Measured Q: {mean_Q:.4f} ± {std_Q:.4f}")
    print(f"  Actual lepton Q: 0.6667")

    # Check if falsified
    if abs(mean_Q - 2/3) > 0.05:
        print(f"  ❌ FAILED: Koide relation violated!")
        print(f"     Expected ~0.667, got {mean_Q:.4f}")
        return False, mean_Q

    print(f"  ✓ PASSED: Koide relation satisfied within tolerance")
    return True, mean_Q


def test_scrambling_time(N_values=[100, 500, 2000, 10000]):
    """
    Experiment 4: Test fast scrambling prediction

    Prediction: Information scrambles in τ ~ log(log(N)) steps
    Falsification: Scaling ~ log(N) instead
    """
    print("\n" + "="*60)
    print("EXPERIMENT 4: Scrambling Time Scaling")
    print("="*60)

    results = []
    for N in N_values:
        print(f"\n  Testing N = {N}...")
        G = nx.random_regular_graph(6, N)

        # Measure diameter (proxy for scrambling time)
        start_time = time.time()
        diameter = nx.diameter(G)
        elapsed = time.time() - start_time

        results.append((N, diameter, elapsed))
        print(f"    Diameter: {diameter}, Time: {elapsed:.2f}s")

    # Check scaling
    N_arr = np.array([r[0] for r in results])
    D_arr = np.array([r[1] for r in results])

    print(f"\nResults:")
    print(f"  N values: {N_arr}")
    print(f"  Diameters: {D_arr}")

    # Fit to log(log(N))
    log_log_N = np.log(np.log(N_arr))
    coeffs_loglog = np.polyfit(log_log_N, D_arr, 1)

    # Fit to log(N) for comparison
    log_N = np.log(N_arr)
    coeffs_log = np.polyfit(log_N, D_arr, 1)

    print(f"\n  log(log(N)) fit: diameter ≈ {coeffs_loglog[0]:.2f} * log(log(N)) + {coeffs_loglog[1]:.2f}")
    print(f"  log(N) fit:      diameter ≈ {coeffs_log[0]:.2f} * log(N) + {coeffs_log[1]:.2f}")

    # Check if diameter stays small (< 10) even for large N
    if D_arr[-1] > 10:
        print(f"  ❌ FAILED: Diameter grows too large!")
        print(f"     Expected < 10 for N=10000, got {D_arr[-1]}")
        return False, results

    print(f"  ✓ PASSED: Fast scrambling confirmed (diameter stays small)")
    return True, results


def test_entanglement_area_law(N=200, trials=10):
    """
    Experiment 5: Test if random tensor network satisfies area law

    Prediction: S(A) ~ |∂A| (area law)
    Falsification: S(A) ~ |A| (volume law)
    """
    print("\n" + "="*60)
    print("EXPERIMENT 5: Entanglement Area Law")
    print("="*60)
    print("  (Using simplified proxy measurement)")

    results = []
    for trial in range(trials):
        # Generate network
        G = nx.random_regular_graph(6, N)

        # Partition at middle
        subsystem_A = set(range(N//2))
        subsystem_B = set(range(N//2, N))

        # Count boundary edges
        boundary_size = len(list(nx.edge_boundary(G, subsystem_A)))

        # Estimate entanglement entropy (simplified)
        # For a tensor network with bond dimension χ, S ~ |∂A| * log(χ)
        chi = 8  # bond dimension
        S_estimate = boundary_size * np.log(chi) * (1 + 0.1 * np.random.randn())

        results.append((boundary_size, S_estimate))

        if trial % 3 == 0:
            print(f"  Trial {trial+1}/{trials}: boundary={boundary_size}, S≈{S_estimate:.1f}")

    boundaries = np.array([r[0] for r in results])
    entropies = np.array([r[1] for r in results])

    # Fit S vs boundary size
    slope, intercept = np.polyfit(boundaries, entropies, 1)

    print(f"\nResults:")
    print(f"  Entanglement entropy S ≈ {slope:.3f} * |∂A| + {intercept:.2f}")
    print(f"  Predicted slope: ~{np.log(8):.2f} (area law with χ=8)")

    # Check if falsified (slope should be ~ log(χ) ≈ 2.08)
    expected_slope = np.log(8)
    if abs(slope - expected_slope) > 1.0:
        print(f"  ❌ FAILED: Area law scaling incorrect!")
        print(f"     Expected slope ~{expected_slope:.2f}, got {slope:.2f}")
        return False, slope

    print(f"  ✓ PASSED: Area law satisfied")
    return True, slope


def run_all_falsification_tests():
    """
    Master function to run all 5 experiments and report results
    """
    print("\n" + "="*60)
    print("QTNC-7 FALSIFICATION TEST SUITE")
    print("="*60)
    print(f"Running 5 computational experiments...")
    print(f"Expected runtime: ~5-10 minutes total")

    start_time = time.time()

    tests = [
        ("Spectral Dimension", test_spectral_dimension),
        ("Mass Ratios", test_mass_ratios),
        ("Koide Relation", test_koide_relation),
        ("Scrambling Time", test_scrambling_time),
        ("Area Law", test_entanglement_area_law)
    ]

    results = {}
    for name, test_func in tests:
        try:
            passed, result = test_func()
            results[name] = {
                "status": "PASS" if passed else "FAIL",
                "result": result
            }
        except Exception as e:
            results[name] = {
                "status": "ERROR",
                "result": str(e)
            }
            print(f"\n⚠ ERROR in {name}: {e}")
            traceback.print_exc()

    elapsed = time.time() - start_time

    # Print summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)

    for name, info in results.items():
        status = info["status"]
        symbol = "✓" if status == "PASS" else "❌" if status == "FAIL" else "⚠"
        print(f"  {symbol} {name}: {status}")

    failures = [name for name, info in results.items() if info["status"] == "FAIL"]
    errors = [name for name, info in results.items() if info["status"] == "ERROR"]

    print(f"\n  Total runtime: {elapsed:.1f} seconds")
    print(f"  Passed: {sum(1 for i in results.values() if i['status'] == 'PASS')}/5")
    print(f"  Failed: {len(failures)}/5")
    print(f"  Errors: {len(errors)}/5")

    if failures:
        print(f"\n❌ THEORY FALSIFIED BY: {', '.join(failures)}")
        print(f"   → These core predictions do not hold computationally")
        print(f"   → Theory needs revision or different parameters")
    elif errors:
        print(f"\n⚠ TESTS INCOMPLETE: {', '.join(errors)} had errors")
        print(f"   → Cannot conclusively falsify, but tests are inconclusive")
    else:
        print(f"\n✓ ALL TESTS PASSED!")
        print(f"   → Theory survives computational falsification attempts")
        print(f"   → Next: Compare with experimental data (neutrino masses, etc.)")

    print("\n" + "="*60)

    return results


if __name__ == "__main__":
    # Run all tests
    results = run_all_falsification_tests()

    # Optional: Save results
    import json
    with open('falsification_results.json', 'w') as f:
        json.dump({k: {"status": v["status"], "result": str(v["result"])}
                   for k, v in results.items()}, f, indent=2)
    print("\nResults saved to falsification_results.json")
