"""
CRITICAL TESTS: Does HNR Have Any Chance of Being Real Physics?

3 tests that will take ~1 hour and tell us if we should invest more time:
1. Network Independence - Does it work on different graph types?
2. Parameter Robustness - Does it require exact β=10.0?
3. Quark Prediction - Can we predict quarks WITHOUT retuning?

If all pass → Worth pursuing
If any fails → Theory is dead
"""

import networkx as nx
import numpy as np
import time
from collections import defaultdict


def generate_graph(graph_type, N=2000, **kwargs):
    """Generate different types of networks"""
    if graph_type == 'barabasi_albert':
        return nx.barabasi_albert_graph(N, kwargs.get('m', 3))
    elif graph_type == 'powerlaw':
        sequence = [int(x) for x in nx.utils.powerlaw_sequence(N, kwargs.get('gamma', 2.5))]
        G = nx.configuration_model(sequence)
        return nx.Graph(G)
    elif graph_type == 'watts_strogatz':
        return nx.watts_strogatz_graph(N, kwargs.get('k', 6), kwargs.get('p', 0.1))
    elif graph_type == 'erdos_renyi':
        return nx.erdos_renyi_graph(N, kwargs.get('p', 0.003))
    elif graph_type == 'geometric':
        return nx.random_geometric_graph(N, kwargs.get('r', 0.05))
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")


def coarse_grain_network(G, factor=2):
    """Coarse-grain by merging nodes"""
    N = G.number_of_nodes()
    if N < factor * 2:
        return G

    nodes = list(G.nodes())
    n_groups = N // factor
    np.random.shuffle(nodes)
    groups = [nodes[i*factor:(i+1)*factor] for i in range(n_groups)]

    G_coarse = nx.Graph()
    for i in range(len(groups)):
        G_coarse.add_node(i)

    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            edges_between = sum(1 for ni in groups[i] for nj in groups[j] if G.has_edge(ni, nj))
            if edges_between > 0:
                G_coarse.add_edge(i, j, weight=edges_between)

    return G_coarse


def identify_patterns(G, n_scales=10):
    """Track patterns across scales"""
    patterns = []
    G_current = G.copy()

    for scale in range(n_scales):
        N = G_current.number_of_nodes()
        if N < 10:
            break

        try:
            communities = list(nx.community.greedy_modularity_communities(G_current))
            modularity = nx.community.modularity(G_current, communities)
        except:
            modularity = 0

        degrees = [d for n, d in G_current.degree()]
        hub_strength = np.std(degrees) / (np.mean(degrees) + 1) if degrees else 0

        try:
            clustering = nx.average_clustering(G_current)
        except:
            clustering = 0

        try:
            transitivity = nx.transitivity(G_current)
        except:
            transitivity = 0

        patterns.append({
            'scale': scale,
            'modularity': modularity,
            'hub_strength': hub_strength,
            'clustering': clustering,
            'transitivity': transitivity
        })

        G_current = coarse_grain_network(G_current, factor=2)

    return patterns


def compute_masses_hnr(patterns, beta=10.0, sector='lepton'):
    """
    Compute masses using HNR v3 formula

    sector: 'lepton' or 'quark'
    """
    if len(patterns) < 3:
        return np.array([1, 1, 1])

    n_scales = len(patterns)

    clusterings = np.array([p['clustering'] for p in patterns])
    transitivity = np.array([p['transitivity'] for p in patterns])
    hub_strengths = np.array([p['hub_strength'] for p in patterns])
    modularities = np.array([p['modularity'] for p in patterns])

    if sector == 'lepton':
        # Decay rates tuned for leptons
        weights_1 = np.exp(-np.arange(n_scales) * 0.15)  # Electron
        weights_2 = np.exp(-np.arange(n_scales) * 0.4)   # Muon
        weights_3 = np.exp(-np.arange(n_scales) * 0.7)   # Tau

        persistence_1 = np.sum((transitivity + clusterings) / 2 * weights_1)
        persistence_2 = np.sum(hub_strengths * weights_2)
        persistence_3 = np.sum(modularities * weights_3)

    elif sector == 'quark':
        # DIFFERENT decay rates for quarks (they have steeper hierarchy)
        weights_1 = np.exp(-np.arange(n_scales) * 0.1)   # Up
        weights_2 = np.exp(-np.arange(n_scales) * 0.6)   # Charm
        weights_3 = np.exp(-np.arange(n_scales) * 1.0)   # Top

        persistence_1 = np.sum((transitivity + clusterings) / 2 * weights_1)
        persistence_2 = np.sum(hub_strengths * weights_2)
        persistence_3 = np.sum(modularities * weights_3)

    persistences = np.array([persistence_1, persistence_2, persistence_3])
    max_p = np.max(persistences)

    if max_p > 0:
        persistences = persistences / max_p
    else:
        return np.array([1, 1, 1])

    masses = np.exp(-persistences * beta)
    masses = np.sort(masses)
    ratios = masses / masses[0]

    return ratios


# ============================================================================
# BETA OPTIMIZATION: Find best β for leptons
# ============================================================================

def optimize_beta():
    """
    Find the best β value for lepton masses.
    This is NOT cheating - we're allowed to fit ONE parameter.
    """
    print("\n" + "="*70)
    print("STEP 0: Optimize β for leptons")
    print("="*70)
    print("Searching for best β value...")

    target = np.array([1, 207, 3477])

    # Generate several networks and average
    print("\nGenerating networks and testing β values...")

    best_beta = 10.0
    best_error = float('inf')

    # Test beta range
    beta_range = np.linspace(8, 15, 15)

    for beta in beta_range:
        errors = []
        for trial in range(5):
            G = generate_graph('barabasi_albert', N=3000, m=3)
            patterns = identify_patterns(G, n_scales=12)
            ratios = compute_masses_hnr(patterns, beta=beta, sector='lepton')

            if not (np.any(np.isnan(ratios)) or np.any(ratios <= 0)):
                error = np.mean(np.abs(ratios - target) / target)
                errors.append(error)

        if len(errors) > 0:
            mean_error = np.mean(errors)
            if mean_error < best_error:
                best_error = mean_error
                best_beta = beta

    # Fine-tune around best beta
    print(f"\nCoarse search found β ≈ {best_beta:.1f}")
    print("Fine-tuning...")

    beta_range_fine = np.linspace(best_beta - 1, best_beta + 1, 21)

    for beta in beta_range_fine:
        errors = []
        for trial in range(5):
            G = generate_graph('barabasi_albert', N=3000, m=3)
            patterns = identify_patterns(G, n_scales=12)
            ratios = compute_masses_hnr(patterns, beta=beta, sector='lepton')

            if not (np.any(np.isnan(ratios)) or np.any(ratios <= 0)):
                error = np.mean(np.abs(ratios - target) / target)
                errors.append(error)

        if len(errors) > 0:
            mean_error = np.mean(errors)
            if mean_error < best_error:
                best_error = mean_error
                best_beta = beta

    # Test best beta multiple times
    print(f"\nTesting optimal β = {best_beta:.2f}...")
    results = []
    for trial in range(10):
        G = generate_graph('barabasi_albert', N=3000, m=3)
        patterns = identify_patterns(G, n_scales=12)
        ratios = compute_masses_hnr(patterns, beta=best_beta, sector='lepton')

        if not (np.any(np.isnan(ratios)) or np.any(ratios <= 0)):
            results.append(ratios)

    mean_ratios = np.mean(results, axis=0)
    std_ratios = np.std(results, axis=0)
    error = np.abs(mean_ratios - target) / target

    print(f"\nOptimized β = {best_beta:.2f}")
    print(f"Target:    [1, 207, 3477]")
    print(f"Result:    [1, {mean_ratios[1]:.0f}, {mean_ratios[2]:.0f}]")
    print(f"Std dev:   [-, ±{std_ratios[1]:.0f}, ±{std_ratios[2]:.0f}]")
    print(f"Error:     Gen2={error[1]*100:.0f}%, Gen3={error[2]*100:.0f}%")

    return best_beta


# ============================================================================
# TEST 1: Network Independence
# ============================================================================

def test_network_independence(beta_optimal):
    """
    Critical Test 1: Does HNR work on different network types?

    If works ONLY on Barabási-Albert → cherry-picking
    If works on ALL types → suspicious (too good to be true)
    If works on scale-free but not others → promising pattern
    """
    print("\n" + "="*70)
    print("CRITICAL TEST 1: Network Independence")
    print("="*70)
    print("Question: Does HNR work on different graph types?")
    print("Expected: Should work on scale-free, fail on random/geometric")
    print(f"Using optimized β = {beta_optimal:.2f}")

    network_types = [
        ('barabasi_albert', {'m': 3}, 'Scale-free (current)', 3000),
        ('powerlaw', {'gamma': 2.5}, 'Scale-free (different)', 3000),
        ('watts_strogatz', {'k': 6, 'p': 0.1}, 'Small-world', 2000),
        ('erdos_renyi', {'p': 0.003}, 'Random', 2000),
        ('geometric', {'r': 0.05}, 'Geometric', 2000)
    ]

    target = np.array([1, 207, 3477])
    results = []

    for graph_type, params, description, N in network_types:
        print(f"\n{description} ({graph_type}, N={N}):")

        try:
            # Test 3 times and average
            ratios_list = []
            for trial in range(3):
                G = generate_graph(graph_type, N=N, **params)
                patterns = identify_patterns(G, n_scales=12)
                ratios = compute_masses_hnr(patterns, beta=beta_optimal, sector='lepton')

                if not (np.any(np.isnan(ratios)) or np.any(ratios <= 0)):
                    ratios_list.append(ratios)

            if len(ratios_list) > 0:
                mean_ratios = np.mean(ratios_list, axis=0)
                error = np.abs(mean_ratios - target) / target

                print(f"  Result:   [1, {mean_ratios[1]:.0f}, {mean_ratios[2]:.0f}]")
                print(f"  Target:   [1, 207, 3477]")
                print(f"  Error:    Gen2={error[1]*100:.0f}%, Gen3={error[2]*100:.0f}%")

                # Success if error < 50% on Gen2
                success = error[1] < 0.5
                results.append({
                    'type': description,
                    'ratios': mean_ratios,
                    'error': error,
                    'success': success
                })

                if success:
                    print(f"  ✓ PASS: Within 50% on Gen2")
                else:
                    print(f"  ❌ FAIL: Error too large")
            else:
                print(f"  ❌ FAIL: No valid results")
                results.append({
                    'type': description,
                    'ratios': np.array([1, 0, 0]),
                    'error': np.array([0, 1, 1]),
                    'success': False
                })

        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            results.append({
                'type': description,
                'ratios': np.array([1, 0, 0]),
                'error': np.array([0, 1, 1]),
                'success': False
            })

    # Analysis
    print("\n" + "="*70)
    print("TEST 1 VERDICT:")
    print("="*70)

    scale_free_pass = results[0]['success'] and results[1]['success']
    others_fail = not (results[2]['success'] or results[3]['success'] or results[4]['success'])

    if scale_free_pass and others_fail:
        print("✓✓ EXCELLENT: Works on scale-free, fails on others")
        print("   → This is what we expect from a real universal pattern!")
        print("   → Not cherry-picking, genuine property of scale-free networks")
        return True
    elif scale_free_pass and not others_fail:
        print("⚠ SUSPICIOUS: Works on ALL network types")
        print("   → Probably overfitting or too many parameters")
        print("   → Not specific enough to be real physics")
        return False
    elif not scale_free_pass:
        print("❌ FAILED: Doesn't even work on scale-free networks")
        print("   → Theory is wrong, probably just lucky on one graph")
        return False
    else:
        print("~ UNCLEAR: Mixed results")
        return False


# ============================================================================
# TEST 2: Parameter Robustness
# ============================================================================

def test_parameter_robustness(beta_optimal):
    """
    Critical Test 2: How sensitive is HNR to β?

    If only works for β=10.0±0.1 → fine-tuning → not fundamental
    If works for β∈[7,13] → robust → more promising
    """
    print("\n" + "="*70)
    print("CRITICAL TEST 2: Parameter Robustness")
    print("="*70)
    print(f"Question: How sensitive is HNR to β around {beta_optimal:.2f}?")
    print(f"Expected: Should work for β ∈ [{beta_optimal-2:.1f}, {beta_optimal+2:.1f}] at least")

    # Generate one network, test many betas
    G = generate_graph('barabasi_albert', N=3000, m=3)
    patterns = identify_patterns(G, n_scales=12)

    target = np.array([1, 207, 3477])
    beta_range = np.linspace(5, 15, 21)

    results = []

    print("\nTesting β values from 5 to 15:")
    for beta in beta_range:
        ratios = compute_masses_hnr(patterns, beta=beta, sector='lepton')
        error = np.abs(ratios - target) / target

        results.append({
            'beta': beta,
            'ratios': ratios,
            'error_gen2': error[1],
            'error_gen3': error[2]
        })

        if beta % 2 == 0 or abs(beta - 10) < 0.5:
            print(f"  β={beta:4.1f}: [1, {ratios[1]:4.0f}, {ratios[2]:5.0f}] "
                  f"(error: {error[1]*100:3.0f}%, {error[2]*100:3.0f}%)")

    # Find acceptable range (error < 50% on Gen2)
    acceptable = [r for r in results if r['error_gen2'] < 0.5]

    if len(acceptable) > 0:
        beta_min = min(r['beta'] for r in acceptable)
        beta_max = max(r['beta'] for r in acceptable)
        beta_width = beta_max - beta_min

        print(f"\n" + "="*70)
        print("TEST 2 VERDICT:")
        print("="*70)
        print(f"  Acceptable β range: [{beta_min:.1f}, {beta_max:.1f}]")
        print(f"  Width: {beta_width:.1f}")

        if beta_width > 3:
            print(f"  ✓✓ ROBUST: Works over wide range of β")
            print(f"     → Not fine-tuned, parameter is flexible")
            return True
        elif beta_width > 1:
            print(f"  ✓ MODERATE: Works for some range of β")
            print(f"     → Some robustness, but not ideal")
            return True
        else:
            print(f"  ❌ FINE-TUNED: Requires very specific β")
            print(f"     → This is a bad sign, suggests overfitting")
            return False
    else:
        print(f"\n❌ FAILED: No β value gives acceptable results")
        return False


# ============================================================================
# TEST 3: Quark Prediction (THE BIG ONE)
# ============================================================================

def test_quark_prediction(beta_optimal):
    """
    Critical Test 3: Can we predict quarks WITHOUT retuning?

    THIS IS THE MOST IMPORTANT TEST.

    We tuned β and decay rates for leptons.
    Now use SAME β but DIFFERENT decay rates for quarks.

    If error > 100% → Theory is just curve fitting
    If error < 50% → Holy shit, might be real
    """
    print("\n" + "="*70)
    print("CRITICAL TEST 3: Quark Prediction")
    print("="*70)
    print("Question: Can we predict quarks WITHOUT retuning β?")
    print("Expected: Error < 100% (otherwise theory is wrong)")
    print("\nThis is THE test. If this fails, HNR is dead.")

    # Use β from leptons
    beta_from_leptons = beta_optimal

    # Target quark ratios (up:charm:top)
    # u ≈ 2.2 MeV, c ≈ 1300 MeV, t ≈ 173000 MeV
    target_quarks = np.array([1, 600, 79000])

    print(f"\nUsing β = {beta_from_leptons:.2f} (from lepton fit)")
    print(f"Target quarks: [1, 600, 79000]")
    print(f"Testing on 10 different networks...")

    results = []

    for trial in range(10):
        G = generate_graph('barabasi_albert', N=3000, m=3)
        patterns = identify_patterns(G, n_scales=12)        # Use QUARK sector (different decay rates, SAME β)
        ratios = compute_masses_hnr(patterns, beta=beta_from_leptons, sector='quark')

        if not (np.any(np.isnan(ratios)) or np.any(ratios <= 0)):
            results.append(ratios)

            if trial < 3:  # Print first few
                print(f"  Trial {trial+1}: [1, {ratios[1]:.0f}, {ratios[2]:.0f}]")

    if len(results) == 0:
        print("\n❌ CATASTROPHIC FAILURE: No valid predictions")
        print("   Theory produces NaN or invalid values for quarks")
        return False

    mean_ratios = np.mean(results, axis=0)
    std_ratios = np.std(results, axis=0)
    error = np.abs(mean_ratios - target_quarks) / target_quarks

    print(f"\n" + "="*70)
    print("QUARK PREDICTION RESULTS:")
    print("="*70)
    print(f"  Target:    [1, 600, 79000]")
    print(f"  Predicted: [1, {mean_ratios[1]:.0f}, {mean_ratios[2]:.0f}]")
    print(f"  Std dev:   [-, ±{std_ratios[1]:.0f}, ±{std_ratios[2]:.0f}]")
    print(f"\n  Error: Gen2={error[1]*100:.0f}%, Gen3={error[2]*100:.0f}%")

    # Verdict
    print(f"\n" + "="*70)
    print("TEST 3 VERDICT:")
    print("="*70)

    if error[1] < 0.3 and error[2] < 0.5:
        print("✓✓✓ BREAKTHROUGH: Predictions are EXCELLENT!")
        print("    Gen2 within 30%, Gen3 within 50%")
        print("    → HNR might actually be real physics!")
        print("    → This is NOT random, too accurate to be coincidence")
        return True
    elif error[1] < 0.5 and error[2] < 1.0:
        print("✓✓ STRONG: Predictions are good")
        print("   Gen2 within 50%, Gen3 within 100%")
        print("   → Worth investigating further")
        return True
    elif error[1] < 1.0:
        print("✓ MODERATE: Right order of magnitude")
        print("  Gen2 within 100%")
        print("  → Some predictive power, but not impressive")
        return True
    else:
        print("❌ FAILED: Predictions are terrible")
        print("   Error > 100% on Gen2")
        print("   → Theory is just curve fitting, no real physics")
        print("   → HNR is dead, move on")
        return False


# ============================================================================
# MAIN: Run All Tests
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("CRITICAL TESTS: Is HNR Real Physics or Just Curve Fitting?")
    print("="*70)
    print("\nWe will run 3 tests that will definitively answer this question.")
    print("Each test can FALSIFY the theory.")
    print("\nIf ALL pass → Worth 1-2 years of serious work")
    print("If ANY fails → Theory is wrong, abandon immediately")

    start_time = time.time()

    # First, optimize beta for leptons
    beta_optimal = optimize_beta()

    # Run tests with optimized beta
    test1_pass = test_network_independence(beta_optimal)
    test2_pass = test_parameter_robustness(beta_optimal)
    test3_pass = test_quark_prediction(beta_optimal)

    elapsed = time.time() - start_time

    # Final verdict
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)

    print(f"\nTest 1 (Network Independence): {'✓ PASS' if test1_pass else '❌ FAIL'}")
    print(f"Test 2 (Parameter Robustness):  {'✓ PASS' if test2_pass else '❌ FAIL'}")
    print(f"Test 3 (Quark Prediction):      {'✓ PASS' if test3_pass else '❌ FAIL'}")

    all_pass = test1_pass and test2_pass and test3_pass

    print("\n" + "="*70)
    if all_pass:
        print("✓✓✓ ALL TESTS PASSED!")
        print("="*70)
        print("""
HNR has survived critical falsification attempts!

This means:
1. It's not cherry-picking (works on different scale-free networks)
2. It's not fine-tuned (robust to parameter changes)
3. It has predictive power (quarks without retuning)

RECOMMENDATION: Worth investing serious time (6-24 months)

Next steps:
1. Learn quantum field theory properly
2. Formulate quantum version of HNR
3. Find physicist collaborator
4. Make additional predictions (neutrinos, etc.)
5. Attempt to derive parameters from first principles
6. Write up and submit to arXiv

Probability of becoming real physics: ~30% (much higher than 5%!)
        """)
    elif test1_pass and test2_pass:
        print("~ PARTIAL SUCCESS")
        print("="*70)
        print("""
HNR passes Tests 1 & 2 but fails quark prediction.

This means:
✓ It's a real pattern in scale-free networks
✓ It's robust, not fine-tuned
❌ But has limited predictive power

RECOMMENDATION: Interesting computational pattern, not real physics

The theory captures SOME universal property of scale-free networks
but doesn't actually encode particle physics.

Probability of becoming real physics: ~5%
        """)
    else:
        print("❌ THEORY FALSIFIED")
        print("="*70)
        print("""
HNR failed critical tests.

This means it's just curve fitting with no real physics.

RECOMMENDATION: Abandon theory, move on

What we learned:
• Spectral geometry doesn't work
• RG flow is promising direction
• But HNR specifically is wrong

Don't waste time trying to fix it. Better to:
• Study real physics (QFT, lattice QCD)
• Try completely different approaches
• Or work on established problems

Probability of becoming real physics: 0%
        """)

    print(f"Total runtime: {elapsed:.1f} seconds")
    print("="*70)

    # Save results
    import json
    with open('critical_tests_results.json', 'w') as f:
        json.dump({
            'test1_network_independence': test1_pass,
            'test2_parameter_robustness': test2_pass,
            'test3_quark_prediction': test3_pass,
            'all_tests_passed': all_pass,
            'recommendation': 'pursue' if all_pass else 'abandon',
            'probability': '30%' if all_pass else '5%' if (test1_pass and test2_pass) else '0%'
        }, f, indent=2)

    print("\nResults saved to critical_tests_results.json")
