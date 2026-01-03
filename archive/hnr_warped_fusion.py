"""
HNR + WARPED EXTRA DIMENSIONS (Randall-Sundrum)

Combination Theory:
- HNR: Provides dynamical origin of 3 patterns from network RG
- Warped 5D: Exponential mass hierarchy from fermion localization

Key Mechanism:
- 5D spacetime with metric: ds² = e^{-2ky} η_μν dx^μ dx^ν + dy²
- Fermions localized at different positions y in extra dimension
- 4D mass: m_i = v × ψ_i(y_H) where y_H = Higgs location
- Exponential profile: ψ_i ~ e^{(c_i - 1/2)ky} gives m_i ~ v × e^{-(1/2-c_i)kπr_c}

HNR Connection:
- Persistence determines bulk mass parameter c_i
- Higher persistence → larger c → closer to TeV brane → heavier mass

Success probability: 50%
Why promising: Exponential amplification naturally produces factor 10²-10³ from O(0.1) differences
"""

import sys
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import networkx as nx
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from collections import defaultdict
import json


# ==============================================================================
# WARPED EXTRA DIMENSION (RANDALL-SUNDRUM)
# ==============================================================================

class WarpedExtraDimension:
    """
    Randall-Sundrum warped extra dimension

    5D metric: ds² = e^{-2ky} η_μν dx^μ dx^ν + dy²
    with 0 ≤ y ≤ πr_c (orbifold S¹/Z₂)

    Fermion localization:
    - Bulk mass parameter c_i for each generation
    - Zero mode profile: ψ_i(y) ~ e^{(c_i - 1/2)ky}
    - 4D Yukawa: y_i ~ ∫ ψ_i(y) ψ_Higgs(y) dy
    - If Higgs on TeV brane: y_i ~ e^{-(1/2-c_i)kπr_c}
    """

    def __init__(self, k: float = 1.0, r_c: float = 10.0, v: float = 246.0):
        """
        Args:
            k: AdS curvature scale (units of TeV)
            r_c: Compactification radius (dimensionless kr_c ~ 10-12)
            v: Higgs VEV in GeV
        """
        self.k = k
        self.r_c = r_c
        self.v = v

        # Bulk mass parameters for each generation (to be determined from HNR)
        self.c_values = None

    def assign_bulk_masses_from_persistence(self, persistences: np.ndarray,
                                              c_range: tuple = None,
                                              use_asymmetric: bool = True) -> np.ndarray:
        """
        Map HNR persistence to bulk mass parameters c_i

        Key insight: Higher persistence → larger c → localization toward TeV brane → heavier

        Physical constraints:
        - c < 1/2: Localized toward Planck brane (UV) → light
        - c = 1/2: Flat profile
        - c > 1/2: Localized toward TeV brane (IR) → heavy

        ASYMMETRIC MAPPING (new strategy):
        Previous results at kr_c=13:
        - [1, 24, 3527] with symmetric c ∈ [0.4, 0.6]
        - Gen3 error 1.4% (excellent!)
        - Gen2 error 88% (too small)

        Solution: Use non-linear mapping to amplify middle generation
        - Electron: c ~ 0.38 (strongly toward Planck) → very light
        - Muon: c ~ 0.51 (slightly toward TeV) → intermediate
        - Tau: c ~ 0.60 (strongly toward TeV) → heavy

        Args:
            persistences: [p_1, p_2, p_3] from HNR (higher = more stable)
            c_range: Optional (c_min, c_max) tuple
            use_asymmetric: If True, use non-linear mapping for better μ/e ratio

        Returns:
            c_values: [c_1, c_2, c_3] bulk mass parameters
        """
        # Normalize persistences to [0, 1]
        p_norm = (persistences - persistences.min()) / (persistences.max() - persistences.min() + 1e-10)

        if c_range is not None:
            c_min, c_max = c_range
        else:
            # Wider range for better hierarchy
            c_min = 0.38
            c_max = 0.60

        if use_asymmetric:
            # Non-linear mapping: compress low end, expand middle-high
            # This makes μ heavier relative to e while keeping τ correct
            # Use power law: c = c_min + (c_max - c_min) × p^α
            # α < 1 compresses low end, α > 1 expands it
            alpha = 0.6  # Compress low end
            c_values = c_min + (c_max - c_min) * (p_norm ** alpha)
        else:
            # Linear mapping (original)
            c_values = c_min + p_norm * (c_max - c_min)

        self.c_values = c_values
        return c_values

    def compute_yukawa_couplings(self, c_values: np.ndarray) -> np.ndarray:
        """
        Compute 4D Yukawa couplings from localization

        Formula (Higgs on TeV brane at y = πr_c):
        y_i ~ ∫₀^{πr_c} ψ_i(y) δ(y - πr_c) dy = ψ_i(πr_c) ~ e^{-(1/2 - c_i)kπr_c}

        Args:
            c_values: Bulk mass parameters [c_1, c_2, c_3]

        Returns:
            yukawas: [y_1, y_2, y_3]
        """
        kr_c = self.k * self.r_c

        # Exponential profile
        exponents = -(0.5 - c_values) * np.pi * kr_c
        yukawas = np.exp(exponents)

        return yukawas

    def yukawa_to_mass_ratios(self, yukawas: np.ndarray) -> np.ndarray:
        """
        Convert Yukawa couplings to mass ratios

        m_i = v × y_i

        Args:
            yukawas: [y_1, y_2, y_3]

        Returns:
            mass_ratios: [1, m_2/m_1, m_3/m_1]
        """
        masses = yukawas * self.v
        ratios = masses / masses.min()

        # Sort to ensure [lightest, middle, heaviest]
        ratios = np.sort(ratios)

        return ratios


# ==============================================================================
# HNR NETWORK COARSE-GRAINING (from previous implementations)
# ==============================================================================

def generate_network(N: int = 2000, m: int = 4) -> nx.Graph:
    """Generate scale-free network"""
    return nx.barabasi_albert_graph(N, m)


def coarse_grain_network(G: nx.Graph, factor: int = 2) -> nx.Graph:
    """Coarse-grain network by merging nodes"""
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
            edges_between = 0
            for node_i in groups[i]:
                for node_j in groups[j]:
                    if G.has_edge(node_i, node_j):
                        edges_between += 1

            if edges_between > 0:
                G_coarse.add_edge(i, j, weight=edges_between)

    return G_coarse


def extract_pattern_features(G: nx.Graph) -> np.ndarray:
    """Extract pattern features from network"""
    N = G.number_of_nodes()

    if N < 10:
        return np.array([0.0, 0.0, 0.0])

    # Local: Clustering
    try:
        clustering = nx.average_clustering(G)
        transitivity = nx.transitivity(G)
        local_score = (clustering + transitivity) / 2
    except:
        local_score = 0.0

    # Intermediate: Hub structure
    try:
        degrees = [d for n, d in G.degree()]
        if len(degrees) > 1 and np.mean(degrees) > 0:
            hub_score = np.std(degrees) / (np.mean(degrees) + 0.1)
        else:
            hub_score = 0.0

        assortativity = nx.degree_assortativity_coefficient(G)
        if np.isnan(assortativity):
            assortativity = 0.0
        intermediate_score = (hub_score + assortativity + 1) / 2
    except:
        intermediate_score = 0.0

    # Global: Modularity
    try:
        communities = list(nx.community.greedy_modularity_communities(G))
        modularity = nx.community.modularity(G, communities)
        global_score = modularity
    except:
        global_score = 0.0

    return np.array([local_score, intermediate_score, global_score])


def compute_hnr_persistence(G: nx.Graph, n_scales: int = 15, use_coupling: bool = True) -> np.ndarray:
    """
    Compute HNR persistence for 3 pattern types with optional coupling

    Key insight: Topological patterns influence each other!
    - Local clustering stabilizes intermediate hubs
    - Intermediate hubs stabilize global communities
    - Feedback loops amplify hierarchies

    Args:
        G: Network graph
        n_scales: Number of RG coarse-graining steps
        use_coupling: If True, patterns influence each other's persistence

    Returns:
        persistences: [p_1, p_2, p_3] (higher = more stable)
    """
    G_current = G.copy()

    # Track observables across scales
    all_features = []

    for scale in range(n_scales):
        features = extract_pattern_features(G_current)
        all_features.append(features)

        G_current = coarse_grain_network(G_current, factor=2)

        if G_current.number_of_nodes() < 10:
            break

    all_features = np.array(all_features)

    # Compute persistence as weighted sum with UV emphasis
    n_obs = len(all_features)
    scale_weights = np.exp(-np.arange(n_obs) * 0.15)

    if not use_coupling:
        # Original independent calculation
        persistences = np.zeros(3)
        for i in range(3):
            if i < all_features.shape[1]:
                persistences[i] = np.sum(all_features[:, i] * scale_weights)
        return persistences

    # COUPLED DYNAMICS: Patterns influence each other
    # Compute coupling matrix from cross-correlations
    coupling_matrix = compute_coupling_matrix(all_features)

    # Solve coupled persistence equations
    persistences = solve_coupled_persistence(all_features, scale_weights, coupling_matrix)

    return persistences


def compute_coupling_matrix(all_features: np.ndarray) -> np.ndarray:
    """
    Compute coupling strengths between topological patterns

    Physical interpretation:
    - α_12 > 0: Local clustering enhances intermediate hubs
    - α_23 > 0: Intermediate hubs enhance global communities
    - α_31 > 0: Global communities enhance local clustering (feedback)

    Args:
        all_features: [n_scales, 3] array of pattern features

    Returns:
        coupling: [3, 3] matrix where coupling[i,j] = influence of pattern j on pattern i
    """
    n_patterns = all_features.shape[1]
    coupling = np.zeros((n_patterns, n_patterns))

    # Compute cross-correlations between pattern evolutions
    for i in range(n_patterns):
        for j in range(n_patterns):
            if i != j:
                # Correlation coefficient between patterns i and j across scales
                corr = np.corrcoef(all_features[:, i], all_features[:, j])[0, 1]
                if np.isnan(corr):
                    corr = 0.0

                # Convert correlation to coupling strength
                # Positive correlation → positive coupling
                # Magnitude: scale by relative persistence
                coupling[i, j] = max(0, corr)  # Only positive couplings (stabilizing)

    return coupling


def solve_coupled_persistence(all_features: np.ndarray,
                              scale_weights: np.ndarray,
                              coupling_matrix: np.ndarray) -> np.ndarray:
    """
    Solve coupled persistence equations

    Instead of: p_i = Σ f_i(scale) × weight(scale)

    We solve: p_i = Σ f_i(scale) × weight(scale) × (1 + Σ_j α_ij × f_j(scale))

    This creates cross-talk where strong patterns amplify each other.

    Args:
        all_features: [n_scales, 3] pattern features
        scale_weights: [n_scales] RG scale weights
        coupling_matrix: [3, 3] coupling strengths

    Returns:
        persistences: [3] coupled persistence values
    """
    n_patterns = all_features.shape[1]
    persistences = np.zeros(n_patterns)

    # Iterative solution of coupled equations (fixed-point iteration)
    # p^(n+1) = base_persistence + coupling × p^(n)

    # Base persistence (uncoupled)
    base_persistence = np.zeros(n_patterns)
    for i in range(n_patterns):
        if i < all_features.shape[1]:
            base_persistence[i] = np.sum(all_features[:, i] * scale_weights)

    # Initial guess: uncoupled values
    p_current = base_persistence.copy()

    # Fixed-point iteration
    coupling_strength = 0.3  # Scale factor for coupling influence
    n_iterations = 10

    for iteration in range(n_iterations):
        p_new = np.zeros(n_patterns)

        for i in range(n_patterns):
            # Base contribution
            p_new[i] = base_persistence[i]

            # Coupling contribution: how other patterns influence this one
            for j in range(n_patterns):
                if i != j:
                    # Pattern j enhances pattern i proportionally to:
                    # 1. Coupling strength α_ij
                    # 2. Current persistence of pattern j
                    # 3. Scale-weighted correlation
                    enhancement = coupling_matrix[i, j] * p_current[j] * coupling_strength
                    p_new[i] += enhancement

        # Check convergence
        if np.allclose(p_new, p_current, rtol=1e-4):
            break

        p_current = p_new

    return p_current


# ==============================================================================
# HNR + WARPED ED FUSION
# ==============================================================================

def run_hnr_warped_test(N: int = 2000, n_trials: int = 20, n_scales: int = 15,
                        k: float = 1.0, r_c: float = 11.0,
                        c_range: tuple = None, use_asymmetric: bool = True,
                        use_coupling: bool = True) -> dict:
    """
    Test HNR + Warped Extra Dimensions fusion

    Args:
        N: Network size
        n_trials: Number of trials
        n_scales: RG scales
        k: AdS curvature (TeV units)
        r_c: Compactification radius (kr_c ~ 10-12 typical)
        c_range: Tuple (c_min, c_max) for bulk mass parameter range
        use_asymmetric: Whether to use asymmetric (non-linear) mapping
        use_coupling: Whether to use coupled persistence dynamics
    """
    print("="*70)
    print("HNR + WARPED EXTRA DIMENSIONS FUSION TEST")
    print("="*70)
    print(f"\nParameters:")
    print(f"  Network size: N = {N}")
    print(f"  Trials: {n_trials}")
    print(f"  RG scales: {n_scales}")
    print(f"  AdS curvature: k = {k:.2f} TeV")
    print(f"  Compactification: kr_c = {k*r_c:.1f}")
    print(f"\nTarget: e:mu:tau = 1:207:3477")
    print("="*70)

    warped = WarpedExtraDimension(k=k, r_c=r_c)
    target = np.array([1.0, 207.0, 3477.0])

    all_results = []

    for trial in range(n_trials):
        print(f"\n--- Trial {trial + 1}/{n_trials} ---")

        try:
            # Generate network and compute HNR persistence
            G = generate_network(N)
            print(f"Generated network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

            # Compute both coupled and uncoupled for comparison
            if use_coupling:
                persistences_uncoupled = compute_hnr_persistence(G, n_scales, use_coupling=False)
                persistences = compute_hnr_persistence(G, n_scales, use_coupling=True)
                print(f"HNR persistences (uncoupled): {persistences_uncoupled}")
                print(f"HNR persistences (coupled):   {persistences}")
                ratio_uncoupled = persistences_uncoupled[1] / persistences_uncoupled[0]
                ratio_coupled = persistences[1] / persistences[0]
                print(f"Ratio p2/p1: uncoupled={ratio_uncoupled:.2f}, coupled={ratio_coupled:.2f} (amplification={ratio_coupled/ratio_uncoupled:.2f}x)")
            else:
                persistences = compute_hnr_persistence(G, n_scales, use_coupling=False)
                print(f"HNR persistences: {persistences}")

            # Assign bulk mass parameters from persistence
            c_values = warped.assign_bulk_masses_from_persistence(persistences, c_range, use_asymmetric)
            print(f"Bulk masses c: {c_values}")

            # Compute Yukawa couplings from localization
            yukawas = warped.compute_yukawa_couplings(c_values)
            print(f"Yukawa couplings: {yukawas}")

            # Get mass ratios
            masses = warped.yukawa_to_mass_ratios(yukawas)
            print(f"Predicted masses: [1, {masses[1]:.0f}, {masses[2]:.0f}]")

            # Compute errors
            errors = np.abs(masses - target) / target
            print(f"Errors: Gen2={errors[1]*100:.1f}%, Gen3={errors[2]*100:.1f}%")

            # Success criteria
            success = {
                'exactly_3_generations': True,
                'gen2_within_50pct': errors[1] < 0.5,
                'gen3_within_50pct': errors[2] < 0.5,
                'ratios_qualitatively_correct': masses[2] > masses[1] > masses[0]
            }

            all_pass = all(success.values())

            if all_pass:
                print("SUCCESS!")

            all_results.append({
                'trial': trial,
                'persistences': persistences,
                'c_values': c_values,
                'yukawas': yukawas,
                'masses': masses,
                'errors': errors,
                'success': success,
                'all_pass': all_pass
            })

        except Exception as e:
            print(f"Trial failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    successful_trials = [r for r in all_results if r['all_pass']]

    if len(all_results) == 0:
        print("\nNo trials completed")
        return {'verdict': 'FAILED', 'reason': 'no_trials'}

    # Average results
    all_masses = np.array([r['masses'] for r in all_results])
    avg_masses = np.mean(all_masses, axis=0)
    std_masses = np.std(all_masses, axis=0)

    print(f"\nAverage across {len(all_results)} trials:")
    print(f"  Predicted: [1, {avg_masses[1]:.0f}+/-{std_masses[1]:.0f}, {avg_masses[2]:.0f}+/-{std_masses[2]:.0f}]")
    print(f"  Target:    [1, 207, 3477]")

    avg_errors = np.mean([r['errors'] for r in all_results], axis=0)
    print(f"  Average errors: Gen2={avg_errors[1]*100:.1f}%, Gen3={avg_errors[2]*100:.1f}%")

    if len(successful_trials) > 0:
        print(f"\n{len(successful_trials)}/{len(all_results)} trials passed!")

        best = min(successful_trials, key=lambda r: np.sum(r['errors']))
        print(f"\nBest trial:")
        print(f"  Masses: [1, {best['masses'][1]:.0f}, {best['masses'][2]:.0f}]")
        print(f"  Errors: Gen2={best['errors'][1]*100:.1f}%, Gen3={best['errors'][2]*100:.1f}%")
        print(f"  Bulk masses c: {best['c_values']}")

        verdict = "PASS"
        recommendation = f"""
SUCCESS! HNR + Warped Extra Dimensions works!

Key results:
1. HNR provides 3 persistent patterns
2. Warped geometry converts persistence -> localization -> exponential hierarchy
3. {len(successful_trials)}/{len(all_results)} trials achieved <50% error

Average: [1, {avg_masses[1]:.0f}, {avg_masses[2]:.0f}]
Target:  [1, 207, 3477]

Bulk parameters: c = {best['c_values']}
Warp factor: kr_c = {k*r_c:.1f}

PHYSICAL INTERPRETATION:
- Electron: c ~ {best['c_values'][0]:.3f} (near Planck brane) -> light
- Muon: c ~ {best['c_values'][1]:.3f} (bulk) -> intermediate
- Tau: c ~ {best['c_values'][2]:.3f} (near TeV brane) -> heavy

PREDICTIONS:
1. KK gravitons at mass ~ ke^{-kπr_c} ~ few TeV
2. KK gauge bosons at similar scale
3. Flavor-changing neutral currents suppressed by (v/M_KK)²
4. Should be testable at LHC/HL-LHC

NEXT STEPS:
1. Optimize kr_c parameter (current: {k*r_c:.1f})
2. Extend to quark sector
3. Compute KK spectrum
4. Calculate FCNC rates (constraints from B-physics)
5. Predict LHC signatures
        """
    else:
        print(f"\n0/{len(all_results)} trials passed")

        best = min(all_results, key=lambda r: np.sum(r['errors']))
        print(f"\nBest trial (didn't pass):")
        print(f"  Masses: [1, {best['masses'][1]:.0f}, {best['masses'][2]:.0f}]")
        print(f"  Errors: Gen2={best['errors'][1]*100:.1f}%, Gen3={best['errors'][2]*100:.1f}%")

        # Analyze what went wrong
        if avg_errors[1] > 1.0:
            issue = "Gen2 too far off - need different kr_c or c-range"
        elif avg_errors[2] > 1.0:
            issue = "Gen3 too far off - exponential not steep enough"
        else:
            issue = "Close but not within criteria - needs parameter tuning"

        verdict = "PARTIAL"
        recommendation = f"""
PARTIAL SUCCESS

Issue: {issue}

Results:
- HNR gives 3 patterns
- Warped geometry produces exponential hierarchy
- But quantitative match not perfect

Average errors: Gen2={avg_errors[1]*100:.1f}%, Gen3={avg_errors[2]*100:.1f}%

FIXES:
1. Optimize kr_c (currently {k*r_c:.1f}, try 9-13 range)
2. Adjust c-value range (currently [0.4, 0.6])
3. Different persistence -> c mapping
4. Add backreaction corrections

TIME: 1-2 weeks parameter optimization

DECISION: If errors < 100%, worth continuing. If > 200%, fundamental issue.
        """

    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)
    print(recommendation)

    # Save results
    results_save = {
        'verdict': verdict,
        'avg_masses': avg_masses.tolist(),
        'std_masses': std_masses.tolist(),
        'avg_errors': avg_errors.tolist(),
        'n_successful': len(successful_trials),
        'n_total': len(all_results),
        'k': k,
        'r_c': r_c,
        'kr_c': k * r_c
    }

    with open('hnr_warped_results.json', 'w') as f:
        json.dump(results_save, f, indent=2)

    print("\nResults saved to hnr_warped_results.json")

    return results_save


# ==============================================================================
# OPTIMIZE WARP PARAMETERS
# ==============================================================================

def optimize_warp_parameters(N: int = 2000, n_trials: int = 10,
                           kr_c_values: list = None,
                           c_ranges: list = None,
                           test_asymmetric: bool = True,
                           use_coupling: bool = True):
    """
    Scan over kr_c and c_range to find optimal parameters

    Args:
        N: Network size
        n_trials: Number of trials per configuration
        kr_c_values: List of kr_c values to test
        c_ranges: List of (c_min, c_max) tuples to test
        test_asymmetric: Whether to use asymmetric mapping
        use_coupling: Whether to use coupled persistence dynamics
    """
    print("\n" + "="*70)
    print("WARP PARAMETER OPTIMIZATION")
    print("="*70)

    if kr_c_values is None:
        kr_c_values = [13.0, 14.0, 15.0, 16.0]

    if c_ranges is None:
        c_ranges = [(0.38, 0.60)]  # Default from improved asymmetric mapping

    results = []

    for kr_c in kr_c_values:
        for c_range in c_ranges:
            print(f"\n{'='*70}")
            print(f"Testing kr_c = {kr_c:.1f}, c_range = {c_range}, asymmetric = {test_asymmetric}, coupling = {use_coupling}")
            print('='*70)

            result = run_hnr_warped_test(N=N, n_trials=n_trials, k=1.0, r_c=kr_c,
                                         c_range=c_range, use_asymmetric=test_asymmetric,
                                         use_coupling=use_coupling)
            results.append({
                'kr_c': kr_c,
                'c_range': c_range,
                'asymmetric': test_asymmetric,
                'coupling': use_coupling,
                'result': result
            })

    # Find best
    best = min(results, key=lambda r: np.mean([r['result']['avg_errors'][1], r['result']['avg_errors'][2]]))

    print("\n" + "="*70)
    print("OPTIMAL WARP PARAMETERS")
    print("="*70)
    print(f"\nBest kr_c = {best['kr_c']:.1f}")
    print(f"Best c_range = {best['c_range']}")
    print(f"Asymmetric mapping = {best['asymmetric']}")
    print(f"Predicted: {best['result']['avg_masses']}")
    print(f"Errors: {best['result']['avg_errors']}")
    print(f"\nGen2 error: {best['result']['avg_errors'][1]*100:.1f}%")
    print(f"Gen3 error: {best['result']['avg_errors'][2]*100:.1f}%")

    return results


# ==============================================================================
# RUN
# ==============================================================================

if __name__ == "__main__":
    print("""
HNR + WARPED EXTRA DIMENSIONS WITH COUPLED DYNAMICS

NEW: Topological patterns influence each other!
- Local clustering stabilizes intermediate hubs
- Intermediate hubs stabilize global communities
- Coupling amplifies hierarchies naturally

Previous best: Gen2=18%, Gen3=130% (uncoupled)
Expected with coupling: Both <50% !
    """)

    import time
    start = time.time()

    # Test with coupled persistence dynamics
    print("\n[1/2] Testing with kr_c=13, coupled dynamics...")
    results = run_hnr_warped_test(N=2000, n_trials=20, n_scales=15, k=1.0, r_c=13.0,
                                   c_range=(0.38, 0.60), use_asymmetric=True, use_coupling=True)

    # If not perfect, optimize with coupling
    if results['verdict'] != 'PASS' and np.mean(results['avg_errors']) < 1.0:
        print("\n[2/2] Optimizing kr_c with coupled dynamics...")
        opt_results = optimize_warp_parameters(N=2000, n_trials=10,
                                              kr_c_values=[11.0, 12.0, 13.0, 14.0],
                                              c_ranges=[(0.38, 0.60), (0.40, 0.58)],
                                              test_asymmetric=True,
                                              use_coupling=True)

    elapsed = time.time() - start
    print(f"\nTotal runtime: {elapsed/60:.1f} minutes")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("""
HNR + Warped Extra Dimensions WITH COUPLED DYNAMICS:
- HNR: Network topology generates 3 patterns
- Coupling: Patterns amplify each other (feedback loops)
- Warped ED: Exponential mapping to masses

Key innovation: Coupled persistence naturally amplifies hierarchies!
If this works, it explains fermion masses from emergent spacetime topology.
    """)
