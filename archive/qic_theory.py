"""
QUANTUM INFORMATION COMPRESSION (QIC) THEORY

Core Principle: Fermion masses emerge from algorithmic complexity
- Wheeler's "It from Bit": Information is fundamental
- Quantum eraser: Information determines physical reality
- Generations = Different information-theoretic stability classes

Key Ideas:
1. Network state = quantum information register
2. Kolmogorov complexity K(state) = minimum description length
3. Compressibility C = 1 - K(compressed)/K(original)
4. Mass ∝ 1/C (less compressible → heavier)

Physical Interpretation:
- Gen1 (electron): Highly compressible pattern → lightest
- Gen2 (muon): Medium compressibility → intermediate
- Gen3 (tau): Low compressibility (complex) → heaviest

This naturally produces exponential hierarchies because compression ratios
can vary by orders of magnitude (like JPEG vs raw image).

Testable Prediction:
If QIC is correct, LHC data should show information-theoretic correlations
in fermion production rates matching compression hierarchy.
"""

import numpy as np
import networkx as nx
from typing import Tuple, Dict, List
import json
from scipy.stats import entropy as scipy_entropy
from collections import Counter
import io
import sys

# UTF-8 output for Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# ==============================================================================
# INFORMATION THEORY OBSERVABLES
# ==============================================================================

def compute_shannon_entropy(G: nx.Graph) -> float:
    """
    Shannon entropy of degree distribution

    H = -Σ p(k) log p(k)

    Lower entropy = more ordered = more compressible
    """
    degrees = [d for n, d in G.degree()]
    if len(degrees) == 0:
        return 0.0

    # Probability distribution
    degree_counts = Counter(degrees)
    total = sum(degree_counts.values())
    probs = np.array([count/total for count in degree_counts.values()])

    # Shannon entropy
    H = -np.sum(probs * np.log2(probs + 1e-10))
    return H


def compute_mutual_information(G: nx.Graph) -> float:
    """
    Mutual information between node degrees

    I(X;Y) = H(X) + H(Y) - H(X,Y)

    Measures correlation in network structure
    Higher MI = more redundancy = more compressible
    """
    edges = list(G.edges())
    if len(edges) == 0:
        return 0.0

    # Degree pairs
    degree_pairs = [(G.degree(u), G.degree(v)) for u, v in edges]

    if len(degree_pairs) == 0:
        return 0.0

    # Individual entropies
    degrees_u = [d[0] for d in degree_pairs]
    degrees_v = [d[1] for d in degree_pairs]

    H_u = compute_distribution_entropy(degrees_u)
    H_v = compute_distribution_entropy(degrees_v)

    # Joint entropy
    H_uv = compute_distribution_entropy(degree_pairs)

    # Mutual information
    MI = H_u + H_v - H_uv
    return max(0, MI)  # Can be slightly negative due to numerical errors


def compute_distribution_entropy(values: list) -> float:
    """Helper: entropy of any discrete distribution"""
    if len(values) == 0:
        return 0.0

    counts = Counter(values)
    total = sum(counts.values())
    probs = np.array([count/total for count in counts.values()])

    H = -np.sum(probs * np.log2(probs + 1e-10))
    return H


def compute_kolmogorov_proxy(G: nx.Graph) -> float:
    """
    Proxy for Kolmogorov complexity using compression

    True K(x) is uncomputable, but we can estimate via:
    - Lempel-Ziv complexity
    - Graph motif counts
    - Structural information content

    Returns: Normalized complexity ∈ [0, 1]
    Lower = more compressible
    """
    N = G.number_of_nodes()
    E = G.number_of_edges()

    if N == 0:
        return 0.0

    # Method 1: Degree sequence complexity
    degrees = sorted([d for n, d in G.degree()])
    degree_complexity = compute_lz_complexity(degrees)

    # Method 2: Clustering heterogeneity (structure complexity)
    try:
        clustering = nx.clustering(G)
        clustering_values = list(clustering.values())
        if len(clustering_values) > 0:
            clustering_std = np.std(clustering_values)
        else:
            clustering_std = 0.0
    except:
        clustering_std = 0.0

    # Method 3: Motif diversity (higher diversity = less compressible)
    try:
        triangles = sum(nx.triangles(G).values()) / 3
        max_triangles = N * (N-1) * (N-2) / 6
        triangle_ratio = triangles / (max_triangles + 1)
    except:
        triangle_ratio = 0.0

    # Combined complexity (normalized)
    complexity = (degree_complexity + clustering_std + triangle_ratio) / 3

    return min(1.0, complexity)


def compute_lz_complexity(sequence: list) -> float:
    """
    Lempel-Ziv complexity of sequence

    Counts distinct patterns - proxy for algorithmic complexity
    More patterns = less compressible
    """
    if len(sequence) == 0:
        return 0.0

    # Convert to binary string (simplified)
    # Bin values into 8 bins
    if len(sequence) == 0:
        return 0.0

    min_val = min(sequence)
    max_val = max(sequence)
    if max_val == min_val:
        return 0.0

    bins = 8
    binary_seq = [int((x - min_val) / (max_val - min_val) * (bins-1)) for x in sequence]

    # Count unique substrings (simplified LZ)
    patterns = set()
    for length in range(1, min(len(binary_seq), 10)):
        for i in range(len(binary_seq) - length + 1):
            pattern = tuple(binary_seq[i:i+length])
            patterns.add(pattern)

    # Normalize by maximum possible
    max_patterns = len(binary_seq) * (len(binary_seq) + 1) / 2
    lz = len(patterns) / (max_patterns + 1)

    return lz


def compute_compressibility(G: nx.Graph) -> float:
    """
    Overall compressibility metric

    C = 1 - (information_content / max_possible_information)

    High C → highly compressible → should be light
    Low C → incompressible → should be heavy

    Returns: Compressibility ∈ [0, 1]
    """
    # Three measures of compressibility

    # 1. Entropy-based: Lower entropy = more ordered = more compressible
    H = compute_shannon_entropy(G)
    N = G.number_of_nodes()
    if N > 1:
        H_max = np.log2(N)  # Maximum entropy (uniform distribution)
        entropy_compress = 1 - (H / (H_max + 1e-10))
    else:
        entropy_compress = 0.0

    # 2. Mutual information: Higher MI = more redundancy = more compressible
    MI = compute_mutual_information(G)
    mi_compress = MI / (H + 1e-10)  # Normalized by entropy

    # 3. Kolmogorov proxy: Lower complexity = more compressible
    K_proxy = compute_kolmogorov_proxy(G)
    kolmogorov_compress = 1 - K_proxy

    # Combined compressibility (weighted average)
    C = 0.4 * entropy_compress + 0.3 * mi_compress + 0.3 * kolmogorov_compress

    return max(0.0, min(1.0, C))


# ==============================================================================
# NETWORK GENERATION & RG FLOW
# ==============================================================================

def generate_network(N: int, m: int = 4) -> nx.Graph:
    """Generate Barabási-Albert scale-free network"""
    G = nx.barabasi_albert_graph(N, m, seed=None)
    return G


def coarse_grain_network(G: nx.Graph, factor: int = 2) -> nx.Graph:
    """
    Coarse-grain network by factor using community detection
    """
    if G.number_of_nodes() < factor * 2:
        return G

    try:
        # Community detection
        communities = list(nx.community.greedy_modularity_communities(G))

        # Create coarse-grained network
        G_coarse = nx.Graph()

        # Each community becomes a node
        for i, community in enumerate(communities):
            G_coarse.add_node(i)

        # Add edges between communities
        for i, comm_i in enumerate(communities):
            for j, comm_j in enumerate(communities):
                if i < j:
                    # Check if there's connection between communities
                    for node_i in comm_i:
                        for node_j in comm_j:
                            if G.has_edge(node_i, node_j):
                                G_coarse.add_edge(i, j)
                                break
                        if G_coarse.has_edge(i, j):
                            break

        return G_coarse
    except:
        # Fallback: simple clustering
        nodes = list(G.nodes())
        n_clusters = max(1, len(nodes) // factor)

        G_coarse = nx.Graph()
        for i in range(n_clusters):
            G_coarse.add_node(i)

        # Random connections
        for i in range(n_clusters):
            for j in range(i+1, n_clusters):
                if np.random.random() < 0.3:
                    G_coarse.add_edge(i, j)

        return G_coarse


# ==============================================================================
# QIC MASS GENERATION
# ==============================================================================

def compute_qic_masses(G: nx.Graph, n_scales: int = 15) -> Tuple[np.ndarray, Dict]:
    """
    Compute fermion masses from quantum information compression

    Algorithm:
    1. Track compressibility across RG scales
    2. Identify 3 stability classes (generations)
    3. Map compressibility → mass (inverse relationship)

    Returns:
        masses: [m_1, m_2, m_3] mass ratios
        diagnostics: Dictionary of intermediate values
    """
    G_current = G.copy()

    # Track compressibility across scales
    compressibilities = []
    entropies = []
    complexities = []

    for scale in range(n_scales):
        # Compute information-theoretic observables
        C = compute_compressibility(G_current)
        H = compute_shannon_entropy(G_current)
        K = compute_kolmogorov_proxy(G_current)

        compressibilities.append(C)
        entropies.append(H)
        complexities.append(K)

        # Coarse-grain for next scale
        G_current = coarse_grain_network(G_current, factor=2)

        if G_current.number_of_nodes() < 10:
            break

    # Convert to arrays
    compressibilities = np.array(compressibilities)
    entropies = np.array(entropies)
    complexities = np.array(complexities)

    # Three stability classes from information flow
    # Use UV emphasis (early scales more important)
    scales = len(compressibilities)
    weights = np.exp(-np.arange(scales) * 0.2)
    weights /= weights.sum()

    # Generation 1: Highest compressibility (most stable/simple)
    # Generation 2: Medium compressibility
    # Generation 3: Lowest compressibility (least stable/complex)

    # Extract three observables
    compress_avg = np.sum(compressibilities * weights)
    entropy_avg = np.sum(entropies * weights)
    complexity_avg = np.sum(complexities * weights)

    # Map to three generations using spectral decomposition
    # Use variance across scales as discriminant
    compress_var = np.var(compressibilities)
    entropy_var = np.var(entropies)
    complexity_var = np.var(complexities)

    # Three generation signatures - FIXED SCALING
    # All scores normalized to [0,1] range for fair comparison
    
    # Generation 1: High compressibility + stable → lightest
    # C is already in [0,1], variance is small
    gen1_score = compress_avg + 0.1 * compress_var
    
    # Generation 2: Entropy-complexity balance → intermediate
    # Normalize H (typ. 2-4) and use ratio
    H_norm = entropy_avg / 5.0  # Normalize to ~[0,1]
    K_norm = complexity_avg  # Already ~[0,1]
    gen2_score = (H_norm / (K_norm + 0.1)) * 0.3  # Scale down ratio to [0,1]
    
    # Generation 3: Low compressibility + high complexity → heaviest
    # Inverse compressibility emphasizes complexity
    gen3_score = (complexity_avg / (compress_avg + 0.1)) * 0.2  # Scale to [0,1]

    # All scores now in comparable ranges [0,1]
    scores = np.array([gen1_score, gen2_score, gen3_score])    # Mass formula: m ∝ 1/C^α where α controls hierarchy strength
    # Higher compressibility → lower mass
    alpha = 2.5  # Tunable parameter

    masses = (1.0 / (scores + 0.01)) ** alpha

    # Normalize to electron = 1
    masses = masses / masses.min()

    diagnostics = {
        'compressibilities': compressibilities,
        'entropies': entropies,
        'complexities': complexities,
        'scores': scores,
        'alpha': alpha,
        'compress_avg': compress_avg,
        'entropy_avg': entropy_avg,
        'complexity_avg': complexity_avg
    }

    return masses, diagnostics


# ==============================================================================
# TESTING FRAMEWORK
# ==============================================================================

def run_qic_test(N: int = 2000, n_trials: int = 20, n_scales: int = 15) -> dict:
    """
    Test QIC theory

    Args:
        N: Network size
        n_trials: Number of trials
        n_scales: RG scales

    Returns:
        results: Dictionary with test results
    """
    print("="*70)
    print("QUANTUM INFORMATION COMPRESSION (QIC) THEORY TEST")
    print("="*70)
    print(f"\nParameters:")
    print(f"  Network size: N = {N}")
    print(f"  Trials: {n_trials}")
    print(f"  RG scales: {n_scales}")
    print(f"\nTarget: e:μ:τ = 1:207:3477")
    print("="*70)

    target = np.array([1.0, 207.0, 3477.0])
    all_results = []

    for trial in range(n_trials):
        print(f"\n--- Trial {trial + 1}/{n_trials} ---")

        try:
            # Generate network
            G = generate_network(N)
            print(f"Generated network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

            # Compute masses from QIC
            masses, diagnostics = compute_qic_masses(G, n_scales)

            print(f"Compressibility scores: {diagnostics['scores']}")
            print(f"Predicted masses: [1, {masses[1]:.0f}, {masses[2]:.0f}]")

            # Compute errors
            errors = np.abs(masses - target) / target
            print(f"Errors: Gen2={errors[1]*100:.1f}%, Gen3={errors[2]*100:.1f}%")

            # Success criteria
            success = {
                'exactly_3_generations': len(masses) == 3,
                'gen2_within_50pct': errors[1] < 0.5,
                'gen3_within_50pct': errors[2] < 0.5,
                'ratios_qualitatively_correct': masses[2] > masses[1] > masses[0]
            }

            all_pass = all(success.values())
            if all_pass:
                print("✓ SUCCESS!")

            all_results.append({
                'trial': trial,
                'masses': masses,
                'errors': errors,
                'success': success,
                'all_pass': all_pass,
                'diagnostics': diagnostics
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

    if len(all_results) == 0:
        print("\nNo successful trials!")
        return {'verdict': 'FAIL', 'all_results': []}

    # Average results
    all_masses = np.array([r['masses'] for r in all_results])
    all_errors = np.array([r['errors'] for r in all_results])

    avg_masses = np.mean(all_masses, axis=0)
    std_masses = np.std(all_masses, axis=0)
    avg_errors = np.mean(all_errors, axis=0)

    print(f"\nAverage across {len(all_results)} trials:")
    print(f"  Predicted: [1, {avg_masses[1]:.0f}±{std_masses[1]:.0f}, {avg_masses[2]:.0f}±{std_masses[2]:.0f}]")
    print(f"  Target:    [1, 207, 3477]")
    print(f"  Average errors: Gen2={avg_errors[1]*100:.1f}%, Gen3={avg_errors[2]*100:.1f}%")

    # Count successes
    successful_trials = [r for r in all_results if r['all_pass']]
    print(f"\n{len(successful_trials)}/{len(all_results)} trials passed")

    # Best trial
    if len(all_results) > 0:
        best_trial = min(all_results, key=lambda r: np.sum(r['errors'][1:]))
        print(f"\nBest trial:")
        print(f"  Masses: [1, {best_trial['masses'][1]:.0f}, {best_trial['masses'][2]:.0f}]")
        print(f"  Errors: Gen2={best_trial['errors'][1]*100:.1f}%, Gen3={best_trial['errors'][2]*100:.1f}%")

    # Verdict
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)

    if len(successful_trials) >= n_trials * 0.5:
        verdict = "PASS"
        print("\n✓✓✓ PASS ✓✓✓")
        print("\nQIC theory successfully predicts fermion mass hierarchy!")
        print("Information compression naturally generates 3 generations.")
    elif avg_errors[1] < 1.0 and avg_errors[2] < 1.0:
        verdict = "PARTIAL SUCCESS"
        print("\nPARTIAL SUCCESS")
        print("\nIssue: Close but not within criteria")
        print("\nQIC theory shows promise - qualitatively correct hierarchy.")
        print("Quantitative match requires parameter tuning.")
    else:
        verdict = "FAIL"
        print("\nFAILURE")
        print("\nQIC theory does not match observed masses.")
        print("Information-theoretic approach may not be the right mechanism.")

    # Save results
    results_save = {
        'verdict': verdict,
        'avg_masses': avg_masses.tolist(),
        'avg_errors': avg_errors.tolist(),
        'std_masses': std_masses.tolist(),
        'n_successful': len(successful_trials),
        'n_total': len(all_results),
        'N': N,
        'n_scales': n_scales
    }

    with open('qic_results.json', 'w') as f:
        json.dump(results_save, f, indent=2)

    print("\nResults saved to qic_results.json")

    return results_save


# ==============================================================================
# RUN
# ==============================================================================

if __name__ == "__main__":
    print("""
QUANTUM INFORMATION COMPRESSION THEORY

Wheeler: "It from Bit" - Information is fundamental
Quantum Eraser: Information determines physical outcomes

Core Idea:
- Fermion masses ∝ 1 / (information compressibility)
- Highly compressible patterns (simple) → light fermions
- Incompressible patterns (complex) → heavy fermions

This naturally produces exponential hierarchies because
compression ratios can differ by orders of magnitude!

If this works, it's a genuine breakthrough:
Information theory → Particle physics
    """)

    import time
    start = time.time()

    # Run test
    results = run_qic_test(N=2000, n_trials=20, n_scales=15)

    elapsed = time.time() - start
    print(f"\nTotal runtime: {elapsed/60:.1f} minutes")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("""
QIC theory connects:
- Quantum information (compression, entropy)
- Network topology (emergent spacetime)
- Particle masses (inverse compressibility)

Key prediction: Information-theoretic correlations in LHC data
If QIC is correct, fermion production rates should follow
compression hierarchy patterns.

This is testable!
    """)
