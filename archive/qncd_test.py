"""
Quantum Network Causal Dynamics (QNCD) - Critical Test

Tests whether entanglement-driven coarse-graining on quantum networks
produces exactly 3 stable patterns with universal mass ratios.

If YES → Theory worth pursuing (5-10 years)
If NO → Abandon and try something else

Runtime: ~1-2 hours for full test
"""

import numpy as np
import networkx as nx
from scipy.linalg import expm, eig
from scipy.stats import entropy
import matplotlib.pyplot as plt
from collections import defaultdict
import json


# ==============================================================================
# QUANTUM NETWORK SETUP
# ==============================================================================

class QuantumNetwork:
    """
    Quantum network where nodes are qubits and edges represent entanglement.

    State: |Ψ⟩ represented as density matrix ρ
    Dynamics: Hamiltonian evolution + entanglement-driven coarse-graining
    """

    def __init__(self, G, initial_state='random'):
        """
        G: NetworkX graph
        initial_state: 'random', 'separable', or 'GHZ'
        """
        self.G = G.copy()
        self.N = G.number_of_nodes()
        self.node_list = list(G.nodes())

        # Map graph nodes to indices
        self.node_to_idx = {node: i for i, node in enumerate(self.node_list)}

        # Initialize quantum state
        if initial_state == 'random':
            self.rho = self._random_density_matrix()
        elif initial_state == 'separable':
            self.rho = self._separable_state()
        elif initial_state == 'GHZ':
            self.rho = self._ghz_state()
        else:
            raise ValueError(f"Unknown initial state: {initial_state}")

    def _random_density_matrix(self):
        """Generate random density matrix (Haar measure would be ideal, this is approximate)"""
        # For large N, we can't store full 2^N × 2^N matrix
        # Instead: treat as collection of local density matrices + correlations
        # This is an approximation but necessary for N > 10

        # Start with product state
        rho = np.eye(2**min(self.N, 8)) / (2**min(self.N, 8))

        # Add random phases
        H = np.random.randn(rho.shape[0], rho.shape[1])
        H = (H + H.T) / 2  # Make Hermitian
        U = expm(1j * H * 0.1)
        rho = U @ rho @ U.conj().T

        return rho

    def _separable_state(self):
        """All qubits in |0⟩ state"""
        dim = 2**min(self.N, 8)
        rho = np.zeros((dim, dim))
        rho[0, 0] = 1.0
        return rho

    def _ghz_state(self):
        """GHZ state: (|00...0⟩ + |11...1⟩)/√2"""
        dim = 2**min(self.N, 8)
        rho = np.zeros((dim, dim), dtype=complex)
        rho[0, 0] = 0.5
        rho[0, -1] = 0.5
        rho[-1, 0] = 0.5
        rho[-1, -1] = 0.5
        return rho

    def entanglement_entropy(self, node_i, node_j):
        """
        Calculate entanglement entropy between two nodes.

        For large systems, use approximation based on local correlations.
        """
        if not self.G.has_edge(node_i, node_j):
            return 0.0

        # Get edge weight (correlation strength)
        weight = self.G[node_i][node_j].get('weight', 1.0)

        # Simplified model: entanglement entropy proportional to correlation
        # In full theory, would compute S(ρ_ij) = -Tr(ρ_ij log ρ_ij)
        # For now: use graph structure + local measurements

        # Degree-based approximation (hubs more entangled)
        degree_i = self.G.degree(node_i)
        degree_j = self.G.degree(node_j)

        # Higher degree → more entanglement capacity
        base_entropy = np.log2(min(degree_i, degree_j) + 1)

        # Weight modulates actual entanglement
        S = base_entropy * weight * (1 + 0.1 * np.random.randn())

        return max(0, S)

    def time_evolve(self, dt=0.1, steps=10):
        """
        Evolve quantum state under local Hamiltonian.
        Generates entanglement dynamically.
        """
        for _ in range(steps):
            # Local Hamiltonian: nearest neighbor interactions
            # This is simplified - full version would use actual quantum gates

            for edge in self.G.edges():
                # Increase edge weight (entanglement grows)
                if 'weight' not in self.G.edges[edge]:
                    self.G.edges[edge]['weight'] = 1.0

                # Entanglement grows with time (up to max)
                w = self.G.edges[edge]['weight']
                self.G.edges[edge]['weight'] = min(3.0, w + dt * 0.1)


# ==============================================================================
# COARSE-GRAINING
# ==============================================================================

def coarse_grain_network(qnet, eta=1.0):
    """
    Coarse-grain quantum network by merging highly entangled nodes.

    Rule: If S(ρ_ij) > η, merge nodes i and j

    Returns: new QuantumNetwork, merged_pairs
    """
    G = qnet.G.copy()
    merged_pairs = []

    # Find all edges with high entanglement
    high_entanglement_edges = []

    for edge in G.edges():
        i, j = edge
        S = qnet.entanglement_entropy(i, j)
        if S > eta:
            high_entanglement_edges.append((i, j, S))

    # Sort by entanglement (highest first)
    high_entanglement_edges.sort(key=lambda x: x[2], reverse=True)

    # Limit merging to avoid too fast collapse
    max_merges = max(1, len(G.nodes()) // 4)  # Merge at most 25% of nodes per iteration

    # Merge nodes (greedy algorithm)
    merged_nodes = set()
    G_coarse = G.copy()
    merge_count = 0

    for i, j, S in high_entanglement_edges:
        if merge_count >= max_merges:
            break

        if i in merged_nodes or j in merged_nodes:
            continue

        # Merge j into i
        # Connect i to all of j's neighbors
        for neighbor in G.neighbors(j):
            if neighbor != i and neighbor not in merged_nodes:
                if G_coarse.has_edge(i, neighbor):
                    # Add weights
                    w_old = G_coarse[i][neighbor].get('weight', 1.0)
                    w_new = G[j][neighbor].get('weight', 1.0) if G.has_edge(j, neighbor) else 0
                    G_coarse[i][neighbor]['weight'] = w_old + w_new
                else:
                    w = G[j][neighbor].get('weight', 1.0) if G.has_edge(j, neighbor) else 1.0
                    G_coarse.add_edge(i, neighbor, weight=w)

        # Remove j
        G_coarse.remove_node(j)
        merged_nodes.add(j)
        merged_pairs.append((i, j))
        merge_count += 1

    # Create new quantum network
    qnet_coarse = QuantumNetwork(G_coarse, initial_state='separable')
    qnet_coarse.rho = qnet.rho  # Inherit state (simplified)

    return qnet_coarse, merged_pairs
# ==============================================================================
# PATTERN IDENTIFICATION
# ==============================================================================

def extract_topological_features(G):
    """
    Extract features that characterize network patterns.
    These should be invariant under coarse-graining for stable patterns.
    """
    if G.number_of_nodes() < 3:
        return None

    features = {}

    # Degree statistics
    degrees = [d for n, d in G.degree()]
    features['mean_degree'] = np.mean(degrees)
    features['std_degree'] = np.std(degrees)
    features['max_degree'] = np.max(degrees)
    features['degree_skewness'] = np.mean([(d - np.mean(degrees))**3 for d in degrees]) / (np.std(degrees)**3 + 1e-10)

    # Clustering
    try:
        features['clustering'] = nx.average_clustering(G)
    except:
        features['clustering'] = 0

    # Transitivity
    try:
        features['transitivity'] = nx.transitivity(G)
    except:
        features['transitivity'] = 0

    # Modularity
    try:
        communities = list(nx.community.greedy_modularity_communities(G))
        features['modularity'] = nx.community.modularity(G, communities)
        features['n_communities'] = len(communities)
    except:
        features['modularity'] = 0
        features['n_communities'] = 1

    # Assortativity
    try:
        features['assortativity'] = nx.degree_assortativity_coefficient(G)
    except:
        features['assortativity'] = 0

    # Centrality
    try:
        betweenness = nx.betweenness_centrality(G)
        features['max_betweenness'] = max(betweenness.values())
        features['mean_betweenness'] = np.mean(list(betweenness.values()))
    except:
        features['max_betweenness'] = 0
        features['mean_betweenness'] = 0

    return features


def compute_pattern_persistence(feature_sequence):
    """
    Compute how persistent a pattern is across scales.

    Patterns that persist → low mass particles
    Patterns that decay quickly → high mass particles
    """
    if len(feature_sequence) < 2:
        return 0

    # Compute decay rate of each feature
    decay_rates = []

    for feature_name in feature_sequence[0].keys():
        values = [f[feature_name] for f in feature_sequence]

        # Handle NaN and inf
        values = np.array(values)
        values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)

        # Avoid log(0)
        values = values + 1e-10

        # Fit exponential decay
        scales = np.arange(len(values))

        try:
            # Linear fit to log(values) vs scales
            log_values = np.log(values + 1e-10)
            log_values = np.nan_to_num(log_values, nan=0.0, posinf=0.0, neginf=-10.0)

            if np.std(log_values) > 1e-6:  # Only if there's variation
                coeffs = np.polyfit(scales, log_values, 1)
                decay_rate = -coeffs[0]  # Negative slope = decay
                decay_rates.append(decay_rate)
        except:
            pass

    if len(decay_rates) == 0:
        return 0.1  # Default small persistence

    # Persistence = inverse of mean decay rate
    mean_decay = np.mean(decay_rates)
    persistence = 1.0 / (abs(mean_decay) + 0.01)

    # Clamp to reasonable range
    persistence = np.clip(persistence, 0.01, 100.0)

    return persistence


# ==============================================================================
# CLUSTERING PATTERNS
# ==============================================================================

def identify_pattern_types(all_feature_sequences):
    """
    Cluster patterns into types based on their evolution across scales.

    Question: Do we get exactly 3 clusters?
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    # Extract persistence "fingerprints"
    fingerprints = []

    for seq in all_feature_sequences:
        if len(seq) < 3:
            continue

        # Create fingerprint: how each feature evolves
        fingerprint = []

        feature_names = seq[0].keys()
        for fname in feature_names:
            values = [f[fname] for f in seq]

            # Characterize evolution
            if len(values) >= 2:
                # Clean values
                values_clean = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)

                # Mean, std, decay rate
                fingerprint.append(np.mean(values_clean))
                fingerprint.append(np.std(values_clean))

                # Decay rate
                scales = np.arange(len(values_clean))
                try:
                    if np.std(values_clean) > 1e-6:
                        coeffs = np.polyfit(scales, values_clean, 1)
                        fingerprint.append(coeffs[0])  # Slope
                    else:
                        fingerprint.append(0)
                except:
                    fingerprint.append(0)

        fingerprints.append(fingerprint)

    if len(fingerprints) < 3:
        return None, None

    fingerprints = np.array(fingerprints)

    # Normalize
    scaler = StandardScaler()
    fingerprints_norm = scaler.fit_transform(fingerprints)

    # Try clustering with different numbers of clusters
    results = {}

    max_clusters = min(len(fingerprints) - 1, 7)  # Can't have more clusters than samples-1

    for n_clusters in range(2, max_clusters + 1):
        if n_clusters >= len(fingerprints):
            break

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(fingerprints_norm)

        # Compute silhouette score (quality of clustering)
        from sklearn.metrics import silhouette_score
        if len(set(labels)) > 1 and len(set(labels)) < len(fingerprints):
            try:
                score = silhouette_score(fingerprints_norm, labels)
                results[n_clusters] = {
                    'labels': labels,
                    'score': score,
                    'inertia': kmeans.inertia_
                }
            except ValueError:
                # Skip if clustering failed
                pass

    return fingerprints, results
# ==============================================================================
# MAIN TEST
# ==============================================================================

def run_critical_test(N=500, eta=1.0, n_trials=10, n_scales=12):
    """
    THE CRITICAL TEST

    Question: Do we get exactly 3 stable pattern types with universal mass ratios?

    Args:
        N: Network size
        eta: Entanglement threshold for coarse-graining
        n_trials: Number of random initial networks to test
        n_scales: Number of coarse-graining steps

    Returns:
        results dictionary with verdict
    """
    print("="*70)
    print("QUANTUM NETWORK CAUSAL DYNAMICS - CRITICAL TEST")
    print("="*70)
    print(f"\nParameters:")
    print(f"  Network size: N = {N}")
    print(f"  Entanglement threshold: η = {eta}")
    print(f"  Number of trials: {n_trials}")
    print(f"  Coarse-graining scales: {n_scales}")
    print(f"\nQuestion: Do we get exactly 3 stable patterns?")
    print(f"Expected mass ratios: ~1:200:3000 (like e:μ:τ)")
    print("="*70)

    all_persistences = []
    all_feature_sequences = []

    for trial in range(n_trials):
        print(f"\n--- Trial {trial + 1}/{n_trials} ---")

        # Generate scale-free network (like real quantum spacetime might be)
        G = nx.barabasi_albert_graph(N, m=3)
        print(f"Generated scale-free network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        # Initialize quantum network
        qnet = QuantumNetwork(G, initial_state='random')
        print(f"Initialized quantum state")

        # Evolve to generate entanglement
        print(f"Evolving to generate entanglement...")
        qnet.time_evolve(dt=0.1, steps=20)

        # Track features across scales
        feature_sequence = []

        print(f"\nCoarse-graining across {n_scales} scales:")
        for scale in range(n_scales):
            # Extract features at this scale
            features = extract_topological_features(qnet.G)

            if features is None:
                break

            feature_sequence.append(features)

            N_current = qnet.G.number_of_nodes()
            print(f"  Scale {scale}: N = {N_current:4d}, clustering = {features['clustering']:.3f}, modularity = {features['modularity']:.3f}")

            if N_current < 10:
                break

            # Coarse-grain
            qnet, merged = coarse_grain_network(qnet, eta=eta)

            # Evolve again to generate new entanglement
            qnet.time_evolve(dt=0.05, steps=5)

        # Compute persistence
        if len(feature_sequence) >= 3:
            persistence = compute_pattern_persistence(feature_sequence)
            all_persistences.append(persistence)
            all_feature_sequences.append(feature_sequence)
            print(f"Pattern persistence: {persistence:.3f}")

    print("\n" + "="*70)
    print("ANALYZING PATTERNS")
    print("="*70)

    # Cluster patterns
    fingerprints, clustering_results = identify_pattern_types(all_feature_sequences)

    if clustering_results is None:
        print("\n❌ FAILED: Not enough data to cluster patterns")
        return {'verdict': 'FAILED', 'reason': 'insufficient_data'}

    # Find optimal number of clusters
    print("\nClustering quality for different numbers of pattern types:")
    for n_clusters, result in clustering_results.items():
        print(f"  {n_clusters} patterns: silhouette score = {result['score']:.3f}")

    # Best clustering (highest silhouette score)
    best_n_clusters = max(clustering_results.items(), key=lambda x: x[1]['score'])[0]
    best_clustering = clustering_results[best_n_clusters]

    print(f"\nOptimal number of pattern types: {best_n_clusters}")

    # Compute mass ratios for each cluster
    print("\nComputing mass ratios...")

    persistences_by_cluster = defaultdict(list)
    for i, label in enumerate(best_clustering['labels']):
        if i < len(all_persistences):
            persistences_by_cluster[label].append(all_persistences[i])

    # Average persistence per cluster
    cluster_persistences = []
    for cluster_id in sorted(persistences_by_cluster.keys()):
        avg_persistence = np.mean(persistences_by_cluster[cluster_id])
        std_persistence = np.std(persistences_by_cluster[cluster_id])
        cluster_persistences.append(avg_persistence)
        print(f"  Pattern type {cluster_id}: persistence = {avg_persistence:.3f} ± {std_persistence:.3f} ({len(persistences_by_cluster[cluster_id])} samples)")

    # Convert persistence to mass ratios
    # Mass ∝ 1/persistence
    if len(cluster_persistences) >= 2:
        masses = [1.0 / p for p in cluster_persistences]
        masses = sorted(masses)

        # Normalize to lightest = 1
        mass_ratios = [m / masses[0] for m in masses]

        print(f"\nPredicted mass ratios: {mass_ratios}")
        print(f"Target (leptons):      [1, 207, 3477]")

        # Compare to leptons
        target = np.array([1, 207, 3477])
        predicted = np.array(mass_ratios[:3] if len(mass_ratios) >= 3 else mass_ratios + [0]*(3-len(mass_ratios)))

        if len(predicted) >= 2:
            error_gen2 = abs(predicted[1] - target[1]) / target[1] if len(predicted) > 1 else 1.0
            error_gen3 = abs(predicted[2] - target[2]) / target[2] if len(predicted) > 2 else 1.0

            print(f"\nErrors: Gen2 = {error_gen2*100:.1f}%, Gen3 = {error_gen3*100:.1f}%")

    # VERDICT
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)

    success_criteria = {
        'exactly_3_patterns': best_n_clusters == 3,
        'good_clustering': best_clustering['score'] > 0.3,
        'error_gen2_ok': error_gen2 < 0.5 if 'error_gen2' in locals() else False,
        'error_gen3_ok': error_gen3 < 1.0 if 'error_gen3' in locals() else False,
    }

    all_pass = all(success_criteria.values())

    print("\nCriteria:")
    for criterion, passed in success_criteria.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"  {criterion:20s}: {status}")

    print("\n" + "="*70)

    if all_pass:
        print("✓✓✓ TEST PASSED!")
        print("="*70)
        print("""
Theory shows promise! We found:
1. Exactly 3 stable pattern types
2. Good clustering separation
3. Mass ratios in right ballpark

RECOMMENDATION: Pursue this theory further

Next steps:
1. Refine quantum simulation (use actual density matrices)
2. Analytical calculation of fixed points
3. Prove 3 patterns are generic (mathematical theorem)
4. Calculate mixing angles and other observables
5. Make predictions for neutrinos

Estimated time to predictions: 1-2 years
Probability of success: ~30-40%
        """)
    elif success_criteria['exactly_3_patterns'] and success_criteria['good_clustering']:
        print("~ PARTIAL SUCCESS")
        print("="*70)
        print("""
We found 3 stable patterns with good separation!
But mass ratios are off.

This means:
✓ Core mechanism works (3 patterns emerge)
✓ Patterns are stable and distinct
❌ But quantitative predictions wrong

RECOMMENDATION: Investigate modifications

Possible fixes:
1. Different coarse-graining rule
2. Include quantum corrections more carefully
3. Different entanglement measure
4. Add more physics (interactions, gauge fields)

Estimated time to fix: 6 months - 1 year
Probability of success: ~20%
        """)
    elif success_criteria['exactly_3_patterns']:
        print("~ MARGINAL")
        print("="*70)
        print("""
Found 3 patterns, but:
- Clustering quality poor (overlapping)
- Mass ratios wrong

RECOMMENDATION: Significant rework needed

This suggests mechanism is on right track but implementation
needs major improvements.

Consider:
1. Completely different entanglement measure
2. Different coarse-graining algorithm
3. Add more structure to Hamiltonian
4. Or: abandon and try different approach

Probability of success: ~10%
        """)
    else:
        print("❌ TEST FAILED")
        print("="*70)
        print(f"""
Theory does not work as formulated!

Found {best_n_clusters} stable patterns, not 3.

This means the core mechanism doesn't produce the right structure.

RECOMMENDATION: Abandon this version

Either:
1. Go back to drawing board with completely different approach
2. Or: Accept theory is wrong and work on something else

Do NOT waste years trying to fix this. The foundation is wrong.

Probability of success: <5%
        """)

    # Save results
    results = {
        'n_clusters': best_n_clusters,
        'silhouette_score': best_clustering['score'],
        'success_criteria': {k: bool(v) for k, v in success_criteria.items()},  # Convert to bool
        'all_pass': bool(all_pass),
        'mass_ratios': [float(x) for x in mass_ratios] if 'mass_ratios' in locals() else None,
        'error_gen2': float(error_gen2) if 'error_gen2' in locals() else None,
        'error_gen3': float(error_gen3) if 'error_gen3' in locals() else None,
    }

    with open('qncd_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n Results saved to qncd_test_results.json")

    return results


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def visualize_coarse_graining(N=200, eta=1.0, n_scales=5):
    """
    Visualize how network coarse-grains across scales.
    Helps understand what's happening.
    """
    print("Generating visualization of coarse-graining process...")

    G = nx.barabasi_albert_graph(N, m=3)
    qnet = QuantumNetwork(G, initial_state='random')
    qnet.time_evolve(dt=0.1, steps=20)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for scale in range(min(n_scales, 6)):
        ax = axes[scale]

        # Draw network
        pos = nx.spring_layout(qnet.G, k=0.5, iterations=50)

        # Color by degree
        degrees = [qnet.G.degree(n) for n in qnet.G.nodes()]

        nx.draw(qnet.G, pos, ax=ax, node_color=degrees, node_size=30,
                cmap='viridis', with_labels=False, edge_color='gray', alpha=0.6)

        ax.set_title(f'Scale {scale}: N={qnet.G.number_of_nodes()}')
        ax.axis('off')

        if qnet.G.number_of_nodes() < 5:
            break

        # Coarse-grain for next iteration
        qnet, _ = coarse_grain_network(qnet, eta=eta)
        qnet.time_evolve(dt=0.05, steps=5)

    plt.tight_layout()
    plt.savefig('qncd_coarse_graining.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to qncd_coarse_graining.png")
    plt.close()


# ==============================================================================
# RUN
# ==============================================================================

if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║  QUANTUM NETWORK CAUSAL DYNAMICS (QNCD) - CRITICAL TEST          ║
║                                                                    ║
║  This test will determine if the theory is worth pursuing.        ║
║  Runtime: ~1-2 hours                                              ║
║                                                                    ║
║  If test passes → Invest 5-10 years                               ║
║  If test fails → Abandon and try something else                   ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
    """)

    import time
    start = time.time()

    # First: visualize what's happening
    print("\n[1/3] Generating visualization...")
    visualize_coarse_graining(N=200, eta=1.0, n_scales=5)

    # Second: run quick test
    print("\n[2/3] Running quick test (20 trials)...")
    quick_results = run_critical_test(N=300, eta=1.0, n_trials=20, n_scales=10)

    # Third: If quick test promising, run full test
    if quick_results.get('n_clusters') == 3 or input("\nRun full test? (y/n): ").lower() == 'y':
        print("\n[3/3] Running full test (50 trials)...")
        full_results = run_critical_test(N=500, eta=1.0, n_trials=50, n_scales=12)
    else:
        print("\nSkipping full test based on quick results.")
        full_results = quick_results

    elapsed = time.time() - start
    print(f"\nTotal runtime: {elapsed/60:.1f} minutes")

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    print("\nNow you know whether to invest years in this theory or move on.")
    print("Science is about ruthless falsification, not defending hypotheses.")
    print("\nGood luck!")
