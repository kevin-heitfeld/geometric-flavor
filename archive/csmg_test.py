"""
Causal Set Mass Genesis (CSMG) - Critical Test

Tests whether causal sets produce exactly 3 stable particle pattern types
with mass ratios matching leptons (1:207:3477).

If YES → Theory worth pursuing (1-2 years)
If NO → Abandon and try something else

Runtime: ~30-60 minutes for full test
"""

import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from collections import defaultdict
import json


# ==============================================================================
# CAUSAL SET CONSTRUCTION
# ==============================================================================

class CausalSet:
    """
    A causal set is a discrete spacetime represented as a directed acyclic graph.

    Events = nodes
    Causal relations = edges (x → y means "x can causally influence y")
    """

    def __init__(self):
        self.G = nx.DiGraph()
        self.events = []
        self.time_slices = defaultdict(list)
        self.next_id = 0

    def add_event(self, time_coord, spatial_coord):
        """
        Add a new event to the causal set.

        Args:
            time_coord: Discrete time coordinate
            spatial_coord: 3D spatial position (for light cone calculations)
        """
        event_id = self.next_id
        self.next_id += 1

        self.G.add_node(event_id,
                       time=time_coord,
                       pos=spatial_coord,
                       depth=0)

        self.events.append(event_id)
        self.time_slices[time_coord].append(event_id)

        return event_id

    def add_causal_link(self, past_event, future_event):
        """Add causal relation: past_event ≺ future_event"""
        if self.G.nodes[past_event]['time'] >= self.G.nodes[future_event]['time']:
            raise ValueError("Causal link must go forward in time")

        self.G.add_edge(past_event, future_event)

    def compute_depths(self):
        """Compute causal depth of each event (# steps from earliest events)"""
        # Topological sort to compute depths
        for node in nx.topological_sort(self.G):
            predecessors = list(self.G.predecessors(node))
            if len(predecessors) == 0:
                self.G.nodes[node]['depth'] = 0
            else:
                max_pred_depth = max(self.G.nodes[p]['depth'] for p in predecessors)
                self.G.nodes[node]['depth'] = max_pred_depth + 1

    def get_past_cone(self, event, max_depth=None):
        """Get all events in the causal past of this event"""
        past = set()
        to_explore = [event]
        depth = 0

        while to_explore and (max_depth is None or depth < max_depth):
            current = to_explore.pop(0)
            predecessors = list(self.G.predecessors(current))

            for pred in predecessors:
                if pred not in past:
                    past.add(pred)
                    to_explore.append(pred)

            depth += 1

        return past

    def get_future_cone(self, event, max_depth=None):
        """Get all events in the causal future of this event"""
        future = set()
        to_explore = [event]
        depth = 0

        while to_explore and (max_depth is None or depth < max_depth):
            current = to_explore.pop(0)
            successors = list(self.G.successors(current))

            for succ in successors:
                if succ not in future:
                    future.add(succ)
                    to_explore.append(succ)

            depth += 1

        return future


def generate_causal_set(n_events=1000, spatial_dim=3, correlation_length=5.0):
    """
    Generate a causal set via sequential growth.

    Algorithm:
    1. Start with Big Bang event at t=0
    2. At each timestep, add events
    3. Create causal links based on light cone structure

    Args:
        n_events: Total number of events
        spatial_dim: Spatial dimensionality (3 for our universe)
        correlation_length: Causal correlation scale (Planck length in units)
    """
    cs = CausalSet()

    # Big Bang: single event at origin
    cs.add_event(time_coord=0, spatial_coord=np.zeros(spatial_dim))

    # Sequential growth
    events_per_time = max(1, n_events // 100)  # Grows over ~100 time steps

    for t in range(1, 100):
        if len(cs.events) >= n_events:
            break

        # Add events at this time slice
        n_new = events_per_time + np.random.poisson(np.sqrt(events_per_time))

        for _ in range(n_new):
            if len(cs.events) >= n_events:
                break

            # Random spatial position
            pos = np.random.randn(spatial_dim) * np.sqrt(t)  # Expanding universe

            new_event = cs.add_event(time_coord=t, spatial_coord=pos)

            # Create causal links to past events
            for past_event in cs.events[:-1]:  # All previous events
                past_time = cs.G.nodes[past_event]['time']
                past_pos = cs.G.nodes[past_event]['pos']

                # Check if in past light cone
                dt = t - past_time
                dr = np.linalg.norm(pos - past_pos)

                if dt > 0 and dr < dt * 1.0:  # Speed of light = 1
                    # Probability of link based on causal distance
                    causal_dist = dt - dr  # How deep in light cone
                    prob = np.exp(-causal_dist / correlation_length)

                    if np.random.rand() < prob:
                        cs.add_causal_link(past_event, new_event)

    # Compute depths
    cs.compute_depths()

    return cs


# ==============================================================================
# PATTERN IDENTIFICATION
# ==============================================================================

class CausalPattern:
    """
    A persistent pattern in the causal set (potential particle).

    Represented as a sequence of event sets across time slices.
    """

    def __init__(self, initial_events, birth_time):
        self.history = [set(initial_events)]
        self.birth_time = birth_time
        self.alive = True

    def propagate(self, cs, current_time):
        """
        Propagate pattern to next time slice by following causal links.
        """
        if not self.alive:
            return

        current_events = self.history[-1]
        next_events = set()

        # Follow causal links forward
        for event in current_events:
            future = cs.get_future_cone(event, max_depth=2)
            # Only include events at current time
            future_at_time = [e for e in future if cs.G.nodes[e]['time'] == current_time]
            next_events.update(future_at_time)

        if len(next_events) == 0:
            self.alive = False
        else:
            self.history.append(next_events)
    
    @property
    def lifetime(self):
        """Return number of time slices this pattern persists"""
        return len(self.history)

    def get_causal_depth(self, cs):
        """
        Compute average causal depth of events in this pattern.
        """
        depths = []
        for event_set in self.history:
            for event in event_set:
                depths.append(cs.G.nodes[event]['depth'])

        return np.mean(depths) if depths else 0

    def get_topology_features(self, cs):
        """
        Compute topological features of this pattern.

        Returns:
            dict with features: dimension, connectivity, depth_spread, etc.
        """
        features = {}

        # Average number of events per time slice
        avg_size = np.mean([len(s) for s in self.history])
        features['avg_size'] = avg_size

        # Causal depth statistics
        depths = []
        for event_set in self.history:
            for event in event_set:
                depths.append(cs.G.nodes[event]['depth'])

        if len(depths) > 0:
            features['mean_depth'] = np.mean(depths)
            features['std_depth'] = np.std(depths)
            features['max_depth'] = np.max(depths)
        else:
            features['mean_depth'] = 0
            features['std_depth'] = 0
            features['max_depth'] = 0

        # Causal connectivity (avg links per event)
        total_links = 0
        total_events = 0

        for event_set in self.history:
            for event in event_set:
                total_links += cs.G.in_degree(event) + cs.G.out_degree(event)
                total_events += 1

        features['avg_connectivity'] = total_links / (total_events + 1)

        # Persistence (how long pattern survives)
        features['lifetime'] = len(self.history)

        # Causal dimension (how pattern scales with depth)
        if len(depths) > 10:
            # Group by depth and count
            depth_counts = defaultdict(int)
            for d in depths:
                depth_counts[d] += 1

            # Fit power law N(d) ~ d^dim
            if len(depth_counts) > 3:
                ds = sorted(depth_counts.keys())
                ns = [depth_counts[d] for d in ds]

                # Log-log fit
                log_ds = np.log(np.array(ds) + 1)
                log_ns = np.log(np.array(ns) + 1)

                if np.std(log_ds) > 0:
                    coeffs = np.polyfit(log_ds, log_ns, 1)
                    features['causal_dimension'] = coeffs[0]
                else:
                    features['causal_dimension'] = 0
            else:
                features['causal_dimension'] = 0
        else:
            features['causal_dimension'] = 0

        return features


def identify_patterns(cs, min_lifetime=5):
    """
    Identify persistent patterns in the causal set.

    Patterns = connected components that persist across time slices.
    """
    patterns = []

    # Start patterns at early times
    for start_time in range(10, 50):
        if start_time not in cs.time_slices:
            continue

        events_at_time = cs.time_slices[start_time]

        # Each event could be start of a pattern
        # But we want spatially coherent groups
        for event in events_at_time:
            # Get nearby events (causally connected)
            nearby = {event}

            # Add causally related events at same time
            for other in events_at_time:
                if other == event:
                    continue

                # Check if they share causal past/future
                past_event = cs.get_past_cone(event, max_depth=3)
                past_other = cs.get_past_cone(other, max_depth=3)

                if len(past_event & past_other) > 0:
                    nearby.add(other)

            # Create pattern
            pattern = CausalPattern(nearby, start_time)

            # Propagate forward
            for t in range(start_time + 1, 90):
                pattern.propagate(cs, t)

                if not pattern.alive:
                    break

            # Keep if long-lived
            if pattern.lifetime >= min_lifetime:
                patterns.append(pattern)

    return patterns


# ==============================================================================
# PATTERN CLASSIFICATION
# ==============================================================================

def classify_patterns(patterns, cs):
    """
    Cluster patterns into types based on topology.

    Question: Do we get exactly 3 types?
    """
    if len(patterns) < 5:
        return None, None

    # Extract features
    feature_vectors = []

    for pattern in patterns:
        features = pattern.get_topology_features(cs)

        # Convert to vector
        vec = [
            features['mean_depth'],
            features['std_depth'],
            features['max_depth'],
            features['avg_connectivity'],
            features['causal_dimension'],
            features['avg_size'],
            features['lifetime']
        ]

        feature_vectors.append(vec)

    feature_vectors = np.array(feature_vectors)

    # Remove NaN/inf
    feature_vectors = np.nan_to_num(feature_vectors, nan=0.0, posinf=1.0, neginf=0.0)

    # Normalize
    scaler = StandardScaler()
    features_norm = scaler.fit_transform(feature_vectors)

    # Try different numbers of clusters
    results = {}
    max_clusters = min(len(patterns) - 1, 7)

    for n_clusters in range(2, max_clusters + 1):
        if n_clusters >= len(patterns):
            break

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        labels = kmeans.fit_predict(features_norm)

        # Silhouette score
        if len(set(labels)) > 1 and len(set(labels)) < len(patterns):
            try:
                score = silhouette_score(features_norm, labels)
                results[n_clusters] = {
                    'labels': labels,
                    'score': score,
                    'centers': kmeans.cluster_centers_
                }
            except:
                pass

    return feature_vectors, results


# ==============================================================================
# MASS CALCULATION
# ==============================================================================

def compute_mass_ratios(patterns, cs, alpha=1.0, beta=1.0):
    """
    Compute mass ratios from causal depth.

    Formula: m = m0 * exp(alpha * depth^beta)
    """
    masses = []

    for pattern in patterns:
        depth = pattern.get_causal_depth(cs)
        mass = np.exp(alpha * (depth ** beta))
        masses.append(mass)

    return np.array(masses)


# ==============================================================================
# MAIN TEST
# ==============================================================================

def run_critical_test(n_events=2000, n_trials=20, alpha=1.0, beta=0.5):
    """
    THE CRITICAL TEST

    Question: Do we get exactly 3 stable pattern types?

    Args:
        n_events: Size of each causal set
        n_trials: Number of causal sets to generate
        alpha, beta: Mass formula parameters (will be fitted)

    Returns:
        results dictionary with verdict
    """
    print("="*70)
    print("CAUSAL SET MASS GENESIS - CRITICAL TEST")
    print("="*70)
    print(f"\nParameters:")
    print(f"  Causal set size: N = {n_events} events")
    print(f"  Number of trials: {n_trials}")
    print(f"  Mass formula: m ∝ exp(α × depth^β)")
    print(f"\nQuestion: Do we get exactly 3 stable pattern types?")
    print(f"Expected mass ratios: ~1:200:3000 (like e:μ:τ)")
    print("="*70)

    all_patterns = []
    all_causal_sets = []

    for trial in range(n_trials):
        print(f"\n--- Trial {trial + 1}/{n_trials} ---")

        # Generate causal set
        cs = generate_causal_set(n_events=n_events, spatial_dim=3, correlation_length=5.0)
        print(f"Generated causal set: {len(cs.events)} events, {cs.G.number_of_edges()} causal links")

        # Identify patterns
        patterns = identify_patterns(cs, min_lifetime=5)
        print(f"Found {len(patterns)} persistent patterns")

        if len(patterns) > 0:
            all_patterns.extend(patterns)
            all_causal_sets.extend([cs] * len(patterns))

    print(f"\n{'='*70}")
    print(f"ANALYZING PATTERNS")
    print("="*70)
    print(f"Total patterns found: {len(all_patterns)}")

    if len(all_patterns) < 10:
        print("\n❌ FAILED: Too few patterns found")
        return {'verdict': 'FAILED', 'reason': 'insufficient_patterns'}

    # Classify patterns
    feature_vectors, clustering_results = classify_patterns(all_patterns, all_causal_sets[0])

    if clustering_results is None or len(clustering_results) == 0:
        print("\n❌ FAILED: Could not cluster patterns")
        return {'verdict': 'FAILED', 'reason': 'clustering_failed'}

    # Find best clustering
    print("\nClustering quality for different numbers of pattern types:")
    for n_clusters, result in clustering_results.items():
        print(f"  {n_clusters} patterns: silhouette score = {result['score']:.3f}")

    best_n_clusters = max(clustering_results.items(), key=lambda x: x[1]['score'])[0]
    best_clustering = clustering_results[best_n_clusters]

    print(f"\nOptimal number of pattern types: {best_n_clusters}")

    # Compute mass ratios for each cluster
    print("\nComputing mass ratios...")

    labels = best_clustering['labels']
    masses_by_cluster = defaultdict(list)
    depths_by_cluster = defaultdict(list)

    for i, (pattern, cs) in enumerate(zip(all_patterns, all_causal_sets)):
        if i >= len(labels):
            break

        cluster_id = labels[i]
        depth = pattern.get_causal_depth(cs)
        mass = np.exp(alpha * (depth ** beta))

        masses_by_cluster[cluster_id].append(mass)
        depths_by_cluster[cluster_id].append(depth)

    # Average by cluster
    cluster_depths = []
    cluster_masses = []

    for cluster_id in sorted(masses_by_cluster.keys()):
        avg_depth = np.mean(depths_by_cluster[cluster_id])
        avg_mass = np.mean(masses_by_cluster[cluster_id])
        std_mass = np.std(masses_by_cluster[cluster_id])

        cluster_depths.append(avg_depth)
        cluster_masses.append(avg_mass)

        print(f"  Type {cluster_id}: depth={avg_depth:.1f}, mass={avg_mass:.2f}±{std_mass:.2f} ({len(masses_by_cluster[cluster_id])} patterns)")

    # Sort by mass
    sorted_indices = np.argsort(cluster_masses)
    cluster_masses = np.array(cluster_masses)[sorted_indices]

    # Normalize to lightest = 1
    mass_ratios = cluster_masses / cluster_masses[0]

    print(f"\nPredicted mass ratios: {mass_ratios}")
    print(f"Target (leptons):      [1, 207, 3477]")

    # Compute errors
    target = np.array([1, 207, 3477])

    if len(mass_ratios) >= 2:
        error_gen2 = abs(mass_ratios[1] - target[1]) / target[1] if len(mass_ratios) > 1 else 1.0
        error_gen3 = abs(mass_ratios[2] - target[2]) / target[2] if len(mass_ratios) > 2 else 1.0

        print(f"\nErrors: Gen2 = {error_gen2*100:.1f}%, Gen3 = {error_gen3*100:.1f}%")
    else:
        error_gen2 = 1.0
        error_gen3 = 1.0

    # VERDICT
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)

    success_criteria = {
        'exactly_3_patterns': best_n_clusters == 3,
        'good_clustering': best_clustering['score'] > 0.25,
        'error_gen2_ok': error_gen2 < 0.5 if error_gen2 < 10 else False,
        'error_gen3_ok': error_gen3 < 1.0 if error_gen3 < 10 else False,
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
Causal Set Mass Genesis shows promise!

Found:
1. Exactly 3 stable pattern types ✓
2. Good clustering separation ✓
3. Mass ratios in right ballpark ✓

RECOMMENDATION: Pursue this theory further

Next steps:
1. Refine mass formula (optimize α and β)
2. Mathematical proof that 3 types are generic
3. Calculate mixing angles from pattern overlaps
4. Make predictions for neutrinos and quarks
5. Develop full quantum formulation

Estimated time to predictions: 1-2 years
Probability of success: ~40-50%
        """)
    elif success_criteria['exactly_3_patterns']:
        print("~ PARTIAL SUCCESS")
        print("="*70)
        print("""
Found 3 pattern types but quantitative predictions off.

This means:
✓ Core mechanism works (3 types emerge)
✓ Causal depth creates hierarchy
❌ But mass formula needs work

RECOMMENDATION: Continue with modifications

Possible fixes:
1. Different mass formula (not just exp(α d^β))
2. Include more topology features
3. Better pattern identification algorithm
4. Quantum corrections

Estimated time to fix: 6 months - 1 year
Probability of success: ~25%
        """)
    else:
        print("❌ TEST FAILED")
        print("="*70)
        print(f"""
Theory does not work as formulated!

Found {best_n_clusters} stable patterns, not 3.

This means causal set dynamics don't naturally produce
the right structure for 3 generations.

RECOMMENDATION: Major rework or abandon

Either:
1. Prove mathematically why 3 types should exist
2. Modify causal set growth rules significantly
3. Or: Accept theory is wrong and try different approach

Do NOT invest years without fixing fundamental issue.

Probability of success: <10%
        """)

    # Save results
    results = {
        'n_clusters': int(best_n_clusters),
        'silhouette_score': float(best_clustering['score']),
        'success_criteria': {k: bool(v) for k, v in success_criteria.items()},
        'all_pass': bool(all_pass),
        'mass_ratios': [float(x) for x in mass_ratios] if len(mass_ratios) > 0 else None,
        'error_gen2': float(error_gen2),
        'error_gen3': float(error_gen3),
        'n_patterns_found': len(all_patterns)
    }

    with open('csmg_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to csmg_test_results.json")

    return results


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def visualize_causal_set(cs, max_events=200):
    """Visualize a causal set"""
    print("Generating causal set visualization...")

    # Limit to first max_events for clarity
    events_to_show = list(cs.events)[:max_events]
    subgraph = cs.G.subgraph(events_to_show)

    # Layout by time
    pos = {}
    for node in subgraph.nodes():
        t = cs.G.nodes[node]['time']
        spatial_pos = cs.G.nodes[node]['pos']

        # 2D projection
        x = spatial_pos[0] if len(spatial_pos) > 0 else 0
        y = t * 2  # Stretch time axis

        pos[node] = (x, y)

    plt.figure(figsize=(12, 10))

    # Color by depth
    depths = [cs.G.nodes[node]['depth'] for node in subgraph.nodes()]

    nodes = nx.draw_networkx_nodes(subgraph, pos, node_color=depths, node_size=20,
                                    cmap='viridis', alpha=0.6)
    nx.draw_networkx_edges(subgraph, pos, edge_color='gray', alpha=0.6,
                          arrows=True, arrowsize=5)

    plt.title(f'Causal Set ({len(subgraph.nodes())} events shown)')
    plt.xlabel('Spatial dimension')
    plt.ylabel('Time →')
    plt.colorbar(nodes, label='Causal Depth')
    plt.tight_layout()
    plt.savefig('csmg_causal_set.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to csmg_causal_set.png")
    plt.close()


# ==============================================================================
# RUN
# ==============================================================================

if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║  CAUSAL SET MASS GENESIS (CSMG) - CRITICAL TEST                  ║
║                                                                    ║
║  Tests if causal sets naturally produce 3 particle types          ║
║  Runtime: ~30-60 minutes                                          ║
║                                                                    ║
║  If test passes → Invest 1-2 years                                ║
║  If test fails → Try different approach                           ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
    """)

    import time
    start = time.time()

    # Generate and visualize one causal set
    print("\n[1/2] Generating example causal set...")
    cs_example = generate_causal_set(n_events=500, spatial_dim=3)
    visualize_causal_set(cs_example, max_events=200)

    # Run full test
    print("\n[2/2] Running critical test...")
    results = run_critical_test(n_events=2000, n_trials=20, alpha=0.8, beta=0.6)

    elapsed = time.time() - start
    print(f"\nTotal runtime: {elapsed/60:.1f} minutes")

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    print("\nScience demands ruthless falsification.")
    print("The data has spoken. Follow where it leads.")
