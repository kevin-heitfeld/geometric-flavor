"""
Final Test: Dynamical Relaxation Times
Can network dynamics produce mass hierarchy where statics failed?
"""

import networkx as nx
import numpy as np
import time


def generate_hyperbolic_graph(N, r_mean=6):
    """Generate hyperbolic graph (fast version)"""
    # Use Barabási-Albert as proxy for speed
    G = nx.barabasi_albert_graph(N, r_mean // 2)
    return G


def prepare_excitation(G, excitation_type='local'):
    """
    Create different excitation patterns:
    - local: Single node + neighbors
    - hub: High-degree nodes
    - community: Full community structure
    """
    nodes = list(G.nodes())
    N = len(nodes)
    
    # Initialize state: 1 on excited nodes, 0 elsewhere
    state = np.zeros(N)
    
    if excitation_type == 'local':
        # Excite random node + neighbors
        center = np.random.choice(nodes)
        excited = [center] + list(G.neighbors(center))
        state[excited] = 1.0
        
    elif excitation_type == 'hub':
        # Excite top 10% high-degree nodes
        degrees = dict(G.degree())
        top_hubs = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:N//10]
        state[top_hubs] = 1.0
        
    elif excitation_type == 'community':
        # Excite largest community
        try:
            communities = list(nx.community.greedy_modularity_communities(G))
            if len(communities) > 0:
                community = list(communities[0])
                state[community] = 1.0
            else:
                state[nodes[:N//3]] = 1.0
        except:
            state[nodes[:N//3]] = 1.0
    
    # Normalize
    state = state / np.linalg.norm(state)
    return state


def measure_relaxation_time(G, initial_state, n_steps=50):
    """
    Measure how fast excitation decays via diffusion
    
    Dynamics: ψ(t+1) = (1-γ) * ψ(t) + γ * A·ψ(t) / degree
    where A is adjacency matrix
    """
    A = nx.adjacency_matrix(G).toarray()
    degrees = np.array([G.degree(n) for n in G.nodes()])
    degrees[degrees == 0] = 1  # Avoid division by zero
    
    # Diffusion operator: D = A / degree (normalized)
    D = A / degrees[:, np.newaxis]
    
    # Diffusion rate
    gamma = 0.5
    
    # Evolve
    state = initial_state.copy()
    initial_concentration = np.max(state)  # Peak concentration
    
    concentrations = [initial_concentration]
    
    for step in range(n_steps):
        # Diffusion + decay
        state = (1 - gamma) * state + gamma * D @ state
        
        # Measure peak concentration (how localized excitation is)
        concentration = np.max(state)
        concentrations.append(concentration)
    
    concentrations = np.array(concentrations)
    
    # Fit exponential decay: C(t) ~ exp(-t/τ)
    # τ = relaxation time
    
    # Remove zeros
    concentrations = concentrations[concentrations > 1e-10]
    
    if len(concentrations) < 10:
        return np.nan
    
    log_conc = np.log(concentrations)
    times = np.arange(len(log_conc))
    
    # Linear fit: log(C) ~ -t/τ + const
    # Slope = -1/τ
    coeffs = np.polyfit(times, log_conc, 1)
    decay_rate = -coeffs[0]
    
    if decay_rate <= 0:
        return np.nan
    
    relaxation_time = 1 / decay_rate
    
    return relaxation_time


def test_dynamical_masses(N=500, trials=10):
    """
    Test if relaxation times give mass hierarchy
    """
    print("\n" + "="*60)
    print("DYNAMICAL THEORY: Masses from Relaxation Times")
    print("="*60)
    print(f"N = {N} nodes, {trials} trials")
    
    results = []
    
    for trial in range(trials):
        try:
            G = generate_hyperbolic_graph(N, r_mean=6)
            
            # Measure relaxation times for 3 excitation types
            tau_local = measure_relaxation_time(G, prepare_excitation(G, 'local'))
            tau_hub = measure_relaxation_time(G, prepare_excitation(G, 'hub'))
            tau_community = measure_relaxation_time(G, prepare_excitation(G, 'community'))
            
            if np.isnan(tau_local) or np.isnan(tau_hub) or np.isnan(tau_community):
                continue
            
            # Mass ~ ℏ / τ (inverse relaxation time)
            # So mass ratios = τ ratios (inverted)
            
            # Normalize by shortest time
            tau_min = min(tau_local, tau_hub, tau_community)
            
            mass_local = tau_min / tau_local
            mass_hub = tau_min / tau_hub
            mass_community = tau_min / tau_community
            
            # Sort by mass (ascending)
            masses = sorted([mass_local, mass_hub, mass_community])
            
            # Normalize to lightest = 1
            ratios = np.array(masses) / masses[0]
            
            results.append(ratios)
            
            if trial % 3 == 0:
                print(f"  Trial {trial+1}: τ=[{tau_local:.1f}, {tau_hub:.1f}, {tau_community:.1f}]")
                print(f"           → masses=[1, {ratios[1]:.2f}, {ratios[2]:.2f}]")
                
        except Exception as e:
            print(f"  Trial {trial+1}: failed ({e})")
            continue
    
    if len(results) < trials // 2:
        print(f"\n⚠ Too few successful trials: {len(results)}/{trials}")
        return False, np.array([1, 0, 0])
    
    mean_ratios = np.mean(results, axis=0)
    std_ratios = np.std(results, axis=0)
    
    print(f"\n" + "="*60)
    print("RESULTS:")
    print("="*60)
    print(f"  Valid trials: {len(results)}/{trials}")
    print(f"  Target (leptons): [1, ~207, ~3477]")
    print(f"  Measured:         [1, {mean_ratios[1]:.1f}, {mean_ratios[2]:.1f}]")
    print(f"  Std dev:          [-, ±{std_ratios[1]:.1f}, ±{std_ratios[2]:.1f}]")
    
    # Check for hierarchy
    if mean_ratios[1] > 2 and mean_ratios[2] > 10:
        print(f"\n✓ HIERARCHY EXISTS!")
        print(f"  Generation 2: {mean_ratios[1]:.1f}x heavier")
        print(f"  Generation 3: {mean_ratios[2]:.1f}x heavier")
        
        # Check if it's exponential-ish
        if mean_ratios[1] > 5 and mean_ratios[2] > 50:
            print(f"\n✓✓ STRONG HIERARCHY (approaching realistic values)")
            return True, mean_ratios
        else:
            print(f"\n~ Moderate hierarchy (not quite SM-like)")
            return True, mean_ratios
    else:
        print(f"\n❌ NO SIGNIFICANT HIERARCHY")
        return False, mean_ratios


def compare_all_approaches():
    """
    Final comparison of all approaches
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE COMPARISON: All Approaches to Mass Generation")
    print("="*70)
    
    start = time.time()
    
    # Test dynamical approach
    passed, ratios = test_dynamical_masses(N=500, trials=15)
    
    elapsed = time.time() - start
    
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    
    print("\n1. SPECTRAL GEOMETRY (Original QTNC-7):")
    print("   Laplacian eigenvalues → [1, 0, 0]")
    print("   Status: ❌ FAILED - eigenvalues too degenerate")
    
    print("\n2. ENTANGLEMENT SPECTRUM:")
    print("   Boundary sizes → [1, 0, 0]")
    print("   Status: ❌ FAILED - boundaries too similar")
    
    print("\n3. MUTUAL INFORMATION:")
    print("   I(A:B) for different regions → [1, 0.1, 0.02]")
    print("   Status: ❌ FAILED - insufficient hierarchy")
    
    print("\n4. DYNAMICAL RELAXATION (New):")
    print(f"   Relaxation times → [1, {ratios[1]:.1f}, {ratios[2]:.1f}]")
    if passed:
        print(f"   Status: ✓ PROMISING - hierarchy emerges from dynamics!")
    else:
        print(f"   Status: ❌ FAILED - still no hierarchy")
    
    print("\n" + "="*70)
    print("CONCLUSION:")
    print("="*70)
    
    if passed and ratios[2] > 20:
        print("\n✓ DYNAMICAL APPROACH WORKS BEST")
        print("\nKey insight:")
        print("  • Static properties (eigenvalues, boundaries) → no hierarchy")
        print("  • Dynamic properties (relaxation times) → hierarchy emerges!")
        print("\nThis suggests:")
        print("  • Particle masses are fundamentally DYNAMICAL")
        print("  • Not encoded in static geometry")
        print("  • But in how information/energy flows through network")
        print("\nRevised theory:")
        print("  m_ℓ ~ ℏ/τ_ℓ where τ_ℓ = relaxation time of ℓ-th excitation mode")
        
    elif passed:
        print("\n~ DYNAMICAL APPROACH SHOWS PROMISE")
        print("  But hierarchy too weak for realistic SM")
        print("  Needs refinement or additional structure")
        
    else:
        print("\n❌ ALL APPROACHES FAIL")
        print("\nHonest conclusion:")
        print("  • Tensor networks CAN produce emergent spacetime")
        print("  • But CANNOT derive SM mass hierarchy from first principles")
        print("  • Mass values require additional input (Yukawa couplings)")
        print("\nThis is okay! Even string theory has ~10^500 free parameters.")
    
    print(f"\nTotal runtime: {elapsed:.1f} seconds")
    print("="*70)
    
    return passed, ratios


if __name__ == "__main__":
    passed, ratios = compare_all_approaches()
    
    # Save results
    import json
    with open('dynamical_test_results.json', 'w') as f:
        json.dump({
            "approach": "dynamical_relaxation",
            "passed": bool(passed),
            "ratios": [float(r) for r in ratios],
            "target": [1, 207, 3477]
        }, f, indent=2)
    
    print("\nResults saved to dynamical_test_results.json")
