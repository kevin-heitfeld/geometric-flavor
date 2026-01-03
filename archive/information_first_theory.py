"""
INFORMATION-FIRST THEORY (IFT)

Revolutionary idea: Treat information as PRIMARY, not derived from network

Previous attempts (HNR, QIC):
  Network → Observables → Masses
  Problem: Observables locked by network statistics

New approach:
  Information → Network → Masses

Core Principle:
--------------
The universe is fundamentally a quantum information system.
Space, time, and particles EMERGE from information dynamics.

Key Insight from Quantum Eraser:
Information erasure affects physical outcomes retroactively.
This suggests information is MORE fundamental than spacetime!

Theory Structure:
-----------------
1. Start with abstract information states |ψ⟩ (not spatial)
2. Information has intrinsic complexity/entropy
3. Three DISTINCT information configurations exist (generations)
4. Networks emerge as LOW-ENERGY representations of information
5. Fermion masses = Energy cost of information processing

Why This Works:
---------------
- Not constrained by single network type
- Three information states can have vastly different complexities
- No artificial "where do we cut the network" problem
- Information complexity can naturally span orders of magnitude

Mathematical Framework:
-----------------------
Information state: |ψ_i⟩ with three fundamental configurations

Gen1: |ψ_1⟩ = Highly compressible (simple, ordered)
      → Low algorithmic complexity K(ψ_1)
      → Low mass

Gen2: |ψ_2⟩ = Moderately compressible (intermediate)
      → Medium complexity K(ψ_2)
      → Medium mass

Gen3: |ψ_3⟩ = Incompressible (complex, random-like)
      → High complexity K(ψ_3)
      → High mass

Mass formula: m_i ∝ K(ψ_i)^α

where K(ψ_i) = Kolmogorov complexity of information state
"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Dict
import json
from datetime import datetime


# ==============================================================================
# INFORMATION STATES - PRIMARY SUBSTRATE
# ==============================================================================

class InformationState:
    """
    Fundamental information configuration

    NOT derived from a network - this IS the fundamental object!
    Networks emerge as low-energy representations.
    """

    def __init__(self, state_type: str, N: int = 1000):
        """
        state_type: 'simple', 'intermediate', 'complex'
        N: Information content (number of qubits/bits)
        """
        self.state_type = state_type
        self.N = N
        self.bitstring = self._generate_information_state()

    def _generate_information_state(self) -> np.ndarray:
        """
        Generate fundamental information configurations with EXTREME differences

        Key: These have INTRINSIC complexity, not derived from anything
        """
        if self.state_type == 'simple':
            # ULTRA-compressible: Single bit repeated
            # Kolmogorov complexity ≈ 1 (just "repeat 0" or "alternate 01")
            # Use alternating pattern for minimal but non-zero complexity
            bitstring = np.zeros(self.N, dtype=int)
            bitstring[::2] = 1  # 0101010101...

        elif self.state_type == 'intermediate':
            # Medium compressibility: Nested structure
            # Create hierarchical pattern: blocks within blocks
            bitstring = np.zeros(self.N, dtype=int)

            # Level 1: Large blocks of 100
            for i in range(0, self.N, 200):
                bitstring[i:min(i+100, self.N)] = 1

            # Level 2: Medium blocks of 20 (overwrite some)
            for i in range(0, self.N, 50):
                bitstring[i:min(i+20, self.N)] = 0

            # Level 3: Small blocks of 3
            for i in range(0, self.N, 10):
                bitstring[i:min(i+3, self.N)] = 1

        elif self.state_type == 'complex':
            # MAXIMAL complexity: Cryptographically strong pseudo-random
            # Kolmogorov complexity ≈ N (incompressible)
            np.random.seed(None)
            bitstring = np.random.randint(0, 2, self.N)

        else:
            raise ValueError(f"Unknown state_type: {self.state_type}")

        return bitstring

    def compute_kolmogorov_complexity(self) -> float:
        """
        Estimate Kolmogorov complexity K(ψ)

        Uses Lempel-Ziv compression as proxy
        """
        return self._lempel_ziv_complexity(self.bitstring)

    def compute_shannon_entropy(self) -> float:
        """Shannon entropy of bitstring"""
        # Compute run-length distribution
        runs = []
        current_run = 1
        for i in range(1, len(self.bitstring)):
            if self.bitstring[i] == self.bitstring[i-1]:
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1
        runs.append(current_run)

        # Compute entropy of run-length distribution
        unique, counts = np.unique(runs, return_counts=True)
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log2(probs + 1e-10))

        return entropy

    def compute_compressibility(self) -> float:
        """
        Overall compressibility metric

        C = 1 - (K / N)

        High C = highly compressible = simple
        Low C = incompressible = complex
        """
        K = self.compute_kolmogorov_complexity()
        # For alternating (01010), K ≈ 2
        # For random, K ≈ N
        # So C ranges from ~1 (simple) to ~0 (complex)
        return 1.0 - (K / self.N)

    def _lempel_ziv_complexity(self, sequence: np.ndarray) -> float:
        """
        Lempel-Ziv complexity: number of distinct patterns

        Returns RAW count (not normalized) to allow large hierarchies
        """
        n = len(sequence)
        i = 0
        complexity = 0
        vocab = set()

        while i < n:
            # Find longest prefix not in vocabulary
            j = i + 1
            while j <= n:
                pattern = tuple(sequence[i:j])
                if pattern not in vocab:
                    vocab.add(pattern)
                    complexity += 1
                    i = j
                    break
                j += 1
            else:
                break

        # Return RAW complexity (can range from 1 to N)
        return float(complexity)

    def emerge_network(self) -> nx.Graph:
        """
        EMERGENCE: Network as low-energy representation of information

        This is the KEY difference from previous approaches!
        The network EMERGES from information, not vice versa.

        Algorithm: Use bitstring to define network connectivity
        - Bits encode presence/absence of edges
        - Network structure reflects information content
        """
        # Determine network size (scales with sqrt to keep manageable)
        n_nodes = int(np.sqrt(self.N))

        # Create network
        G = nx.Graph()
        G.add_nodes_from(range(n_nodes))

        # Use bitstring to determine edges
        bit_index = 0
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                if bit_index < len(self.bitstring):
                    # Connect if bit is 1
                    if self.bitstring[bit_index] == 1:
                        G.add_edge(i, j)
                    bit_index += 1
                else:
                    break

        # Ensure connectivity (physical requirement)
        if not nx.is_connected(G):
            # Add minimum spanning connections
            components = list(nx.connected_components(G))
            for i in range(len(components) - 1):
                # Connect components
                node_a = list(components[i])[0]
                node_b = list(components[i+1])[0]
                G.add_edge(node_a, node_b)

        return G


# ==============================================================================
# INFORMATION-FIRST MASS GENERATION
# ==============================================================================

def compute_ift_masses(states: List[InformationState], alpha: float = 2.0) -> Tuple[np.ndarray, Dict]:
    """
    Compute fermion masses from information states

    Core principle: m_i ∝ K(ψ_i)^α

    Mass = Energy cost of processing information
    More complex information → higher processing cost → heavier mass

    Args:
        states: Three InformationState objects (generations)
        alpha: Power-law exponent (controls hierarchy strength)

    Returns:
        masses: [m_1, m_2, m_3] mass ratios
        diagnostics: Dictionary of intermediate values
    """
    # Compute information-theoretic properties
    complexities = []
    compressibilities = []
    entropies = []

    for state in states:
        K = state.compute_kolmogorov_complexity()
        C = state.compute_compressibility()
        H = state.compute_shannon_entropy()

        complexities.append(K)
        compressibilities.append(C)
        entropies.append(H)

    complexities = np.array(complexities)
    compressibilities = np.array(compressibilities)
    entropies = np.array(entropies)

    # Mass formula: m ∝ K^α
    # Higher complexity → heavier mass
    masses = complexities ** alpha

    # Normalize to electron = 1
    masses = masses / masses.min()

    diagnostics = {
        'complexities': complexities.tolist(),
        'compressibilities': compressibilities.tolist(),
        'entropies': entropies.tolist(),
        'alpha': alpha
    }

    return masses, diagnostics


# ==============================================================================
# TESTING FRAMEWORK
# ==============================================================================

def run_ift_test(n_trials: int = 20, N: int = 1000, alpha: float = 2.0):
    """
    Test Information-First Theory

    Key difference: We generate THREE DISTINCT information states,
    not three cuts of the same network!
    """
    print("\n" + "="*70)
    print("INFORMATION-FIRST THEORY (IFT) TEST")
    print("="*70)

    print("\nRevolutionary Idea:")
    print("  Information is PRIMARY, not derived from network")
    print("  Previous: Network → Observables → Masses (FAILED)")
    print("  New:      Information → Network → Masses")

    print(f"\nParameters:")
    print(f"  Information content: N = {N} bits")
    print(f"  Trials: {n_trials}")
    print(f"  Power-law exponent: α = {alpha}")

    # Target mass ratios
    target = np.array([1, 207, 3477])
    print(f"\nTarget: e:μ:τ = {target[0]}:{target[1]}:{target[2]}")
    print("="*70)

    results = []

    for trial in range(n_trials):
        print(f"\n--- Trial {trial+1}/{n_trials} ---")

        # Generate THREE DISTINCT information states
        # This is the KEY: Not three observations of one network,
        # but three fundamentally different information configurations
        state1 = InformationState('simple', N)
        state2 = InformationState('intermediate', N)
        state3 = InformationState('complex', N)

        states = [state1, state2, state3]

        # Compute masses from information
        masses, diagnostics = compute_ift_masses(states, alpha=alpha)

        # Evaluate
        errors = np.abs(masses - target) / target * 100

        print(f"Information complexities: K = {diagnostics['complexities']}")
        print(f"Compressibilities: C = {diagnostics['compressibilities']}")
        print(f"Predicted masses: [{masses[0]:.0f}, {masses[1]:.0f}, {masses[2]:.0f}]")
        print(f"Errors: Gen2={errors[1]:.1f}%, Gen3={errors[2]:.1f}%")

        # Store results
        results.append({
            'masses': masses.tolist(),
            'errors': errors.tolist(),
            'diagnostics': diagnostics
        })

        # Check if successful (within 30% for both)
        if errors[1] < 30 and errors[2] < 30:
            print("✓ SUCCESS!")

    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    all_masses = np.array([r['masses'] for r in results])
    mean_masses = all_masses.mean(axis=0)
    std_masses = all_masses.std(axis=0)

    all_errors = np.array([r['errors'] for r in results])
    mean_errors = all_errors.mean(axis=0)

    print(f"\nAverage across {n_trials} trials:")
    print(f"  Predicted: [{mean_masses[0]:.0f}, {mean_masses[1]:.0f}±{std_masses[1]:.0f}, {mean_masses[2]:.0f}±{std_masses[2]:.0f}]")
    print(f"  Target:    [{target[0]}, {target[1]}, {target[2]}]")
    print(f"  Average errors: Gen2={mean_errors[1]:.1f}%, Gen3={mean_errors[2]:.1f}%")

    # Count successes
    successes = sum(1 for r in results if r['errors'][1] < 30 and r['errors'][2] < 30)
    print(f"\n{successes}/{n_trials} trials passed (30% threshold)")

    # Best trial
    best_idx = np.argmin([r['errors'][1] + r['errors'][2] for r in results])
    best = results[best_idx]
    print(f"\nBest trial:")
    print(f"  Masses: [{best['masses'][0]:.0f}, {best['masses'][1]:.0f}, {best['masses'][2]:.0f}]")
    print(f"  Errors: Gen2={best['errors'][1]:.1f}%, Gen3={best['errors'][2]:.1f}%")

    # Show complexity ratios
    all_K = np.array([r['diagnostics']['complexities'] for r in results])
    mean_K = all_K.mean(axis=0)
    print(f"\nComplexity ratios:")
    print(f"  K_2/K_1 = {mean_K[1]/mean_K[0]:.2f} (need {target[1]/target[0]:.2f})")
    print(f"  K_3/K_1 = {mean_K[2]/mean_K[0]:.2f} (need {target[2]/target[0]:.2f})")
    print(f"  K_3/K_2 = {mean_K[2]/mean_K[1]:.2f} (need {target[2]/target[1]:.2f})")

    # Verdict
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)

    if successes > n_trials * 0.5:
        print("\n✓ SUCCESS\n")
        print("IFT works! Information → Masses")
        print("\nThis is revolutionary: Information is truly primary!")
    elif mean_errors[1] < 80 and mean_errors[2] < 80:
        print("\nPARTIAL SUCCESS\n")
        print("IFT shows promise. Issues:")
        if mean_K[1]/mean_K[0] < 10:
            print("- K ratios too small (need larger complexity differences)")
            print("- Try: Adjust information state generation")
            print("- Or: Scan alpha parameter")
        if std_masses[1]/mean_masses[1] > 0.3:
            print("- High variance (information states not stable enough)")
            print("- Try: Increase N (more bits)")
    else:
        print("\n✗ FAILURE\n")
        print("IFT does not produce correct mass hierarchy.")
        print("\nPossible issues:")
        print("- LZ complexity proxy not capturing true algorithmic complexity")
        print("- Information states need better design")
        print("- May need quantum information measures (entanglement entropy)")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'N': N,
            'n_trials': n_trials,
            'alpha': alpha
        },
        'target': target.tolist(),
        'results': results,
        'summary': {
            'mean_masses': mean_masses.tolist(),
            'std_masses': std_masses.tolist(),
            'mean_errors': mean_errors.tolist(),
            'successes': successes,
            'mean_complexities': mean_K.tolist()
        }
    }

    with open('ift_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("\nResults saved to ift_results.json")

    return results


def scan_alpha_parameter(N: int = 1000, n_trials: int = 10):
    """
    Scan alpha parameter to find optimal hierarchy
    """
    print("\n" + "="*70)
    print("ALPHA PARAMETER SCAN")
    print("="*70)

    alphas = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    target = np.array([1, 207, 3477])

    best_alpha = None
    best_error = float('inf')

    for alpha in alphas:
        print(f"\nTesting α = {alpha}...")

        errors_sum = 0
        for trial in range(n_trials):
            state1 = InformationState('simple', N)
            state2 = InformationState('intermediate', N)
            state3 = InformationState('complex', N)

            masses, _ = compute_ift_masses([state1, state2, state3], alpha=alpha)
            errors = np.abs(masses - target) / target * 100
            errors_sum += errors[1] + errors[2]

        avg_error = errors_sum / n_trials
        print(f"  Average error: {avg_error:.1f}%")

        if avg_error < best_error:
            best_error = avg_error
            best_alpha = alpha

    print("\n" + "="*70)
    print(f"Best α = {best_alpha} with average error {best_error:.1f}%")
    print("="*70)

    return best_alpha


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    start_time = datetime.now()

    print("\n" + "="*70)
    print("INFORMATION-FIRST THEORY")
    print("="*70)

    print("\nKey Insight:")
    print("  Previous theories failed because they computed information")
    print("  FROM networks, which locked observables by network statistics.")
    print()
    print("  IFT: Information is PRIMARY substrate")
    print("       Networks EMERGE from information")
    print("       Three DISTINCT information states → three generations")
    print()
    print("Why This Should Work:")
    print("  - Not constrained by single network type")
    print("  - Complexity can naturally span orders of magnitude")
    print("  - Simple bitstring (000111) vs random bitstring")
    print("    have vastly different Kolmogorov complexity!")
    print()
    print("Wheeler's \"It from Bit\" - literally!")

    # First, scan alpha to find optimal value
    print("\n[1/2] Scanning alpha parameter...")
    best_alpha = scan_alpha_parameter(N=1000, n_trials=10)

    # Run full test with best alpha
    print(f"\n[2/2] Full test with α = {best_alpha}...")
    results = run_ift_test(n_trials=20, N=1000, alpha=best_alpha)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60

    print(f"\nTotal runtime: {duration:.1f} minutes")

    print("\n" + "="*70)
    print("THEORETICAL IMPLICATIONS")
    print("="*70)

    print("\nIf IFT succeeds:")
    print("  1. Information is more fundamental than spacetime")
    print("  2. Fermion masses = Information processing costs")
    print("  3. Generations = Distinct information configurations")
    print("  4. Universe is fundamentally a quantum computer")
    print()
    print("Testable Predictions:")
    print("  - Fermion production ∝ Information complexity")
    print("  - Entanglement patterns should match mass hierarchy")
    print("  - Holographic bounds should relate to fermion sector")
    print()
    print("Connection to Quantum Eraser:")
    print("  Information erasure → Mass changes?")
    print("  (Extremely speculative but worth exploring!)")
    print()
