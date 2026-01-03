"""
Quick diagnostic: Check actual ranges of info-theoretic observables
"""
import numpy as np
import networkx as nx
from qic_theory import (generate_network, compute_compressibility,
                         compute_shannon_entropy, compute_kolmogorov_proxy,
                         coarse_grain_network)

# Generate test network
G = generate_network(2000, m=4)

print("Checking ranges of observables across RG scales:")
print("="*60)

for scale in range(5):
    C = compute_compressibility(G)
    H = compute_shannon_entropy(G)
    K = compute_kolmogorov_proxy(G)

    print(f"\nScale {scale}:")
    print(f"  N = {G.number_of_nodes()}, E = {G.number_of_edges()}")
    print(f"  Compressibility C = {C:.4f}")
    print(f"  Shannon entropy H = {H:.4f}")
    print(f"  Kolmogorov proxy K = {K:.4f}")
    print(f"  Ratio H/K = {H/(K+0.1):.4f}")

    G = coarse_grain_network(G, factor=2)
    if G.number_of_nodes() < 10:
        break

print("\n" + "="*60)
print("Observations:")
print("- C is in range [0, 1]")
print("- H is typically [2, 4] (bits)")
print("- K is typically [0.2, 0.6]")
print("- H/K ratio is typically [5, 20]")
print("\nProblem: H/K ratio is ~10-100x larger than C and K!")
print("Solution: Need to normalize all scores to same scale")
