"""
Critical diagnostic: Are information-theoretic observables also locked
by scale-free network statistics? (Like HNR persistence was)
"""
import numpy as np
import networkx as nx
from qic_theory import (generate_network, compute_compressibility,
                         compute_shannon_entropy, compute_kolmogorov_proxy)

print("Testing variance of info-theoretic observables across 50 networks:")
print("="*70)

compressibilities = []
entropies = []
complexities = []

for trial in range(50):
    G = generate_network(2000, m=4)

    C = compute_compressibility(G)
    H = compute_shannon_entropy(G)
    K = compute_kolmogorov_proxy(G)

    compressibilities.append(C)
    entropies.append(H)
    complexities.append(K)

C_arr = np.array(compressibilities)
H_arr = np.array(entropies)
K_arr = np.array(complexities)

print(f"\nCompressibility C:")
print(f"  Mean: {C_arr.mean():.4f}")
print(f"  Std:  {C_arr.std():.4f}")
print(f"  Range: [{C_arr.min():.4f}, {C_arr.max():.4f}]")
print(f"  Coefficient of variation: {C_arr.std()/C_arr.mean():.4f}")

print(f"\nShannon entropy H:")
print(f"  Mean: {H_arr.mean():.4f}")
print(f"  Std:  {H_arr.std():.4f}")
print(f"  Range: [{H_arr.min():.4f}, {H_arr.max():.4f}]")
print(f"  Coefficient of variation: {H_arr.std()/H_arr.mean():.4f}")

print(f"\nKolmogorov complexity K:")
print(f"  Mean: {K_arr.mean():.4f}")
print(f"  Std:  {K_arr.std():.4f}")
print(f"  Range: [{K_arr.min():.4f}, {K_arr.max():.4f}]")
print(f"  Coefficient of variation: {K_arr.std()/K_arr.mean():.4f}")

print("\n" + "="*70)
print("VERDICT:")
print("="*70)

cv_threshold = 0.10  # Need at least 10% variation
if C_arr.std()/C_arr.mean() < cv_threshold:
    print("\n❌ FUNDAMENTAL PROBLEM IDENTIFIED:")
    print("\nInformation-theoretic observables are LOCKED by scale-free statistics!")
    print("Just like HNR persistence was locked at ~2:1 ratio.")
    print("\nAll BA networks have similar:")
    print("- Degree distributions (power-law with γ ≈ 3)")
    print("- Entropy (determined by power-law exponent)")
    print("- Complexity (also determined by power-law)")
    print("\nThis explains why QIC can't produce large hierarchies:")
    print("The observables don't vary enough between different 'generations'")
    print("because they're all computed from the SAME underlying network!")
    print("\nWe need fundamentally DIFFERENT network types for each generation,")
    print("not just different cuts/bipartitions of the same network.")
else:
    print("\n✓ Observables have sufficient variation")
    print("QIC theory may work with better score formulas")
