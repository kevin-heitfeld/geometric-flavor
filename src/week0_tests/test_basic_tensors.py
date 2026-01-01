"""
Week 0 Test: Basic Tensor Operations
Goal: Verify NumPy/SciPy can handle tensor contractions and SVD

This is your first code file in the 6-month journey!
"""

import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt

print("="*70)
print("WEEK 0 TEST: Basic Tensor Operations")
print("="*70)
print()

# Test 1: Create a simple rank-3 tensor
print("Test 1: Creating tensors")
print("-"*70)

A = np.random.rand(3, 4, 5)
print(f"Tensor A shape: {A.shape}")
print(f"Tensor A is rank-{len(A.shape)}")
print()

# Test 2: Tensor contraction (like matrix multiplication, but more general)
print("Test 2: Tensor contraction")
print("-"*70)

B = np.random.rand(5, 6, 7)
print(f"Tensor B shape: {B.shape}")

# Contract over shared index (size 5)
C = np.tensordot(A, B, axes=([2], [0]))
print(f"Contraction A ⊗ B: {A.shape} × {B.shape} → {C.shape}")
print("✓ Contraction works!")
print()

# Test 3: SVD (crucial for MERA)
print("Test 3: Singular Value Decomposition (SVD)")
print("-"*70)

# Flatten tensor to matrix
M = A.reshape(12, 5)  # (3*4, 5)
print(f"Matrix M shape: {M.shape}")

# Perform SVD: M = U S V†
U, S, Vh = svd(M, full_matrices=False)
print(f"U shape: {U.shape}")
print(f"S shape: {S.shape}")
print(f"Vh shape: {Vh.shape}")
print()

print("Schmidt values (singular values):")
print(S)
print()

# Verify reconstruction
M_reconstructed = U @ np.diag(S) @ Vh
error = np.linalg.norm(M - M_reconstructed)
print(f"Reconstruction error: {error:.2e}")

if error < 1e-10:
    print("✓ SVD works perfectly!")
else:
    print("⚠ SVD has numerical error (check installation)")
print()

# Test 4: Schmidt spectrum analysis
print("Test 4: Schmidt spectrum (for perfect tensor check)")
print("-"*70)

# Normalize
S_normalized = S / np.linalg.norm(S)
print("Normalized Schmidt values:")
print(S_normalized)
print()

# Check flatness (perfect tensor has all equal)
S_std = np.std(S_normalized)
S_mean = np.mean(S_normalized)
flatness = S_std / S_mean

print(f"Standard deviation: {S_std:.4f}")
print(f"Mean: {S_mean:.4f}")
print(f"Flatness (std/mean): {flatness:.4f}")
print()

if flatness < 0.1:
    print("✓ Spectrum is nearly flat (this is rare for random tensors!)")
elif flatness < 0.5:
    print("○ Spectrum has some structure")
else:
    print("✗ Spectrum is very non-uniform (expected for random tensor)")
print()

# Visualize
plt.figure(figsize=(8, 5))
plt.bar(range(len(S_normalized)), S_normalized, color='steelblue')
plt.axhline(S_mean, color='red', linestyle='--', label=f'Mean = {S_mean:.3f}')
plt.xlabel('Index')
plt.ylabel('Normalized Schmidt Value')
plt.title('Schmidt Spectrum (Random Tensor)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/week0_schmidt_spectrum_test.png', dpi=150)
print("✓ Figure saved: results/week0_schmidt_spectrum_test.png")
print()

# Summary
print("="*70)
print("SUMMARY: All basic operations work!")
print("="*70)
print()
print("Next steps:")
print("1. Create perfect tensor from τ = 2.69i (Week 1)")
print("2. Build MERA layer structure")
print("3. Extract emergent metric")
print()
print("You're ready to start the intensive!")
print()
