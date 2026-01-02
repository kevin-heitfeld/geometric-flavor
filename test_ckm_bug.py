"""Debug CKM calculation discrepancy."""
import numpy as np

# Mass ratios
y_up = np.array([1.0, 577.0, 78636.0])
y_down = np.array([1.0, 20.3, 890.0])

# Epsilon parameters
eps_up = np.array([-3.56395815, 0.11670461, 0.01016270])
eps_down = np.array([1.29928251, 0.59493618, 0.07393132])

# Build Yukawa matrices
Y_up = np.diag(y_up).astype(complex)
Y_up[0, 1] = eps_up[0] * np.sqrt(y_up[0] * y_up[1])
Y_up[1, 0] = Y_up[0, 1]
Y_up[1, 2] = eps_up[1] * np.sqrt(y_up[1] * y_up[2])
Y_up[2, 1] = Y_up[1, 2]
Y_up[0, 2] = eps_up[2] * np.sqrt(y_up[0] * y_up[2])
Y_up[2, 0] = Y_up[0, 2]

Y_down = np.diag(y_down).astype(complex)
Y_down[0, 1] = eps_down[0] * np.sqrt(y_down[0] * y_down[1])
Y_down[1, 0] = Y_down[0, 1]
Y_down[1, 2] = eps_down[1] * np.sqrt(y_down[1] * y_down[2])
Y_down[2, 1] = Y_down[1, 2]
Y_down[0, 2] = eps_down[2] * np.sqrt(y_down[0] * y_down[2])
Y_down[2, 0] = Y_down[0, 2]

print("Y_up:")
print(Y_up)
print()
print("Y_down:")
print(Y_down)
print()

# Method 1: SVD as in main script
U_uL_svd, _, _ = np.linalg.svd(Y_up)
U_dL_svd, _, _ = np.linalg.svd(Y_down)
V_CKM_svd = U_uL_svd @ U_dL_svd.conj().T

print("Method 1 (SVD):")
print("V_CKM:")
print(V_CKM_svd)
s12_svd = np.abs(V_CKM_svd[0, 1])
s23_svd = np.abs(V_CKM_svd[1, 2])
s13_svd = np.abs(V_CKM_svd[0, 2])
print(f"sin²θ₁₂ = {s12_svd**2:.6f}")
print(f"sin²θ₂₃ = {s23_svd**2:.6f}")
print(f"sin²θ₁₃ = {s13_svd**2:.6f}")
print()

# Method 2: Diagonalize Y†Y as in optimizer
YuYu = Y_up.conj().T @ Y_up
YdYd = Y_down.conj().T @ Y_down

# Get eigenvectors
_, U_uL_eig = np.linalg.eigh(YuYu)
_, U_dL_eig = np.linalg.eigh(YdYd)

V_CKM_eig = U_uL_eig @ U_dL_eig.conj().T

print("Method 2 (Y†Y diagonalization):")
print("V_CKM:")
print(V_CKM_eig)
s12_eig = np.abs(V_CKM_eig[0, 1])
s23_eig = np.abs(V_CKM_eig[1, 2])
s13_eig = np.abs(V_CKM_eig[0, 2])
print(f"sin²θ₁₂ = {s12_eig**2:.6f}")
print(f"sin²θ₂₃ = {s23_eig**2:.6f}")
print(f"sin²θ₁₃ = {s13_eig**2:.6f}")
print()

# Observed
sin2_12_obs = 0.0510
sin2_23_obs = 0.00157
sin2_13_obs = 0.000128

print("Observed:")
print(f"sin²θ₁₂ = {sin2_12_obs:.6f}")
print(f"sin²θ₂₃ = {sin2_23_obs:.6f}")
print(f"sin²θ₁₃ = {sin2_13_obs:.6f}")
print()

print("Errors (SVD method):")
print(f"  sin²θ₁₂: {abs(s12_svd**2 - sin2_12_obs)/sin2_12_obs*100:.1f}%")
print(f"  sin²θ₂₃: {abs(s23_svd**2 - sin2_23_obs)/sin2_23_obs*100:.1f}%")
print(f"  sin²θ₁₃: {abs(s13_svd**2 - sin2_13_obs)/sin2_13_obs*100:.1f}%")
print()

print("Errors (Y†Y method):")
print(f"  sin²θ₁₂: {abs(s12_eig**2 - sin2_12_obs)/sin2_12_obs*100:.1f}%")
print(f"  sin²θ₂₃: {abs(s23_eig**2 - sin2_23_obs)/sin2_23_obs*100:.1f}%")
print(f"  sin²θ₁₃: {abs(s13_eig**2 - sin2_13_obs)/sin2_13_obs*100:.1f}%")
