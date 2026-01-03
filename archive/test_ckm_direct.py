import numpy as np

# Exact same code as optimize_ckm.py
m_up = np.array([1.0, 577.0, 78636.0])
m_down = np.array([1.0, 18.3, 890.0])

eps_up = np.array([-3.56395815, 0.11670461, 0.01016270])
eps_down = np.array([1.29928251, 0.59493618, 0.07393132])

# Build Yukawa matrices
Y_up = np.diag(m_up).astype(complex)
Y_up[0, 1] = eps_up[0] * np.sqrt(m_up[0] * m_up[1])
Y_up[1, 0] = Y_up[0, 1]
Y_up[1, 2] = eps_up[1] * np.sqrt(m_up[1] * m_up[2])
Y_up[2, 1] = Y_up[1, 2]
Y_up[0, 2] = eps_up[2] * np.sqrt(m_up[0] * m_up[2])
Y_up[2, 0] = Y_up[0, 2]

Y_down = np.diag(m_down).astype(complex)
Y_down[0, 1] = eps_down[0] * np.sqrt(m_down[0] * m_down[1])
Y_down[1, 0] = Y_down[0, 1]
Y_down[1, 2] = eps_down[1] * np.sqrt(m_down[1] * m_down[2])
Y_down[2, 1] = Y_down[1, 2]
Y_down[0, 2] = eps_down[2] * np.sqrt(m_down[0] * m_down[2])
Y_down[2, 0] = Y_down[0, 2]

print("Y_up:")
print(Y_up)
print()
print("Y_down:")
print(Y_down)
print()

# Diagonalize
U_uL, _, _ = np.linalg.svd(Y_up)
U_dL, _, _ = np.linalg.svd(Y_down)

# CKM
V_CKM = U_uL @ U_dL.conj().T

print("V_CKM:")
print(V_CKM)
print()

# Extract angles
s12 = np.abs(V_CKM[0, 1])
s23 = np.abs(V_CKM[1, 2])
s13 = np.abs(V_CKM[0, 2])

print(f"sin²θ₁₂ = {s12**2:.6f} (obs: 0.051000)")
print(f"sin²θ₂₃ = {s23**2:.6f} (obs: 0.001570)")
print(f"sin²θ₁₃ = {s13**2:.6f} (obs: 0.000128)")
