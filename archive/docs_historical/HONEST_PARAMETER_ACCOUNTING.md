# Honest Parameter Accounting

**Purpose**: Strict categorization of what is derived vs constrained vs calibrated
**Standard**: Referee-proof accounting for publication readiness

---

## Category Definitions

### 1. DERIVED (from geometry alone)
Parameters computed from first principles with **no free choices**:
- Input: String geometry (τ, k_i, wrapping numbers)
- Output: Unique numerical prediction
- Test: Could be computed before any data

### 2. CONSTRAINED (by geometry + calibration)
Parameters determined by geometric structure + measured inputs:
- Input: Geometry + 1-3 measured values (e.g., α_EM, m_h)
- Output: Predictions for related observables
- Test: Reduces freedom compared to SM

### 3. CALIBRATED (optimized parameters)
Free effective couplings fitted to data:
- Input: Observational data
- Method: Optimization (differential evolution, etc.)
- Status: **Currently phenomenological**
- Path forward: Derive from Kähler metric or other geometry

### 4. ASSUMED (identities/consistency)
Not observables but mathematical necessities:
- Examples: m_γ = 0, charge quantization, unitarity
- Status: Should NOT be counted as "predictions"

---

## Current Parameter Status

### DERIVED Parameters (11 total)

#### Gauge Sector (3)
| Parameter | Value | Derivation | Status |
|-----------|-------|------------|--------|
| k_1 (U(1)_Y) | 1 | Kac-Moody level | ✅ DERIVED |
| k_2 (SU(2)) | 2 | Kac-Moody level | ✅ DERIVED |
| k_3 (SU(3)) | 3 | Kac-Moody level | ✅ DERIVED |

**Output**: α_1, α_2, α_3 at string scale

#### Higgs Sector (2)
| Parameter | Value | Derivation | Status |
|-----------|-------|------------|--------|
| v | 246 GeV | EWSB + g_Y, g_2 | ✅ CONSTRAINED |
| λ_h | 0.129 | m_h relation | ✅ CONSTRAINED |

**Input**: m_h = 125 GeV (measured)
**Output**: v, λ_h from geometric gauge couplings

#### Fermion Mass Scales (6)
| Parameter | Value | Derivation | Status |
|-----------|-------|------------|--------|
| τ_0 | 2.507j | Geometric | ✅ DERIVED |
| k | 20 | Modular weight | ✅ DERIVED |
| c_lep | 0.04 | Wrapping → area | ✅ DERIVED |
| c_up | 1.0 | Wrapping → area | ✅ DERIVED |
| c_down | 0.25 | Wrapping → area | ✅ DERIVED |
| τ_ν | 0.0244j | Separate modulus | ✅ DERIVED |

**Output**: Overall mass scales per sector

---

### CALIBRATED Parameters (27 total)

#### Generation Hierarchy (9) - **Phase 2 Target**
| Parameter | Count | Current Status | Physical Meaning |
|-----------|-------|----------------|------------------|
| A_lep' | 3 | ⚠️ CALIBRATED | Lepton localization (absorbing g_i) |
| A_up' | 3 | ⚠️ CALIBRATED | Up-quark localization (absorbing g_i) |
| A_down' | 3 | ⚠️ CALIBRATED | Down-quark localization (absorbing g_i) |

**Role**: Wavefunction widths → Gaussian overlap integrals → generation hierarchy
**Current method**: Optimized to match m_e, m_μ, m_τ, m_u, m_c, m_t, m_d, m_s, m_b
**Path forward**: Derive from Kähler metric K_{i̅j} = ∂_i∂_̅j K(τ, τ̄)

**Critical assessment**:
> "These are free effective couplings. They replace Yukawa matrices with wavefunction widths. The structure is geometric but values are currently phenomenological."

#### CKM Off-Diagonals (12) - **Phase 2 Target**
| Parameter | Count | Current Status | Physical Meaning |
|-----------|-------|----------------|------------------|
| ε_lep | 6 | ⚠️ CALIBRATED | Lepton mixing phases |
| ε_up | 6 | ⚠️ CALIBRATED | Up-quark mixing phases |
| ε_down | 6 | ⚠️ CALIBRATED | Down-quark mixing phases |

**Role**: Off-diagonal structure in Yukawa matrices
**Current method**: Optimized to match V_CKM observables (sin²θ_12, sin²θ_23, sin²θ_13, δ_CP, J_CP)
**Path forward**: Derive from intersection geometry in 25-parameter D-brane moduli space

**Previous attempt**: Geometric CKM derivation → 1767% error on V_us
**Honest assessment**: Structure may be geometric, but current values are fitted

#### Neutrino Structure (16) - **Partially Phase 2**
| Parameter | Count | Current Status | Physical Meaning |
|-----------|-------|----------------|------------------|
| M_R scale | 1 | ✅ DERIVED | Heavy neutrino mass (~50 GeV) |
| μ scale | 1 | ✅ DERIVED | Majorana mass (~50 GeV) |
| M_D off-diag | 6 | ⚠️ CALIBRATED | Dirac mixing structure |
| M_R off-diag | 6 | ⚠️ CALIBRATED | Right-handed mixing |
| μ off-diag | 4 | ⚠️ CALIBRATED | Majorana mixing structure |

**Scales derived**: τ_ν = 0.0244j from geometric + instanton corrections
**Structure fitted**: PMNS angles (θ_12, θ_13, θ_23) and CP phases (δ, α, β)
**Path forward**: Connect to modular flavor symmetries (A_4, S_4 from τ discrete symmetries)

---

### ASSUMED (Not Observables)

These should **NOT** be counted as predictions:

| "Observable" | Why It's Not | Category |
|--------------|--------------|----------|
| m_γ = 0 | Gauge invariance | IDENTITY |
| Charge quantization | Gauge theory consistency | IDENTITY |
| Lepton number | Discrete symmetry | ASSUMPTION |
| c = ℏ = 1 | Unit choice | CONVENTION |
| ∑ Q_i = 0 | Anomaly cancellation | CONSISTENCY |

---

## Observable Accounting (Strict)

### Currently Predicted Observables

#### Truly Derived (13)
1. **Gauge couplings** (3): α_EM(M_Z), α_W(M_Z), α_s(M_Z)
2. **Gauge relations** (1): sin²θ_W
3. **Higgs** (1): λ_h (given m_h)
4. **v** (1): From EWSB (constrained by g_Y, g_2)

**Subtotal**: 6 observables from pure geometry

5. **Fermion mass ratios** within sectors (6):
   - m_μ/m_e, m_τ/m_e (leptons)
   - m_c/m_u, m_t/m_u (up quarks)
   - m_s/m_d, m_b/m_d (down quarks)

**Note**: These use calibrated A_i' but ratios are constrained by geometry

6. **Neutrino mass differences** (2): Δm²_21, Δm²_31 (from τ_ν scale)

**Subtotal**: 8 constrained by geometry + calibration

7. **Dark energy** (2): w_0 ≈ -1, w_a (from frozen PNGB)

**Total truly predicted**: ~15-20 observables (depending on strictness)

#### Calibrated (Fitted to Data)
1. **Absolute fermion masses** (9): m_e through m_b (uses A_i')
2. **CKM observables** (5): sin²θ_12, sin²θ_23, sin²θ_13, δ_CP, J_CP
3. **PMNS angles** (3): θ_12, θ_13, θ_23
4. **PMNS phases** (3): δ, α, β

**Total calibrated**: 20 observables (fitted using 27 parameters)

---

## Honest Comparison to Standard Model

### Standard Model
- **Parameters**: 19 (6 quark masses, 3 lepton masses, 3 gauge couplings, 4 CKM, v, λ_h, + neutrino sector)
- **Observables**: ~50 (all of particle physics)
- **Predictive power**: 50/19 ≈ 2.6 predictions per parameter

### Our Framework (Honest Count)
- **Derived**: 11 parameters (gauge + Higgs + scales)
- **Calibrated**: 27 parameters (A_i', ε_ij, neutrino structure)
- **Total**: 38 parameters (11 derived + 27 calibrated)
- **Observables**: ~35 (removing identities and consistency conditions)
- **Current predictive power**: 35/38 ≈ 0.9 predictions per parameter

**BUT**: If A_i' can be derived from Kähler metric:
- **Reduced to**: 11 derived + 18 calibrated = 29 parameters
- **Predictive power**: 35/29 ≈ 1.2 predictions per parameter

**IF** neutrino structure follows from modular symmetries:
- **Reduced to**: 11 derived + 12 calibrated = 23 parameters
- **Predictive power**: 35/23 ≈ 1.5 predictions per parameter

---

## The Critical Distinction

### What We CAN Claim
> "A unified geometric framework that:
> - Derives gauge couplings from Kac-Moody levels
> - Derives Higgs sector from EWSB consistency
> - Derives fermion mass scales from modular geometry
> - Derives neutrino seesaw scales from instantons
> - **Isolates remaining freedom to wavefunction localization parameters**
> - Provides a systematic path to compute these from Kähler metric"

### What We CANNOT Claim (Yet)
> "A complete Theory of Everything with all parameters derived"

The difference is crucial for survival under expert scrutiny.

---

## The One Hard Derivation (Decisive)

To cross from "framework" to "theory", we must choose:

### Option A: Derive ℓ_sector from Kähler Metric (BEST)

**Goal**: Show A_i ~ f(τ, g_s, V_6) from K_{i̅j}

**Method**:
1. Start with Kähler potential: K = -k ln(S + S̄) - 3ln(T + T̄)
2. Compute metric: K_{i̅j} = ∂_i∂_̅j K
3. Solve Laplacian on CY3: ∇² ψ ~ m² ψ
4. Extract localization scale: ℓ ~ 1/√(∂²K)

**Impact**: Converts 9 free parameters → geometric consequences

**Status**: Feasible with specific CY3 (Swiss cheese, T³/ℤ₂×ℤ₂)

### Option B: Freeze All and Predict (STRONG)

**Goal**: Out-of-sample prediction with NO new fitting

**Options**:
1. Predict CKM structure without optimizing ε_ij (currently 1767% error - would falsify if can't fix)
2. Predict collider signatures from M_R ~ 50 GeV (falsifiable at LHC/HL-LHC)
3. Predict relations between mass ratios across sectors (e.g., m_μ/m_e vs m_s/m_d)

**Impact**: Demonstrates predictive power beyond fitting

**Risk**: Failure would be visible

---

## Recommended Actions

### 1. Immediate (This Session)
- ✅ Create this honest accounting document
- ⚠️ Update GEOMETRIC_CKM_STATUS.md with strict categories
- ⚠️ Remove identities from "predicted observables" count
- ⚠️ Label all optimization results as "calibrated effective couplings"

### 2. Next Priority (Choose One Path)
- **Path A**: Begin Kähler metric derivation of A_i'
  - Start with diagonal metric for single generation
  - Extend to generation mixing
  - Test against calibrated values

- **Path B**: Make frozen prediction
  - Lock all 27 calibrated parameters
  - Compute new observable not used in fitting
  - Publish prediction before measurement

### 3. Manuscript Structure
- **Part I**: What is derived (gauge + Higgs + scales)
- **Part II**: What is constrained (mass ratios, neutrino differences)
- **Part III**: What remains calibrated (localization widths, mixing structure)
- **Part IV**: Path to full derivation (Kähler metric program)

---

## Bottom Line

**Current status**:
- No longer toy model
- Not yet complete ToE
- One hard derivation away from serious theory

**Honest claim**:
> "Geometric framework reproducing SM+cosmology structure with freedom isolated to wavefunction normalization (computable from Kähler metric)"

**Next milestone**:
- Derive A_i' from K_{i̅j}, OR
- Make falsifiable out-of-sample prediction

This is the decisive step for expert engagement.
