# Phase 1 Completion Plan: Parameter Identification

**Goal:** Transform all 38 fitted parameters into geometrically-identified objects.
**Timeline:** 2-4 weeks focused work
**Status:** Implementation roadmap

---

## Executive Summary

**Phase 1 is 85-90% complete** but has critical gaps:
- ✅ All 50 observables predicted (<5% error)
- ✅ Internal consistency verified
- ✅ Modular structure demonstrated
- ❌ Only 4/38 parameters have clear geometric identity

**Remaining work:** Parameter identification, not new physics.

---

## Critical Path: 4 Major Tasks

### TASK 1: Collapse CP Violation to Single Spurion ⚡ HIGHEST PRIORITY

**Problem:** 28 free complex parameters for mixing (12 CKM + 16 neutrino)
**Solution:** ONE complex modular form generates all mixing

**Implementation:**

#### 1.1 Define Froggatt-Nielsen/Modular Spurion
```python
def get_CP_spurion(tau, sector='FN'):
    """
    Single source of CP violation for entire theory.

    Options:
    - 'FN': Froggatt-Nielsen VEV ⟨θ⟩ ~ λ_C e^{iδ}
    - 'modular': Modular form ⟨Z⟩ = η(τ)^w or G_2(τ)
    - 'flux': Flux-induced VEV from ISD(3,1) fluxes
    """
    if sector == 'FN':
        # Cabibbo-scale VEV with observable phase
        return 0.22 * np.exp(1j * 1.2)  # |⟨θ⟩| ~ λ_C, arg ∈ [0, 2π]

    elif sector == 'modular':
        # Weight-2 modular form (Eisenstein)
        G2 = eisenstein_G2(tau)
        return G2 / np.abs(G2)  # normalized for mixing

    else:
        raise ValueError(f"Unknown spurion sector: {sector}")
```

#### 1.2 Generate CKM from Single Source
```python
def generate_CKM_from_spurion(Z_spurion, U1_charges):
    """
    CKM matrix elements from ONE complex spurion.

    Y_ij = Y₀ × [δ_ij + ε_ij]
    ε_ij = C_ij × Z^{q_i - q_j}

    where:
    - C_ij: Clebsch-Gordan (from family symmetry, discrete)
    - q_i: U(1)_FN charges (integers, from anomaly cancellation)
    - Z: Complex spurion (ONE free parameter)
    """
    epsilon_up = np.zeros((3, 3), dtype=complex)
    epsilon_down = np.zeros((3, 3), dtype=complex)

    # Example charge assignment (Froggatt-Nielsen)
    q_up = np.array([3, 2, 0])    # u, c, t generations
    q_down = np.array([3, 2, 0])  # d, s, b generations

    # Clebsch coefficients (from SO(10) or SU(5) breaking)
    # These are FIXED by symmetry, not fitted
    C_up = get_clebsch_coefficients('up', family_symmetry='A4')
    C_down = get_clebsch_coefficients('down', family_symmetry='A4')

    for i in range(3):
        for j in range(3):
            if i != j:
                power = q_up[i] - q_up[j]
                epsilon_up[i, j] = C_up[i, j] × Z_spurion**power

                power = q_down[i] - q_down[j]
                epsilon_down[i, j] = C_down[i, j] × Z_spurion**power

    return epsilon_up, epsilon_down
```

**Benefit:** Reduces 28 free parameters → 1 complex spurion + discrete data

#### 1.3 Unify Neutrino CP Phase
```python
def generate_neutrino_mixing_from_spurion(Z_spurion, charges_L, charges_N):
    """
    Neutrino Dirac matrix from SAME spurion as CKM.

    M_D[i,j] = Y₀ × [δ_ij + ε_ij] × v
    ε_ij = C_ij^ν × Z^{q_L[i] - q_N[j]}

    Key: phase_CP is NOT independent, it comes from arg(Z).
    """
    M_D_offdiag = np.zeros(3, dtype=complex)

    # Off-diagonal structure from spurion
    M_D_offdiag[0] = C_01 × Z_spurion**(charges_L[0] - charges_N[1])
    M_D_offdiag[1] = C_12 × Z_spurion**(charges_L[1] - charges_N[2])
    M_D_offdiag[2] = C_02 × Z_spurion**(charges_L[0] - charges_N[2])

    return M_D_offdiag
```

**Code changes required:**
- Delete `fit_ckm_parameters()` optimization
- Delete independent `phase_CP` parameter
- Add `get_mixing_from_spurion(Z, charges, symmetry)`
- Optimize only: |Z|, arg(Z), charge assignments

**Success metric:** 28 parameters → 2 (|Z|, arg(Z)) + discrete charges

---

### TASK 2: Lock Yukawa Normalizations to Kähler Geometry

**Problem:** Y₀^(u,d,ℓ) are arbitrary normalization constants
**Solution:** Relate to Kähler metric + instantons

**Implementation:**

#### 2.1 Define Kähler Metric from Modulus
```python
def kahler_metric(tau, g_s, sector='uptype'):
    """
    Kähler metric component for Yukawa couplings.

    K = -k log(T + T̄) - log(S + S̄) + ...
    where T = τ (Kähler modulus), S = dilaton

    Yukawa ~ e^{-K_i/2 - K_j/2 - K_H/2} × ⟨ψ_i ψ_j H⟩
    """
    T = tau
    S = 1/g_s  # dilaton

    # Kähler potential (tree level)
    k_T = 3  # dimension of internal manifold piece
    K = -k_T × np.log(T.imag) - np.log(S.real)

    # Matter field Kähler metric (from localization)
    if sector == 'uptype':
        K_matter = K  # untwisted sector
    elif sector == 'downtype':
        K_matter = K + 0.5  # twisted sector shift
    elif sector == 'lepton':
        K_matter = K + 0.3  # different localization

    return np.exp(-K_matter / 2)
```

#### 2.2 Add Worldsheet Instanton Suppression
```python
def instanton_action(tau, g_s, sector):
    """
    Worldsheet instanton contributing to Yukawa.

    Y ~ e^{-S_inst} where S_inst = Area / ℓ_s²

    For different sectors, different 2-cycles are wrapped.
    """
    # Instanton action (real part)
    if sector == 'uptype':
        S_inst = 15.2  # wrapped 2-cycle area
    elif sector == 'downtype':
        S_inst = 14.5  # different 2-cycle
    elif sector == 'lepton':
        S_inst = 15.0  # third 2-cycle

    return np.exp(-S_inst)
```

#### 2.3 Compute Y₀ from Geometry
```python
def yukawa_normalization(tau, g_s, sector):
    """
    Yukawa normalization from Kähler + instantons.

    Y₀ = K(τ, g_s) × e^{-S_inst} × C_overlap

    where:
    - K: Kähler metric factor
    - S_inst: Worldsheet instanton action
    - C_overlap: Wavefunction overlap (order 1)
    """
    K_factor = kahler_metric(tau, g_s, sector)
    inst_factor = instanton_action(tau, g_s, sector)

    # Overlap integral (geometric, ~ 1)
    C_overlap = 1.0  # placeholder, from explicit CY calculation

    Y0 = K_factor × inst_factor × C_overlap

    return Y0
```

**Code changes required:**
- Replace fitted Y₀ with computed yukawa_normalization()
- Add Kähler potential K(τ, g_s)
- Add instanton action S_inst per sector
- Keep only C_overlap as order-1 factor

**Success metric:** 3 fitted Y₀ → 3 geometric expressions + 3 O(1) overlaps

---

### TASK 3: Discretize Localization Parameters

**Problem:** 12 continuous parameters (g_i, A_i)
**Solution:** Lock to discrete charges and geometric distances

**Implementation:**

#### 3.1 Lock g_i to Modular Weights
```python
def get_modular_weights_from_charges(U1_charges):
    """
    Modular weights are NOT free parameters.
    They are determined by U(1) charge assignments.

    For modular forms: f(τ) → (cτ+d)^{-w} f((aτ+b)/(cτ+d))
    Weight w is fixed by field's transformation law.
    """
    # Modular weights from anomaly cancellation
    # For standard embedding: w_i = q_i (U(1)_FN charge)

    weights = U1_charges.copy()  # integers or half-integers

    return weights
```

#### 3.2 Lock A_i to Brane Geometry
```python
def get_localization_factors_from_geometry(brane_positions, higgs_position):
    """
    Localization factors from physical brane separations.

    A_i = d_i / ℓ_s

    where d_i is distance between generation-i brane and Higgs brane.
    """
    distances = np.linalg.norm(brane_positions - higgs_position, axis=1)

    # In string units
    ell_string = 1.0  # set scale
    A_i = distances / ell_string

    return A_i
```

#### 3.3 Replace Continuous Optimization
```python
# OLD (wrong):
def fit_mass_parameters():
    """Optimize g_i, A_i continuously"""
    result = minimize(objective, x0=[...], bounds=[(-5,5), ...])
    return result.x

# NEW (correct):
def get_mass_parameters_from_geometry(tau, g_s, charges, brane_config):
    """Compute g_i, A_i from discrete geometry"""
    g_i = get_modular_weights_from_charges(charges)  # discrete
    A_i = get_localization_factors_from_geometry(brane_config['positions'],
                                                  brane_config['higgs'])  # geometric
    return g_i, A_i
```

**Code changes required:**
- Delete continuous g_i, A_i optimization
- Add discrete charge assignment (from anomaly cancellation)
- Add brane position configuration (from CY geometry)
- Optimize only: charge assignment (discrete) + brane positions (geometric)

**Success metric:** 12 continuous → 6 integer charges + 6 geometric distances

---

### TASK 4: Identify Neutrino Scales with Moduli

**Problem:** M_R, μ are arbitrary mass scales
**Solution:** Relate to modulus VEV and SUSY breaking

**Implementation:**

#### 4.1 Majorana Scale from Modulus
```python
def get_majorana_scale(tau, M_string=2e16):
    """
    Right-handed neutrino mass from modulus VEV.

    M_R ~ M_string × e^{-Re(τ)}

    This is GUT-scale suppressed by modulus.
    """
    suppression = np.exp(-tau.real)
    M_R = M_string × suppression  # GeV

    return M_R
```

#### 4.2 LNV Scale from SUSY Breaking
```python
def get_LNV_scale(m_gravitino, F_term):
    """
    Lepton number violation from SUSY breaking.

    μ ~ F / M_Pl ~ m_{3/2}^2 / M_Pl

    This explains why μ ≪ M_R naturally.
    """
    M_Planck = 1.22e19  # GeV
    mu = m_gravitino**2 / M_Planck

    return mu
```

**Code changes required:**
- Replace fitted M_R with modulus-dependent expression
- Replace fitted μ with SUSY-breaking expression
- Add m_gravitino as input (from SUSY sector)

**Success metric:** 2 fitted masses → 2 geometric expressions

---

## Implementation Timeline

### Week 1: CP Spurion (CRITICAL PATH)
- [ ] Day 1-2: Implement single spurion mechanism
- [ ] Day 3-4: Recompute CKM with spurion
- [ ] Day 5-7: Unify neutrino CP phase, validate predictions

**Deliverable:** CKM code using 1 spurion instead of 12 parameters

### Week 2: Yukawa Identification
- [ ] Day 8-9: Implement Kähler metric K(τ, g_s)
- [ ] Day 10-11: Add instanton actions
- [ ] Day 12-14: Recompute mass predictions, validate errors

**Deliverable:** Y₀ computed from geometry, not fitted

### Week 3: Discretization
- [ ] Day 15-17: Lock g_i to integer modular weights
- [ ] Day 18-20: Compute A_i from brane geometry
- [ ] Day 21: Validate mass hierarchy predictions

**Deliverable:** All localization parameters from discrete/geometric data

### Week 4: Neutrino Scales + Validation
- [ ] Day 22-23: Implement M_R(τ), μ(m_3/2)
- [ ] Day 24-26: Full system validation, error check
- [ ] Day 27-28: Parameter dictionary completion, documentation

**Deliverable:** Complete Phase 1, all parameters identified

---

## Success Metrics

### Quantitative
- [ ] Parameter count: 38 fitted → 38 identified
- [ ] Fitted degrees of freedom: 38 → ~5 (charges + spurion)
- [ ] All errors still <5%
- [ ] No new free parameters introduced

### Qualitative
- [ ] Every parameter has named geometric origin
- [ ] CP violation from single source
- [ ] Discrete/quantized where expected
- [ ] Clear path to Phase 2 computation

---

## Validation Checklist

After each task, verify:
- [ ] All 50 observables still predicted
- [ ] Maximum error still <5%
- [ ] No arbitrary normalizations
- [ ] No continuous parameters where discrete expected
- [ ] Clear geometric interpretation documented

---

## Phase 1 Completion Declaration

Phase 1 is complete when:

```
✅ All 38 parameters have geometric identity
✅ CP violation from single spurion (not 28 parameters)
✅ Yukawa normalizations from Kähler + instantons
✅ Localization from discrete charges + brane positions
✅ Neutrino scales from moduli + SUSY
✅ All errors <5%
✅ No unconstrained continuous parameters
✅ Clear boundary to Phase 2
```

Then and only then: **Phase 2 begins** (compute values from geometry).

---

## Next Steps

1. **Review this plan** - Are priorities correct?
2. **Start TASK 1** - CP spurion collapse (highest impact)
3. **Weekly checkpoints** - Track progress vs. metrics
4. **Document as you go** - Update parameter dictionary

**Estimated total work:** 80-120 hours focused coding + validation.

**Current status:** Plan approved, ready to implement.
