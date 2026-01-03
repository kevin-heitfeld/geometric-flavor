# Generalization: Z₂ → Z₃×Z₄ Modular Weights

**Date**: December 28, 2025 (Day 2)  
**Branch**: `exploration/cft-modular-weights`  
**Goal**: Derive w_e=-2, w_μ=0, w_τ=1 from T⁶/(Z₃×Z₄) geometry

---

## 1. What We Learned from Z₂ Paper (arXiv:2410.05788)

### Formula for T²/Z₂ Localized Modes
```
w = 3ℓ - 2a
```
where:
- ℓ = localized magnetic flux quantum number
- a = mode index, a ∈ {0, 1, ..., ℓ-1}
- w = modular weight (even ℓ → S₃, odd ℓ → S'₄)

### Key Insight
Modular weight arises from:
1. **Flux contribution**: ℓ (creates ℓ zero modes at each fixed point)
2. **Mode index**: -2a (from gauge transformation structure)
3. **Total**: w = 2(ℓ-a) + ℓ = 3ℓ - 2a

---

## 2. Our Case: T⁶/(Z₃×Z₄) = (T²)³ Factorized

### Geometry
```
T⁶ = T²₁ × T²₂ × T²₃
```

**Orbifold action**:
- Z₃: θ₃ = (1/3, 1/3, -2/3) on (z₁, z₂, z₃)
- Z₄: θ₄ = (1/4, 1/4, -1/2) on (z₁, z₂, z₃)

**Each T²ᵢ has**:
- Complex structure modulus τᵢ
- Localized fluxes at fixed points
- Independent modular group action

### Hypothesis: Additive Weights

For fields localized on product space:
```
w_total = w₁ + w₂ + w₃
```

where each wᵢ follows appropriate formula for respective orbifold.

---

## 3. Generalized Formula for Z_N Orbifolds

### Conjecture
For T²/Z_N with localized flux:
```
w = N·ℓ - (N-1)·a  [CONJECTURE]
```

**Check against known Z₂ case**:
```
N=2: w = 2ℓ - 1·a = 2ℓ - a  [WRONG! Should be 3ℓ - 2a]
```

**Alternative formula**:
```
w = (N+1)·ℓ - 2a  [ALTERNATIVE CONJECTURE]
```

Check Z₂:
```
N=2: w = 3ℓ - 2a  [✓ MATCHES!]
```

Check Z₃:
```
N=3: w = 4ℓ - 2a
```

Check Z₄:
```
N=4: w = 5ℓ - 2a
```

---

## 4. Application to Our Phenomenology

### Target Weights
| Family | Weight |
|--------|--------|
| e      | w_e = -2 |
| μ      | w_μ = 0  |
| τ      | w_τ = 1  |

### Strategy 1: Single Torus Dominates

If lepton sector primarily from one T² (say T²₃ with Z₃ action):
```
w = 4ℓ - 2a  (for Z₃)
```

**Test electron (w=-2)**:
```
4ℓ - 2a = -2
→ 4ℓ = 2a - 2
→ 2ℓ = a - 1
```

For ℓ=1: a = 3 [INVALID: a must be < ℓ]
For ℓ=2: a = 5 [INVALID]

**PROBLEM**: Still no solution with positive ℓ!

### Strategy 2: Factorized Contributions

Lepton families distributed across three tori:
```
w_total = w₁(ℓ₁,a₁) + w₂(ℓ₂,a₂) + w₃(ℓ₃,a₃)
```

**Example decomposition**:
```
Electron: w_e = -2 = w₁ + w₂ + w₃
Muon:     w_μ = 0  = w₁' + w₂' + w₃'
Tau:      w_τ = 1  = w₁'' + w₂'' + w₃''
```

**Possible solution**:
```
w₁: Using 4ℓ₁ - 2a₁ = ?
w₂: Using 4ℓ₂ - 2a₂ = ?  
w₃: Using 5ℓ₃ - 2a₃ = ? (Z₄ contribution)
```

---

## 5. CRITICAL REALIZATION: Bulk vs Localized Modes

### Paper Context Re-examined

**arXiv:2410.05788** studies:
- **Localized modes** at orbifold fixed points
- Created by **localized magnetic flux** at fixed points
- Mode structure: ℓ modes at EACH fixed point

**Our case** (from existing code):
- **Magnetized D7-branes** wrapping cycles
- **Bulk magnetic flux** on entire brane worldvolume
- Zero modes from **Dirac equation** on magnetized brane

### Key Distinction

**Localized flux** (paper):
```python
# From arXiv:2410.05788
ψ_L^a(z,τ) = (singular_gauge_transform) × ψ_bulk
w = 3ℓ - 2a  (for Z₂)
```

**Bulk flux** (our case):
```python
# From our Papers 1-3
ψ_zero(z) = exp(πi F z̄ z) × θ(z|τ)
w = modular_weight_from_theta_function_characteristics
```

**This is THE critical difference!**

---

## 6. Modular Weights from Bulk Magnetized Branes

### Wave Function Structure (Cremades-Ibanez-Marchesano)

For D7-brane with magnetic flux M wrapping T²:
```
ψ_zero^(a)(z,τ) = N × exp(πiMz̄z/Imτ) × θ[α; β](Mz|τ)
```

where:
- M = magnetic flux quantum (integer)
- (α,β) = characteristics from orbifold quantum numbers
- a = 0, 1, ..., |M|-1 (degeneracy = |M|)

### Theta Function Characteristics

**Jacobi theta function**:
```
θ[α; β](z|τ) = Σ_n exp(πi(n+α)²τ + 2πi(n+α)(z+β))
```

**Modular transformation**:
```
θ[α; β](z/(cτ+d) | (aτ+b)/(cτ+d)) = χ(γ) × (cτ+d)^(1/2) × exp(πic z²/(cτ+d)) × θ[α'; β'](z|τ)
```

where χ(γ) is a phase.

### Modular Weight Extraction

For matter field ψ with characteristics (α,β):
```
ψ(γ(z,τ)) = (cτ+d)^w × ρ(γ) × ψ(z,τ)
```

**Weight w depends on**:
1. Theta function: contributes w = 1/2
2. Exponential prefactor: contributes additional weight from flux M
3. Orbifold quantum numbers (α,β): shift weight

**Generic formula** (from Ibanez-Uranga):
```
w = (1/2) + (flux_contributions) + (orbifold_shifts)
```

---

## 7. Connecting to Our Z₃×Z₄ Framework

### From Existing Code Analysis

**Our identified geometry** (from `identify_calabi_yau.py`):
```python
T⁶/(Z₃ × Z₄) with magnetized D7-branes
- Quarks on 4-cycle → Γ₀(4)
- Leptons on 3-cycle → Γ₀(3)
- χ = -6 → 3 generations
```

**Orbifold twists** (from `CALABI_YAU_IDENTIFIED.md`):
```
Z₃: θ₃ = (1/3, 1/3, -2/3)
Z₄: θ₄ = (1/4, 1/4, -1/2)
```

**Modular weights in code** (from `theory14_modular_weights.py`):
```python
# Phenomenological assignment
w_e = -2
w_μ = 0
w_τ = 1

# Used to construct Yukawa matrices via:
Y = modular_form(tau, weight=w)
```

### The Question

**Can we DERIVE these weights from**:
1. D7-brane wrapping numbers?
2. Magnetic flux quanta M_i?
3. Orbifold quantum numbers from Z₃×Z₄?

---

## 8. Action Plan for Week 1 Completion

### Days 2-3: Read Key Papers (IN PROGRESS)

**Priority papers**:
1. ✅ arXiv:2410.05788 - Localized modes (COMPLETED)
2. ⏳ arXiv:2101.00826 - 3-generation modes (TODAY)
3. ⏳ arXiv:2001.07972 - Type IIB flux vacua
4. ⏳ arXiv:hep-th/0404229 - Cremades-Ibanez-Marchesano (Yukawa from magnetized branes)

**Extract**:
- Bulk mode modular weight formula
- Orbifold quantum number → (α,β) characteristics
- How Z₃ and Z₄ twists affect weights

### Days 4-5: Synthesis

**Construct explicit formulas**:
```
w_i = f(M_i, α_i, β_i, orbifold_twists)
```

**Test against phenomenology**:
```
Does f give (-2, 0, 1) for reasonable (M, α, β)?
```

### Days 6-7: Feasibility Decision

**If YES**:
- Compute full wave functions
- Verify Yukawa couplings
- GO decision for Weeks 2-4 calculation

**If NO**:
- Document why weights must remain phenomenological
- Update Paper 4 Discussion section
- Pivot to phenomenology Papers 5-7

---

## 9. Hypothesis Summary

### Hypothesis A: Localized vs Bulk

**Claim**: Paper formula w=3ℓ-2a is for **localized modes**, but we need **bulk mode** formula.

**Test**: Find Cremades-Ibanez-Marchesano formula for bulk magnetized branes.

**Prediction**: Bulk formula will involve M (flux) and (α,β) (characteristics), not (ℓ,a).

### Hypothesis B: Factorized Weights

**Claim**: For T⁶ = (T²)³, weights are **additive**:
```
w_total = w₁(Z₃ sector) + w₂(Z₄ sector) + w₃(common)
```

**Test**: Each sector contributes fractional weight, sum gives integer.

**Prediction**: 
```
w_e = w₁^e + w₂^e = (-1) + (-1) = -2
w_μ = w₁^μ + w₂^μ = (0) + (0) = 0
w_τ = w₁^τ + w₂^τ = (1/2) + (1/2) = 1
```

### Hypothesis C: Orbifold Shifts

**Claim**: Z₃ and Z₄ twists **shift** base weights by fractional amounts related to twist eigenvalues.

**Test**: Orbifold quantum numbers q₃ ∈ {0,1,2} and q₄ ∈ {0,1,2,3} give:
```
w_shift = (q₃/3) + (q₄/4) [mod Z]
```

**Prediction**:
```
Electron: q₃=2, q₄=0 → shift = 2/3 → base=-2-2/3, round to -2
Muon:     q₃=0, q₄=0 → shift = 0   → base=0
Tau:      q₃=1, q₄=0 → shift = 1/3 → base=2/3, round to 1
```

---

## 10. Literature Search Strategy

### Key Papers to Find

1. **Cremades-Ibanez-Marchesano (2004)**: "Computing Yukawa Couplings from Magnetized Extra Dimensions"
   - arXiv:hep-th/0404229
   - Should have **exact formula** for modular weights of bulk modes

2. **Blumenhagen et al.**: D-branes on orbifolds with modular symmetry
   - Look for "modular weight" + "orbifold" + "magnetized"

3. **Kobayashi group reviews**: 
   - arXiv:2307.03384 - "Modular flavor models" (review, 40 pages)
   - Should synthesize bulk vs localized distinction

### Search Terms

- "modular weight" + "magnetized D-branes" + "bulk modes"
- "theta function characteristics" + "orbifold" + "Z_3" OR "Z_4"
- "Yukawa couplings" + "modular transformation" + "wave functions"

---

## Status

**Day 2 Morning**: 
- ✅ Identified bulk vs localized mode distinction (CRITICAL!)
- ✅ Formulated three testable hypotheses (A, B, C)
- ✅ Defined clear action plan for Days 2-7

**Next Step**: 
Search for Cremades-Ibanez-Marchesano paper (arXiv:hep-th/0404229) to get bulk mode formula.

**Timeline**: On track. The bulk/localized distinction is exactly the kind of insight needed to break through Wall #1!

**Branch**: `exploration/cft-modular-weights`  
**Last Commit**: 4e30ff4
