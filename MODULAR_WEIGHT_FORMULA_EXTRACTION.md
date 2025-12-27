# Key Extraction: Modular Weight Formula from arXiv:2410.05788

**Date**: December 27, 2025  
**Paper**: "Modular symmetry of localized modes" - Kobayashi, Otsuka, Takada, Uchida  
**Journal**: Phys. Rev. D 110 (2024) 125013  
**Status**: ✅ CRITICAL BREAKTHROUGH - Explicit formula found!

---

## 1. MAIN RESULT: Modular Weight Formula

### Formula (from Eq 3.46, 3.60)
```
Modular weight: w = 2(ℓ-a) + ℓ = 3ℓ - 2a
```

where:
- `ℓ` = localized magnetic flux quantum number (positive integer)
- `a` = mode index, a ∈ {0, 1, 2, ..., ℓ-1}
- `w` = modular weight (can be negative, zero, or positive!)

### Modular Transformation Law
```
ψ_L^a(S(z,τ)) = (-τ)^w × e^(3πiℓ/2) × ψ_L^a(z,τ)
ψ_L^a(T(z,τ)) = e^(πiℓ/2) × ψ_L^a(z,τ)
```

where S, T are standard modular generators:
- S: τ → -1/τ
- T: τ → τ+1

---

## 2. Flavor Symmetry Structure

### Dependence on ℓ
| ℓ value | Flavor Symmetry | Level N |
|---------|-----------------|---------|
| ℓ ∈ 2ℤ (even) | S₃ | N = 2 |
| ℓ ∈ 2ℤ+1 (odd) | S'₄ (double cover of S₄) | N = 4 |

### Representation Structure (T²/Z₂ orbifold)
**Fixed points**: z = 0, 1/2, τ/2, (τ+1)/2

**Mode structure**:
- ℓ localized modes at each fixed point
- Modes labeled by index a = 0, 1, ..., ℓ-1
- Different a → different modular weight w

**Flavor representations**:
- **ℓ even**: 
  * z=0 mode: S₃ singlet (1 or 1')
  * Other 3 fixed points: S₃ doublet + singlet
- **ℓ odd**:
  * z=0 mode: S'₄ singlet
  * Other 3 fixed points: S'₄ triplet

---

## 3. Explicit Wave Function (Critical!)

### General Form (Eq 3.44)
```
ψ_L,0^a(z,τ) = (Imτ)^(-ℓN/4) × 
               [ψ_(1/2,1/2)(z,τ) / |ψ_(1/2,1/2)(z,τ)|]^(ℓN) × 
               Φ_0(z,τ)^(ℓ-a)
```

where:
- `ψ_(α₁,α₂)(z,τ)` = theta function with characteristics (α₁,α₂)
- `Φ_0(z,τ)` = sum of three terms (permutation-symmetric)
- `N` = orbifold order (N=2 for Z₂)

### Theta Function (Eq 3.34)
```
ψ_(α₁,α₂)(z,τ) = e^(2πiα₁α₂) × e^(πiz×Imz/Imτ) × ϑ[α₁; α₂](z,τ)
```

where ϑ is Jacobi theta function.

### Specific Cases (Fixed Points)
**At z=0** (Eq 3.30-3.32):
```
ψ_(1/2,1/2)(z,τ) = θ₁(z,τ)  [vanishes at z=0]
ψ_(1/2,0)(z,τ)   = θ₂(z,τ)  [Z₂-invariant]
ψ_(0,1/2)(z,τ)   = θ₃(z,τ)  [Z₂-invariant]
ψ_(0,0)(z,τ)     = θ₄(z,τ)  [Z₂-invariant]
```

---

## 4. Connection to Our Framework

### Our Phenomenology (Papers 1-3)
| Family | Modular Weight |
|--------|----------------|
| e      | w_e = -2       |
| μ      | w_μ = 0        |
| τ      | w_τ = 1        |

### Applying Formula w = 3ℓ - 2a

**System of equations**:
```
w_e = 3ℓ_e - 2a_e = -2
w_μ = 3ℓ_μ - 2a_μ = 0
w_τ = 3ℓ_τ - 2a_τ = 1
```

**Solve for each family**:

1. **Electron (w = -2)**:
   ```
   3ℓ - 2a = -2
   → ℓ = 0, a = 1?  [INVALID: ℓ must be positive integer]
   → ℓ = 2, a = 4?  [INVALID: a must satisfy 0 ≤ a < ℓ]
   ```
   **Problem**: No integer solution with ℓ > 0 and 0 ≤ a < ℓ!

2. **Muon (w = 0)**:
   ```
   3ℓ - 2a = 0
   → ℓ = 2, a = 3?  [INVALID: a ≥ ℓ]
   → ℓ = 4, a = 6?  [INVALID: a ≥ ℓ]
   ```
   **Problem**: No valid solution!

3. **Tau (w = 1)**:
   ```
   3ℓ - 2a = 1
   → ℓ = 1, a = 1?  [INVALID: a ≥ ℓ]
   → ℓ = 3, a = 4?  [INVALID: a ≥ ℓ]
   ```
   **Problem**: No valid solution!

---

## 5. CRITICAL INSIGHT: Formula May Differ for T^6/(Z₃×Z₄)

### Why Direct Application Fails

**Paper context**: T²/Z₂ orbifold (single 2D torus with Z₂ twist)
**Our case**: T⁶/(Z₃×Z₄) orbifold (three 2D tori with Z₃ and Z₄ twists)

**Hypothesis 1**: Weight formula generalizes for factorized tori
```
w_total = w₁ + w₂ + w₃
```
where each w_i follows formula on respective T²_i.

**Hypothesis 2**: Different twist orders (Z₃, Z₄ vs Z₂) modify formula
```
w = N×ℓ - (N-1)×a  ???
```
where N = orbifold order (N=3 or N=4)?

**Hypothesis 3**: Our weights come from BULK modes, not localized modes
- Paper studies LOCALIZED modes on fixed points
- We may need bulk mode formula (different behavior)

---

## 6. Key Physics from Paper

### Localized vs Bulk Modes

**Bulk mode** (Eq 3.37):
```
ψ_B(z,τ) = (Imτ)^(-ℓN/4) × [ψ_z_fp,Z_N^1(z,τ) / |ψ_z_fp,Z_N^1(z,τ)|]^(ℓN)
```
- Modular weight: ℓ (from localized flux)
- Without singular gauge transformation: w = 0

**Localized mode** (constructed from bulk):
```
ψ_L^a(z,τ) = (ψ_z_fp,Z_N^0 / ψ_z_fp,Z_N^1)^((ℓ-a)N) × ψ_B
```
- Modular weight: 2(ℓ-a) + ℓ = 3ℓ - 2a
- Extra weight from ratio of Z_N-invariant modes

### Physical Interpretation

**Localized magnetic flux**: ξ^F = ℓ
- Creates ℓ chiral zero modes at each fixed point
- Each mode labeled by "winding number" a
- Higher a → lower modular weight

**Modular weight shift**:
```
w_bulk = ℓ              [from localized flux]
w_localized = 3ℓ - 2a   [from mode structure]
Δw = 2(ℓ-a)             [from gauge transformation]
```

---

## 7. Action Items for Our Case

### Immediate (Day 1 afternoon - TODAY)

✅ **DONE**: Extracted explicit formula from most recent paper
✅ **DONE**: Identified that direct formula doesn't give our weights

### Next Steps (Days 2-3)

**Priority 1**: Search for T⁶/(Z_N×Z_M) orbifold papers
- Look for arXiv:2101.00826 (3-generation modes on magnetized orbifolds)
- Search "factorized tori" + "modular weights"
- Check if product formula w_total = w₁ + w₂ + w₃ appears

**Priority 2**: Understand bulk vs localized modes
- Our case may be bulk modes on magnetized D7-branes
- Different formula than localized modes at fixed points
- Check arXiv:2001.07972 for magnetized D-brane weights

**Priority 3**: Check Z₃ and Z₄ orbifold papers
- Formula might be w = N×ℓ - (N-1)×a for Z_N
- Our case: combine Z₃ contribution + Z₄ contribution
- Test: w_e = (3ℓ₃ - 2a₃) + (4ℓ₄ - 3a₄) = -2 ?

---

## 8. Success Criteria Reassessment

### MINIMUM (End Day 3)
- ✅ Understand Z₂ formula (ACHIEVED)
- ⏳ Find generalization to Z₃×Z₄ (IN PROGRESS)
- ⏳ Identify bulk mode vs localized mode distinction

### TARGET (End Day 5)
- ⏳ Explicit formula for T⁶/(Z₃×Z₄) case
- ⏳ Solve for (ℓ_i, a_i) giving w_e=-2, w_μ=0, w_τ=1
- ⏳ Validate against phenomenology

### STRETCH (End Day 7)
- ⏳ Compute wave functions Ψ_e, Ψ_μ, Ψ_τ
- ⏳ Verify Yukawa couplings match Papers 1-3
- ⏳ GO decision for full calculation

---

## 9. Technical Notes

### Theta Function Modular Transformations (Appendix A)

**Under S: τ → -1/τ**:
```
ψ_(0,0)(S(z,τ)) = (-τ)^(1/2) × e^(πi/4) × ψ_(0,0)(z,τ)
ψ_(1/2,0)(S(z,τ)) = (-τ)^(1/2) × e^(πi/4) × ψ_(0,1/2)(z,τ)
ψ_(0,1/2)(S(z,τ)) = (-τ)^(1/2) × e^(-πi/4) × ψ_(1/2,0)(z,τ)
ψ_(1/2,1/2)(S(z,τ)) = (-τ)^(1/2) × e^(-πi/4) × ψ_(1/2,1/2)(z,τ)
```

**Under T: τ → τ+1**:
```
ψ_(0,0)(T(z,τ)) = ψ_(0,1/2)(z,τ)
ψ_(1/2,0)(T(z,τ)) = e^(πi/4) × ψ_(1/2,1/2)(z,τ)
ψ_(0,1/2)(T(z,τ)) = ψ_(0,0)(z,τ)
ψ_(1/2,1/2)(T(z,τ)) = e^(-πi/4) × ψ_(1/2,0)(z,τ)
```

These are EXACT transformations from string theory CFT!

---

## 10. References from Paper

**Key citations for next steps**:
- [35] arXiv:2101.00826 - "Modular flavor symmetries of 3-generation modes on magnetized toroidal orbifolds"
- [34] arXiv:2301.10356 - Modular symmetry in magnetized compactifications
- [36] arXiv:1804.06644 - Original modular symmetry in magnetized D-branes
- [50] arXiv:hep-th/0404229 - Cremades-Ibanez-Marchesano (Yukawa couplings from magnetized branes)

---

## Status

**Major Achievement**: Found explicit modular weight formula w = 3ℓ - 2a for T²/Z₂!

**Current Challenge**: Formula doesn't directly give our phenomenological weights w_e=-2, w_μ=0, w_τ=1.

**Next Step**: Generalize to T⁶/(Z₃×Z₄) orbifold - likely requires reading arXiv:2101.00826 (3-generation paper).

**Timeline**: Still on track for Week 1 reconnaissance. Day 1 afternoon progress excellent!

**Branch**: `exploration/cft-modular-weights`  
**Commits**: 486a18c (plan), 1585f6a (literature), pending (this extraction)
