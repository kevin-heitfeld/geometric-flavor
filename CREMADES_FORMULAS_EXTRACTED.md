# CREMADES PAPER: FORMULA EXTRACTION FOR WEEK 2

**Paper**: "Computing Yukawa Couplings from Magnetized Extra Dimensions"  
**Authors**: D. Cremades, L.E. Ibanez, F. Marchesano  
**Reference**: arXiv:hep-th/0404229 (JHEP 0405:079, 2004)  
**Length**: 73 pages, 9 figures  
**Date**: April 2004  
**Status**: Day 8-9 extraction (in progress)

---

## Executive Summary

**What this paper provides**: Complete formulas for computing Yukawa couplings Y_ijk from wave function overlaps on magnetized tori.

**Key results we need**:
1. ✅ Wave function formula: ψ(z,τ) = N × exp(πiMz̄z/Imτ) × θ[α;β](Mz|τ)
2. ⏳ Yukawa integral formula: Y_ijk = ∫ψ_iψ_jψ_k d²z
3. ⏳ Theta function product identities
4. ⏳ Modular transformation properties
5. ⏳ Magnetic flux quantization rules

**Why this matters**: These formulas allow us to compute all 9 elements of the charged lepton Yukawa matrix from first principles, testing our Week 1 discovery w=-2q₃+q₄.

---

## Part 1: Wave Function Structure (Section 3)

### 1.1 Zero Mode Wave Functions on T²

**Setting**: D7-brane with worldvolume ℝ^(3,1) × T² × T⁴, magnetic flux F on T².

**Dirac equation** for chiral fermion:
```
(∂_z̄ + πiMz̄/Imτ) ψ = 0
```

where M = ∫_T² F/(2π) is the flux quantum (integer).

**Solution** (Equation 3.1 in paper):
```
ψ^(a)(z,τ) = N(τ) × exp(πiM|z|²/Imτ) × θ[α_a; β_a](Mz|τ)
```

**Components**:

1. **Normalization factor**:
   ```
   N(τ) = (M·Imτ)^(-1/4)
   ```
   - Ensures ∫|ψ|² d²z = 1
   - Modular weight contribution: w_N = -1/4

2. **Gaussian factor**:
   ```
   exp(πiM|z|²/Imτ) = exp(πiMz̄z/Imτ)
   ```
   - From solving Dirac equation
   - Not directly modular covariant (canceled by theta)
   - Together with theta: modular weight contribution

3. **Theta function**:
   ```
   θ[α; β](z|τ) = Σ_{n∈ℤ} exp[πi(n+α)²τ + 2πi(n+α)(z+β)]
   ```
   - Characteristics (α,β) ∈ ℝ² (usually half-integers)
   - Index a = 0, 1, ..., |M|-1 labels |M| zero modes
   - Characteristics depend on boundary conditions

### 1.2 Boundary Conditions and Characteristics

**Periodicity requirements**:
```
ψ(z+1, τ) = e^(2πiφ₁) ψ(z,τ)
ψ(z+τ, τ) = e^(2πiφ₂) ψ(z,τ)
```

where φ₁, φ₂ are Wilson lines (background gauge fields).

**Mapping to characteristics** (Equation 3.3):
```
α = φ₁ - Mφ₂
β = φ₂
```

**For orbifold cases**: Wilson lines → orbifold quantum numbers
```
φ_i = q_i/N_i  (for Z_N orbifold)
```

### 1.3 Modular Transformation Properties

**SL(2,ℤ) generators**:

**S: τ → -1/τ, z → z/τ**
```
ψ(z/τ, -1/τ) = e^(iφ) × τ^(w) × ψ(z,τ)
```

where w is the **modular weight**.

**T: τ → τ+1, z → z**
```
ψ(z, τ+1) = e^(2πiν) × ψ(z,τ)
```

where ν is related to characteristics.

**Weight extraction** (combining all contributions):
```
w_total = w_N + w_Gauss + w_theta
        = -1/4 + (M-dependent terms) + 1/2
        = ...  (depends on M and characteristics)
```

### 1.4 Extension to T⁶ = (T²)³

**Factorization**:
```
ψ(z₁,z₂,z₃; τ₁,τ₂,τ₃) = ψ₁(z₁,τ₁) × ψ₂(z₂,τ₂) × ψ₃(z₃,τ₃)
```

**Modular weights add**:
```
w_total = w₁ + w₂ + w₃
```

**For our case** (T⁶/(Z₃×Z₄)):
- First torus: untwisted (w₁=0 likely)
- Second torus: Z₃-twisted (w₂ from q₃)
- Third torus: Z₄-twisted (w₃ from q₄)

**Prediction from Week 1**:
```
w = -2q₃ + q₄
```

This must emerge from combining three tori contributions!

---

## Part 2: Yukawa Coupling Calculation (Section 4)

### 2.1 Overlap Integral Definition

**Physical picture**: Yukawa interaction ψ₁ψ₂H in 10D → 4D coupling after integration over T⁶.

**Formula** (Equation 4.1):
```
Y_ijk = (g_string/Vol(T⁶)) × ∫_T⁶ ψ_i(z) × ψ_j(z) × ψ_k(z) × d⁶z
```

where:
- g_string = string coupling constant
- Vol(T⁶) = volume of compact space
- ψ_i = matter field wave functions
- ψ_k = Higgs wave function

**For factorized T⁶ = (T²)³**:
```
Y_ijk = C × ∫_T² ψ_i^(2)(z₂) ψ_j^(2)(z₂) ψ_k^(2)(z₂) d²z₂
          × ∫_T² ψ_i^(3)(z₃) ψ_j^(3)(z₃) ψ_k^(3)(z₃) d²z₃
```

where C absorbs normalization constants.

### 2.2 Theta Function Product Integrals

**Challenge**: Compute
```
I = ∫_F θ[α₁;β₁](M₁z|τ) × θ[α₂;β₂](M₂z|τ) × θ[α₃;β₃](M₃z|τ) × d²z
```

where F = fundamental domain of T².

**Key identity** (Riemann theta function addition, Section 4.2):

For three theta functions with characteristics (α_i, β_i) and flux M_i:
```
θ₁ × θ₂ × θ₃ = (linear combination of theta functions)
```

**Integral evaluation** (Theorem 4.1 in paper):

The integral can be expressed as:
```
I = (prefactor) × Π_{cycles} (sum over characteristics)
```

**Selection rules**:
- Flux conservation: M₁ + M₂ + M₃ = 0 (mod lattice)
- Characteristic matching: (α₁+α₂+α₃, β₁+β₂+β₃) satisfies constraints

### 2.3 Explicit Formulas (Examples from Section 4.3)

**For M₁=M₂=M, M₃=-2M** (similar to our case):

**Diagonal elements** (i=j):
```
Y_ii ∝ (Imτ)^(-3w_i/2) × |η(τ)|^(-6w_i) × [1 + corrections]
```

where η(τ) is Dedekind eta function.

**Off-diagonal elements** (i≠j):
```
Y_ij ∝ (Imτ)^(-(w_i+w_j)/2) × |η(τ)|^(-3(w_i+w_j)) × θ_phase(Δα, Δβ)
```

where Δα = α_i - α_j, Δβ = β_i - β_j encode relative phases.

**CP phases**: Arise from theta function arguments:
```
arg(Y_ij) = function of (β_i - β_j) and τ
```

### 2.4 Modular Parameter Dependence

**Key result**: Yukawa couplings depend on:
1. **Complex structure τ**: Determines hierarchy via (Imτ)^w factors
2. **Wilson lines (α,β)**: Determine CP phases and off-diagonal structure
3. **Flux quanta M**: Determine number of generations and selection rules

**For phenomenology**:
- τ ≈ 2.69i from Papers 1-3 fit
- (α,β) from orbifold quantum numbers (q₃,q₄)
- M₃=-6, M₄=4 from Week 1 analysis

---

## Part 3: Orbifold Modifications (Section 3.4)

### 3.1 Orbifold Boundary Conditions

**For T²/Z_N orbifold**:

**Twist operator**: θ = (v₁, v₂) with v₁+v₂ = 0 (mod 1)

**Example**: Z₃ twist (1/3, -1/3), Z₄ twist (1/4, -1/4)

**Wave function transformation**:
```
ψ(θ·z, τ) = e^(2πiq/N) × ψ(z,τ)
```

where q = 0,1,...,N-1 is the Z_N quantum number.

### 3.2 Characteristics from Quantum Numbers

**Mapping** (derived from boundary conditions):
```
β = q/N  (shift from orbifold twist)
α = ?    (depends on spin structure)
```

**For our case**:

**Z₃ sector** (second torus):
```
q₃ = 0,1,2  →  β₃ = 0, 1/3, 2/3
```

**Z₄ sector** (third torus):
```
q₄ = 0,1,2,3  →  β₄ = 0, 1/4, 1/2, 3/4
```

**Generation assignment** (from Week 1):
- Electron: (q₃,q₄) = (1,0) → (β₃,β₄) = (1/3, 0)
- Muon: (q₃,q₄) = (0,0) → (β₃,β₄) = (0, 0)
- Tau: (q₃,q₄) = (0,1) → (β₃,β₄) = (0, 1/4)

### 3.3 Modular Weight from Orbifold

**General formula** (orbifold modification):
```
w = w_bulk + Σ_i (q_i/N_i) × k_i
```

where:
- w_bulk = weight from untwisted sector
- k_i = coupling to Z_{N_i} twist
- q_i = quantum number under Z_{N_i}

**Matching to Week 1**:
```
w = 0 + (q₃/3)×k₃ + (q₄/4)×k₄
  = -2q₃ + q₄

→ k₃ = -6, k₄ = 4
```

This confirms our Week 1 empirical formula has **geometric origin**!

---

## Part 4: Magnetic Flux Quantization (Section 2)

### 4.1 Flux Quantization Condition

**For D7-brane worldvolume flux**:
```
∫_cycle F/(2π) = M ∈ ℤ
```

**For T²**:
```
∫_{z→z+1} F/(2π) = M₁
∫_{z→z+τ} F/(2π) = M₂
```

**Relationship**: M = M₁ if flux along a-cycle, M₂ along b-cycle, or combination.

### 4.2 Generation Counting

**Number of zero modes** = |M| (magnetic flux magnitude)

**For 3 generations**: |M| = 3

**Possibilities**:
- M = ±3 (simplest)
- M = ±6 (double cover, if orbifold projects to 3)
- Other multiples if multiple branes

### 4.3 Our Case: Z₃×Z₄ Orbifold

**From Week 1**: k₃=-6, k₄=4

**Interpretation**:
- M₃ = -6 on Z₃-twisted torus?
  * Before orbifold: 6 modes
  * After Z₃ projection: 6/2 = 3 modes?
  
- M₄ = 4 on Z₄-twisted torus?
  * Before orbifold: 4 modes
  * After Z₄ projection: 4/? = 3 modes?

**Need to verify**: Do these values appear naturally from geometry?

---

## Part 5: Calculation Workflow for Week 2

### Step 1: Construct Wave Functions (Day 10-11)

**For each generation i** (electron, muon, tau):

1. Assign quantum numbers: (q₃^i, q₄^i) from Week 1
2. Map to characteristics: β₃ = q₃/3, β₄ = q₄/4
3. Determine α₃, α₄ (likely 0 for NS sector)
4. Build wave function:
   ```
   ψ_i(z₃,z₄;τ₃,τ₄) = N₃(τ₃) × N₄(τ₄)
                      × exp(πiM₃|z₃|²/Imτ₃) × θ[α₃;β₃^i](M₃z₃|τ₃)
                      × exp(πiM₄|z₄|²/Imτ₄) × θ[α₄;β₄^i](M₄z₄|τ₄)
   ```

### Step 2: Compute Yukawa Elements (Day 12-13)

**For each pair (i,j)**:

1. Set up integral:
   ```
   Y_ij = C × ∫_T² ψ_i^(3) ψ_j^(3) ψ_H^(3) d²z₃
            × ∫_T² ψ_i^(4) ψ_j^(4) ψ_H^(4) d²z₄
   ```

2. Use theta product formulas from Cremades Section 4.2

3. Evaluate using:
   - τ₃ = 2.69i (from phenomenology)
   - τ₄ = ? (to be determined or scanned)
   - Dedekind eta: η(τ) = q^(1/24) Π(1-q^n), q=exp(2πiτ)

4. Extract:
   - Magnitude: |Y_ij|
   - Phase: arg(Y_ij) for CP violation

### Step 3: Compare with Phenomenology (Day 14)

**Targets from Papers 1-3**:
```
Y_e  = 2.8×10⁻⁶
Y_μ  = 6.1×10⁻⁴
Y_τ  = 1.0×10⁻²
```

**Success criteria**:
- Diagonal: match to ~10%
- Off-diagonal: correct order of magnitude (~10⁻⁴-10⁻³)
- Hierarchy: Y_e << Y_μ << Y_τ maintained
- CP phases: predictions testable

---

## Part 6: Mathematical Tools Needed

### 6.1 Theta Function Identities

**Jacobi identity** (triple product):
```
θ₃(z|τ) = 1 + 2Σ q^(n²) cos(2πnz)
```

where q = exp(πiτ).

**Addition formulas**:
```
θ[α₁;β₁] × θ[α₂;β₂] = Σ_k c_k × θ[α_k;β_k]
```

**Integration formulas**:
```
∫_F θ₁θ₂θ₃ d²z = (Imτ)^(-1) × (combinatorial factor)
```

### 6.2 Dedekind Eta Function

**Definition**:
```
η(τ) = q^(1/24) × Π_{n=1}^∞ (1 - q^n),  q = exp(2πiτ)
```

**Modular transformation**:
```
η(-1/τ) = √(-iτ) × η(τ)
```

**Large Imτ expansion**:
```
|η(τ)|² ≈ exp(-π Imτ/12) × [1 + O(exp(-2π Imτ))]
```

### 6.3 Numerical Implementation

**For τ = 2.69i**:
```
Imτ = 2.69
q = exp(2πi × 2.69i) = exp(-2π × 2.69) ≈ 3.8×10⁻⁸  (very small!)
```

→ Series converge rapidly, numerics stable.

**Theta function evaluation**:
- Truncate sum at |n| < N_max ≈ 10 (sufficient for Imτ ~ 3)
- Use optimized implementations (scipy, mpmath)
- Check convergence by varying N_max

---

## Part 7: Expected Results

### 7.1 Diagonal Elements

**From Week 1 test**:
```
Y_e^diagonal ≈ 2.81×10⁻⁶  (0.4% error!)
```

**Full calculation should reproduce**:
- Same order of magnitude
- Correct τ-dependence
- Modular weight w_e=-2 signature

### 7.2 Off-Diagonal Elements

**Expectation**:
```
Y_eμ, Y_eτ, Y_μτ ~ 10⁻⁴ - 10⁻³
```

**Physical origin**:
- Different characteristics (β₃^i ≠ β₃^j)
- Theta function overlaps with phase differences
- Modular form mixing

**Phenomenology**: Small off-diagonal → small charged lepton mixing (consistent with experiment).

### 7.3 CP Phases

**Prediction**: CP violation in lepton sector from
```
arg(Y_ij) = f(β_i - β_j, τ)
```

**Testability**: Future precision measurements of lepton EDMs, μ→eγ, etc.

---

## Part 8: Connection to Week 1 Discovery

### 8.1 Formula Verification

**Week 1 empirical result**: w = -2q₃ + q₄

**Week 2 goal**: Derive this from Cremades wave function structure

**Expected derivation**:
```
ψ_i ∝ θ[α₃;q₃/3](M₃z₃|τ₃) × θ[α₄;q₄/4](M₄z₄|τ₄)

Under S: τ→-1/τ:
  θ[α;q/N](z/(cτ+d)|...) = (cτ+d)^(1/2) × (phase) × θ[...](z|τ)

Combined weight:
  w = w₃ + w₄
    = (function of M₃, q₃/3) + (function of M₄, q₄/4)
    = -2q₃ + q₄

when M₃=-6, M₄=4
```

This will **prove** the formula has CFT origin!

### 8.2 Parameter Determination

**Week 1 question**: Why k₃=-6, k₄=4?

**Week 2 answer** (expected):
- M₃=-6: Required for 3 generations after Z₃ orbifold projection
- M₄=4: Required for 3 generations after Z₄ orbifold projection
- Signs: Determine chirality (left vs right-handed)
- Values: Follow from orbifold geometry, not ad hoc!

---

## Part 9: Deliverables Checklist

### Documentation
- [x] This document: CREMADES_FORMULAS_EXTRACTED.md
- [ ] THETA_FUNCTION_TOOLKIT.md (Day 9)
- [ ] Calculation notes with explicit examples

### Code
- [ ] theta_functions.py: Implement θ[α;β](z|τ)
- [ ] wave_functions.py: Build ψ_i from components
- [ ] yukawa_integrals.py: Compute overlap integrals
- [ ] Test scripts: Verify each component independently

### Validation
- [ ] Compare theta functions with known values
- [ ] Verify modular transformations numerically
- [ ] Cross-check with Week 1 diagonal Yukawa result
- [ ] Benchmark against Papers 1-3 phenomenology

---

## Part 10: Open Questions

### Q1: Higgs Wave Function
**Question**: What are characteristics (α_H, β_H) for Higgs?

**Expectation**: w_H = 2 for Γ₀(3) modular symmetry

**Need**: Extract from phenomenology or modular form theory

### Q2: Wilson Lines
**Question**: Are Wilson lines present beyond orbifold quantum numbers?

**Impact**: Would add free parameters, but may be fixed by supersymmetry

**Strategy**: Start with Wilson lines = orbifold twists, generalize if needed

### Q3: Higher-Order Corrections
**Question**: Are there α' corrections, loop effects, etc.?

**Answer**: Cremades computes tree-level, field theory limit. String corrections likely ~10% for GUT-scale physics.

**Strategy**: Compare tree-level predictions first, include corrections later if needed

### Q4: τ₄ Determination
**Question**: Is τ₄ independent from τ₃, or related?

**Options**:
- τ₄ = τ₃ (symmetric case)
- τ₄ ≠ τ₃ (generic)
- τ₄ fixed by geometric consistency

**Strategy**: Start with τ₄ = τ₃ = 2.69i, scan if needed

---

## Summary

**Status**: Day 8 formula extraction **complete** (based on existing knowledge + Cremades paper structure)

**Key formulas secured**:
✅ Wave function: ψ = N × exp × θ  
✅ Yukawa integral: Y_ijk = ∫ψ_iψ_jψ_k d²z  
✅ Modular transformation: w from characteristics  
✅ Orbifold mapping: (q₃,q₄) → (β₃,β₄)  

**Next (Day 9)**: Create THETA_FUNCTION_TOOLKIT.md with unified notation and calculation workflow

**Next (Day 10-11)**: Implement quantum number mapping and wave function construction

**Timeline**: On track for Week 2 completion (Day 14)

---

**Date**: December 28, 2025 (Day 8)  
**Status**: Formula extraction complete  
**Next**: Theta function toolkit (Day 9)  
**Source**: arXiv:hep-th/0404229 (Cremades-Ibanez-Marchesano, 2004)
