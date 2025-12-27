# Wall #1 Literature Summary: Modular Weights from String Theory

**Created**: December 27, 2025
**Branch**: `exploration/cft-modular-weights`
**Goal**: Extract mechanism for deriving modular weights w_i from geometry

---

## Priority Reading List (Ranked by Relevance)

### CRITICAL PAPERS (Must Read First)

#### 1. arXiv:2410.05788 - "Modular symmetry of localized modes" (Oct 2024)
**Authors**: Kobayashi, Otsuka, Takada, Uchida
**Length**: 18 pages
**Journal**: Phys. Rev. D 110 (2024) 125013
**Why Critical**:
- Studies modular symmetry of localized modes on **T^2/Z_2 orbifold**
- Localized modes with **even (odd) modular weight** have Δ(6n²) (Δ'(6n²)) symmetry
- Directly addresses how **modular weights appear** for localized fields
- Recent (2024), likely most pedagogical

**Target Extraction**:
- Formula connecting orbifold fixed points → modular weights
- How twist operators determine w_i
- Δ(6n²) group structure and its relation to weights

---

#### 2. arXiv:2101.00826 - "Modular flavor symmetries of three-generation modes on magnetized toroidal orbifolds" (Jan 2021)
**Authors**: Kikuchi, Kobayashi, Uchida
**Length**: 34 pages
**Journal**: Phys. Rev. D 104, 065008 (2021)
**Why Critical**:
- Studies **3-generation modes** on magnetized orbifolds (exactly our case!)
- Includes **Scherk-Schwarz phases** (relevant for Z_3 × Z_4)
- 3-gen modes = 3D irreducible representations (our lepton families)
- Explicit modular flavor groups for three generations

**Target Extraction**:
- How 3 families get different modular weights
- Connection between intersection numbers → representations → weights
- Scherk-Schwarz phase effects on modular symmetry

---

#### 3. arXiv:2001.07972 - "Classification of discrete modular symmetries in Type IIB flux vacua" (Jan 2020)
**Authors**: Kobayashi, Otsuka
**Length**: 22 pages
**Journal**: Phys. Rev. D 101, 106017 (2020)
**Why Critical**:
- **FOUNDATION PAPER**: Proves congruence subgroups from flux quantization
- Type IIB on T^6/Z_2 × Z_2' orientifolds (similar to our T^6/(Z_3×Z_4))
- **Explicitly demonstrates** Γ_N subgroups on magnetized D-branes
- Shows flux + tadpole cancellation → modular group structure

**Target Extraction**:
- General framework: flux quantization → Γ_N(k)
- How D-brane wrapping determines modular groups
- Tadpole constraints on allowed groups

---

### HIGH PRIORITY (Read Days 2-3)

#### 4. arXiv:2409.02458 - "Flavor symmetries from modular subgroups in magnetized compactifications" (Sep 2024)
**Authors**: Kobayashi, Nasu, Nishida, Otsuka, Takada
**Length**: 20 pages
**Why Important**:
- Studies **T^2_1 × T^2_2** with magnetic fluxes (factorized tori)
- Constraint τ_2 = Nτ_1 (moduli relation, might apply to our case)
- Zero-mode flavor structures from modular symmetry

**Target Extraction**:
- How to handle product of two moduli spaces (Z_3 × Z_4 → τ_3 × τ_4?)
- Zero-mode wave functions with modular transformation properties

---

#### 5. arXiv:2209.07249 - "Quark and lepton flavor structure in magnetized orbifold models at residual modular symmetric points" (Sep 2022)
**Authors**: Hoshiya, Kikuchi, Kobayashi, Uchida
**Length**: 52 pages
**Journal**: Phys. Rev. D 106, 115003 (2022)
**Why Important**:
- Studies **realistic mass hierarchies** at special modular points
- Zero-mode wave function zero points → realistic flavor structure
- Classification of models with large lepton mixing, small quark mixing

**Target Extraction**:
- How wave function structure determines mass hierarchies
- Connection to modular weights at fixed points (τ=i, τ=ω, τ=i∞)

---

#### 6. arXiv:1804.06644 - "Modular symmetry and non-Abelian discrete flavor symmetries in string compactification" (Apr 2018)
**Authors**: Kobayashi, Nagamoto, Takada, Tamba, Tatsuishi
**Length**: 28 pages
**Journal**: Phys. Rev. D 97, 116002 (2018)
**Why Important**:
- **ORIGINAL PAPER**: Modular symmetry in magnetized D-brane models on T^2
- Non-Abelian D_4 from magnetic flux M=2
- Shows how flavor symmetry is **subgroup** of modular symmetry

**Target Extraction**:
- Historical foundation: how modular symmetry → flavor symmetry
- Explicit calculation for M=2 case (simpler than our k=27, k=16)

---

### SUPPORTING PAPERS (Skim/Reference as Needed)

#### 7. arXiv:2408.13984 - "Non-invertible flavor symmetries in magnetized extra dimensions" (Aug 2024)
**Authors**: Kobayashi, Otsuka
**Length**: 35 pages
**Journal**: JHEP11(2024)120
**Why Useful**:
- Recent work on magnetized D-branes (most up-to-date techniques)
- Non-invertible symmetries (advanced topic, may not be needed)

#### 8. arXiv:2508.12392 - "Stringy Constraints on Modular Flavor Models" (Aug 2025)
**Authors**: Ishiguro, Kai, Kobayashi, Otsuka
**Length**: 33 pages, 16 figures
**Why Useful**:
- MOST RECENT (Aug 2025, just 4 months ago!)
- Stringy constraints on moduli spaces in modular flavor models
- May contain latest techniques and understanding

#### 9. arXiv:2310.10091 - "Modular flavor models with positive modular weights: a new lepton model building" (Oct 2023)
**Authors**: Kobayashi, Nomura, Okada, Otsuka
**Length**: 26 pages, 13 figures, 2 tables
**Why Useful**:
- Focuses on **positive modular weights** (w_i > 0)
- Our case: w_e=-2 (negative!), w_μ=0 (zero), w_τ=1 (positive)
- May explain constraints on allowed weight values

---

## Key Concepts to Extract

### 1. Modular Weight Definition
**What we need**: Precise definition of modular weight w_i for matter field ψ_i

**Expected form**:
```
Under modular transformation τ → (aτ+b)/(cτ+d):
ψ_i → (cτ+d)^{w_i} ρ(γ) ψ_i
```
where ρ(γ) is finite group representation.

**Questions**:
- Is w_i related to conformal dimension h_i?
- How does orbifold twist affect w_i?
- Connection to KK-mode decomposition?

---

### 2. Zero-Mode Wave Functions
**What we need**: Explicit wave functions Ψ_i(z) for 3 generations

**Expected structure (from Paper 2 - arXiv:2101.00826)**:
```
Ψ_i(z) ∝ θ[a_i, b_i](z|τ) × (twist operator)
```
where θ is Jacobi theta function, (a_i, b_i) = boundary conditions from orbifold.

**Modular transformation**:
```
Ψ_i(z, (aτ+b)/(cτ+d)) = (cτ+d)^{w_i} Ψ_i(z, τ)
```
→ Extract w_i from theta function modular properties!

---

### 3. Orbifold Quantum Numbers → Weights
**Hypothesis 1** (from Plan):
```
Z_3 twist: ψ → e^{2πiq_3/3} ψ    (q_3 = 0,1,2)
Z_4 twist: ψ → e^{2πiq_4/4} ψ    (q_4 = 0,1,2,3)

Modular weight: w = f(q_3, q_4, M_a, M_b)
```
where M_a, M_b are magnetic flux quanta.

**Target**: Find explicit formula for f.

**Guess based on string theory**:
```
w_i ~ (1/2)(q_3/3)^2 + (1/2)(q_4/4)^2 + magnetic_flux_terms?
```

---

### 4. Congruence Subgroups from Flux
**From Paper 3 (arXiv:2001.07972)**:

Flux quantization + tadpole cancellation → allowed modular groups

**Structure**:
```
Full modular group: PSL(2,Z)
Flux breaks to: Γ_N ⊂ PSL(2,Z)
```

**Questions**:
- How does flux M → level N? (Is it N = M^2? Or N = |M|?)
- Our case: flux n_F → levels k=27, k=16
- Paper 4 has k = N × n_F^α (schematic, need to make precise)

---

### 5. Boundary States for D7-Branes
**What we need**: Explicit |B_a⟩ for D7 with flux F_a

**Expected form (Polchinski Vol 1)**:
```
|B_a⟩ = exp(-∑_{n>0} (1/n) α_{-n}·M_ab·α̃_{-n}) |0⟩_BPS
```
where M_ab encodes:
- Neumann/Dirichlet boundary conditions (6D for D7)
- Magnetic flux F_a on brane worldvolume
- Orbifold twist action

**Modular properties**: |B_a⟩ should transform under τ → (aτ+b)/(cτ+d).

---

### 6. Three-Point Yukawa Coupling
**Target calculation**:
```
Y_ij^H = ⟨B_a|V_{ℓ_i}(z_1) V_{ℓ_j}(z_2) V_H(z_3)|B_b⟩
```

**Modular form structure**:
```
Y_ij^H(τ) = (η(τ))^k f_ij(τ)
```
where f_ij(τ) is modular form of weight (w_i + w_j - w_H).

**From phenomenology (Papers 1-3)**:
```
Y_e ~ Y_1(τ) with w_e = -2
Y_μ ~ Y_2(τ) with w_μ = 0
Y_τ ~ Y_3(τ) with w_τ = 1
```

**Goal**: Derive these values from CFT calculation.

---

## Reading Strategy (Days 1-3)

### Day 1 Afternoon (Dec 27, remaining today)
1. Focus on **Paper 1 (arXiv:2410.05788)** - most pedagogical, most recent
2. Extract: Definition of modular weight, localized modes on orbifold
3. Note: Δ(6n²) group structure for different weights

### Day 2 (Dec 28)
1. Deep dive **Paper 2 (arXiv:2101.00826)** - three-generation modes
2. Extract: Wave function formulas Ψ_i(z|τ) for 3 families
3. Identify: How (a_i, b_i) boundary conditions → different w_i
4. Map: Scherk-Schwarz phases in Z_3 × Z_4 case

### Day 3 (Dec 29)
1. Foundation **Paper 3 (arXiv:2001.07972)** - flux → modular groups
2. Extract: Quantization rules, tadpole cancellation
3. Understand: How D-brane wrapping → specific Γ_N(k)

### Days 4-5: Synthesis (in separate document)

---

## Expected Outcomes by Paper

**Paper 1 → Modular weight definition**:
- Precise transformation law ψ_i → (cτ+d)^{w_i} ψ_i
- How orbifold fixed points affect weights
- Group theory: Δ(6n²) ↔ w values

**Paper 2 → Three-generation mechanism**:
- Why we get exactly 3 families with different weights
- Wave function formulas Ψ₁, Ψ₂, Ψ₃
- Explicit w₁, w₂, w₃ for a simple case (hopefully!)

**Paper 3 → String consistency constraints**:
- What fluxes are allowed (tadpole cancellation)
- How flux determines level k of Γ_N(k)
- Whether our k=27, k=16 are consistent with string theory

---

## Success Criteria for Literature Review

**MINIMUM (Day 3 end)**:
✓ Understand modular weight definition (transformation law)
✓ Identify mechanism: orbifold + flux → w_i values
✓ Extract at least one worked example from papers

**TARGET (Day 3 end)**:
✓ Explicit formula for w_i in terms of orbifold quantum numbers
✓ Wave function formulas for 3-generation case
✓ Can reproduce example calculation from paper

**STRETCH (Day 3 end)**:
✓ Apply formulas to our T^6/(Z_3×Z_4) case
✓ Preliminary calculation: w_e=?, w_μ=?, w_τ=?
✓ Check if matches phenomenology (-2, 0, 1)

---

## Next Steps After Literature Review

**If mechanism is CLEAR** → Proceed to synthesis (Days 4-5)
**If mechanism is MURKY** → Search for pedagogical reviews, textbook chapters
**If mechanism is IMPOSSIBLE** → Document findings, make NO-GO decision

---

## Notes on Paper Access

**arXiv PDFs**: Cannot fetch directly via web tools (PDF format limitation)

**Alternative approach**:
1. Focus on arXiv abstract pages (HTML)
2. Search for related papers citing these works
3. Look for review articles or thesis (e.g., arXiv:2403.17280 - PhD thesis, 238 pages!)
4. Extract key formulas from HTML equation rendering

**PhD Thesis Option**: arXiv:2403.17280 (Kikuchi, 238 pages, Feb 2024)
- "The flavor structures on magnetized orbifold models and 4D modular symmetric models"
- Likely most comprehensive, pedagogical treatment
- Should contain worked examples and explicit calculations

---

## Status

**Current**: Day 1, literature search complete (39 Kobayashi papers found)
**Priority papers identified**: Top 9 ranked by relevance
**Next action**: Read Paper 1 (arXiv:2410.05788) - modular symmetry of localized modes
**Timeline**: On track for Week 1 reconnaissance

**Branch**: `exploration/cft-modular-weights`
**Commit**: 486a18c "Wall #1 attack: CFT modular weights derivation plan"
