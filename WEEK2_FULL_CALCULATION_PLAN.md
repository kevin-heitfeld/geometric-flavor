# WEEK 2: FULL LEPTON SECTOR CALCULATION

**Date**: December 28, 2025
**Branch**: `exploration/cft-modular-weights`
**Timeline**: Days 8-14 (January 2026)
**Goal**: Calculate complete 3×3 charged lepton Yukawa matrix from first principles

---

## Executive Summary

**Week 1 Achievement**: Discovered formula **w = -2q₃ + q₄** that reproduces modular weights from orbifold quantum numbers.

**Week 2 Goal**: Go beyond the formula to **explicit calculation** of all Yukawa couplings Y_ij (i,j = e,μ,τ).

**Why This Matters**:
- Week 1 proved weights are *derivable* → Week 2 proves they give *correct phenomenology*
- Diagonal electron Yukawa already matches to 0.4% → Now test all 9 matrix elements
- Full Yukawa matrix = complete test of framework at fundamental level
- Success here → Week 3 quark sector becomes straightforward extension

**Success Criteria**:
- **Minimum**: Calculate all 9 Y_ij elements with explicit theta function integrals
- **Target**: Off-diagonal elements match Papers 1-3 phenomenology to ~10%
- **Stretch**: Extract CP violation phases, predict lepton mixing effects

---

## The Challenge

### What We Know (Week 1)
✅ Modular weights: w_e=-2, w_μ=0, w_τ=1
✅ Formula: w = -2q₃ + q₄
✅ Quantum numbers: electron (1,0), muon (0,0), tau (0,1)
✅ Parameters: k₃=-6, k₄=4
✅ Diagonal electron Yukawa matches to 0.4%

### What We Need (Week 2)
❓ Explicit wave functions ψ_e(z,τ), ψ_μ(z,τ), ψ_τ(z,τ)
❓ Theta characteristics (α,β) from quantum numbers
❓ Magnetic flux M values for each generation
❓ Yukawa overlap integrals Y_ij = ∫ψ_iψ_jψ_H d²z
❓ Off-diagonal elements and CP phases
❓ Full comparison with phenomenology

---

## Technical Background

### CFT Wave Functions (Cremades-Ibanez-Marchesano)

**General structure** (arXiv:hep-th/0404229, Section 3):
```
ψ(z,τ) = N(τ) × exp(πiMz̄z/Imτ) × θ[α;β](Mz|τ)
```

**Components**:
1. **N(τ)**: Normalization factor (modular weight contribution)
   - N(τ) ~ (Imτ)^(-1/4) for single T²
   - Product over three tori for T⁶

2. **exp(πiMz̄z/Imτ)**: Magnetic flux contribution
   - M = magnetic flux quantum (integer for D7-branes)
   - |M| = number of zero modes = 3 generations
   - Sign determines chirality

3. **θ[α;β](Mz|τ)**: Riemann theta function
   - α,β = characteristics (half-integers mod 1)
   - Encodes orbifold boundary conditions
   - Transforms under modular group

**Modular transformation**:
```
ψ(z/(cτ+d), (aτ+b)/(cτ+d)) = (cτ+d)^w × ρ(γ) × ψ(z,τ)

where γ = [a b; c d] ∈ SL(2,ℤ), ρ(γ) = phase factor
```

### Yukawa Couplings

**Definition**:
```
Y_ij = C × ∫_T² ψ_i(z,τ) × ψ_j(z,τ) × ψ_H(z,τ) × d²z
```

where:
- C = overall normalization (string coupling)
- T² = compact torus (one factor of T⁶)
- ψ_H = Higgs wave function (w_H=2 for Γ₀(3))

**Theta function integrals** (Cremades Section 4):
```
∫_T² θ[α₁;β₁](M₁z|τ) × θ[α₂;β₂](M₂z|τ) × θ[α₃;β₃](M₃z|τ) d²z
  = (analytic formula using theta function identities)
```

Key: These integrals are **exactly calculable** using:
- Riemann theta function addition formulas
- Jacobi triple product identity
- Residue theorem for poles in τ

---

## Day-by-Day Roadmap

### **Days 8-9: Literature Extraction Phase**

**Day 8 (January 2026): Deep Dive Cremades Paper**

**Primary source**: arXiv:hep-th/0404229 (Cremades-Ibanez-Marchesano, 73 pages)

**Sections to extract**:
1. **Section 3.1-3.2**: D7-brane wave functions on magnetized tori
   - Equation (3.1): Complete ψ(z,τ) formula
   - Table 1: Theta characteristics for different boundary conditions
   - Equation (3.15): Modular transformation properties

2. **Section 3.3**: Factorization for T⁶ = (T²)³
   - How to combine three tori
   - Modular weight additivity
   - Flux quantization on each cycle

3. **Section 4.1-4.2**: Yukawa coupling calculation
   - Equation (4.3): Triple overlap integral
   - Theorem 4.1: Theta function product formula
   - Example 4.2: Explicit 3-generation case

**Deliverable**: `CREMADES_FORMULAS_EXTRACTED.md` (~400 lines)
- Complete wave function formula with all terms explained
- Yukawa overlap integral formula ready to implement
- Theta function identities needed for calculation
- References to specific equations in paper

**Day 9: Secondary Literature**

**Supporting papers**:
1. **arXiv:0904.0910** (Antoniadis-Kumar-Panda, 77 pages)
   - Alternative theta function conventions
   - Cross-check formulas
   - Additional calculation techniques

2. **arXiv:2101.00826** (Kobayashi et al., 3-generation modes)
   - Verification of M=-6, M=4 values
   - 3-generation counting
   - Orbifold modifications

**Deliverable**: `THETA_FUNCTION_TOOLKIT.md`
- Unified notation across papers
- Complete list of needed identities
- Calculation workflow checklist

---

### **Days 10-11: Quantum Number Mapping Phase**

**Day 10: Answer Q2 - Theta Characteristics**

**Goal**: Determine precise map (q₃,q₄) → (α,β)

**Approach**:
1. **Z₃ sector**: q₃ ∈ {0,1,2} → β₃ = q₃/3
   - Orbifold twist: θ₃ = (1/3, 1/3, -2/3)
   - Wave function picks up phase: exp(2πiq₃/3)
   - Theta characteristic: θ[α₃; q₃/3]

2. **Z₄ sector**: q₄ ∈ {0,1,2,3} → β₄ = q₄/4
   - Orbifold twist: θ₄ = (1/4, 1/4, -1/2)
   - Similar phase analysis
   - Theta characteristic: θ[α₄; q₄/4]

3. **Spin structure α**: Periodic vs anti-periodic
   - Check D7-brane boundary conditions
   - NS vs R sector on worldsheet
   - Likely α=0 for all (even spin structure)

**Verification**:
- Construct explicit ψ with these (α,β)
- Check modular transformation gives w = -2q₃ + q₄
- Verify periodicity under z → z+1 and z → z+τ

**Code**: `map_quantum_numbers_to_characteristics.py`

**Deliverable**: `QUANTUM_NUMBER_MAPPING.md` (~200 lines)
- Complete derivation of (q₃,q₄) → (α,β) map
- Verification calculations
- Table of characteristics for all generations

**Day 11: Answer Q1 - Magnetic Flux**

**Goal**: Relate (k₃,k₄) = (-6,4) to magnetic flux quantization

**Key insights**:
1. **3-generation requirement**: |M| = 3 for each torus
   - But M can be ±3 (chirality choice)
   - Or factors: M₃ = -6, M₄ = 4 from product structure?

2. **Flux quantization formula**:
   ```
   ∫_cycle F = 2π × M
   ```
   - F = magnetic field strength on D7-brane
   - M = integer quantum number

3. **Orbifold corrections**:
   - Z₃: Maybe requires M₃ = 3k for k=-2 → M₃=-6?
   - Z₄: Maybe requires M₄ = 4k for k=1 → M₄=4?

**Verification**:
- Check if M₃=-6, M₄=4 gives 3 generations
- Verify these values appear in literature
- Connect to k₃=-6, k₄=4 in w = -2q₃ + q₄

**Code**: `verify_flux_quantization.py`

**Deliverable**: `MAGNETIC_FLUX_DERIVATION.md` (~250 lines)
- Flux quantization conditions
- Connection to (k₃,k₄) parameters
- Physical interpretation of signs and magnitudes

---

### **Days 12-13: Calculation Phase**

**Day 12: Wave Function Construction**

**Goal**: Build explicit ψ_e, ψ_μ, ψ_τ from components

**For each generation i** (electron, muon, tau):

1. **Quantum numbers**: (q₃^i, q₄^i) from Week 1
   - Electron: (1,0)
   - Muon: (0,0)
   - Tau: (0,1)

2. **Theta characteristics**: (α^i, β^i) from Day 10 mapping
   - β^i = (q₃^i/3, q₄^i/4)
   - α^i = (0,0) likely

3. **Magnetic flux**: (M₃, M₄) from Day 11
   - M₃ = -6
   - M₄ = 4

4. **Wave function**:
   ```
   ψ_i(z₃,z₄; τ₃,τ₄) = N₃(τ₃) × N₄(τ₄)
                      × exp(πiM₃z̄₃z₃/Imτ₃) × θ[α₃^i;β₃^i](M₃z₃|τ₃)
                      × exp(πiM₄z̄₄z₄/Imτ₄) × θ[α₄^i;β₄^i](M₄z₄|τ₄)
   ```

**Verification checks**:
- Modular transformation: ψ(S(z,τ)) = (-iτ)^(w_i) × ψ(z,τ)
- Periodicity: ψ(z+1,τ) = phase × ψ(z,τ)
- Normalization: ∫|ψ|² d²z = 1

**Code**: `construct_wave_functions.py` (~400 lines)
- Functions for each component
- Full wave function builder
- Verification tests
- Visualization of |ψ|² on fundamental domain

**Deliverable**: Wave functions ready for Yukawa calculation

**Day 13: Yukawa Matrix Calculation**

**Goal**: Calculate all 9 elements Y_ij

**Diagonal elements** (Y_ee, Y_μμ, Y_ττ):
```
Y_ii = C × ∫∫ ψ_i(z₃,τ₃) × ψ_i(z₃,τ₃) × ψ_H(z₃,τ₃) d²z₃
         × ∫∫ ψ_i(z₄,τ₄) × ψ_i(z₄,τ₄) × ψ_H(z₄,τ₄) d²z₄
```

**Off-diagonal elements** (Y_ij for i≠j):
```
Y_ij = C × ∫∫ ψ_i(z₃,τ₃) × ψ_j(z₃,τ₃) × ψ_H(z₃,τ₃) d²z₃
         × ∫∫ ψ_i(z₄,τ₄) × ψ_j(z₄,τ₄) × ψ_H(z₄,τ₄) d²z₄
```

**Calculation method**:
1. Use theta function product formula from Cremades
2. Apply Riemann theta function identities
3. Evaluate using τ₃=2.69i, τ₄=? from phenomenology
4. Extract overall scale C from Y_ee match

**Challenges**:
- Theta function products with different characteristics
- Integration over fundamental domain F = {z: |z|≤1, |Re(z)|≤1/2}
- Numerical stability for large Imτ

**Code**: `compute_yukawa_matrix_full.py` (~500 lines)
- Theta function integral calculator
- Matrix element computation
- Optimization and numerical checks
- Export results for comparison

**Deliverable**: Complete 3×3 Yukawa matrix with uncertainties

---

### **Day 14: Validation & Assessment**

**Day 14 Morning: Phenomenology Comparison**

**Comparison with Papers 1-3 results**:

From Papers 1-3 phenomenological fit (τ=2.69i, Γ₀(3)):
```
Y_e^fit  ≈ 2.8×10⁻⁶    (GUT scale)
Y_μ^fit  ≈ 6.1×10⁻⁴
Y_τ^fit  ≈ 1.0×10⁻²

Off-diagonal ≈ (modular form overlaps, ~10⁻⁴ - 10⁻³ range)
```

**Tests**:
1. **Diagonal match**: |Y_ii^calc - Y_ii^fit| / Y_ii^fit < 10%
2. **Hierarchy**: Y_e << Y_μ << Y_τ correct?
3. **Off-diagonal structure**: Correct order of magnitude?
4. **CP phases**: Match predictions from modular symmetry?
5. **Mixing**: Any charged lepton mixing predicted?

**Success criteria**:
- **MINIMUM**: Diagonal elements within factor of 2
- **TARGET**: All elements match to ~10%
- **STRETCH**: Off-diagonal phases match, predict new effects

**Code**: `validate_yukawa_predictions.py`

**Deliverable**: `WEEK2_YUKAWA_RESULTS.md`
- Complete comparison table
- Discussion of agreements and discrepancies
- Physics interpretation
- Publication-quality figures

**Day 14 Afternoon: Assessment & Decision**

**Questions to answer**:
1. ✅ or ❌ Did we calculate all 9 Yukawa elements?
2. ✅ or ❌ Do results match Papers 1-3 phenomenology?
3. ✅ or ❌ Is physics interpretation clear and compelling?
4. ✅ or ❌ Is quark sector extension (Week 3) straightforward?

**GO/NO-GO for Week 3**:

**GO conditions**:
- Full Yukawa matrix calculated successfully
- Agreement with phenomenology at ~10-20% level
- Clear path to quark sector extension
- No fundamental conceptual blockers

**If GO**: Week 3 = Quark sector (same method, different cycles)

**If NO-GO**:
- Address Week 2 gaps first
- May need additional papers/methods
- Reassess timeline for Paper 8

**Deliverable**: `WEEK2_ASSESSMENT.md` (~300 lines)
- Summary of Week 2 results
- Evaluation against success criteria
- GO/NO-GO decision with justification
- Week 3 roadmap (if GO)

---

## Technical Challenges & Mitigations

### Challenge 1: Theta Function Integration

**Problem**: ∫θ₁θ₂θ₃ d²z integrals are complex

**Literature solution**:
- Cremades Section 4.2 provides explicit formulas
- Use Riemann identity: Product of thetas = sum of thetas
- Residue theorem for contour integrals

**Mitigation**:
- Extract exact formulas from paper (Day 8)
- Implement symbolic calculation first
- Verify with numerical integration as cross-check
- Use Mathematica/SymPy for symbolic manipulation

### Challenge 2: Theta Characteristic Mapping

**Problem**: (q₃,q₄) → (α,β) map not explicitly in literature

**Physical reasoning**:
- Orbifold twist determines β via phase matching
- Spin structure determines α via fermion boundary conditions
- Can derive from first principles using CFT methods

**Mitigation**:
- Multiple approaches: geometric, CFT, D-brane boundary conditions
- Cross-check against known cases (Z₂, Z₃)
- Verify by checking modular transformation properties

### Challenge 3: Off-Diagonal Elements

**Problem**: More complex than diagonal (different characteristics)

**Why solvable**:
- Same theta function machinery
- Different (α,β) pairs but same integration method
- Literature has examples (Cremades Section 4.3)

**Mitigation**:
- Start with diagonal (already ~working from Week 1)
- Generalize to off-diagonal systematically
- Use symmetries to reduce independent calculations

### Challenge 4: Parameter Matching

**Problem**: τ₃, τ₄ values not precisely known

**Current status**:
- τ₃ ≈ 2.69i from Papers 1-3 fit
- τ₄ = ? (may be different or related)

**Mitigation**:
- Use τ₃=2.69i as starting point
- Scan τ₄ to match quark/neutrino sectors
- Or use geometric constraints from Calabi-Yau
- Worst case: treat as 1-2 free parameters (still better than 10!)

---

## Expected Outcomes

### Best Case Scenario

**Result**: All 9 Y_ij elements match phenomenology to ~5-10%

**Implications**:
- **Framework validated at fundamental level**
- ~10 parameters → 0-2 (τ values, possibly fixed by geometry)
- Wall #1 completely demolished
- Week 3 quark sector straightforward
- Paper 8 becomes high-impact breakthrough

**Timeline**: Submit Paper 8 by March 2026

### Target Scenario

**Result**: Diagonal elements match well, off-diagonal ~20-30% agreement

**Implications**:
- Core mechanism validated
- Small discrepancies explainable (higher-order corrections, RG running)
- Framework transformation still achieved
- Week 3 feasible with lessons learned

**Timeline**: Submit Paper 8 by April 2026

### Minimum Acceptable

**Result**: Correct order of magnitude for all elements, qualitative hierarchy

**Implications**:
- Proof of concept successful
- Quantitative refinements needed (2-loop RG, threshold corrections)
- Still major achievement: first derivation of Yukawas from geometry
- Week 3 may need additional tools

**Timeline**: More work needed, Paper 8 by May 2026

### Failure Scenario

**Result**: Large discrepancies (factor >5) or calculation intractable

**Response**:
- Debug Week 2 approach
- May need different CFT techniques
- Consult additional literature
- Consider collaborator with string CFT expertise
- Reassess if Papers 5-7 should come first

---

## Deliverables

### Documentation (6 files)
1. `WEEK2_FULL_CALCULATION_PLAN.md` - This document
2. `CREMADES_FORMULAS_EXTRACTED.md` - Complete formula extraction
3. `THETA_FUNCTION_TOOLKIT.md` - Calculation methods
4. `QUANTUM_NUMBER_MAPPING.md` - (q₃,q₄)→(α,β) derivation
5. `MAGNETIC_FLUX_DERIVATION.md` - k₃,k₄ from flux quantization
6. `WEEK2_YUKAWA_RESULTS.md` - Full results and comparison
7. `WEEK2_ASSESSMENT.md` - Final evaluation and GO/NO-GO

### Code (4 scripts)
1. `map_quantum_numbers_to_characteristics.py` - (q,α,β) mapping
2. `verify_flux_quantization.py` - M values verification
3. `construct_wave_functions.py` - Build ψ_e, ψ_μ, ψ_τ
4. `compute_yukawa_matrix_full.py` - Calculate all Y_ij
5. `validate_yukawa_predictions.py` - Compare with experiment

### Visualizations (3 figures)
1. `wave_functions_fundamental_domain.png` - |ψ|² for all generations
2. `yukawa_matrix_heatmap.png` - All 9 elements visualized
3. `phenomenology_comparison.png` - Theory vs experiment

### Git Commits
- Target: 8-12 commits over Week 2
- Clean history with descriptive messages
- Branch: `exploration/cft-modular-weights` (continue from Week 1)
- Merge to main after Week 2 assessment

---

## Success Metrics

### Quantitative
- [ ] All 9 Y_ij elements calculated explicitly
- [ ] Diagonal elements match to <20%
- [ ] Off-diagonal elements correct order of magnitude
- [ ] CP phases extracted from calculation
- [ ] Computational time <1 hour per matrix element

### Qualitative
- [ ] Physics interpretation clear and compelling
- [ ] Method generalizable to quark sector
- [ ] Paper 8 story coherent and complete
- [ ] Community reception: "This is major progress"

### Impact
- [ ] ~10 free parameters eliminated
- [ ] First-principles Yukawa calculation from string theory
- [ ] Framework elevated to fundamental theory status
- [ ] Clear path to complete flavor theory

---

## Risk Management

### Risk 1: Literature Extraction Incomplete
**Probability**: Low (papers are comprehensive)
**Impact**: High (can't proceed without formulas)
**Mitigation**: Allocate full 2 days to extraction, consult multiple papers

### Risk 2: Theta Integrals Too Complex
**Probability**: Medium (technically challenging)
**Impact**: Medium (can use numerical methods)
**Mitigation**: Start symbolic, fall back to numerical, use Mathematica

### Risk 3: Poor Phenomenology Match
**Probability**: Medium (first-principles calculation)
**Impact**: Low (still proof of concept)
**Mitigation**: Understand discrepancies, identify corrections needed

### Risk 4: Time Overrun
**Probability**: Medium (ambitious timeline)
**Impact**: Low (scientific value remains)
**Mitigation**: Flexible weekly schedule, prioritize critical tasks

---

## Relation to Papers 1-7

### Papers 1-3 (Published)
- Established modular symmetry phenomenology
- Fitted modular weights w_e,w_μ,w_τ as free parameters
- **Week 2 derives** what Papers 1-3 fitted

### Paper 4 (Submitted)
- Identified T⁶/(Z₃×Z₄) as geometric origin
- Connected to Γ₀(3) and Γ₀(4) flavor symmetries
- **Week 2 completes** what Paper 4 initiated

### Papers 5-7 (Planned Phenomenology)
- Proton decay, LFV, EDM predictions
- Use phenomenological modular weights
- **Will benefit** from Week 2 fundamental derivation

### Paper 8 (Target)
- **Week 2 is core calculation** for Paper 8
- Full lepton + quark + neutrino sectors
- First-principles flavor physics from string theory

---

## Timeline Summary

**Week 1** (Dec 28, 2025): ✅ COMPLETE
- Formula w=-2q₃+q₄ discovered
- Phenomenology validated
- STRONG GO decision

**Week 2** (Days 8-14, January 2026): **STARTING NOW**
- Day 8-9: Literature extraction
- Day 10-11: Quantum number mapping
- Day 12-13: Yukawa calculation
- Day 14: Validation & assessment

**Week 3** (Days 15-21, January 2026): If GO from Week 2
- Quark sector extension
- CKM matrix prediction
- b-quark outlier resolution

**Week 4** (Days 22-28, February 2026): If GO from Week 3
- Neutrino sector
- Type-I seesaw masses
- PMNS mixing predictions

**Paper 8 Writing** (February-March 2026)
- Complete manuscript
- Submit to PRD

---

## Conclusion

Week 2 is where we go from **"We discovered a formula"** to **"We calculated everything from first principles."**

This is the heart of Wall #1 attack: explicit, quantitative, testable predictions for all charged lepton Yukawa couplings starting from nothing but string theory geometry.

If Week 2 succeeds → Paper 8 becomes a landmark achievement in string phenomenology.

**Status**: Ready to begin!
**Next action**: Deep dive Cremades paper (Day 8)

---

**Date**: December 28, 2025
**Plan**: Week 2 (Days 8-14)
**Goal**: Complete 3×3 Yukawa matrix from first principles
**Timeline**: January 2026
