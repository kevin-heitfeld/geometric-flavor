# CFT Modular Weights Derivation Plan
**Wall #1 Attack: Derive w_i from First Principles**

Created: December 27, 2025
Branch: `exploration/cft-modular-weights`

---

## 1. Goal

Derive modular weights **w_e = -2, w_μ = 0, w_τ = 1** from string theory geometry, eliminating ~10 fitted parameters from the framework.

**Current Status**: Weights are phenomenological inputs that correctly predict fermion masses/mixings.

**Target**: Prove weights emerge from D7-brane boundary states and orbifold quantum numbers in worldsheet CFT.

**Success Criterion**: Compute w_i from T^6/(Z_3 × Z_4) geometry + D7-brane intersection numbers and match phenomenology.

---

## 2. Key References

### Primary Papers (Kobayashi-Otsuka)

1. **arXiv:2001.07972** "Classification of discrete modular symmetries in Type IIB flux vacua" (2020)
   - Proves congruence subgroups Γ_N appear on magnetized D-branes
   - Shows flux quantization determines modular group structure
   - Type IIB on T^6/Z_2 × Z_2' with three-form fluxes
   - **Key insight**: Modular groups from D-brane wrapping + flux constraints

2. **arXiv:2408.13984** "Non-invertible flavor symmetries in magnetized extra dimensions" (2024)
   - Recent work on modular flavor symmetries from magnetized D-branes
   - Effective action from Type IIB + D7-branes
   - May contain explicit vertex operator calculations

### Required Background

3. **Polchinski Vol 1 Ch 2-3**: Worldsheet CFT basics, vertex operators, boundary states
4. **Blumenhagen et al.**: D-branes and magnetized compactifications
5. **Ibanez-Uranga Ch 10**: Yukawa couplings from string theory

---

## 3. Technical Approach

### Phase 1: Literature Review (Days 1-3)

**Download and study:**
- arXiv:2001.07972 PDF (22 pages)
- arXiv:2408.13984 PDF (35 pages)
- Identify sections with explicit CFT calculations
- Extract formulas for boundary states |B_a⟩ on magnetized D7-branes

**Key formulas to find:**
```
|B_a⟩ = boundary state for D7-brane with flux F_a
V_ψ(z) = vertex operator for matter field ψ
⟨B_a|V_ψ₁(z₁)V_ψ₂(z₂)V_H(z₃)|B_b⟩ = Yukawa coupling amplitude
```

**Target extraction:**
- How do orbifold twists θ_3, θ_4 act on boundary states?
- Where do modular weights appear in the calculation?
- Connection to conformal dimensions h_ψ of twist fields?

### Phase 2: Map Modular Weight Emergence (Days 4-5)

**Hypothesis 1**: Weights from orbifold quantum numbers
```
Under Z_3: ψ → e^(2πiq_3/3) ψ
Under Z_4: ψ → e^(2πiq_4/4) ψ
Modular weight w = f(q_3, q_4, intersection numbers)
```

**Hypothesis 2**: Weights from conformal blocks
```
Yukawa ∝ ∫ d²z ⟨V_ψ₁V_ψ₂V_H⟩ ~ η(τ)^w₁ η(τ)^w₂ η(τ)^w_H
Extract w_i from conformal dimensions
```

**Hypothesis 3**: Weights from modular forms of D-brane action
```
Effective action: S = ∫ (∂_μτ)² + Yukawa[τ, ψ]
τ = complex structure modulus
Yukawa modular form of weight (w₁+w₂-w_H)
```

### Phase 3: Explicit Calculation Setup (Days 6-7)

**If feasible, start computing:**

1. **Boundary state for D7-brane on T^6/(Z_3 × Z_4)**
   ```
   |B⟩ = exp(-∑_n α₋ₙ·M·α̃₋ₙ) |0⟩
   M = matrix encoding brane position + flux
   ```

2. **Vertex operator for lepton doublet**
   ```
   V_ℓ(z) = :e^(ipX) σ_twist(z):
   σ_twist = twist field with h = f(q_3, q_4)
   ```

3. **Three-point function**
   ```
   ⟨V_ℓ₁(z₁)V_ℓ₂(z₂)V_H(z₃)⟩ on disk with boundary |B_a⟩, |B_b⟩
   ```

4. **Extract modular weight**
   ```
   Yukawa coupling Y_ij(τ) ∝ (η(τ)/η(Nτ))^k f_ij(τ)
   f_ij(τ) = modular form of weight w_i + w_j - w_H
   Read off w_i from conformal calculation
   ```

---

## 4. Feasibility Assessment

### Known Challenges

1. **CFT complexity**: Need full worldsheet calculation with orbifold twists
2. **Boundary state formalism**: Magnetized D-branes require advanced boundary CFT
3. **Modular form technology**: Must connect CFT to Γ_N(k) modular forms
4. **Orbifold technicalities**: Z_3 × Z_4 more complex than simple Z_N

### Required Expertise

- Conformal field theory (vertex operators, OPEs, correlation functions)
- Boundary CFT (Cardy states, boundary states, open string spectrum)
- Modular forms (η-function, Dedekind eta, modular weights)
- D-brane physics (DBI action, Chern-Simons terms, intersection numbers)

### AI Capabilities Assessment

**Strengths:**
- Can follow explicit calculations step-by-step
- Good at algebraic manipulations
- Can check dimensional analysis and consistency

**Weaknesses:**
- No intuition for "which terms dominate"
- Cannot innovate beyond what's in references
- May miss subtle CFT conventions

### Decision Criteria (End of Day 7)

**GO if:**
- ✓ Found explicit boundary state formulas in Kobayashi-Otsuka papers
- ✓ Clear connection between orbifold quantum numbers and modular weights
- ✓ Calculation looks systematic (even if tedious)
- ✓ Human+AI can reproduce example calculations from papers

**NO-GO if:**
- ✗ Papers only give schematic arguments, no explicit formulas
- ✗ Calculation requires unpublished or "known to experts" techniques
- ✗ Modularity weight emergence mechanism unclear after literature review
- ✗ Human+AI cannot understand/reproduce paper's example calculations

---

## 5. Expected Outcomes

### Best Case (Wall Broken!)

**Derived:** w_e = -2, w_μ = 0, w_τ = 1 from orbifold quantum numbers
**Result:** Framework has ZERO fitted parameters for lepton quantum numbers
**Impact:** Transforms from "excellent phenomenology" to "candidate fundamental theory"
**Next:** Similar calculation for quark sector (w_u, w_c, w_t, w_d, w_s, w_b)
**Timeline:** 3-4 weeks full calculation + validation

### Partial Success

**Understood:** Mechanism for weight emergence, but not full calculation
**Result:** Can constrain weights (e.g., w_τ - w_e = 3 from topology)
**Impact:** Reduces free parameters from ~10 to ~3
**Next:** Document mechanism, use constraints in phenomenology
**Timeline:** 1-2 weeks mechanism documentation

### Learning Outcome (Wall Stands)

**Understood:** Why calculation is hard, what's missing from current knowledge
**Result:** Document technical roadblocks (e.g., "need full D7 CFT partition function")
**Impact:** Honest assessment of framework limitations
**Next:** Pivot to phenomenology Papers 5-7, flag weights as "future work"
**Timeline:** 1 week documentation, then switch tracks

---

## 6. Week 1 Roadmap

### Day 1 (Dec 27): Literature Download
- [x] Created exploration/cft-modular-weights branch
- [x] Found Kobayashi-Otsuka papers (arXiv:2001.07972, arXiv:2408.13984)
- [ ] Download PDFs
- [ ] First read: identify sections with explicit formulas

### Day 2: Deep Dive arXiv:2001.07972
- [ ] Extract boundary state formalism for magnetized D7-branes
- [ ] Understand how flux quantization → congruence subgroups
- [ ] Map connection: D-brane wrapping → modular groups

### Day 3: Deep Dive arXiv:2408.13984
- [ ] Look for explicit vertex operator calculations
- [ ] Extract modular weight emergence mechanism
- [ ] Document any CFT technology we'll need

### Day 4-5: Synthesis
- [ ] Write out explicit formulas for our case (T^6/(Z_3 × Z_4))
- [ ] Identify modular weight mechanism (Hypothesis 1, 2, or 3?)
- [ ] Assess: can we compute w_e, w_μ, w_τ?

### Day 6-7: Feasibility Decision
- [ ] Attempt toy calculation (e.g., Z_3 sector only)
- [ ] Test if human+AI can reproduce paper examples
- [ ] **GO/NO-GO decision** for full Wall #1 attack

### Day 8 (Decision Point): Branch
- **If GO**: Commit to Weeks 2-4 full CFT calculation
- **If NO-GO**: Document findings, switch to Paper 5 (proton decay)

---

## 7. Success Metrics

**Wall #1 Broken:**
- Derived w_e = -2, w_μ = 0, w_τ = 1 from geometry (match phenomenology)
- Framework now has 0 free parameters for lepton flavor structure
- Paper 8: "First-Principles Derivation of Modular Weights from String Geometry"

**Partial Wall Broken:**
- Established constraints (e.g., w_τ - w_e = 3) reducing free parameters
- Documented mechanism even if not full calculation
- Incorporated constraints into Papers 5-7

**Wall Stands (Learning):**
- Documented why calculation is beyond current capabilities
- Identified missing techniques (e.g., "full orbifold CFT partition function")
- Honest assessment in Paper 4 Discussion section remains accurate

---

## 8. Risk Mitigation

**Risk 1**: Papers too abstract, no explicit formulas
→ Search for earlier Kobayashi-Otsuka work with pedagogical examples
→ Look for reviews on magnetized D-branes (Blumenhagen, Cvetic, et al.)

**Risk 2**: Calculation requires unpublished expertise
→ Document "calculation exists in principle" and cite papers
→ Pivot to phenomenology, note weights as "future work"

**Risk 3**: Weeks 2-4 fail after committing
→ Hybrid approach: always keep 40% effort on phenomenology Papers 5-7
→ Even if Wall #1 fails, expand falsifiable predictions

**Risk 4**: Derived weights don't match phenomenology
→ CRITICAL: Would invalidate entire framework
→ Must check intermediate steps carefully against known results

---

## Next Steps (Immediate)

1. Download arXiv:2001.07972 and arXiv:2408.13984 PDFs
2. First pass: identify all equations with "modular weight" or "conformal dimension"
3. Extract boundary state formulas for D7-branes with flux
4. Map orbifold quantum numbers q_3, q_4 for lepton families

**Status**: Day 1 in progress
**Branch**: `exploration/cft-modular-weights`
**Commitment Level**: 1 week reconnaissance, then GO/NO-GO decision
