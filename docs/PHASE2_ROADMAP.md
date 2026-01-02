# Phase 2: Explicit Calabi-Yau Geometry and Kähler Metric

## Objective (Refined from Surgical Attack)

**Goal**: Compute g_i and A_i from explicit CY geometry with Kähler metric, eliminating the 2 calibration factors (δg, λ) from surgical attack.

**Target physics**: 
- g_i from holomorphic/modular data (already 2.5% in surgical attack)
- A_i from **Kähler metric and wavefunction normalization** (need ×2-5 enhancement over naive distance)

**Success criterion**: <10% error overall (validates framework)

## Key Insight from Surgical Attack

The split between g_i success (2.5%) and A_i failure (28%) is NOT a bug — it's a **signal**:

| Parameter | Physical Origin | Topology Sufficient? | Result |
|-----------|----------------|---------------------|---------|
| g_i | Holomorphic/modular weights | ✅ YES | 2.5% error |
| A_i | Kähler metric/wavefunction norm | ❌ NO | 28% error |

**ChatGPT verdict**: "TOPOLOGY MAXED OUT — METRIC PHYSICS REQUIRED"

The A_i systematic undershooting by ×2-5 is **not a failure** — it's the theory telling us exactly what's missing:
1. Kähler metric (not just topology)
2. Wavefunction normalization integrals
3. Moduli mixing (τ₁, τ₂, τ₃ cross-talk)
4. Possible warping corrections

## Phase 2 Strategy: Minimum Viable CY

### Geometry Choice: Resolved T²×T²×T²

**Why this geometry:**
- ✅ Explicit metric known (toric variety)
- ✅ Kähler moduli identified (τ₁, τ₂, τ₃ plus blow-up modes)
- ✅ D7-brane embedding computable
- ✅ Wavefunction profiles solvable (at least numerically)
- ✅ Warping perturbative (small throats, not deep)

**What we add vs. surgical attack:**
1. **Blow-up exceptional divisors** (resolved singularities)
2. **Full Kähler potential** K = -log(∫ J ∧ J ∧ J) with corrections
3. **Matter metric** G_ij = ∂ᵢ∂̄ⱼK evaluated explicitly
4. **Wavefunction overlap integrals** ∫ ψ_i* ψ_j ψ_H on CY

**What we don't need (yet):**
- ❌ Complex structure moduli (fixes Yukawa structure, not magnitudes)
- ❌ Full flux stabilization (use approximate LARGE volume regime)
- ❌ α' corrections (expect ~10% effects, smaller than current A_i errors)
- ❌ Deep warped throats (add if first pass fails)

### Technical Approach

**Week 1-2: Explicit CY Setup**

1. **Resolved geometry** (Days 1-3)
   - Start with T²×T²×T² = (ℂ/Λ)³
   - Blow up 27 fixed points (T²×T²×T² has 64 → blow 27 for CY)
   - Toric polytope: vertices from (n,m) lattice points
   - Kähler cone: identify which blow-ups are independent

2. **Kähler potential** (Days 4-5)
   - Volume form: V = τ₁τ₂τ₃ + corrections from blow-ups
   - Kähler potential: K = -3log(V)
   - Check: K_τᵢτ̄ⱼ = ∂τᵢ∂τ̄ⱼK (matter metric)
   - Include mixing: τ₁τ₂ terms from blow-up geometry

3. **D7-brane embedding** (Days 6-7)
   - Use bulk brane picture from surgical attack v3 (all wrap all tori)
   - Embedding: X⁷ ⊂ CY³ given by wrapping numbers (n,m) on each T²
   - DBI action: ∫_X⁷ √det(P[G + B]) where P is pullback
   - Check: tadpole cancellation, anomaly cancellation

**Week 3: Wavefunction Profiles**

4. **Fermion zero modes** (Days 8-10)
   - Massless mode equation: ∂̄ψ = 0 (Dolbeault cohomology)
   - Expansion: ψ(z) = Σ_α c_α ω_α where ω_α are harmonic forms
   - Localization: ψ ∝ exp(-d²/2σ²) near brane locus
   - Solve: either analytically (approximate) or numerically (exact)

5. **Overlap integrals** (Days 11-12)
   - Yukawa coupling: Y_ijk ~ ∫_CY ψ_i ∧ ψ_j ∧ ψ_k ∧ Ω
   - Matter metric normalization: G_ij = ∫_CY ψ_i* ∧ ψ_j
   - Combine: Physical Yukawa = Y_ijk / √(G_ii G_jj G_kk)
   - Extract: g_i and A_i from ratio Y_ijk / Y_111

**Week 4: Analysis and Refinement**

6. **Compare with fitted values** (Days 13-14)
   - Compute g_i from modular weights (should match surgical attack ~2.5%)
   - Compute A_i from wavefunction overlaps (target: match fitted values within 10%)
   - Check: Is ×2-5 enhancement present from Kähler metric normalization?

7. **Add corrections if needed** (Days 15-16)
   - If still >10% error on A_i:
     - Add warping: warp factor a(y) from flux backreaction
     - Add α' corrections: K → K + α' R where R is Ricci scalar
     - Add quantum corrections: loop effects on Kähler potential
   - Iterate until convergence or clear failure

8. **Document and decide** (Days 17-18)
   - Success (<10%): Write up methodology, prepare for paper
   - Partial (10-20%): Identify remaining missing physics
   - Failure (>20%): Honest assessment of what doesn't work

## Mathematical Framework

### Kähler Potential for Resolved T²×T²×T²

General form:
```
K = -log(8τ₁τ₂τ₃) + corrections
```

With blow-ups (simplified, 3 key exceptional divisors):
```
V = τ₁τ₂τ₃ + ε₁τ₁τ₂ + ε₂τ₁τ₃ + ε₃τ₂τ₃ + ...
K = -3log(V) + O(ε²)
```

Matter metric (bulk branes):
```
G_ij = ∂τᵢ∂τ̄ⱼK = (3/V²)[δᵢⱼV² - (∂ᵢV)(∂ⱼV)]
```

### Wavefunction Ansatz

For D7-brane with wrapping (n_i, m_i) on torus T_i²:
```
ψ(z₁,z₂,z₃) = A × θ[n₁,m₁](z₁;τ₁) × θ[n₂,m₂](z₂;τ₂) × θ[n₃,m₃](z₃;τ₃)
```

Where θ[n,m] is Jacobi theta function with characteristics.

Normalization:
```
G = ∫_CY |ψ|² √g d⁶y = |A|² × (matter metric factor)
```

### Yukawa from Overlap

Physical Yukawa:
```
Y_physical = Y_holomorphic / √(G₁G₂G₃)
            = [g_i × exp(A_i)] × reference
```

Where:
- g_i comes from modular weight ω_i → θ[n,m] normalization
- A_i comes from ∫ |ψ_i|² √g (Kähler metric contribution)

**Key point**: The √g factor in normalization integral is where ×2-5 enhancement enters!

## Computational Tools

### Required Capabilities

1. **Symbolic**: 
   - Kähler potential derivatives
   - Metric computation
   - Modular forms (Jacobi theta, Dedekind eta)

2. **Numerical**:
   - Integration over CY (if needed)
   - Theta function evaluation
   - Optimization (find best blow-up parameters ε_i)

3. **Already have**:
   - Dedekind eta (in kmass_compute.py)
   - Modular weight calculation (surgical attack scripts)
   - Fitted values for comparison (unified_predictions_complete.py)

### New Tools Needed

**Create**: `src/phase2_kahler_metric.py`
- KahlerPotential class (V, K, derivatives)
- BlownUpTorus class (exceptional divisors, volumes)
- MatterMetric class (G_ij from K)

**Create**: `src/phase2_wavefunctions.py`
- JacobiTheta class (theta functions with characteristics)
- D7Wavefunction class (ψ from wrapping numbers)
- OverlapIntegral class (compute ∫ ψ*ψψ)

**Create**: `src/phase2_yukawa_computation.py`
- Main script: tie everything together
- Compute g_i and A_i from first principles
- Compare with fitted values
- Output: validation plots and error analysis

## Success Criteria

### Minimal Success (validates framework)
- ✅ Overall error <10%
- ✅ g_i maintain ~2-5% error (already achieved)
- ✅ A_i improve from 28% → <15%
- ✅ Understand ×2-5 enhancement mechanism

### Target Success (competitive with fit)
- 🎯 Overall error <5%
- 🎯 g_i <3% error
- 🎯 A_i <7% error
- 🎯 Calibration factors (δg, λ) derived, not fitted

### Aspirational Success (parameter-free)
- 🌟 k_mass predicted from geometry
- 🌟 All 18 parameters (g_i, A_i) within 3%
- 🌟 Zero free parameters (fully geometric)

## Failure Modes and Response

### Scenario 1: A_i still >15% error after Kähler metric
**Diagnosis**: Warping or α' corrections needed
**Response**: Add warp factor a(y) = 1 + w(y) perturbatively
**Timeline**: +1 week

### Scenario 2: g_i degrades (currently 2.5% → >10%)
**Diagnosis**: Blow-up corrections interfering with modular structure
**Response**: Use minimal blow-up (fewer exceptional divisors)
**Timeline**: +3 days (iterate geometry choice)

### Scenario 3: Both >20% error
**Diagnosis**: Fundamental issue with toric CY or bulk brane picture
**Response**: Consider alternative:
- Option A: Different CY (e.g., quintic in ℙ⁴)
- Option B: Intersecting brane stacks (not bulk)
- Option C: Non-geometric (F-theory) approach
**Timeline**: Major pivot (2-3 weeks to restart)

### Scenario 4: Success (<10%) but k_mass still phenomenological
**Diagnosis**: Framework works but needs additional constraint
**Response**: Accept as "partial unification" - geometric origin for 18/21 params
**Outcome**: Strong result, publishable, motivates further work

## Timeline

**Total: 3-4 weeks** (18-20 working days)

| Week | Focus | Deliverable |
|------|-------|-------------|
| 1-2 | Explicit CY geometry | Kähler potential, D7 embedding |
| 3 | Wavefunction profiles | Overlap integrals computed |
| 4 | Analysis & refinement | Comparison with data, error <10%? |
| +1 (if needed) | Corrections | Warping, α', or quantum effects |

**Checkpoints**:
- Day 7: Kähler metric explicit, sanity checks pass
- Day 12: Wavefunction overlaps computed, first g_i/A_i estimates
- Day 16: Error analysis complete, decision point
- Day 20: Final results documented

## Commitment

This is the **decisive test** of the framework:
- ✅ Success (<10%): Framework validated, proceed to publication
- ⚠️ Partial (10-20%): Identify missing physics, decide if addressable
- ❌ Failure (>20%): Honest assessment, framework needs major revision

No more exploratory work after this. Phase 2 either works or it doesn't.

## Next Steps (Immediate)

1. **Create Phase 2 roadmap** (this document) ✅
2. **Set up Kähler potential framework** (geometry classes)
3. **Implement wavefunction module** (Jacobi theta, overlaps)
4. **Run first computation** (toric CY with minimal blow-ups)
5. **Iterate and refine** (add corrections as needed)

Ready to begin implementation.
