# Framework Documentation - START HERE

**Last Updated**: December 28, 2025

This directory contains the CANONICAL description of the established framework.

---

## ⚡ Core Principle: Single Universal τ

**ALL sectors use the SAME modular parameter**:

```
τ = 2.69i ± 0.05 (purely imaginary)
```

Determined by global χ² fit to all 30 observables across flavor, cosmology, and dark energy.

### What This Means

- **Leptons**: Use τ = 2.69i
- **Quarks**: Use τ = 2.69i (SAME VALUE)
- **Cosmology**: Use τ = 2.69i (SAME VALUE)
- **Dark Energy**: Use τ = 2.69i (SAME VALUE)

### What Differs Between Sectors

NOT the modular parameter τ, but the **mathematical structures**:

| Sector | Modular Group | Level | Modular Forms | Parameter |
|--------|--------------|-------|---------------|-----------|
| **Leptons** | Γ₀(3) | k=27 | η(τ) - Dedekind eta | τ = 2.69i |
| **Quarks** | Γ₀(4) | k=16 | E₄(τ) - Eisenstein | τ = 2.69i |

**Key insight**: Different modular forms (η vs E₄), NOT different τ values.

---

## 🎯 Results: What This Framework Achieves

### Paper 1: Standard Model Flavor
- **19 flavor parameters** from single τ = 2.69i
- 6 quark masses, 4 CKM elements
- 3 charged lepton masses
- 3 neutrino mixing angles, 2 mass splittings, 1 CP phase
- **χ²/dof = 1.18** (excellent fit)

### Paper 2: Cosmological Consequences  
- Inflation from τ* = 2.69i moduli stabilization
- Dark matter from modular symmetry breaking
- Leptogenesis from CP violation
- **8 cosmological observables** explained

### Paper 3: Dark Energy
- Quintessence from pseudo-Nambu-Goldstone boson
- Mass, decay constant, instanton coefficient from τ = 2.69i
- Subdominant dark energy component
- **Testable deviations** from ΛCDM

### Paper 4: String Theory Origin
- **T⁶/(Z₃×Z₄) compactification** produces modular structure
- Complex structure modulus U = 2.69i
- Worldvolume fluxes determine levels (k=27, k=16)
- D7-branes on (1,3)-cycles realize Yukawa couplings

**Total**: 30 observables explained from single input τ = 2.69i

---

## ❌ What This Framework Does NOT Have

Common misconceptions to avoid:

- ❌ Different τ values for different sectors (τ_leptons ≠ τ_quarks)
- ❌ "τ-ratio = 7/16" connecting different modular parameters
- ❌ Δk=2 as universal law across all sectors (it's leptonic only)
- ❌ Multiple moduli (τ₁, τ₂, ...) for different branes

**These were explored historically but ABANDONED**. See `docs/historical/` for context on failed approaches.

---

## 📚 Canonical Sources (In Order of Priority)

### 1. Papers (Final Authority)
Located in `manuscript*/` directories:

- **manuscript/**: Paper 1 - Flavor physics
- **manuscript_cosmology/**: Paper 2 - Cosmological consequences  
- **manuscript_dark_energy/**: Paper 3 - Dark energy
- **manuscript_paper4_string_origin/**: Paper 4 - String theory origin

**If any documentation contradicts these papers, the papers are correct.**

### 2. Framework Documentation (This Directory)
- `SINGLE_TAU_FRAMEWORK.md` - Detailed technical documentation
- `PAPERS_1-4_SUMMARY.md` - Executive summaries
- This README - Quick orientation

### 3. Research Questions
- `docs/research/OPEN_QUESTIONS.md` - Verified open questions
- `docs/research/PATH_A_PROGRESS.md` - Mathematical origins research
- `docs/research/PATH_B_PROGRESS.md` - Extensions research

---

## 🚀 If You're New: Start Here

### Complete Beginner
1. **Read this page** (you're here!)
2. Understand: τ = 2.69i for ALL sectors
3. Different sectors use different modular forms with SAME τ

### Want Technical Details
1. Read `SINGLE_TAU_FRAMEWORK.md` (detailed math)
2. Check relevant Paper in `manuscript*/`
3. Look at verified scripts in `src/framework/tau_2p69i/`

### Want to Contribute Research
1. Check `docs/research/OPEN_QUESTIONS.md` for verified questions
2. Read `docs/research/CHECKLIST_BEFORE_INVESTIGATING.md`
3. **Do NOT** start from `docs/historical/` files

### Confused About τ Values?
Read `docs/CONFUSION_SOURCE_ANALYSIS.md` - explains historical vs current framework.

---

## ⚠️ Common Pitfalls

### Pitfall 1: Reading Historical Documents First
❌ **Wrong**: Start with `FALSIFICATION_DISCOVERY.md` → get confused about multiple τ values  
✅ **Right**: Start with this README → understand single τ = 2.69i framework

### Pitfall 2: Trusting Old Scripts
❌ **Wrong**: Use `src/why_quarks_need_eisenstein.py` (has τ=3.25i, τ=1.422i)  
✅ **Right**: Use `src/framework/tau_2p69i/verify_tau_2p69i.py` (correct τ=2.69i)

### Pitfall 3: Assuming Δk=2 is Universal
❌ **Wrong**: Apply Δk=2 to all sectors  
✅ **Right**: Δk=2 is leptonic only (see `docs/historical/2024_07_delta_k_universality.md`)

---

## 📞 Quick Reference Card

```
┌─────────────────────────────────────────────────┐
│  FRAMEWORK AT A GLANCE                          │
├─────────────────────────────────────────────────┤
│  τ = 2.69i (UNIVERSAL, ALL SECTORS)             │
│                                                 │
│  Leptons:  Γ₀(3), k=27, η(τ),  χ²=1.2         │
│  Quarks:   Γ₀(4), k=16, E₄(τ), χ²=1.1         │
│                                                 │
│  30 observables explained                       │
│  Papers 1-4: Ready for submission               │
└─────────────────────────────────────────────────┘
```

**Key Papers**:
- Heitfeld et al. (2024a) - Flavor  
- Heitfeld et al. (2024b) - Cosmology
- Heitfeld et al. (2024c) - Dark Energy
- Heitfeld et al. (2024d) - String Origin

**String Construction**: Type IIB on T⁶/(Z₃×Z₄) with magnetized D7-branes

**Status**: Framework established ✅ | Papers ready ✅ | Extensions in progress 🔄

---

## 🔗 Navigation

- **Up**: `docs/` (all documentation)
- **Sideways**: `docs/research/` (open questions), `docs/historical/` (old explorations)
- **Down**: Papers in `manuscript*/`
- **Code**: `src/framework/`, `src/papers/`

Last updated: 2025-12-28 | Maintained by: Kevin Heitfeld
