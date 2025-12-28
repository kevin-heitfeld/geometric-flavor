# Active Research Questions

**Last Updated**: December 28, 2025

This directory contains VERIFIED open research questions that build on the established framework (Papers 1-4).

---

## ⚠️ Before Investigating Any Question

**Use the checklist**: [CHECKLIST_BEFORE_INVESTIGATING.md](CHECKLIST_BEFORE_INVESTIGATING.md)

**Key verification**:
1. ✅ Is it mentioned in Papers 1-4 as unresolved?
2. ✅ Uses τ = 2.69i (not multiple τ values)?
3. ✅ Consistent with established framework?
4. ✅ Not tested in `docs/historical/`?
5. ✅ Clear motivation from Papers 1-4?

If ANY fails → STOP and verify.

---

## Current Research Tracks

### Path A: Mathematical Origins

**Goal**: Understand WHY the framework structure emerges from first principles

**Status**: Partially complete (Steps 1-3 done, Step 4 needs verification)

**Questions**:
1. ✅ **E₄ from gauge anomalies** - COMPLETED
   - Derived E₄(τ) structure from SU(3) gauge anomaly cancelation
   - See: [PATH_A_PROGRESS_REPORT.md](PATH_A_PROGRESS_REPORT.md)

2. ✅ **3 generations from topology** - COMPLETED
   - Derived from tadpole constraint + Z₃ orbifold order
   - N_gen = 3 from C = 2k_avg + 1 pattern
   - See: Path A Step 2 results

3. ✅ **C=13 pattern** - UNDERSTOOD
   - C = 2k_avg + 1 with k_avg ≈ 6 gives C=13
   - Universal across lepton sector
   - See: Path A Step 3 results

4. ❓ **Step 4** - NEEDS CLARIFICATION
   - Original plan investigated "τ-ratio = 7/16" (WRONG - no ratio exists!)
   - Need to determine what ACTUAL question remains
   - Verify against Papers 1-4 what's truly open

**Next Actions**:
- Review Path A progress with correct understanding (single τ=2.69i)
- Verify Steps 1-3 claims against Papers 1-4
- Determine if Path A is complete or if real questions remain

---

### Path B: Extensions and Predictions

**Goal**: Extend framework to new domains, make testable predictions

**Possible Questions** (need verification against Papers 1-4):

1. **Gauge coupling unification**
   - Does modular structure predict GUT scale?
   - Threshold corrections at τ = 2.69i
   - Status: Partially addressed in Paper 4?

2. **Quantum corrections**
   - How stable is τ = 2.69i under RG running?
   - Kähler corrections at large volume
   - Status: Discussed in Paper 1 Section 6.2?

3. **Gravitational sector**
   - Does framework extend to gravity?
   - Planck scale emergence from moduli
   - Status: Unknown, needs investigation

4. **Neutrino mass scale**
   - What determines absolute mass (only splittings fit)?
   - Right-handed neutrino masses from geometry
   - Status: Seesaw mechanism in Paper 1, absolute scale open?

**Status**: Need systematic review of what Papers 1-4 actually leave open

---

## Questions That Are NOT Open (Already Answered)

### ❌ Multi-τ Framework
**Question**: "Can we use different τ values for different sectors?"
**Status**: TESTED AND FAILED (Phase 1 exploration)
**See**: `docs/historical/2025_12_22_multi_tau_exploration.md`

### ❌ Δk=2 Universality
**Question**: "Is Δk=2 universal across all sectors?"
**Status**: TESTED - NO (leptonic feature only)
**See**: `docs/historical/2025_12_24_delta_k_universality.md`

### ❌ Different k-orderings
**Question**: "Do different k-orderings give same τ?"
**Status**: TESTED - NO (different orderings → different τ)
**See**: `docs/historical/2025_12_25_k_pattern_stress_test.md`

### ❌ τ-ratio = 7/16
**Question**: "Why does τ_leptons/τ_quarks = 7/16?"
**Status**: INVALID - Framework uses single τ=2.69i
**See**: `docs/CONFUSION_SOURCE_ANALYSIS.md`

---

## How to Add New Questions

### Requirements

A question should be added here ONLY if:

1. **Verified open**: Papers 1-4 explicitly mention as future work
2. **Framework consistent**: Uses τ = 2.69i, established structures
3. **Not historical**: Checked `docs/historical/` - not a failed attempt
4. **Clear motivation**: Can cite specific Paper 1-4 location
5. **Testable**: Has clear success/failure criteria

### Format

```markdown
### [Question Title]

**Paper Reference**: [Paper X, Section Y mentions this as open]
**Motivation**: [Why this matters for framework]
**Current Understanding**: [What we know]
**Open Question**: [Specific thing to investigate]
**Success Criteria**: [What would answer this]
**Status**: [Not started / In progress / Blocked]
```

### Process

1. Read Papers 1-4 carefully
2. Identify explicit mentions of "future work" or "open questions"
3. Check it's not in `docs/historical/` (already tested)
4. Verify consistency with τ = 2.69i framework
5. Add to appropriate track (Path A or Path B)
6. Update this README

---

## Current Priority

**HIGH PRIORITY**: Systematic review of Papers 1-4 to extract ACTUAL open questions

**Method**:
1. Search Papers 1-4 for: "future work", "open question", "remains to be shown"
2. List all explicit mentions
3. Verify each is truly open (not answered elsewhere in papers)
4. Organize by track (Path A vs Path B)
5. Update this document with verified list

**Why**: Prevents wasting time on:
- Non-existent questions (like τ-ratio)
- Already-answered questions (scattered across papers)
- Historical dead ends (already tested and failed)

---

## Files in This Directory

| File | Purpose | Status |
|------|---------|--------|
| `README.md` | This file - navigation and overview | Current |
| `OPEN_QUESTIONS.md` | Verified list of open questions | Needs creation |
| `PATH_A_PROGRESS_REPORT.md` | Path A status and results | Needs review |
| `PATH_B_PROGRESS_REPORT.md` | Path B status and plans | Needs creation |
| `CHECKLIST_BEFORE_INVESTIGATING.md` | Prevention checklist | To be created |

---

## Navigation

- **Current Framework**: `docs/framework/` ← Established results
- **Up**: `docs/` (all documentation)
- **Historical**: `docs/historical/` ← Failed approaches
- **Papers**: `manuscript*/` ← Final authority

**Remember**: Always verify against Papers 1-4 before starting new investigation!

Last updated: 2025-12-28 | Maintained by: Kevin Heitfeld
