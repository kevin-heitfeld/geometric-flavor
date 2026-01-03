# Repository Cleanup Plan

## Problem Statement

The repository contains a mix of:
1. **Historical explorations** (failed attempts, superseded approaches)
2. **Final framework** (Papers 1-4, established results)
3. **Current research** (Path A/B, active investigations)

This mixing caused confusion where an AI agent pursued a "τ-ratio = 7/16" problem that was actually a **failed historical exploration** from Phase 1, not a current research question. The agent spent hours investigating a non-existent problem because documentation didn't clearly separate history from current work.

---

## Proposed Directory Structure

```
qtnc/
├── docs/
│   ├── framework/              # NEW: Canonical framework documentation
│   │   ├── README.md          # Start here - single source of truth
│   │   ├── PAPERS_1-4_SUMMARY.md
│   │   └── SINGLE_TAU_FRAMEWORK.md  # τ = 2.69i for ALL sectors
│   │
│   ├── research/               # NEW: Active research questions
│   │   ├── PATH_A_PROGRESS.md
│   │   ├── PATH_B_PROGRESS.md
│   │   └── OPEN_QUESTIONS.md
│   │
│   ├── historical/             # NEW: Old explorations (for context only)
│   │   ├── README.md          # "These are SUPERSEDED approaches"
│   │   ├── 2024_06_multi_tau_exploration.md  # FAILED
│   │   ├── 2024_07_delta_k_universality.md   # FAILED (not universal)
│   │   └── 2024_08_k_pattern_stress_test.md
│   │
│   └── [current mixed files]   # TO BE ORGANIZED
│
├── src/
│   ├── framework/              # NEW: Core framework calculations
│   │   ├── tau_2p69i/         # Scripts using correct τ = 2.69i
│   │   └── modular_forms/      # η, E₄ implementations
│   │
│   ├── papers/                 # NEW: Paper-specific analysis
│   │   ├── paper1_flavor/
│   │   ├── paper2_cosmology/
│   │   ├── paper3_dark_energy/
│   │   └── paper4_string_origin/
│   │
│   ├── research/               # NEW: Path A/B investigations
│   │   ├── path_a/
│   │   └── path_b/
│   │
│   ├── historical/             # NEW: Old scripts (for reference)
│   │   ├── README.md          # "Uses superseded values"
│   │   └── multi_tau_tests/   # Scripts with τ=3.25i, τ=1.422i
│   │
│   └── [159 Python scripts]    # TO BE ORGANIZED
│
├── results/                     # Organized by date/topic
├── figures/                     # Organized by paper/topic
├── manuscript*/                 # Paper directories (OK as is)
└── scripts/                     # Build utilities (OK)
```

---

## Phase 1: Identify and Label (Week 1)

### Step 1.1: Create Framework Documentation Hub

**Create docs/framework/README.md**:
```markdown
# Framework Documentation - START HERE

This directory contains the CANONICAL description of the established framework.

## The Single-τ Framework (Current)

**Core Principle**: All sectors (leptons, quarks, cosmology, dark energy) use the SAME modular parameter:

τ = 2.69i (purely imaginary)

**What differs between sectors**: The modular forms and groups, NOT τ itself.

- **Leptons**: Γ₀(3) at level k=27, using η(τ) modular forms
- **Quarks**: Γ₀(4) at level k=16, using E₄(τ) Eisenstein series
- **Both**: Same τ = 2.69i

## Papers (Canonical Sources)

1. **Paper 1 (Flavor)**: 19 SM flavor parameters from τ = 2.69i
2. **Paper 2 (Cosmology)**: Inflation, DM, leptogenesis from τ = 2.69i
3. **Paper 3 (Dark Energy)**: Quintessence from τ = 2.69i
4. **Paper 4 (String Origin)**: T⁶/(Z₃×Z₄) compactification realizes τ = 2.69i

χ²/dof = 1.18 for all 30 observables.

## What This Framework Does NOT Have

❌ Different τ values for different sectors (τ_leptons ≠ τ_quarks)
❌ τ-ratio = 7/16 connecting sectors
❌ Δk=2 as universal law (it's leptonic only)

These were explored historically but ABANDONED. See docs/historical/ for context.

## If You're New

1. Read SINGLE_TAU_FRAMEWORK.md (this directory)
2. Check PAPERS_1-4_SUMMARY.md for results
3. For open questions, see docs/research/OPEN_QUESTIONS.md
4. Do NOT read docs/historical/ unless you need context on what was tried and failed
```

### Step 1.2: Add Status Labels to Confusing Files

**Files needing immediate labels**:

1. **docs/FALSIFICATION_DISCOVERY.md**
   - Add header:
   ```markdown
   # ⊘ HISTORICAL EXPLORATION - SUPERSEDED ⊘
   
   **Status**: FAILED APPROACH - DO NOT USE
   **Date**: ~June 2024 (before Papers 1-4)
   **Outcome**: Abandoned in favor of single-τ framework
   
   This documents an attempt to use DIFFERENT τ values for leptons (3.25i) and quarks (1.422i).
   While the τ-ratio matched gauge couplings perfectly, the approach failed to achieve minimal
   unification and was abandoned.
   
   **Current Framework**: Uses τ = 2.69i for ALL sectors. See docs/framework/README.md
   
   ---
   
   [Original content below - kept for historical context]
   ```

2. **docs/COMPLETE_FLAVOR_UNIFICATION.md**
   - Add header section:
   ```markdown
   # Complete Flavor Unification
   
   ⚠️ **NOTE**: This document contains MIXED content from different phases.
   
   **Lines 1-200**: Historical multi-τ exploration (SUPERSEDED)
   **Lines 200+**: Discussion of single-τ framework (CURRENT)
   
   For clean documentation of current framework, see docs/framework/SINGLE_TAU_FRAMEWORK.md
   
   ---
   ```

3. **docs/DELTA_K_UNIVERSALITY_REPORT.md**
   - Add outcome at top:
   ```markdown
   # Δk Universality Test Results
   
   **Status**: COMPLETED ✅
   **Outcome**: Δk=2 is LEPTONIC feature, NOT universal
   **Implication**: Different sectors have different Δk values
   
   This falsifies the hypothesis that Δk=2 is a universal geometric law.
   
   ---
   ```

### Step 1.3: Label Historical Scripts

Add headers to scripts using old τ values:

**src/why_quarks_need_eisenstein.py** (line 1):
```python
#!/usr/bin/env python3
"""
HISTORICAL SCRIPT - Uses superseded τ values

This script demonstrates why quarks cannot use η(τ) and require E₄(τ) instead,
using the OLD multi-τ approach (τ_leptons=3.25i, τ_quarks=1.422i).

SUPERSEDED: Current framework uses τ=2.69i for BOTH sectors. The key difference
is modular forms (η vs E₄), not different τ values.

For current framework, see: src/framework/tau_2p69i/
Date: ~June 2024 (before Paper 1 finalization)
"""
```

Similar headers for:
- src/test_e4_beta_connection.py
- src/multi_brane_scenario_test.py
- src/tau_ratio_coupling_test.py

---

## Phase 2: Create New Structure (Week 2)

### Step 2.1: Create New Directories

```bash
mkdir docs/framework docs/research docs/historical
mkdir src/framework src/papers src/research src/historical
mkdir src/framework/tau_2p69i src/framework/modular_forms
mkdir src/papers/paper1_flavor src/papers/paper2_cosmology src/papers/paper3_dark_energy src/papers/paper4_string_origin
mkdir src/research/path_a src/research/path_b
mkdir src/historical/multi_tau_tests
```

### Step 2.2: Move Historical Documents

```bash
# Documents about failed approaches
mv docs/FALSIFICATION_DISCOVERY.md docs/historical/2024_06_multi_tau_exploration.md
mv docs/DELTA_K_UNIVERSALITY_REPORT.md docs/historical/2024_07_delta_k_universality.md
mv docs/K_PATTERN_STRESS_TEST_RESULTS.md docs/historical/2024_08_k_pattern_stress_test.md
```

### Step 2.3: Move Historical Scripts

```bash
# Scripts using old τ values
mv src/why_quarks_need_eisenstein.py src/historical/multi_tau_tests/
mv src/test_e4_beta_connection.py src/historical/multi_tau_tests/
mv src/multi_brane_scenario_test.py src/historical/multi_tau_tests/
mv src/tau_ratio_coupling_test.py src/historical/multi_tau_tests/
```

### Step 2.4: Organize Current Framework Scripts

**Identify scripts using τ = 2.69i** (grep found 30+):
```bash
# Core framework scripts
mv src/verify_tau_2p69i.py src/framework/tau_2p69i/
mv src/yukawa_numerical_overlaps.py src/framework/tau_2p69i/
mv src/yukawa_kahler_normalized.py src/framework/tau_2p69i/
mv src/understanding_errors.py src/framework/tau_2p69i/

# Paper-specific scripts
mv src/theory14_seesaw_cp.py src/papers/paper1_flavor/
mv src/theory14_rg_twoloop.py src/papers/paper1_flavor/
mv src/theory14_complete_fit*.py src/papers/paper1_flavor/

# Path A research
mv src/neutrino_k_pattern_stress_test.py src/research/path_a/
mv src/explain_k_pattern.py src/research/path_a/
```

---

## Phase 3: Create Canonical Documentation (Week 3)

### Step 3.1: Write docs/framework/SINGLE_TAU_FRAMEWORK.md

```markdown
# The Single-τ Framework: Definitive Documentation

## Core Principle

ALL sectors in the framework use the SAME modular parameter:

τ = 2.69i ± 0.05 (purely imaginary)

This is determined by global χ² fit to all 30 observables (flavor + cosmology + dark energy).

## What Differs Between Sectors

NOT the modular parameter τ, but the MATHEMATICAL STRUCTURES:

### Leptons
- **Modular group**: Γ₀(3) ⊂ SL(2,ℤ)
- **Level**: k = 27 (from worldvolume flux n_F = 3)
- **Modular forms**: η(τ) = q^(1/24) ∏(1-q^n), pure modular
- **Mass structure**: m_ℓ ∝ |η(τ)|^k_ℓ
- **Parameter**: τ = 2.69i

[continues with detailed documentation]
```

### Step 3.2: Write docs/research/OPEN_QUESTIONS.md

Carefully review Papers 1-4 to identify what is ACTUALLY unresolved:

```markdown
# Open Research Questions

This document lists questions that are:
- ✓ Verified as open (not answered in Papers 1-4)
- ✓ Relevant to current framework (τ = 2.69i)
- ✓ Not historical failures being revisited

## Path A: Mathematical Origins

### Question 1: E₄ from Gauge Anomaly
**Status**: Claimed derived in [check source]
**Verification needed**: Does Paper 4 actually derive this?

[Continue only with verified open questions]
```

---

## Phase 4: Add Prevention Mechanisms (Week 4)

### Step 4.1: Create docs/historical/README.md

```markdown
# Historical Explorations

⚠️ **WARNING**: Files in this directory document FAILED or SUPERSEDED approaches.

They are kept for:
- Understanding what was tried and why it didn't work
- Historical context for current framework decisions
- Preventing re-exploration of dead ends

**DO NOT** use these as basis for new research without checking current framework first.

## Timeline

**Phase 1: Multi-τ Exploration** (~June 2024)
- Hypothesis: Different sectors have different τ values
- Result: Failed minimal unification
- Status: ABANDONED

**Phase 2: Single-τ Framework** (June-August 2024)
- Hypothesis: Same τ, different modular forms
- Result: Success! χ²/dof = 1.18
- Status: ESTABLISHED (Papers 1-4)

**Phase 3: Path A/B Research** (August 2024 - Present)
- Building on established framework
- See docs/research/ for current questions

## Files in This Directory

[Table with status, date, outcome]
```

### Step 4.2: Create Pre-Research Checklist

**docs/research/CHECKLIST_BEFORE_INVESTIGATING.md**:

```markdown
# Checklist Before Investigating New Questions

Before spending time on a research question, verify:

## 1. Is this already answered?
- [ ] Checked Papers 1-4 (manuscripts/)
- [ ] Searched docs/framework/
- [ ] Searched results/ for completed tests

## 2. Is this consistent with established framework?
- [ ] Uses τ = 2.69i (not multiple τ values)
- [ ] Refers to correct modular groups (Γ₀(3), Γ₀(4))
- [ ] Doesn't contradict Paper 1-4 results

## 3. Is this a historical dead end?
- [ ] Checked docs/historical/ for similar attempts
- [ ] Verified not in FAILED approaches
- [ ] Confirmed not superseded by later work

## 4. Where did this question come from?
- [ ] Papers 1-4 explicitly list as open question
- [ ] Path A/B progress reports identify as next step
- [ ] NOT from mixed historical/current documents

## 5. Validation
- [ ] Can cite specific location in Papers 1-4 motivating this
- [ ] Question survives framework consistency checks
- [ ] Has clear success/failure criteria

If ANY item fails → STOP and verify before proceeding.
```

---

## Phase 5: Testing and Validation (Week 5)

### Step 5.1: Verify Links Still Work

After reorganization:
```bash
# Check for broken relative paths in documents
grep -r "](docs/" docs/
grep -r "](src/" docs/
# Update paths as needed
```

### Step 5.2: Test Script Imports

```python
# Many scripts import from other scripts
# Check if moves broke anything
python src/framework/tau_2p69i/verify_tau_2p69i.py
python src/papers/paper1_flavor/theory14_seesaw_cp.py
```

### Step 5.3: Update Main README.md

Add navigation section:
```markdown
## Repository Navigation

- **📁 docs/framework/**: Start here - canonical framework documentation
- **📁 docs/research/**: Active research questions (Path A/B)
- **📁 docs/historical/**: Old explorations (for context only)
- **📁 manuscripts/**: Papers 1-4 (published/ready for submission)
- **📁 src/framework/**: Core framework calculations
- **📁 src/papers/**: Paper-specific analyses
- **📁 src/research/**: Current investigation scripts

⚠️ If you're confused about τ values, read docs/CONFUSION_SOURCE_ANALYSIS.md
```

---

## Success Criteria

After cleanup, a new researcher (human or AI) should be able to:

1. **Quickly find canonical framework**:
   - Start at docs/framework/README.md
   - See "τ = 2.69i for ALL sectors" immediately
   - Find Papers 1-4 summaries

2. **Identify open questions**:
   - Go to docs/research/OPEN_QUESTIONS.md
   - See verified, up-to-date questions
   - Understand why each is worth investigating

3. **Avoid historical dead ends**:
   - See "HISTORICAL" or "SUPERSEDED" labels clearly
   - Understand why approaches failed
   - Not waste time re-exploring failures

4. **Verify against sources**:
   - Use checklist before investigating
   - Check Papers 1-4 first
   - Validate consistency with framework

---

## Rollout Plan

### Week 1: Labels (Non-disruptive)
- Add status headers to confusing files
- Create docs/framework/ with README
- Create docs/CONFUSION_SOURCE_ANALYSIS.md ✅ DONE

### Week 2: New Structure
- Create new directories
- Write README files for each
- Set up navigation

### Week 3: Content Creation
- Write SINGLE_TAU_FRAMEWORK.md
- Write PAPERS_1-4_SUMMARY.md
- Review and update OPEN_QUESTIONS.md

### Week 4: Migration
- Move files to new structure
- Update imports and paths
- Add historical context

### Week 5: Testing
- Verify all scripts still run
- Check documentation links
- Update main README.md

---

## Priority Recommendations

**Do FIRST** (immediate, non-disruptive):
1. ✅ Create docs/CONFUSION_SOURCE_ANALYSIS.md - DONE
2. Create docs/framework/README.md with "τ = 2.69i for all" statement
3. Add SUPERSEDED headers to FALSIFICATION_DISCOVERY.md and similar files

**Do NEXT** (Week 2-3):
4. Create directory structure
5. Write canonical framework documentation

**Do LATER** (Week 4-5):
6. Migrate files
7. Update imports
8. Full testing

**Can DEFER**:
- Detailed historical analysis
- Complete file reorganization
- Advanced navigation tools

---

## Estimated Effort

- **Phase 1** (Labels): 4-6 hours
- **Phase 2** (Structure): 6-8 hours
- **Phase 3** (Documentation): 8-12 hours
- **Phase 4** (Prevention): 4-6 hours
- **Phase 5** (Testing): 6-8 hours

**Total**: 28-40 hours over 4-5 weeks

**High-value quick wins** (do first):
- docs/framework/README.md: 2 hours, prevents most confusion
- SUPERSEDED labels: 2 hours, immediate clarity
- CONFUSION_SOURCE_ANALYSIS.md: Already done!

---

## Open Questions for User

1. **Scope**: Full reorganization or just docs/?
   - Full = move 159 Python scripts into structure
   - Docs only = just organize documentation, scripts stay flat

2. **Timing**: Do gradually or all at once?
   - Gradual = less disruptive, can continue research
   - All at once = cleaner result, but blocks other work for ~week

3. **Historical preservation**: Keep or delete?
   - Keep in docs/historical/ = educational, shows evolution
   - Delete entirely = cleaner, less clutter

4. **Git history**: Preserve file history on moves?
   - git mv = preserves history, more complex
   - Manual move = simpler, loses file history

Please advise on preferences before proceeding!
