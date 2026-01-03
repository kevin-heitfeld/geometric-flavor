# DAY 3 EXECUTION PLAN: Confirming Formula Novelty
**Date**: December 29, 2025
**Goal**: Increase confidence from 85% → 95%+ that τ = 27/10 formula is novel
**Time**: 4 hours

---

## CURRENT STATUS (After Day 2)

**Finding**: τ = k_lepton/X formula does NOT appear in standard references
**Confidence**: 85% novel
**Remaining uncertainty**: Possible obscure/recent papers

---

## DAY 3 TASKS

### Task 1: Recent Papers Search (90 min)
**Target**: Papers from 2020-2024 (most likely to have new results)

#### 1A: Kobayashi-Otsuka Recent Work (30 min)
Search INSPIRE-HEP for latest papers:
```
author:Kobayashi AND author:Otsuka AND year:2020-2024
keywords: modular, flavor, complex structure
```

**Check for**:
- Any formula relating τ to topological data
- New results on D7-brane moduli
- Updates to magnetized brane framework

#### 1B: Modular Flavor Theory Updates (30 min)
Key authors to check (2020-2024):
- Feruglio group (SISSA/Padova)
- Ding group (USTC)
- Novichkov et al. (CERN)

**Query**: "modular flavor symmetry" AND "complex structure" AND year>2020

#### 1C: String Phenomenology Reviews (30 min)
Recent comprehensive reviews might mention formula:
- Ibanez-Uranga updates?
- Weigand F-theory reviews (check newer arXiv:2xxx.xxxxx)
- Marchesano et al. D-brane reviews

---

### Task 2: Targeted ArXiv Queries (60 min)

#### Query A: Direct Mathematical Formula
```
abs:"complex structure modulus" AND abs:"orbifold" AND abs:"Hodge"
year: 2010-2024
```
**Expected**: 20-50 papers
**Action**: Scan abstracts for τ formulas

#### Query B: Type IIB Moduli Stabilization
```
abs:"Type IIB" AND abs:"moduli" AND abs:"formula"
year: 2015-2024
```
**Check**: Any explicit τ computation methods

#### Query C: Product Orbifolds Specifically
```
abs:"Z_3" AND abs:"Z_4" AND abs:"orbifold"
```
**Focus**: Papers on Z₃×Z₄ specifically

---

### Task 3: Google Scholar Deep Dive (60 min)

#### 3A: Cited-By Analysis (20 min)
Take Dixon et al. (1985) "Strings on Orbifolds"
- Check papers citing it (thousands)
- Filter for: "complex structure" + "formula"
- Look for generalizations

#### 3B: Forward Citations (20 min)
Take Kobayashi-Otsuka (2016) main paper
- Check who cites them
- Look for: attempts to compute τ from geometry

#### 3C: "Orphan" Formula Search (20 min)
Try unusual search terms:
- "tau equals" + "orbifold order"
- "complex structure" + "determined by"
- "modulus formula" + "topology"

---

### Task 4: Check Textbook Problem Sets (30 min)

Sometimes formulas hide in exercises!

**Sources**:
- Polchinski Vol II exercises on orbifolds
- Kiritsis "String Theory in a Nutshell" problems
- Becker-Becker-Schwarz exercises
- Blumenhagen-Lüst-Theisen problem sets

**Method**: PDF search for "complex structure" in exercise sections

---

## SUCCESS CRITERIA

### If Formula is Novel (Expected):
**Evidence needed**:
- ✅ No hits in 2020-2024 papers
- ✅ No formula in recent reviews
- ✅ No Z₃×Z₄ papers mention τ computation
- ✅ No textbook exercises derive it

**Confidence increase**: 85% → 95%

**Next action**: Draft paper section claiming novelty

### If Precedent Found (Unlikely):
**Action**:
- Document exact paper/equation
- Compare with your formula (notation differences?)
- Cite appropriately
- Focus on YOUR INSIGHT: Uniqueness of Z₃×Z₄

---

## OUTPUT DELIVERABLES

### 1. Updated Literature Findings
**File**: Update `DAY2_LITERATURE_FINDINGS.md` → `DAY2-3_LITERATURE_FINDINGS.md`

**Add sections**:
- "Recent Papers (2020-2024)" - results
- "ArXiv Systematic Search" - query results
- "Citation Analysis" - forward/backward citations
- "Final Confidence Assessment" - 95%+ or found precedent

### 2. Bibliography File
**File**: Create `tau_formula_bibliography.bib`

**Include**:
- All papers checked
- Papers with related (but not same) formulas
- Key negative results ("We checked X, formula not there")

### 3. Search Methodology Documentation
**File**: Update `WEEK1_PROGRESS_TRACKER.md`

**Document**:
- Specific queries run
- Number of papers scanned
- Time spent per query
- Justification for final confidence level

---

## PARALLEL TASK: Start Day 4 Prep

While waiting for searches to complete, start:

### Generalization Test Setup
**File**: Create `research/tau_formula_generalization_tests.py`

**Prepare tests for Day 4**:
1. List of 10+ orbifolds to test
2. Known CY manifolds with published h^{1,1} values
3. Limiting case predictions
4. Parameter sensitivity analysis

**Time**: 30 min (in parallel with searches)

---

## TIMELINE

```
Hour 1 (9:00-10:00): Recent papers search (Kobayashi-Otsuka, reviews)
Hour 2 (10:00-11:00): Targeted ArXiv queries A, B, C
Hour 3 (11:00-12:00): Google Scholar cited-by + forward citations
Hour 4 (12:00-13:00): Textbook exercises + document findings

Parallel: Day 4 prep (30 min during query waits)
```

---

## EXPECTED OUTCOME

**Most likely**: Confirm formula is novel (95% confidence)

**Supporting argument**:
1. Not in standard references (Day 2) ✅
2. Not in recent papers (Day 3) ✅ (expected)
3. Not in citation network (Day 3) ✅ (expected)
4. Different from literature approach (Day 2) ✅

**Paper claim**:
> "To our knowledge, this formula relating the complex structure modulus directly to orbifold group orders and Hodge numbers has not appeared in the literature. We have systematically searched standard references [citations], recent papers (2020-2024), and forward/backward citation networks without finding precedent."

**Backup position** (if something found):
> "While the formula τ = k/X appeared in [citation], we independently derived it and provide the first systematic study of its uniqueness properties and phenomenological implications."

---

## START HERE

**First action** (5 min):
1. Open INSPIRE-HEP: https://inspirehep.net
2. Search: `author:Kobayashi AND author:Otsuka AND year:2020-2024`
3. Scan abstracts for "complex structure" mentions
4. Document hits in findings file

**Then proceed** through Tasks 1-4 systematically.

**Stop condition**: If clear precedent found, halt search and document immediately.

---

## NOTES FOR EXECUTOR

- Be thorough but efficient
- Document negative results (important!)
- If searches take too long, sample representative papers
- Goal is confidence, not perfection
- Even 90% confidence is sufficient for "appears novel" claim

**Remember**: Your Day 2 search was already quite comprehensive. Day 3 is about catching edge cases and increasing confidence from "likely novel" to "very likely novel".
