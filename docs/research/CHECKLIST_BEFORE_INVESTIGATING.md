# Checklist Before Investigating New Questions

Use this checklist BEFORE spending time on any research question.

**Purpose**: Prevent wasted effort on non-existent problems, historical dead ends, or already-answered questions.

**Time saved**: 1-10+ hours per investigation by catching issues early

---

## The τ-Ratio Incident (December 2025)

**What happened**: Agent spent hours investigating "τ-ratio = 7/16" connecting supposedly different τ values for leptons (3.25i) and quarks (1.422i).

**Reality**: Framework uses **single τ = 2.69i** for all sectors. The "τ-ratio" was a failed Phase 1 exploration, not current framework.

**Cost**: ~1000+ lines of code, 3 files created, committed to GitHub before error caught.

**Prevention**: This checklist. Use it every time.

---

## Required Checks (All Must Pass)

### ☐ 1. Papers 1-4 Verification

**Check**: Is this question mentioned in Papers 1-4 as explicitly unresolved?

**How to verify**:
```bash
# Search all manuscript files
grep -r "future work" manuscript*/
grep -r "open question" manuscript*/
grep -r "remains to" manuscript*/
grep -r "needs further" manuscript*/
```

**Pass criteria**: Find specific citation like "Paper 1, Section 6, mentions X as future work"

**Fail indicators**:
- ❌ Can't find question mentioned anywhere in Papers 1-4
- ❌ Found in historical docs but not papers
- ❌ Based on assumption not in papers

**If fails**: Question likely doesn't exist or is already answered. STOP.

---

### ☐ 2. Framework Consistency

**Check**: Does this use τ = 2.69i (not multiple τ values)?

**How to verify**:
- Read question description carefully
- Look for mentions of "τ_leptons", "τ_quarks", "τ_hadronic", "τ-ratio"
- Check if it assumes different sectors have different modular parameters

**Pass criteria**:
- ✅ Uses single τ = 2.69i universally
- ✅ May investigate different modular FORMS (η, E₄) but same τ
- ✅ Consistent with Papers 1-4 framework

**Fail indicators**:
- ❌ Mentions "τ_leptons = 3.25i" and "τ_quarks = 1.422i"
- ❌ Investigates "τ-ratio" or "why different τ values"
- ❌ Assumes Δk=2 is universal (it's leptonic only)

**If fails**: Question is based on superseded Phase 1 framework. STOP.

---

### ☐ 3. Historical Check

**Check**: Has this been tested before in `docs/historical/`?

**How to verify**:
```bash
# Search historical documents
ls docs/historical/
grep -r "key terms" docs/historical/

# Check specific known dead ends
cat docs/historical/2025_12_22_multi_tau_exploration.md
cat docs/historical/2025_12_24_delta_k_universality.md
```

**Pass criteria**: Question doesn't appear in historical failed attempts

**Fail indicators**:
- ❌ Found in historical/ with "FAILED" or "SUPERSEDED" status
- ❌ Similar question tested and rejected (e.g., Δk universality)
- ❌ Listed in historical/README.md as explored

**If fails**: Question was already investigated and failed. Read why before re-attempting.

---

### ☐ 4. Source Validation

**Check**: Where did this question come from?

**Valid sources**:
- ✅ Papers 1-4 explicitly list as "future work"
- ✅ Logical extension of established results (with clear paper citation)
- ✅ Reviewer question on papers (documented)
- ✅ New experimental data requiring framework update

**Invalid sources**:
- ❌ Found in mixed historical/current document (e.g., COMPLETE_FLAVOR_UNIFICATION.md)
- ❌ From exploration notes without "current framework" label
- ❌ Based on misreading of historical documents as current
- ❌ "Seems interesting" without paper justification

**If invalid**: Verify against Papers 1-4 before proceeding.

---

### ☐ 5. Testability

**Check**: Can this question be answered definitively?

**Pass criteria**:
- ✅ Clear success condition (e.g., "If χ² < 2, hypothesis supported")
- ✅ Clear failure condition (e.g., "If Δχ² > 3.84, hypothesis rejected")
- ✅ Specific observables to compute
- ✅ Known data to compare against

**Fail indicators**:
- ❌ Vague question like "explore connections between X and Y"
- ❌ No way to know when investigation is complete
- ❌ No clear criteria for success vs failure
- ❌ Philosophical rather than computational

**If fails**: Refine question to be specific and testable before starting.

---

## Red Flags (Stop Immediately If You See These)

### 🚩 Multiple τ Values
```
"τ_leptonic = 3.25i, τ_hadronic = 1.422i"
"τ-ratio = 7/16"
"Different sectors have different modular parameters"
```
→ **WRONG**: Framework uses single τ = 2.69i

### 🚩 Universal Δk=2
```
"Δk=2 should extend to quarks"
"Test Δk=2 as universal geometric law"
"All sectors must have Δk=2 spacing"
```
→ **WRONG**: Δk=2 is leptonic only (tested and rejected for quarks)

### 🚩 Historical Document as Source
```
"FALSIFICATION_DISCOVERY.md says..."
"COMPLETE_FLAVOR_UNIFICATION.md shows..."
"TOE_PATHWAY.md mentions..."
```
→ **DANGER**: These mix historical and current content. Verify against Papers 1-4.

### 🚩 k-Level Confusion
```
"k₂=16 (tadpole), k₃=7 (orbifold) → 7/16"
"Derive τ-ratio from gauge kinetic levels"
```
→ **WRONG**: No τ-ratio exists (k₂, k₃ are modular levels, not τ values)

---

## What to Do If Checklist Fails

### Option 1: Verify Against Papers (Recommended)
1. Read relevant section of Papers 1-4 carefully
2. Check if question is explicitly mentioned
3. Verify framework parameters (τ = 2.69i, etc.)
4. If passes after verification, proceed with caution

### Option 2: Consult Historical Docs
1. Check `docs/historical/README.md` for similar attempts
2. Read why previous approach failed
3. Determine if your question avoids those pitfalls
4. Document why your approach is different

### Option 3: Ask for Clarification
1. Document the ambiguity you found
2. List which sources contradict each other
3. Ask which source is authoritative (answer: Papers 1-4)
4. Wait for clarification before proceeding

### Option 4: Mark as "Needs Review"
1. Add question to parking lot with "NEEDS REVIEW" status
2. Note which checklist items failed
3. Don't start investigation until reviewed
4. Continue with verified questions instead

---

## Examples

### ✅ GOOD: Question Passes All Checks

**Question**: "Does τ = 2.69i remain stable under two-loop RG running?"

**Checklist**:
1. ✅ Papers 1-4: Paper 1 Section 6.2 mentions "two-loop corrections not yet computed"
2. ✅ Framework: Uses τ = 2.69i (correct)
3. ✅ Historical: Not in docs/historical/ (new question)
4. ✅ Source: Paper 1 explicitly lists as future work
5. ✅ Testable: Compute two-loop β-functions, check if Δτ << 0.05

**Action**: Proceed with investigation ✓

---

### ❌ BAD: Question Fails Multiple Checks

**Question**: "Why does τ_leptons/τ_quarks = 7/16 match gauge coupling ratio?"

**Checklist**:
1. ❌ Papers 1-4: No mention of different τ values (single τ = 2.69i everywhere)
2. ❌ Framework: Assumes multiple τ values (contradicts established framework)
3. ❌ Historical: Found in docs/historical/2024_06_multi_tau_exploration.md (FAILED)
4. ❌ Source: From FALSIFICATION_DISCOVERY.md (historical doc, not paper)
5. ✅ Testable: Could compute, but based on false premise

**Red flags**: 🚩 Multiple τ values, 🚩 τ-ratio, 🚩 Historical doc as source

**Action**: STOP - This was already investigated and failed. Framework uses single τ = 2.69i.

---

### ⚠️ UNCERTAIN: Needs More Verification

**Question**: "Does C = 2k_avg + 1 extend to quark sector?"

**Checklist**:
1. ❓ Papers 1-4: Not explicitly mentioned (need to search carefully)
2. ✅ Framework: Uses τ = 2.69i, talks about k-values (correct)
3. ❓ Historical: Similar to Δk=2 test (which failed), but different question
4. ⚠️ Source: From Path A notes (need to verify against papers)
5. ✅ Testable: Compute k_avg for quarks, check if C = 2k_avg + 1

**Action**:
1. Search Papers 1-4 for any mention of C = 2k_avg + 1 or k_avg
2. Check if similar to failed Δk=2 universality test
3. If genuinely new and not contradicted, proceed cautiously
4. If can't verify, mark "NEEDS REVIEW"

---

## Quick Reference Card

```
┌────────────────────────────────────────────────┐
│  BEFORE INVESTIGATING:                         │
├────────────────────────────────────────────────┤
│  1. In Papers 1-4 as open?          ☐ Yes/No  │
│  2. Uses τ = 2.69i (not multi-τ)?   ☐ Yes/No  │
│  3. Not in docs/historical/?         ☐ Yes/No  │
│  4. Valid source (not mixed docs)?  ☐ Yes/No  │
│  5. Testable with clear criteria?   ☐ Yes/No  │
│                                                │
│  RED FLAGS:                                    │
│  🚩 τ_leptons ≠ τ_quarks                       │
│  🚩 τ-ratio = 7/16                             │
│  🚩 Δk=2 universal                             │
│  🚩 Source: FALSIFICATION_DISCOVERY.md         │
│                                                │
│  ALL CHECKS PASS? → Proceed ✓                 │
│  ANY CHECK FAILS? → Verify or STOP ✗          │
└────────────────────────────────────────────────┘
```

---

## Template for Documenting Checks

When starting investigation, document your checklist:

```markdown
## Investigation: [Question Title]

### Pre-Investigation Checklist

**1. Papers 1-4 Verification**:
- Searched for: [search terms]
- Found in: [Paper X, Section Y, page Z]
- Status: ✅ PASS / ❌ FAIL / ❓ UNCERTAIN

**2. Framework Consistency**:
- Uses τ = 2.69i: ✅ Yes / ❌ No
- Consistent with established results: ✅ Yes / ❌ No / ❓ Uncertain
- Status: ✅ PASS / ❌ FAIL

**3. Historical Check**:
- Searched: docs/historical/
- Found: [None / Similar in file X]
- Status: ✅ PASS / ❌ FAIL

**4. Source Validation**:
- Source: [Papers 1-4 / Other]
- Reliability: [High / Medium / Low]
- Status: ✅ PASS / ❌ FAIL

**5. Testability**:
- Success criteria: [Specific condition]
- Failure criteria: [Specific condition]
- Status: ✅ PASS / ❌ FAIL

**Overall**: ✅ ALL PASS - Proceed / ❌ FAILED - STOP / ⚠️ Review needed

**Approval**: [Date, Name]
```

---

## Maintenance

**Update this checklist when**:
- New failure mode discovered (add to red flags)
- Papers submitted/revised (update verification process)
- Historical directory grows (update what to check)
- New confusion sources identified (add examples)

**Review frequency**: After each major investigation or mistake

**Owner**: Repository maintainer

Last updated: 2025-12-28 | Maintained by: Kevin Heitfeld
