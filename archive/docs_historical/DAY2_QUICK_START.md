# Quick Start: Day 2 Literature Search

**Date**: December 29, 2025
**Goal**: Find precedent for τ = 27/10 formula
**Time Budget**: 4 hours

---

## Morning Session (2 hours): Weigand + ArXiv Q1

### Task 1: Weigand Lectures (90 min)
**Paper**: arXiv:1806.01854 "Lectures on F-theory compactifications"
**Download**: https://arxiv.org/abs/1806.01854

**Read these sections**:
1. Section 3: Complex structure moduli space (~30 min)
   - Look for: How τ is determined in F-theory/Type IIB
   - Search for: "complex structure", "modular parameter", "stabilization"

2. Section 5: Rational points and special loci (~30 min)
   - Look for: Special values of moduli, rational points
   - Search for: "rational", "special", "discrete"

3. Skim introduction and conclusions (~30 min)
   - Get context on moduli space structure
   - Note any relevant references to follow

**Record in**: TAU_LITERATURE_SEARCH.md under "Findings Log"

---

### Task 2: ArXiv Query Q1 (30 min)
**Query**: Direct Formula Search (HIGH priority)

**Steps**:
1. Go to: https://arxiv.org/search/advanced
2. Copy this query into "Abstract" field:
   ```
   abs:"complex structure" AND abs:"orbifold" AND (abs:"formula" OR abs:"determination")
   ```
3. Filter: hep-th category, last 10 years
4. Scan first 20 results
5. Download 3-5 most promising papers

**Look for in titles/abstracts**:
- "complex structure determination"
- "orbifold moduli"
- "formula" + "orbifold"
- Explicit formulas in abstracts

**Expected**: 50-100 results, ~5 worth downloading

---

## Afternoon Session (2 hours): Paper Downloads + Quick Scans

### Task 3: Download Priority Papers (30 min)

**Must download** (if available):
1. arXiv:2001.07972 (Kobayashi-Otsuka 2020) - Modular symmetries
2. arXiv:2408.13984 (Kobayashi-Otsuka 2024) - Recent work
3. Top 3-5 from ArXiv Q1 results

**How to download**:
- Click paper title → Click "PDF" button
- Save to: `research/papers/` folder
- Name: `YYMM_FirstAuthor_ShortTitle.pdf`

---

### Task 4: Quick Scan All Downloaded Papers (90 min)

**For each paper** (~15 min each):

1. **Read abstract** (2 min)
   - Does it mention complex structure determination?
   - Any formula for moduli values?

2. **Scan introduction** (3 min)
   - What's the main topic?
   - Relevant to our question?

3. **Jump to sections on moduli** (5 min)
   - Search PDF for: "complex structure", "formula", "modular parameter"
   - Read any relevant paragraphs

4. **Check conclusions** (2 min)
   - Summary of results
   - Any formulas mentioned?

5. **Skim references** (3 min)
   - Note any papers titled about "complex structure" or "orbifolds"
   - Add promising ones to download list

**Record findings**:
```bash
# Use interactive mode
python research/literature_search_helper.py --interactive

# For each finding:
Command: add
Source: arXiv:XXXX.XXXXX
Content: [Quote or summary of relevant content]
Assessment: EXACT_MATCH / RELATED / NOT_RELEVANT
```

---

## End of Day 2 Assessment (15 min)

### Questions to Answer:
1. Did I find the exact formula τ = k/(N₁+N₂+h^{1,1})?
   - [ ] YES → Record source, prepare to cite
   - [ ] NO → Continue search

2. Did I find related formulas?
   - [ ] YES → List them, assess similarity
   - [ ] NO → Note this

3. How many papers checked?
   - Target: 6-10 papers
   - Actual: _____

4. Confidence level on novelty?
   - [ ] High (no precedent found)
   - [ ] Medium (need more searching)
   - [ ] Low (found something similar)

### Update Documents:
- [ ] TAU_LITERATURE_SEARCH.md - Add all findings
- [ ] WEEK1_PROGRESS_TRACKER.md - Mark Day 2 complete
- [ ] literature_findings.json - Auto-saved from interactive mode

---

## Common Pitfalls to Avoid

❌ **Don't**: Read entire papers cover-to-cover
✓ **Do**: Strategic scanning of relevant sections

❌ **Don't**: Get distracted by interesting but irrelevant content
✓ **Do**: Stay focused on complex structure formulas

❌ **Don't**: Download 50 papers "just in case"
✓ **Do**: Be selective, download only promising ones

❌ **Don't**: Forget to record findings
✓ **Do**: Use interactive tool or document immediately

---

## Success Criteria for Day 2

**Minimum (Must achieve)**:
- ✅ Read Weigand sections on complex structure
- ✅ Run ArXiv Q1 and scan results
- ✅ Download 5+ relevant papers
- ✅ Quick scan all downloaded papers
- ✅ Record findings (even if negative)

**Target (Should achieve)**:
- ✅ Clear sense if formula exists in standard references
- ✅ List of related formulas (if any)
- ✅ 3-5 papers identified for deep read (Day 3)

**Stretch (Nice to have)**:
- ✅ Found exact formula (precedent)
- ✅ Identified this as genuinely novel
- ✅ Expert contact identified for consultation

---

## If You Find Exact Match

🎯 **EXACT FORMULA FOUND!**

**Immediate actions**:
1. Screenshot the relevant page
2. Record full citation
3. Note where formula appears (equation number, page)
4. Check: Do they derive it or cite someone else?
5. If they cite → Follow that reference!

**Then**:
- Update TAU_LITERATURE_SEARCH.md with "PRECEDENT FOUND"
- Prepare citation for Paper 4
- Move to Week 2 (modular weights verification)

---

## If No Match Found

⚠️ **No precedent yet**

**Don't panic** - This is expected after just 1 day

**Continue**:
- Day 3: Run remaining ArXiv queries (Q2-Q6)
- Day 3: Deep dive on Kobayashi-Otsuka papers
- Day 3: Assess if genuinely novel

**Remember**: "Not found" ≠ "doesn't exist"
- May be expressed differently
- May be implicit in literature
- May genuinely be novel

---

## Tools Available

### Literature Search
```bash
# Get search instructions
python research/literature_search_helper.py

# Record findings interactively
python research/literature_search_helper.py --interactive
```

### Verification (If needed)
```bash
# Re-run numerical checks
python research/tau_27_10_verification.py
```

### Progress Tracking
- `docs/research/WEEK1_PROGRESS_TRACKER.md` - Daily updates
- `docs/research/TAU_LITERATURE_SEARCH.md` - Systematic search log
- `research/literature_findings.json` - Structured findings database

---

## Contact for Help

If you get stuck or find something exciting:
- Document it immediately
- Continue with remaining tasks
- Plan expert consultation for Day 3 if needed

---

**Remember**: The goal is NOT to prove novelty today.
**The goal is**: Systematically check if formula exists in standard places.

Tomorrow (Day 3) we'll make the final assessment.

Good luck! 🚀
