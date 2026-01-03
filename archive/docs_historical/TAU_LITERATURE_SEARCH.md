# Literature Search: τ = k/(N₁ + N₂ + h^{1,1})

**Goal**: Find precedent or establish novelty of the formula τ = 27/10 from orbifold topology.

**Date Started**: December 28, 2025  
**Status**: Systematic search in progress

---

## The Formula

```
τ = k_lepton / X

where:
  k_lepton = N_Z3³ = 27
  X = N_Z3 + N_Z4 + h^{1,1} = 3 + 4 + 3 = 10
  
Result: τ = 2.7 (matches phenomenological τ = 2.69 ± 0.05)
```

---

## Search Progress

### ✓ COMPLETED: Initial Scan

**Checked**:
- PATH_A_PROGRESS_REPORT.md - Derivation documented
- PAPER4_COMPLETE_SUMMARY.md - No formula in Paper 4
- No obvious matches in local documentation

**Status**: No precedent in our existing papers

---

## SYSTEMATIC LITERATURE SEARCH

### Priority 1: Standard Textbooks ⏳ IN PROGRESS

#### Book 1: Ibanez-Uranga "String Theory and Particle Physics"
**Relevance**: THE textbook for Type IIB compactifications
**Chapters to check**:
- [ ] Ch 10: Toroidal orientifolds and orbifolds
- [ ] Ch 11: Intersecting brane worlds  
- [ ] Ch 12: Magnetized D-branes
- [ ] Appendix: Complex structure moduli

**What to look for**:
- Formulas determining τ from orbifold structure
- Relation between modular parameter and group orders
- Role of h^{1,1} in complex structure
- Rational values of τ in orbifold CFT

**Search keywords**: "complex structure", "orbifold", "modular parameter", "Z_N", "rational"

**Status**: ⏸ Awaiting book access

---

#### Book 2: Blumenhagen-Lüst-Theisen "Basic Concepts of String Theory"
**Relevance**: Comprehensive orbifold CFT treatment
**Chapters to check**:
- [ ] Ch 10: Toroidal compactifications
- [ ] Ch 11: Orbifold compactifications
- [ ] Sections on moduli space geometry

**Status**: ⏸ Awaiting book access

---

#### Book 3: Weigand "Lectures on F-theory"
**Relevance**: F-theory has Type IIB as limit, moduli space structure
**Sections to check**:
- [ ] Complex structure moduli space
- [ ] Rational points and special loci
- [ ] Connection to Type IIB orbifolds

**Status**: Available online - https://arxiv.org/abs/1806.01854

---

### Priority 2: Key Research Papers 📄

#### Must-Check Papers

##### 1. Kobayashi-Otsuka (Modular Flavor from Magnetized Branes)
**Papers**:
- [ ] arXiv:2001.07972 - "Classification of discrete modular symmetries" (2020)
- [ ] arXiv:2408.13984 - "Non-invertible flavor symmetries" (2024)

**Why relevant**: Directly addresses modular symmetry emergence from branes
**What to look for**: Complex structure determination from flux/magnetization

---

##### 2. Cremades-Ibanez-Marchesano (Original Magnetized Branes)
**Papers**:
- [ ] hep-th/0308001 - "Computing Yukawa couplings from magnetized branes" (2003)
- [ ] hep-th/0607160 - "Yukawa couplings in intersecting D-brane models" (2006)

**Why relevant**: Wave function formulas, modular transformations
**What to look for**: How τ appears in wave functions

---

##### 3. Dixon et al. (Orbifold Compactifications - Classic Papers)
**Papers**:
- [ ] Dixon-Harvey-Vafa-Witten, Nucl.Phys.B261:678 (1985) - Original orbifold paper
- [ ] Dixon-Kaplunovsky-Louis, Nucl.Phys.B329:27 (1990) - Orbifold phenomenology

**Why relevant**: Foundational orbifold CFT
**What to look for**: Complex structure in twisted sectors

---

##### 4. Aspinwall et al. (Calabi-Yau Moduli Space)
**Papers**:
- [ ] Aspinwall-Greene-Morrison, Nucl.Phys.B416:414 (1994) - Calabi-Yau moduli
- [ ] Aspinwall, hep-th/9611137 - Enhanced gauge symmetries

**Why relevant**: Geometry of moduli space
**What to look for**: Special points in complex structure moduli space

---

##### 5. Recent Modular Flavor Papers
**Papers**:
- [ ] Feruglio et al. - Modular invariance and neutrino masses
- [ ] Kobayashi-Shimizu - Modular symmetry in string compactifications
- [ ] Baur et al. - Modular flavor from F-theory

**Why relevant**: Current state of modular flavor research
**What to look for**: Any mention of τ determination from geometry

---

### Priority 3: ArXiv Systematic Search 🔍

#### Search 1: Direct Formula Search
**Query**: `"complex structure" AND "orbifold" AND ("formula" OR "determination")`
**Filter**: hep-th, hep-ph (last 10 years)
**Expected results**: ~50-100 papers
**Status**: ⏸ Not started

---

#### Search 2: Modular Parameter Search  
**Query**: `"modular parameter" AND ("Z_3" OR "Z_4") AND "compactification"`
**Filter**: hep-th (all time)
**Expected results**: ~20-50 papers
**Status**: ⏸ Not started

---

#### Search 3: Rational Tau Search
**Query**: `"rational" AND "tau" AND "string theory" AND "modular"`
**Filter**: hep-th, math-ph
**Expected results**: ~30-60 papers
**Status**: ⏸ Not started

---

#### Search 4: Product Orbifold Search
**Query**: `("Z_3 x Z_4" OR "Z_3 times Z_4") AND "complex structure"`
**Filter**: hep-th
**Expected results**: ~10-20 papers
**Status**: ⏸ Not started

---

### Priority 4: Expert Consultation 👥

#### Prepared Questions

**Question 1: Formula Recognition**
> "We derived the relation τ = k/(N₁+N₂+h^{1,1}) for product orbifolds T⁶/(Z_N₁×Z_N₂), 
> where k is the modular level and h^{1,1} is the Hodge number. 
> For Z₃×Z₄ on CY₃, this gives τ = 27/10 = 2.7, matching phenomenology.
> Is this formula known in the literature?"

**Question 2: Complex Structure Determination**
> "For Type IIB compactified on T⁶/(Z₃×Z₄), how is the complex structure modulus 
> typically determined? Are there selection rules from orbifold consistency?"

**Question 3: Modular Level Connection**
> "We observe k_lepton = N³ for the Z_N factor acting on lepton branes. 
> Is there a known geometric reason for this cubic relation?"

**Question 4: Alternative Realizations**
> "Are there other Z_N×Z_M product orbifolds that give τ ≈ 2.7? 
> Our scan suggests Z₃×Z₄ is unique—is this expected?"

**Potential experts to contact**:
- [ ] Fernando Marchesano (IFT Madrid) - fernando.marchesano@csic.es
- [ ] Tatsuo Kobayashi (Hokkaido) - kobayashi@particle.sci.hokudai.ac.jp  
- [ ] Arthur Hebecker (Heidelberg) - a.hebecker@thphys.uni-heidelberg.de
- [ ] Timo Weigand (Hamburg) - timo.weigand@desy.de

**Status**: ⏸ Awaiting literature search completion before contacting

---

## Findings Log

### Date: [TBD]
**Source**: [Paper/Book/Expert]
**Relevant content**:
```
[Quote or summary]
```
**Assessment**: [Precedent? / Related? / Not relevant]

---

## Similar Formulas Found

### None yet ⚠

Will document any related formulas here, even if not exact matches.

---

## Preliminary Assessment (After Day 1)

**Evidence this may be novel**:
1. No mention in our Papers 1-4
2. Not in PATH_A documentation despite extensive derivation
3. Clean rational formula (27/10) suggests fundamental significance
4. Uniqueness in parameter space (only Z₃×Z₄ works)

**Evidence this should exist in literature**:
1. Orbifold compactifications studied for 40+ years
2. Formula is "too simple" to have been missed
3. Type IIB moduli space well-studied
4. Modular flavor papers should have encountered this

**Most likely scenarios**:
1. **Exists but in different language** (50% probability)
   - Formula known, but expressed differently
   - Need to translate between formalisms
   
2. **Implicit in literature but not explicit** (30% probability)
   - Relationship understood but not written as formula
   - Can be derived from existing results
   
3. **Genuinely novel** (15% probability)
   - First explicit formula of this type
   - Significant if correct
   
4. **Incorrect or coincidence** (5% probability)
   - Formula is numerological
   - Deeper check reveals issues

---

## Next Actions (Days 2-3)

### Day 2 Morning
- [ ] Download Weigand F-theory lectures (arXiv:1806.01854)
- [ ] Read sections on complex structure moduli space
- [ ] Check if rational τ values discussed

### Day 2 Afternoon
- [ ] ArXiv search #1: Direct formula search
- [ ] Scan abstracts, download ~10 most relevant papers
- [ ] Quick read for any matching formulas

### Day 3 Morning
- [ ] ArXiv searches #2-4
- [ ] Deep dive on Kobayashi-Otsuka papers (modular flavor)
- [ ] Check Dixon orbifold classics

### Day 3 Afternoon
- [ ] Compile findings
- [ ] Assess: precedent found OR novel result
- [ ] Decide: proceed to derivation OR cite literature

---

## Success Criteria

By end of Day 3, we should know:
- ✓ or ✗ Formula exists in literature
- If ✓: Where? How expressed? How to cite?
- If ✗: Confidence level this is novel? (High/Medium/Low)

---

## Updates

**December 28, 2025 - Day 1**
- Initial search of local documentation: negative
- Numerical verification: formula works precisely (0.37% error)
- Uniqueness: Z₃×Z₄ only match in [2,10]×[2,10] scan
- Assessment: High probability of novelty OR exists in different formalism

**[To be continued Day 2...]**
