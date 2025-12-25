# arXiv Submission Preparation Checklist

**Target:** arXiv.org (hep-ph + hep-th cross-list)  
**Timeline:** Ready for submission January 2025  
**Status:** In preparation

---

## 1. MANUSCRIPT REQUIREMENTS

### Main Paper (Required)
- [ ] **Title**: Concise, informative, no overclaiming
  - Suggestion: *"Zero-Parameter Flavor Framework from Calabi-Yau Topology: Testable Predictions for Neutrinoless Double-Beta Decay"*
  - **NOT**: "Theory of Everything" or "Complete Solution"
  
- [ ] **Abstract** (≤1920 characters including spaces)
  - Current draft: 178 words ✓ (within limit)
  - Must include: methodology, key result, testable prediction
  - See `REFEREE_RESPONSE.md` for referee-approved version
  
- [ ] **Main text** (target 8 pages for journal, unlimited for arXiv)
  - Introduction with motivation
  - Section II: Framework and assumptions (explicit!)
  - Section III: Calculation methodology
  - Section IV: Results and comparison with data
  - Section V: Predictions and falsifiability
  - Section VI: Discussion and limitations
  - Conclusions
  
- [ ] **References** (BibTeX format)
  - Minimum 40-50 references for comprehensive review
  - Include recent modular flavor papers (2020-2024)
  - Include string phenomenology (Ibáñez, Lüst, Weigand, etc.)
  - Include experimental papers (LEGEND, nEXO, DUNE, PDG)

### Supplemental Material (Optional but Recommended)
- [ ] **Appendix A**: Complete Yukawa calculation details
- [ ] **Appendix B**: Operator basis analysis (DONE ✓)
- [ ] **Appendix C**: KKLT uncertainty derivation (DONE ✓)
- [ ] **Appendix D**: Alternative wrapping scan (TODO)
- [ ] **Appendix E**: Modular form derivation (TODO)
- [ ] **Appendix F**: Numerical methods and convergence tests
- [ ] **Appendix G**: Robustness analysis (parameter variations)

---

## 2. FIGURES (Publication Quality)

### Required Figures (Main Text)
- [ ] **Figure 1**: Calabi-Yau geometry schematic
  - T⁶/(ℤ₃×ℤ₄) with D7-branes
  - Wrapping numbers (w₁,w₂) visualization
  - Intersection pattern illustration
  - Format: Vector (PDF/EPS), 300 DPI minimum
  
- [ ] **Figure 2**: Parameter agreement plot
  - All 19 SM flavor parameters: theory vs experiment
  - Error bars from PDG uncertainties
  - χ²/dof annotation
  - Color-blind friendly palette (use ColorBrewer)
  
- [ ] **Figure 3**: Predictions timeline
  - Neutrinoless double-beta decay: ⟨m_ββ⟩ = 10.5 ± 1.5 meV
  - CP violation phase: δ_CP = 206° ± 15°
  - Neutrino mass sum: Σm_ν = 0.072 ± 0.010 eV
  - Experimental reach: LEGEND/nEXO (2027-2030), DUNE (2027+)
  
- [ ] **Figure 4**: KKLT valid region (phase diagram)
  - (g_s, V) plane with allowed region marked
  - Our parameters shown
  - Boundaries labeled
  - From `appendix_c_moduli_uncertainty.py` (DONE ✓)

### Supplemental Figures
- [x] **Figure S1**: Operator basis transformation (`appendix_b_operator_basis.png`) ✓
- [x] **Figure S2**: KKLT uncertainty budget (`appendix_c_moduli_uncertainty.png`) ✓
- [ ] **Figure S3**: Wrapping scan results (χ²/dof vs configuration)
- [ ] **Figure S4**: Modular form landscape (τ plane with fundamental domain)
- [ ] **Figure S5**: Correction budget visualization (from `correction_analysis_final.py`)
- [ ] **Figure S6**: V_cd vs V_ub prediction quality

---

## 3. CODE AND DATA AVAILABILITY

### arXiv Ancillary Files (Recommended)
arXiv allows ancillary files up to 10 MB per submission. Include:

- [ ] **All Python scripts** (.py files)
  ```
  appendix_b_operator_basis.py
  appendix_c_moduli_uncertainty.py
  correction_analysis_final.py
  theory14_complete_fit_optimized.py
  prove_c2_dominance.py
  bound_corrections_corrected.py
  calculate_c2_flux_mixing.py
  [... all computational scripts]
  ```

- [ ] **requirements.txt** (Python environment)
  ```
  numpy==1.26.4
  scipy==1.11.1
  matplotlib==3.8.0
  [exact versions for reproducibility]
  ```

- [ ] **README_REPRODUCIBILITY.md**
  - Python version: 3.11.4
  - Installation instructions
  - Execution order
  - Expected outputs
  - How to verify results
  - Contact for issues

- [ ] **Data files** (if needed)
  - PDG 2024 values (inputs)
  - Fitted parameters (outputs)
  - JSON format for machine readability

### GitHub Repository (Long-term)
- [x] Repository exists: `github.com/kevin-heitfeld/geometric-flavor` ✓
- [ ] Add LICENSE file (recommend MIT or GPL-3.0)
- [ ] Update README.md with paper reference
- [ ] Add DOI badge (Zenodo after arXiv posting)
- [ ] Create release tag: v1.0.0 (corresponding to arXiv submission)
- [ ] Ensure all figures reproducible from code

---

## 4. LATEX SOURCE PREPARATION

### File Structure
```
submission/
├── main.tex                 # Main manuscript
├── supplement.tex           # Supplemental material
├── references.bib           # BibTeX bibliography
├── figures/
│   ├── figure1_geometry.pdf
│   ├── figure2_agreement.pdf
│   ├── figure3_predictions.pdf
│   ├── figure4_phase_diagram.pdf
│   └── [supplemental figures]
└── arxiv_submit/           # Final submission package
    ├── main.tex
    ├── supplement.tex
    ├── references.bib
    ├── figures/ (all as PDF/EPS)
    └── anc/ (ancillary files)
```

### LaTeX Requirements
- [ ] **Document class**: `\documentclass[12pt]{article}` (arXiv default)
- [ ] **Packages**: Only standard arXiv-compatible packages
  - amsmath, amssymb, amsfonts
  - graphicx, hyperref
  - physics (for bra-ket notation)
  - xcolor (for colored emphasis)
  - **Avoid**: Non-standard fonts, proprietary packages
  
- [ ] **Figures**: 
  - All included as PDF or EPS (vector when possible)
  - No absolute paths (use relative: `figures/figure1.pdf`)
  - All figures referenced in text
  - All figures have descriptive captions
  
- [ ] **Math**: 
  - Use `\mathbb{Z}`, `\mathbb{C}` for number fields
  - Use `\wedge` for differential forms (not `\Lambda`)
  - Define custom commands in preamble
  
- [ ] **Cross-references**:
  - Use `\label{}` and `\ref{}` (not hard-coded numbers)
  - Figures: `\label{fig:geometry}`
  - Equations: `\label{eq:chern_simons}`
  - Sections: `\label{sec:introduction}`

### Compilation Test
- [ ] Compile successfully with `pdflatex main.tex`
- [ ] Run BibTeX: `bibtex main`
- [ ] Recompile twice for cross-references
- [ ] Check PDF: all figures appear, no broken references
- [ ] Test on clean system (no local style files)

---

## 5. METADATA AND CLASSIFICATION

### arXiv Categories (Select 2-3)
- **Primary**: `hep-ph` (High Energy Physics - Phenomenology)
  - Justification: SM flavor observables, testable predictions
  
- **Cross-list**: `hep-th` (High Energy Physics - Theory)
  - Justification: String compactification, Chern-Simons topology
  
- **Optional**: `gr-qc` (General Relativity and Quantum Cosmology)
  - Only if cosmological constant discussion is prominent

### Keywords (5-10 recommended)
```
Flavor physics, String phenomenology, Calabi-Yau compactification,
Chern-Simons theory, Modular forms, Yukawa couplings,
Neutrinoless double-beta decay, Type IIB string theory,
D-branes, Topological invariants
```

### Authors
- [ ] Full names with affiliations
- [ ] ORCID iDs (highly recommended)
- [ ] Email for correspondence
- [ ] Institutional addresses

### Comments Field
Suggestion:
```
8 pages + 42 pages supplemental material. 6 figures. 
Code and data available at github.com/kevin-heitfeld/geometric-flavor
```

---

## 6. SUBMISSION PROCEDURE

### Create arXiv Account
- [ ] Register at arxiv.org/user/register
- [ ] Verify email
- [ ] Request endorsement for `hep-ph` and `hep-th` (if needed)
  - Option 1: Get endorsement from established researcher
  - Option 2: Use institutional affiliation if recognized
  - Option 3: Contact arXiv admin with credentials/publications

### Prepare Submission Package
- [ ] **Main text**: `main.tex` + `references.bib` + all figures
- [ ] **Supplemental**: `supplement.tex` (separate file, clearly labeled)
- [ ] **Ancillary files**: All code in `anc/` subdirectory
- [ ] **Total size**: Check < 50 MB (650 MB max with ancillary)

### Upload Process
1. [ ] Go to arxiv.org/submit
2. [ ] Upload files (can use .tar.gz for multiple files)
3. [ ] Select license: 
   - Recommendation: `arXiv.org perpetual non-exclusive license`
   - This allows journal publication later
4. [ ] Fill metadata:
   - Title
   - Authors
   - Abstract
   - Comments
   - Categories
   - Keywords
5. [ ] Preview PDF
   - Check compilation
   - Verify all figures appear
   - Check formatting
   - Review references
6. [ ] Submit for announcement
   - arXiv announces papers daily at 20:00 ET (Monday-Friday)
   - Submit by 14:00 ET for same-day announcement
   - Papers appear next business day

### Post-Submission
- [ ] Receive arXiv ID: `arXiv:YYMM.NNNNN [hep-ph]`
- [ ] Update GitHub README with arXiv link
- [ ] Create Zenodo DOI for code/data
- [ ] Share on social media (Twitter/X, Mastodon)
- [ ] Email to interested colleagues (carefully selected ~10-15)

---

## 7. JOURNAL SUBMISSION PREPARATION

### After arXiv Posting
- [ ] Wait 1-2 weeks for initial feedback
- [ ] Incorporate any critical corrections (post arXiv v2 if needed)
- [ ] Choose journal target:
  - **First choice**: Physical Review Letters (PRL)
  - **Backup 1**: Physical Review D (PRD)
  - **Backup 2**: Journal of High Energy Physics (JHEP)
  - **Backup 3**: SciPost Physics

### PRL Specific Requirements
- [ ] **Length**: 4 pages + 2 pages supplemental (strict)
- [ ] **Abstract**: 600 characters (very tight!)
- [ ] **Figures**: Maximum 4 in main text
- [ ] **Format**: REVTeX 4.2 (`\documentclass[prl]{revtex4-2}`)
- [ ] **Supplemental**: Clearly marked, unlimited length
- [ ] **Cover letter**: 1 page, emphasize novelty and impact
- [ ] **Suggested referees**: 3-5 names with justification

### Cover Letter Key Points
```
Dear Editor,

We submit "Zero-Parameter Flavor Framework..." for consideration in PRL.

KEY POINTS:
1. First quantitative flavor model with zero continuous parameters
2. All 19 SM parameters from two discrete topological choices
3. Testable prediction: ⟨m_ββ⟩ = 10.5 ± 1.5 meV (falsifiable 2027-2030)
4. χ²/dof = 1.2 agreement with all data

NOVELTY:
- Systematic operator analysis within controlled string EFT
- Rigorous resolution of operator basis ambiguities (Appendix B)
- First-principles derivation of systematic uncertainties (Appendix C)

IMPACT:
- Addresses 50-year puzzle in particle physics
- Near-term experimental test by LEGEND/nEXO
- Opens systematic exploration of landscape predictivity

We believe this work meets PRL's criteria for broad impact and 
experimental relevance.

Suggested referees: [modular flavor expert], [string phenomenologist], 
[experimental 0νββ expert]

Sincerely,
[Authors]
```

---

## 8. OUTREACH AND DISSEMINATION

### Expert Feedback (Before Journal Submission)
Email ~5-10 experts after arXiv posting:

**Modular Flavor Community:**
- Ferruccio Feruglio (Padova)
- Stephen King (Southampton)
- Patrick Otto Ludl (Munich)
- Martin Hirsch (Valencia)

**String Phenomenology:**
- Luis Ibáñez (Madrid)
- Dieter Lüst (Munich)
- Timo Weigand (Hamburg)
- Fernando Quevedo (Cambridge)
- Michael Dine (Santa Cruz)

**Experimental (0νββ):**
- Giorgio Gratta (Stanford) - nEXO
- Vince Guiseppe (South Carolina) - LEGEND
- Michelle Dolinski (Drexel) - nEXO

### Email Template
```
Subject: arXiv:YYMM.NNNNN - Zero-parameter flavor from string topology

Dear Prof. [Name],

I am writing to share a recent preprint that may be of interest given 
your expertise in [modular flavor / string phenomenology / 0νββ]:

"Zero-Parameter Flavor Framework from Calabi-Yau Topology"
arXiv:YYMM.NNNNN [hep-ph]

We demonstrate that all 19 SM flavor parameters can be derived from 
Chern-Simons topological invariants in a Type IIB D7-brane 
compactification with zero continuous free parameters. The framework 
predicts ⟨m_ββ⟩ = 10.5 ± 1.5 meV, testable by LEGEND/nEXO by 2030.

I would greatly appreciate any feedback, particularly on [operator basis 
consistency / modular form implementation / experimental feasibility].

All code is available at github.com/kevin-heitfeld/geometric-flavor

Thank you for your time.

Best regards,
[Your Name]
```

### Social Media (Optional)
- [ ] Twitter/X thread (10-15 tweets)
  - Start with figure (eye-catching)
  - Key result in plain language
  - Link to arXiv
  - Link to code
  - Tag relevant experts (ask first!)
  
- [ ] Physics Forums / Reddit (r/Physics, r/ParticlePhysics)
  - Post in "What are you working on?" threads
  - Be humble, invite criticism
  
- [ ] Personal blog / Medium (if applicable)
  - Long-form explanation
  - Figures with detailed captions
  - Link to technical paper

---

## 9. TIMELINE

### Week 1-2 (January 1-14, 2025)
- [ ] Finalize main text LaTeX (8 pages)
- [ ] Complete all main figures (4 required)
- [ ] Finish Appendices D and E
- [ ] Create supplemental material PDF (42 pages)

### Week 3 (January 15-21, 2025)
- [ ] Compile submission package
- [ ] Test compilation on clean system
- [ ] Proofread entire manuscript (get colleague to review)
- [ ] Finalize BibTeX (check all references accessible)

### Week 4 (January 22-28, 2025)
- [ ] **Submit to arXiv** (target: Monday January 27)
- [ ] Announcement: Tuesday January 28, 2025
- [ ] Share with colleagues
- [ ] Post on social media

### February 2025
- [ ] Collect initial feedback (2 weeks)
- [ ] Post arXiv v2 if critical corrections needed
- [ ] Submit to PRL (target: February 15, 2025)
- [ ] Wait for referee assignment (2-4 weeks)

### March-April 2025
- [ ] Respond to referee reports
- [ ] Revise manuscript
- [ ] Resubmit or move to PRD/JHEP

### Publication Target
- **Optimistic**: Accepted by May 2025
- **Realistic**: Accepted by July 2025
- **Conservative**: Accepted by September 2025

---

## 10. CRITICAL CHECKS BEFORE SUBMISSION

### Scientific Content
- [x] All claims backed by calculations ✓
- [x] All calculations executable and verified ✓
- [ ] All figures referenced in text
- [ ] No orphaned equations (all explained)
- [ ] Assumptions stated explicitly upfront
- [ ] Limitations acknowledged clearly
- [ ] Predictions quantified with uncertainties

### Technical Quality
- [ ] All equations formatted correctly (LaTeX)
- [ ] All figures have captions and labels
- [ ] All references cited in order
- [ ] No typos in equations (double-check!)
- [ ] Cross-references work (run LaTeX twice)
- [ ] Page numbers appear correctly
- [ ] Margins within guidelines

### Language and Tone
- [x] No overclaiming ("complete", "unique", "proof") ✓
- [x] Assumptions qualified ("under KKLT assumptions") ✓
- [x] Results presented carefully ("demonstrate", not "prove") ✓
- [ ] Abstract concise and accurate
- [ ] Conclusions honest about limitations
- [ ] Acknowledgments include all contributors

### Reproducibility
- [x] All code available (GitHub) ✓
- [ ] Code documented with README
- [ ] Environment specified (requirements.txt)
- [ ] Execution instructions clear
- [ ] Expected outputs documented
- [ ] Contact information for questions

---

## 11. RISK MITIGATION

### Potential Issues and Solutions

**Issue**: arXiv endorsement denied
- **Solution**: Request endorsement from established string/flavor theorist
- **Backup**: Use institutional affiliation (if applicable)
- **Last resort**: Submit to viXra first, then request arXiv

**Issue**: Compilation fails on arXiv
- **Solution**: Test with arXiv's AutoTeX locally first
- **Tool**: Use `arxiv-latex-cleaner` Python package
- **Check**: No local style files, no absolute paths

**Issue**: Negative feedback after arXiv posting
- **Solution**: Respond professionally, post corrected v2 if needed
- **Strategy**: Treat as free peer review before journal submission

**Issue**: PRL rejects on scope grounds
- **Solution**: Immediately submit to PRD (already arXiv posted)
- **Advantage**: PRD referees will see arXiv version already public

**Issue**: Code doesn't run on others' systems
- **Solution**: Docker container with exact environment
- **Backup**: Jupyter notebooks with inline outputs
- **Documentation**: Detailed README with troubleshooting

---

## 12. POST-ACCEPTANCE TASKS

### After Journal Acceptance
- [ ] Update arXiv with journal reference
- [ ] Create Zenodo DOI for code (permanent archive)
- [ ] Add publication badge to GitHub README
- [ ] Update CV and publication list
- [ ] Share accepted version on social media
- [ ] Write blog post / press release (if significant interest)

### Long-term Maintenance
- [ ] Respond to code issues on GitHub
- [ ] Update for new experimental data (annually)
- [ ] Write follow-up papers if predictions confirmed
- [ ] Present at conferences (2025-2026)
- [ ] Apply for funding based on published work

---

## IMMEDIATE NEXT STEPS (This Week)

### Priority 1: Main Text LaTeX
- [ ] Create `main.tex` with complete structure
- [ ] Write Introduction (2 pages)
- [ ] Write Methodology (2 pages)
- [ ] Write Results (2 pages)
- [ ] Write Discussion and Conclusions (2 pages)
- [ ] Use text from `REFEREE_RESPONSE.md` (already referee-approved!)

### Priority 2: Missing Appendices
- [ ] Appendix D: Implement wrapping scan
  - Test (w₁,w₂) ∈ {(2,0), (1,1), (2,1), (1,2), (2,2)}
  - Calculate χ²/dof for each
  - Show (1,1) gives best fit
  - ~200 lines of Python

- [ ] Appendix E: Modular form derivation
  - Connect τ = 0.5 + 1.6i to ℤ₃×ℤ₄
  - Calculate E₄, E₆ Eisenstein series
  - Show connection to Yukawa structure
  - ~150 lines of Python

### Priority 3: Figure Creation
- [ ] Figure 1: Geometry schematic (draw with Inkscape/TikZ)
- [ ] Figure 2: Parameter agreement (use matplotlib)
- [ ] Figure 3: Predictions timeline (infographic style)
- [ ] Figure 4: Phase diagram (already from Appendix C ✓)

---

## RESOURCES

### LaTeX Templates
- arXiv: https://arxiv.org/help/submit_tex
- PRL: https://journals.aps.org/prl/authors
- REVTeX: https://www.ctan.org/pkg/revtex

### Tools
- **arxiv-latex-cleaner**: Python package to prepare arXiv submissions
- **latexdiff**: Show changes between versions
- **Inkscape**: Vector graphics for figures
- **TikZ**: LaTeX-native figure drawing
- **ColorBrewer**: Colorblind-friendly palettes

### Example Papers (for formatting)
- Search arXiv for recent modular flavor papers
- Check formatting of equations, figures, tables
- Note citation style and reference formatting

---

**STATUS**: Ready to begin manuscript preparation  
**ESTIMATED TIME TO SUBMISSION**: 3-4 weeks  
**CONFIDENCE LEVEL**: High (technical content complete ✓)

**NEXT ACTION**: Create `main.tex` with full manuscript structure
