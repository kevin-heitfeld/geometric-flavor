# Manuscript: Zero-Parameter Flavor Framework from Calabi-Yau Topology

**Status:** In preparation for arXiv submission (January 2025)

## Directory Structure

```
manuscript/
├── main.tex                    # Main LaTeX document
├── references.bib              # Bibliography (BibTeX format)
├── README.md                   # This file
│
├── sections/                   # Main text sections
│   ├── 01_introduction.tex     # Introduction and motivation
│   ├── 02_framework.tex        # Framework and assumptions
│   ├── 03_calculation.tex      # Calculation methodology (TODO)
│   ├── 04_results.tex          # Results and data comparison (TODO)
│   ├── 05_predictions.tex      # Testable predictions (TODO)
│   ├── 06_discussion.tex       # Discussion and robustness (TODO)
│   └── 07_conclusions.tex      # Conclusions (TODO)
│
├── appendices/                 # Supplemental material
│   ├── appendix_a_yukawa_details.tex      # Complete Yukawa derivation (TODO)
│   ├── appendix_b_operator_basis.tex      # Operator basis analysis (TODO)
│   ├── appendix_c_kklt_uncertainty.tex    # KKLT uncertainty derivation (TODO)
│   ├── appendix_d_wrapping_scan.tex       # Alternative configurations (TODO)
│   ├── appendix_e_modular_forms.tex       # Modular form derivation (TODO)
│   └── appendix_f_numerical_methods.tex   # Numerical implementation (TODO)
│
└── figures/                    # All figures (vector format preferred)
    ├── figure1_geometry.pdf            # CY geometry schematic (TODO)
    ├── figure2_agreement.pdf           # Parameter agreement plot (TODO)
    ├── figure3_predictions.pdf         # Predictions timeline (TODO)
    ├── figure4_phase_diagram.pdf       # KKLT valid region (TODO)
    └── supplemental/                   # Supplemental figures
        ├── figS1_operator_basis.pdf    # From appendix_b_operator_basis.py
        ├── figS2_kklt_uncertainty.pdf  # From appendix_c_moduli_uncertainty.py
        └── ...

```

## Compilation Instructions

### Standard Compilation

```bash
cd manuscript/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Using latexmk (Recommended)

```bash
cd manuscript/
latexmk -pdf main.tex
```

### Clean Build Files

```bash
latexmk -c    # Remove auxiliary files
latexmk -C    # Remove auxiliary files + PDF
```

## Dependencies

Required LaTeX packages (all standard in TeX Live / MiKTeX):
- amsmath, amssymb, amsfonts, amsthm
- graphicx, xcolor, tikz
- hyperref, cleveref
- natbib (for bibliography)
- booktabs, multirow, array (for tables)
- physics (for bra-ket notation)

## Current Status

### Completed ✓
- [x] Main document structure (`main.tex`)
- [x] Section 1: Introduction (`01_introduction.tex`) - 2,200 words
- [x] Section 2: Framework (`02_framework.tex`) - 2,400 words
- [x] Section 3: Calculation (`03_calculation.tex`) - 3,800 words
- [x] Section 4: Results (`04_results.tex`) - 3,200 words
- [x] Section 5: Predictions (`05_predictions.tex`) - 4,200 words
- [x] Section 6: Discussion (`06_discussion.tex`) - 5,800 words
- [x] Section 7: Conclusions (`07_conclusions.tex`) - 2,600 words
- [x] Bibliography skeleton (`references.bib`) - 30+ entries
- [x] All 6 appendices (15,800 words total)
- [x] 4 main figures generated (PDF + PNG formats)

**Main text complete: 24,200 words (7 sections)**
**Appendices complete: 15,800 words (6 appendices)**
**Figures: 5/10 complete (4 main + 1 summary figure)**

### In Progress 🚧
- [ ] Supplemental figures (6 figures from appendix analyses)

**Appendices complete: 6/6 appendices (15,800 words)**
- [x] Appendix A: Complete Yukawa derivation (3,200 words)
- [x] Appendix B: Operator basis analysis (3,400 words)
- [x] Appendix C: KKLT moduli uncertainty (2,800 words)
- [x] Appendix D: Wrapping scan (2,600 words)
- [x] Appendix E: Modular forms (2,200 words)
- [x] Appendix F: Numerical methods (1,600 words)

### Todo 📝
- [ ] Generate 6 supplemental figures (for appendices)
- [ ] Proofreading and consistency checks (cross-references, equation numbering)
- [ ] Compile full LaTeX document and verify all references
- [ ] arXiv submission package preparation (ancillary files, metadata)

**Bibliography:** 30+ references complete (sufficient coverage)

**Main Figures Generated:**
1. `figure1_geometry.pdf/.png` - CY geometry with D7-branes ✓
2. `figure2_agreement.pdf/.png` - Parameter comparison (6 panels) ✓
3. `figure2_agreement_summary.pdf/.png` - Pull plot (19 parameters) ✓
4. `figure3_predictions.pdf/.png` - Experimental timeline ✓
5. `figure4_phase_diagram.pdf/.png` - KKLT moduli space ✓

## Relation to Python Code

The appendices should incorporate results from:
- `appendix_b_operator_basis.py` → `appendices/appendix_b_operator_basis.tex`
- `appendix_c_moduli_uncertainty.py` → `appendices/appendix_c_kklt_uncertainty.tex`
- `theory14_complete_fit_optimized.py` → Results tables in Section 4
- `prove_c2_dominance.py` → Supplement to Appendix B

Figures are generated by Python scripts and saved to `figures/` or `figures/supplemental/`.

## Word Count Summary

- **Main text:** 24,200 words (7 sections complete)
  - Section 1 (Introduction): 2,200 words
  - Section 2 (Framework): 2,400 words
  - Section 3 (Calculation): 3,800 words
  - Section 4 (Results): 3,200 words
  - Section 5 (Predictions): 4,200 words
  - Section 6 (Discussion): 5,800 words
  - Section 7 (Conclusions): 2,600 words

- **Appendices:** 15,800 words (6 appendices complete)
  - Appendix A: 3,200 words
  - Appendix B: 3,400 words
  - Appendix C: 2,800 words
  - Appendix D: 2,600 words
  - Appendix E: 2,200 words
  - Appendix F: 1,600 words

- **Total:** 40,000 words (~80-100 pages estimated)

## Contact

Kevin Heitfeld
Email: kheitfeld@gmail.com (UPDATE)
GitHub: github.com/kevin-heitfeld/geometric-flavor

## License

This manuscript is licensed under CC BY 4.0 (Creative Commons Attribution 4.0 International) upon publication. Code remains under MIT License.
