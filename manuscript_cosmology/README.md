# Paper 2: Complete Cosmology from Modular String Compactifications

**Status**: In preparation (initial structure created December 26, 2025)

## Overview

This manuscript (Paper 2) demonstrates that the same Type IIB string compactification that explains SM flavor (Paper 1) naturally accommodates:

- **Inflation** (α-attractor, n_s = 0.967, r = 0.003)
- **Dark Matter** (sterile neutrinos 83% + axions 17%)
- **Baryogenesis** (resonant leptogenesis, η_B exact match)
- **Strong CP** (modular axion from ρ Kähler modulus)

**Total**: 25 observables from unified modular framework

## Structure

### Completed Sections
- ✅ `main.tex` - Main LaTeX file with preamble, abstract, structure
- ✅ `01_introduction.tex` - Full introduction (~10 pages)
- ✅ `02_multimoduli_framework.tex` - Multi-moduli framework (~8 pages)

### To Be Completed (from exploration branch analysis)
- ⏳ `03_inflation.tex` - From `modular_inflation_analysis.py`
- ⏳ `04_dark_matter.tex` - From `sterile_neutrino_constraints.py`
- ⏳ `05_baryogenesis.tex` - From `LEPTOGENESIS_INVESTIGATION_COMPLETE.md`
- ⏳ `06_strong_cp.tex` - From `modular_axion_strong_cp.py`
- ⏳ `07_timeline.tex` - Synthesis of all sections
- ⏳ `08_predictions.tex` - Testable predictions catalog
- ⏳ `09_discussion.tex` - Theoretical discussion
- ⏳ `10_conclusions.tex` - Conclusions and outlook

## Source Data

All analysis scripts and results are on the `exploration/dark-matter-from-flavor` branch:

- `modular_inflation_analysis.py` → Section 3
- `sterile_neutrino_constraints.py` → Section 4
- `leptogenesis_final_parameter_table.py` → Section 5
- `modular_axion_strong_cp.py` → Section 6
- `EXPLORATION_BRANCH_SUMMARY.md` → Overview
- `INFLATION_EXPLORATION_COMPLETE.md` → Inflation details
- `LEPTOGENESIS_INVESTIGATION_COMPLETE.md` → Leptogenesis details

## Figures

Figures to be copied from exploration branch:
- `modular_inflation_analysis.png` → Figure for Section 3
- `sterile_neutrino_constraints.png` → Figure for Section 4
- `leptogenesis_parameter_space.png` → Figure for Section 5
- `modular_axion_parameter_space.png` → Figure for Section 6

## Timeline

- **Dec 2025**: Structure created, Introduction + Framework complete
- **Jan 2025**: Complete Sections 3-6 from exploration analysis
- **Feb 2025**: Complete Sections 7-10, polish abstract
- **Mar 2025**: Final revisions, prepare for submission
- **Apr-May 2025**: Submit after Paper 1 acceptance

## Compilation

```bash
cd manuscript_cosmology
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Key Achievement

**Inflation is no longer assumed** - it's now **derived** from the modular Kähler geometry! This strengthens Paper 2 significantly compared to initial plans.

## Notes

- Companion Paper 1 reference: `Heitfeld:2025flavor` (to be updated with arXiv number when submitted)
- Keep consistent notation with Paper 1 (tau, rho, sigma for moduli)
- Emphasize testability and falsifiability throughout
- Target journals: PRD, JHEP, or PLB
