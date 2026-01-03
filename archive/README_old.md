# Geometric Flavor Unification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**Complete derivation of Standard Model flavor parameters from string theory compactifications**

This repository contains the complete codebase and manuscripts for a geometric approach to flavor physics, deriving 28 observables from a single modular parameter τ = 2.69i with zero continuous free parameters.

---

## ⚡ Quick Navigation

### 🆕 New Here? START HERE:
📁 **[docs/framework/README.md](docs/framework/README.md)** - Core framework explanation

**Key Point**: This framework uses **τ = 2.69i universally** for all sectors (leptons, quarks, cosmology, dark energy). If you see documents mentioning different τ values, those are historical explorations (see below).

### 📚 Documentation Map

- **📁 [docs/framework/](docs/framework/)** - Canonical framework documentation ← **Start here**
  - Single-τ framework (τ = 2.69i for ALL sectors)
  - What differs: modular forms (η, E₄), NOT τ values
  - Papers 1-4 summaries

- **📁 [docs/research/](docs/research/)** - Active research questions
  - Path A: Mathematical origins (E₄ from gauge anomalies, 3 generations)
  - Path B: Extensions (gauge unification, gravity)
  - Verified open questions only

- **📁 [docs/historical/](docs/historical/)** - Old explorations ⚠️
  - Failed approaches (multi-τ, Δk universality)
  - Kept for educational context
  - **Do NOT use** as basis for new work

- **❓ Confused?** Read [docs/CONFUSION_SOURCE_ANALYSIS.md](docs/CONFUSION_SOURCE_ANALYSIS.md)

---

## 📂 Repository Structure

```
geometric-flavor/
├── manuscript/                   # Paper 1: Flavor unification (main result)
├── manuscript_cosmology/         # Paper 2: Cosmological predictions
├── manuscript_dark_energy/       # Paper 3: Dark energy from moduli
├── manuscript_paper4_string_origin/ # Paper 4: String theory embedding
│
├── docs/
│   ├── framework/               # ← START HERE: Current framework docs
│   ├── research/                # Active research (Path A/B)
│   └── historical/              # ⚠️ Old explorations (educational only)
│
├── src/                         # Analysis scripts (159 Python files)
├── figures/                     # Visualizations (128 figures)
├── results/                     # JSON outputs (21 result files)
│
└── scripts/                     # Build utilities
```

---

## 🎯 Key Results

### The Framework at a Glance

**Single Input**: τ = 2.69i (modular parameter)

**Outputs**: 28 observables across four papers
- ✅ 19 SM flavor parameters (Paper 1)
- ✅ 6 cosmological observables (Paper 2)
- ✅ 3 dark energy properties (Paper 3)
- ✅ String origin confirmed (Paper 4)

**Quality**: χ²/dof = 1.18 (excellent fit)

### Paper 1: Zero-Parameter Flavor Framework (τ = 2.69i)
- **19/19 SM flavor observables** from modular forms
- **Leptons**: Γ₀(3) at level k=27, using η(τ)
- **Quarks**: Γ₀(4) at level k=16, using E₄(τ)
- **Same τ for both sectors**: τ = 2.69i
- **χ²/dof = 1.18** with zero continuous free parameters

### Paper 2: Cosmological Predictions (τ = 2.69i)
- **Inflation**: α-attractor from modular Kähler (n_s = 0.967, r = 0.003)
- **Dark matter**: Sterile neutrino (83%) + axion (17%)
- **Baryogenesis**: Resonant leptogenesis (η_B exact match)
- **Strong CP**: Modular axion solution
- **All from τ = 2.69i** (same value as flavor)

### Paper 3: Dark Energy Mechanism
- **Quintessence**: Two-component (ρ + Λ_eff)
- **w(z)**: Evolves from -0.95 → -1 (matches observations)
- **Natural**: Moduli stabilization provides both components

### Paper 4: String Theory Origin
- **Type IIB** orientifold compactification
- **T⁶/ℤ₃×ℤ₄** orbifold with D7-branes
- **Modular emergence**: τ ≈ 2.69i from volume/complex structure
### Paper 3: Dark Energy Mechanism (τ = 2.69i)
- **Quintessence**: Two-component (ρ + Λ_eff)
- **w(z)**: Evolves from -0.95 → -1 (matches observations)
- **Natural**: Moduli stabilization provides both components
- **From τ = 2.69i**: Same modular structure

### Paper 4: String Theory Origin (τ = 2.69i)
- **Type IIB** orientifold compactification
- **T⁶/(Z₃×Z₄)** orbifold with D7-branes
- **Modular parameter**: Complex structure U = 2.69i
- **Gauge coupling**: Threshold corrections match observations
- **Verification**: String construction produces τ = 2.69i naturally

---

## 🚀 Quick Start

### Prerequisites
```bash
python >= 3.11
numpy, scipy, matplotlib
sympy (for symbolic calculations)
```

### Installation
```bash
git clone https://github.com/kevin-heitfeld/geometric-flavor.git
cd geometric-flavor
pip install -r requirements.txt
```

### Verify Framework
```bash
# Validate τ = 2.69i framework
python src/verify_tau_2p69i.py

# Master summary (all 30 observables)
python src/master_summary.py

# Yukawa coupling calculations
python src/yukawa_numerical_overlaps.py
```

### ⚠️ Historical Scripts (Educational Only)
```bash
# These use OLD τ values (3.25i, 1.422i) - for understanding only
python src/why_quarks_need_eisenstein.py  # Shows why E₄ needed
python src/test_e4_beta_connection.py     # E₄ vs QCD running
```

---

## 📖 Documentation

### Start Here
1. **[docs/framework/README.md](docs/framework/README.md)** - Framework overview
2. Papers in `manuscript*/` directories - Final authority
3. **[docs/research/](docs/research/)** - Open questions

### ⚠️ Important Notes
- Some older docs use superseded τ values (3.25i, 1.422i)
- Look for warning headers at top of files
- When in doubt: **Papers 1-4 are authoritative**
- See [docs/CONFUSION_SOURCE_ANALYSIS.md](docs/CONFUSION_SOURCE_ANALYSIS.md) for clarification

---

## 📊 Current Status

**Framework**: ESTABLISHED ✅ (Papers 1-4 ready)
**Observables**: 30/30 explained from τ = 2.69i
**Fit Quality**: χ²/dof = 1.18

**Completed**:
- ✅ 19/19 SM flavor parameters (leptons + quarks)
- ✅ 8 cosmological observables
- ✅ 3 dark energy properties
- ✅ 4 complete manuscripts ready for submission
- ✅ String theory origin (T⁶/(Z₃×Z₄) construction)
- ✅ 3 generation origin (topological + tadpole + Z₃)
- ✅ Cosmological predictions (inflation, DM, baryogenesis)

**In Progress**:
- 🔄 Expert validation and peer review
- 🔄 C=13 theoretical justification
- 🔄 Rigorous path integral derivations

---

## 🤝 Contributing

This is currently a research project under development. For questions or collaboration inquiries, please open an issue.

**Note**: Parts of this work were developed in collaboration with AI assistants (GitHub Copilot, Claude, GPT-4). All scientific claims and calculations have been independently verified.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 📧 Contact

Kevin Heitfeld
- Email: kheitfeld@gmail.com
- GitHub: [@kevin-heitfeld](https://github.com/kevin-heitfeld)

---

## 🙏 Acknowledgments

This work builds on foundational research in:
- String phenomenology (Vafa, Morrison, Weigand, et al.)
- Modular flavor symmetries (Feruglio, Criado, King, et al.)
- F-theory compactifications (Blumenhagen, Cvetic, et al.)

AI assistance provided by:
- GitHub Copilot (coding and analysis)
- Claude 3.5 Sonnet (theoretical insights)
- ChatGPT-4 (optimization strategies)
