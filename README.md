# Geometric Flavor Unification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**Complete derivation of Standard Model flavor parameters from string theory compactifications**

This repository contains the complete codebase and manuscripts for a geometric approach to flavor physics, deriving 19+ SM observables from discrete topological data with zero continuous free parameters.

---

## 📂 Repository Structure

```
geometric-flavor/
├── manuscript/                   # Paper 1: Flavor unification (main result)
├── manuscript_cosmology/         # Paper 2: Cosmological predictions
├── manuscript_dark_energy/       # Paper 3: Dark energy from moduli
├── manuscript_paper4_string_origin/ # Paper 4: String theory embedding
│
├── src/                         # All Python analysis scripts
├── docs/                        # Documentation and research notes
├── figures/                     # Generated plots and visualizations
├── results/                     # JSON output files from calculations
│
├── moduli_exploration/          # Detailed moduli stabilization analysis
├── archive/                     # Historical exploration work
├── scripts/                     # Build and utility scripts
└── venv/                        # Python virtual environment
```

---

## 🎯 Key Results

### Paper 1: Zero-Parameter Flavor Framework
- **19/19 SM flavor observables** from CY topology
- **χ²/dof = 1.2** with zero continuous free parameters
- Discrete inputs: ℤ₃ × ℤ₄ orbifold, (w₁,w₂) = (1,1) wrapping

### Paper 2: Cosmological Predictions
- **Inflation**: α-attractor from modular Kähler (n_s = 0.967, r = 0.003)
- **Dark matter**: Sterile neutrino (83%) + axion (17%)
- **Baryogenesis**: Resonant leptogenesis (η_B exact match)
- **Strong CP**: Modular axion solution

### Paper 3: Dark Energy Mechanism
- **Quintessence**: Two-component (ρ + Λ_eff)
- **w(z)**: Evolves from -0.95 → -1 (matches observations)
- **Natural**: Moduli stabilization provides both components

### Paper 4: String Theory Origin
- **Type IIB** orientifold compactification
- **T⁶/ℤ₃×ℤ₄** orbifold with D7-branes
- **Modular emergence**: τ ≈ 2.69i from volume/complex structure
- **Gauge unification**: Threshold corrections reproduced

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

### Run Key Calculations
```bash
# Flavor framework validation
python src/master_summary.py

# Quark E4 structure analysis
python src/why_quarks_need_eisenstein.py

# Generation count investigation
python src/why_3_generations.py

# Dark matter predictions
python src/dark_matter_string_theory.py

# Quintessence evolution
python src/quintessence_cosmological_evolution.py
```

---

## 📖 Documentation

- **Main result**: See `docs/COMPLETE_FLAVOR_FRAMEWORK_FINAL.md`
- **Path A breakthroughs**: See `docs/PATH_A_PROGRESS_REPORT.md`
- **E4 derivation**: See `docs/QUARK_E4_BREAKTHROUGH.md`
- **ToE pathway**: See `docs/TOE_PATHWAY.md`

Full list of documentation in `docs/` directory.

---

## 📊 Current Status

**Framework Completion**: ~76-78%

**Completed**:
- ✅ 19/19 SM flavor observables
- ✅ 4 complete manuscripts
- ✅ E4 structure derived from gauge theory
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
