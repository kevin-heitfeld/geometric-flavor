# Week 0 Reference Papers

Download these papers for Week 0 preparation (Jan 1-7, 2026).

## MERA & Tensor Networks

1. **Vidal (2007)** - Original MERA paper
   - Title: "Entanglement Renormalization"
   - arXiv: [cond-mat/0512165](https://arxiv.org/abs/cond-mat/0512165)
   - Priority: **CRITICAL** (start reading Day 2)
   - Topics: MERA construction, disentanglers, isometries

2. **Vidal (2008)** - MERA review
   - Title: "Class of Quantum Many-Body States That Can Be Efficiently Simulated"
   - arXiv: [quant-ph/0610099](https://arxiv.org/abs/quant-ph/0610099)
   - Priority: High
   - Topics: Efficient simulation, area law

## AdS/CFT Connection

3. **Swingle (2012)** - Holography from tensor networks
   - Title: "Entanglement Renormalization and Holography"
   - arXiv: [0905.1317](https://arxiv.org/abs/0905.1317)
   - Priority: **CRITICAL** (read Days 4-5)
   - Topics: MERA = hyperbolic space, AdS geometry

4. **Pastawski et al. (2015)** - HaPPY code
   - Title: "Holographic quantum error-correcting codes: Toy models for the bulk/boundary correspondence"
   - arXiv: [1503.06237](https://arxiv.org/abs/1503.06237)
   - Priority: **CRITICAL** (read Days 4-5)
   - Topics: Perfect tensors, quantum error correction, holography

## Quantum Error Correction

5. **Almheiri et al. (2015)** - Bulk locality from error correction
   - Title: "Bulk Locality and Quantum Error Correction in AdS/CFT"
   - arXiv: [1411.7041](https://arxiv.org/abs/1411.7041)
   - Priority: High
   - Topics: Code subspace, operator algebra

6. **Nielsen & Chuang** - Chapter 10 (Quantum Error Correction)
   - Textbook: "Quantum Computation and Quantum Information"
   - Priority: Medium (review during Week 5)
   - Topics: Stabilizer formalism, CSS codes

## Conformal Bootstrap (Optional Week 0)

7. **Poland et al. (2018)** - Bootstrap review
   - Title: "The Conformal Bootstrap"
   - arXiv: [1805.04405](https://arxiv.org/abs/1805.04405)
   - Priority: Medium (overview only Week 0, deep dive Week 9)
   - Topics: Crossing symmetry, numerical methods

8. **Simmons-Duffin (2015)** - SDPB paper
   - Title: "A Semidefinite Program Solver for the Conformal Bootstrap"
   - arXiv: [1502.02033](https://arxiv.org/abs/1502.02033)
   - Priority: Low (Week 9)
   - Topics: Numerical implementation

## HKLL Reconstruction (Optional Week 0)

9. **Hamilton et al. (2006)** - Original HKLL paper
   - Title: "Local bulk operators in AdS/CFT: A boundary view of horizons and locality"
   - arXiv: [hep-th/0506118](https://arxiv.org/abs/hep-th/0506118)
   - Priority: Low (Week 19)
   - Topics: Smearing functions, bulk reconstruction

10. **Harlow (2018)** - HKLL review
    - Title: "TASI Lectures on the Emergence of Bulk Physics in AdS/CFT"
    - arXiv: [1802.01040](https://arxiv.org/abs/1802.01040)
    - Priority: Low (Week 19)
    - Topics: Code subspace, quantum error correction perspective

---

## Week 0 Reading Plan

**Day 2 (Jan 2)**: Vidal (2007) - Sections I-III
**Day 4-5 (Jan 4-5)**: Swingle (2012) + Pastawski (2015)
**Day 6 (Jan 6)**: Poland (2018) - Abstract + Section 2 (overview)

**Download Command** (PowerShell):
```powershell
# Navigate to references/week0/
cd references/week0/

# Download using wget or Invoke-WebRequest
# Example for Vidal:
Invoke-WebRequest -Uri "https://arxiv.org/pdf/cond-mat/0512165.pdf" -OutFile "vidal_2007_mera.pdf"
```

**Alternative**: Use browser to download from arXiv links above.

---

## Notes

- Papers 1-4 are **essential** for Phase 1 (Weeks 1-8)
- Papers 5-6 are useful for understanding quantum error correction framework
- Papers 7-10 are for later phases (can skip Week 0)

**Priority for Week 0**: Read Vidal, Swingle, Pastawski thoroughly. Skim Poland for intuition.
