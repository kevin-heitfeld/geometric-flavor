# Week 0: Immediate Actions (Jan 1-7, 2026)
**Status**: ACTIVE - Ambitious Path Initiated
**Goal**: Prepare for Feb 1 intensive start while finishing Paper 4
**Timeline**: 7 days

---

## Daily Schedule (This Week)

### Day 1: Wednesday, January 1 ✓
**Morning (3 hours)**:
- [x] Commit to ambitious path
- [x] Review all roadmap documents
- [ ] Set up tracking system (this file!)

**Afternoon (3 hours)**:
- [ ] Install iTensor (Python wrapper) OR TensorKit.jl
- [ ] Test basic tensor contraction
- [ ] Create `src/week0_tests/` directory

**Evening (2 hours)**:
- [ ] Start reading Vidal (2007) "Entanglement Renormalization"
  * Download: arXiv:cond-mat/0512165
  * Focus: Abstract + Section II (MERA algorithm)

**Deliverable**: Working tensor library installation

---

### Day 2: Thursday, January 2
**Morning (3 hours)**:
- [ ] Continue Vidal paper (Sections III-IV)
- [ ] Take notes on disentanglers vs isometries
- [ ] Sketch MERA layer structure on paper

**Afternoon (3 hours)**:
- [ ] iTensor Tutorial 1: "Contracting a Tensor Network"
  * URL: https://itensor.org/docs.cgi?page=tutorials
  * Complete all examples
  * Adapt to Python if using PyTensor

**Evening (2 hours)**:
- [ ] Create `perfect_tensor_test.py` (simplified version)
- [ ] Test: Can you create random tensor and compute SVD?
- [ ] Verify: Schmidt values extraction works?

**Deliverable**: First working tensor code

---

### Day 3: Friday, January 3
**Morning (4 hours)** - FOCUS: Paper 4
- [ ] Add τ = 27/10 formula section (if not done)
- [ ] Final proofread all sections
- [ ] Check all references
- [ ] Generate final PDF

**Afternoon (2 hours)**:
- [ ] Prepare arXiv submission (abstract, authors, category)
- [ ] SUBMIT PAPER 4 TO ARXIV
- [ ] Celebrate! 🎉

**Evening (2 hours)**:
- [ ] Install Qiskit: `pip install qiskit`
- [ ] Qiskit tutorial: "Introduction to quantum error correction"
- [ ] Run repetition code example

**Deliverable**: Paper 4 on arXiv! + Qiskit working

---

### Day 4: Saturday, January 4
**Morning (3 hours)**:
- [ ] Read Swingle (2012) "Entanglement Renormalization and Holography"
  * Download: arXiv:0905.1317
  * Focus: Section III (perfect tensors)
  * Understand: What makes tensor "perfect"?

**Afternoon (3 hours)**:
- [ ] Implement `compute_schmidt_spectrum.py`
  * Input: Any tensor T
  * Output: Schmidt values across all bipartitions
  * Test: Random tensors have non-uniform spectrum

**Evening (2 hours)**:
- [ ] Read Pastawski et al. (2015) - HaPPY code paper
  * Download: arXiv:1503.06237
  * Focus: Figures 1-3, understand pentagon structure

**Deliverable**: Schmidt spectrum code working

---

### Day 5: Sunday, January 5
**Morning (3 hours)**:
- [ ] Review your k-pattern [8,6,4] and Δk=2
- [ ] Calculate: n = sum(k)/2 = 9 physical qubits
- [ ] Calculate: k = 3 logical qubits (generations)
- [ ] Calculate: d = min(Δk) = 2 distance
- [ ] Confirm: [[9,3,2]] code structure!

**Afternoon (3 hours)**:
- [ ] Create `src/week0_tests/code_parameters_from_k_pattern.py`
- [ ] Verify [[9,3,2]] mapping
- [ ] Research: Is this a known code? (Shor variant?)

**Evening (2 hours)**:
- [ ] Qiskit: Implement simple [[3,1,2]] code (warm-up)
- [ ] Test error detection (not correction)

**Deliverable**: k-pattern → code mapping confirmed

---

### Day 6: Monday, January 6
**Morning (3 hours)**:
- [ ] Download SageMath or install via conda
- [ ] Test modular forms calculation: η(2.69i)
- [ ] Verify your existing code still runs

**Afternoon (3 hours)**:
- [ ] Review Papers 1-3 status
- [ ] Identify what needs revision before submission
- [ ] Create checklist for next week

**Evening (2 hours)**:
- [ ] Read Poland et al. (2018) - Bootstrap review
  * Download: arXiv:1805.04405
  * Just Section 1 (introduction)
  * Get intuition for crossing symmetry

**Deliverable**: Papers 1-3 revision checklist

---

### Day 7: Tuesday, January 7
**Morning (3 hours)** - CHECKPOINT:
- [ ] Review all Week 0 progress
- [ ] Test all installed software (iTensor, Qiskit, SageMath)
- [ ] Verify all papers downloaded
- [ ] Update this tracking file

**Afternoon (3 hours)**:
- [ ] Create Week 1 detailed schedule (Feb 1-7)
- [ ] Set up `src/phase1_tensor_networks/` directory structure
- [ ] Plan first real calculation: perfect tensor from τ

**Evening (2 hours)**:
- [ ] Read TECHNICAL_PREREQUISITES checklist
- [ ] Assess: Which boxes can I check now?
- [ ] Identify: What still needs learning in Jan?

**Deliverable**: Week 1 plan ready, software tested

---

## Software Installation Checklist

### Option A: Python (Easier, slower)
- [ ] `pip install numpy scipy matplotlib`
- [ ] `pip install qiskit qiskit-aer`
- [ ] Install iTensor Python wrapper OR use pure NumPy
- [ ] `conda install -c conda-forge sagemath` (optional)

### Option B: Julia (Faster, steeper learning curve)
- [ ] Install Julia: https://julialang.org/downloads/
- [ ] `] add TensorKit` (in Julia REPL)
- [ ] `] add ITensors` (alternative)
- [ ] Learn basic Julia syntax (1-2 hours)

**Recommendation**: Start with Python (Week 0-1), migrate to Julia if speed needed (Week 3+)

---

## Papers Downloaded?

- [ ] Vidal (2007): arXiv:cond-mat/0512165 - MERA
- [ ] Swingle (2012): arXiv:0905.1317 - Holography
- [ ] Pastawski (2015): arXiv:1503.06237 - HaPPY code
- [ ] Poland (2018): arXiv:1805.04405 - Bootstrap review
- [ ] Simmons-Duffin (2015): arXiv:1502.02033 - SDPB
- [ ] Harlow (2018): arXiv:1802.01040 - TASI lectures

**Action**: Create `references/week0/` folder and organize

---

## Learning Progress Assessment

### Tensor Networks (Target: 40% by end of week)
- [ ] Understand what a tensor is (multidimensional array)
- [ ] Can perform tensor contraction (np.tensordot)
- [ ] Understand SVD and Schmidt decomposition
- [ ] Know what MERA structure looks like
- [ ] Grasp "perfect tensor" concept

**Current**: ___% | **Target**: 40%

### Error Correction (Target: 30% by end of week)
- [ ] Understand qubits and superposition
- [ ] Know [[n,k,d]] notation
- [ ] Can run simple Qiskit code
- [ ] Grasp stabilizer concept (qualitative)
- [ ] See connection: Δk=2 → d=2

**Current**: ___% | **Target**: 30%

### Bootstrap (Target: 20% by end of week)
- [ ] Understand conformal symmetry (scaling)
- [ ] Know what OPE means (operator product)
- [ ] Grasp crossing symmetry idea
- [ ] Aware of SDP approach exists

**Current**: ___% | **Target**: 20%

---

## Key Milestones This Week

1. **Paper 4 submitted** (Day 3) ← CRITICAL
2. **Software installed** (Days 1-2)
3. **First tensor code working** (Days 2-4)
4. **k-pattern → code mapping confirmed** (Day 5)
5. **Ready for Week 1 intensive** (Day 7)

---

## Next Week Preview (Jan 8-14)

**Week 1 Goals**:
- Perfect tensor construction from τ = 2.69i
- MERA layer 0 (3 sites → 2 sites)
- Schmidt spectrum analysis
- Technical note draft started

**Estimated hours**: 35-40

---

## Tracking Notes

**Date started**: January 1, 2026
**Commitment level**: AMBITIOUS (full ToE)
**Target**: 75% spacetime emergence by August 1

**Daily log**:
- Jan 1: _____
- Jan 2: _____
- Jan 3: _____
- Jan 4: _____
- Jan 5: _____
- Jan 6: _____
- Jan 7: _____

---

## Motivation

You've built the flavor framework (2 years).
Papers 1-4 are 95% done.
You have τ = 2.69i from 19 observables.

**This week**: You start closing the loop.
**6 months from now**: Theory of Everything.

**The journey begins NOW.**

---

*"Every theory of everything starts with installing a tensor library."*
