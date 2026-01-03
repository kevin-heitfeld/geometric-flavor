# Day 2 Reading Guide: Vidal (2007) MERA Paper

**File**: `references/week0/vidal_2007_mera.pdf`
**Time**: 3-4 hours
**Goal**: Understand MERA construction algorithm

---

## What to Focus On

### Part 1: Motivation (30 min)
- **Abstract**: Why MERA? (Entanglement renormalization)
- **Introduction**: Area law, why tensor networks work
- **Key insight**: Short-range entanglement can be removed at each scale

### Part 2: MERA Construction (2 hours) ⭐ CRITICAL
- **Section II**: MERA algorithm
  * **Disentanglers** (u): Remove short-range entanglement
  * **Isometries** (w): Coarse-grain (reduce sites)
  * **Causal cone**: Which tensors affect a given site?

**Questions to answer**:
1. What is the difference between disentanglers and isometries?
2. How many parameters in each tensor?
3. How do you connect layers? (draw the network!)
4. What is the "branching ratio" (usually 3:1)?

### Part 3: Properties (1 hour)
- **Section III**: MERA properties
  * Area law satisfied
  * Computational cost (polynomial)
  * Relation to critical systems

---

## Key Concepts for Your Work

### 1. Perfect Tensors (you have χ=6 version)
- Vidal uses generic tensors, optimized via variational methods
- You have: **Constructed from τ=2.69i** (phenomenology → geometry!)
- Difference: Your tensor has physical origin (modular symmetry)

### 2. Network Structure
```
Layer 0: ●●●●●●●●  (2^L sites, L=3 → 8 sites for χ=6)
          \u/\u/\u/\u/   (disentanglers)
           ●●●●●●●●
            \w/w/w/w     (isometries, 3:1)
Layer 1:    ●●●●       (fewer sites)
             \u/\u/
              ●●●●
               \w/w
Layer 2:        ●●     (coarse)
                 \u
                  ●
Layer 3:          ●    (single site)
```

### 3. Holography Connection (Swingle 2012, read Day 4)
- MERA layers = radial direction in AdS
- Coarse-graining = moving into bulk
- Your τ=2.69i → emergent metric g_μν

---

## What You'll Implement (Week 1)

**Week 1 Goal**: Build 5-layer MERA from your perfect tensor

```python
# Pseudocode
def build_mera_layer(tensor_in, chi):
    """
    Input: tensor_in (rank-6, shape (chi,chi,chi,chi,chi,chi))
    Output:
      - disentanglers (u): Remove entanglement
      - isometries (w): Coarse-grain 3→1
      - tensor_out: Next layer tensor (smaller)
    """
    # Step 1: Apply disentanglers (your perfect tensor has this built in!)
    # Step 2: Contract with isometries (need to construct)
    # Step 3: Get reduced tensor for next layer
    return disentanglers, isometries, tensor_out

# Loop over layers
tensors = [perfect_tensor]  # Start with your χ=6 tensor
for layer in range(5):
    u, w, T_next = build_mera_layer(tensors[-1], chi=6)
    tensors.append(T_next)
```

---

## Questions to Think About

While reading Vidal, keep asking:

1. **How does my perfect tensor fit into this framework?**
   - Your tensor is rank-6 (6 indices, χ=6)
   - Vidal's tensors are generic (optimized numerically)
   - You have: Physical construction from τ → Can you skip optimization?

2. **What is the emergent metric?**
   - MERA defines a network geometry
   - Swingle (2012): This geometry IS hyperbolic space (AdS!)
   - Week 3 goal: Extract g_μν from entanglement structure

3. **How do I verify Einstein equations?**
   - Once you have g_μν, compute Ricci tensor R_μν
   - Check: R_μν = constant × g_μν (cosmological constant)
   - Week 4 goal: Verify this within ~30% (stringy corrections!)

---

## Day 2 Deliverables

By end of tomorrow:
- [ ] Read Vidal Abstract + Intro + Section II
- [ ] Draw MERA network structure (by hand or digitally)
- [ ] Understand: disentanglers vs isometries
- [ ] Create notes: `references/week0/vidal_2007_notes.md`
- [ ] Decision: iTensor or pure NumPy? (χ=6 works in NumPy!)

---

## Bonus: iTensor vs NumPy

**iTensor** (C++ library with Python bindings):
- Pros: Optimized for tensor networks, auto-contraction
- Cons: Installation, learning curve, may not support χ=6 directly

**NumPy** (you're already using):
- Pros: You know it, χ=6 proven viable, full control
- Cons: Manual contraction indices, slower for large χ

**Recommendation**: Start with NumPy for Week 0-1. If χ > 20 needed later, consider iTensor.

---

## Summary

**Tomorrow's Focus**: Understand MERA algorithm deeply (disentanglers + isometries). By end of Day 2, you should be able to sketch the network structure and explain how your perfect tensor (χ=6, c=8.92, τ=2.69i) will be used as the starting point for Week 1 construction.

**Week 1 Goal** (starting Feb 1): Build full 5-layer MERA network from your perfect tensor, extract emergent AdS geometry, verify spacetime emergence is working!

Good luck! 🚀
