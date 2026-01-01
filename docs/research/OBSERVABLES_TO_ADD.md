# Comprehensive Observable List for unified_predictions.py

## Standard Model Parameters: Complete List

### Currently Predicted (11 observables):
1. ✅ AdS₃ geometry
2-7. ✅ 6 mass ratios (m_μ/m_e, m_τ/m_e, m_c/m_u, m_t/m_u, m_s/m_d, m_b/m_d)
8-10. ✅ 3 CKM angles (θ₁₂, θ₂₃, θ₁₃)
11. ✅ α₂ (weak coupling)

### TO ADD (Priority 1 - Critical):

#### Absolute Mass Scales (3 observables):
12. m_e = 0.511 MeV (electron mass)
13. m_u = 2.16 MeV (up quark mass)
14. m_d = 4.67 MeV (down quark mass)

**Implementation**:
- Need Higgs VEV: v = 246 GeV
- Need Yukawa normalization: Y₀ (fitted initially)
- Formula: m_i = Y₀ × v × |η(τ)|^(k_i/2) × exp(-A_i Im[τ])

#### Neutrino Sector (5 observables):
15. Δm²₂₁ = 7.5 × 10⁻⁵ eV² (solar mass splitting)
16. Δm²₃₁ = 2.5 × 10⁻³ eV² (atmospheric mass splitting)
17. θ₁₂^PMNS ≈ 34° (solar angle)
18. θ₂₃^PMNS ≈ 42° (atmospheric angle)
19. θ₁₃^PMNS ≈ 8.5° (reactor angle)

**Implementation**:
- Use seesaw: m_ν = m_D M_R⁻¹ m_D^T
- m_D from k_PMNS = [5,3,1]
- M_R = Majorana scale (fitted initially)
- Already have utils/pmns_seesaw.py!

#### CP Violation (2 observables):
20. δ_CP^CKM ≈ 70° (CKM CP phase)
21. J_CP ≈ 3 × 10⁻⁵ (Jarlskog invariant)

**Implementation**:
- Need Re[τ] ≠ 0 (complex moduli)
- Or instanton corrections
- Off-diagonal Yukawa elements
- Already have utils/instanton_corrections.py partially!

#### Complete Gauge Sector (2 observables):
22. α₁ ≈ 0.0102 (U(1) hypercharge at M_Z)
23. α₃ ≈ 0.1184 (SU(3) QCD at M_Z)

**Implementation**:
- Gauge kinetic function from string theory
- RG running from string scale
- Already have α₂, add α₁ and α₃

### TO ADD (Priority 2 - Important):

#### Higgs Sector (2 observables):
24. v = 246 GeV (Higgs VEV)
25. m_h = 125 GeV (Higgs mass)

**Implementation**:
- Electroweak symmetry breaking
- Radiative corrections (SUSY?)
- Moduli stabilization

### Total Observable Count:
- Current: 11
- After Priority 1 additions: 23
- After Priority 2: 25
- Standard Model total: ~25-30

This gives us 92-100% coverage of SM observables!
