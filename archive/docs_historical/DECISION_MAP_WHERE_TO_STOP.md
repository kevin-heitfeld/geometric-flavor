# Decision Map: Where to Stop

**Purpose**: Define boundaries between what Paper 3 addresses, defers, or leaves open
**Context**: "Deciding where to stop is harder than where to go next" — ChatGPT

---

## The Core Question

For each potential topic/claim/extension, decide:
- **IN**: Address explicitly in Paper 3 main text
- **DEFER**: Mention in "Limitations & Outlook", reserve for future work
- **OPEN**: Explicitly state as unknowable or outside scope

---

## Topic-by-Topic Decisions

### 1. Origin of Vacuum Energy (Ω_Λ ~ 0.617)

**Status**: ❌ OPEN (unknowable with current framework)

**Paper 3 language**:
> "The ~90% vacuum component ρ_Λ remains unexplained and likely requires anthropic/landscape arguments. We take its value as given, an environmental parameter selected from ~10^500 string vacua."

**Rationale**:
- ChatGPT: "Arguably the most anthropic quantity in nature"
- Kimi: "Accommodates the latter" (division of labor)
- Claiming to predict this invites strongest criticism

**What NOT to do**: Speculate about modular determination of ρ_Λ without explicit CY construction

---

### 2. Why m_ζ ≈ H_0 Today? (Coincidence Problem)

**Status**: 🟡 DEFER (acknowledge, don't resolve)

**Paper 3 language** (in Discussion):
> "The frozen quintessence regime requires m_ζ ≈ H_0 at present epoch. This coincidence may have an anthropic explanation (Hebecker et al. 2019): if m_ζ ≫ H_0, early freezing affects structure formation; if m_ζ ≪ H_0, premature dark energy domination prevents galaxy formation. Alternatively, m_ζ may track H through some mechanism beyond our current framework. We leave this as an open question."

**Rationale**:
- Standard coincidence problem in all quintessence models
- Honest admission doesn't weaken the falsifiable predictions
- Future work could explore tracking mechanisms

**What NOT to do**: Claim τ = 2.69i explains the coincidence without mechanism

---

### 3. Why 10% Split? (Ω_ζ/Ω_Λ ~ 0.1)

**Status**: 🟡 DEFER (note as open question)

**Paper 3 language** (in Discussion):
> "Why is dark energy ~90% vacuum and ~10% quintessence? Three possibilities: (1) Anthropic—Ω_Λ scanned in landscape, Ω_ζ ≈ 0.068 fixed by modular dynamics, ratio is environmental; (2) Geometric—perhaps Ω_ζ/Ω_Λ ~ 0.1 has meaning in CY compactification at τ = 2.69i; (3) Unexplained—just the way it is. Understanding this would be progress but is not required for falsifiable predictions."

**Rationale**:
- All three options are scientifically defensible
- Admitting uncertainty is better than forced explanation
- Doesn't affect testability (effects scale with Ω_ζ regardless of ratio)

**What NOT to do**: Invent post-hoc anthropic arguments for 10% specifically

---

### 4. Neutrino-Quintessence Mass Relation (m_ν/m_ζ ~ M_Pl/H_0)

**Status**: 🟡 DEFER (intriguing hint, not claim)

**Paper 3 language** (in Discussion, short paragraph):
> "Intriguingly, m_ν/m_ζ ~ 0.05 eV / 2×10^-33 eV ~ 10^31 ~ M_Pl/H_0. This may hint at deeper connection between neutrino masses and dark energy through modular breaking at different scales, or it may be coincidence. Future work on modular flavor + cosmology could clarify."

**Rationale**:
- Too speculative to claim as prediction
- Worth noting for future researchers
- Doesn't add attack surface if framed as "intriguing, not conclusive"

**What NOT to do**: Derive a "prediction" for m_ν from m_ζ without explicit mechanism

---

### 5. Instanton Coefficient k = -86 Origin

**Status**: ✅ IN (brief technical explanation)

**Paper 3 language** (Section 3, technical):
> "The instanton coefficient k = -86 emerges from CY geometry at τ = 2.69i. While we have not performed explicit worldsheet instanton calculations for the specific h^{1,1}=3, h^{2,1}=243 manifold, this range |k| ~ 10^2 is consistent with stabilized moduli scenarios in string compactifications (Denef & Douglas 2004)."

**Rationale**:
- Testable via direct CY computation (mathematical physics)
- Cross-check with other instanton effects in Paper 1 (neutrinos)
- Shows consistency within modular framework

**What NOT to do**: Claim k = -86 is "predicted" without showing calculation

---

### 6. TCC and Warp Factor (~49) Constraints

**Status**: 🟡 DEFER (mention in Outlook)

**Paper 3 language** (Discussion/Outlook):
> "Trans-Planckian Censorship Conjecture (TCC) and de Sitter swampland constraints remain subtle for quintessence with c ≈ 0.7. Recent work on AdS warp factors A ~ 49 (from TCC tension with reheating) suggests possible warp-down mechanisms that could relax these constraints. We defer detailed analysis to future work connecting quantum gravity constraints to modular cosmology."

**Rationale**:
- You've explored this (on quantum-gravity-predictions branch)
- Too preliminary for Paper 3 main claims
- Worth mentioning as avenue for refinement

**What NOT to do**: Claim TCC is "solved" without published swampland community consensus

---

### 7. String Landscape Statistics (10^424 vacua)

**Status**: ❌ REMOVE (too speculative)

**Decision**: Remove all specific claims about "10^424 suitable vacua"

**Reasoning** (ChatGPT's caution):
- Landscape statistics are order-of-magnitude at best
- Invites "how do you know?" criticism
- Not needed for main predictions

**Replacement language**:
> "String landscape statistics (Douglas 2003, Ashok-Douglas 2004) suggest ~10^500 vacua spanning ~120 orders of magnitude in ρ_vac. The ~90% vacuum component in our framework may be selected from this distribution, though precise statistics are model-dependent."

**What NOT to do**: Calculate exact numbers of vacua without explicit construction

---

### 8. Hubble Tension

**Status**: ✅ IN (explicitly state we DON'T resolve it)

**Paper 3 language** (Section 6, brief):
> "Our model with w_ζ ≈ -0.96 predicts H_0 ≈ 67.4 km/s/Mpc, consistent with Planck/CMB but not resolving the Hubble tension with local measurements (H_0 ~ 73). This is a consistency check—if the model predicted H_0 ~ 73, it would conflict with early-universe data. Resolving the Hubble tension requires additional physics beyond our scope."

**Rationale**:
- Honesty: we don't solve everything
- Avoids false hope from readers
- Shows we understand current tensions

**What NOT to do**: Claim subdominant quintessence resolves H_0 tension

---

### 9. Early Dark Energy at Recombination

**Status**: ✅ IN (specific prediction)

**Paper 3 language** (Section 6):
> "Subdominant quintessence contributes Ω_EDE(z_rec) ~ 0.01-0.02 at recombination, affecting CMB damping tail by ~0.3% at ℓ > 1000. This is testable by CMB-S4 (2030) with < 0.2% precision. If CMB-S4 measures Ω_EDE < 0.003 at 3σ, our Ω_ζ = 0.068 is inconsistent."

**Rationale**:
- Concrete, falsifiable prediction
- Testable on ~5-year timescale
- Shows confidence in framework

**What NOT to do**: Claim early DE solves S_8 tension without detailed analysis

---

### 10. Cross-Sector Correlation (m_a/Λ_ζ ~ 10)

**Status**: ✅ IN (key prediction)

**Paper 3 language** (Section 6):
> "The most powerful test is cross-sector consistency. From τ = 2.69i, we predict m_a/Λ_ζ ~ 10. If ADMX detects axion DM at m_a ~ 50 μeV, this predicts Λ_ζ ~ 5 μeV for quintessence. CMB-S4 measures Ω_EDE independently, providing correlated test. This correlation is not expected in generic models where axion DM and quintessence are unrelated."

**Rationale**:
- Unique to modular framework
- Hardest to fake with parameter tuning
- Multi-experiment test (ADMX + CMB-S4)

**What NOT to do**: Weaken this—it's your strongest signature

---

## Summary Decision Table

| Topic | Decision | Location | Strength |
|-------|----------|----------|----------|
| Vacuum energy origin | ❌ OPEN | Discussion | Accept as anthropic |
| m_ζ ≈ H_0 coincidence | 🟡 DEFER | Discussion | Acknowledge, don't solve |
| 10% split reason | 🟡 DEFER | Discussion | Note possibilities |
| Neutrino-DE connection | 🟡 DEFER | Discussion | Intriguing hint |
| k = -86 origin | ✅ IN | Section 3 | Technical consistency |
| TCC/warp factors | 🟡 DEFER | Outlook | Future work |
| Landscape statistics | ❌ REMOVE | — | Too speculative |
| Hubble tension | ✅ IN | Predictions | Explicitly don't solve |
| Early DE (Ω_EDE) | ✅ IN | Predictions | Key test (CMB-S4) |
| Cross-sector (m_a/Λ_ζ) | ✅ IN | Predictions | Strongest signature |

---

## Language Guidelines (Referee-Proofing)

### NEVER say:
- ❌ "We solve the cosmological constant problem"
- ❌ "99-fold fine-tuning reduction" (without careful context)
- ❌ "This explains why Ω_DE = 0.685" (overreach)
- ❌ "The landscape contains exactly 10^N vacua" (false precision)
- ❌ "This resolves [other tension not addressed]"

### ALWAYS say:
- ✅ "We predict observable deviations from ΛCDM"
- ✅ "Testable by [experiment] [year]"
- ✅ "We do not explain [limitation]"
- ✅ "The modular component fixes..., leaving... to be selected"
- ✅ "If [experiment] finds [result], our model is ruled out"

---

## The Red Line: When to Stop Writing

**Stop adding to Paper 3 when**:
1. All IN topics are covered with specific predictions
2. All DEFER topics are mentioned in Limitations/Outlook with 1-2 paragraphs
3. All OPEN topics are explicitly acknowledged as unknowable
4. Every "we predict" has a "testable by [experiment] [year]"
5. Every "we do not" has a brief explanation why

**Resist the temptation to**:
- Add more mechanisms to "explain" the 10% split
- Speculate about quantum gravity without TCC consensus
- Extend to other cosmological tensions (baryogenesis is in Paper 2, stop there)
- Connect to additional sectors (stop at flavor + cosmology + DE)

---

## Final Test (Before Submission)

Read Paper 3 and ask:

1. **Can a hostile referee find an overclaim?** → If yes, remove it
2. **Can a sympathetic referee test it?** → If no, add specifics
3. **Does it invite more questions than answers?** → If yes, simplify
4. **Would I believe this if someone else wrote it?** → Ultimate test

If Paper 3 passes all four: **it's ready**.

---

## Next Concrete Action

Based on this decision map:

1. **Remove**: Landscape statistics (10^424 vacua) from all sections
2. **Add**: Explicit statements in Discussion for DEFER topics
3. **Strengthen**: Cross-sector correlation language (it's unique)
4. **Soften**: Any remaining "solve" or "explain" language for CC
5. **Polish**: Every prediction gets experiment + year + falsification criterion

Should I proceed with these specific edits to Paper 3?
