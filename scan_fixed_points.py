"""
SCAN MODULAR SUBGROUP FIXED POINTS

Check if τ ≈ 2.7i is a fixed point of any Γ(N) subgroup for N = 2-10

The modular group SL(2,ℤ) acts on the upper half-plane by:
    τ → (aτ + b)/(cτ + d)  where ad - bc = 1

Principal congruence subgroups:
    Γ(N) = {(a,b,c,d) ∈ SL(2,ℤ) : a,d ≡ 1 (mod N), b,c ≡ 0 (mod N)}

Fixed points satisfy: τ = (aτ + b)/(cτ + d)
    → cτ² + (d-a)τ - b = 0
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# Our observed value from fits
TAU_OBSERVED = 0.0 + 2.7j
TAU_THEORY14 = 0.0 + 2.69j

print("="*70)
print("MODULAR FIXED POINT SCANNER")
print("="*70)
print(f"\nSearching for fixed points of Γ(N) subgroups")
print(f"Target: τ ≈ {TAU_OBSERVED} (from RG fits)")
print(f"        τ ≈ {TAU_THEORY14} (from Theory #14)")
print("="*70)

def generate_SL2Z_elements(max_c=20, max_d=20):
    """
    Generate elements of SL(2,ℤ) with bounded entries

    For (a,b,c,d) with ad - bc = 1:
    Given c, d, we can solve for a, b
    """
    elements = []

    # Special case: c = 0 → d = ±1
    for d in [1, -1]:
        for b in range(-20, 21):
            a = d  # Since ad - bc = 1 and c=0
            elements.append((a, b, 0, d))

    # General case: c ≠ 0
    for c in range(1, max_c + 1):
        for d in range(-max_d, max_d + 1):
            # Need ad - bc = 1
            # For each b, solve for a: a = (1 + bc)/d
            for b in range(-20, 21):
                if d == 0:
                    continue
                numerator = 1 + b * c
                if numerator % d == 0:
                    a = numerator // d
                    # Verify
                    if a * d - b * c == 1:
                        elements.append((a, b, c, d))

    # Also negative c
    for c in range(1, max_c + 1):
        for d in range(-max_d, max_d + 1):
            c_neg = -c
            for b in range(-20, 21):
                if d == 0:
                    continue
                numerator = 1 + b * c_neg
                if numerator % d == 0:
                    a = numerator // d
                    if a * d - b * c_neg == 1:
                        elements.append((a, b, c_neg, d))

    return elements

def is_in_gamma_N(a, b, c, d, N):
    """
    Check if (a,b,c,d) is in Γ(N)

    Γ(N): a ≡ d ≡ 1 (mod N), b ≡ c ≡ 0 (mod N)
    """
    return (a % N == 1) and (d % N == 1) and (b % N == 0) and (c % N == 0)

def find_fixed_point(a, b, c, d):
    """
    Find fixed point of transformation τ → (aτ + b)/(cτ + d)

    Solve: cτ² + (d-a)τ - b = 0
    """
    if c == 0:
        # Linear: (d-a)τ = b - aτ → τ(d-a+a) = b → dτ = b
        # But if c=0 and ad=1, then d=±1
        # τ → aτ + b (if d=1) or τ → -aτ - b (if d=-1)
        # No fixed point unless a=1, b=0 (identity)
        return []

    # Quadratic formula
    discriminant = (d - a)**2 + 4*b*c

    if discriminant < 0:
        # Complex solutions - upper half plane
        sqrt_disc = np.sqrt(-discriminant) * 1j
        tau1 = (-(d-a) + sqrt_disc) / (2*c)
        tau2 = (-(d-a) - sqrt_disc) / (2*c)

        # Return only those in upper half plane
        fixed_points = []
        if tau1.imag > 0.01:
            fixed_points.append(tau1)
        if tau2.imag > 0.01 and abs(tau2 - tau1) > 0.01:
            fixed_points.append(tau2)
        return fixed_points
    else:
        # Real discriminant → real fixed points (boundary)
        # Not in upper half plane
        return []

def scan_gamma_N_fixed_points(N, elements):
    """
    Find all fixed points of Γ(N)
    """
    gamma_N_elements = [e for e in elements if is_in_gamma_N(*e, N)]

    print(f"\n{'='*70}")
    print(f"Γ({N}): Found {len(gamma_N_elements)} transformations")

    all_fixed_points = []

    for a, b, c, d in gamma_N_elements:
        if a == 1 and b == 0 and c == 0 and d == 1:
            # Identity - skip
            continue

        fps = find_fixed_point(a, b, c, d)

        for tau_fp in fps:
            # Check if already found (deduplication)
            is_new = True
            for tau_existing in all_fixed_points:
                if abs(tau_fp - tau_existing) < 0.01:
                    is_new = False
                    break

            if is_new:
                all_fixed_points.append(tau_fp)

                # Check distance to our target
                dist_obs = abs(tau_fp - TAU_OBSERVED)
                dist_th14 = abs(tau_fp - TAU_THEORY14)

                if dist_obs < 0.5 or dist_th14 < 0.5:
                    print(f"  *** CLOSE MATCH! ***")
                    print(f"  Transformation: ({a}, {b}, {c}, {d})")
                    print(f"  Fixed point: τ = {tau_fp.real:.4f} + {tau_fp.imag:.4f}i")
                    print(f"  Distance to 2.70i: {dist_obs:.4f}")
                    print(f"  Distance to 2.69i: {dist_th14:.4f}")

    return all_fixed_points

# Generate SL(2,Z) elements
print("\nGenerating SL(2,ℤ) elements...")
elements = generate_SL2Z_elements(max_c=15, max_d=15)
print(f"Generated {len(elements)} transformations\n")

# Known fixed points for reference
print("KNOWN FIXED POINTS:")
print(f"  τ = i (Γ(2) and higher): {1j}")
print(f"  τ = ρ = e^(2πi/3) (Γ(3)): {np.exp(2j*np.pi/3)}")
print(f"  τ = i√2 (some subgroups): {1j*np.sqrt(2)}")
print(f"  τ = i√3 (some subgroups): {1j*np.sqrt(3)}")

# Scan each level
all_results = {}

for N in range(2, 11):
    fixed_points = scan_gamma_N_fixed_points(N, elements)
    all_results[N] = fixed_points

    if fixed_points:
        print(f"\n  Total unique fixed points for Γ({N}): {len(fixed_points)}")

        # Find closest to our targets
        if fixed_points:
            closest_to_obs = min(fixed_points, key=lambda t: abs(t - TAU_OBSERVED))
            closest_to_th14 = min(fixed_points, key=lambda t: abs(t - TAU_THEORY14))

            print(f"  Closest to 2.70i: {closest_to_obs.real:.4f} + {closest_to_obs.imag:.4f}i " +
                  f"(distance: {abs(closest_to_obs - TAU_OBSERVED):.4f})")
            print(f"  Closest to 2.69i: {closest_to_th14.real:.4f} + {closest_to_th14.imag:.4f}i " +
                  f"(distance: {abs(closest_to_th14 - TAU_THEORY14):.4f})")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

best_match_level = None
best_match_tau = None
best_match_distance = float('inf')

for N, fps in all_results.items():
    for tau_fp in fps:
        dist = min(abs(tau_fp - TAU_OBSERVED), abs(tau_fp - TAU_THEORY14))
        if dist < best_match_distance:
            best_match_distance = dist
            best_match_tau = tau_fp
            best_match_level = N

print(f"\nBest match overall:")
print(f"  Level: Γ({best_match_level})")
print(f"  Fixed point: τ = {best_match_tau.real:.4f} + {best_match_tau.imag:.4f}i")
print(f"  Distance to observed: {abs(best_match_tau - TAU_OBSERVED):.4f}")
print(f"  Distance to Theory #14: {abs(best_match_tau - TAU_THEORY14):.4f}")

# Interpretation
print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)

if best_match_distance < 0.1:
    print("\n✓✓✓ EXCELLENT MATCH!")
    print(f"τ ≈ 2.7i is likely a fixed point of Γ({best_match_level})!")
    print("\nThis means:")
    print(f"  • Modular parameter is stabilized by Γ({best_match_level}) symmetry")
    print("  • Enhanced symmetry point in moduli space")
    print("  • Dynamical explanation: τ flows to this symmetric configuration")
    print("\nImplication:")
    print("  → τ is not arbitrary - selected by symmetry!")
    print("  → Reduces free parameters from theory")

elif best_match_distance < 0.5:
    print("\n✓ CLOSE MATCH")
    print(f"τ ≈ 2.7i is close to Γ({best_match_level}) fixed point")
    print("\nPossibilities:")
    print("  1. Higher-level subgroup not scanned (N > 10)")
    print("  2. Quantum corrections shift τ slightly from fixed point")
    print("  3. τ near fixed point but not exactly on it")
    print("\nFurther investigation:")
    print("  • Scan higher levels (N = 11-20)")
    print("  • Include quantum corrections to moduli potential")
    print("  • Check if τ satisfies other stabilization conditions")

else:
    print("\n⊗ NO CLOSE MATCH FOUND")
    print(f"τ ≈ 2.7i is NOT close to Γ(N) fixed points for N = 2-10")
    print(f"Best match distance: {best_match_distance:.4f}")
    print("\nPossibilities:")
    print("  1. τ is fixed point of higher-level subgroup (N > 10)")
    print("  2. τ from different mechanism (flux stabilization, etc.)")
    print("  3. τ is 'generic' value not tied to symmetry")
    print("  4. τ from landscape/anthropic selection")
    print("\nNext steps:")
    print("  • Scan much higher levels (N up to 50)")
    print("  • Consider other subgroups (Γ₀(N), Γ₁(N))")
    print("  • Check string compactification predictions")

# Visualization
print("\n" + "="*70)
print("Creating visualization...")
print("="*70)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: All fixed points
ax1.set_xlim(-0.6, 0.6)
ax1.set_ylim(0, 4)
ax1.set_xlabel('Re(τ)', fontsize=12)
ax1.set_ylabel('Im(τ)', fontsize=12)
ax1.set_title('Γ(N) Fixed Points (N=2-10)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)

# Fundamental domain
theta = np.linspace(0, np.pi, 100)
x_domain = np.cos(theta)
y_domain = np.sin(theta)
ax1.fill_between(x_domain, y_domain, 4, alpha=0.1, color='gray', label='Fundamental domain')

# Plot fixed points by level
colors = plt.cm.tab10(np.linspace(0, 1, 10))
for i, N in enumerate(range(2, 11)):
    fps = all_results[N]
    if fps:
        re_parts = [fp.real for fp in fps]
        im_parts = [fp.imag for fp in fps]
        ax1.scatter(re_parts, im_parts, s=50, alpha=0.7,
                   color=colors[i], label=f'Γ({N})', marker='o', edgecolors='black')

# Mark our targets
ax1.plot(TAU_OBSERVED.real, TAU_OBSERVED.imag, 'r*', markersize=20,
        label='τ ≈ 2.70i (RG fit)', markeredgecolor='black', markeredgewidth=1.5)
ax1.plot(TAU_THEORY14.real, TAU_THEORY14.imag, 'g*', markersize=20,
        label='τ ≈ 2.69i (Theory #14)', markeredgecolor='black', markeredgewidth=1.5)

ax1.legend(loc='upper right', fontsize=8, ncol=2)

# Plot 2: Zoom near τ ≈ 2.7i
ax2.set_xlim(-0.3, 0.3)
ax2.set_ylim(2.0, 3.5)
ax2.set_xlabel('Re(τ)', fontsize=12)
ax2.set_ylabel('Im(τ)', fontsize=12)
ax2.set_title('Zoom: Near τ ≈ 2.7i', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='k', linewidth=0.5)
ax2.axvline(x=0, color='k', linewidth=0.5)

# Plot only nearby fixed points
for i, N in enumerate(range(2, 11)):
    fps = all_results[N]
    nearby_fps = [fp for fp in fps if abs(fp - TAU_OBSERVED) < 1.0]
    if nearby_fps:
        re_parts = [fp.real for fp in nearby_fps]
        im_parts = [fp.imag for fp in nearby_fps]
        ax2.scatter(re_parts, im_parts, s=100, alpha=0.7,
                   color=colors[i], label=f'Γ({N})', marker='o', edgecolors='black')

        # Label them
        for fp in nearby_fps:
            ax2.text(fp.real + 0.02, fp.imag + 0.05, f'{fp.imag:.2f}i',
                    fontsize=8, alpha=0.7)

# Mark targets
ax2.plot(TAU_OBSERVED.real, TAU_OBSERVED.imag, 'r*', markersize=25,
        label='Target (2.70i)', markeredgecolor='black', markeredgewidth=2)
ax2.plot(TAU_THEORY14.real, TAU_THEORY14.imag, 'g*', markersize=25,
        label='Theory #14 (2.69i)', markeredgecolor='black', markeredgewidth=2)

# Draw circle of radius 0.1 around targets
circle1 = plt.Circle((TAU_OBSERVED.real, TAU_OBSERVED.imag), 0.1,
                     color='red', fill=False, linestyle='--', alpha=0.5)
circle2 = plt.Circle((TAU_THEORY14.real, TAU_THEORY14.imag), 0.1,
                     color='green', fill=False, linestyle='--', alpha=0.5)
ax2.add_patch(circle1)
ax2.add_patch(circle2)

ax2.legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig('modular_fixed_points_scan.png', dpi=150, bbox_inches='tight')
print("Visualization saved: modular_fixed_points_scan.png")

print("\n" + "="*70)
print("SCAN COMPLETE!")
print("="*70)
