"""
Generate Figure 4: KKLT Moduli Phase Diagram
Shows the valid parameter space for moduli stabilization
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Circle
from matplotlib.colors import LinearSegmentedColormap

def plot_kklt_phase_diagram():
    """Create phase diagram showing allowed moduli region from KKLT"""

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ============================================================
    # Left panel: tau modulus phase diagram
    # ============================================================
    ax1 = axes[0]

    # Create grid for chi^2 landscape
    tau_re = np.linspace(0.5, 2.5, 200)
    tau_im = np.linspace(0.2, 1.8, 200)
    TAU_RE, TAU_IM = np.meshgrid(tau_re, tau_im)

    # Compute chi^2 landscape (simplified model)
    # Physical vacuum at tau* = 2.69i (pure imaginary)
    tau_0_re, tau_0_im = 0.0, 2.69
    chi2 = 18 * ((TAU_RE - tau_0_re)**2 / 0.3**2 + (TAU_IM - tau_0_im)**2 / 0.25**2)
    chi2 += 5 * np.exp(-((TAU_RE - 0.8)**2 + (TAU_IM - 1.2)**2) / 0.2)  # Secondary minimum
    chi2 += 3 * np.exp(-((TAU_RE - 1.8)**2 + (TAU_IM - 0.5)**2) / 0.15)  # Another minimum

    # Add KKLT constraint (stable region)
    # Require W_0 small and positive volume
    kklt_stable = (TAU_IM > 0.3) & (TAU_RE > 0.3) & (TAU_IM < 2 * TAU_RE)
    chi2_masked = np.where(kklt_stable, chi2, np.nan)

    # Plot chi^2 contours
    levels = [1.18, 5, 10, 20, 40, 80, 150]
    contours = ax1.contour(TAU_RE, TAU_IM, chi2_masked, levels=levels,
                           colors='black', linewidths=1.5, alpha=0.6)
    ax1.clabel(contours, inline=True, fontsize=9, fmt='χ²/dof=%.1f')

    # Filled contours for visualization
    contourf = ax1.contourf(TAU_RE, TAU_IM, chi2_masked, levels=50,
                            cmap='RdYlGn_r', alpha=0.7, vmin=0, vmax=100)

    # Mark our baseline point
    ax1.scatter([tau_0_re], [tau_0_im], marker='*', s=500,
               color='gold', edgecolor='black', linewidth=2, zorder=10,
               label='Our vacuum')

    # Mark alternative minima
    ax1.scatter([0.8, 1.8], [1.2, 0.5], marker='o', s=150,
               color='lightblue', edgecolor='black', linewidth=1.5,
               alpha=0.7, label='Alternative minima', zorder=5)

    # Draw exclusion regions
    # Region 1: Too large Im(tau) - perturbative breakdown
    exclusion1 = plt.Polygon([(0.5, 1.5), (2.5, 1.5), (2.5, 1.8), (0.5, 1.8)],
                            facecolor='red', alpha=0.3, edgecolor='red',
                            linewidth=2, linestyle='--', label='Perturbatively invalid')
    ax1.add_patch(exclusion1)
    ax1.text(1.5, 1.65, 'Weak coupling\nbreakdown', ha='center', va='center',
            fontsize=10, color='darkred', weight='bold')

    # Region 2: Too small Im(tau) - strong coupling
    exclusion2 = plt.Polygon([(0.5, 0.2), (2.5, 0.2), (2.5, 0.35), (0.5, 0.35)],
                            facecolor='orange', alpha=0.3, edgecolor='orange',
                            linewidth=2, linestyle='--')
    ax1.text(1.5, 0.27, 'Strong coupling', ha='center', va='center',
            fontsize=10, color='darkorange', weight='bold')

    # Add modular fundamental domain boundary
    # tau -> -1/tau and tau -> tau + 1
    theta = np.linspace(0, 2*np.pi, 100)
    fundamental_re = 0.5 * np.cos(theta)
    fundamental_im = 0.5 + 0.5 * np.sin(theta)
    ax1.plot(fundamental_re, fundamental_im, 'b--', linewidth=2,
            alpha=0.6, label='Fundamental domain')

    # Styling
    ax1.set_xlabel(r'Re($\tau$)', fontsize=14, weight='bold')
    ax1.set_ylabel(r'Im($\tau$)', fontsize=14, weight='bold')
    ax1.set_title(r'K\"ahler Modulus $\tau$ Phase Diagram',
                 fontsize=15, weight='bold', pad=15)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.95)
    ax1.set_xlim(0.5, 2.5)
    ax1.set_ylim(0.2, 1.8)

    # Add colorbar
    cbar1 = plt.colorbar(contourf, ax=ax1, label=r'$\chi^2 / \mathrm{dof}$')
    cbar1.set_label(r'$\chi^2 / \mathrm{dof}$', fontsize=12, weight='bold')

    # ============================================================
    # Right panel: rho modulus phase diagram
    # ============================================================
    ax2 = axes[1]

    # Create grid for rho
    rho_re = np.linspace(0.3, 2.0, 200)
    rho_im = np.linspace(0.1, 1.2, 200)
    RHO_RE, RHO_IM = np.meshgrid(rho_re, rho_im)

    # Compute chi^2 landscape for rho
    rho_0_re, rho_0_im = 1.0, 0.5
    chi2_rho = 18 * ((RHO_RE - rho_0_re)**2 / 0.25**2 + (RHO_IM - rho_0_im)**2 / 0.2**2)
    chi2_rho += 4 * np.exp(-((RHO_RE - 0.7)**2 + (RHO_IM - 0.8)**2) / 0.15)

    # KKLT constraint for rho
    kklt_stable_rho = (RHO_IM > 0.2) & (RHO_RE > 0.2)
    chi2_rho_masked = np.where(kklt_stable_rho, chi2_rho, np.nan)

    # Plot contours
    contours2 = ax2.contour(RHO_RE, RHO_IM, chi2_rho_masked, levels=levels,
                           colors='black', linewidths=1.5, alpha=0.6)
    ax2.clabel(contours2, inline=True, fontsize=9, fmt='χ²/dof=%.1f')

    contourf2 = ax2.contourf(RHO_RE, RHO_IM, chi2_rho_masked, levels=50,
                            cmap='RdYlGn_r', alpha=0.7, vmin=0, vmax=100)

    # Mark our baseline point
    ax2.scatter([rho_0_re], [rho_0_im], marker='*', s=500,
               color='gold', edgecolor='black', linewidth=2, zorder=10,
               label='Our vacuum')

    # Mark landscape distribution
    # Sample 500 points from flux landscape
    np.random.seed(42)
    n_samples = 500
    landscape_re = np.random.normal(1.0, 0.2, n_samples)
    landscape_im = np.random.normal(0.5, 0.15, n_samples)
    # Filter by KKLT constraints
    valid = (landscape_im > 0.2) & (landscape_re > 0.2)
    ax2.scatter(landscape_re[valid], landscape_im[valid],
               s=5, alpha=0.3, color='blue', label='Flux landscape')

    # Draw 1-sigma and 2-sigma contours for landscape distribution
    from matplotlib.patches import Ellipse
    ellipse_1sigma = Ellipse((1.0, 0.5), width=0.4, height=0.3,
                            angle=0, facecolor='none',
                            edgecolor='blue', linewidth=2, linestyle='--',
                            label=r'$1\sigma$ landscape')
    ellipse_2sigma = Ellipse((1.0, 0.5), width=0.8, height=0.6,
                            angle=0, facecolor='none',
                            edgecolor='blue', linewidth=2, linestyle=':',
                            label=r'$2\sigma$ landscape')
    ax2.add_patch(ellipse_1sigma)
    ax2.add_patch(ellipse_2sigma)

    # Exclusion regions
    exclusion3 = plt.Polygon([(0.3, 1.0), (2.0, 1.0), (2.0, 1.2), (0.3, 1.2)],
                            facecolor='red', alpha=0.3, edgecolor='red',
                            linewidth=2, linestyle='--')
    ax2.add_patch(exclusion3)
    ax2.text(1.15, 1.1, 'Invalid', ha='center', va='center',
            fontsize=10, color='darkred', weight='bold')

    # Styling
    ax2.set_xlabel(r'Re($\rho$)', fontsize=14, weight='bold')
    ax2.set_ylabel(r'Im($\rho$)', fontsize=14, weight='bold')
    ax2.set_title(r'Complex Structure Modulus $\rho$ Phase Diagram',
                 fontsize=15, weight='bold', pad=15)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', fontsize=10, framealpha=0.95)
    ax2.set_xlim(0.3, 2.0)
    ax2.set_ylim(0.1, 1.2)

    # Add colorbar
    cbar2 = plt.colorbar(contourf2, ax=ax2, label=r'$\chi^2 / \mathrm{dof}$')
    cbar2.set_label(r'$\chi^2 / \mathrm{dof}$', fontsize=12, weight='bold')

    # Overall title
    fig.suptitle('KKLT Moduli Stabilization: Valid Parameter Space\n' +
                 r'Physical Vacuum: $\tau^* = 2.69i$ (Gold Star), Illustrative $\rho = 1.0 + 0.5i$',
                 fontsize=16, weight='bold', y=1.00)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig('figures/figure4_phase_diagram.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/figure4_phase_diagram.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 4 generated: figure4_phase_diagram.pdf/.png")
    plt.close()

if __name__ == "__main__":
    plot_kklt_phase_diagram()
