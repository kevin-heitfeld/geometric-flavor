"""
Summary visualization of all three moduli constraints.

Shows:
- τ = 2.69i from 30 observables
- g_s ~ 0.5-1.0 from gauge unification
- Im(T) ~ 0.8 from anomaly + KKLT + Yukawas
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D

# Set up publication-quality plot
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = False

fig = plt.figure(figsize=(14, 10))

# Create three panels
ax1 = plt.subplot(2, 2, 1)  # Complex structure τ
ax2 = plt.subplot(2, 2, 2)  # Dilaton g_s
ax3 = plt.subplot(2, 2, 3)  # Kähler Im(T)
ax4 = plt.subplot(2, 2, 4)  # Summary table

# Colors
color_determined = '#2E7D32'  # Dark green
color_constrained = '#1976D2'  # Blue
color_converge = '#C62828'  # Red

#=============================================================================
# Panel 1: Complex Structure τ = 2.69i
#=============================================================================
ax1.set_xlim(1.5, 4.0)
ax1.set_ylim(0, 1)
ax1.axvline(2.69, color=color_determined, linewidth=3, label='τ = 2.69i')
ax1.axvspan(2.64, 2.74, alpha=0.3, color=color_determined, label='±2% uncertainty')

# Add annotation
ax1.text(2.69, 0.5, 'Im(τ) = 2.69 ± 0.05\nFrom 30 observables\n(flavor + cosmology)',
         ha='center', va='center', fontsize=10,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=color_determined, linewidth=2))

ax1.set_xlabel('Im(τ) = Im(U)', fontsize=12, fontweight='bold')
ax1.set_ylabel('', fontsize=12)
ax1.set_title('(a) Complex Structure Modulus', fontsize=13, fontweight='bold')
ax1.set_yticks([])
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left', fontsize=9)

#=============================================================================
# Panel 2: Dilaton g_s
#=============================================================================
ax2.set_xlim(0.2, 1.5)
ax2.set_ylim(0, 3)

# MSSM result
g_s_mssm = 0.72
g_s_mssm_range = [0.55, 1.0]  # k=1 to k=2
ax2.barh(2, width=g_s_mssm_range[1]-g_s_mssm_range[0], left=g_s_mssm_range[0],
         height=0.4, color=color_constrained, alpha=0.5, label='MSSM (k=1-2)')
ax2.plot(g_s_mssm, 2, 'o', color=color_constrained, markersize=10, markeredgecolor='black', markeredgewidth=1.5)

# SM result
g_s_sm = 0.55
g_s_sm_range = [0.50, 0.60]  # ±10%
ax2.barh(1, width=g_s_sm_range[1]-g_s_sm_range[0], left=g_s_sm_range[0],
         height=0.4, color='#FF6F00', alpha=0.5, label='SM (4% spread)')
ax2.plot(g_s_sm, 1, 's', color='#FF6F00', markersize=10, markeredgecolor='black', markeredgewidth=1.5)

# Agnostic bracket
ax2.axvspan(0.5, 1.0, alpha=0.2, color='gray', label='Agnostic range')

# Annotations
ax2.text(0.72, 2.5, 'MSSM: M_GUT = 2.1×10¹⁶ GeV\nα_GUT = 0.0412 (0.1% spread)',
         ha='center', fontsize=9)
ax2.text(0.55, 0.5, 'SM: M_GUT = 1.8×10¹⁴ GeV\nα_GUT = 0.0242 (4% spread)',
         ha='center', fontsize=9)

ax2.set_xlabel('g_s = exp(Im S)', fontsize=12, fontweight='bold')
ax2.set_ylabel('', fontsize=12)
ax2.set_title('(b) Dilaton (String Coupling)', fontsize=13, fontweight='bold')
ax2.set_yticks([1, 2])
ax2.set_yticklabels(['SM', 'MSSM'])
ax2.grid(True, alpha=0.3, axis='x')
ax2.legend(loc='upper right', fontsize=9)

#=============================================================================
# Panel 3: Kähler Modulus Im(T)
#=============================================================================
ax3.set_xlim(0, 2.0)
ax3.set_ylim(0, 4)

# Three independent estimates
methods = ['Anomaly\n(volume-corrected)', 'KKLT\n(a=0.25)', 'Yukawa\nprefactors']
values = [0.81, 0.80, 0.80]  # Average of SM/MSSM for first, then consistent values
errors_low = [0.04, 0.10, 0.15]
errors_high = [0.05, 0.10, 0.15]

colors_methods = ['#7B1FA2', '#0288D1', '#D32F2F']

for i, (method, val, err_low, err_high, col) in enumerate(zip(methods, values, errors_low, errors_high, colors_methods)):
    y_pos = 3 - i
    ax3.errorbar(val, y_pos, xerr=[[err_low], [err_high]],
                 fmt='o', markersize=12, color=col, markeredgecolor='black',
                 markeredgewidth=1.5, capsize=5, capthick=2, linewidth=2,
                 label=method)

# Convergence region
Im_T_best = 0.80
Im_T_range = [0.65, 0.95]
ax3.axvspan(Im_T_range[0], Im_T_range[1], alpha=0.3, color=color_converge,
            label='Convergence region')
ax3.axvline(Im_T_best, color=color_converge, linewidth=3, linestyle='--',
            label='Best value: 0.80')

# Add text box
ax3.text(1.5, 2.0, 'All three methods\nconverge to\nIm(T) ~ 0.8 ± 0.2\n\nKey: a ~ 0.25\n(not a = 1!)',
         ha='center', va='center', fontsize=10,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=color_converge, linewidth=2))

ax3.set_xlabel('Im(T)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Method', fontsize=12, fontweight='bold')
ax3.set_title('(c) Kähler Modulus - Three Independent Constraints', fontsize=13, fontweight='bold')
ax3.set_yticks([3, 2, 1])
ax3.set_yticklabels(methods)
ax3.grid(True, alpha=0.3, axis='x')
ax3.legend(loc='upper left', fontsize=8)

#=============================================================================
# Panel 4: Summary Table
#=============================================================================
ax4.axis('off')

# Title
ax4.text(0.5, 0.95, 'Summary: All Three Moduli Constrained',
         ha='center', va='top', fontsize=14, fontweight='bold')

# Table data
table_data = [
    ['Modulus', 'Value', 'Method', 'Precision'],
    ['', '', '', ''],
    ['Im(U) = Im(τ)', '2.69 ± 0.05', '30 observables fit', '±2% (unique)'],
    ['Im(S) = g_s', '0.5 - 1.0', 'Gauge unification', 'Factor ~2'],
    ['Im(T)', '0.8 ± 0.2', 'Anomaly+KKLT+Yukawa', '±25%'],
]

# Draw table
cell_height = 0.12
cell_widths = [0.15, 0.25, 0.35, 0.25]
y_start = 0.80

# Header row
y = y_start
for j, (text, width) in enumerate(zip(table_data[0], cell_widths)):
    x = sum(cell_widths[:j])
    ax4.add_patch(Rectangle((x, y), width, cell_height,
                            facecolor='lightgray', edgecolor='black', linewidth=1.5))
    ax4.text(x + width/2, y + cell_height/2, text,
             ha='center', va='center', fontsize=10, fontweight='bold')

# Data rows
for i, row in enumerate(table_data[2:], start=1):
    y = y_start - i * cell_height

    # Color code by precision
    if i == 1:  # τ
        row_color = color_determined
        alpha = 0.2
    elif i == 2:  # g_s
        row_color = color_constrained
        alpha = 0.2
    else:  # Im(T)
        row_color = color_converge
        alpha = 0.2

    for j, (text, width) in enumerate(zip(row, cell_widths)):
        x = sum(cell_widths[:j])
        ax4.add_patch(Rectangle((x, y), width, cell_height,
                                facecolor=row_color, alpha=alpha,
                                edgecolor='black', linewidth=1))
        ax4.text(x + width/2, y + cell_height/2, text,
                 ha='center', va='center', fontsize=9)

# Key implications
implications_y = y_start - len(table_data[2:]) * cell_height - 0.05

ax4.text(0.5, implications_y, 'Physical Implications:',
         ha='center', va='top', fontsize=11, fontweight='bold')

implications = [
    '• String scale: M_s ~ 0.8 × M_Planck ~ 10¹⁹ GeV',
    '• GUT scale: M_GUT ~ 2×10¹⁶ GeV (MSSM)',
    '• Compactification: V_CY ~ 0.7 l_s⁶ (quantum regime)',
    '• Perturbative: g_s < 1 (calculations reliable)',
    '• Testable: Proton decay, SUSY scale, Yukawa running',
]

for i, text in enumerate(implications):
    ax4.text(0.05, implications_y - 0.07 - i*0.06, text,
             ha='left', va='top', fontsize=9)

# Status box
status_y = 0.08
ax4.add_patch(FancyBboxPatch((0.15, status_y), 0.7, 0.06,
                             boxstyle='round,pad=0.01',
                             facecolor='#4CAF50', edgecolor='black', linewidth=2))
ax4.text(0.5, status_y + 0.03, 'STATUS: All three moduli determined by phenomenological consistency!',
         ha='center', va='center', fontsize=10, fontweight='bold', color='white')

ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)

#=============================================================================
# Overall title
#=============================================================================
fig.suptitle('Moduli Stabilization from Phenomenology: Complete Solution',
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('moduli_summary_all_three.png', dpi=300, bbox_inches='tight')
print("Saved: moduli_summary_all_three.png")

plt.savefig('moduli_exploration/moduli_summary_all_three.png', dpi=300, bbox_inches='tight')
print("Saved: moduli_exploration/moduli_summary_all_three.png")

plt.show()

print("\n" + "="*70)
print("MODULI DETERMINATION COMPLETE")
print("="*70)
print(f"\nComplex Structure:  Im(U) = Im(τ) = 2.69 ± 0.05  (±2%)")
print(f"Dilaton:            Im(S) = g_s = 0.5-1.0        (factor 2)")
print(f"Kähler Modulus:     Im(T) = 0.80 ± 0.20         (±25%)")
print(f"\nAll three constrained by phenomenological consistency!")
print("="*70)
