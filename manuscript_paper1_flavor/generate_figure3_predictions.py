"""
Generate Figure 3: Experimental Predictions Timeline
Shows the three falsifiable predictions and when they'll be tested
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np

def plot_predictions_timeline():
    """Create timeline showing experimental tests of our three predictions"""

    fig, ax = plt.subplots(figsize=(16, 10))

    # Timeline years
    years = np.arange(2024, 2036)
    year_positions = np.arange(len(years))

    # Predictions data: [name, our_prediction, current_limit, target_sensitivity, test_year_start, test_year_end, experiments]
    predictions = [
        {
            'name': r'Neutrinoless Double-Beta Decay: $\langle m_{\beta\beta} \rangle$',
            'prediction': r'$(10.5 \pm 1.5)$ meV',
            'prediction_val': 10.5,
            'prediction_err': 1.5,
            'current_limit': '< 60 meV (90% CL)',
            'target': '5-10 meV sensitivity',
            'years': [(2025, 2030, 'LEGEND-1000', 'steelblue'),
                     (2027, 2032, 'nEXO', 'darkgreen'),
                     (2028, 2033, 'CUPID', 'purple')],
            'y_pos': 3.0,
            'decisive_year': 2030
        },
        {
            'name': r'Leptonic CP Violation: $\delta_{CP}$',
            'prediction': r'$(206 \pm 15)°$',
            'prediction_val': 206,
            'prediction_err': 15,
            'current_limit': r'$(195 \pm 25)°$ (1$\sigma$)',
            'target': r'$\pm 10°$ precision',
            'years': [(2025, 2029, 'DUNE', 'crimson'),
                     (2026, 2031, 'Hyper-K', 'darkorange'),
                     (2030, 2035, 'IceCube-Gen2', 'teal')],
            'y_pos': 2.0,
            'decisive_year': 2029
        },
        {
            'name': r'Absolute Neutrino Mass Scale: $\Sigma m_\nu$',
            'prediction': r'$(60 \pm 8)$ meV',
            'prediction_val': 60,
            'prediction_err': 8,
            'current_limit': '< 120 meV (95% CL)',
            'target': '< 40 meV (95% CL)',
            'years': [(2026, 2030, 'CMB-S4', 'royalblue'),
                     (2024, 2028, 'KATRIN', 'forestgreen'),
                     (2028, 2032, 'Planck+LSS', 'brown')],
            'y_pos': 1.0,
            'decisive_year': 2030
        }
    ]

    # Plot timeline bars for each prediction
    for pred in predictions:
        y = pred['y_pos']

        # Draw horizontal line for this prediction
        ax.hlines(y, 0, len(years)-1, colors='gray', linestyles='--', alpha=0.3, linewidth=1)

        # Plot experimental timeline bars
        for start, end, exp_name, color in pred['years']:
            start_idx = start - 2024
            end_idx = end - 2024
            width = end_idx - start_idx

            rect = FancyBboxPatch((start_idx, y - 0.15), width, 0.3,
                                  boxstyle="round,pad=0.05",
                                  facecolor=color, edgecolor='black',
                                  linewidth=1.5, alpha=0.7)
            ax.add_patch(rect)

            # Add experiment label
            ax.text(start_idx + width/2, y, exp_name,
                   ha='center', va='center', fontsize=9,
                   weight='bold', color='white',
                   bbox=dict(boxstyle='round', facecolor=color,
                            edgecolor='black', linewidth=1, alpha=0.9))

        # Add prediction text on the left
        ax.text(-0.5, y, pred['name'], ha='right', va='center',
               fontsize=12, weight='bold')
        ax.text(-0.5, y - 0.35, f"Prediction: {pred['prediction']}",
               ha='right', va='top', fontsize=10, style='italic',
               color='darkred')
        ax.text(-0.5, y - 0.50, f"Current: {pred['current_limit']}",
               ha='right', va='top', fontsize=9, color='gray')

        # Mark decisive year with a star
        decisive_idx = pred['decisive_year'] - 2024
        ax.scatter([decisive_idx], [y], marker='*', s=500,
                  color='gold', edgecolor='black', linewidth=2, zorder=10)
        ax.text(decisive_idx, y + 0.4, 'Decisive\nTest',
               ha='center', va='bottom', fontsize=9, weight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # Add vertical lines for key years
    key_years = [2025, 2028, 2030, 2033]
    for year in key_years:
        idx = year - 2024
        ax.axvline(idx, color='lightgray', linestyle='-', alpha=0.5, linewidth=1)
        ax.text(idx, 4.2, str(year), ha='center', va='bottom',
               fontsize=11, weight='bold')

    # Add "NOW" marker
    now_idx = 2025 - 2024
    ax.axvline(now_idx, color='red', linestyle='-', alpha=0.8, linewidth=3)
    ax.text(now_idx, 4.5, 'NOW\n(2025)', ha='center', va='bottom',
           fontsize=12, weight='bold', color='red',
           bbox=dict(boxstyle='round', facecolor='white',
                    edgecolor='red', linewidth=2))

    # Add outcome regions
    # Success region (right side)
    ax.add_patch(Rectangle((10, 0.5), 2, 3,
                           facecolor='lightgreen', alpha=0.3,
                           edgecolor='green', linewidth=2, linestyle='--'))
    ax.text(11, 4.0, 'FRAMEWORK\nCONFIRMED', ha='center', va='center',
           fontsize=13, weight='bold', color='darkgreen',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Falsification region (if tests fail)
    ax.text(11, 0.3, 'or FALSIFIED\nif predictions fail',
           ha='center', va='center', fontsize=10, style='italic',
           color='darkred',
           bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.8))

    # Styling
    ax.set_xlim(-2.5, len(years))
    ax.set_ylim(0.2, 4.8)
    ax.set_xlabel('Year', fontsize=14, weight='bold')
    ax.set_title('Experimental Roadmap: Testing Zero-Parameter Flavor Predictions\n' +
                 'Three Falsifiable Tests Within Next Decade',
                 fontsize=18, weight='bold', pad=20)

    # Remove y-axis
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Add legend for experiment colors
    legend_elements = []
    all_experiments = set()
    for pred in predictions:
        for _, _, exp, color in pred['years']:
            if exp not in all_experiments:
                all_experiments.add(exp)
                legend_elements.append(mpatches.Patch(color=color, label=exp, alpha=0.7))

    ax.legend(handles=legend_elements, loc='upper left',
             fontsize=10, title='Experiments', framealpha=0.95,
             ncol=3)

    # Add footnote
    fig.text(0.5, 0.01,
            'If all three predictions confirmed → first successful a priori predictions from string theory in particle physics',
            ha='center', fontsize=11, style='italic', weight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig('figures/figure3_predictions.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/figure3_predictions.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 3 generated: figure3_predictions.pdf/.png")
    plt.close()

if __name__ == "__main__":
    plot_predictions_timeline()
