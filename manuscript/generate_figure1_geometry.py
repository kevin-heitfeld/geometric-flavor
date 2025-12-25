"""
Generate Figure 1: Calabi-Yau Geometry Schematic
Shows D7-branes wrapping four-cycles in the CY threefold
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Circle, Ellipse
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d import Axes3D

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

def plot_cy_geometry():
    """Create schematic representation of CY threefold with D7-branes"""

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Draw CY threefold as a torus (schematic)
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, 2 * np.pi, 50)
    U, V = np.meshgrid(u, v)

    # Major and minor radii for torus
    R = 2.0  # major radius
    r = 0.8  # minor radius

    X = (R + r * np.cos(V)) * np.cos(U)
    Y = (R + r * np.cos(V)) * np.sin(U)
    Z = r * np.sin(V)

    # Plot the CY manifold (torus as proxy)
    ax.plot_surface(X, Y, Z, alpha=0.15, color='lightblue',
                    edgecolor='none', shade=True)

    # Draw D7-brane wrapping cycle D1 (red)
    u1 = np.linspace(0, 2 * np.pi, 100)
    v1 = np.pi / 4  # Fixed angle
    X1 = (R + r * np.cos(v1)) * np.cos(u1)
    Y1 = (R + r * np.cos(v1)) * np.sin(u1)
    Z1 = r * np.sin(v1) * np.ones_like(u1)
    ax.plot(X1, Y1, Z1, 'r-', linewidth=3, label=r'$D_1$ cycle', alpha=0.8)

    # Draw D7-brane wrapping cycle D2 (blue)
    u2 = np.pi / 3  # Fixed angle
    v2 = np.linspace(0, 2 * np.pi, 100)
    X2 = (R + r * np.cos(v2)) * np.cos(u2)
    Y2 = (R + r * np.cos(v2)) * np.sin(u2)
    Z2 = r * np.sin(v2)
    ax.plot(X2, Y2, Z2, 'b-', linewidth=3, label=r'$D_2$ cycle', alpha=0.8)

    # Mark intersection point (Yukawa point)
    yukawa_u = u2
    yukawa_v = v1
    yukawa_x = (R + r * np.cos(yukawa_v)) * np.cos(yukawa_u)
    yukawa_y = (R + r * np.cos(yukawa_v)) * np.sin(yukawa_u)
    yukawa_z = r * np.sin(yukawa_v)
    ax.scatter([yukawa_x], [yukawa_y], [yukawa_z],
               color='gold', s=200, marker='*',
               label='Yukawa point', edgecolor='black', linewidth=1.5, zorder=10)

    # Add 4D spacetime representation
    spacetime_x = np.array([-3.5, -3.5, -2.5, -2.5, -3.5])
    spacetime_y = np.array([-3, -2, -2, -3, -3])
    spacetime_z = np.array([0, 0, 0, 0, 0])
    ax.plot(spacetime_x, spacetime_y, spacetime_z, 'k-', linewidth=2)
    ax.text(-3, -2.5, -0.5, r'$\mathbb{R}^{1,3}$', fontsize=14, weight='bold')

    # Add arrows showing dimensional reduction
    arrow = Arrow3D([-2.5, -1], [-2.5, -0.5], [0, 0],
                    mutation_scale=20, lw=2, arrowstyle='-|>', color='black')
    ax.add_artist(arrow)
    ax.text(-1.7, -1.5, 0.3, 'Compactification', fontsize=11, style='italic')

    # Add labels
    ax.text(0, 0, 2, r'$X = \mathbb{P}_{11226}[12]$', fontsize=16, weight='bold',
            ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add wrapping annotation
    ax.text(2.5, -1, 1.5, r'$\Sigma_4 = w_1 D_1 + w_2 D_2$', fontsize=13,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    ax.text(2.5, -1, 1.0, r'$(w_1, w_2) = (1, 1)$', fontsize=12)

    # Styling
    ax.set_xlabel('$x_1$', fontsize=12, labelpad=10)
    ax.set_ylabel('$x_2$', fontsize=12, labelpad=10)
    ax.set_zlabel('$x_3$', fontsize=12, labelpad=10)
    ax.set_title('D7-Brane Configuration on Calabi-Yau Threefold',
                 fontsize=16, weight='bold', pad=20)

    # Set viewing angle
    ax.view_init(elev=20, azim=45)

    # Remove grid and adjust limits
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-4, 3])
    ax.set_ylim([-4, 3])
    ax.set_zlim([-2, 3])

    # Legend
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)

    plt.tight_layout()
    plt.savefig('figures/figure1_geometry.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/figure1_geometry.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 1 generated: figure1_geometry.pdf/.png")
    plt.close()

if __name__ == "__main__":
    plot_cy_geometry()
