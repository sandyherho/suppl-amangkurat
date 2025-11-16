#!/usr/bin/env python
"""Figure 1: Spatiotemporal evolution heatmaps (2×2 grid, shared colorbar)."""

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from pathlib import Path


def load_data(filename):
    """Load x, t, phi from NetCDF."""
    filepath = Path('../model_outputs') / filename
    with Dataset(filepath, 'r') as nc:
        x = nc.variables['x'][:]
        t = nc.variables['t'][:]
        phi = nc.variables['phi'][:]
    return x, t, phi


def compute_center_of_mass(x, phi):
    """Compute energy center <x>(t)."""
    phi2 = phi**2
    norm = np.trapz(phi2, x, axis=1)
    xc = np.trapz(phi2 * x[np.newaxis, :], x, axis=1) / norm
    return xc


def main():
    # Configuration
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'text.usetex': False,
    })

    # Cases
    CASES = [
        ('case_1_linear_wave.nc', 'Linear Wave'),
        ('case_2_kink_soliton.nc', 'Kink Soliton'),
        ('case_3_breather.nc', 'Breather'),
        ('case_4_kink_antikink_collision.nc', 'Collision')
    ]

    # Create output directories
    STATS_DIR = Path('../stats')
    FIGS_DIR = Path('../figs')
    STATS_DIR.mkdir(exist_ok=True)
    FIGS_DIR.mkdir(exist_ok=True)

    # Load all data first to determine global color limits
    print("Loading data and computing global limits...")
    all_data = []
    phi_global_min = np.inf
    phi_global_max = -np.inf
    
    for filename, label in CASES:
        x, t, phi = load_data(filename)
        all_data.append((x, t, phi, label))
        
        phi_min = phi.min()
        phi_max = phi.max()
        
        if phi_min < phi_global_min:
            phi_global_min = phi_min
        if phi_max > phi_global_max:
            phi_global_max = phi_max
    
    # Use symmetric colorbar around zero
    vmax = max(abs(phi_global_min), abs(phi_global_max))
    vmin = -vmax
    
    print(f"  Global φ range: [{vmin:.3f}, {vmax:.3f}]")

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Statistics file
    stats_file = STATS_DIR / 'fig1_spatiotemporal_stats.txt'
    with open(stats_file, 'w') as f:
        f.write("FIGURE 1: SPATIOTEMPORAL EVOLUTION\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Global colorbar limits: [{vmin:.4f}, {vmax:.4f}]\n\n")

    # Plot each case
    print("\nCreating spatiotemporal plots...")
    for idx, ((x, t, phi, label), ax) in enumerate(zip(all_data, axes)):
        nt, nx = phi.shape
        
        print(f"  Processing {label}...")
        
        # Plot heatmap with global color limits
        extent = [x.min(), x.max(), t.min(), t.max()]
        im = ax.imshow(phi, aspect='auto', origin='lower', 
                       extent=extent, cmap='RdBu_r', 
                       vmin=vmin, vmax=vmax)
        
        # Overlay energy center trajectory
        xc = compute_center_of_mass(x, phi)
        ax.plot(xc, t, 'k--', linewidth=1.5, alpha=0.8)
        
        # Labels
        ax.set_xlabel(r'Position $x$')
        ax.set_ylabel(r'Time $t$')
        
        # Subplot label only (no title)
        ax.text(0.02, 0.98, f'({chr(97+idx)})', transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='white', 
                         edgecolor='none', alpha=0.8))
        
        # Statistics
        velocity = (xc[-1] - xc[0]) / (t[-1] - t[0])
        
        with open(stats_file, 'a') as f:
            f.write(f"Case {idx+1}: {label}\n")
            f.write(f"  Grid: {nx} × {nt}\n")
            f.write(f"  Domain: x ∈ [{x.min():.2f}, {x.max():.2f}], ")
            f.write(f"t ∈ [{t.min():.2f}, {t.max():.2f}]\n")
            f.write(f"  Field amplitude: max|φ| = {np.abs(phi).max():.4f}\n")
            f.write(f"  Field range: [{phi.min():.4f}, {phi.max():.4f}]\n")
            f.write(f"  Center drift: Δx = {xc[-1] - xc[0]:.4f}\n")
            f.write(f"  Propagation velocity: v = {velocity:.4f}\n")
            f.write(f"  Interpretation: ")
            
            if idx == 0:
                f.write("Linear dispersive propagation.\n")
            elif idx == 1:
                f.write("Static topological soliton.\n")
            elif idx == 2:
                f.write("Periodic breather oscillation.\n")
            elif idx == 3:
                f.write("Collision with bounce dynamics.\n")
            f.write("\n")

    # Add single colorbar for entire figure
    fig.subplots_adjust(right=0.90)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r'$\phi$', rotation=270, labelpad=25, fontsize=13)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout(rect=[0, 0, 0.90, 1])
    
    # Save
    print("\nSaving figures...")
    plt.savefig(FIGS_DIR / 'fig1_spatiotemporal.pdf', transparent=False)
    plt.savefig(FIGS_DIR / 'fig1_spatiotemporal.png', transparent=False)
    plt.savefig(FIGS_DIR / 'fig1_spatiotemporal.eps', transparent=False)
    
    print("Figure 1 saved: ../figs/fig1_spatiotemporal.{pdf,png,eps}")
    print(f"Statistics: {stats_file}")
    plt.close()
    print("\nDone!")


if __name__ == '__main__':
    main()
