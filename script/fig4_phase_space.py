#!/usr/bin/env python
"""Figure 4: Phase space structure (2×2 grid, scenario-colored, unified limits)."""

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


def time_derivative(phi, t):
    """Compute dphi/dt using central differences."""
    dt = t[1] - t[0]
    phi_dot = np.gradient(phi, dt, axis=0)
    return phi_dot


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

    # Cases with colors
    CASES = [
        ('case_1_linear_wave.nc', 'Linear Wave', 'blue'),
        ('case_2_kink_soliton.nc', 'Kink Soliton', 'orange'),
        ('case_3_breather.nc', 'Breather', 'green'),
        ('case_4_kink_antikink_collision.nc', 'Collision', 'red')
    ]

    # Create output directories
    STATS_DIR = Path('../stats')
    FIGS_DIR = Path('../figs')
    STATS_DIR.mkdir(exist_ok=True)
    FIGS_DIR.mkdir(exist_ok=True)

    # Load all data and compute global limits
    print("Loading data and computing global limits...")
    all_data = []
    phi_all = []
    phi_dot_all = []
    
    for filename, label, color in CASES:
        x, t, phi = load_data(filename)
        center_idx = len(x) // 2
        phi_center = phi[:, center_idx]
        phi_dot = time_derivative(phi, t)
        phi_dot_center = phi_dot[:, center_idx]
        
        all_data.append({
            'label': label,
            'color': color,
            'phi': phi_center,
            'phi_dot': phi_dot_center,
            't': t
        })
        
        phi_all.extend(phi_center)
        phi_dot_all.extend(phi_dot_center)
    
    # Global limits (with 10% margin)
    phi_min, phi_max = np.min(phi_all), np.max(phi_all)
    phi_dot_min, phi_dot_max = np.min(phi_dot_all), np.max(phi_dot_all)
    
    phi_range = phi_max - phi_min
    phi_dot_range = phi_dot_max - phi_dot_min
    
    xlim = (phi_min - 0.1 * phi_range, phi_max + 0.1 * phi_range)
    ylim = (phi_dot_min - 0.1 * phi_dot_range, phi_dot_max + 0.1 * phi_dot_range)
    
    print(f"  Global φ range: [{xlim[0]:.3f}, {xlim[1]:.3f}]")
    print(f"  Global ∂ₜφ range: [{ylim[0]:.3f}, {ylim[1]:.3f}]")

    # Create figure (2×2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Statistics file
    stats_file = STATS_DIR / 'fig4_phase_space_stats.txt'
    with open(stats_file, 'w') as f:
        f.write("FIGURE 4: PHASE SPACE ANALYSIS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Global limits:\n")
        f.write(f"  φ(x=0,t): [{xlim[0]:.4f}, {xlim[1]:.4f}]\n")
        f.write(f"  ∂ₜφ(x=0,t): [{ylim[0]:.4f}, {ylim[1]:.4f}]\n\n")

    # Plot each case
    print("\nCreating phase space plots...")
    for idx, (ax, data) in enumerate(zip(axes, all_data)):
        label = data['label']
        color = data['color']
        phi_center = data['phi']
        phi_dot_center = data['phi_dot']
        
        print(f"  Processing {label}...")
        
        # Scatter plot with bigger points, scenario color, no time encoding
        ax.scatter(phi_center, phi_dot_center, 
                  c=color, s=30, alpha=0.7, 
                  edgecolors='face', linewidths=0,
                  rasterized=True)
        
        # Set same limits for all subplots
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        # Labels (no [dimensionless])
        ax.set_xlabel(r'$\phi(x=0, t)$')
        ax.set_ylabel(r'$\partial_t \phi(x=0, t)$')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Subplot label only (no title)
        ax.text(0.02, 0.98, f'({chr(97+idx)})', transform=ax.transAxes,
               fontsize=14, fontweight='bold', va='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='white', 
                        edgecolor='none', alpha=0.8))
        
        # Statistics
        corr = np.corrcoef(phi_center, phi_dot_center)[0, 1]
        rms_phi = np.sqrt(np.mean(phi_center**2))
        rms_dot = np.sqrt(np.mean(phi_dot_center**2))
        
        with open(stats_file, 'a') as f:
            f.write(f"Case {idx+1}: {label}\n")
            f.write(f"  Field φ(x=0, t):\n")
            f.write(f"    Mean: {phi_center.mean():.4f}\n")
            f.write(f"    Std: {phi_center.std():.4f}\n")
            f.write(f"    Range: [{phi_center.min():.4f}, {phi_center.max():.4f}]\n")
            f.write(f"    RMS: {rms_phi:.4f}\n")
            f.write(f"  Derivative ∂ₜφ(x=0, t):\n")
            f.write(f"    Mean: {phi_dot_center.mean():.4f}\n")
            f.write(f"    Std: {phi_dot_center.std():.4f}\n")
            f.write(f"    Range: [{phi_dot_center.min():.4f}, {phi_dot_center.max():.4f}]\n")
            f.write(f"    RMS: {rms_dot:.4f}\n")
            f.write(f"  Correlation: ρ = {corr:.4f}\n")
            f.write(f"  Interpretation: ")
            
            if idx == 0:
                f.write("Dispersive wave trajectory in phase space.\n")
            elif idx == 1:
                f.write("Near-stationary point (static soliton).\n")
            elif idx == 2:
                f.write("Closed orbit (periodic breather dynamics).\n")
            elif idx == 3:
                f.write("Complex trajectory from collision events.\n")
            f.write("\n")

    plt.tight_layout()
    
    # Save
    print("\nSaving figures...")
    plt.savefig(FIGS_DIR / 'fig4_phase_space.pdf', transparent=False)
    plt.savefig(FIGS_DIR / 'fig4_phase_space.png', transparent=False)
    plt.savefig(FIGS_DIR / 'fig4_phase_space.eps', transparent=False)
    
    print("Figure 4 saved: ../figs/fig4_phase_space.{pdf,png,eps}")
    print(f"Statistics: {stats_file}")
    plt.close()
    print("\nDone!")


if __name__ == '__main__':
    main()
