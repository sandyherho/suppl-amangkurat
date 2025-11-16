#!/usr/bin/env python
"""Figure 2: Wave envelope evolution at three time snapshots (4×3 grid)."""

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


def main():
    # Configuration
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'text.usetex': False,
    })

    # Cases with associated colors
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

    # Create figure (4 rows × 3 columns)
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    
    linestyles = ['-', '--', '-.']
    time_labels = ['Initial', 'Middle', 'Final']

    # Statistics file
    stats_file = STATS_DIR / 'fig2_envelope_stats.txt'
    with open(stats_file, 'w') as f:
        f.write("FIGURE 2: WAVE ENVELOPE EVOLUTION\n")
        f.write("=" * 70 + "\n\n")

    # Load all data first to determine global y-limits for each row
    all_data = []
    y_limits = []
    
    for filename, label, color in CASES:
        x, t, phi = load_data(filename)
        all_data.append((x, t, phi, label, color))
        
        # Determine y-limit for this case (with 10% margin)
        phi_max = np.abs(phi).max()
        y_limits.append((-phi_max * 1.1, phi_max * 1.1))

    # Plot each case
    print("Creating wave envelope plots...")
    for row_idx, ((x, t, phi, label, color), ylim) in enumerate(zip(all_data, y_limits)):
        nt = len(t)
        
        print(f"  Processing {label}...")
        
        # Select three time snapshots
        time_indices = [0, nt // 2, nt - 1]
        
        # Plot each time snapshot in separate column
        for col_idx, (tidx, tlabel, ls) in enumerate(zip(time_indices, time_labels, linestyles)):
            ax = axes[row_idx, col_idx]
            
            phi_snapshot = phi[tidx, :]
            # Use scenario color for all time snapshots
            ax.plot(x, phi_snapshot, color=color, linestyle=ls, linewidth=2.5, alpha=0.9)
            
            # Set same y-limits for the entire row
            ax.set_ylim(ylim)
            ax.set_xlim([x.min(), x.max()])
            
            # Labels (remove [dimensionless])
            if row_idx == 3:
                ax.set_xlabel(r'Position $x$')
            if col_idx == 0:
                ax.set_ylabel(r'$\phi(x,t)$')
            
            # Time as subtitle for all subplots
            ax.set_title(f'$t = {t[tidx]:.2f}$', fontsize=10, pad=8)
            
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            
            # Subplot label
            subplot_letter = chr(97 + row_idx * 3 + col_idx)
            ax.text(0.02, 0.98, f'({subplot_letter})', transform=ax.transAxes,
                   fontsize=12, fontweight='bold', va='top', ha='left',
                   bbox=dict(boxstyle='round', facecolor='white', 
                            edgecolor='none', alpha=0.8))
        
        # Statistics
        with open(stats_file, 'a') as f:
            f.write(f"Case {row_idx+1}: {label}\n")
            for tidx, tlabel in zip(time_indices, time_labels):
                phi_snap = phi[tidx, :]
                peak = np.max(np.abs(phi_snap))
                peak_pos = x[np.argmax(np.abs(phi_snap))]
                rms = np.sqrt(np.trapz(phi_snap**2, x) / (x.max() - x.min()))
                
                f.write(f"  {tlabel} (t = {t[tidx]:.2f}):\n")
                f.write(f"    Peak amplitude: {peak:.4f}\n")
                f.write(f"    Peak position: {peak_pos:.4f}\n")
                f.write(f"    RMS amplitude: {rms:.4f}\n")
            
            f.write(f"  Interpretation: ")
            if row_idx == 0:
                f.write("Wave packet disperses and propagates.\n")
            elif row_idx == 1:
                f.write("Soliton profile remains unchanged.\n")
            elif row_idx == 2:
                f.write("Breather breathes with periodic modulation.\n")
            elif row_idx == 3:
                f.write("Solitons collide and separate.\n")
            f.write("\n")

    plt.tight_layout()
    
    # Save
    print("\nSaving figures...")
    plt.savefig(FIGS_DIR / 'fig2_envelope.pdf', transparent=False)
    plt.savefig(FIGS_DIR / 'fig2_envelope.png', transparent=False)
    plt.savefig(FIGS_DIR / 'fig2_envelope.eps', transparent=False)
    
    print("Figure 2 saved: ../figs/fig2_envelope.{pdf,png,eps}")
    print(f"Statistics: {stats_file}")
    plt.close()
    print("\nDone!")


if __name__ == '__main__':
    main()
