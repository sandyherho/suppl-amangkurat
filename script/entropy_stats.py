#!/usr/bin/env python
"""Entropy analysis: Shannon, Rényi, Tsallis + Composite metric."""

import numpy as np
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


def compute_probability(phi, x):
    """Compute normalized probability p(x,t) = |phi|^2 / integral."""
    phi2 = phi**2
    norm = np.trapz(phi2, x, axis=1)[:, np.newaxis]
    # Avoid division by zero
    norm = np.where(norm > 1e-15, norm, 1.0)
    return phi2 / norm


def shannon_entropy(p):
    """Shannon entropy S = -sum(p * log(p))."""
    p_pos = p[p > 1e-15]
    if len(p_pos) == 0:
        return 0.0
    return -np.sum(p_pos * np.log(p_pos))


def renyi_entropy(p, alpha):
    """Rényi entropy H_α = log(sum(p^α)) / (1 - α)."""
    if abs(alpha - 1.0) < 1e-10:
        return shannon_entropy(p)
    p_pos = p[p > 1e-15]
    if len(p_pos) == 0:
        return 0.0
    # Avoid division by zero or invalid operations
    sum_p_alpha = np.sum(p_pos**alpha)
    if sum_p_alpha <= 0:
        return 0.0
    return np.log(sum_p_alpha) / (1 - alpha)


def tsallis_entropy(p, q):
    """Tsallis entropy S_q = (1 - sum(p^q)) / (q - 1)."""
    if abs(q - 1.0) < 1e-10:
        return shannon_entropy(p)
    p_pos = p[p > 1e-15]
    if len(p_pos) == 0:
        return 0.0
    return (1 - np.sum(p_pos**q)) / (q - 1)


def composite_entropy(p):
    """
    Composite entropy metric: weighted combination.
    C = 0.5*S + 0.3*H_2 + 0.2*S_2
    Combines spatial spread (Shannon), concentration (Rényi), 
    and non-extensivity (Tsallis).
    """
    S = shannon_entropy(p)
    H2 = renyi_entropy(p, 2.0)
    S2 = tsallis_entropy(p, 2.0)
    return 0.5 * S + 0.3 * H2 + 0.2 * S2


def main():
    # Cases
    CASES = [
        ('case_1_linear_wave.nc', 'Linear Wave'),
        ('case_2_kink_soliton.nc', 'Kink Soliton'),
        ('case_3_breather.nc', 'Breather'),
        ('case_4_kink_antikink_collision.nc', 'Collision')
    ]

    # Create output directory
    STATS_DIR = Path('../stats')
    STATS_DIR.mkdir(exist_ok=True)

    # Statistics file
    stats_file = STATS_DIR / 'entropy_analysis.txt'
    with open(stats_file, 'w') as f:
        f.write("ENTROPY ANALYSIS: SHANNON, RÉNYI, TSALLIS, COMPOSITE\n")
        f.write("=" * 70 + "\n\n")

    for idx, (filename, label) in enumerate(CASES):
        # Load data
        x, t, phi = load_data(filename)
        prob = compute_probability(phi, x)
        
        # Compute entropy time series
        S_shannon = np.array([shannon_entropy(p) for p in prob])
        H_renyi0 = np.array([renyi_entropy(p, 0.5) for p in prob])  # α=0.5
        H_renyi2 = np.array([renyi_entropy(p, 2.0) for p in prob])  # Collision
        H_renyi_inf = np.array([renyi_entropy(p, 10.0) for p in prob])  # Approx ∞
        S_tsallis05 = np.array([tsallis_entropy(p, 0.5) for p in prob])
        S_tsallis2 = np.array([tsallis_entropy(p, 2.0) for p in prob])
        C_composite = np.array([composite_entropy(p) for p in prob])
        
        # Write statistics
        with open(stats_file, 'a') as f:
            f.write(f"Case {idx+1}: {label}\n")
            f.write(f"  Time range: t ∈ [{t.min():.2f}, {t.max():.2f}]\n")
            f.write(f"\n")
            
            f.write(f"  Shannon Entropy S:\n")
            f.write(f"    Mean: {S_shannon.mean():.4f}\n")
            f.write(f"    Std: {S_shannon.std():.4f}\n")
            f.write(f"    Range: [{S_shannon.min():.4f}, {S_shannon.max():.4f}]\n")
            f.write(f"    Initial: {S_shannon[0]:.4f}\n")
            f.write(f"    Final: {S_shannon[-1]:.4f}\n")
            f.write(f"\n")
            
            f.write(f"  Rényi Entropy H₀.₅:\n")
            f.write(f"    Mean: {H_renyi0.mean():.4f}\n")
            f.write(f"    Range: [{H_renyi0.min():.4f}, {H_renyi0.max():.4f}]\n")
            f.write(f"\n")
            
            f.write(f"  Rényi Entropy H₂ (Collision):\n")
            f.write(f"    Mean: {H_renyi2.mean():.4f}\n")
            f.write(f"    Std: {H_renyi2.std():.4f}\n")
            f.write(f"    Range: [{H_renyi2.min():.4f}, {H_renyi2.max():.4f}]\n")
            f.write(f"\n")
            
            f.write(f"  Rényi Entropy H₁₀ (approx H_∞):\n")
            f.write(f"    Mean: {H_renyi_inf.mean():.4f}\n")
            f.write(f"    Range: [{H_renyi_inf.min():.4f}, {H_renyi_inf.max():.4f}]\n")
            f.write(f"\n")
            
            f.write(f"  Tsallis Entropy S₀.₅:\n")
            f.write(f"    Mean: {S_tsallis05.mean():.4f}\n")
            f.write(f"    Range: [{S_tsallis05.min():.4f}, {S_tsallis05.max():.4f}]\n")
            f.write(f"\n")
            
            f.write(f"  Tsallis Entropy S₂:\n")
            f.write(f"    Mean: {S_tsallis2.mean():.4f}\n")
            f.write(f"    Std: {S_tsallis2.std():.4f}\n")
            f.write(f"    Range: [{S_tsallis2.min():.4f}, {S_tsallis2.max():.4f}]\n")
            f.write(f"\n")
            
            f.write(f"  Composite Entropy C = 0.5*S + 0.3*H₂ + 0.2*S₂:\n")
            f.write(f"    Mean: {C_composite.mean():.4f}\n")
            f.write(f"    Std: {C_composite.std():.4f}\n")
            f.write(f"    Range: [{C_composite.min():.4f}, {C_composite.max():.4f}]\n")
            f.write(f"    Initial: {C_composite[0]:.4f}\n")
            f.write(f"    Final: {C_composite[-1]:.4f}\n")
            f.write(f"\n")
            
            f.write(f"  Interpretation: ")
            if idx == 0:
                f.write("Linear wave: moderate, stable entropy.\n")
            elif idx == 1:
                f.write("Kink: low entropy (highly coherent structure).\n")
            elif idx == 2:
                f.write("Breather: periodic entropy oscillations.\n")
            elif idx == 3:
                f.write("Collision: complex entropy dynamics.\n")
            
            f.write("\n" + "-" * 70 + "\n\n")

    # Comparative summary
    with open(stats_file, 'a') as f:
        f.write("COMPARATIVE SUMMARY:\n")
        f.write("\n")
        f.write("Shannon Entropy (S):\n")
        f.write("  Quantifies spatial information spread.\n")
        f.write("  Higher S → more delocalized field.\n")
        f.write("\n")
        f.write("Rényi Entropies (H_α):\n")
        f.write("  H₀.₅: emphasizes rare events\n")
        f.write("  H₂: focuses on high-probability regions (collision entropy)\n")
        f.write("  H₁₀: approximates min-entropy (maximum probability)\n")
        f.write("\n")
        f.write("Tsallis Entropies (S_q):\n")
        f.write("  Capture non-extensive statistics.\n")
        f.write("  Useful for systems with long-range correlations.\n")
        f.write("\n")
        f.write("Composite Entropy (C):\n")
        f.write("  Weighted combination: C = 0.5*S + 0.3*H₂ + 0.2*S₂\n")
        f.write("  Balances spatial spread, concentration, non-extensivity.\n")
        f.write("  Single metric for overall complexity comparison.\n")
        f.write("\n")
        f.write("Physical Interpretation:\n")
        f.write("  Lower entropy → coherent, localized structures (kink)\n")
        f.write("  Higher entropy → spread-out, dispersed fields (linear wave)\n")
        f.write("  Oscillating entropy → periodic dynamics (breather)\n")
        f.write("  Complex entropy evolution → nonlinear interactions (collision)\n")

    print(f"Entropy statistics saved: {stats_file}")


if __name__ == '__main__':
    main()
