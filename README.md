# Supplementary Data Analysis Scripts: "`amangkurat`: A Python library for symplectic pseudo-spectral solution of the idealized (1+1)D nonlinear Klein-Gordon equation"

[![DOI](https://zenodo.org/badge/1097858665.svg)](https://doi.org/10.5281/zenodo.17625800)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains supplementary data analysis scripts for processing and visualizing outputs from the `amangkurat` solver.

## Contents

This repository includes Python scripts for:

- **Entropy Analysis** (`entropy_stats.py`): Computes Shannon, Rényi, Tsallis, and composite entropy metrics
- **Figure 1** (`fig1_spatiotemporal_evolution.py`): Spatiotemporal evolution heatmaps (2×2 grid)
- **Figure 2** (`fig2_wave_envelope_evolution.py`): Wave envelope evolution at three time snapshots (4×3 grid)
- **Figure 3** (`fig3_intensity_distributions.py`): Statistical distributions of field intensity
- **Figure 4** (`fig4_phase_space.py`): Phase space structure analysis (2×2 grid)

## Data Repository

All simulation outputs, including:
- GIF animations
- Computational logs of 4 test case experiments
- NetCDF output files
- Statistical calculations
- Generated figures

are publicly available on the **OSF Repository**: [https://doi.org/10.17605/OSF.IO/BKM2P](https://doi.org/10.17605/OSF.IO/BKM2P)

## The `amangkurat` Solver

The `amangkurat` solver is a Python library for symplectic pseudo-spectral solution of the idealized (1+1)D nonlinear Klein-Gordon equation.

- **Source Code**: [https://github.com/sandyherho/amangkurat](https://github.com/sandyherho/amangkurat)
- **PyPI Package**: [https://pypi.org/project/amangkurat/](https://pypi.org/project/amangkurat/)

### Installation
```bash
pip install amangkurat
```

## Usage

1. Download the NetCDF output files from the OSF repository
2. Place them in a `model_outputs/` directory
3. Run the analysis scripts:
```bash
python script/entropy_stats.py
python script/fig1_spatiotemporal_evolution.py
python script/fig2_wave_envelope_evolution.py
python script/fig3_intensity_distributions.py
python script/fig4_phase_space.py
```

## Dependencies

- numpy
- matplotlib
- netCDF4
- scipy
- pathlib

## Output

Scripts generate:
- Statistical summaries in `stats/` directory
- Publication-quality figures in `figs/` directory (PDF, PNG, EPS formats)

## License

Please refer to the main `amangkurat` repository for licensing information.

## Citation

If you use this code or data, please cite the associated publication and the OSF repository.
