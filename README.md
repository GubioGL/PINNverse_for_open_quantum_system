# Inverse Physics-informed Neural Networks Procedure for Detecting Noise in Open Quantum Systems

This repository contains the code and supplementary materials for the article **"Inverse Physics-informed neural networks procedure for detecting noise in open quantum systems"**.

**arXiv Link:** https://arxiv.org/abs/2507.12552
---

## Overview

<p align="center">
  <img src="No%20fields/Predic_withfields_subplots_2x2.png" alt="PINN Predictions" width="700">
</p>

This work addresses the inverse problem of identifying noise parameters in open quantum systems using Physics-Informed Neural Networks (PINNs). The dynamics of open quantum systems are governed by the **Lindblad master equation**:

$$
\frac{d\rho}{dt} = -i[H, \rho] + \sum_k \gamma_k \left( L_k \rho L_k^\dagger - \frac{1}{2}\{L_k^\dagger L_k, \rho\} \right)
$$

where:
- $\rho$ is the density matrix of the quantum system
- $H$ is the Hamiltonian operator
- $L_k$ are the Lindblad operators describing different noise channels
- $\gamma_k$ are the decay rates (noise parameters to be estimated)
- $[\cdot, \cdot]$ denotes the commutator and $\{\cdot, \cdot\}$ the anticommutator

Our PINN approach learns to solve this equation while simultaneously estimating the unknown noise parameters $\gamma_k$ from measurement data.

---

## Repository Structure

This repository is organized into multiple folders, each corresponding to different quantum system configurations investigated in the article. The main structure contains:

### Problem-Specific Folders

The repository contains dedicated folders for each quantum system problem studied:

#### 1. **1 qubit/**
Contains implementations for single-qubit quantum systems.
- **CSV data files**: `sx.csv`, `sx_new.csv`, `sy.csv`, `sz.csv` - Store expectation values of Pauli operators for comparison.

#### 2. **No fields/**
Contains implementations for quantum systems without external fields. This folder includes:

- **`raw_data/`**: Stores the parameters found during simulation runs. Each CSV file contains the results from different seeds and noise levels (standard deviations).

- **`data_erro/`**: Contains error analysis files for different parameter configurations:
  - Files follow the naming convention: `erros_parametro_nofields_N50_seed{X}_std{Y}.csv`
  - Different seeds (1, 10, 11, 12, 100, etc.) and standard deviations (0, 0.02, 0.04, 0.06, 0.08, 0.1) are explored.

- **`modelo de script para o cluster/`**: Contains scripts (`job.sh`) that were executed in parallel on a computing cluster for large-scale simulations.

- **`function.py`**: Defines classes and helper functions required to execute the PINN (Physics-Informed Neural Network) models.

- **`Ode_equations_generator.ipynb`**: Jupyter notebook that generates the differential equations (Lindblad master equations) that the PINN will solve.

- **`version_pytorch.ipynb`**: Main training notebook where you can execute simulations for a single seed and visualize:
  - Complete training process
  - Loss function evolution
  - Parameter convergence
  - Model performance metrics

- **Jupyter notebooks for analysis and visualization**:
  - **`figure_3.ipynb`**: Generates Figure 3 from the article
  - **`figure_4.ipynb`**: Generates Figure 4 from the article
  - **`figure_Densidade_KDE.ipynb`**: Creates density plots using Kernel Density Estimation
  - **`analise_dos_resultados.ipynb`**: Comprehensive results analysis
  - **`analyzing_data.ipynb`**: Additional data analysis and statistics
  - **`data_erro_folder.ipynb`**: Error data processing and aggregation

#### 3. **With fields/**
Contains implementations for quantum systems with external fields applied. The folder structure mirrors the **No fields/** directory:

- **`raw_data/`**: Parameters found during simulations with external fields
- **`data_erro/`**: Error analysis for field-driven systems
- **`modelo de script para o cluster/`**: Cluster execution scripts for parallel simulations
- **`function.py`**: PINN model classes and utilities specific to systems with fields
- **`Ode_equations_generator.ipynb`**: Differential equation generator for field-driven dynamics
- **`version_pytorch.ipynb`**: Training and visualization notebook
- **`figure_1.py`**: Script to generate Figure 1 from the article
- **`figure_2.py`**: Script to generate Figure 2 from the article
- **`analise_dos_resultados.ipynb`**: Results analysis
- **`analyzing_data.ipynb`**: Data exploration and statistics
- **`data_erro_folder.ipynb`**: Error data processing

---

## Key Features

- **Physics-Informed Neural Networks (PINNs)**: Neural network architectures that incorporate physical laws (Lindblad equations) as constraints during training.
- **Inverse Problem Solving**: Estimate noise parameters in open quantum systems from measurement data.
- **Multiple System Configurations**: Investigate both field-free and field-driven quantum dynamics.
- **Robust Error Analysis**: Extensive testing across different noise levels and random seeds.
- **Reproducible Research**: All code, data, and analysis scripts needed to reproduce article results.

---

## Citation

If you use this code in your research, please cite our article:

```
arXiv: https://arxiv.org/abs/2507.12552
```

---

Feel free to explore the folders to understand the implementation and the experiments conducted in the context of inverse problems for open quantum systems.
