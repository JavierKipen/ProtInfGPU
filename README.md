
# Protein Abundance Inference via Expectation Maximization in Fluorosequencing

This repository contains the code associated with the manuscript:

Kipen, J., Smith, M. B., Blom, T., Zhou, S. B., Marcotte, E. M., & Jaldén, J. (2025).  
**_Protein Abundance Inference via Expectation Maximization in Fluorosequencing_**.  
Submitted to *PLOS Computational Biology*.  
https://www.biorxiv.org/content/10.1101/2025.07.10.664057v1

## Overview

Fluorosequencing produces millions of short peptide-level fluorescence reads. This project implements a scalable and accurate framework for estimating protein abundances using Expectation Maximization (EM), based on posterior probabilities from peptide inference classifiers such as Probeam or Whatprot.

The repository includes:
- A simple Python (NumPy) implementation for five-protein synthetic experiments (CPU-based).
- A GPU-accelerated implementation for full-proteome scale datasets (written with CUDA/CuPy).
- Scripts to reproduce all experiments and generate the figures in the paper, under both standard and reduced-error simulation settings.

## Repository Structure

```
ProtInfGPU/
│
├── code/                  # Core EM implementations
│   ├── cuda/              # CUDA implementation for full proteome inference (optimized)
│   ├── cupy/              # CuPy implementation (simpler, less optimized GPU version)
│   ├── exporting/whatprot # Tools to extract and prepare datasets using Whatprot
│   └── numpy/             # NumPy-based EM implementation for 5-protein case (CPU)
│
├── ext/                   # External dependencies (Probeam and Whatprot clones)
├── scripts/               # Scripts to generate plots and run experiments
├── results/               # Saved outputs used to generate paper figures
└── README.md              # This file
```

## Reproducing Results

### Dataset Generation

Clone the dependencies in the `ext/` directory:

- [Probeam](https://github.com/JavierKipen/probeam), using the `probeam_sparsity_scores` branch.
- [Whatprot](https://github.com/marcottelab/whatprot), using the `publication-1` branch.

Data export path: `code/exporting/whatprot/`

#### Steps:

1. **Generate Sim Table**  
   `GenSimTable.py` — Maps fluorescence strings to proteins. > For the 5-protein setup, use `gen5Prot.py` to generate a tailored SimTable.

2. **Generate DyeSeqs**  
   `ExportDyeSeqs.py` — Creates DyeSeqs file required by Whatprot.

3. **Protein Info Table**  
   `genProtInfoTable.py` — Maps protein indices to names.

4. **Generate Equal-Representation Dataset**  
   Use Whatprot to generate uniform sample counts per fluorescence string. See `scripts/GenDatasets.ipynb`.

5. **Posterior Probabilities via Probeam**  
   Use Probeam to compute posterior estimates. See `scripts/GenDatasets.ipynb`.



## 5-Protein Experiment (CPU)

To reproduce all results for the 5-protein setup see scripts/5ProtAnalysis.ipynb

Internally uses:
- `code/numpy/ProteinInferenceEMv3.py`
- `code/numpy/WrapperEMCrossValv3.py`

## Whole Proteome (GPU)

1. Prepare datasets as described above.
2. Navigate to `code/cuda/`.
3. Follow instructions in `Notes.txt` to compile and run the EM algorithm on GPU.
4. Compatible with NVIDIA Tesla V100 GPUs.

---

## Reproducing figures

Once the results are obtained, scripts/20642ProtPlots.ipynb and scripts/5ProtAnalysis.ipynb generate the plots for the paper. 
