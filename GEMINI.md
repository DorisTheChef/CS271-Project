# GEMINI Mandates

This file provides foundational instructions and context for Gemini CLI when working on the CS271-Project.

## Project Overview
This project is for CS271 and involves analyzing single-cell data, specifically related to the "Virtual Cell Challenge." The primary data format used is HDF5 (`.h5`).

## Critical Mandates
- **Data Protection:** The file `jurkat.h5` is ignored by `.gitignore` and must never be committed or staged.
- **HDF5 Handling:** Use `h5py` or `pandas` for reading `.h5` files. Always verify the structure of the H5 file before processing, as it typically contains groups like `X`, `obs`, and `var`.
- **Notebook Development:** When modifying notebooks, maintain the experimental and iterative nature of the analysis while ensuring code blocks are well-documented.
- **Normalization:** The `X` matrix in `jurkat.h5` is already log-normalized. Skip `sc.pp.normalize_total` and `sc.pp.log1p` to prevent "double-normalization." Verification shows a Linear Row Sum mean of ~14,178 with a low CV (~5.5%), confirming pre-normalization.
## HVG Strategy: 
    - Prefer `n_top_genes` over hard thresholds (`min_disp`, `min_mean`) for stable feature counts.
    - **Selection:** Use `n_top_genes=3000` with `flavor='seurat'` to ensure sufficient biological signal.
    - **Storage:** The `processed_adata.h5ad` file preserves the **full 21412 x 18080 dimensions**. The 3000 selected genes are identified via the `adata.var['highly_variable']` boolean mask.
    - **Metrics:** `means` (average expression) and `dispersions` (variation relative to mean) are calculated in bins. `dispersions_norm` is the ranking criteria.
    - **Target Genes:** Always force target genes into the HVG list to ensure perturbation effects are captured in PCA/UMAP.

## Environment
- Use the `.venv` virtual environment located in the root directory.
- Dependencies include `h5py`, `numpy`, and standard data science libraries.
