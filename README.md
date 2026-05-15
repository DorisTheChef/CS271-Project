# CS271 Project — Virtual Cell Challenge (Jurkat Perturb-seq)

Single-cell analysis and perturbation-response modeling for the [ARC Virtual Cell Challenge](https://virtualcellchallenge.org/) Jurkat Perturb-seq dataset. The project explores CRISPR knockdown effects in Jurkat T cells and trains baseline models to predict post-perturbation gene expression from control expression and perturbation identity.

## Dataset

| Property | Value |
|----------|-------|
| Cell line | Jurkat (human T lymphocyte) |
| Cells | 21,412 |
| Genes | 18,080 |
| Perturbations | 68 target genes + `non-targeting` control |
| Raw format | HDF5 (`jurkat.h5`), AnnData-like structure (`X`, `obs`, `var`) |

The raw `jurkat.h5` file is **not tracked in git** (see [Data setup](#data-setup)). For full experimental context, metadata fields, and preprocessing notes, see [`DATA_UNDERSTANDING.md`](DATA_UNDERSTANDING.md).

### Important preprocessing rules

- The `X` matrix is **already log-normalized** — do **not** run `sc.pp.normalize_total` or `sc.pp.log1p`.
- Use **3,051 highly variable genes** (`n_top_genes=3051`, `flavor='seurat'`), and **force all 68 target genes** into the HVG set.
- `processed_adata.h5ad` keeps the full **21,412 × 18,080** matrix; HVGs are marked in `adata.var['highly_variable']`.

## Repository structure

```
CS271-Project/
├── getdata.ipynb                 # Download / inspect raw jurkat.h5
├── DATA_UNDERSTANDING.md         # Dataset documentation
├── INSTRUCTIONS.md               # Agent / contributor conventions
│
├── zyt/                          # Exploratory Scanpy analysis
│   ├── data_analysis.ipynb       # Cell counts, preprocessing
│   ├── perturbation_analysis.ipynb
│   ├── perturbation_cell_counts.csv
│   └── STATE_for_Virtual_Cell_Challenge.ipynb
│
├── baseline/                     # Early linear baseline notebook
│   └── baseline.ipynb
│
├── Linear Model/                 # Linear baseline (HVG 3051 setup)
│   ├── LinearModel.ipynb
│   ├── linear_baseline_final.pt
│   └── model_epoch_005.pt
│
├── data/processed/               # Cached test tensors for evaluation
│   ├── X_test.pt
│   └── y_test.pt
│
├── MLP_HVG_3051_mlflow.py        # MLP training (HVG setup, MLflow)
├── MLP_HVG_3051_sweep_mlflow.py  # MLP hyperparameter sweep
├── Linear_HVG_3051_sweep_mlflow.py
├── evaluate_linear_mlflow.py     # Recompute / log linear test loss
├── knockdown_mlp_*.json/csv/pth  # Full-gene MLP experiment artifacts
├── mlp_hvg_3051_*.json/csv/pth   # HVG MLP experiment artifacts
└── workflow.jpg                  # End-to-end pipeline diagram
```

## Modeling approach

### Prediction task

Given a **control** expression profile and a **perturbation** (one-hot over 68 genes), predict the **perturbed** expression profile. Loss is **MSE** between predicted and observed expression.

### HVG baseline (primary comparison setup)

Used by the linear notebook and the `*_HVG_3051_*` scripts:

| Component | Dimension |
|-----------|-----------|
| Input (HVG expression + perturbation one-hot) | 3,051 + 68 = **3,119** |
| Output (perturbed HVG expression) | **3,051** |

**Linear:** single `nn.Linear(3119 → 3051)`  
**MLP:** 5 hidden layers of 512 units, ReLU (`3119 → 512 → … → 3051`)

### Full-gene MLP (`knockdown_mlp_*`)

Alternative architecture over all genes (~18k input/output). See `knockdown_mlp_config.json` for architecture and target-gene list.

## Setup

### Environment

Use the project virtual environment at the repo root:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

Core dependencies: `h5py`, `numpy`, `pandas`, `scanpy`, `torch`, `mlflow`, `matplotlib`, `seaborn`.

### Data setup

1. Obtain `jurkat.h5` from the Virtual Cell Challenge / ARC source (see the bundled data-source PDF in the repo).
2. Place it in the project root as `jurkat.h5`.
3. Run preprocessing notebooks in `zyt/` to produce `processed_adata.h5ad` (or copy to repo root — training scripts search multiple paths).

Files ignored by git (local only):

- `*.h5`, `*.h5ad` — raw and processed AnnData
- `mlruns/`, `outputs/` — MLflow and run artifacts

## Running experiments

Activate `.venv`, ensure `processed_adata.h5ad` is available, then:

```bash
# Train MLP on HVG setup (logs to ./mlruns)
python MLP_HVG_3051_mlflow.py

# Hyperparameter sweeps (linear or MLP)
python Linear_HVG_3051_sweep_mlflow.py
python MLP_HVG_3051_sweep_mlflow.py

# Evaluate saved linear checkpoint on test set
python evaluate_linear_mlflow.py
```

MLflow experiment name: `jurkat_perturbation_prediction`.  
View runs locally:

```bash
mlflow ui --backend-store-uri file:./mlruns
```

### Saved artifacts (in repo)

| File | Description |
|------|-------------|
| `Linear Model/linear_baseline_final.pt` | Linear model weights |
| `mlp_hvg_3051_final.pth` / `mlp_hvg_3051_best_val.pth` | MLP checkpoints |
| `mlp_hvg_3051_config.json` | MLP hyperparameters and test metrics |
| `knockdown_mlp_model.pth` | Full-gene MLP weights |
| `zyt/perturbation_cell_counts.csv` | Cells per `target_gene` |
| `workflow.jpg` | Pipeline overview |

## Analysis workflow

1. **Load & QC** — `getdata.ipynb`, `zyt/data_analysis.ipynb`
2. **Preprocess** — HVG selection, PCA/UMAP, force target genes into HVGs → `processed_adata.h5ad`
3. **Explore perturbations** — `zyt/perturbation_analysis.ipynb` (DE, UMAP, knockdown validation)
4. **Train baselines** — `Linear Model/LinearModel.ipynb` or Python training scripts
5. **Track experiments** — MLflow sweeps and metric CSVs

Regenerate the workflow figure:

```bash
python create_workflow_diagram.py
```

## Contributing

- Never commit `jurkat.h5` or other `*.h5` / `*.h5ad` files.
- Follow conventions in [`INSTRUCTIONS.md`](INSTRUCTIONS.md) when editing notebooks or preprocessing code.
- Large checkpoints (>50 MB) may trigger GitHub warnings; consider Git LFS for future model files.

## References

- [`DATA_UNDERSTANDING.md`](DATA_UNDERSTANDING.md) — dataset structure, normalization verification, visualization framework
- [`INSTRUCTIONS.md`](INSTRUCTIONS.md) — preprocessing mandates and HVG strategy
- `Data Source Jurkat Perturb-seq (ARC Virtual Cell Challenge) jurkat.h5 21,412 cells x 18,080 genes.pdf` — data provenance

## License

Course project for CS271 (SJSU). Dataset terms follow the ARC Virtual Cell Challenge. Submodule `zyt/state/` has its own license — see `zyt/state/LICENSE`.
