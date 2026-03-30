# Data Understanding: Jurkat Single-Cell Perturbation Dataset

This document summarizes the structure and experimental context of the `jurkat.h5` dataset based on initial exploration and analysis.

## 1. Overview
The dataset contains single-cell RNA sequencing (scRNA-seq) data from **Jurkat cells** (a human T lymphocyte cell line). The presence of a `target_gene` metadata field indicates that this is a **Perturb-seq** (or similar CRISPR-screen) experiment, where specific genes were knocked down to observe their effect on the cellular transcriptome.

- **Total Cells:** 21,412
- **Total Genes:** 18,080
- **Format:** HDF5 (structured similarly to an AnnData object)

## 2. Experimental Design (Perturbations)
The dataset includes **69 distinct treatment groups** tracked in the `obs/target_gene` column:
- **68 Gene Targets:** Includes key regulators such as:
    - **Chromatin Remodelers:** `SMARCB1`, `SMARCE1`, `HDAC3`, `KDM1A`.
    - **Transcriptional Machinery:** `MED1`, `MED12`, `MED24`, `TAF13`, `MAX`.
    - **Metabolic/Structural:** `ARPC2`, `ATP6V0B`, `HMGCR`, `MAT2A`.
- **1 Control Group:** Labeled `non-targeting`. This serves as the baseline for all differential expression analyses.

## 3. Data Structure (HDF5/AnnData Mapping)
| HDF5 Group | Mapping | Description |
| :--- | :--- | :--- |
| `X` | `adata.X` | Gene expression matrix (Normalised/Raw counts). |
| `obs` | `adata.obs` | Cell-level metadata (UMI counts, mitochondrial %, target gene). |
| `var` | `adata.var` | Gene-level metadata (Gene symbols/IDs). |
| `layers` | `adata.layers` | Alternative versions of the expression matrix. |
| `obsm` | `adata.obsm` | Multi-dimensional embeddings (e.g., PCA, UMAP). |

## 4. Key Metadata Fields
- **`UMI_count`**: Number of unique transcripts detected per cell; used to assess library complexity.
- **`mitopercent`**: Percentage of reads mapping to mitochondrial genes; used as a proxy for cell stress or damage.
- **`target_gene`**: The specific gene targeted by CRISPR in that cell.
- **`batch_var` / `gem_group`**: Technical variables indicating different experimental batches or microfluidic channels.

## 5. Visual Analysis Framework (Interpretation)
To evaluate the success and impact of the perturbations, we use five primary visualization strategies. **Note:** All visualisations (except basic UMAP) depend on first performing a global Differential Expression (DE) analysis against the `non-targeting` control to identify key regulated genes.

- **Magnitude of Effect (Bar Chart):** Measures the "earthquake level" of each perturbation. A high number of Significantly DE genes (Abs LFC > 1.0, Adj P-val < 0.05) indicates a "master regulator" gene that significantly alters the cell's internal state when removed.
- **Experimental Validation (Dotplot):** Acts as a "surgery success report." We look for reduced expression of the targeted gene within its own treatment group. If the targeted gene is absent or low on the diagonal, the CRISPR knockdown worked.
- **Quantitative Knockdown Spread (Violin & Strip Plots):** Evaluates the per-cell knockdown efficiency distribution. Because single-cell expression is highly stochastic (due to biological bursting and technical dropout), this visualization helps us understand the *spread* of the perturbation's effectiveness. Negative efficiency values represent cells stochastically expressing the target gene higher than the control mean, while the density at 100% indicates perfect or near-perfect gene silencing across the population.
- **Cellular Identity Shifts (UMAP):** Shows how far a perturbation pushes a cell away from its original "personality" (the control group). By highlighting the "Top 5" perturbations (those with the most DE genes), we can visualize which knockdowns cause the most dramatic transcriptomic shifts.
- **Regulatory Details (Matrixplot):** Provides a "damage map," showing exactly which top-ranked downstream genes are affected by specific upstream perturbations, helping to map biological pathways.

## 6. Critical Technical Insights
- **Log-Normalization Status:** Analysis of the `X` matrix in `jurkat.h5` reveals that the data is **already log-normalized**. 
    - **Evidence:** Values are non-integers, and the maximum value is characteristic of log-transformed data ($\sim 6.5-8.5$).
    - **Verification of Pre-normalization (CV Analysis):** To confirm the data was scaled before the log-transform, we calculate the **Linear Row Sums** (undoing the transform via `np.expm1(adata.X).sum(axis=1)`). 
        - **Results:** Mean $\approx 14178.57$, Std $\approx 783.35$.
        - **Interpretation:** While the standard deviation (783) might appear high, the **Coefficient of Variation (CV = Std/Mean)** is only **$\sim 5.5\%$**. In scRNA-seq, a CV this low indicates the cells have been successfully scaled to a nearly constant total count (likely 14,000). Minor fluctuations are typical due to subsequent gene filtering or floating-point precision during the $e^x - 1$ calculation.
    - **Action:** Standard normalization steps (`sc.pp.normalize_total` and `sc.pp.log1p`) **must be skipped** to avoid "double-normalization," which would distort the biological signal. 
- **HDF5 Indexing (`_index`):** The HDF5 format uses a specialized `_index` array within the `var` group to store the primary identifiers (gene symbols/IDs) for the 18,080 genes. When loading this data into a Pandas DataFrame or AnnData object, it is critical to use this array to set the dataframe's index and then *remove* the `_index` column from the dataframe itself. Failing to remove it will cause errors (e.g., `ValueError`) when attempting to save the object as an `.h5ad` file, as `_index` is a reserved internal keyword for AnnData storage.
- **Gene Preservation:** During preprocessing, standard "Highly Variable Gene" (HVG) selection may filter out the targeted genes themselves if they aren't naturally variable across the population. It is **mandatory** to manually add the 68 target genes back into the HVG list to ensure knockdown efficiency can be verified. Furthermore, the `processed_adata.h5ad` file preserves all **18,080 genes** to allow for comprehensive downstream analysis; the 3,000 selected HVGs are identified via the `adata.var['highly_variable']` boolean mask.
- **Control Referencing:** All statistical tests (like Wilcoxon rank-sum) must be performed relative to the `non-targeting` group to isolate the biological effect of the CRISPR guide from the baseline technical noise.
