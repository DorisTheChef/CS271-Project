"""
MLP hyperparameter sweep for Jurkat Perturb-seq prediction.

This script uses the SAME data setup as the linear baseline:
- Highly variable genes only: 3051 genes
- Input: control expression vector + perturbation one-hot vector
- Input dimension: 3051 + 68 = 3119
- Output: perturbed expression vector
- Output dimension: 3051

Hyperparameter sweep:
- learning_rate: [1e-4, 1e-3, 1e-2]
- hidden_dim: [128, 256, 512]
- dropout: [0.0, 0.2]

Fixed:
- batch_size = 64
- optimizer = Adam
- loss = MSELoss
- epochs = 10

All runs are logged to MLflow under:
    jurkat_perturbation_prediction
"""

from pathlib import Path
import json
import itertools

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import mlflow


# -------------------------
# Fixed settings
# -------------------------
SEED = 42
NUM_EPOCHS = 10
BATCH_SIZE = 64

CONTROL_LABEL = "non-targeting"
MIN_CELLS_PER_PERTURBATION = 60
VAL_RATIO = 0.10
TEST_TARGET_RATIO = 0.10
NUM_PERTURBATIONS = 68

INPUT_DIM = 3119
OUTPUT_DIM = 3051
NUM_GENES = 3051
NUM_HIDDEN_LAYERS = 5

EXPERIMENT_NAME = "jurkat_perturbation_prediction"

PROCESSED_DIR = Path("data") / "processed"
SWEEP_OUTPUT_DIR = Path("outputs") / "mlp_sweep"
SWEEP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Sweep levels
# -------------------------
LEARNING_RATES = [1e-4, 1e-3, 1e-2]
HIDDEN_DIMS = [128, 256, 512]
DROPOUTS = [0.0, 0.2]


# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------
# Data path
# -------------------------
def find_data_path() -> Path:
    candidates = [
        Path("processed_adata.h5ad"),
        Path("processed_data.h5ad"),
        Path("Linear Model") / "processed_adata.h5ad",
        Path("Linear Model") / "processed_data.h5ad",
        Path("baseline") / "processed_adata.h5ad",
        Path("baseline") / "processed_data.h5ad",
        Path("zyt") / "processed_adata.h5ad",
        Path("zyt") / "processed_data.h5ad",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find processed_adata.h5ad or processed_data.h5ad. "
        "Put this script in the project root or update find_data_path()."
    )


# -------------------------
# Dataset
# -------------------------
class PerturbationDataset(Dataset):
    def __init__(
        self,
        control_matrix,
        target_matrix,
        perturb_idx,
        num_perturbations=68,
        train=True,
        fixed_control_indices=None,
    ):
        self.control_matrix = torch.tensor(control_matrix, dtype=torch.float32)
        self.target_matrix = torch.tensor(target_matrix, dtype=torch.float32)
        self.perturb_idx = torch.tensor(perturb_idx, dtype=torch.long)
        self.num_perturbations = num_perturbations
        self.train = train
        self.fixed_control_indices = fixed_control_indices

        if (not train) and (fixed_control_indices is None):
            raise ValueError("Validation/Test dataset needs fixed_control_indices.")

    def __len__(self):
        return len(self.target_matrix)

    def __getitem__(self, idx):
        target = self.target_matrix[idx]
        p_idx = self.perturb_idx[idx].item()

        one_hot = torch.zeros(self.num_perturbations, dtype=torch.float32)
        one_hot[p_idx] = 1.0

        if self.train:
            control_idx = np.random.randint(0, len(self.control_matrix))
        else:
            control_idx = self.fixed_control_indices[idx]

        control = self.control_matrix[control_idx]
        x = torch.cat([control, one_hot], dim=0)

        return x, target


# -------------------------
# MLP model
# -------------------------
class MLPBaseline(nn.Module):
    def __init__(
        self,
        input_dim=3119,
        output_dim=3051,
        hidden_dim=512,
        num_hidden_layers=5,
        dropout=0.0,
    ):
        super().__init__()

        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be at least 1")

        layers = []

        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# -------------------------
# Helpers
# -------------------------
def to_numpy_matrix(adata):
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def prepare_linear_style_data():
    """
    Reproduce the same data setup used by the linear baseline:
    - HVG only
    - control cells as input reference
    - perturbed cells as target
    - validation/test split based on perturbation groups
    """
    data_path = find_data_path()
    print("Loading data from:", data_path)

    adata = sc.read_h5ad(data_path)
    print("Original adata shape:", adata.shape)

    if "highly_variable" not in adata.var:
        raise KeyError("adata.var must contain 'highly_variable'.")
    if "target_gene" not in adata.obs:
        raise KeyError("adata.obs must contain 'target_gene'.")

    adata = adata[:, adata.var["highly_variable"]].copy()
    print("HVG adata shape:", adata.shape)

    ControlData = adata[adata.obs["target_gene"] == CONTROL_LABEL].copy()
    PertubatedData = adata[adata.obs["target_gene"] != CONTROL_LABEL].copy()

    print("ControlData observations:", ControlData.n_obs)
    print("PertubatedData observations:", PertubatedData.n_obs)

    target_counts = PertubatedData.obs["target_gene"].value_counts().sort_values(ascending=False)

    # 1-based index first, then later convert to 0-based.
    gene_to_idx = {gene: i + 1 for i, gene in enumerate(target_counts.index)}
    PertubatedData.obs["perturb_idx"] = PertubatedData.obs["target_gene"].map(gene_to_idx)

    high_count_genes = target_counts[target_counts > MIN_CELLS_PER_PERTURBATION]
    high_count_targets = set(high_count_genes.index)

    print("Number of perturbations:", len(target_counts))
    print("Perturbations with >60 cells:", len(high_count_genes))

    rng = np.random.default_rng(SEED)

    # Validation split
    high_count_mask = PertubatedData.obs["target_gene"].isin(high_count_targets)
    high_count_obs = PertubatedData.obs_names[high_count_mask]

    val_sample_size = max(1, int(np.ceil(len(high_count_obs) * VAL_RATIO)))
    sampled_obs = rng.choice(high_count_obs, size=val_sample_size, replace=False)
    ValidationData = PertubatedData[sampled_obs].copy()

    # Test split
    rest_data = PertubatedData[~PertubatedData.obs_names.isin(ValidationData.obs_names)].copy()
    rest_high_count_targets = list(high_count_targets)

    n_test_targets = max(1, int(np.ceil(len(rest_high_count_targets) * TEST_TARGET_RATIO)))
    test_targets = rng.choice(rest_high_count_targets, size=n_test_targets, replace=False)

    test_mask = rest_data.obs["target_gene"].isin(test_targets)
    TestData = rest_data[test_mask].copy()

    train_mask = ~PertubatedData.obs_names.isin(
        TestData.obs_names.union(ValidationData.obs_names)
    )
    TrainingData = PertubatedData[train_mask].copy()

    print("TrainingData shape:", TrainingData.shape)
    print("ValidationData shape:", ValidationData.shape)
    print("TestData shape:", TestData.shape)
    print("Selected test targets:", list(map(str, test_targets)))

    return data_path, ControlData, TrainingData, ValidationData, TestData


def build_dataloaders(ControlData, TrainingData, ValidationData, TestData, batch_size):
    np.random.seed(SEED)

    control_matrix = to_numpy_matrix(ControlData)
    train_matrix = to_numpy_matrix(TrainingData)
    val_matrix = to_numpy_matrix(ValidationData)
    test_matrix = to_numpy_matrix(TestData)

    train_idx = np.asarray(TrainingData.obs["perturb_idx"].values, dtype=np.int64)
    val_idx = np.asarray(ValidationData.obs["perturb_idx"].values, dtype=np.int64)
    test_idx = np.asarray(TestData.obs["perturb_idx"].values, dtype=np.int64)

    # Linear notebook used 1~68, convert to 0~67.
    if train_idx.min() == 1:
        train_idx = train_idx - 1
        val_idx = val_idx - 1
        test_idx = test_idx - 1

    val_fixed = np.random.randint(0, len(control_matrix), size=len(val_matrix))
    test_fixed = np.random.randint(0, len(control_matrix), size=len(test_matrix))

    train_dataset = PerturbationDataset(
        control_matrix,
        train_matrix,
        train_idx,
        num_perturbations=NUM_PERTURBATIONS,
        train=True,
    )

    val_dataset = PerturbationDataset(
        control_matrix,
        val_matrix,
        val_idx,
        num_perturbations=NUM_PERTURBATIONS,
        train=False,
        fixed_control_indices=val_fixed,
    )

    test_dataset = PerturbationDataset(
        control_matrix,
        test_matrix,
        test_idx,
        num_perturbations=NUM_PERTURBATIONS,
        train=False,
        fixed_control_indices=test_fixed,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = criterion(pred, y)

            total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


def save_test_tensors(test_loader):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_x = []
    all_y = []

    for x_batch, y_batch in test_loader:
        all_x.append(x_batch.cpu())
        all_y.append(y_batch.cpu())

    X_test = torch.cat(all_x, dim=0)
    y_test = torch.cat(all_y, dim=0)

    x_path = PROCESSED_DIR / "X_test.pt"
    y_path = PROCESSED_DIR / "y_test.pt"

    torch.save(X_test, x_path)
    torch.save(y_test, y_path)

    print("Saved X_test:", X_test.shape, "to", x_path)
    print("Saved y_test:", y_test.shape, "to", y_path)

    return x_path, y_path, X_test, y_test


def run_one_experiment(
    learning_rate,
    hidden_dim,
    dropout,
    data_path,
    train_loader,
    val_loader,
    test_loader,
    device,
):
    set_seed(SEED)

    run_name = f"mlp_sweep_lr{learning_rate}_hidden{hidden_dim}_dropout{dropout}"

    model = MLPBaseline(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        hidden_dim=hidden_dim,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        dropout=dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    architecture_string = (
        f"{INPUT_DIM} -> "
        + " -> ".join([str(hidden_dim)] * NUM_HIDDEN_LAYERS)
        + f" -> {OUTPUT_DIM}"
    )

    safe_lr = str(learning_rate).replace(".", "p").replace("-", "m")
    safe_dropout = str(dropout).replace(".", "p")
    run_file_prefix = f"mlp_lr{safe_lr}_hidden{hidden_dim}_dropout{safe_dropout}"

    model_save_path = SWEEP_OUTPUT_DIR / f"{run_file_prefix}_final.pth"
    best_model_save_path = SWEEP_OUTPUT_DIR / f"{run_file_prefix}_best_val.pth"
    epoch_losses_path = SWEEP_OUTPUT_DIR / f"{run_file_prefix}_epoch_losses.csv"
    config_save_path = SWEEP_OUTPUT_DIR / f"{run_file_prefix}_config.json"

    epoch_history = []
    best_val_loss = float("inf")
    best_epoch = -1

    print("\n" + "=" * 80)
    print("Starting run:", run_name)
    print("Architecture:", architecture_string)
    print("=" * 80)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_type", "mlp")
        mlflow.log_param("run_group", "mlp_hvg_3051_hyperparameter_sweep")
        mlflow.log_param("data_setup", "same_as_linear_hvg_3051")
        mlflow.log_param("data_path", str(data_path))

        mlflow.log_param("input_dim", INPUT_DIM)
        mlflow.log_param("output_dim", OUTPUT_DIM)
        mlflow.log_param("num_genes", NUM_GENES)
        mlflow.log_param("num_perturbations", NUM_PERTURBATIONS)

        mlflow.log_param("hidden_dim", hidden_dim)
        mlflow.log_param("num_hidden_layers", NUM_HIDDEN_LAYERS)
        mlflow.log_param("dropout", dropout)
        mlflow.log_param("learning_rate", learning_rate)

        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("num_epochs", NUM_EPOCHS)
        mlflow.log_param("seed", SEED)
        mlflow.log_param("loss_function", "MSELoss")
        mlflow.log_param("optimizer", "Adam")

        mlflow.log_param("train_samples", len(train_loader.dataset))
        mlflow.log_param("val_samples", len(val_loader.dataset))
        mlflow.log_param("test_samples", len(test_loader.dataset))

        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss = evaluate(model, val_loader, criterion, device)

            epoch_history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                }
            )

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

            print(
                f"{run_name} | Epoch {epoch:03d} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Validation Loss: {val_loss:.6f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save(model.state_dict(), best_model_save_path)

        test_loss = evaluate(model, test_loader, criterion, device)

        torch.save(model.state_dict(), model_save_path)

        pd.DataFrame(epoch_history).to_csv(epoch_losses_path, index=False)

        config = {
            "model_type": "MLPBaseline",
            "run_group": "mlp_hvg_3051_hyperparameter_sweep",
            "data_setup": "same_as_linear_hvg_3051",
            "input_dim": INPUT_DIM,
            "output_dim": OUTPUT_DIM,
            "num_genes": NUM_GENES,
            "num_perturbations": NUM_PERTURBATIONS,
            "hidden_dim": hidden_dim,
            "num_hidden_layers": NUM_HIDDEN_LAYERS,
            "dropout": dropout,
            "architecture": architecture_string,
            "activation": "ReLU",
            "final_activation": None,
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": learning_rate,
            "optimizer": "Adam",
            "loss_function": "MSELoss",
            "seed": SEED,
            "control_label": CONTROL_LABEL,
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
            "test_samples": len(test_loader.dataset),
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
            "test_loss": float(test_loss),
        }

        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.log_metric("best_epoch", best_epoch)
        mlflow.log_metric("test_loss", test_loss)

        mlflow.log_artifact(str(model_save_path), artifact_path="model_checkpoint")
        mlflow.log_artifact(str(best_model_save_path), artifact_path="model_checkpoint")
        mlflow.log_artifact(str(epoch_losses_path), artifact_path="metrics")
        mlflow.log_artifact(str(config_save_path), artifact_path="config")

    print("Finished run:", run_name)
    print(f"Best Val Loss: {best_val_loss:.6f}")
    print(f"Test Loss: {test_loss:.6f}")

    return {
        "run_name": run_name,
        "learning_rate": learning_rate,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "test_loss": test_loss,
    }


# -------------------------
# Main
# -------------------------
def main():
    set_seed(SEED)

    data_path, ControlData, TrainingData, ValidationData, TestData = prepare_linear_style_data()

    train_loader, val_loader, test_loader = build_dataloaders(
        ControlData,
        TrainingData,
        ValidationData,
        TestData,
        batch_size=BATCH_SIZE,
    )

    # Save the test tensors once for consistent re-evaluation.
    x_test_path, y_test_path, X_test, y_test = save_test_tensors(test_loader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(EXPERIMENT_NAME)

    all_results = []

    sweep_combinations = list(itertools.product(LEARNING_RATES, HIDDEN_DIMS, DROPOUTS))

    print("\nTotal sweep runs:", len(sweep_combinations))
    print("Learning rates:", LEARNING_RATES)
    print("Hidden dims:", HIDDEN_DIMS)
    print("Dropouts:", DROPOUTS)

    for learning_rate, hidden_dim, dropout in sweep_combinations:
        result = run_one_experiment(
            learning_rate=learning_rate,
            hidden_dim=hidden_dim,
            dropout=dropout,
            data_path=data_path,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
        )
        all_results.append(result)

    results_df = pd.DataFrame(all_results)
    results_path = SWEEP_OUTPUT_DIR / "mlp_hvg_3051_sweep_summary.csv"
    results_df.to_csv(results_path, index=False)

    best_row = results_df.sort_values("test_loss", ascending=True).iloc[0]

    print("\n" + "=" * 80)
    print("SWEEP FINISHED")
    print("=" * 80)
    print("Best run by test_loss:")
    print(best_row)
    print("\nSummary saved to:", results_path)

    # Log sweep summary as a separate MLflow run.
    with mlflow.start_run(run_name="mlp_hvg_3051_sweep_summary"):
        mlflow.log_param("run_group", "mlp_hvg_3051_hyperparameter_sweep_summary")
        mlflow.log_param("num_runs", len(results_df))
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("num_epochs", NUM_EPOCHS)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("loss_function", "MSELoss")

        mlflow.log_metric("best_test_loss", float(best_row["test_loss"]))
        mlflow.log_metric("best_val_loss", float(best_row["best_val_loss"]))

        mlflow.log_param("best_learning_rate", float(best_row["learning_rate"]))
        mlflow.log_param("best_hidden_dim", int(best_row["hidden_dim"]))
        mlflow.log_param("best_dropout", float(best_row["dropout"]))

        mlflow.log_artifact(str(results_path), artifact_path="sweep_summary")
        mlflow.log_artifact(str(x_test_path), artifact_path="test_data")
        mlflow.log_artifact(str(y_test_path), artifact_path="test_data")

    print("Logged sweep summary to MLflow.")


if __name__ == "__main__":
    main()