"""
MLP baseline for Jurkat Perturb-seq prediction using the SAME data setup as the linear baseline.

This version is modified from the original MLP.py so that it matches the linear model dimensions:
- Use highly variable genes only: 3051 genes
- Input: control expression vector + perturbation one-hot vector
- Input dimension: 3051 + 68 = 3119
- Output: perturbed expression vector
- Output dimension: 3051

It also logs results to MLflow under experiment:
    jurkat_perturbation_prediction
"""

from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import mlflow


# -------------------------
# Hyperparameters
# -------------------------
SEED = 42
NUM_EPOCHS = 10          # match the linear baseline first
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

HIDDEN_DIM = 512
NUM_HIDDEN_LAYERS = 5
DROPOUT = 0.0            # original MLP had no dropout

CONTROL_LABEL = "non-targeting"
MIN_CELLS_PER_PERTURBATION = 60
VAL_RATIO = 0.10
TEST_TARGET_RATIO = 0.10
NUM_PERTURBATIONS = 68

EXPERIMENT_NAME = "jurkat_perturbation_prediction"
RUN_NAME = "mlp_hvg_3051_retrained"

MODEL_SAVE_PATH = Path("mlp_hvg_3051_final.pth")
BEST_MODEL_SAVE_PATH = Path("mlp_hvg_3051_best_val.pth")
CONFIG_SAVE_PATH = Path("mlp_hvg_3051_config.json")
EPOCH_LOSSES_PATH = Path("mlp_hvg_3051_epoch_losses.csv")

PROCESSED_DIR = Path("data") / "processed"


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
    """Find the processed AnnData file used by the linear model."""
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
    """Reproduce the linear model data setup exactly."""
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
    gene_to_idx = {gene: i + 1 for i, gene in enumerate(target_counts.index)}
    PertubatedData.obs["perturb_idx"] = PertubatedData.obs["target_gene"].map(gene_to_idx)

    high_count_genes = target_counts[target_counts > MIN_CELLS_PER_PERTURBATION]
    high_count_targets = set(high_count_genes.index)

    print("Number of perturbations:", len(target_counts))
    print("Perturbations with >60 cells:", len(high_count_genes))

    rng = np.random.default_rng(SEED)

    # Validation split, same logic as linear notebook
    high_count_mask = PertubatedData.obs["target_gene"].isin(high_count_targets)
    high_count_obs = PertubatedData.obs_names[high_count_mask]

    val_sample_size = max(1, int(np.ceil(len(high_count_obs) * VAL_RATIO)))
    sampled_obs = rng.choice(high_count_obs, size=val_sample_size, replace=False)
    ValidationData = PertubatedData[sampled_obs].copy()

    # Test split, same logic as linear notebook
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


def build_dataloaders(ControlData, TrainingData, ValidationData, TestData):
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
        control_matrix, train_matrix, train_idx,
        num_perturbations=NUM_PERTURBATIONS,
        train=True,
    )
    val_dataset = PerturbationDataset(
        control_matrix, val_matrix, val_idx,
        num_perturbations=NUM_PERTURBATIONS,
        train=False,
        fixed_control_indices=val_fixed,
    )
    test_dataset = PerturbationDataset(
        control_matrix, test_matrix, test_idx,
        num_perturbations=NUM_PERTURBATIONS,
        train=False,
        fixed_control_indices=test_fixed,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

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


# -------------------------
# Main
# -------------------------
def main():
    set_seed(SEED)

    data_path, ControlData, TrainingData, ValidationData, TestData = prepare_linear_style_data()
    train_loader, val_loader, test_loader = build_dataloaders(
        ControlData, TrainingData, ValidationData, TestData
    )

    input_dim = 3051 + NUM_PERTURBATIONS
    output_dim = 3051

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = MLPBaseline(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=HIDDEN_DIM,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    architecture_string = (
        f"{input_dim} -> "
        + " -> ".join([str(HIDDEN_DIM)] * NUM_HIDDEN_LAYERS)
        + f" -> {output_dim}"
    )
    print("MLP architecture:", architecture_string)

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(EXPERIMENT_NAME)

    epoch_history = []
    best_val_loss = float("inf")
    best_epoch = -1

    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.log_param("model_type", "mlp")
        mlflow.log_param("data_setup", "same_as_linear_hvg_3051")
        mlflow.log_param("data_path", str(data_path))
        mlflow.log_param("input_dim", input_dim)
        mlflow.log_param("output_dim", output_dim)
        mlflow.log_param("num_genes", 3051)
        mlflow.log_param("num_perturbations", NUM_PERTURBATIONS)
        mlflow.log_param("hidden_dim", HIDDEN_DIM)
        mlflow.log_param("num_hidden_layers", NUM_HIDDEN_LAYERS)
        mlflow.log_param("dropout", DROPOUT)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("num_epochs", NUM_EPOCHS)
        mlflow.log_param("seed", SEED)
        mlflow.log_param("loss_function", "MSE")
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("train_samples", len(train_loader.dataset))
        mlflow.log_param("val_samples", len(val_loader.dataset))
        mlflow.log_param("test_samples", len(test_loader.dataset))

        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss = evaluate(model, val_loader, criterion, device)

            epoch_history.append(
                {"epoch": epoch, "train_loss": float(train_loss), "val_loss": float(val_loss)}
            )

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_loss:.6f} | Validation Loss: {val_loss:.6f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save(model.state_dict(), BEST_MODEL_SAVE_PATH)

        test_loss = evaluate(model, test_loader, criterion, device)
        print(f"Test Loss: {test_loss:.6f}")

        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print("Final MLP model saved to", MODEL_SAVE_PATH)

        # Save test tensors for consistent re-evaluation later.
        x_test_path, y_test_path, X_test, y_test = save_test_tensors(test_loader)

        pd.DataFrame(epoch_history).to_csv(EPOCH_LOSSES_PATH, index=False)

        config = {
            "model_type": "MLPBaseline",
            "data_setup": "same_as_linear_hvg_3051",
            "input_dim": input_dim,
            "output_dim": output_dim,
            "num_genes": 3051,
            "num_perturbations": NUM_PERTURBATIONS,
            "hidden_dim": HIDDEN_DIM,
            "num_hidden_layers": NUM_HIDDEN_LAYERS,
            "dropout": DROPOUT,
            "architecture": architecture_string,
            "activation": "ReLU",
            "final_activation": None,
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "seed": SEED,
            "control_label": CONTROL_LABEL,
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
            "test_samples": len(test_loader.dataset),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "test_loss": test_loss,
        }

        with open(CONFIG_SAVE_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.log_metric("best_epoch", best_epoch)
        mlflow.log_metric("test_loss", test_loss)

        mlflow.log_artifact(str(MODEL_SAVE_PATH), artifact_path="model_checkpoint")
        mlflow.log_artifact(str(BEST_MODEL_SAVE_PATH), artifact_path="model_checkpoint")
        mlflow.log_artifact(str(EPOCH_LOSSES_PATH), artifact_path="metrics")
        mlflow.log_artifact(str(CONFIG_SAVE_PATH), artifact_path="config")
        mlflow.log_artifact(str(x_test_path), artifact_path="test_data")
        mlflow.log_artifact(str(y_test_path), artifact_path="test_data")

    print("Logged MLP HVG 3051 run to MLflow.")


if __name__ == "__main__":
    main()
