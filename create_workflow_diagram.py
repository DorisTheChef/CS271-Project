import matplotlib.pyplot as plt
from matplotlib.patches import (
    FancyBboxPatch,
    FancyArrowPatch,
    Ellipse,
    Rectangle,
)
from pathlib import Path


OUTPUT_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "jurkat_mlflow_workflow_polished.png"


# -------------------------
# Drawing helpers
# -------------------------
def add_round_box(
    ax,
    x,
    y,
    w,
    h,
    title,
    body="",
    facecolor="#f7fbff",
    edgecolor="#4a90e2",
    title_color="black",
    title_size=10,
    body_size=8,
    linewidth=1.5,
    dashed=False,
    align="center",
):
    linestyle = "--" if dashed else "-"

    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=0.06",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        linestyle=linestyle,
    )
    ax.add_patch(box)

    ax.text(
        x + w / 2,
        y + h - 0.16,
        title,
        ha="center",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color=title_color,
    )

    if body:
        if align == "center":
            body_x = x + w / 2
            ha = "center"
        else:
            body_x = x + 0.18
            ha = "left"

        ax.text(
            body_x,
            y + h - 0.46,
            body,
            ha=ha,
            va="top",
            fontsize=body_size,
            color="black",
            linespacing=1.35,
        )


def add_step_box(ax, x, y, w, h, text):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        linewidth=1.1,
        edgecolor="#6aa6e8",
        facecolor="#eaf4ff",
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=8.3,
        color="#0b2545",
    )


def add_arrow(ax, start, end, color="black", lw=1.5, dashed=False):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=lw,
        color=color,
        linestyle="--" if dashed else "-",
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arrow)


def add_down_arrow(ax, x, y_top, y_bottom, color="#1565c0"):
    add_arrow(ax, (x, y_top), (x, y_bottom), color=color, lw=1.4)


def add_cylinder(ax, x, y, w, h, title, body, facecolor="#4db7e8", edgecolor="#1565c0"):
    # Body
    rect = Rectangle(
        (x, y),
        w,
        h,
        linewidth=1.4,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(rect)

    # Top and bottom ellipses
    top = Ellipse(
        (x + w / 2, y + h),
        w,
        h * 0.32,
        linewidth=1.4,
        edgecolor=edgecolor,
        facecolor="#dff4ff",
    )
    bottom = Ellipse(
        (x + w / 2, y),
        w,
        h * 0.32,
        linewidth=1.4,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(bottom)
    ax.add_patch(top)

    ax.text(
        x + w / 2,
        y + h * 0.64,
        title,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#073763",
    )
    ax.text(
        x + w / 2,
        y + h * 0.30,
        body,
        ha="center",
        va="center",
        fontsize=8,
        color="#073763",
        linespacing=1.25,
    )


def add_small_tool(ax, x, y, w, h, label, icon_text=""):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.1,
        edgecolor="#f0a84b",
        facecolor="#fff8e8",
    )
    ax.add_patch(box)

    if icon_text:
        ax.text(
            x + 0.18,
            y + h / 2,
            icon_text,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color="#2b6cb0",
        )
        text_x = x + 0.38
        ha = "left"
    else:
        text_x = x + w / 2
        ha = "center"

    ax.text(
        text_x,
        y + h / 2,
        label,
        ha=ha,
        va="center",
        fontsize=8.5,
        color="black",
    )


def add_model_card(ax, x, y, w, h, title, body, edgecolor="#ff6b6b", facecolor="#fff3f3"):
    add_round_box(
        ax,
        x,
        y,
        w,
        h,
        title=title,
        body="",
        facecolor=facecolor,
        edgecolor=edgecolor,
        title_color="#d7191c" if edgecolor == "#ff6b6b" else "#2e7d32",
        title_size=10,
        body_size=8,
        linewidth=1.3,
    )

    # Inner light panel
    inner = FancyBboxPatch(
        (x + 0.20, y + 0.20),
        w - 0.40,
        h - 0.60,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=0,
        facecolor="white",
        alpha=0.8,
    )
    ax.add_patch(inner)

    ax.text(
        x + 0.35,
        y + h - 0.62,
        body,
        ha="left",
        va="top",
        fontsize=8,
        color="black",
        linespacing=1.35,
    )


# -------------------------
# Main figure
# -------------------------
def main():
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.axis("off")

    fig.patch.set_facecolor("white")

    # Title
    ax.text(
        10,
        11.55,
        "Jurkat Perturb-seq Gene Expression Prediction Workflow with MLflow",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
    )
    ax.text(
        10,
        11.20,
        "Linear Baseline, MLP Hyperparameter Tuning, and Future State Model Extension",
        ha="center",
        va="center",
        fontsize=11,
        color="#333333",
    )

    # -------------------------
    # Top data source and real samples
    # -------------------------
    add_round_box(
        ax,
        0.45,
        9.85,
        2.9,
        1.25,
        "Data Source",
        "Jurkat Perturb-seq\nARC Virtual Cell Challenge\n\njurkat.h5 / processed_adata.h5ad\n21,412 cells × 18,080 genes",
        facecolor="#fff8e8",
        edgecolor="#f0a84b",
        title_size=10,
        body_size=7.3,
        align="center",
    )

    add_cylinder(
        ax,
        4.45,
        9.90,
        2.1,
        1.05,
        "Real Samples",
        "Observed cells\nControl + Perturbed\n68 perturbations",
    )

    add_arrow(ax, (3.35, 10.48), (4.45, 10.48))

    # -------------------------
    # Left pipeline modules
    # -------------------------
    # Preprocessing outer
    add_round_box(
        ax,
        0.25,
        7.05,
        4.1,
        2.35,
        "1. Preprocessing",
        "",
        facecolor="white",
        edgecolor="#777777",
        dashed=True,
        title_size=11,
    )

    step_x = 0.48
    step_w = 3.65
    add_step_box(ax, step_x, 8.78, step_w, 0.32, "Extract highly variable genes (3051)")
    add_step_box(ax, step_x, 8.30, step_w, 0.32, "Use provided log-normalized expression")
    add_step_box(ax, step_x, 7.82, step_w, 0.32, "Separate control and perturbed cells")
    add_step_box(ax, step_x, 7.34, step_w, 0.32, "Encode perturbation labels (68)")

    add_down_arrow(ax, 2.30, 8.78, 8.62)
    add_down_arrow(ax, 2.30, 8.30, 8.14)
    add_down_arrow(ax, 2.30, 7.82, 7.66)

    # Feature engineering
    add_round_box(
        ax,
        0.25,
        4.95,
        4.1,
        1.75,
        "2. Feature Engineering",
        "",
        facecolor="white",
        edgecolor="#777777",
        dashed=True,
        title_size=11,
    )

    add_step_box(ax, step_x, 6.22, step_w, 0.32, "Control cells as input reference")
    add_step_box(ax, step_x, 5.75, step_w, 0.32, "One-hot perturbation vector")
    add_step_box(ax, step_x, 5.28, step_w, 0.32, "Concatenate: [3051 genes + 68 pert] = 3119")
    add_step_box(ax, step_x, 5.02, step_w, 0.22, "Target: perturbed gene expression (3051)")

    add_down_arrow(ax, 2.30, 6.22, 6.07)
    add_down_arrow(ax, 2.30, 5.75, 5.60)

    # Data splitting
    add_round_box(
        ax,
        0.25,
        2.35,
        4.1,
        2.15,
        "3. Data Splitting",
        "",
        facecolor="white",
        edgecolor="#777777",
        dashed=True,
        title_size=11,
    )

    add_round_box(
        ax,
        0.55,
        3.25,
        1.05,
        0.75,
        "Train Set",
        "Remaining cells",
        facecolor="#e8f8e8",
        edgecolor="#5aa85a",
        title_size=8.5,
        body_size=7,
    )
    add_round_box(
        ax,
        1.78,
        3.25,
        1.10,
        0.75,
        "Validation",
        "10% from\nhigh-count groups",
        facecolor="#fff8e8",
        edgecolor="#f0a84b",
        title_size=8.5,
        body_size=7,
    )
    add_round_box(
        ax,
        3.05,
        3.25,
        1.00,
        0.75,
        "Test Set",
        "Same split\nfor comparison",
        facecolor="#f3f3ff",
        edgecolor="#7b70d6",
        title_size=8.5,
        body_size=7,
    )

    ax.text(
        2.30,
        2.75,
        "Save tensors → data/processed/X_test.pt, y_test.pt",
        ha="center",
        va="center",
        fontsize=8,
        color="#333333",
    )

    # Arrows from Real Samples to left modules
    add_arrow(ax, (5.50, 9.90), (5.50, 8.20))
    add_arrow(ax, (5.50, 8.20), (4.35, 8.20))
    add_arrow(ax, (5.50, 8.20), (5.50, 5.80))
    add_arrow(ax, (5.50, 5.80), (4.35, 5.80))
    add_arrow(ax, (5.50, 5.80), (5.50, 3.40))
    add_arrow(ax, (5.50, 3.40), (4.35, 3.40))

    # -------------------------
    # Modeling module
    # -------------------------
    add_round_box(
        ax,
        6.35,
        1.65,
        7.55,
        7.95,
        "4. Modeling / Current and Future Models",
        "",
        facecolor="white",
        edgecolor="#777777",
        dashed=True,
        title_size=12,
        linewidth=1.5,
    )

    add_model_card(
        ax,
        6.75,
        8.05,
        6.75,
        1.15,
        "4.1 Linear Baseline",
        "Input x: control expression + perturbation one-hot\n"
        "Model: y = Wx + b\n"
        "Loss: MSELoss    Optimizer: Adam\n"
        "Logged to MLflow as linear baseline run",
    )

    add_model_card(
        ax,
        6.75,
        6.55,
        6.75,
        1.15,
        "4.2 MLP Neural Network",
        "Input dim: 3119    Output dim: 3051\n"
        "Multi-layer neural network with ReLU activations\n"
        "Hidden dim tuned: 128, 256, 512\n"
        "Dropout tuned: 0.0, 0.2",
    )

    add_model_card(
        ax,
        6.75,
        5.05,
        6.75,
        1.15,
        "4.3 MLP Hyperparameter Sweep",
        "Learning rate: 1e-4, 1e-3, 1e-2\n"
        "Batch size fixed: 64\n"
        "Optimizer fixed: Adam\n"
        "Total sweep: 18 MLflow runs",
        edgecolor="#f0a84b",
        facecolor="#fff8f1",
    )

    add_model_card(
        ax,
        6.75,
        3.55,
        6.75,
        1.15,
        "4.4 State Model Placeholder",
        "Future work: Arc Institute State model\n"
        "Will use the same processed HVG setup\n"
        "Input dim: 3119    Output dim: 3051\n"
        "Reserved for later MLflow comparison",
        edgecolor="#7b70d6",
        facecolor="#f3f3ff",
    )

    # Input arrows into modeling
    add_arrow(ax, (4.35, 8.20), (6.35, 8.60))
    add_arrow(ax, (4.35, 5.80), (6.35, 7.10))
    add_arrow(ax, (4.35, 3.40), (6.35, 5.60))

    # -------------------------
    # Right evaluation and MLflow
    # -------------------------
    add_round_box(
        ax,
        15.0,
        7.05,
        4.25,
        2.35,
        "5. Evaluation",
        "",
        facecolor="white",
        edgecolor="#777777",
        dashed=True,
        title_size=12,
    )

    eval_x = 15.35
    eval_w = 3.55
    add_step_box(ax, eval_x, 8.70, eval_w, 0.38, "Regression: gene expression prediction")
    add_step_box(ax, eval_x, 8.15, eval_w, 0.38, "Training loss: MSE")
    add_step_box(ax, eval_x, 7.60, eval_w, 0.38, "Validation loss / best epoch")
    add_step_box(ax, eval_x, 7.05, eval_w, 0.38, "Final test loss comparison")

    add_round_box(
        ax,
        15.0,
        3.20,
        4.25,
        3.35,
        "6. Experiment Tracking (MLflow)",
        "",
        facecolor="white",
        edgecolor="#777777",
        dashed=True,
        title_size=12,
    )

    ml_x = 15.35
    ml_w = 3.55
    add_round_box(
        ax,
        ml_x,
        5.75,
        ml_w,
        0.55,
        "Log Parameters",
        "model, lr, batch_size, hidden_dim, dropout, seed",
        facecolor="#f8f3ff",
        edgecolor="#9b72d9",
        title_size=8.5,
        body_size=7.3,
    )
    add_round_box(
        ax,
        ml_x,
        5.05,
        ml_w,
        0.55,
        "Log Metrics",
        "train_loss, val_loss, best_val_loss, test_loss",
        facecolor="#f8f3ff",
        edgecolor="#9b72d9",
        title_size=8.5,
        body_size=7.3,
    )
    add_round_box(
        ax,
        ml_x,
        4.35,
        ml_w,
        0.55,
        "Log Artifacts",
        "checkpoints, configs, CSV summaries, figures",
        facecolor="#f8f3ff",
        edgecolor="#9b72d9",
        title_size=8.5,
        body_size=7.3,
    )
    add_round_box(
        ax,
        ml_x,
        3.65,
        ml_w,
        0.55,
        "Compare Runs",
        "linear vs MLP vs future State model",
        facecolor="#f8f3ff",
        edgecolor="#9b72d9",
        title_size=8.5,
        body_size=7.3,
    )

    ax.text(
        17.15,
        3.35,
        "MLflow",
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
        color="#2d6cdf",
        alpha=0.85,
    )

    # Model to evaluation arrows
    add_arrow(ax, (13.50, 8.60), (15.00, 8.45))
    add_arrow(ax, (13.50, 7.10), (15.00, 8.20))
    add_arrow(ax, (13.50, 5.60), (15.00, 7.95))
    add_arrow(ax, (13.50, 4.10), (15.00, 7.70), dashed=True)

    # Evaluation to MLflow
    add_arrow(ax, (17.15, 7.05), (17.15, 6.55))

    # -------------------------
    # Bottom tooling bar
    # -------------------------
    add_round_box(
        ax,
        0.25,
        0.35,
        15.9,
        0.85,
        "7. Infrastructure & Tooling",
        "",
        facecolor="white",
        edgecolor="#777777",
        dashed=True,
        title_size=10,
    )

    tool_y = 0.50
    tool_h = 0.38
    x0 = 0.55
    gap = 0.18
    tools = [
        ("Python 3", "Py"),
        ("PyTorch", "🔥"),
        ("NumPy", "N"),
        ("Pandas", "Pd"),
        ("Scanpy / AnnData", "Sc"),
        ("Matplotlib", "M"),
        ("MLflow", "ML"),
        ("CSV / JSON", "{}"),
    ]

    widths = [1.45, 1.35, 1.25, 1.25, 1.90, 1.55, 1.25, 1.45]

    cur_x = x0
    for (label, icon), width in zip(tools, widths):
        add_small_tool(ax, cur_x, tool_y, width, tool_h, label, icon)
        cur_x += width + gap

    # Legend
    add_round_box(
        ax,
        16.55,
        0.35,
        2.65,
        1.45,
        "Legend",
        "",
        facecolor="white",
        edgecolor="#bbbbbb",
        title_size=9,
    )
    add_arrow(ax, (16.85, 1.30), (17.45, 1.30), lw=1.2)
    ax.text(17.58, 1.30, "Data flow", va="center", fontsize=8)
    add_arrow(ax, (16.85, 0.92), (17.45, 0.92), lw=1.2, dashed=True)
    ax.text(17.58, 0.92, "Future / planned", va="center", fontsize=8)
    ax.add_patch(
        Rectangle(
            (16.82, 0.55),
            0.25,
            0.18,
            facecolor="#f8f3ff",
            edgecolor="#9b72d9",
            linewidth=1,
        )
    )
    ax.text(17.58, 0.64, "MLflow artifact", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved polished workflow diagram to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()