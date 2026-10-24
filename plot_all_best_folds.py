import json
import matplotlib.pyplot as plt

def find_best_fold(variant, opt_key):
    """Find the fold with highest val_accuracy for a given optimizer."""
    best_fold = 1
    best_acc = 0
    for fold in range(1, 6):
        path = f"{variant}/history_{opt_key}_fold{fold}.json"
        try:
            with open(path) as f:
                data = json.load(f)
            va = max(data.get("val_accuracy", [0]))
            if va > best_acc:
                best_acc = va
                best_fold = fold
        except FileNotFoundError:
            pass
    return best_fold

def generate_plot(variant):
    opt_keys = ["baseline", "adam", "sgd", "adamw"]
    opt_labels = ["Baseline (RMSprop)", "Adam", "SGD", "AdamW"]
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#F44336"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"Perbandingan Proses Training 4 Optimizer — EfficientNet-{variant}",
        fontsize=16, fontweight="bold"
    )

    for opt_key, opt_label, color in zip(opt_keys, opt_labels, colors):
        best_fold = find_best_fold(variant, opt_key)
        json_path = f"{variant}/history_{opt_key}_fold{best_fold}.json"

        with open(json_path) as f:
            data = json.load(f)

        val_loss = data.get("val_loss", [])
        val_acc = data.get("val_accuracy", [])
        epochs = range(1, len(val_loss) + 1)
        legend_name = f"{opt_label} (Fold {best_fold})"

        axes[0].plot(epochs, val_loss, label=legend_name, color=color, linewidth=2.0)
        axes[1].plot(epochs, val_acc, label=legend_name, color=color, linewidth=2.0)

        print(f"  {variant} | {opt_label}: Best Fold = {best_fold}")

    # Subplot 1: Loss
    axes[0].set_title("Validation Loss per Epoch", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Loss", fontsize=11)
    axes[0].grid(True, linestyle="-", alpha=0.3)
    axes[0].legend(fontsize=10, loc="upper right")

    # Subplot 2: Accuracy
    axes[1].set_title("Validation Accuracy per Epoch", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Accuracy", fontsize=11)
    axes[1].grid(True, linestyle="-", alpha=0.3)
    axes[1].legend(fontsize=10, loc="lower right")

    plt.tight_layout()
    save_path = f"{variant}/optimizer_comparison_training_history_{variant.lower()}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Saved => {save_path}\n")

if __name__ == "__main__":
    for v in ["B0", "B3", "B7"]:
        generate_plot(v)
