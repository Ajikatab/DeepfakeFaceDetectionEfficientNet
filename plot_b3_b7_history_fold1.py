import json
import matplotlib.pyplot as plt

def generate_plot(variant):
    optimizers = [
        ("Baseline (RMSprop) (Fold 1)", f"{variant}/history_baseline_fold1.json", "#2196F3"),
        ("Adam (Fold 1)", f"{variant}/history_adam_fold1.json", "#FF9800"),
        ("SGD (Fold 1)", f"{variant}/history_sgd_fold1.json", "#4CAF50"),
        ("AdamW (Fold 1)", f"{variant}/history_adamw_fold1.json", "#F44336"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Perbandingan Proses Training 4 Optimizer — EfficientNet-{variant}", fontsize=16, fontweight="bold")

    for name, json_path, color in optimizers:
        with open(json_path) as f:
            data = json.load(f)
        
        val_loss = data.get("val_loss", [])
        val_acc = data.get("val_accuracy", [])
        epochs = range(1, len(val_loss) + 1)
        
        # Plot Loss
        axes[0].plot(epochs, val_loss, label=name, color=color, linewidth=2.0)
        
        # Plot Accuracy
        axes[1].plot(epochs, val_acc, label=name, color=color, linewidth=2.0)

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
    print(f"[OK] Gambar berhasil disimpan ke {save_path}")

if __name__ == "__main__":
    generate_plot("B3")
    generate_plot("B7")
