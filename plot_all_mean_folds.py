import json
import os
import numpy as np
import matplotlib.pyplot as plt

def get_mean_history(variant, opt_key):
    """
    Load all 5 fold history JSONs for an optimizer and compute the average
    val_loss and val_accuracy per epoch across the folds.
    """
    all_val_losses = []
    all_val_accs = []
    
    for fold in range(1, 6):
        path = f"{variant}/history_{opt_key}_fold{fold}.json"
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            vl = d.get("val_loss", [])
            va = d.get("val_accuracy", [])
            if vl and va:
                all_val_losses.append(vl)
                all_val_accs.append(va)
    
    if not all_val_losses:
        return [], []
    
    max_epochs = max(len(vl) for vl in all_val_losses)
    
    mean_val_loss = []
    mean_val_acc = []
    
    for ep in range(max_epochs):
        losses_at_ep = [vl[ep] for vl in all_val_losses if ep < len(vl)]
        accs_at_ep = [va[ep] for va in all_val_accs if ep < len(va)]
        
        mean_val_loss.append(np.mean(losses_at_ep))
        mean_val_acc.append(np.mean(accs_at_ep))
        
    return mean_val_loss, mean_val_acc

def generate_mean_plot(variant):
    opt_keys = ["baseline", "adam", "sgd", "adamw"]
    opt_labels = ["Baseline (RMSprop)", "Adam", "SGD", "AdamW"]
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#F44336"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"Perbandingan Proses Training 4 Optimizer (Rata-rata 5-Fold) — EfficientNet-{variant}",
        fontsize=16, fontweight="bold"
    )

    for opt_key, opt_label, color in zip(opt_keys, opt_labels, colors):
        mean_loss, mean_acc = get_mean_history(variant, opt_key)
        epochs = range(1, len(mean_loss) + 1)
        
        axes[0].plot(epochs, mean_loss, label=f"{opt_label}", color=color, linewidth=2.2)
        axes[1].plot(epochs, mean_acc, label=f"{opt_label}", color=color, linewidth=2.2)

    # Subplot 1: Loss
    axes[0].set_title("Validation Loss per Epoch (Mean 5-Fold)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Loss", fontsize=11)
    axes[0].grid(True, linestyle="-", alpha=0.3)
    axes[0].legend(fontsize=10, loc="upper right")

    # Subplot 2: Accuracy
    axes[1].set_title("Validation Accuracy per Epoch (Mean 5-Fold)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Accuracy", fontsize=11)
    axes[1].grid(True, linestyle="-", alpha=0.3)
    axes[1].legend(fontsize=10, loc="lower right")

    plt.tight_layout()
    save_path = f"{variant}/optimizer_comparison_training_history_{variant.lower()}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Saved => {save_path}")

if __name__ == "__main__":
    for v in ["B0", "B3", "B7"]:
        print(f"Generating mean 5-fold training history plot for {v}...")
        generate_mean_plot(v)
