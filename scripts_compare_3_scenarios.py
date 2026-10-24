import os, json
import pandas as pd
import numpy as np

print("=" * 85)
print("  REKAPITULASI DETAIL 3 SKENARIO EFFICIENTNET-B0 + SGD")
print("=" * 85)

# 1. 10k + 50 Epochs
print("\n[1] SKENARIO: 10K DATASET + 50 EPOCHS")
v_10k_50 = []
for f in range(1, 6):
    p = f"B0/training_log_sgd_fold{f}.csv"
    if os.path.exists(p):
        df = pd.read_csv(p)
        best_acc = df['val_accuracy'].max()
        stopped_ep = len(df)
        v_10k_50.append(best_acc)
        print(f"  Fold {f}: Selesai Epoch {stopped_ep}/50 | Best Val Acc: {best_acc:.2f}")
print(f"  --> Rata-rata CV: {np.mean(v_10k_50):.2f} +- {np.std(v_10k_50):.2f}")

if os.path.exists("B0/kfold_test_results_b0.json"):
    with open("B0/kfold_test_results_b0.json") as f:
        tr = json.load(f).get("SGD", {})
        print(f"  --> Test Set (2.000 foto): Akurasi = {tr.get('accuracy', 0):.2f} | F1 = {tr.get('f1_score', 0):.2f} | Precision = {tr.get('precision', 0):.2f} | Recall = {tr.get('recall', 0):.2f}")

# 2. 10k + 100 Epochs
print("\n[2] SKENARIO: 10K DATASET + 100 EPOCHS")
v_10k_100 = []
for f in range(1, 6):
    p = f"B0/100_EPOCH/training_log_sgd_100ep_fold{f}.csv"
    if os.path.exists(p):
        df = pd.read_csv(p)
        best_acc = df['val_accuracy'].max()
        stopped_ep = len(df)
        v_10k_100.append(best_acc)
        print(f"  Fold {f}: Selesai Epoch {stopped_ep}/100 | Best Val Acc: {best_acc:.2f}")
print(f"  --> Rata-rata CV: {np.mean(v_10k_100):.2f} +- {np.std(v_10k_100):.2f}")

# 3. 20k + 50 Epochs
print("\n[3] SKENARIO: 20K PURE HUMAN FACE DATASET + 50 EPOCHS")
v_20k_50 = []
for f in range(1, 6):
    p = f"PERCOBAAN_GENIMAGE_B0_SGD/training_log_sgd_fold{f}.csv"
    if os.path.exists(p):
        df = pd.read_csv(p)
        best_acc = df['val_accuracy'].max()
        stopped_ep = len(df)
        v_20k_50.append(best_acc)
        print(f"  Fold {f}: Selesai Epoch {stopped_ep}/50 | Best Val Acc: {best_acc:.2f}")
print(f"  --> Rata-rata CV: {np.mean(v_20k_50):.2f} +- {np.std(v_20k_50):.2f}")
