import os
import sys
import time
import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd

sys.path.append(os.path.abspath('.'))
from gui_tester import MODEL_CONFIGS, build_exact_training_model, load_exact_weights_robust

print("=" * 110)
print("  LAPORAN HASIL PREDIKSI DETAIL LENGKAP 12 SKENARIO MODEL (FOTO SIMULASI D:\\PICTURES\\SIMULASI\\FIX)")
print("=" * 110)

target_fix_dir = r"D:\PICTURES\SIMULASI\FIX"
if os.path.exists(target_fix_dir) and len(os.listdir(target_fix_dir)) > 0:
    SIMULATION_DIR = target_fix_dir
else:
    SIMULATION_DIR = os.path.abspath('FOTO SIMULASI')

image_files = sorted([f for f in os.listdir(SIMULATION_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))])

ground_truths = []
filepaths = []
for fname in image_files:
    fp = os.path.join(SIMULATION_DIR, fname)
    filepaths.append(fp)
    if fname.startswith(('ga', '202608')):
        ground_truths.append("ASLI (REAL)")
    elif fname.startswith(('kb', 'ekb', 'ChatGPT')):
        ground_truths.append("AI (FAKE)")
    else:
        ground_truths.append("UNKNOWN")

summary_results = []
scen_counter = 1

for variant_name, cfg in MODEL_CONFIGS.items():
    img_size = cfg['img_size']
    
    # Preprocess image batch for this variant
    batch_arrs = []
    for fp in filepaths:
        img = Image.open(fp).convert("RGB")
        img_resized = img.resize(img_size, getattr(Image, 'Resampling', Image).NEAREST)
        arr = tf.keras.preprocessing.image.img_to_array(img_resized)
        batch_arrs.append(arr)
    images_batch = np.array(batch_arrs, dtype=np.float32)
    
    for opt_name, pattern in cfg['patterns'].items():
        best_fold = cfg['best_folds'][opt_name]
        
        print("\n" + "=" * 110)
        print(f"SKENARIO {scen_counter}: {variant_name} + {opt_name}")
        print("=" * 110)
        
        # 1. Single Best Fold Prediction
        w_path_best = os.path.join(cfg['folder'], f"{pattern}fold{best_fold}.weights.h5")
        tf.keras.backend.clear_session()
        model_best, _ = build_exact_training_model(variant_name)
        load_exact_weights_robust(model_best, w_path_best)
        preds_single = model_best(images_batch, training=False).numpy()
        
        # 2. 5-Fold Ensemble Prediction
        ensemble_probs = np.zeros((len(image_files), 2))
        for f_num in range(1, 6):
            w_path_f = os.path.join(cfg['folder'], f"{pattern}fold{f_num}.weights.h5")
            tf.keras.backend.clear_session()
            m_f, _ = build_exact_training_model(variant_name)
            load_exact_weights_robust(m_f, w_path_f)
            p_f = m_f(images_batch, training=False).numpy()
            ensemble_probs += p_f
        ensemble_probs /= 5.0
        
        # Print Detail Table for this Scenario
        header_fmt = f"{'No':^4s} | {'Nama Gambar':42s} | {'Kategori Asli':13s} | {'Pred (Single Fold)':18s} | {'Conf Single':11s} | {'Pred (5-Fold Ens)':17s} | {'Conf Ens':9s} | {'Status Ens':10s}"
        print(header_fmt)
        print("-" * 110)
        
        single_correct = 0
        ens_correct = 0
        
        for i, fname in enumerate(image_files):
            gt = ground_truths[i]
            
            # Single fold
            p_s = preds_single[i]
            idx_s = int(np.argmax(p_s))
            label_s = "ASLI (REAL)" if idx_s == 1 else "AI (FAKE)"
            conf_s = p_s[1] if idx_s == 1 else p_s[0]
            if label_s == gt: single_correct += 1
            
            # Ensemble 5-fold
            p_e = ensemble_probs[i]
            idx_e = int(np.argmax(p_e))
            label_e = "ASLI (REAL)" if idx_e == 1 else "AI (FAKE)"
            conf_e = p_e[1] if idx_e == 1 else p_e[0]
            is_e_correct = (label_e == gt)
            if is_e_correct: ens_correct += 1
            
            st_icon = "BENAR [V]" if is_e_correct else "SALAH [X]"
            
            row_fmt = f"{i+1:^4d} | {fname:42s} | {gt:13s} | {label_s:18s} | {conf_s*100:9.2f}%  | {label_e:17s} | {conf_e*100:7.2f}%  | {st_icon:10s}"
            print(row_fmt)
            
        print("-" * 110)
        acc_s_pct = (single_correct / len(image_files)) * 100
        acc_e_pct = (ens_correct / len(image_files)) * 100
        print(f"REKAP SKENARIO {scen_counter}: Single Best Fold (Fold {best_fold}) = {single_correct}/{len(image_files)} ({acc_s_pct:.2f}%) | 5-Fold Ensemble = {ens_correct}/{len(image_files)} ({acc_e_pct:.2f}%)")
        print("-" * 110)
        
        summary_results.append({
            'No': scen_counter,
            'Arsitektur Model': variant_name,
            'Optimizer': opt_name,
            'Best Fold': f"Fold {best_fold}",
            'Akurasi Single Fold': f"{acc_s_pct:.2f}% ({single_correct}/{len(image_files)})",
            'Akurasi 5-Fold Ensemble': f"{acc_e_pct:.2f}% ({ens_correct}/{len(image_files)})"
        })
        scen_counter += 1

print("\n" + "=" * 110)
print("  RINGKASAN KOMPARASI SELURUH 12 SKENARIO MODEL")
print("=" * 110)
df_sum = pd.DataFrame(summary_results)
print(df_sum.to_string(index=False))
print("=" * 110)
