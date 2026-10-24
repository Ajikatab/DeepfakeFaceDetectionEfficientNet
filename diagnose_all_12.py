"""Diagnosa B3+Adam: Apakah bobot classifier termuat dengan benar?"""
import os, glob
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import h5py

from gui_tester import build_exact_training_model, load_exact_weights_robust, MODEL_CONFIGS
from PIL import Image

FIX_DIR = r'd:/Aji Katab/KULIAH/SEMESTER 8/TUGAS AKHIR/KODING TA/FOTO SIMULASI/FIX'

# ===== CEK LOADING BOBOT B3 =====
weight_path = 'B3/efficientnetb3_adam_fold2.weights.h5'
print("=" * 80)
print("  DIAGNOSA LOADING BOBOT B3 + ADAM (FOLD 2)")
print("=" * 80)

# Cek struktur h5
with h5py.File(weight_path, 'r') as f:
    top_keys = list(f.keys())
    print(f"Top-level keys: {top_keys}")
    if 'layers' in f:
        h5l = f['layers']
        print(f"Format: Keras 3 (layers/ group)")
        # Cek dense, dense_1
        for target in ['dense', 'dense_1', 'batch_normalization']:
            if target in h5l and 'vars' in h5l[target]:
                vg = h5l[target]['vars']
                keys = sorted(vg.keys(), key=lambda x: int(x))
                shapes = [str(np.array(vg[k]).shape) for k in keys]
                print(f"  {target}: {shapes}")
            else:
                print(f"  {target}: NOT FOUND or no vars")
    else:
        print(f"Format: Keras 2 (model_weights)")

# Build dan load
model, img_size = build_exact_training_model('EfficientNet-B3')
dense_before = model.get_layer('dense').get_weights()[0].copy()

load_exact_weights_robust(model, weight_path)
dense_after = model.get_layer('dense').get_weights()[0].copy()

# Verifikasi
with h5py.File(weight_path, 'r') as f:
    if 'layers' in f and 'dense' in f['layers'] and 'vars' in f['layers']['dense']:
        h5_dense = np.array(f['layers']['dense']['vars']['0'])
        match = np.allclose(dense_after, h5_dense)
        print(f"\nDense berubah: {not np.allclose(dense_before, dense_after)}")
        print(f"Dense COCOK h5: {match}")
    else:
        print("\nTidak bisa verifikasi - format berbeda")

# ===== TEST SEMUA B0, B3, B7 PADA FOLDER FIX =====
print("\n" + "=" * 120)
print("  TEST FOLDER FIX PADA SEMUA 12 SKENARIO (B0, B3, B7 x 4 OPTIMIZER) - SINGLE BEST FOLD")
print("=" * 120)

files = sorted(glob.glob(os.path.join(FIX_DIR, '*.*')))

for variant in ['EfficientNet-B0', 'EfficientNet-B3', 'EfficientNet-B7']:
    cfg = MODEL_CONFIGS[variant]
    print(f"\n{'='*100}")
    print(f"  {variant} (img_size={cfg['img_size']})")
    print(f"{'='*100}")
    
    for opt in ['Baseline (RMSprop)', 'Adam', 'SGD', 'AdamW']:
        best_fold = cfg['best_folds'][opt]
        pattern = cfg['patterns'][opt]
        w_file = f"{pattern}fold{best_fold}.weights.h5"
        w_path = os.path.join(cfg['folder'], w_file)
        
        tf.keras.backend.clear_session()
        m, isz = build_exact_training_model(variant)
        load_exact_weights_robust(m, w_path)
        
        correct = 0
        total = len(files)
        details = []
        
        for p in files:
            fname = os.path.basename(p)
            true_label = 'ASLI' if fname.startswith('ga') else 'AI'
            img = Image.open(p).convert('RGB').resize(isz, getattr(Image, 'Resampling', Image).NEAREST)
            arr = np.expand_dims(tf.keras.preprocessing.image.img_to_array(img), axis=0)
            preds = m(arr, training=False).numpy()[0]
            pred_idx = int(np.argmax(preds))
            pred = 'ASLI' if pred_idx == 1 else 'AI'
            conf = preds[pred_idx] * 100
            ok = pred == true_label
            if ok: correct += 1
            details.append(f"{fname[:30]:30} -> {pred} ({conf:.1f}%) {'OK' if ok else 'SALAH'}")
        
        acc = correct / total * 100
        marker = " <<<" if acc >= 60 else ""
        print(f"  {opt:20} Fold {best_fold}: {correct}/{total} ({acc:.1f}%){marker}")
        for d in details:
            print(f"    {d}")
