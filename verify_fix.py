"""Verifikasi bahwa fix load_exact_weights_robust berhasil memuat bobot classifier."""
import os, sys
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import h5py

sys.path.insert(0, '.')
from gui_tester import build_exact_training_model, load_exact_weights_robust

weight_path = 'B7/efficientnetb7_baseline_fold2.weights.h5'

print("=" * 80)
print("  VERIFIKASI LOADING BOBOT SEBELUM DAN SESUDAH FIX")
print("=" * 80)

# Build model
model, img_size = build_exact_training_model('EfficientNet-B7')

# Simpan bobot SEBELUM loading (random/ImageNet init)
dense_before = model.get_layer('dense').get_weights()[0].copy()
dense1_before = model.get_layer('dense_1').get_weights()[0].copy()
bn_before = model.get_layer('batch_normalization').get_weights()[0].copy()

print(f"\nSEBELUM load_exact_weights_robust:")
print(f"  dense  kernel mean={np.mean(dense_before):.6f} std={np.std(dense_before):.6f}")
print(f"  dense_1 kernel mean={np.mean(dense1_before):.6f} std={np.std(dense1_before):.6f}")
print(f"  BN gamma mean={np.mean(bn_before):.6f} std={np.std(bn_before):.6f}")

# Load weights dengan fungsi yang sudah diperbaiki
load_exact_weights_robust(model, weight_path)

# Simpan bobot SESUDAH loading
dense_after = model.get_layer('dense').get_weights()[0].copy()
dense1_after = model.get_layer('dense_1').get_weights()[0].copy()
bn_after = model.get_layer('batch_normalization').get_weights()[0].copy()

print(f"\nSESUDAH load_exact_weights_robust:")
print(f"  dense  kernel mean={np.mean(dense_after):.6f} std={np.std(dense_after):.6f}")
print(f"  dense_1 kernel mean={np.mean(dense1_after):.6f} std={np.std(dense1_after):.6f}")
print(f"  BN gamma mean={np.mean(bn_after):.6f} std={np.std(bn_after):.6f}")

# Verifikasi perubahan
dense_changed = not np.allclose(dense_before, dense_after)
dense1_changed = not np.allclose(dense1_before, dense1_after)
bn_changed = not np.allclose(bn_before, bn_after)

print(f"\nDense weights berubah setelah loading: {dense_changed}")
print(f"Dense_1 weights berubah setelah loading: {dense1_changed}")
print(f"BN weights berubah setelah loading: {bn_changed}")

# Cross-check dengan h5py langsung
with h5py.File(weight_path, 'r') as f:
    h5_dense = np.array(f['layers']['dense']['vars']['0'])
    h5_dense1 = np.array(f['layers']['dense_1']['vars']['0'])
    
    match_dense = np.allclose(dense_after, h5_dense)
    match_dense1 = np.allclose(dense1_after, h5_dense1)
    
    print(f"\nDense weights COCOK dengan file h5: {match_dense}")
    print(f"Dense_1 weights COCOK dengan file h5: {match_dense1}")

if match_dense and match_dense1:
    print("\n✅ FIX BERHASIL! Bobot classifier termuat dengan benar!")
else:
    print("\n❌ FIX GAGAL! Bobot classifier TIDAK termuat!")
