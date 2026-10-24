"""Generate Grad-CAM visualization for misclassified samples (False Negative & False Positive)."""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

sys.path.insert(0, 'helpers')
from visualize_hierarchical_gradcam import generate_hierarchical_visualization

model_path = 'B3/efficientnetb3_adamw_fold2.weights.h5'
img_size = (300, 300)

# 1. Pilih sampel False Negative (AI ditebak Asli)
fn_dir = 'B3/error_analysis/false_negatives'
fn_files = [f for f in os.listdir(fn_dir) if f.endswith(('.png', '.jpg'))]

# 2. Pilih sampel False Positive (Asli ditebak AI)
fp_dir = 'B3/error_analysis/false_positives'
fp_files = [f for f in os.listdir(fp_dir) if f.endswith(('.png', '.jpg'))]

print(f"Total False Negatives: {len(fn_files)}")
print(f"Total False Positives: {len(fp_files)}")

# Ambil sampel pertama yang menarik
fn_sample = os.path.join(fn_dir, fn_files[0])
fp_sample = os.path.join(fp_dir, fp_files[0])

print(f"\nMembuat visualisasi Grad-CAM untuk False Negative: {fn_sample}")
generate_hierarchical_visualization(
    model_path=model_path,
    image_path=fn_sample,
    variant_name='B3',
    img_size=img_size,
    output_save_path='B3/gradcam_misclassified_fn.png'
)

print(f"\nMembuat visualisasi Grad-CAM untuk False Positive: {fp_sample}")
generate_hierarchical_visualization(
    model_path=model_path,
    image_path=fp_sample,
    variant_name='B3',
    img_size=img_size,
    output_save_path='B3/gradcam_misclassified_fp.png'
)

print("\n[OK] Selesai membuat visualisasi Grad-CAM untuk sampel salah prediksi!")
