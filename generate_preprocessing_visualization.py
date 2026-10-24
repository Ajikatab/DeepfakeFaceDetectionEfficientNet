"""
Script untuk menghasilkan Visualisasi Tahapan Pra-Pemrosesan Data & Augmentasi
Menggunakan Citra dari dataset/train_val (Dataset Asli Pelatihan)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

# Path sample dari dataset/train_val
ai_sample_path = r"dataset/train_val/AI/0137J1GGN2.jpg"
asli_sample_path = r"dataset/train_val/Asli/00042.jpg"

def generate_preprocessing_grid(image_path, title_prefix, save_path):
    # 1. Load Original Image from Dataset
    raw_img = Image.open(image_path)
    img_orig = ImageOps.exif_transpose(raw_img).convert('RGB')
    orig_w, orig_h = img_orig.size

    # 2. Resize to 300x300 (EfficientNet-B3 standard)
    img_resized = img_orig.resize((300, 300), Image.Resampling.BILINEAR)
    arr_resized = np.array(img_resized)

    # 3. Normalized / Rescaled (Float 0..1 represented as pixel distribution)
    arr_norm = arr_resized.astype(np.float32) / 255.0

    # 4. Augmentation: Horizontal Flip
    img_flipped = img_resized.transpose(Image.FLIP_LEFT_RIGHT)

    # 5. Augmentation: Random Rotation (15 degrees)
    img_rotated = img_resized.rotate(15, resample=Image.Resampling.BILINEAR, expand=False)

    # 6. Augmentation: Zoom In (120%) & Center Crop to 300x300
    zoom_factor = 1.2
    zw, zh = int(300 * zoom_factor), int(300 * zoom_factor)
    img_zoomed = img_resized.resize((zw, zh), Image.Resampling.BILINEAR)
    left = (zw - 300) // 2
    top = (zh - 300) // 2
    img_zoomed_cropped = img_zoomed.crop((left, top, left + 300, top + 300))

    # Plot 2x3 Grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Visualisasi Tahap Pra-Pemrosesan", fontsize=16, fontweight='bold', y=0.98)

    # Panel 1: Original Image
    axes[0, 0].imshow(img_orig)
    axes[0, 0].set_title(f"1. Citra Input\n(Resolusi Asli Dataset: {orig_w} × {orig_h} px)", fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')

    # Panel 2: Resized Image
    axes[0, 1].imshow(img_resized)
    axes[0, 1].set_title("2. Penyesuaian Dimensi (Resize)\n(Dimensi Input Model: 300 × 300 px)", fontsize=11, fontweight='bold')
    axes[0, 1].axis('off')

    # Panel 3: Normalization / Rescaling
    axes[0, 2].imshow(arr_norm)
    axes[0, 2].set_title("3. Normalisasi Nilai Piksel", fontsize=11, fontweight='bold')
    axes[0, 2].axis('off')

    # Panel 4: Horizontal Flip
    axes[1, 0].imshow(img_flipped)
    axes[1, 0].set_title("4. Augmentasi: Horizontal Flip", fontsize=11, fontweight='bold')
    axes[1, 0].axis('off')

    # Panel 5: Rotation
    axes[1, 1].imshow(img_rotated)
    axes[1, 1].set_title("5. Augmentasi: Random Rotation", fontsize=11, fontweight='bold')
    axes[1, 1].axis('off')

    # Panel 6: Zoom
    axes[1, 2].imshow(img_zoomed_cropped)
    axes[1, 2].set_title("6. Augmentasi: Random Zoom", fontsize=11, fontweight='bold')
    axes[1, 2].axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OK] Visualisasi berhasil disimpan: {save_path}")


if __name__ == '__main__':
    print("Memproses visualisasi pra-pemrosesan dari dataset/train_val...")
    generate_preprocessing_grid(ai_sample_path, "Citra Buatan AI", "B3/visualisasi_prapemrosesan_ai.png")
    generate_preprocessing_grid(asli_sample_path, "Citra Asli Kamera", "B3/visualisasi_prapemrosesan_asli.png")
    print("Selesai!")
