import os
import numpy as np
from PIL import Image, ImageOps

# ---------------------------------------------------------------------------
# Preprocess seluruh dataset (train/val) dan simpan hasilnya ke folder
# ``preprocessed_dataset`` yang meniru struktur kelas (AI / Asli).
# ---------------------------------------------------------------------------

SRC_ROOT = r"dataset/train_val"          # folder sumber (asli)
DST_ROOT = r"preprocessed_dataset/train_val"  # folder tujuan (hasil pra‑pemrosesan)
TARGET_SIZE = (300, 300)                # ukuran yang dipakai EfficientNet‑B3 (resize)

def preprocess_image(img_path: str) -> Image.Image:
    """Baca gambar, lakukan resize ke ``TARGET_SIZE``.
    Normalisasi (0‑1) diperlukan hanya saat melatih model, sehingga untuk
    penyimpanan kembali ke PNG/JPG kami menyimpan gambar yang sudah di‑resize
    dengan nilai piksel 0‑255.
    """
    raw = Image.open(img_path)
    img = ImageOps.exif_transpose(raw).convert('RGB')
    img_resized = img.resize(TARGET_SIZE, Image.Resampling.BILINEAR)
    return img_resized

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def process_class(class_name: str):
    src_dir = os.path.join(SRC_ROOT, class_name)
    dst_dir = os.path.join(DST_ROOT, class_name)
    ensure_dir(dst_dir)
    for fname in os.listdir(src_dir):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            continue
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, fname)
        img = preprocess_image(src_path)
        img.save(dst_path)
        print(f"[OK] {src_path} -> {dst_path}")

if __name__ == "__main__":
    print("Memproses seluruh dataset dan menyimpan ke", DST_ROOT)
    for class_name in os.listdir(SRC_ROOT):
        class_path = os.path.join(SRC_ROOT, class_name)
        if os.path.isdir(class_path):
            process_class(class_name)
    print("Selesai pra‑pemrosesan seluruh dataset.")
