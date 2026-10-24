import os, glob
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

TEST_P_DIR = r"d:/Aji Katab/KULIAH/SEMESTER 8/TUGAS AKHIR/KODING TA/P"
IMG_SIZE = (224, 224)

def build_model():
    base_model = tf.keras.applications.EfficientNetB0(weights=None, include_top=False, input_shape=(*IMG_SIZE, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax', dtype='float32')(x)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=outputs)
    return model

def load_ensemble(weight_dir, prefix):
    models = []
    for f in range(1, 6):
        w_path = os.path.join(weight_dir, f"{prefix}{f}.weights.h5")
        if os.path.exists(w_path):
            m = build_model()
            m.load_weights(w_path)
            models.append(m)
    return models

print("Memuat bobot ensemble untuk 3 skenario...")
models_10k_50  = load_ensemble("B0", "efficientnetb0_sgd_fold")
models_10k_100 = load_ensemble("B0/100_EPOCH", "efficientnetb0_sgd_100ep_fold")
models_20k_50  = load_ensemble("PERCOBAAN_GENIMAGE_B0_SGD", "efficientnetb0_sgd_fold")

files = sorted(glob.glob(os.path.join(TEST_P_DIR, "*.*")))
results = []
corrupt_files = []

for path in files:
    fname = os.path.basename(path)
    true_label = "AI"
    
    try:
        img = Image.open(path).convert('RGB').resize(IMG_SIZE, Image.BILINEAR)
    except Exception as e:
        corrupt_files.append(fname)
        continue
        
    arr = np.expand_dims(tf.keras.preprocessing.image.img_to_array(img), axis=0)
    
    # 1. 10k 50ep (0: AI, 1: Asli)
    p1 = np.mean([m(arr, training=False).numpy()[0] for m in models_10k_50], axis=0)
    pred1 = "ASLI" if p1[1] >= p1[0] else "AI"
    conf1 = p1[1]*100 if pred1 == "ASLI" else p1[0]*100
    stat1 = "BENAR" if pred1 == true_label else "SALAH"
    
    # 2. 10k 100ep (0: AI, 1: Asli)
    p2 = np.mean([m(arr, training=False).numpy()[0] for m in models_10k_100], axis=0)
    pred2 = "ASLI" if p2[1] >= p2[0] else "AI"
    conf2 = p2[1]*100 if pred2 == "ASLI" else p2[0]*100
    stat2 = "BENAR" if pred2 == true_label else "SALAH"
    
    # 3. 20k 50ep (0: fake/AI, 1: real/Asli)
    p3 = np.mean([m(arr, training=False).numpy()[0] for m in models_20k_50], axis=0)
    pred3 = "ASLI" if p3[1] >= p3[0] else "AI"
    conf3 = p3[1]*100 if pred3 == "ASLI" else p3[0]*100
    stat3 = "BENAR" if pred3 == true_label else "SALAH"
    
    results.append({
        'File': fname,
        'Label Asli': true_label,
        'Skenario 1 (10k 50ep)': f"{pred1} ({conf1:.1f}%) [{stat1}]",
        'Skenario 2 (10k 100ep)': f"{pred2} ({conf2:.1f}%) [{stat2}]",
        'Skenario 3 (20k 50ep)': f"{pred3} ({conf3:.1f}%) [{stat3}]",
        'Stat1': stat1,
        'Stat2': stat2,
        'Stat3': stat3
    })

df_res = pd.DataFrame(results)

print("=" * 115)
print("  HASIL PENGUJIAN GAMBAR AI DARI STABLE DIFFUSION (FOLDER P) PADA 3 SKENARIO")
print("=" * 115)
for idx, r in df_res.iterrows():
    print(f"{r['File']:25} | 10k 50ep: {r['Skenario 1 (10k 50ep)']:23} | 10k 100ep: {r['Skenario 2 (10k 100ep)']:23} | 20k 50ep: {r['Skenario 3 (20k 50ep)']:23}")

acc1 = (df_res['Stat1'] == 'BENAR').sum()
acc2 = (df_res['Stat2'] == 'BENAR').sum()
acc3 = (df_res['Stat3'] == 'BENAR').sum()
total = len(df_res)

print("\n" + "=" * 115)
print(f"  RINGKASAN AKURASI DETEKSI GAMBAR STABLE DIFFUSION ({total} GAMBAR VALID):")
print(f"  - Skenario 1 (10k Dataset + 50 Epoch) : {acc1} / {total} Terdeteksi Benar ({acc1/total*100:.1f}%)")
print(f"  - Skenario 2 (10k Dataset + 100 Epoch): {acc2} / {total} Terdeteksi Benar ({acc2/total*100:.1f}%)")
print(f"  - Skenario 3 (20k Dataset + 50 Epoch) : {acc3} / {total} Terdeteksi Benar ({acc3/total*100:.1f}%)")
print("=" * 115)

if corrupt_files:
    print(f"\n[CATATAN]: File berikut bukan format gambar (berisi teks HTML webpage): {corrupt_files}")
