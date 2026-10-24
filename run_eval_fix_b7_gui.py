import os, sys, glob
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from gui_tester import get_gui_model, MODEL_CONFIGS

FIX_DIR = r'd:/Aji Katab/KULIAH/SEMESTER 8/TUGAS AKHIR/KODING TA/FOTO SIMULASI/FIX'
files = sorted(glob.glob(os.path.join(FIX_DIR, '*.*')))

opts = ['Baseline (RMSprop)', 'Adam', 'SGD', 'AdamW']
models = {}

for opt in opts:
    best_f = MODEL_CONFIGS['EfficientNet-B7']['best_folds'][opt]
    print(f'Memuat B7 {opt} (Fold {best_f})...', flush=True)
    m, img_size, _ = get_gui_model('EfficientNet-B7', opt, best_f)
    models[opt] = (m, img_size, best_f)

print('\n' + '=' * 125, flush=True)
print('  PENGUJIAN FOTO SIMULASI FIX MENGGUNAKAN FOLD TERBAIK EFFICIENTNET-B7 (PIPELINE GUI / NOTEBOOK)', flush=True)
print('=' * 125, flush=True)

results = []
for p in files:
    fname = os.path.basename(p)
    true_label = 'ASLI' if fname.startswith('ga') else 'AI'
    img = Image.open(p).convert('RGB')
    
    row = {'File': fname, 'Label Asli': true_label}
    for opt, (m, img_sz, bf) in models.items():
        img_resized = img.resize(img_sz, getattr(Image, 'Resampling', Image).NEAREST)
        arr = np.expand_dims(tf.keras.preprocessing.image.img_to_array(img_resized), axis=0)
        preds = m(arr, training=False).numpy()[0]
        # 0: AI, 1: Asli
        pred_idx = int(np.argmax(preds))
        pred = 'ASLI' if pred_idx == 1 else 'AI'
        conf = preds[pred_idx] * 100
        stat = 'BENAR' if pred == true_label else 'SALAH'
        row[opt] = f"{pred} ({conf:.2f}%) [{stat}]"
        row[f"{opt}_stat"] = stat
    results.append(row)

df = pd.DataFrame(results)
for idx, r in df.iterrows():
    f_short = r['File'][:38]
    print(f"{f_short:38} | Asli: {r['Label Asli']:4} | Base: {r['Baseline (RMSprop)']:22} | Adam: {r['Adam']:22} | SGD: {r['SGD']:22} | AdamW: {r['AdamW']:22}", flush=True)

print('\n' + '=' * 125, flush=True)
print('  RINGKASAN AKURASI PER OPTIMIZER B7 (SINGLE BEST FOLD):')
for opt, (_, _, bf) in models.items():
    c = (df[f"{opt}_stat"] == 'BENAR').sum()
    print(f"  - EfficientNet-B7 + {opt:18} (Fold {bf}): {c}/{len(df)} Benar ({c/len(df)*100:.1f}%)", flush=True)
print('=' * 125, flush=True)
