import os, glob, re
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import h5py

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

IMG_SIZE = (600, 600)
FIX_DIR = r"d:/Aji Katab/KULIAH/SEMESTER 8/TUGAS AKHIR/KODING TA/FOTO SIMULASI/FIX"

def get_optimizer(opt_name):
    if opt_name == 'Baseline (RMSprop)':
        return tf.keras.optimizers.RMSprop(learning_rate=1e-4)
    elif opt_name == 'Adam':
        return tf.keras.optimizers.Adam(learning_rate=1e-4)
    elif opt_name == 'SGD':
        return tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9)
    elif opt_name == 'AdamW':
        return tf.keras.optimizers.Adam(learning_rate=1e-4)
    return tf.keras.optimizers.Adam(learning_rate=1e-4)

def build_b7_model(opt_name):
    base_model = tf.keras.applications.EfficientNetB7(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax', dtype='float32')(x)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=outputs)
    model.compile(
        optimizer=get_optimizer(opt_name),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def load_robust_weights(model, filepath):
    try:
        model.load_weights(filepath)
        print(f'  [OK] Weights loaded directly from {filepath}', flush=True)
        return
    except Exception:
        pass
    try:
        with h5py.File(filepath, 'r') as f:
            if 'layers' not in f:
                model.load_weights(filepath, by_name=True, skip_mismatch=True)
                return
            h5 = f['layers']
            loaded = 0
            if 'dense' in h5 and 'vars' in h5['dense']:
                vg = h5['dense']['vars']
                keys = sorted(vg.keys(), key=lambda x: int(x))
                w = [np.array(vg[k]) for k in keys]
                model.get_layer('dense').set_weights(w)
                loaded += len(w)
            if 'dense_1' in h5 and 'vars' in h5['dense_1']:
                vg = h5['dense_1']['vars']
                keys = sorted(vg.keys(), key=lambda x: int(x))
                w = [np.array(vg[k]) for k in keys]
                model.get_layer('dense_1').set_weights(w)
                loaded += len(w)
            def _ns(s):
                return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]
            bn_names = sorted([n for n in h5.keys() if n.startswith('batch_normalization')], key=_ns)
            for bn_name in bn_names:
                if 'vars' in h5[bn_name]:
                    vg = h5[bn_name]['vars']
                    keys = sorted(vg.keys(), key=lambda x: int(x))
                    if keys and np.array(vg['0']).shape == (256,):
                        w = [np.array(vg[k]) for k in keys]
                        model.get_layer('batch_normalization').set_weights(w)
                        loaded += len(w)
                        break
            print(f'  [OK] Top layers loaded: {loaded} weight arrays from {filepath}', flush=True)
    except Exception as e:
        print(f'  [WARN] h5py load failed: {e}, falling back to by_name', flush=True)
        model.load_weights(filepath, by_name=True, skip_mismatch=True)

best_folds = {
    'Baseline (RMSprop)': ('B7/efficientnetb7_baseline_fold2.weights.h5', 'Fold 2'),
    'Adam': ('B7/efficientnetb7_adam_fold1.weights.h5', 'Fold 1'),
    'SGD': ('B7/efficientnetb7_sgd_fold2.weights.h5', 'Fold 2'),
    'AdamW': ('B7/efficientnetb7_adamw_fold1.weights.h5', 'Fold 1')
}

models = {}
for opt_name, (w_path, fold_name) in best_folds.items():
    print(f"Memuat Model {opt_name} ({fold_name})...", flush=True)
    m = build_b7_model(opt_name)
    load_robust_weights(m, w_path)
    models[opt_name] = m

files = sorted(glob.glob(os.path.join(FIX_DIR, "*.*")))
results = []

# Class mapping in 10k: 0 = AI, 1 = Asli
for p in files:
    fname = os.path.basename(p)
    if fname.startswith("ga"):
        true_label = "ASLI"
    else:
        true_label = "AI"
        
    img = Image.open(p).convert('RGB').resize(IMG_SIZE, Image.BILINEAR)
    arr = np.expand_dims(tf.keras.preprocessing.image.img_to_array(img), axis=0)
    
    row = {'File': fname, 'Label Sebenarnya': true_label}
    for opt_name, m in models.items():
        probs = m(arr, training=False).numpy()[0]
        # 0: AI, 1: Asli
        pred = "ASLI" if probs[1] >= probs[0] else "AI"
        conf = probs[1]*100 if pred == "ASLI" else probs[0]*100
        stat = "BENAR" if pred == true_label else "SALAH"
        row[opt_name] = f"{pred} ({conf:.1f}%) [{stat}]"
        row[f"{opt_name}_stat"] = stat
        row[f"{opt_name}_pred"] = pred
        row[f"{opt_name}_conf"] = conf
        
    results.append(row)

df = pd.DataFrame(results)

print("\n" + "=" * 125, flush=True)
print("  HASIL PENGUJIAN FOTO SIMULASI FIX PADA FOLD TERBAIK MODEL EFFICIENTNET-B7 (SEMUA OPTIMIZER)", flush=True)
print("=" * 125, flush=True)

for idx, r in df.iterrows():
    print(f"{r['File']:48} | Asli: {r['Label Sebenarnya']:4} | Base: {r['Baseline (RMSprop)']:22} | Adam: {r['Adam']:22} | SGD: {r['SGD']:22} | AdamW: {r['AdamW']:22}", flush=True)

print("\n" + "=" * 125, flush=True)
print("  RINGKASAN AKURASI PER OPTIMIZER B7:")
for opt_name in best_folds.keys():
    correct = (df[f"{opt_name}_stat"] == "BENAR").sum()
    total = len(df)
    print(f"  - EfficientNet-B7 + {opt_name:18} ({best_folds[opt_name][1]}): {correct}/{total} Benar ({correct/total*100:.1f}%)", flush=True)
print("=" * 125, flush=True)
