import os, sys, json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=C:/Users/katab/miniconda3/envs/tf-gpu/Library/bin'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 64

def build_model():
    base_model = tf.keras.applications.EfficientNetB0(weights=None, include_top=False, input_shape=(*IMG_SIZE, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax', dtype='float32')(x)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=outputs)
    return model

def evaluate_test(name, weight_dir, weight_prefix, test_dir, class_names):
    print(f"\n=======================================================")
    print(f" [EVALUASI TEST SET] {name}")
    print(f"=======================================================")
    
    filepaths, labels = [], []
    for c in class_names:
        cdir = os.path.join(test_dir, c)
        for f in os.listdir(cdir):
            filepaths.append(os.path.join(cdir, f))
            labels.append(c)
            
    df_test = pd.DataFrame({'filepath': filepaths, 'label': labels})
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    test_gen = test_datagen.flow_from_dataframe(
        df_test, x_col='filepath', y_col='label', target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode='categorical', classes=class_names, shuffle=False)
    
    y_true = test_gen.classes
    
    # 1. Per-fold eval
    fold_probs = []
    fold_accs = []
    for f in range(1, 6):
        w_path = os.path.join(weight_dir, f"{weight_prefix}{f}.weights.h5")
        if not os.path.exists(w_path):
            continue
        model = build_model()
        model.load_weights(w_path)
        test_gen.reset()
        preds = model.predict(test_gen, verbose=0)
        fold_probs.append(preds)
        
        y_pred = np.argmax(preds, axis=1)
        acc = accuracy_score(y_true, y_pred) * 100
        fold_accs.append(acc)
        print(f"  Fold {f}: Akurasi Test = {acc:.2f}%")
        
    print(f"  --> Rata-rata Single Fold Test Acc: {np.mean(fold_accs):.2f}% +- {np.std(fold_accs):.2f}%")
    
    # 2. Ensemble eval
    ens_prob = np.mean(fold_probs, axis=0)
    ens_pred = np.argmax(ens_prob, axis=1)
    ens_acc = accuracy_score(y_true, ens_pred) * 100
    ens_prec = precision_score(y_true, ens_pred, average='weighted', zero_division=0) * 100
    ens_rec = recall_score(y_true, ens_pred, average='weighted', zero_division=0) * 100
    ens_f1 = f1_score(y_true, ens_pred, average='weighted', zero_division=0) * 100
    cm = confusion_matrix(y_true, ens_pred)
    
    print(f"  🏆 ENSEMBLE 5-FOLD: Akurasi = {ens_acc:.2f}% | Precision = {ens_prec:.2f}% | Recall = {ens_rec:.2f}% | F1-Score = {ens_f1:.2f}%")
    print(f"  Confusion Matrix:\n{cm}")
    return ens_acc, ens_prec, ens_rec, ens_f1

# 1. 10k + 50 Epochs
evaluate_test("10K DATASET + 50 EPOCHS", "B0", "efficientnetb0_sgd_fold", "dataset/test", ['AI', 'Asli'])

# 2. 10k + 100 Epochs
evaluate_test("10K DATASET + 100 EPOCHS", "B0/100_EPOCH", "efficientnetb0_sgd_100ep_fold", "dataset/test", ['AI', 'Asli'])

# 3. 20k + 50 Epochs
evaluate_test("20K DATASET + 50 EPOCHS", "PERCOBAAN_GENIMAGE_B0_SGD", "efficientnetb0_sgd_fold", "PERCOBAAN_GENIMAGE_B0_SGD/dataset/test", ['fake', 'real'])
