import os, glob
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

def build_model():
    base_model = tf.keras.applications.EfficientNetB0(weights=None, include_top=False, input_shape=(*IMG_SIZE, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax', dtype='float32')(x)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=outputs)
    return model

def eval_scenario(name, weight_dir, weight_prefix, test_dir, class_names):
    print(f"\n=======================================================")
    print(f" EVALUASI TEST SET: {name}")
    print(f" Test Dir: {test_dir} | Classes: {class_names}")
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
    
    # 1. Per Fold Evaluation
    fold_probs = []
    for f in range(1, 6):
        w_path = os.path.join(weight_dir, f"{weight_prefix}{f}.weights.h5")
        if not os.path.exists(w_path):
            print(f"Weight not found: {w_path}")
            continue
        model = build_model()
        model.load_weights(w_path)
        test_gen.reset()
        preds = model.predict(test_gen, verbose=0)
        fold_probs.append(preds)
        
        y_pred = np.argmax(preds, axis=1)
        acc = accuracy_score(y_true, y_pred) * 100
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100
        rec = recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100
        print(f"  Fold {f}: Akurasi={acc:.2f}% | Precision={prec:.2f}% | Recall={rec:.2f}% | F1={f1:.2f}%")
        
    # 2. Ensemble 5-Fold Evaluation
    ensemble_prob = np.mean(fold_probs, axis=0)
    ens_pred = np.argmax(ensemble_prob, axis=1)
    ens_acc = accuracy_score(y_true, ens_pred) * 100
    ens_prec = precision_score(y_true, ens_pred, average='weighted', zero_division=0) * 100
    ens_rec = recall_score(y_true, ens_pred, average='weighted', zero_division=0) * 100
    ens_f1 = f1_score(y_true, ens_pred, average='weighted', zero_division=0) * 100
    cm = confusion_matrix(y_true, ens_pred)
    
    print(f"  🏆 ENSEMBLE 5-FOLD: Akurasi={ens_acc:.2f}% | Precision={ens_prec:.2f}% | Recall={ens_rec:.2f}% | F1={ens_f1:.2f}%")
    print(f"  Confusion Matrix:\n{cm}")
    return ens_acc, ens_f1

# 1. 10k + 50 Epochs
eval_scenario(
    name="10k Dataset + 50 Epochs",
    weight_dir="B0",
    weight_prefix="efficientnetb0_sgd_fold",
    test_dir="dataset/test",
    class_names=['AI', 'Asli']
)

# 2. 10k + 100 Epochs
eval_scenario(
    name="10k Dataset + 100 Epochs",
    weight_dir="B0/100_EPOCH",
    weight_prefix="efficientnetb0_sgd_100ep_fold",
    test_dir="dataset/test",
    class_names=['AI', 'Asli']
)

# 3. 20k + 50 Epochs
eval_scenario(
    name="20k Pure Face Dataset + 50 Epochs",
    weight_dir="PERCOBAAN_GENIMAGE_B0_SGD",
    weight_prefix="efficientnetb0_sgd_fold",
    test_dir="PERCOBAAN_GENIMAGE_B0_SGD/dataset/test",
    class_names=['fake', 'real']
)
