import os
import sys
import time
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import streamlit as st

# Force CPU execution for fast, stable, and reproducible GUI inference
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Set Page Configuration
st.set_page_config(
    page_title="Deteksi Citra AI vs Asli - Deep Learning",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration for models, image sizes, patterns, and best folds
MODEL_CONFIGS = {
    "EfficientNet-B0": {
        "folder": "B0",
        "img_size": (224, 224),
        "patterns": {
            "Baseline (RMSprop)": "efficientnetb0_baseline_",
            "Adam": "efficientnetb0_adam_",
            "SGD": "efficientnetb0_sgd_",
            "AdamW": "efficientnetb0_adamw_"
        },
        "best_folds": {
            "Baseline (RMSprop)": 5,
            "Adam": 4,
            "SGD": 3,
            "AdamW": 1
        }
    },
    "EfficientNet-B3": {
        "folder": "B3",
        "img_size": (300, 300),
        "patterns": {
            "Baseline (RMSprop)": "efficientnetb3_baseline_",
            "Adam": "efficientnetb3_adam_",
            "SGD": "efficientnetb3_sgd_",
            "AdamW": "efficientnetb3_adamw_"
        },
        "best_folds": {
            "Baseline (RMSprop)": 2,
            "Adam": 2,
            "SGD": 2,
            "AdamW": 2
        }
    },
    "EfficientNet-B7": {
        "folder": "B7",
        "img_size": (600, 600),
        "patterns": {
            "Baseline (RMSprop)": "efficientnetb7_baseline_",
            "Adam": "efficientnetb7_adam_",
            "SGD": "efficientnetb7_sgd_",
            "AdamW": "efficientnetb7_adamw_"
        },
        "best_folds": {
            "Baseline (RMSprop)": 2,
            "Adam": 1,
            "SGD": 2,
            "AdamW": 1
        }
    }
}

def build_exact_training_model(variant_name):
    """Membangun arsitektur model presisi yang 100% identik dengan notebook K-Fold Training."""
    img_size = MODEL_CONFIGS[variant_name]["img_size"]
    
    if variant_name == "EfficientNet-B0":
        base_model = tf.keras.applications.EfficientNetB0(
            weights="imagenet", include_top=False, input_shape=(*img_size, 3)
        )
    elif variant_name == "EfficientNet-B3":
        base_model = tf.keras.applications.EfficientNetB3(
            weights="imagenet", include_top=False, input_shape=(*img_size, 3)
        )
    elif variant_name == "EfficientNet-B7":
        base_model = tf.keras.applications.EfficientNetB7(
            weights="imagenet", include_top=False, input_shape=(*img_size, 3)
        )
    else:
        raise ValueError(f"Varian {variant_name} tidak valid.")

    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(256, activation="relu", name="dense")(x)
    x = tf.keras.layers.BatchNormalization(name="batch_normalization")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax", dtype="float32", name="dense_1")(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=outputs, name=f"GUI_{variant_name}")
    return model, img_size

def load_exact_weights_robust(model, weight_path):
    """Memuat bobot .weights.h5 (Keras 3 format) ke model Keras 2 secara presisi.
    
    Bug Fix: load_weights(by_name=True) GAGAL memuat bobot classifier (dense, BN, dense_1)
    dari format Keras 3 karena struktur HDF5 berbeda (layers/name/vars/0 vs model_weights/name/kernel:0).
    Solusi: Gunakan h5py untuk membaca bobot secara manual dan set_weights() per layer.
    """
    import h5py
    import re
    
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"File bobot tidak ditemukan: {weight_path}")
    
    # Coba direct load (hanya berhasil jika format Keras 2 standar)
    try:
        model.load_weights(weight_path)
        return  # Berhasil langsung
    except Exception:
        pass
    
    # Fallback: Manual load via h5py untuk format Keras 3
    with h5py.File(weight_path, 'r') as f:
        if 'layers' not in f:
            # Bukan format Keras 3, gunakan by_name
            model.load_weights(weight_path, by_name=True, skip_mismatch=True)
            return
        
        h5_layers = f['layers']
        loaded_count = 0
        
        # 1. Load semua layer model yang cocok nama & shape-nya dari file h5
        for layer in model.layers:
            if layer.name in h5_layers and 'vars' in h5_layers[layer.name]:
                vg = h5_layers[layer.name]['vars']
                keys = sorted(vg.keys(), key=lambda x: int(x))
                if not keys:
                    continue
                h5_weights = [np.array(vg[k]) for k in keys]
                model_weights = layer.get_weights()
                
                # Verifikasi shape cocok sebelum set_weights
                if len(h5_weights) == len(model_weights):
                    shapes_match = all(
                        h5_w.shape == m_w.shape 
                        for h5_w, m_w in zip(h5_weights, model_weights)
                    )
                    if shapes_match:
                        layer.set_weights(h5_weights)
                        loaded_count += 1
        
        # 2. Khusus BatchNormalization top classifier: cari yang shape (256,)
        #    Karena di h5 file, 'batch_normalization' merujuk ke base model BN (shape 64),
        #    bukan top classifier BN (shape 256). Perlu cari nama yang benar.
        top_bn_layer = model.get_layer('batch_normalization')
        top_bn_expected_shape = top_bn_layer.get_weights()[0].shape if top_bn_layer.get_weights() else None
        
        if top_bn_expected_shape is not None:
            def _natural_sort_key(s):
                return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]
            
            bn_names = sorted(
                [n for n in h5_layers.keys() if n.startswith('batch_normalization')],
                key=_natural_sort_key
            )
            
            for bn_name in bn_names:
                if 'vars' in h5_layers[bn_name]:
                    vg = h5_layers[bn_name]['vars']
                    keys = sorted(vg.keys(), key=lambda x: int(x))
                    if keys and np.array(vg['0']).shape == top_bn_expected_shape:
                        h5_weights = [np.array(vg[k]) for k in keys]
                        if len(h5_weights) == len(top_bn_layer.get_weights()):
                            top_bn_layer.set_weights(h5_weights)
                            loaded_count += 1
                            break

# Alias untuk kompatibilitas impor dengan simulasi_tester.py
load_exact_weights = load_exact_weights_robust

# MODEL LOAD CACHED IN STREAMLIT
@st.cache_resource(show_spinner=False)
def get_gui_model(variant_name, optimizer_name, fold_num):
    cfg = MODEL_CONFIGS[variant_name]
    pattern = cfg["patterns"][optimizer_name]
    weight_filename = f"{pattern}fold{fold_num}.weights.h5"
    weight_path = os.path.join(cfg["folder"], weight_filename)
    
    tf.keras.backend.clear_session()
    model, img_size = build_exact_training_model(variant_name)
    load_exact_weights_robust(model, weight_path)
    return model, img_size, weight_path

# ====================================================================
# SIDEBAR NAVIGATION & CONFIGURATION
# ====================================================================
st.sidebar.title("🛠️ Konfigurasi Model")

selected_variant = st.sidebar.selectbox(
    "1. Arsitektur Model:",
    options=list(MODEL_CONFIGS.keys()),
    index=2  # Default EfficientNet-B7
)

selected_opt = st.sidebar.selectbox(
    "2. Optimizer:",
    options=["Baseline (RMSprop)", "Adam", "SGD", "AdamW"],
    index=3  # Default AdamW
)

cfg_current = MODEL_CONFIGS[selected_variant]
best_fold_num = cfg_current["best_folds"][selected_opt]

st.sidebar.markdown("---")

# Load model Single Best Fold
try:
    with st.spinner(f"Memuat model Fold {best_fold_num} (Fold Terbaik)..."):
        model, img_size, loaded_weight_path = get_gui_model(selected_variant, selected_opt, best_fold_num)
    st.sidebar.success(f"✅ **Model Siap Digunakan!**\n\n- Arsitektur: **{selected_variant}**\n- Optimizer: **{selected_opt}**\n- Best Fold: **Fold {best_fold_num}**")
except Exception as err:
    st.sidebar.error(f"❌ Gagal memuat model: {str(err)}")
    st.stop()

# ====================================================================
# MAIN DASHBOARD UI
# ====================================================================
st.title("🖼️ Sistem Deteksi Citra Wajah (AI vs Asli)")
st.caption(f"🎯 **Model Aktif**: {selected_variant} + {selected_opt} — **Best Fold {best_fold_num}**")

st.markdown(
    """
    Aplikasi pengujian deteksi keaslian citra berbasis **EfficientNet (Transfer Learning)**. 
    Unggah foto wajah untuk memprediksi apakah citra tersebut merupakan **Foto Asli** atau **Foto Generasi AI**.
    """
)

st.markdown("---")

uploaded_files = st.file_uploader(
    "📤 Unggah Gambar (JPG, JPEG, PNG, WEBP)",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)

if uploaded_files:
    st.markdown(f"### 📊 Hasil Pengujian ({len(uploaded_files)} Gambar)")
    
    c1, c2, c3 = st.columns(3)
    total_ph = c1.empty()
    asli_ph = c2.empty()
    fake_ph = c3.empty()
    
    total_count = len(uploaded_files)
    count_asli = 0
    count_ai = 0
    
    st.markdown("---")
    
    for uploaded_file in uploaded_files:
        try:
            raw_img = Image.open(uploaded_file)
            img = ImageOps.exif_transpose(raw_img).convert("RGB")
            
            # Preprocessing: Resize & Float32 conversion [0..255]
            img_resized = img.resize(img_size, getattr(Image, 'Resampling', Image).NEAREST)
            img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)  # Shape (1, H, W, 3)
            
            t0 = time.time()
            preds = model(img_array, training=False).numpy()[0]
            t_elapsed = time.time() - t0
            
            # Prediksi menggunakan Argmax pada output Dense(2) Softmax
            pred_class_idx = int(np.argmax(preds))  # 0: AI, 1: Asli
            
            prob_ai = float(preds[0])
            prob_asli = float(preds[1])
            
            if pred_class_idx == 1:
                pred_label = "ASLI (REAL)"
                confidence = prob_asli * 100
                count_asli += 1
                color = "#28a745"
            else:
                pred_label = "AI (GENERATED)"
                confidence = prob_ai * 100
                count_ai += 1
                color = "#dc3545"
                
            # Render Row Output
            col_img, col_info, col_prob = st.columns([2, 3, 4])
            
            with col_img:
                st.image(img, use_container_width=True)
                
            with col_info:
                st.markdown(f"**Nama File**: `{uploaded_file.name}`")
                st.markdown(f"**Resolusi Input**: `{img.size[0]} x {img.size[1]} px`")
                st.markdown(f"**Waktu Respon**: `{t_elapsed*1000:.1f} ms`")
                st.markdown(f"**Metode**: `Single Best Fold (Fold {best_fold_num})`")
                
            with col_prob:
                st.markdown(f"### Hasil: <span style='color:{color};'>{pred_label}</span>", unsafe_allow_html=True)
                st.markdown(f"**Tingkat Keyakinan**: **{confidence:.2f}%**")
                st.progress(int(confidence))
                st.caption(f"🔵 Prob. Asli: **{prob_asli*100:.2f}%** | 🔴 Prob. AI: **{prob_ai*100:.2f}%**")
                
            st.markdown("---")
            
        except Exception as e:
            st.error(f"Gagal memproses file `{uploaded_file.name}`: {str(e)}")
            
    # Metric updates
    total_ph.metric("Total Gambar Diuji", f"{total_count} Citra")
    asli_ph.metric("Terdeteksi ASLI", f"{count_asli} Citra", delta=f"{count_asli/total_count*100:.1f}%")
    fake_ph.metric("Terdeteksi AI", f"{count_ai} Citra", delta=f"{count_ai/total_count*100:.1f}%", delta_color="inverse")

else:
    st.info("💡 **Petunjuk**: Gunakan area pengunggah di atas untuk memasukkan 1 atau beberapa foto sekaligus.")
