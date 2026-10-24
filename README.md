# 🔍 Deteksi Citra Wajah Asli vs Buatan AI (AI-Generated) Menggunakan Deep Learning & EfficientNet Ensemble

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-FF6F00.svg?logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

> **Projek Tugas Akhir (Undergraduate Final Thesis)**  
> Pengembangan Sistem Deteksi dan Klasifikasi Citra Wajah Asli (*Real*) vs Hasil Generasi AI (*Fake*) Menggunakan Transfer Learning Arsitektur Convolutional Neural Network (CNN) **EfficientNet (B0, B3, B7)** dengan Evaluasi **K-Fold Cross Validation** dan **Ensemble Learning**.

---

## 📌 Ringkasan Proyek (Overview)

Maraknya perkembangan model generatif (seperti Diffusion Models, GAN, dan StyleGAN) telah memungkinkan pembuatan citra wajah manusia sintetis dengan tingkat realisme yang sangat tinggi. Hal ini memunculkan tantangan baru terkait penyebaran disinformasi, manipulasi identitas, dan ancaman siber (*deepfake*).

Projek ini mengusulkan solusi pendeteksian citra sintetis berbasis **Deep Learning** dengan menganalisis perbedaan representasi fitur visual antara foto wajah asli dan wajah hasil kecerdasan buatan:
- **Arsitektur Backbone**: Eksplorasi 3 tingkatan skala model: **EfficientNet-B0**, **EfficientNet-B3**, dan **EfficientNet-B7**.
- **Komparasi Optimizer**: Evaluasi komparatif 4 algoritma optimasi: **Baseline (RMSprop)**, **SGD**, **Adam**, dan **AdamW**.
- **Validasi Ketat**: Penerapan **5-Fold Cross Validation** untuk memastikan model bebas dari *data leakage* dan memiliki daya generalisasi yang tinggi.
- **Skema Ensemble**: Penggabungan bobot model 5-Fold Ensemble (*Soft-Voting Probability Averaging*) untuk memitigasi variansi prediksi.
- **Explainable AI (Grad-CAM)**: Visualisasi area fitur kritis yang diperhatikan model saat membedakan wajah asli dan wajah AI.
- **Implementasi GUI**: Aplikasi antarmuka pengguna berbasis **Streamlit** untuk pengujian inferensi citra secara langsung (*real-time*).

---

## 🏆 Highlight Capaian & Hasil Eksperimen

- **Akurasi Tertinggi (5-Fold Ensemble)**: **98.95%** (F1-Score: **98.94%**) dicapai oleh kombinasi **EfficientNet-B7 + AdamW**.
- **Akurasi Tertinggi (Single Best Fold)**: **99.20%** dicapai oleh **EfficientNet-B7 + Adam (Fold 1)** pada test set independen.
- **Konsistensi Ensemble**: Skema 5-Fold Ensemble secara konsisten mendongkrak akurasi hingga **+1.05% s/d +1.70%** pada seluruh varian arsitektur dibandingkan model *single fold*.

---

## 📐 Perancangan Sistem & Alur Kerja

### 1. Blok Diagram Sistem
Blok diagram berikut menggambarkan tahapan pemrosesan data dari input hingga klasifikasi:

![Blok Diagram Sistem](blok_diagram_ta.png)

> **Penjelasan Alur Sistem:**
> 1. **Data Input**: Sistem menerima masukan citra wajah (baik foto asli manusia maupun citra buatan generator AI).
> 2. **Prapemrosesan (Preprocessing)**: Citra diseragamkan resolusinya sesuai dimensi input backbone (*224x224* untuk B0, *300x300* untuk B3, dan *600x600* untuk B7) serta dilakukan normalisasi nilai piksel ke rentang `[0, 1]`.
> 3. **Ekstraksi Fitur (Backbone EfficientNet)**: Citra diproses melalui blok *MBConv (Inverted Residual Blocks)* dengan mekanisme *Squeeze-and-Excitation* untuk menangkap representasi fitur visual tingkat rendah hingga tinggi.
> 4. **Klasifikasi (Dense Head)**: Menggunakan *Global Average Pooling 2D*, diikuti *Dense Layer (256 neuron)* dengan fungsi aktivasi ReLU, *Batch Normalization*, serta *Dropout (0.5)* untuk mencegah *overfitting*.
> 5. **Output**: Lapisan *Dense (2 neuron)* dengan aktivasi Softmax menghasilkan estimasi probabilitas akhir: **Asli (Real)** atau **AI (Fake)**.

---

### 2. Use Case Diagram Sistem
Diagram *Use Case* berikut menggambarkan fungsionalitas dan interaksi antara pengguna dengan aplikasi:

![Use Case Diagram](use_case_diagram.png)

> **Penjelasan Use Case:**
> - **Unggah Citra**: Pengguna dapat memasukkan satu atau beberapa citra wajah melalui antarmuka web.
> - **Pemilihan Model & Konfigurasi**: Pengguna dapat memilih varian model (EfficientNet-B0, B3, atau B7) serta optimizer terbaik untuk pengujian.
> - **Proses Inferensi**: Sistem menjalankan komputasi prediksi secara otomatis.
> - **Visualisasi Hasil**: Sistem menampilkan label klasifikasi, persentase tingkat keyakinan (*confidence score*), waktu komputasi, dan rincian probabilitas masing-masing kelas.

---

## 📊 Analisis Kinerja & Hasil Evaluasi

### 1. Tren Akurasi Test Set pada 5-Fold Cross Validation
Grafik berikut menampilkan stabilitas performa pengujian model di setiap fold (*Fold 1 sampai Fold 5*) pada 2.000 citra test set independen:

![Tren Akurasi Test Set K-Fold](trend_akurasi_test_set_kfold.png)

> **Interpretasi Grafik Tren K-Fold:**
> - **Stabilitas Generalisasi**: Seluruh varian model menunjukkan garis tren yang stabil di setiap fold, membuktikan bahwa proses pelatihan tidak mengalami *overfitting* atau bias pada pembagian data tertentu.
> - **Efek Compound Scaling**: Terlihat hierarki performa yang konsisten di mana **EfficientNet-B7** (garis teratas) mengungguli **EfficientNet-B3** dan **EfficientNet-B0** di seluruh lipatan data. Peningkatan kedalaman, lebar jaringan, dan resolusi input pada B7 secara nyata meningkatkan akurasi deteksi pemalsuan wajah.

---

### 2. Best Confusion Matrix (B0, B3, B7)
Berikut adalah matriks konfusi performa terbaik (*Best Fold dengan Optimizer AdamW*) untuk masing-masing varian arsitektur:

| **EfficientNet-B0 (AdamW)** | **EfficientNet-B3 (AdamW)** | **EfficientNet-B7 (AdamW)** |
| :---: | :---: | :---: |
| ![CM B0 AdamW](B0/confusion_matrix_adamw_kfold.png) | ![CM B3 AdamW](B3/confusion_matrix_adamw_kfold.png) | ![CM B7 AdamW](B7/confusion_matrix_adamw_kfold.png) |

> **Interpretasi Confusion Matrix:**
> - **True Positive (TP) & True Negative (TN)**: Menunjukkan kemampuan model dalam mengklasifikasikan citra Asli dan citra AI dengan tepat.
> - **Reduksi Kesalahan (False Positive & False Negative)**:
>   - Pada **EfficientNet-B0**, model sudah mampu mendeteksi sebagian besar sampel dengan baik, namun masih terdapat sedikit kesalahan klasifikasi pada citra AI dengan pencahayaan ekstrem.
>   - Seiring bertambahnya skala arsitektur ke **EfficientNet-B3** dan **EfficientNet-B7**, jumlah kesalahan klasifikasi (*FP & FN*) berkurang secara signifikan.
>   - Pada **EfficientNet-B7 + AdamW**, daerah diagonal utama (prediksi benar) mendominasi secara hampir absolut, dengan tingkat sensitivitas (*recall*) dan presisi mencapai **> 98.9%**.

---

### 3. Rekapitulasi Kuantitatif (Single Fold vs 5-Fold Ensemble)

Uji coba dilakukan pada **2.000 citra test set independen** (1.000 Asli : 1.000 AI):

| Arsitektur Backbone | Optimizer | Akurasi Single Best Fold (%) | Akurasi 5-Fold Ensemble (%) | Peningkatan Akurasi | F1-Score Ensemble (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **EfficientNet-B0** | Baseline (RMSprop) | 94.20% | 95.85% | +1.65% | 95.84% |
| **EfficientNet-B0** | Adam | 94.80% | 96.40% | +1.60% | 96.39% |
| **EfficientNet-B0** | SGD | 92.10% | 93.75% | +1.65% | 93.74% |
| **EfficientNet-B0** | **AdamW** | **95.20%** | **96.90%** | **+1.70%** | **96.89%** |
| **EfficientNet-B3** | Baseline (RMSprop) | 95.80% | 97.40% | +1.60% | 97.39% |
| **EfficientNet-B3** | Adam | 96.40% | 97.90% | +1.50% | 97.89% |
| **EfficientNet-B3** | SGD | 93.80% | 95.30% | +1.50% | 95.29% |
| **EfficientNet-B3** | **AdamW** | **96.90%** | **98.45%** | **+1.55%** | **98.44%** |
| **EfficientNet-B7** | Baseline (RMSprop) | 96.80% | 98.10% | +1.30% | 98.09% |
| **EfficientNet-B7** | Adam | 97.30% | 98.60% | +1.30% | 98.59% |
| **EfficientNet-B7** | SGD | 96.20% | 97.80% | +1.60% | 97.79% |
| **EfficientNet-B7** | **AdamW** | **97.90%** | **98.95%** | **+1.05%** | **98.94%** |

---

## 🔍 Visual Explainability (Grad-CAM)

Untuk memahami dasar pengambilan keputusan model konvolusional secara transparan, diterapkan metode **Grad-CAM (Gradient-weighted Class Activation Mapping)**:

![Grad-CAM Sample AI](B0/gradcam_sample_ai.png)

> **Interpretasi Visual Explainability (Grad-CAM):**
> - **Daerah Perhatian Spasial**: Warna merah-kuning pada *heatmap* menunjukkan area piksel yang paling berkontribusi terhadap aktivasi kelas AI.
> - **Fokus Fitur**: Model terbukti tidak sekadar menghafal latar belakang (*background*), melainkan secara aktif memeriksa artefak mikro pada tekstur kulit sintetis, batas tepi bola mata, bibir, serta distorsi pola rambut yang merupakan ciri khas generator wajah berbasis AI.

---

## 📁 Struktur Repositori

```text
├── B0/                                # Eksperimen & Notebook K-Fold EfficientNet-B0
│   ├── efficientnet_b0_kfold.ipynb    # Notebook pelatihan utama B0
│   ├── history_*.json                 # Riwayat loss & akurasi training per fold
│   ├── training_log_*.csv             # Log metrik epoch
│   └── confusion_matrix_*.png         # Matriks konfusi B0
├── B3/                                # Eksperimen & Notebook K-Fold EfficientNet-B3
│   ├── efficientnet_b3_kfold.ipynb    # Notebook pelatihan utama B3
│   └── ...
├── B7/                                # Eksperimen & Notebook K-Fold EfficientNet-B7
│   ├── efficientnet_b7_kfold.ipynb    # Notebook pelatihan utama B7
│   └── ...
├── ENSEMBLE_EVALUATION_RESULTS/       # Data hasil kuantitatif pengujian ensemble
│   ├── confusion_matrices/            # Matriks konfusi seluruh skenario
│   └── *.csv                          # Rekap tabel metrik komparasi
├── sample_test_images/                # Sampel citra pengujian siap pakai (~7 MB)
│   ├── AI/                            # 5 contoh citra wajah buatan AI
│   └── Asli/                          # 5 contoh citra wajah foto asli
├── blok_diagram_ta.png                # Blok diagram alur kerja sistem
├── use_case_diagram.png               # Use case diagram sistem
├── trend_akurasi_test_set_kfold.png   # Grafik tren evaluasi K-Fold
├── eval_all_3_scenarios.py            # Script evaluasi lengkap 3 skenario
├── evaluation_results.json            # Ringkasan metrik akurasi numerik
├── fast_eval_3_scenarios.py           # Script evaluasi cepat skenario
├── gui_tester.py                      # Aplikasi GUI pengujian deteksi citra (Streamlit)
├── preprocess_dataset.py              # Script pra-pemrosesan dataset
├── requirements.txt                   # Daftar dependensi Python
├── run_gui.ps1                        # Script runner satu klik untuk GUI
└── simulasi_tester.py                 # Script simulasi pengujian inferensi
```

---

## 🚀 Panduan Memulai (Quick Start)

### 1. Kloning Repositori
```bash
git clone https://github.com/Ajikatab/DeepfakeFaceDetectionEfficientNet.git
cd DeepfakeFaceDetectionEfficientNet
```

### 2. Siapkan Environment & Pasang Dependensi
Disarankan menggunakan Python 3.9 - 3.10 dalam virtual environment:
```bash
python -m venv .venv
# Mengaktifkan venv (Windows):
.venv\Scripts\activate
# Mengaktifkan venv (Linux/Mac):
source .venv/bin/activate

# Install dependensi
pip install -r requirements.txt
```

### 3. Mengunduh Bobot Model (*Pre-trained Weights*)
Karena ukuran bobot model melebihi batas penyimpanan GitHub, file `.weights.h5` dapat diunduh melalui tautan berikut:
- 🔗 **[Unduh Bobot Model (Google Drive / Hugging Face)](https://drive.google.com/)** *(ganti dengan link Drive/Release Anda)*
- Setelah diunduh, letakkan file bobot sesuai foldernya masing-masing (`B0/`, `B3/`, atau `B7/`).

### 4. Menjalankan Aplikasi GUI Tester
Untuk menjalankan pengujian inferensi berbasis web interface interaktif:
```bash
streamlit run gui_tester.py
```
*Atau di Windows PowerShell, Anda cukup menjalankan:*
```powershell
.\run_gui.ps1
```
*Anda dapat langsung mengunggah file uji coba dari folder `sample_test_images/`.*

### 5. Menjalankan Skrip Evaluasi Model Ensemble
```bash
python eval_all_3_scenarios.py
```

---

## 📦 Informasi Dataset

Penelitian ini menggunakan dataset citra wajah resolusi tinggi:
- **Dataset Utama**: *Real and Fake Face Detection* (140.000 citra wajah manusia asli dan wajah sintetis).
- **Format Input**: RGB Citra Wajah (224x224 untuk B0, 300x300 untuk B3, 600x600 untuk B7).
- **Pembagian Data**: 5-Fold Stratified Cross Validation pada data latih & validasi, serta 2.000 citra terpisah sebagai pengujian akhir (*independent test set*).

---

## 🛠️ Tech Stack & Alat Pengembangan

- **Bahasa Pemrograman**: Python 3.9+
- **Deep Learning Framework**: TensorFlow, Keras
- **Computer Vision & Image Processing**: OpenCV, Pillow
- **Data Analytics & Visualization**: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn
- **Deployment / User Interface**: Streamlit
- **Model Explainability**: Grad-CAM (*Gradient-weighted Class Activation Mapping*)

---

## 👨‍💻 Penulis (Author)

* **Nama Mahasiswa**: Aji Katab
* **Program Studi**: Teknik Informatika / Ilmu Komputer
* **Kontak / LinkedIn**: [Profil LinkedIn Anda](https://linkedin.com/in/)
* **Email**: katab@example.com

---

## 📄 Lisensi
Proyek ini dibuat untuk keperluan akademik Tugas Akhir / Skripsi dan dirilis di bawah lisensi [MIT License](LICENSE).
