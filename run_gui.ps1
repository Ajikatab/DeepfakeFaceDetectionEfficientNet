$CONDA_ENV = "$env:USERPROFILE\miniconda3\envs\tf-gpu"
$CUDA_PATH = "$CONDA_ENV\Library\bin"
$PYTHON = "$CONDA_ENV\python.exe"
$SCRIPT = "gui_tester.py"

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  Aplikasi Web GUI Pengujian Gambar Real vs Fake (AI) - Tugas Akhir  " -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan

# Set environment variables
$env:PATH = "$CUDA_PATH;$env:PATH"
$env:TF_ENABLE_ONEDNN_OPTS = "0"
$env:TF_CPP_MIN_LOG_LEVEL = "2"

# Memaksa TensorFlow menggunakan CPU untuk GUI
# Menghindari inisialisasi CUDA yang lambat (GUI terbuka instan <2 detik) dan mencegah OOM
$env:CUDA_VISIBLE_DEVICES = "-1"

Write-Host "[INFO] Path CUDA berhasil disiapkan." -ForegroundColor Green
Write-Host "[INFO] Memaksa mode CPU untuk kecepatan startup GUI instan." -ForegroundColor Green

# Cek apakah library streamlit sudah terinstal di environment conda
Write-Host "[INFO] Memeriksa instalasi Streamlit..." -ForegroundColor Yellow
& $PYTHON -c "import streamlit" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[INFO] Streamlit belum terinstal. Menginstal Streamlit ke environment conda..." -ForegroundColor Yellow
    & $PYTHON -m pip install streamlit -q
    Write-Host "[OK] Streamlit berhasil diinstal!" -ForegroundColor Green
} else {
    Write-Host "[OK] Streamlit sudah terinstal." -ForegroundColor Green
}

Write-Host "[INFO] Membuka Aplikasi Web Localhost di Browser..." -ForegroundColor Yellow
Write-Host ""

# Menjalankan Streamlit secara terprogram melalui module python
& $PYTHON -m streamlit run $SCRIPT

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  Web GUI Ditutup." -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Cyan
Read-Host "Tekan Enter untuk keluar"
