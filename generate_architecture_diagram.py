"""Generate an academic swimlane activity diagram for the image classifier."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, Polygon


OUTPUT_PATH = Path(__file__).parent / "arsitektur_sistem_klasifikasi_citra_ai.png"
BLACK = "#222222"
LINE = "#4b5563"
BLUE = "#1683ff"
BLUE_FILL = "#eef6ff"
ORANGE = "#f28c28"
ORANGE_FILL = "#fff4e6"
GREEN = "#35a853"
GREEN_FILL = "#effaf1"
GRAY_FILL = "#f5f6f7"


def draw_box(axis, x, y, width, height, text, fill=GRAY_FILL, edge=BLACK, fontsize=8.7, linewidth=1.2):
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.018,rounding_size=0.045",
        facecolor=fill, edgecolor=edge, linewidth=linewidth,
    )
    axis.add_patch(patch)
    axis.text(x + width / 2, y + height / 2, text, ha="center", va="center",
              fontsize=fontsize, color=BLACK, linespacing=1.15, wrap=True)


def draw_decision(axis, x, y, width, height, text):
    points = [(x + width / 2, y + height), (x + width, y + height / 2),
              (x + width / 2, y), (x, y + height / 2)]
    axis.add_patch(Polygon(points, closed=True, facecolor="white", edgecolor=LINE, linewidth=1.1))
    axis.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=8.1, color=BLACK)


def draw_actor(axis, x, y, label):
    axis.plot(x, y + 0.22, marker="o", markersize=10, markerfacecolor="white", markeredgecolor=BLACK, linewidth=1.2)
    axis.plot([x, x], [y + 0.03, y + 0.16], color=BLACK, linewidth=1.2)
    axis.plot([x - 0.10, x + 0.10], [y + 0.12, y + 0.12], color=BLACK, linewidth=1.2)
    axis.plot([x, x - 0.09], [y + 0.03, y - 0.11], color=BLACK, linewidth=1.2)
    axis.plot([x, x + 0.09], [y + 0.03, y - 0.11], color=BLACK, linewidth=1.2)
    axis.text(x, y - 0.25, label, ha="center", va="top", fontsize=8.5, weight="bold")


def draw_arrow(axis, start, end, label=None, label_offset=(0, 0.10), color=LINE, linestyle="-"):
    axis.annotate("", xy=end, xytext=start,
                  arrowprops={"arrowstyle": "-|>", "color": color, "linewidth": 1.15, "linestyle": linestyle})
    if label:
        x = (start[0] + end[0]) / 2 + label_offset[0]
        y = (start[1] + end[1]) / 2 + label_offset[1]
        axis.text(x, y, label, ha="center", va="center", fontsize=7.2,
                  color=color, backgroundcolor="white")


def draw_label(axis, x, y, text, fontsize=10, weight="normal", color=BLACK):
    axis.text(x, y, text, ha="center", va="center", fontsize=fontsize, weight=weight, color=color)


def main():
    figure, axis = plt.subplots(figsize=(13.333, 7.5), dpi=120)
    figure.patch.set_facecolor("white")
    axis.set_xlim(0, 13.333)
    axis.set_ylim(0, 7.5)
    axis.axis("off")

    draw_label(axis, 6.67, 7.26, "Visualisasi Arsitektur Sistem Klasifikasi Citra AI", fontsize=15, weight="bold")

    # Three academic swimlanes, matching the requested reference style.
    lane_x = [0.30, 4.36, 8.42]
    lane_w = 4.06
    lane_bottom = 0.38
    lane_top = 7.02
    headers = ["Pengguna / User", "Sistem Aplikasi", "Pipeline Model EfficientNet"]
    colors = [BLUE, BLUE, ORANGE]
    fills = ["#f8fbff", "#fbfdff", "#fffdf9"]
    for x, header, color, fill in zip(lane_x, headers, colors, fills):
        axis.add_patch(Rectangle((x, lane_bottom), lane_w, lane_top - lane_bottom,
                                 facecolor=fill, edgecolor=color, linewidth=1.2))
        axis.add_patch(Rectangle((x, 6.72), lane_w, 0.30,
                                 facecolor=color, edgecolor=color, linewidth=1.0))
        draw_label(axis, x + lane_w / 2, 6.87, header, fontsize=9.0, weight="bold", color="white")

    # User lane.
    draw_actor(axis, 2.33, 5.82, "Pengguna / User")
    draw_box(axis, 1.05, 4.92, 2.56, 0.56, "1. Memasukkan / Mengunggah\nCitra Wajah", fill=BLUE_FILL, edge=BLUE, fontsize=8.3)
    draw_box(axis, 1.05, 2.94, 2.56, 0.62, "Melihat Hasil Prediksi\nAI atau Asli", fill=GREEN_FILL, edge=GREEN, fontsize=8.4, linewidth=1.4)
    draw_box(axis, 1.05, 1.56, 2.56, 0.62, "Melihat Confidence\ndan Probabilitas", fill=GREEN_FILL, edge=GREEN, fontsize=8.4, linewidth=1.4)
    draw_box(axis, 1.05, 0.70, 2.56, 0.58, "Melihat Grad-CAM\n(Opsional)", fill=GREEN_FILL, edge=GREEN, fontsize=8.3, linewidth=1.4)

    # System lane.
    draw_box(axis, 5.10, 5.72, 2.58, 0.58, "2. Menerima dan Memvalidasi\nFile Citra", fill=BLUE_FILL, edge=BLUE)
    draw_box(axis, 5.10, 4.82, 2.58, 0.58, "3. Pra-pemrosesan\nResize dan Array Float32", fill=BLUE_FILL, edge=BLUE)
    draw_box(axis, 5.10, 3.92, 2.58, 0.58, "4. Memuat Model Terlatih\ndan Bobot .weights.h5", fill=BLUE_FILL, edge=BLUE)
    draw_box(axis, 5.10, 3.02, 2.58, 0.58, "5. Menjalankan\nKlasifikasi oleh Model", fill=BLUE_FILL, edge=BLUE)
    draw_box(axis, 5.10, 2.12, 2.58, 0.58, "6. Menghitung Label,\nConfidence, dan Probabilitas", fill=BLUE_FILL, edge=BLUE)
    draw_box(axis, 5.10, 1.22, 2.58, 0.58, "7. Menghasilkan\nVisualisasi Grad-CAM", fill=BLUE_FILL, edge=BLUE)

    draw_arrow(axis, (2.33, 5.78), (5.10, 6.01), "1. Citra", label_offset=(0, 0.15), color=BLUE)
    draw_arrow(axis, (6.39, 5.72), (6.39, 5.40), "2")
    draw_arrow(axis, (6.39, 4.82), (6.39, 4.50), "3")
    draw_arrow(axis, (6.39, 3.92), (6.39, 3.60), "4")
    draw_arrow(axis, (6.39, 3.02), (6.39, 2.70), "5")
    draw_arrow(axis, (6.39, 2.12), (6.39, 1.80), "6")
    draw_arrow(axis, (5.10, 3.31), (3.61, 3.25), "7. Hasil", label_offset=(0, 0.16), color=GREEN)
    draw_arrow(axis, (5.10, 2.41), (3.61, 1.87), "8", label_offset=(0, 0.12), color=GREEN)
    draw_arrow(axis, (5.10, 1.51), (3.61, 0.99), "9", label_offset=(0, 0.12), color=GREEN)

    # Model lane: inference and training paths.
    draw_box(axis, 9.05, 5.77, 2.80, 0.58, "Klasifikasi oleh Model\n(EfficientNet-B0/B3/B7)", fill=ORANGE_FILL, edge=ORANGE)
    draw_box(axis, 9.05, 4.88, 2.80, 0.58, "Ekstraksi Fitur", fill=ORANGE_FILL, edge=ORANGE)
    draw_box(axis, 9.05, 3.99, 2.80, 0.58, "Output Softmax\nProbabilitas AI / Asli", fill=ORANGE_FILL, edge=ORANGE)
    draw_decision(axis, 9.55, 2.98, 1.80, 0.70, "Kelas prediksi?")
    draw_box(axis, 8.75, 2.04, 1.72, 0.55, "Gambar\nBuatan AI", fill=GREEN_FILL, edge=GREEN, fontsize=8.2, linewidth=1.3)
    draw_box(axis, 10.75, 2.04, 1.72, 0.55, "Gambar\nAsli", fill=GREEN_FILL, edge=GREEN, fontsize=8.2, linewidth=1.3)
    draw_arrow(axis, (7.68, 3.31), (9.05, 6.06), "10. Input", label_offset=(0.12, 0.0), color=ORANGE)
    draw_arrow(axis, (10.45, 5.77), (10.45, 5.46), "11")
    draw_arrow(axis, (10.45, 4.88), (10.45, 4.57), "12")
    draw_arrow(axis, (10.45, 3.99), (10.45, 3.68), "13")
    draw_arrow(axis, (10.45, 2.98), (9.61, 2.59), "AI", label_offset=(-0.08, 0.08), color=GREEN)
    draw_arrow(axis, (10.45, 2.98), (11.61, 2.59), "Asli", label_offset=(0.10, 0.08), color=GREEN)

    # Training flow below the inference branch in the same model lane.
    axis.plot([8.65, 12.20], [1.70, 1.70], color=BLACK, linewidth=2.5)
    draw_box(axis, 8.75, 0.70, 1.72, 0.58, "Dataset\nAI dan Asli", fill=ORANGE_FILL, edge=ORANGE, fontsize=8.1)
    draw_box(axis, 10.75, 0.70, 1.72, 0.58, "Model Terlatih\n(Bobot H5)", fill=GREEN_FILL, edge=GREEN, fontsize=8.1, linewidth=1.4)
    draw_arrow(axis, (9.61, 1.70), (9.61, 1.28), "14. Training", label_offset=(0.30, 0.05), color=ORANGE)
    draw_arrow(axis, (10.47, 0.99), (10.75, 0.99), "15", color=ORANGE)
    draw_arrow(axis, (11.61, 1.28), (11.61, 1.70), "16. Digunakan", label_offset=(0.35, 0.02), color=GREEN)
    axis.text(10.42, 1.80, "Training: Stratified 5-Fold -> Augmentasi -> Fine-tuning 30 Layer -> Optimizer",
              ha="center", va="bottom", fontsize=7.0, color=BLACK, backgroundcolor="white")
    axis.text(10.42, 0.49, "Optimizer: RMSprop | SGD | Adam | AdamW", ha="center", va="center", fontsize=7.1, color=LINE)

    # Connect trained model to the inference model with a one-way line.
    draw_arrow(axis, (11.61, 1.28), (11.61, 5.77), "Model terlatih digunakan untuk prediksi", label_offset=(0.58, 0), color=GREEN, linestyle="--")

    figure.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[OK] Diagram tersimpan di: {OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
