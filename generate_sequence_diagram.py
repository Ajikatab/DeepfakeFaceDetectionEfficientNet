"""Generate an academic sequence diagram for the AI image classifier."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


OUTPUT_PATH = Path(__file__).parent / "sequence_diagram_klasifikasi_citra_ai.png"
BLACK = "#222222"
GRAY = "#68737d"
BLUE = "#2878c8"
BLUE_FILL = "#eef6ff"
ORANGE = "#c97918"
ORANGE_FILL = "#fff5e6"
GREEN = "#2f855a"
GREEN_FILL = "#edf8f0"
RED = "#b23b3b"
RED_FILL = "#fff0f0"


def draw_participant(axis, x, y, width, height, text, fill=BLUE_FILL, edge=BLACK):
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        facecolor=fill, edgecolor=edge, linewidth=1.1,
    )
    axis.add_patch(patch)
    axis.text(x, y, text, ha="center", va="center", fontsize=7.4,
              weight="bold", color=BLACK, linespacing=1.1)


def draw_lifeline(axis, x, top, bottom, color=GRAY):
    axis.plot([x, x], [top, bottom], color=color, linewidth=0.8,
              linestyle=(0, (3, 3)), zorder=0)


def draw_activation(axis, x, y, height, color=BLUE):
    axis.add_patch(Rectangle((x - 0.035, y - height / 2), 0.07, height,
                             facecolor=color, edgecolor=color, linewidth=0.5,
                             alpha=0.82, zorder=2))


def draw_message(axis, start, end, y, number, text, dashed=False,
                 color=BLACK, label_offset=0.075):
    linestyle = "--" if dashed else "-"
    axis.annotate(
        "", xy=(end, y), xytext=(start, y),
        arrowprops={
            "arrowstyle": "-|>", "color": color, "linewidth": 0.95,
            "linestyle": linestyle,
        },
    )
    prefix = f"{number}. " if number else ""
    axis.text((start + end) / 2, y + label_offset, prefix + text,
              ha="center", va="bottom", fontsize=6.7, color=color,
              backgroundcolor="white")


def draw_self_message(axis, x, y, number, text, height=0.34, color=BLACK):
    """Draw a looped message sent to the same participant."""
    left = x + 0.04
    right = x + 0.38
    axis.plot([left, right, right, left], [y, y, y - height, y - height],
              color=color, linewidth=0.9)
    axis.annotate("", xy=(left, y - height), xytext=(left, y - height),
                  arrowprops={"arrowstyle": "-|>", "color": color, "linewidth": 0.9})
    axis.text(right + 0.08, y - height / 2, f"{number}. {text}",
              ha="left", va="center", fontsize=6.4, color=color,
              backgroundcolor="white")


def draw_fragment(axis, x, y, width, height, title, edge=GRAY, fill="none"):
    axis.add_patch(Rectangle((x, y), width, height, facecolor=fill,
                             edgecolor=edge, linewidth=0.9))
    axis.add_patch(Rectangle((x, y + height - 0.30), width, 0.30,
                             facecolor="white", edgecolor=edge, linewidth=0.9))
    axis.text(x + 0.14, y + height - 0.15, title, ha="left", va="center",
              fontsize=7.5, weight="bold", color=edge)


def draw_note(axis, x, y, width, height, text, fill="#fafafa", edge=GRAY):
    axis.add_patch(Rectangle((x, y), width, height, facecolor=fill,
                             edgecolor=edge, linewidth=0.8))
    axis.text(x + width / 2, y + height / 2, text, ha="center", va="center",
              fontsize=6.8, color=BLACK, wrap=True)


def main():
    figure, axis = plt.subplots(figsize=(16, 10), dpi=120)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    axis.set_xlim(0, 16)
    axis.set_ylim(0, 10)
    axis.axis("off")

    axis.text(8, 9.72, "Sequence Diagram Sistem Klasifikasi Citra AI",
              ha="center", va="center", fontsize=15, weight="bold", color=BLACK)
    axis.text(8, 9.43, "Alur Prediksi Menggunakan Model EfficientNet Terlatih",
              ha="center", va="center", fontsize=8.5, color=GRAY)

    participant_x = [1.05, 3.20, 5.55, 7.90, 10.25, 12.55, 14.75]
    participant_data = [
        ("Pengguna /\nUser", BLUE_FILL, BLUE),
        ("GUI Website", BLUE_FILL, BLUE),
        ("Preprocessing\nCitra", ORANGE_FILL, ORANGE),
        ("Model EfficientNet\nTerlatih", ORANGE_FILL, ORANGE),
        ("Ekstraksi\nFitur", ORANGE_FILL, ORANGE),
        ("Klasifikasi\nModel", GREEN_FILL, GREEN),
        ("Hasil Prediksi\nAI / Asli", GREEN_FILL, GREEN),
    ]
    for x, (text, fill, edge) in zip(participant_x, participant_data):
        draw_participant(axis, x, 8.92, 1.75, 0.62, text, fill, edge)
        draw_lifeline(axis, x, 8.58, 0.72)

    # Activation bars for the prediction interaction.
    draw_activation(axis, participant_x[1], 7.10, 5.85, BLUE)
    draw_activation(axis, participant_x[2], 6.55, 3.55, ORANGE)
    draw_activation(axis, participant_x[3], 5.65, 4.25, ORANGE)
    draw_activation(axis, participant_x[4], 5.95, 1.00, ORANGE)
    draw_activation(axis, participant_x[5], 4.55, 2.20, GREEN)
    draw_activation(axis, participant_x[6], 3.70, 1.05, GREEN)

    # Main prediction messages.
    draw_message(axis, participant_x[0], participant_x[1], 8.08, 1, "Upload citra wajah", color=BLUE)
    draw_message(axis, participant_x[1], participant_x[2], 7.72, 2, "Kirim citra untuk diproses", color=BLUE)
    draw_self_message(axis, participant_x[2], 7.32, 3, "Resize citra")
    draw_self_message(axis, participant_x[2], 6.72, 4, "Konversi citra ke array")
    draw_self_message(axis, participant_x[2], 6.12, 5, "Preprocessing input model")
    draw_message(axis, participant_x[2], participant_x[3], 5.48, 6, "Citra terproses", color=ORANGE)
    draw_message(axis, participant_x[3], participant_x[4], 5.02, 7, "Forward pass / ekstraksi fitur", color=ORANGE)
    draw_message(axis, participant_x[4], participant_x[3], 4.56, 8, "Fitur hasil ekstraksi", dashed=True, color=GRAY)
    draw_message(axis, participant_x[3], participant_x[5], 4.10, 9, "Klasifikasi citra", color=GREEN)
    draw_message(axis, participant_x[5], participant_x[6], 3.64, 10, "Label dan probabilitas", color=GREEN)

    # Alternative fragment for the two output classes.
    draw_fragment(axis, 0.60, 1.98, 14.82, 1.38, "alt", edge=GRAY)
    axis.text(0.84, 2.74, "[probabilitas AI >= probabilitas Asli]", fontsize=7.0, color=RED)
    axis.text(0.84, 2.42, "[probabilitas Asli > probabilitas AI]", fontsize=7.0, color=GREEN)
    draw_message(axis, participant_x[5], participant_x[6], 2.66, None,
                 "Gambar Buatan AI", color=RED, label_offset=0.05)
    draw_message(axis, participant_x[5], participant_x[6], 2.34, None,
                 "Gambar Asli", color=GREEN, label_offset=0.05)
    draw_message(axis, participant_x[6], participant_x[1], 1.64, 11,
                 "Hasil prediksi", dashed=True, color=GREEN, label_offset=0.08)
    draw_message(axis, participant_x[1], participant_x[0], 1.20, 12,
                 "Tampilkan hasil prediksi", dashed=True, color=BLUE, label_offset=0.08)

    # Offline training fragment.
    draw_fragment(axis, 0.60, 0.12, 14.82, 0.88, "Proses Training Model (Offline)", edge=ORANGE_FILL, fill="white")
    train_y = 0.72
    training_steps = [
        ("Dataset AI dan Asli", "Stratified 5-Fold"),
        ("Pra-pemrosesan + augmentasi", "EfficientNet-B0/B3/B7"),
        ("Training + fine-tuning", "RMSprop | SGD | Adam | AdamW"),
        ("Evaluasi model", "Simpan model terlatih"),
    ]
    step_x = [1.00, 4.25, 7.50, 10.75]
    for index, ((left_text, right_text), x) in enumerate(zip(training_steps, step_x), start=1):
        draw_note(axis, x, train_y, 2.50, 0.30, f"{index}. {left_text}\n{right_text}", fill=ORANGE_FILL, edge=ORANGE)
        if index < len(step_x):
            draw_message(axis, x + 2.50, step_x[index], train_y + 0.15, None, "", color=ORANGE, label_offset=0)
    draw_message(axis, 13.25, 7.90, train_y + 0.15, 9,
                 "Model terlatih digunakan pada prediksi", color=GREEN, label_offset=0.07)

    draw_note(axis, 0.82, 0.01, 4.0, 0.09, "Optimizer hanya digunakan pada tahap training.", fill="white", edge=ORANGE)
    draw_note(axis, 5.25, 0.01, 5.25, 0.09, "Pengguna tidak memilih optimizer saat prediksi.", fill="white", edge=BLUE)
    draw_note(axis, 10.95, 0.01, 4.0, 0.09, "Output akhir: Gambar Buatan AI atau Gambar Asli.", fill="white", edge=GREEN)

    figure.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[OK] Sequence diagram tersimpan di: {OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
