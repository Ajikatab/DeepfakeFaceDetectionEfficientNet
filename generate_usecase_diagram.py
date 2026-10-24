"""Generate a UML use-case diagram for the AI-vs-real image classifier."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle


OUTPUT_PATH = Path(__file__).parent / "usecase_diagram_sistem.png"


def draw_actor(axis, x, y, label):
    """Draw a simple UML stick-figure actor."""
    axis.plot(x, y + 0.22, marker="o", markersize=14, color="#17324d", zorder=5)
    axis.plot([x, x], [y - 0.02, y + 0.16], color="#17324d", linewidth=2)
    axis.plot([x - 0.09, x + 0.09], [y + 0.10, y + 0.10], color="#17324d", linewidth=2)
    axis.plot([x, x - 0.09], [y - 0.02, y - 0.14], color="#17324d", linewidth=2)
    axis.plot([x, x + 0.09], [y - 0.02, y - 0.14], color="#17324d", linewidth=2)
    axis.text(x, y - 0.25, label, ha="center", va="top", fontsize=11, weight="bold")


def draw_use_case(axis, x, y, width, height, label, color="#e9f3ff"):
    ellipse = Ellipse(
        (x, y), width, height, facecolor=color, edgecolor="#2878c8", linewidth=1.5
    )
    axis.add_patch(ellipse)
    axis.text(x, y, label, ha="center", va="center", fontsize=9, wrap=True)


def connect(axis, start, end, style="-"):
    axis.plot([start[0], end[0]], [start[1], end[1]], color="#405466", linewidth=1.1, linestyle=style, zorder=0)


def main():
    figure, axis = plt.subplots(figsize=(16, 10))
    axis.set_xlim(0, 16)
    axis.set_ylim(0, 10)
    axis.axis("off")
    figure.patch.set_facecolor("white")

    # System boundary
    boundary = Rectangle(
        (3.0, 0.7), 11.0, 8.6, facecolor="#fbfdff", edgecolor="#17324d", linewidth=2
    )
    axis.add_patch(boundary)
    axis.text(
        8.5,
        9.05,
        "Sistem Klasifikasi Citra AI dan Citra Asli",
        ha="center",
        va="center",
        fontsize=15,
        weight="bold",
        color="#17324d",
    )

    draw_actor(axis, 1.45, 7.2, "Pengguna")
    draw_actor(axis, 1.45, 2.6, "Peneliti / Admin")

    user_cases = [
        (5.0, 7.7, "Memilih arsitektur\nB0 / B3 / B7"),
        (8.5, 7.7, "Memilih optimizer\ndan fold"),
        (11.8, 7.7, "Mengunggah citra"),
        (5.0, 6.0, "Melakukan\nklasifikasi citra"),
        (8.5, 6.0, "Melihat hasil\nAI atau Asli"),
        (11.8, 6.0, "Melihat confidence\ndan probabilitas"),
        (8.5, 4.3, "Melihat visualisasi\nGrad-CAM"),
    ]
    for x, y, label in user_cases:
        draw_use_case(axis, x, y, 2.5, 0.85, label)

    research_cases = [
        (5.0, 2.9, "Menyiapkan dataset\nAI dan Asli"),
        (8.5, 2.9, "Membagi data dengan\nStratified 5-Fold"),
        (11.8, 2.9, "Melakukan augmentasi\ndata training"),
        (5.0, 1.45, "Melatih model\nEfficientNet"),
        (8.5, 1.45, "Fine-tuning 30\nlayer terakhir"),
        (11.8, 1.45, "Mengevaluasi dan\nmembandingkan 12 skenario"),
    ]
    for x, y, label in research_cases:
        draw_use_case(axis, x, y, 2.5, 0.85, label, color="#fff3df")

    # Actor associations
    for end in [(3.75, 7.7), (3.75, 6.0), (3.75, 2.9), (3.75, 1.45)]:
        connect(axis, (1.55, 7.2 if end[1] > 5 else 2.6), end)
    connect(axis, (1.55, 7.2), (3.75, 4.3))
    for end in [(13.05, 7.7), (13.05, 6.0), (13.05, 2.9), (13.05, 1.45)]:
        connect(axis, (1.55, 2.6), end)
    connect(axis, (1.55, 2.6), (3.75, 4.3))

    # Include relations inside the system.
    connect(axis, (6.15, 7.25), (6.15, 6.45), "--")
    connect(axis, (9.65, 7.25), (9.65, 6.45), "--")
    connect(axis, (10.0, 6.0), (10.0, 4.75), "--")
    axis.text(10.15, 6.45, "<<include>>", fontsize=8, color="#405466")
    axis.text(10.15, 5.38, "<<extend>>", fontsize=8, color="#405466")

    # Legend and explanatory note.
    axis.text(3.35, 0.92, "Batas sistem", fontsize=8, color="#405466")
    axis.text(
        8.5,
        0.25,
        "Oval biru: fitur penggunaan sistem | Oval oranye: aktivitas penelitian/training",
        ha="center",
        fontsize=9,
        color="#405466",
    )

    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"[OK] Use case diagram tersimpan: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
