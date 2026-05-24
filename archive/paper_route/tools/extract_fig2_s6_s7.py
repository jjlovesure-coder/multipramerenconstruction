from pathlib import Path
from PIL import Image
import csv


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "extracted"
OUT.mkdir(parents=True, exist_ok=True)


def crop_image(src, box, name):
    image = Image.open(ROOT / src)
    crop = image.crop(box)
    path = OUT / name
    crop.save(path)
    return path


def write_csv(name, fieldnames, rows):
    path = OUT / name
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


# Coordinates are from the 300 dpi PDF renders in results/.
fig2_boxes = {
    "fig2a.png": (450, 435, 1215, 1180),
    "fig2b.png": (1215, 435, 2010, 1180),
    "fig2c.png": (450, 1170, 1215, 1940),
    "fig2d.png": (1215, 1170, 2010, 1940),
    "fig2a_d.png": (430, 420, 2020, 1955),
}

for name, box in fig2_boxes.items():
    crop_image("results/main_fig2_page-13.png", box, name)

crop_image("results/supp_S6_S7_page-11.png", (930, 360, 1515, 1000), "figure_s6.png")
crop_image("results/supp_S6_S7_page-12.png", (550, 430, 1135, 1015), "figure_s7.png")


# Approximate values digitized from Supplementary Figures S6 and S7.
s6_rows = [
    {"figure": "S6", "series": "348 K", "delta_h_kj_mol": 0.45, "delta_h_peak_kj_mol": 0.136, "source": "digitized_from_supplementary_figure"},
    {"figure": "S6", "series": "348 K", "delta_h_kj_mol": 0.54, "delta_h_peak_kj_mol": 0.165, "source": "digitized_from_supplementary_figure"},
    {"figure": "S6", "series": "348 K", "delta_h_kj_mol": 0.59, "delta_h_peak_kj_mol": 0.175, "source": "digitized_from_supplementary_figure"},
    {"figure": "S6", "series": "363 K", "delta_h_kj_mol": 0.48, "delta_h_peak_kj_mol": 0.100, "source": "digitized_from_supplementary_figure"},
    {"figure": "S6", "series": "363 K", "delta_h_kj_mol": 0.55, "delta_h_peak_kj_mol": 0.123, "source": "digitized_from_supplementary_figure"},
    {"figure": "S6", "series": "363 K", "delta_h_kj_mol": 0.62, "delta_h_peak_kj_mol": 0.133, "source": "digitized_from_supplementary_figure"},
    {"figure": "S6", "series": "363 K", "delta_h_kj_mol": 0.68, "delta_h_peak_kj_mol": 0.143, "source": "digitized_from_supplementary_figure"},
    {"figure": "S6", "series": "373 K", "delta_h_kj_mol": 0.44, "delta_h_peak_kj_mol": 0.057, "source": "digitized_from_supplementary_figure"},
    {"figure": "S6", "series": "373 K", "delta_h_kj_mol": 0.52, "delta_h_peak_kj_mol": 0.087, "source": "digitized_from_supplementary_figure"},
    {"figure": "S6", "series": "373 K", "delta_h_kj_mol": 0.59, "delta_h_peak_kj_mol": 0.100, "source": "digitized_from_supplementary_figure"},
    {"figure": "S6", "series": "373 K", "delta_h_kj_mol": 0.65, "delta_h_peak_kj_mol": 0.108, "source": "digitized_from_supplementary_figure"},
    {"figure": "S6", "series": "373 K", "delta_h_kj_mol": 0.72, "delta_h_peak_kj_mol": 0.118, "source": "digitized_from_supplementary_figure"},
]

s7_data = {
    "348 K": [(0.1, 0), (0.2, 0), (0.5, 0), (1, 0), (2, 0), (5, 0), (10, 17), (20, 9), (50, 9), (100, 15), (200, 31), (500, 86), (1000, 108), (2000, 115), (5000, 115)],
    "363 K": [(0.1, 0), (0.2, 0), (0.5, 0), (1, 1), (2, 4), (5, 14), (10, 16), (20, 38), (50, 95), (100, 122), (200, 132), (500, 135), (1000, 132), (2000, 132), (5000, 131)],
    "373 K": [(0.1, 2), (0.2, 2), (0.5, 3), (1, 5), (2, 8), (5, 25), (10, 82), (20, 127), (50, 144), (100, 144), (200, 146), (500, 140), (1000, 139), (2000, 143)],
    "383 K": [(0.1, 12), (0.2, 13), (0.5, 14), (1, 20), (2, 45), (5, 132), (10, 158), (20, 163), (50, 157), (100, 157), (200, 154), (500, 153), (1000, 157), (2000, 158)],
}

s7_rows = []
for series, pairs in s7_data.items():
    for t_s, s_star in pairs:
        s7_rows.append({
            "figure": "S7",
            "series": series,
            "annealing_time_s": t_s,
            "s_star_j_mol_k": s_star,
            "source": "digitized_from_supplementary_figure",
        })

write_csv("figure_s6_digitized.csv", ["figure", "series", "delta_h_kj_mol", "delta_h_peak_kj_mol", "source"], s6_rows)
write_csv("figure_s7_digitized.csv", ["figure", "series", "annealing_time_s", "s_star_j_mol_k", "source"], s7_rows)

readme = OUT / "README.md"
readme.write_text(
    "\n".join([
        "# Extracted Figure/Data Assets",
        "",
        "- `fig2a.png` to `fig2d.png`: cropped panels from main-paper Figure 2.",
        "- `fig2a_d.png`: combined crop containing Figure 2a-d.",
        "- `figure_s6.png`, `figure_s7.png`: cropped supplementary figures.",
        "- `figure_s6_digitized.csv`: approximate experimental points digitized from Figure S6.",
        "- `figure_s7_digitized.csv`: approximate experimental points digitized from Figure S7.",
        "",
        "The PDFs do not contain raw source tables for S6/S7. CSV values were digitized from the rendered figures and should be treated as approximate.",
    ]),
    encoding="utf-8",
)

print(f"Wrote extracted assets to {OUT}")
