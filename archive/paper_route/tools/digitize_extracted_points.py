from pathlib import Path
from PIL import Image, ImageDraw
import csv
import math
import numpy as np
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "extracted"
OUT = SRC / "data_points"
OUT.mkdir(parents=True, exist_ok=True)


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_rgb_at(image, x, y, radius=5):
    h, w = image.shape[:2]
    x0, x1 = max(0, int(x - radius)), min(w, int(x + radius + 1))
    y0, y1 = max(0, int(y - radius)), min(h, int(y + radius + 1))
    patch = image[y0:y1, x0:x1]
    mx = patch.max(axis=2)
    mn = patch.min(axis=2)
    mask = (mx - mn > 35) & (mx > 110)
    if mask.any():
        rgb = patch[mask].mean(axis=0)
    else:
        rgb = patch.reshape(-1, 3).mean(axis=0)
    return tuple(int(v) for v in rgb)


def rgb_label(rgb):
    r, g, b = rgb
    if r > 180 and g < 110 and b < 110:
        return "red"
    if r > 180 and b > 150 and g < 140:
        return "magenta"
    if r > 190 and g > 130 and b < 100:
        return "orange_yellow"
    if g > 130 and r < 170 and b < 170:
        return "green"
    if b > 150 and r < 130:
        return "blue"
    return f"rgb_{r}_{g}_{b}"


def cluster_centers(centers, min_dist=10):
    picked = []
    for x, y, score in sorted(centers, key=lambda item: item[2], reverse=True):
        if all((x - px) ** 2 + (y - py) ** 2 >= min_dist ** 2 for px, py, _ in picked):
            picked.append((x, y, score))
    return sorted(picked, key=lambda item: (item[0], item[1]))


def detect_colored_marker_centers(image_path, plot_box, excludes=(), threshold=42):
    image = np.array(Image.open(image_path).convert("RGB"))
    yy, xx = np.indices(image.shape[:2])
    left, top, right, bottom = plot_box
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    mx = image.max(axis=2)
    mn = image.min(axis=2)
    mask = (
        (xx >= left) & (xx <= right) & (yy >= top) & (yy <= bottom)
        & (mx - mn > 45) & (mx > 115)
    )
    for x0, y0, x1, y1 in excludes:
        mask[y0:y1, x0:x1] = False

    density = ndimage.uniform_filter(mask.astype(float), size=17) * 17 * 17
    local_max = ndimage.maximum_filter(density, size=21)
    peaks = (density == local_max) & (density > threshold) & mask
    labels, _ = ndimage.label(peaks)

    centers = []
    for label_id, slc in enumerate(ndimage.find_objects(labels), start=1):
        if slc is None:
            continue
        pts = np.where(labels == label_id)
        y = float(pts[0].mean())
        x = float(pts[1].mean())
        centers.append((x, y, float(density[int(round(y)), int(round(x))])))
    return image, cluster_centers(centers, min_dist=12)


def log_x(px, tick_at_1e_minus1, tick_step):
    return 10 ** (-1 + (px - tick_at_1e_minus1) / tick_step)


def linear_y_down(py, y_at_min, min_value, per_value_0_1_px):
    return min_value + (py - y_at_min) / per_value_0_1_px * 0.1


def draw_overlay(image_path, centers, out_path):
    im = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(im)
    for idx, (x, y, _) in enumerate(centers, start=1):
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline="black", width=2)
        draw.text((x + 8, y - 8), str(idx), fill="black")
    im.save(out_path)


fig2_configs = {
    "fig2a": {
        "file": "fig2a.png",
        "plot_box": (153, 31, 735, 611),
        "excludes": [(180, 270, 360, 430), (420, 30, 735, 105)],
        "x_tick_0": 218.0,
        "x_step": 123.5,
        "y_top": 31.0,
        "y_step": 89.0,
        "threshold": 38,
    },
    "fig2b": {
        "file": "fig2b.png",
        "plot_box": (159, 31, 741, 611),
        "excludes": [(180, 270, 365, 435), (430, 30, 741, 105)],
        "x_tick_0": 222.0,
        "x_step": 123.5,
        "y_top": 31.0,
        "y_step": 89.0,
        "threshold": 38,
    },
    "fig2c": {
        "file": "fig2c.png",
        "plot_box": (153, 18, 735, 596),
        "excludes": [(430, 18, 735, 105), (150, 560, 360, 596)],
        "x_tick_0": 218.0,
        "x_step": 123.5,
        "y_top": 18.0,
        "y_step": 89.0,
        "threshold": 38,
    },
}

fig2_rows = []
for panel, cfg in fig2_configs.items():
    image_path = SRC / cfg["file"]
    image, centers = detect_colored_marker_centers(
        image_path,
        cfg["plot_box"],
        cfg["excludes"],
        cfg["threshold"],
    )
    draw_overlay(image_path, centers, OUT / f"{panel}_detected_points_overlay.png")
    for point_id, (px, py, score) in enumerate(centers, start=1):
        rgb = mean_rgb_at(image, px, py)
        fig2_rows.append({
            "figure": "Fig2",
            "panel": panel[-1],
            "point_id": point_id,
            "pixel_x": round(px, 2),
            "pixel_y": round(py, 2),
            "t2_s": round(log_x(px, cfg["x_tick_0"], cfg["x_step"]), 5),
            "delta_h_kj_mol": round(linear_y_down(py, cfg["y_top"], 0.1, cfg["y_step"]), 5),
            "marker_color": rgb_label(rgb),
            "mean_rgb": rgb,
            "detection_score": round(score, 1),
            "source": "semi_automatic_digitization_from_cropped_image",
        })

write_csv(
    OUT / "fig2a_c_detected_points.csv",
    fig2_rows,
    ["figure", "panel", "point_id", "pixel_x", "pixel_y", "t2_s", "delta_h_kj_mol", "marker_color", "mean_rgb", "detection_score", "source"],
)


# Manually digitized experimental markers for Fig.2d / Supplementary Fig.S6 and S7.
# These use the same cropped images and axis calibration, but the sparse points are read directly.
fig2d_s6_rows = [
    ("348 K", 0.45, 0.136), ("348 K", 0.54, 0.165), ("348 K", 0.59, 0.175),
    ("363 K", 0.48, 0.100), ("363 K", 0.55, 0.123), ("363 K", 0.62, 0.133), ("363 K", 0.68, 0.143),
    ("373 K", 0.44, 0.057), ("373 K", 0.52, 0.087), ("373 K", 0.59, 0.100), ("373 K", 0.65, 0.108), ("373 K", 0.72, 0.118),
]

write_csv(
    OUT / "fig2d_s6_experimental_points.csv",
    [
        {
            "figure": "Fig2d/S6",
            "series": series,
            "delta_h_kj_mol": x,
            "delta_h_peak_kj_mol": y,
            "source": "manual_digitization_from_cropped_image",
        }
        for series, x, y in fig2d_s6_rows
    ],
    ["figure", "series", "delta_h_kj_mol", "delta_h_peak_kj_mol", "source"],
)

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
            "source": "manual_digitization_from_cropped_image",
        })

write_csv(
    OUT / "s7_experimental_points.csv",
    s7_rows,
    ["figure", "series", "annealing_time_s", "s_star_j_mol_k", "source"],
)

readme = OUT / "README.md"
readme.write_text(
    "\n".join([
        "# Digitized Experimental Points",
        "",
        "- `fig2a_c_detected_points.csv`: semi-automatic marker detections for Fig.2a-c. Curves and symbols overlap, so inspect the overlay PNGs before using as final data.",
        "- `fig2d_s6_experimental_points.csv`: sparse experimental points from Fig.2d / Supplementary Fig.S6.",
        "- `s7_experimental_points.csv`: experimental points from Supplementary Fig.S7.",
        "- `*_overlay.png`: marker detection overlays for quick visual QA.",
        "",
        "All values are digitized from cropped figure images, not from original source tables.",
    ]),
    encoding="utf-8",
)

print(f"Wrote digitized point data to {OUT}")
