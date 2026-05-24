from pathlib import Path
from PIL import Image, ImageDraw
import csv
import numpy as np
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "extracted"
OUT = SRC / "data_points" / "overlap_enhanced"
OUT.mkdir(parents=True, exist_ok=True)


CONFIGS = {
    "fig2a": {
        "file": "fig2a.png",
        "plot_box": (153, 31, 735, 611),
        "exclude": [(180, 270, 360, 430), (420, 30, 735, 105)],
        "x_tick_1e_minus1": 218.0,
        "x_tick_step_px": 123.5,
        "y_at_0p1": 31.0,
        "y_step_0p1_px": 89.0,
    },
    "fig2b": {
        "file": "fig2b.png",
        "plot_box": (159, 31, 741, 611),
        "exclude": [(180, 270, 365, 435), (430, 30, 741, 105)],
        "x_tick_1e_minus1": 222.0,
        "x_tick_step_px": 123.5,
        "y_at_0p1": 31.0,
        "y_step_0p1_px": 89.0,
    },
    "fig2c": {
        "file": "fig2c.png",
        "plot_box": (153, 18, 735, 596),
        "exclude": [(430, 18, 735, 105), (150, 560, 360, 596)],
        "x_tick_1e_minus1": 218.0,
        "x_tick_step_px": 123.5,
        "y_at_0p1": 18.0,
        "y_step_0p1_px": 89.0,
    },
}


def t_from_x(px, x0, step):
    return 10 ** (-1 + (px - x0) / step)


def h_from_y(py, y0, step):
    return 0.1 + (py - y0) / step * 0.1


def color_name(rgb):
    r, g, b = rgb
    if r > 210 and g < 90 and b < 80:
        return "red/orange"
    if r > 215 and g > 150 and b < 120:
        return "orange/yellow"
    if r > 210 and g > 200 and b < 130:
        return "yellow"
    if g > 170 and r < 210 and b < 180:
        return "green"
    if b > 190 and r < 180:
        return "blue/purple"
    if r > 200 and b > 190 and g < 150:
        return "magenta"
    return f"rgb({r},{g},{b})"


def mean_rgb(im, x, y, radius=5):
    h, w = im.shape[:2]
    x0, x1 = max(0, int(x - radius)), min(w, int(x + radius + 1))
    y0, y1 = max(0, int(y - radius)), min(h, int(y + radius + 1))
    patch = im[y0:y1, x0:x1]
    mx = patch.max(axis=2)
    mn = patch.min(axis=2)
    mask = (mx - mn > 35) & (mx > 110)
    if mask.any():
        rgb = patch[mask].mean(axis=0)
    else:
        rgb = patch.reshape(-1, 3).mean(axis=0)
    return tuple(int(v) for v in rgb)


def cluster_close(peaks, min_dist=8):
    chosen = []
    for x, y, score in sorted(peaks, key=lambda p: p[2], reverse=True):
        if all((x - cx) ** 2 + (y - cy) ** 2 >= min_dist ** 2 for cx, cy, _ in chosen):
            chosen.append((x, y, score))
    return sorted(chosen, key=lambda p: (p[0], p[1]))


def detect_marker_cores(path, cfg):
    im = np.array(Image.open(path).convert("RGB"))
    yy, xx = np.indices(im.shape[:2])
    r, g, b = im[:, :, 0], im[:, :, 1], im[:, :, 2]
    mx = im.max(axis=2)
    mn = im.min(axis=2)
    left, top, right, bottom = cfg["plot_box"]

    colored = (
        (xx >= left) & (xx <= right) & (yy >= top) & (yy <= bottom)
        & (mx - mn > 45) & (mx > 115)
    )
    for x0, y0, x1, y1 in cfg["exclude"]:
        colored[y0:y1, x0:x1] = False

    # Thin TNM curves usually have distance <= 2 px. Filled markers keep a
    # larger interior even when a curve of the same color passes through them.
    dist = ndimage.distance_transform_edt(colored)
    local_max = ndimage.maximum_filter(dist, size=13)
    peak_mask = (dist == local_max) & (dist >= 4.8)
    labels, _ = ndimage.label(peak_mask)

    peaks = []
    for label_id, slc in enumerate(ndimage.find_objects(labels), start=1):
        if slc is None:
            continue
        pts = np.where(labels == label_id)
        y = float(pts[0].mean())
        x = float(pts[1].mean())
        score = float(dist[pts].max())
        peaks.append((x, y, score))
    return im, cluster_close(peaks)


def draw_overlay(image_path, rows, out_path):
    im = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(im)
    for row in rows:
        x = float(row["pixel_x"])
        y = float(row["pixel_y"])
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline="black", width=2)
        draw.text((x + 8, y - 8), str(row["point_id"]), fill="black")
    im.save(out_path)


def write_csv(path, rows):
    fields = [
        "figure", "panel", "point_id", "pixel_x", "pixel_y",
        "t2_s", "delta_h_kj_mol", "marker_color", "mean_rgb",
        "core_radius_px", "overlap_note", "source",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


all_rows = []
for panel, cfg in CONFIGS.items():
    path = SRC / cfg["file"]
    im, peaks = detect_marker_cores(path, cfg)
    rows = []
    for idx, (x, y, score) in enumerate(peaks, start=1):
        rgb = mean_rgb(im, x, y)
        # Points on the far-right converged tail often represent several
        # physically overlapping series; the figure no longer contains enough
        # visual information to split them uniquely.
        note = "visible_marker_core"
        if t_from_x(x, cfg["x_tick_1e_minus1"], cfg["x_tick_step_px"]) >= 80:
            note = "converged_overlap_cluster"
        row = {
            "figure": "Fig2",
            "panel": panel[-1],
            "point_id": idx,
            "pixel_x": round(x, 2),
            "pixel_y": round(y, 2),
            "t2_s": round(t_from_x(x, cfg["x_tick_1e_minus1"], cfg["x_tick_step_px"]), 5),
            "delta_h_kj_mol": round(h_from_y(y, cfg["y_at_0p1"], cfg["y_step_0p1_px"]), 5),
            "marker_color": color_name(rgb),
            "mean_rgb": rgb,
            "core_radius_px": round(score, 2),
            "overlap_note": note,
            "source": "distance_transform_marker_core_digitization",
        }
        rows.append(row)
        all_rows.append(row)
    write_csv(OUT / f"{panel}_overlap_enhanced_points.csv", rows)
    draw_overlay(path, rows, OUT / f"{panel}_overlap_enhanced_overlay.png")

write_csv(OUT / "fig2a_c_overlap_enhanced_points.csv", all_rows)

(OUT / "README.md").write_text(
    "\n".join([
        "# Overlap-Enhanced Digitization",
        "",
        "This pass uses a distance transform on colored pixels. Thin TNM curves are suppressed because they have a small pixel radius, while filled experimental markers keep a larger core even when a same-color curve crosses them.",
        "",
        "Limitations:",
        "- If several markers lie exactly on top of each other in the converged tail, the raster image contains only one visible core. Those rows are marked `converged_overlap_cluster`.",
        "- Values are still digitized from cropped raster images, not original source tables.",
        "",
        "Use the overlay PNGs to audit point IDs against the source figures.",
    ]),
    encoding="utf-8",
)

print(f"Wrote overlap-enhanced data to {OUT}")
