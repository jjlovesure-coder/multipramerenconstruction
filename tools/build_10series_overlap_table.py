from pathlib import Path
import csv
import math
from collections import defaultdict


ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "results" / "extracted" / "data_points" / "overlap_enhanced" / "fig2a_c_overlap_enhanced_points.csv"
OUT = ROOT / "results" / "extracted" / "data_points" / "overlap_enhanced"

T1_SERIES = [
    (0.1, "blue"),
    (0.2, "orange"),
    (0.5, "light_green"),
    (1.0, "lavender"),
    (2.0, "peach_orange"),
    (5.0, "yellow"),
    (10.0, "yellow_green"),
    (20.0, "magenta"),
    (50.0, "blue_star"),
    (100.0, "red"),
]

# Standard experimental t2 values visible in Fig.2a-c. The far-right tail is
# strongly converged; values beyond 1000 s are not cleanly separable in the
# raster crop, so the table uses the last clearly represented decade point.
T2_VALUES = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]


def nearest_t2(t):
    return min(T2_VALUES, key=lambda x: abs(math.log10(t / x)))


def read_rows():
    with IN.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def assign_ranked_points(points):
    """Return 10 point assignments, top-to-bottom.

    If fewer than 10 markers are visually separable, multiple t1 ranks share
    the nearest visible marker. This encodes indistinguishable overlaps as the
    same experimental value instead of inventing extra hidden coordinates.
    """
    points = sorted(points, key=lambda r: float(r["delta_h_kj_mol"]))
    n = len(points)
    if n == 0:
        return []
    assignments = []
    for rank in range(10):
        if n >= 10:
            idx = rank
        else:
            idx = round(rank * (n - 1) / 9)
        assignments.append(points[idx])
    return assignments


def main():
    raw = read_rows()
    grouped = defaultdict(list)
    for row in raw:
        t = float(row["t2_s"])
        t2 = nearest_t2(t)
        if abs(math.log10(t / t2)) <= 0.22:
            grouped[(row["panel"], t2)].append(row)

    completed = []
    for panel in ["a", "b", "c"]:
        for t2 in T2_VALUES:
            points = grouped[(panel, t2)]
            assignments = assign_ranked_points(points)
            if not assignments:
                continue
            visible_count = len(points)
            method = "direct_10_visible_markers" if visible_count >= 10 else "shared_overlap_visible_markers"
            for rank, ((t1, curve_color), point) in enumerate(zip(T1_SERIES, assignments), start=1):
                completed.append({
                    "figure": "Fig2",
                    "panel": panel,
                    "t2_s": t2,
                    "t1_s": t1,
                    "fit_curve_rank_top_to_bottom": rank,
                    "fit_curve_color": curve_color,
                    "delta_h_kj_mol": point["delta_h_kj_mol"],
                    "source_pixel_x": point["pixel_x"],
                    "source_pixel_y": point["pixel_y"],
                    "source_point_id": point["point_id"],
                    "source_marker_color": point["marker_color"],
                    "visible_markers_at_t2": visible_count,
                    "assignment_method": method,
                    "note": "overlapped ranks share a visible marker at this t2" if visible_count < 10 else "all 10 ranks visually separable",
                })

    fields = [
        "figure", "panel", "t2_s", "t1_s", "fit_curve_rank_top_to_bottom",
        "fit_curve_color", "delta_h_kj_mol", "source_pixel_x", "source_pixel_y",
        "source_point_id", "source_marker_color", "visible_markers_at_t2",
        "assignment_method", "note",
    ]
    out_csv = OUT / "fig2a_c_10series_overlap_filled_points.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(completed)

    summary = OUT / "10series_README.md"
    summary.write_text(
        "\n".join([
            "# 10-Series Overlap-Filled Fig.2a-c Table",
            "",
            "`fig2a_c_10series_overlap_filled_points.csv` expands each visible t2 time point to the 10 t1 series listed in the paper caption: 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, and 100 s.",
            "",
            "When fewer than 10 markers are separable at a time point, the missing ranks share the nearest visible marker value. These rows are marked `shared_overlap_visible_markers`.",
            "",
            "The `fit_curve_color` column records the intended fitted-curve series color/order. The `source_marker_color` column records the actual visible raster marker/cluster color used for that value.",
        ]),
        encoding="utf-8",
    )
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
