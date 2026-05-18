# 10-Series Overlap-Filled Fig.2a-c Table

`fig2a_c_10series_overlap_filled_points.csv` expands each visible t2 time point to the 10 t1 series listed in the paper caption: 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, and 100 s.

When fewer than 10 markers are separable at a time point, the missing ranks share the nearest visible marker value. These rows are marked `shared_overlap_visible_markers`.

The `fit_curve_color` column records the intended fitted-curve series color/order. The `source_marker_color` column records the actual visible raster marker/cluster color used for that value.