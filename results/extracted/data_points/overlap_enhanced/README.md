# Overlap-Enhanced Digitization

This pass uses a distance transform on colored pixels. Thin TNM curves are suppressed because they have a small pixel radius, while filled experimental markers keep a larger core even when a same-color curve crosses them.

Limitations:
- If several markers lie exactly on top of each other in the converged tail, the raster image contains only one visible core. Those rows are marked `converged_overlap_cluster`.
- Values are still digitized from cropped raster images, not original source tables.

Use the overlay PNGs to audit point IDs against the source figures.