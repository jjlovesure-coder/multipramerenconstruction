# Archived Paper Route

This directory preserves the historical Song-paper digitization and sparse
paper-style modeling route. It is kept for provenance and method comparison,
but it is not the active PS DSC workflow.

Archived material includes:

- source PDFs under `paper/`
- digitized and cropped figure outputs under `results/extracted/`
- old paper-route model outputs under `results/models/` and `results/predictions/`
- one-off extraction and digitization scripts under `tools/`
- archived paper-route tests under `tests/`
- the original paper-route README as `README_modeling.md`

Some compatibility modules remain in `src/models/` because the current PS
workflow still imports shared modeling utilities such as `KernelRegressor`, and
one PS diagnostic still references paper-style helper functions. Treat those
modules as compatibility support rather than the main workflow.

To revive this route, move or adapt the archived scripts and data deliberately
instead of adding it back to the active README.
