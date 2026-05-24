# Archive Index

The active documentation in `docs/` is intentionally small. Older stage reports
and paper-route materials have been moved out of the main workflow.

## Paper Route

`archive/paper_route/` contains the historical Song-paper digitization route:
PDFs, extracted figure data, one-off digitization scripts, old model outputs,
and the original `README_modeling.md`.

## Stage Reports

`archive/reports/` contains older planning, comparison, and diagnostic reports
that are useful for provenance but are no longer active entrypoints. The current
reports retained in `docs/` are:

- `physics_forward_inverse_diagnostic.md`
- `ps_70_105_tnm_arrt_audit.md`
- `ps_arrt_kissinger_calculation.md`
- `ps_physics_informed_model_update.md`
- `ps_required_missing_data_for_prediction.md`
- `ps_tnm_curve_fit_forward_report.md`
- `ps_tnm_enthalpy_time_fit_report.md`

When adding a new report, keep only durable summaries in `docs/` and move
one-off experiment logs to `archive/reports/`.

One-off report-generation scripts that depended on removed historical
`before_*` snapshots live in `archive/reports/scripts/`.
