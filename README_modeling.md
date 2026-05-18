# Physics-Informed Sparse Annealing Predictor

This MVP predicts two-step annealing responses and searches annealing parameters from sparse digitized data.

## Data Sources

- Fig.2a-c: two-step annealing enthalpy change, expanded to 10 `t1` series.
- Fig.2d / S6: memory peak strength `delta_h_peak`.
- S7: activation entropy `S*` for single-step annealing at 348, 363, 373, and 383 K.

The source PDFs do not contain raw tables, so the current dataset is digitized from figures.

## Tp And H*

The paper states that the relaxation peak temperature `Tp` is obtained from the relaxation peak after subtracting two DSC traces, with heating rate `Rh=1000 K/s` in the supplementary methods. If the raw DSC traces are unavailable, `Tp` can still be approximated by digitizing Fig.3a.

The repository now contains an approximate Fig.3 digitization:

- `results/extracted/data_points/fig3b_hstar_tp_digitized.csv`

`H*` is read from Fig.3b. A coarse `Tp` estimate at `Rh=1000 K/s` is included from Fig.3a for traceability. Because a single heating rate is not enough to determine the Fig.3a slope robustly, the model uses Fig.3b `H*` as the practical approximate activation enthalpy.

ARRT-derived `H*` is still retained as a fallback physics-informed feature:

```text
k = (kB T / h) exp(S*/R) exp(-H*/RT)
k ~= 1 / t
```

This gives an approximate `H*` feature from `T`, `t`, and `S*`.

## Run

Build the processed dataset:

```powershell
python -m src.models.dataset
```

Train the forward predictor:

```powershell
python -m src.models.train_forward
```

Run inverse design search for the example target:

```powershell
python -m src.models.inverse_design
```

Run the multi-objective example with activation entropy targets:

```powershell
python tools/run_inverse_design_example.py
```

Outputs:

- `data/processed/annealing_dataset.csv`
- `results/models/forward_kernel_model.json`
- `results/predictions/inverse_design_top10.json`

Current leave-one-out comparison:

```text
Baseline ARRT H* features:
  delta_h MAE       ~= 0.03827 kJ/mol
  delta_h_peak MAE  ~= 0.00801 kJ/mol
  S2* MAE           ~= 5.47 J/mol.K
  delta_S* MAE      ~= 7.37 J/mol.K
  memory label MAE  ~= 0.10851

Enhanced Fig.3 H*/Tp features:
  delta_h MAE       ~= 0.03773 kJ/mol
  delta_h_peak MAE  ~= 0.00790 kJ/mol
  S2* MAE           ~= 4.93 J/mol.K
  delta_S* MAE      ~= 5.94 J/mol.K
  memory label MAE  ~= 0.09003
```

## Current Model

The first implementation uses a small numpy RBF-kernel regressor. Features include:

- annealing temperatures and log times
- `S*` looked up from S7
- ARRT-derived `H*`
- approximate Fig.3-derived `H*` and `Tp`
- simplified TNM dose using the paper parameters `H*=164 kJ/mol`, `A=6e-22 s`, `beta=0.43`

This is intended for rough screening and workflow validation, not high-precision prediction.

The inverse search can include any subset of these targets:

- `delta_h_kj_mol`
- `delta_h_peak_kj_mol`
- `s2_j_mol_k`
- `delta_s_j_mol_k`
- `memory_effect_label`

Adding `S*` targets helps reduce non-uniqueness because it rejects candidates with similar enthalpy response but different relaxation-entropy state.
