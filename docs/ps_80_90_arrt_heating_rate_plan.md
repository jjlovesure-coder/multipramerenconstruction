# PS 80/90 Multi-Heating-Rate ARRT Plan

## Purpose

This is the minimum heating-rate experiment set for the reduced inverse problem:

```text
fixed T1/T2 = 80/90 or 90/80, known t1 ~= 50 s or 500 s -> infer t2
```

The previous minimal inverse check showed that final heating curves alone recover `t2` only coarsely, while the digitized paper data improve strongly when Figure-3-like `H*/S*` information is added. Therefore this plan does not expand the two-step matrix. It first builds strict single-step `H*/S*` anchors at 80 C and 90 C.

## Experiment Matrix

- Temperatures: `80 C`, `90 C`
- Annealing times: `50 s`, `500 s`, `1000 s`
- Heating rates: `5, 10, 20, 40 C/min`
- Total recommended runs: `24`
- Core runs if time is limited: `16` (`50 s` and `500 s` only)

Each run uses:

```text
200 C -> annealing temperature at 0.1 C/s, hold t
annealing temperature -> 30 C at 0.1 C/s, hold 600 s
30 C -> 200 C at selected heating rate
```

## Why This Is Enough For The Next Check

The original paper used single-step heating-rate data to build the kinetic `H*/S*` coordinate. For the current PS question, the most economical analogue is to measure single-step 80 C and 90 C anchors first, then reuse those `time -> H*/S*` relationships as priors when `T1/T2` are fixed and only `t2` is unknown.

The `1000 s` points are marked as long-time extension. They stabilize the long-time end of the `H*/S*` curve and are useful because the existing 80/90 two-step data include `t2 ~= 1000 s`.

## Outputs

- CSV plan: `data/ps/ps_80_90_arrt_heating_rate_plan.csv`
- MTD method: `data/kovactrain26111103R9-06-80-90-arrt.mtd`
