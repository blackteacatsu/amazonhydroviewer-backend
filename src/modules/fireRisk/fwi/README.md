# The Canadian Fire Weather Index [Van Wagner 1987]

## Background

The methods reproduced here in this module follows the instruction provided in the original written equations in Van Wagner 1987 FWI Doc.

This file documents the FWI by its components

### Codespace at a glance

### Fine Fuel Moisture Code (FFMC)

### Duff Moisture Code

# Amazon LDAS FWI Verification

## Introduction

The verification processes will follows the methods outlined in Kansara et al., 2026.

## Spatial Agreement between retrospective & forecast [Roebber, 2009]

Spatial agreement between the forecasted and retrospective drought extents is evaluated using the following detection metrics: probability of detection (POD), false alarm ratio (FAR), critical success index (CSI), and frequency bias (FBIAS).

### Probability of detection (POD)

### False alarm ratio (FAR)

## Deterministic skill metrics

### Pearson correlation, $r$

For each grid cell and month,

```math
r=\frac{\sum_y(F_y-\bar F)(R_y-\bar R)}{\sqrt{\sum_y(F_y-\bar F)^2\sum_y(R_y-\bar R)^2}}
```

where $F_y$ is ensemble-mean monthly forecast code and $R_y$ is the corresponding retrospective value.

### Anomaly Correlation Coefficient (ACC)

Define forecast and retrospective anomalies using their respective climatologies:

```math
F'_y = F_y - \overline{F}, \quad R'_y = R_y - \overline{R}
```

If these climatologies already use the same paired initialization years, then the Pearson coefficient equals to ACC

### Spearman `spearman_r`

Rank agreement ,

### Root mean squared error `rmse`, `bias`

magnitude errors in FWI units.

```math
RMSE = \sqrt{\frac{1}{N}\sum_y(F_y-R_y)^2}, \quad \text{Bias} = \frac{1}{N}\sum_y(F_y-R_y).
```

### `paired-years`

## Probabilistic skill metrics

Ensemble-size debiasing:

```math
RPSS_D=1-\frac{\overline{RPS_{\mathrm{forecast}}}}
{\overline{RPS_{\mathrm{climatology}}}+D},
\qquad

D=\frac{K^2-1}{6KM}.
```

### Ranked probability skill score (RPSS)
