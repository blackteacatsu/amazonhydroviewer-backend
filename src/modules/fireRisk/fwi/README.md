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

## To start the runner

Run this snippet w/ the write working environment activated:

```bash
conda activate analytics
python --version # make sure it is 3.12+

results_root=/mnt/vast/ksu/lis_products/fwi_amazon/fcst_skill_deterministic # change metrics type
run_dir="$results_root/original_dc_trial_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$results_root"

# Test the first monthly period across all 24 initialization years.
nohup python -u scripts/python/get_fwi_fcst_skill.py \
    --fcst-dir /mnt/vast/ksu/lis_products/fwi_amazon/forecast/daily/original_dc \
    --retro-dir /mnt/vast/ksu/lis_products/fwi_amazon/retrospective/daily/original_dc \
    --start-yr 2001 \
    --end-yr 2024 \
    --init-mmdd 07-01 \
    --months 6 \
    --min-years 10 \
    --stage all \
    --output-dir "$run_dir" \
    > "$run_dir.log" 2>&1 < /dev/null &

echo "PID: $!"
echo "Results: $run_dir"
echo "Log: $run_dir.log"

tail -f "$run_dir.log"
```

When `--months 1`, for july 1 initialization, this represents the monthly means of July 2-31, correlated across initialization years.

Try, `--months 3` and `--months 6` for longer lead-time into the future.

Snippet for `soilMoist_dc` method:

```bash
conda activate analytics
python --version # make sure it is 3.12+

results_root=/mnt/vast/ksu/lis_products/fwi_amazon/fcst_skill_deterministic # change metrics type
run_dir="$results_root/soilMoist_dc_trial_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$results_root"

# Test the first monthly period across all 24 initialization years.
nohup python -u scripts/python/get_fwi_fcst_skill.py \
    --fcst-dir /mnt/vast/ksu/lis_products/fwi_amazon/forecast/daily/soilMoist_dc \
    --retro-dir /mnt/vast/ksu/lis_products/fwi_amazon/retrospective/daily/soilMoist_dc \
    --start-yr 2001 \
    --end-yr 2024 \
    --init-mmdd 07-01 \
    --months 6 \
    --min-years 10 \
    --stage all \
    --output-dir "$run_dir" \
    > "$run_dir.log" 2>&1 < /dev/null &

echo "PID: $!"
echo "Results: $run_dir"
echo "Log: $run_dir.log"

tail -f "$run_dir.log"
```

## Deterministic skill metrics

### Pearson correlation, $r$

For each grid cell and month,

```math
r=\frac{\sum_y(F_y-\bar F)(R_y-\bar R)}{\sqrt{\sum_y(F_y-\bar F)^2\sum_y(R_y-\bar R)^2}}
```

where $F_y$ is ensemble-mean monthly forecast code and $R_y$ is the corresponding retrospective value.

### Spearman's Rank Correlation Coefficient [Jolliffe and Stephenson 2012] / Anomaly Correlation Coefficient (ACC)

Define forecast and retrospective anomalies using their respective climatologies:

```math
F'_y = F_y - \overline{F}, \quad R'_y = R_y - \overline{R}
```

If these climatologies already use the same paired initialization years, then the Pearson coefficient equals to ACC

Rank agreement ,

```math
\text{ACC}(\rho) = 1 - \frac{6\Sigma d_i^2}{n(n^2-1)}
```

- $d_i$ : the difference between the ranks of the forecast & observed anomalies
- $n$ : total number of observation

This value should range from -1 to 1, with 0 indicating no forecast skill and 1 denoting perfect skill. The statistical significance of Spearman's correlation coefficient was tested using a two-tailed $p$ value derived from the Student's $t$ test, with a significance level of 0.05 to reject the null hypothesis.

### Root mean squared error `rmse`, `bias`

magnitude errors in FWI units.

```math
RMSE = \sqrt{\frac{1}{N}\sum_y(F_y-R_y)^2}, \quad \text{Bias} = \frac{1}{N}\sum_y(F_y-R_y).
```

### Uncertainty

```math
z=\operatorname{atanh}(r),\qquad
CI_{95\%}=\tanh\left(z\pm\frac{1.96}{\sqrt{N-3}}\right).
```

## Probabilistic skill metrics

<!-- Ensemble-size debiasing:

### Tercile probability -->

### Ranked probability skill score (RPSS)

The ranked probability score (RPS) is calculated as:

```math
RPS_y=\frac{1}{n}\sum_{t=1}^{n}\sum_{k=1}^{K}
\left(\hat{P}_{t,k}-\hat{Y}_{t,k}\right)^2.
```

Where $K$ is the number of categories (in our case $K=3$). $n$ is the total number of observations. $\hat{P}_{t,k}$ is the forecast probability for the $k$th category at time $t$, and $\hat{Y}_{t,k}$ is the historical forecast probability for the same category.

```math
Y_{t,k}=
\begin{cases}
1,& \text{if the observation at time }t\text{ belongs to category }k,\\
0,& \text{otherwise}.
\end{cases}
```

In this method, if the $t$th observation is in category $k$ then $Y_{t,k} = 1$. Otherwise, when the observation is not in $k$, then $Y_{t,k} = 0$.

```math
RPSS=1-\frac{{RPS_{\mathrm{forecast}}}}
{{RPS_{\mathrm{climatology}}}},
\qquad
```

$RPSS$ should range from $-\infty$ to 1:

- and 1 indicates perfect skill values
- $\leq0$ indicate no skill compared to climatology.

### Debiased ranked probability skill score ($RPSS_D$)

As noted in Kansara et al. 2025, conventional $RPSS$ is highly sensitive to ensemble size, therefore, we incorporate a debiased $RPSS_D$

```math
RPSS_D=1-\frac{ RPS_{\text{forecast} } }
{ RPS_{ \text{climatology} } + D }
\qquad

D=\frac{K^2-1}{6KM}
```

where $K$ is the number of categories and $M$ is the ensemble size.

In our case,

<!-- In our case, $K=3$, and latest data contains seven ensemble members, so -->

```math
D(K= 3, M=1)= \frac{9-1}{6 \times 3\times 1} = \frac{4}{9}, \quad D(K= 3, M=7) = \frac{9-1}{6 \times 3\times 7} = \frac{4}{63}
```

Since ensemble member presented in the dataset varies across years, we add each year's correction factor togethe and find the mean correction factor for our test windows.

### Interpreting $RPSS_D$

For $RPSS_D$, higher is better. Since small forecast error makes fraction smaller, and therefore the $RPSS_D$ is closer to one:

- if $RPSS_D$ is less than 0, then the forecast is worse than reference
- if $RPSS_D$ is greater than 0, then the forecast is better than reference
- if $RPSS_D = 0$, then the forecast is the same as the reference

### Note on $RPS_\text{climatology}$

The main assumption is that **below-normal, near-normal, and above-normal are defined as three equally probable parts of a reference climate distribution**.

That requires distinguishing the definition from how well your data support it:

- **Tercile thresholds:** the boundaries are the reference distribution’s \(1/3\) and \(2/3\) quantiles.
- **Matching climate context:** thresholds apply to the same variable, grid cell, and target month. For forecast thresholds, also keep initialization and lead time consistent.
- **Representative reference period:** using those probabilities for verification assumes the reference climate reasonably represents the verification period. Trends or regime changes can weaken this assumption.
- **No extra predictive information:** climatology assigns the same probabilities regardless of that year’s initialization conditions, ENSO state, or forecast.
- **Ties need attention:** exact equal probabilities follow naturally for a continuous distribution. Repeated values—especially zero FWI—can prevent fixed thresholds from putting exactly one-third of observations into each category.

**Your forecast ensemble does not need equal numbers of members in each category.** The `[1/3, 1/3, 1/3]` vector describes the climatological reference, not the active forecast.

There is also a distinction relevant to your \(4/9\):

> [!CAUTION]
> You can prescribe an equal-probability climatological forecast even when the verification observations are not equally distributed. But its **expected RPS equals \(4/9\)** only when the verifying outcomes are also equally likely across the three categories.

For actual observed frequencies \(q*{\mathrm{below}},q*{\mathrm{near}},q\_{\mathrm{above}}\), the mean reference score is instead

\[
\overline{RPS*{\mathrm{clim}}}
=\frac59q*{\mathrm{below}}
+\frac29q*{\mathrm{near}}
+\frac59q*{\mathrm{above}}.
\]

Setting each \(q=1/3\) gives \(4/9\). Your selected verification years need not satisfy that equality exactly, even when the thresholds were estimated correctly.
