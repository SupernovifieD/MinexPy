# Statistical Foundations for MinexPy

<!-- Enable MathJax on this page so LaTeX renders when browsing the local MkDocs site -->
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

This note explains the mathematical foundations behind the statistical functions exposed in `minexpy.stats` and the `StatisticalAnalyzer` class. It focuses on univariate sample statistics commonly used in geochemical exploration and environmental datasets.

## Purpose and scope

- Provide clear formulas for each metric implemented in MinexPy.
- Clarify which estimators are used (sample vs. population) and how missing values are handled.
- Offer guidance on when each statistic is informative for geoscience data.

## Data model and notation

- A dataset consists of observations with sample size n >= 2:
  $$
  x_1, x_2, \ldots, x_n
  $$
- Sample mean:
  $$
  \bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i
  $$
- This is the arithmetic average. It is sensitive to extreme values because every observation contributes equally to the sum.
- MinexPy drops `NaN` values before computation (matching NumPy/Pandas `nan` handling and SciPy's `nan_policy='omit'`).
- Many functions allow specifying a degrees-of-freedom offset (`ddof`). The default `ddof=1` yields unbiased sample estimators; `ddof=0` yields population estimators. Use `ddof=1` for field samples, and `ddof=0` only when you truly have the full population.

## Measures of central tendency

- **Arithmetic mean** (function: `mean`):
  $$
  \bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i
  $$
- Best for roughly symmetric distributions without extreme outliers. Use it to summarize typical concentration levels.
- **Median** (function: `median`): sorted values follow the ordering below:
  $$
  x_{(1)} \leq x_{(2)} \leq \dots \leq x_{(n)}
  $$
  $$
  \operatorname{median}(x) = \begin{cases}
  x_{(k)} & n = 2k-1 \\
  \frac{x_{(k)} + x_{(k+1)}}{2} & n = 2k
  \end{cases}
  $$
- The median is robust to outliers and skew. Prefer it for heavy-tailed or log-normal geochemical data.
- **Mode**: most frequent value in discrete samples (function: `mode`, returns value and count).
- Mode highlights the most common measurement. It is most useful for discrete classes or rounded lab results.

## Measures of dispersion

- **Sample variance** (default):
  $$
  s^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2,
  $$
  implemented with `variance(data, ddof=1)`; set `ddof=0` for population variance.
- Variance measures spread in squared units. Higher variance means measurements are widely scattered around the mean.
- **Sample standard deviation** (function: `std`):
  $$
  s = \sqrt{s^2}
  $$
- Standard deviation is the square root of variance and returns to the original units (e.g., ppm). It summarizes typical deviation from the mean.
- **Range**:
  $$
  \operatorname{range} = x_{\max} - x_{\min}
  $$
- The range shows total spread but is very sensitive to outliers.
- **Coefficient of variation (CV)**: a unitless measure of relative dispersion,
  $$
  \operatorname{CV} = \frac{s}{\bar{x}},
  $$
  used when the mean is non-zero (function: `coefficient_of_variation`).
- CV compares variability between elements with different scales. Values under about 0.15 suggest low relative variability; values above about 0.5 suggest high relative variability.

## Percentiles and interquartile range

- **Percentiles (p-th)**: value denoted as `q_p` such that p percent of observations are less than or equal to it. MinexPy uses NumPy's percentile estimator (default linear interpolation) via `percentile(data, p)`.
  $$
  q_p = \operatorname{percentile}(x, p)
  $$
- Percentiles summarize distribution shape without assuming normality. They are useful for reporting regulatory thresholds or resource cutoff grades.
- **Quartiles**:
  $$
  Q_1 = q_{25}, \quad Q_2 = q_{50} \; (\text{median}), \quad Q_3 = q_{75}
  $$
- **Interquartile range (IQR)** (function: `iqr`):
  $$
  \operatorname{IQR} = Q_3 - Q_1
  $$
  IQR is robust to outliers and is often preferred for skewed geochemical distributions.
- IQR measures the spread of the middle 50% of values. It is less affected by a few extreme assays than the standard deviation.

## Shape metrics

- **Skewness (Fisher, bias-corrected)**:
  $$
  G_1 = \frac{n}{(n-1)(n-2)} \sum_{i=1}^{n} \left( \frac{x_i - \bar{x}}{s} \right)^3,
  $$
  implemented via `scipy.stats.skew` (function: `skewness`). Positive skew indicates a right tail; negative skew indicates a left tail.
- Skewness describes asymmetry. Right-skewed data often indicate a few high-grade samples in exploration datasets.
- **Excess kurtosis (Fisher)**:
  $$
  G_2 = \frac{n(n+1)}{(n-1)(n-2)(n-3)} \sum_{i=1}^{n} \left( \frac{x_i - \bar{x}}{s} \right)^4 - \frac{3(n-1)^2}{(n-2)(n-3)},
  $$
  implemented via `scipy.stats.kurtosis` with `fisher=True` (function: `kurtosis`). G2 = 0 for a normal distribution; G2 > 0 indicates heavier tails.
- Kurtosis captures tail weight. Heavy tails (positive kurtosis) suggest more extreme values than expected under a normal model.

## Standardization and outlier detection

- **Z-scores**: standardize observations to units of standard deviation,
  $$
  z_i = \frac{x_i - \bar{x}}{s}
  $$
  computed with `z_score`. Common outlier rules use the thresholds below (assuming approximate normality):
  $$
  |z_i| > 2 \quad \text{(liberal)} \qquad |z_i| > 3 \quad \text{(conservative)}
  $$
- Z-scores allow you to compare how unusual a value is relative to the sample. Large absolute z-scores flag potential anomalies for further QA/QC or domain review.

## Correlation metrics

MinexPy's `minexpy.correlation` module provides several complementary dependence metrics because geoscience variables are often skewed, noisy, and not strictly linear.

- **Pearson correlation** (`pearson_correlation`): linear association
  $$
  r_{xy} =
  \frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}
       {\sqrt{\sum_{i=1}^{n}(x_i-\bar{x})^2}\sqrt{\sum_{i=1}^{n}(y_i-\bar{y})^2}}
  $$
- Pearson is efficient for approximately normal, homoscedastic relationships but sensitive to outliers.

- **Spearman correlation** (`spearman_correlation`): Pearson correlation on ranks.
- When there are no ties, an equivalent form is:
  $$
  \rho_s = 1 - \frac{6\sum_{i=1}^{n} d_i^2}{n(n^2-1)},
  $$
  where $d_i$ is the rank difference for pair $i$.
- Spearman is preferred for monotonic but nonlinear trends.

- **Kendall correlation** (`kendall_correlation`): concordance-based rank metric
  $$
  \tau = \frac{N_c - N_d}{\binom{n}{2}}
  $$
  where $N_c$ and $N_d$ are numbers of concordant and discordant pairs.
- Kendall is robust for ordinal structure and often stable in small samples.

- **Distance correlation** (`distance_correlation`): nonlinear dependence via centered pairwise distances.
- Let $A$ and $B$ be double-centered distance matrices for $x$ and $y$:
  $$
  \mathcal{V}^2(x,y)=\frac{1}{n^2}\sum_{i,j}A_{ij}B_{ij}, \quad
  \mathcal{R}(x,y)=\frac{\mathcal{V}(x,y)}{\sqrt{\mathcal{V}(x,x)\mathcal{V}(y,y)}}
  $$
- Distance correlation can detect nonlinear relationships that Pearson can miss.

- **Biweight midcorrelation** (`biweight_midcorrelation`): robust correlation with outlier downweighting.
- Using median $M_x$ and MAD scale, define
  $$
  u_i=\frac{x_i-M_x}{c\cdot MAD_x}, \qquad
  w_i=(1-u_i^2)^2 \text{ for } |u_i|<1, \text{ else } 0.
  $$
- The same is computed for $y$, then weighted centered values are correlated.
- This is useful for assay datasets with occasional extreme values.

- **Partial correlation** (`partial_correlation`): correlation after removing control-variable effects.
- If $r_x$ and $r_y$ are residuals from regressing $x$ and $y$ on controls $Z$, then:
  $$
  r_{xy\cdot Z} = \operatorname{corr}(r_x, r_y)
  $$
- Useful for asking whether two elements remain associated after removing depth/lithology trends.

## Comprehensive summaries

`describe` aggregates the above into a single dictionary containing: count, mean, median, standard deviation, variance, min, max, range, skewness, kurtosis, coefficient of variation, Q1, Q3, IQR, and any user-specified percentiles. `StatisticalAnalyzer.summary` applies `describe` column-wise for DataFrames and returns a tidy DataFrame.

## Visualization diagnostics

The `minexpy.statviz` module supports the core graphics used to validate distribution and dependence assumptions:

- `plot_histogram`: frequency/density inspection in linear or log-x scale.
- `plot_box_violin`: robust spread and shape comparison across variables.
- `plot_ecdf`: nonparametric cumulative-distribution comparison.
- `plot_qq`: sample quantiles against theoretical quantiles.
- `plot_pp`: empirical probabilities against theoretical probabilities.
- `plot_scatter`: bivariate pattern checks with optional trend line.

These figures should be interpreted together with summary statistics and domain context (sampling support, assay precision, compositional closure, and spatial dependence).

## Implementation notes in MinexPy

- **Libraries**: NumPy (core array math), Pandas (DataFrame operations), SciPy (skewness/kurtosis estimators).
- **Missing data**: `NaN` values are removed before calculations (`nan_policy='omit'`).
- **Estimator choices**: defaults favor sample estimators (`ddof=1`) to reduce bias when working with finite field samples.
- **Return types**: functional API returns Python floats/arrays; `StatisticalAnalyzer` returns Pandas Series/DataFrames for convenient downstream plotting or export.

## Geoscience-specific considerations

- **Log-normal tendencies**: Element concentrations often follow log-normal or mixed distributions driven by multiplicative geological processes. Check skewness and, if right-skewed, assess the data on a log scale before applying z-score thresholds or comparing means.
- **Relative variability**: CV enables comparing dispersion across elements with very different absolute magnitudes (e.g., ppm vs. ppb). High CV in trace elements may reflect true geological variability or analytical noise—interpret alongside detection limits.
- **Robustness and outliers**: IQR-based fences and median statistics are usually more stable than mean/z-score for datasets with a few extreme assays (common in mineralized zones). Use robust stats to avoid chasing single-sample spikes.
- **Compositional effects**: Major oxides and some geochemical suites are closed to 100%. Classical statistics can be distorted by the constant-sum constraint. Consider log-ratio methods (clr/alr) before comparing elements on a closed basis.
- **Censored and detection-limit data**: Replace-with-half-dl is crude. Track how many values are below detection, and prefer methods that account for censoring when proportions are large.
- **Spatial structure**: Nearby samples are not independent. Summary stats ignore spatial correlation; for spatial decision-making, follow up with variography or kriging to respect spatial dependence.
- **Sampling design and bias**: Clustered sampling can overweight specific zones. When comparing groups (e.g., lithologies), ensure similar sample support or use weighted summaries if sample spacing differs.

## References and further reading

1. NIST/SEMATECH. (2013). *e-Handbook of Statistical Methods*. [https://www.itl.nist.gov/div898/handbook/](https://www.itl.nist.gov/div898/handbook/)
2. Virtanen, P., et al. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. *Nature Methods*, 17, 261–272. [https://doi.org/10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2)
3. Reimann, C., & Filzmoser, P. (2000). Normal and lognormal data distribution in geochemistry: death of a myth. *Chemosphere*, 52(7), 787–793. [https://doi.org/10.1016/S0045-6535(00)00321-5](https://doi.org/10.1016/S0045-6535(00)00321-5)
4. Schuenemeyer, J. H., & Drew, L. J. (2011). *Statistics for Earth and Environmental Scientists*. Wiley. [https://doi.org/10.1002/9780470650707](https://doi.org/10.1002/9780470650707)
