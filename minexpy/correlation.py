"""Correlation analysis tools for geoscience datasets.

This module provides pairwise correlation functions for continuous and
rank-based relationships that are commonly used in geoscience workflows.
It includes classical parametric correlation (Pearson), rank correlations
(Spearman and Kendall), and robust/nonlinear alternatives (biweight
midcorrelation and distance correlation).

The API is designed for practical exploration use-cases where geochemical
or geophysical variables may contain missing values, outliers, non-normal
shape, or monotonic but non-linear relationships.
"""

from typing import Dict, Mapping, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

ArrayLike1D = Union[np.ndarray, pd.Series, Sequence[float]]


__all__ = [
    "pearson_correlation",
    "spearman_correlation",
    "kendall_correlation",
    "distance_correlation",
    "biweight_midcorrelation",
    "partial_correlation",
    "correlation_matrix",
]


def _to_1d_float_array(data: ArrayLike1D, name: str) -> np.ndarray:
    """Convert array-like input into a validated 1D float NumPy array.

    Args:
        data: Input sequence of numeric values.
        name: Parameter name used for error messages.

    Returns:
        A one-dimensional ``numpy.ndarray`` with ``dtype=float``.

    Raises:
        ValueError: If input is empty or not one-dimensional.
    """
    if isinstance(data, pd.Series):
        values = data.to_numpy(dtype=float)
    else:
        values = np.asarray(data, dtype=float)

    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if values.size == 0:
        raise ValueError(f"{name} must not be empty")
    return values


def _prepare_pair(
    x: ArrayLike1D,
    y: ArrayLike1D,
    min_n: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare two aligned numeric vectors and remove invalid pairs.

    Args:
        x: First input vector.
        y: Second input vector.
        min_n: Minimum required number of valid pairs.

    Returns:
        A tuple of cleaned vectors ``(x_clean, y_clean)`` with pairwise
        finite-value filtering applied.

    Raises:
        ValueError: If lengths differ or if too few valid pairs remain.
    """
    x_values = _to_1d_float_array(x, "x")
    y_values = _to_1d_float_array(y, "y")

    if x_values.size != y_values.size:
        raise ValueError("x and y must have the same length")

    mask = np.isfinite(x_values) & np.isfinite(y_values)
    x_clean = x_values[mask]
    y_clean = y_values[mask]

    if x_clean.size < min_n:
        raise ValueError(
            f"At least {min_n} valid paired observations are required; "
            f"received {x_clean.size}"
        )

    return x_clean, y_clean


def _extract_stat_pvalue(result: object) -> Tuple[float, float]:
    """Extract statistic and p-value from SciPy correlation results.

    Args:
        result: Object returned by a SciPy correlation function.

    Returns:
        A tuple of ``(statistic, p_value)``.
    """
    if hasattr(result, "statistic"):
        statistic = float(getattr(result, "statistic"))
    elif hasattr(result, "correlation"):
        statistic = float(getattr(result, "correlation"))
    else:
        statistic = float(result[0])

    if hasattr(result, "pvalue"):
        p_value = float(getattr(result, "pvalue"))
    else:
        p_value = float(result[1])

    return statistic, p_value


def pearson_correlation(
    x: ArrayLike1D,
    y: ArrayLike1D,
    alternative: str = "two-sided",
) -> Dict[str, Union[float, int]]:
    """Compute Pearson product-moment correlation.

    Pearson correlation quantifies linear association between two variables.
    It is sensitive to outliers and assumes approximately linear behavior.

    Args:
        x: First numeric variable.
        y: Second numeric variable.
        alternative: Alternative hypothesis passed to ``scipy.stats.pearsonr``.
            Supported values are ``"two-sided"``, ``"greater"``, and
            ``"less"``.

    Returns:
        A dictionary with:
        - ``correlation``: Pearson's r in ``[-1, 1]``.
        - ``p_value``: p-value for testing zero correlation.
        - ``n``: Number of valid paired observations used.

    Raises:
        ValueError: If inputs are invalid or contain too few valid pairs.

    Examples:
        >>> from minexpy.correlation import pearson_correlation
        >>> x = [10, 12, 15, 20, 21]
        >>> y = [3.1, 3.8, 4.2, 5.7, 6.0]
        >>> pearson_correlation(x, y)
        {'correlation': 0.985..., 'p_value': 0.002..., 'n': 5}
    """
    x_clean, y_clean = _prepare_pair(x, y, min_n=2)
    result = scipy_stats.pearsonr(x_clean, y_clean, alternative=alternative)
    statistic, p_value = _extract_stat_pvalue(result)

    return {
        "correlation": statistic,
        "p_value": p_value,
        "n": int(x_clean.size),
    }


def spearman_correlation(
    x: ArrayLike1D,
    y: ArrayLike1D,
    alternative: str = "two-sided",
) -> Dict[str, Union[float, int]]:
    """Compute Spearman rank correlation.

    Spearman correlation evaluates monotonic association using ranked values.
    It is often preferred for skewed geochemical variables and non-normal
    distributions where monotonic trends are expected.

    Args:
        x: First numeric variable.
        y: Second numeric variable.
        alternative: Alternative hypothesis passed to
            ``scipy.stats.spearmanr``. Supported values are ``"two-sided"``,
            ``"greater"``, and ``"less"``.

    Returns:
        A dictionary with:
        - ``correlation``: Spearman's rho in ``[-1, 1]``.
        - ``p_value``: p-value for testing zero association.
        - ``n``: Number of valid paired observations used.

    Raises:
        ValueError: If inputs are invalid or contain too few valid pairs.

    Examples:
        >>> from minexpy.correlation import spearman_correlation
        >>> x = [1, 2, 3, 4, 5]
        >>> y = [1, 4, 9, 16, 25]
        >>> spearman_correlation(x, y)
        {'correlation': 1.0, 'p_value': 0.0..., 'n': 5}
    """
    x_clean, y_clean = _prepare_pair(x, y, min_n=2)
    result = scipy_stats.spearmanr(x_clean, y_clean, alternative=alternative)
    statistic, p_value = _extract_stat_pvalue(result)

    return {
        "correlation": statistic,
        "p_value": p_value,
        "n": int(x_clean.size),
    }


def kendall_correlation(
    x: ArrayLike1D,
    y: ArrayLike1D,
    method: str = "auto",
    alternative: str = "two-sided",
) -> Dict[str, Union[float, int]]:
    """Compute Kendall's tau correlation.

    Kendall's tau is a rank-based coefficient that compares concordant and
    discordant pairs. It is robust for ordinal data and small samples,
    and is frequently used in environmental and geoscience trend analysis.

    Args:
        x: First numeric variable.
        y: Second numeric variable.
        method: Method for p-value computation. Forwarded to
            ``scipy.stats.kendalltau``.
        alternative: Alternative hypothesis. Supported values are
            ``"two-sided"``, ``"greater"``, and ``"less"``.

    Returns:
        A dictionary with:
        - ``correlation``: Kendall's tau in ``[-1, 1]``.
        - ``p_value``: p-value for testing zero association.
        - ``n``: Number of valid paired observations used.

    Raises:
        ValueError: If inputs are invalid or contain too few valid pairs.

    Examples:
        >>> from minexpy.correlation import kendall_correlation
        >>> x = [2, 4, 6, 8, 10]
        >>> y = [5, 7, 8, 11, 13]
        >>> kendall_correlation(x, y)
        {'correlation': 0.999..., 'p_value': 0.016..., 'n': 5}
    """
    x_clean, y_clean = _prepare_pair(x, y, min_n=2)
    result = scipy_stats.kendalltau(
        x_clean,
        y_clean,
        method=method,
        alternative=alternative,
    )
    statistic, p_value = _extract_stat_pvalue(result)

    return {
        "correlation": statistic,
        "p_value": p_value,
        "n": int(x_clean.size),
    }


def partial_correlation(
    x: ArrayLike1D,
    y: ArrayLike1D,
    controls: Union[np.ndarray, pd.DataFrame, pd.Series, Sequence[float], Sequence[Sequence[float]]],
    alternative: str = "two-sided",
) -> Dict[str, Union[float, int]]:
    """Compute linear partial correlation between ``x`` and ``y``.

    Partial correlation quantifies the linear relationship between ``x`` and
    ``y`` after regressing out one or more control variables. In geoscience,
    this is useful when assessing element-element relations while controlling
    for depth, lithology proxy variables, or other covariates.

    Args:
        x: First numeric variable.
        y: Second numeric variable.
        controls: One or multiple control variables. Accepted shapes are
            ``(n,)`` or ``(n, k)``.
        alternative: Alternative hypothesis for the test on residual
            correlation. Supported values are ``"two-sided"``, ``"greater"``,
            and ``"less"``.

    Returns:
        A dictionary with:
        - ``correlation``: Partial correlation coefficient.
        - ``p_value``: p-value under a t-distribution approximation.
        - ``n``: Number of valid paired observations.
        - ``df``: Residual degrees of freedom, ``n - k - 2``.

    Raises:
        ValueError: If inputs are invalid or insufficient for estimation.

    Examples:
        >>> from minexpy.correlation import partial_correlation
        >>> x = [10, 12, 13, 16, 20]
        >>> y = [2, 3, 3.2, 4.1, 5.2]
        >>> z = [100, 110, 105, 120, 130]
        >>> partial_correlation(x, y, z)
        {'correlation': 0.97..., 'p_value': 0.12..., 'n': 5, 'df': 2}
    """
    x_values = _to_1d_float_array(x, "x")
    y_values = _to_1d_float_array(y, "y")

    if isinstance(controls, pd.DataFrame):
        control_array = controls.to_numpy(dtype=float)
    elif isinstance(controls, pd.Series):
        control_array = controls.to_numpy(dtype=float)
    else:
        control_array = np.asarray(controls, dtype=float)

    if control_array.ndim == 1:
        control_array = control_array.reshape(-1, 1)
    elif control_array.ndim != 2:
        raise ValueError("controls must be one-dimensional or two-dimensional")

    if x_values.size != y_values.size or x_values.size != control_array.shape[0]:
        raise ValueError("x, y, and controls must have the same number of rows")

    valid_mask = np.isfinite(x_values) & np.isfinite(y_values)
    valid_mask = valid_mask & np.all(np.isfinite(control_array), axis=1)

    x_clean = x_values[valid_mask]
    y_clean = y_values[valid_mask]
    z_clean = control_array[valid_mask]

    n_obs = int(x_clean.size)
    n_controls = int(z_clean.shape[1])
    df = n_obs - n_controls - 2

    if n_obs < 3:
        raise ValueError("At least 3 valid observations are required")
    if df <= 0:
        raise ValueError(
            "Not enough observations for partial correlation with the given controls; "
            "need n > number_of_controls + 2"
        )

    design = np.column_stack([np.ones(n_obs), z_clean])

    beta_x, _, _, _ = np.linalg.lstsq(design, x_clean, rcond=None)
    beta_y, _, _, _ = np.linalg.lstsq(design, y_clean, rcond=None)

    residual_x = x_clean - design @ beta_x
    residual_y = y_clean - design @ beta_y

    r, _ = _extract_stat_pvalue(scipy_stats.pearsonr(residual_x, residual_y))
    r = float(np.clip(r, -1.0, 1.0))

    if abs(r) >= 1.0:
        p_value = 0.0
    else:
        t_stat = r * np.sqrt(df / max(1.0 - r * r, np.finfo(float).eps))
        if alternative == "two-sided":
            p_value = float(2.0 * scipy_stats.t.sf(abs(t_stat), df=df))
        elif alternative == "greater":
            p_value = float(scipy_stats.t.sf(t_stat, df=df))
        elif alternative == "less":
            p_value = float(scipy_stats.t.cdf(t_stat, df=df))
        else:
            raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    return {
        "correlation": r,
        "p_value": p_value,
        "n": n_obs,
        "df": int(df),
    }


def distance_correlation(x: ArrayLike1D, y: ArrayLike1D) -> float:
    """Compute distance correlation for nonlinear dependence detection.

    Distance correlation detects both linear and nonlinear associations.
    Unlike Pearson correlation, it can be close to zero only when variables
    are statistically independent (for finite second moments).

    Args:
        x: First numeric variable.
        y: Second numeric variable.

    Returns:
        Distance correlation value in ``[0, 1]``.

    Raises:
        ValueError: If inputs are invalid or contain too few valid pairs.

    Examples:
        >>> from minexpy.correlation import distance_correlation
        >>> x = np.linspace(-2, 2, 50)
        >>> y = x ** 2
        >>> round(distance_correlation(x, y), 3)
        0.54
    """
    x_clean, y_clean = _prepare_pair(x, y, min_n=2)

    # Pairwise distance matrices (1D absolute distances).
    x_dist = np.abs(x_clean[:, None] - x_clean[None, :])
    y_dist = np.abs(y_clean[:, None] - y_clean[None, :])

    # Double-centering transforms distance matrices to zero-mean space.
    x_centered = (
        x_dist
        - x_dist.mean(axis=0, keepdims=True)
        - x_dist.mean(axis=1, keepdims=True)
        + x_dist.mean()
    )
    y_centered = (
        y_dist
        - y_dist.mean(axis=0, keepdims=True)
        - y_dist.mean(axis=1, keepdims=True)
        + y_dist.mean()
    )

    dcov2 = float(np.mean(x_centered * y_centered))
    dvarx = float(np.mean(x_centered * x_centered))
    dvary = float(np.mean(y_centered * y_centered))

    if dvarx <= 0.0 or dvary <= 0.0:
        return 0.0

    dcor2 = dcov2 / np.sqrt(dvarx * dvary)
    dcor = np.sqrt(max(dcor2, 0.0))
    return float(np.clip(dcor, 0.0, 1.0))


def biweight_midcorrelation(
    x: ArrayLike1D,
    y: ArrayLike1D,
    c: float = 9.0,
    epsilon: float = 1e-12,
) -> float:
    """Compute robust biweight midcorrelation.

    Biweight midcorrelation downweights extreme observations using median
    and MAD scaling. It is useful for geochemical datasets where outliers
    can strongly distort classical correlations.

    Args:
        x: First numeric variable.
        y: Second numeric variable.
        c: Tuning constant controlling outlier downweighting. Lower values
            increase robustness and reduce efficiency in near-normal data.
        epsilon: Small positive stabilizer to avoid division by zero.

    Returns:
        Robust correlation in ``[-1, 1]``. Returns ``np.nan`` when robust
        dispersion of either variable is effectively zero.

    Raises:
        ValueError: If inputs are invalid or contain too few valid pairs.

    Examples:
        >>> from minexpy.correlation import biweight_midcorrelation
        >>> x = np.array([1, 2, 3, 4, 5, 100])
        >>> y = np.array([1, 2, 3, 4, 5, -100])
        >>> round(biweight_midcorrelation(x, y), 3)
        0.996
    """
    x_clean, y_clean = _prepare_pair(x, y, min_n=3)

    x_med = np.median(x_clean)
    y_med = np.median(y_clean)

    x_mad = np.median(np.abs(x_clean - x_med))
    y_mad = np.median(np.abs(y_clean - y_med))

    if x_mad <= epsilon or y_mad <= epsilon:
        return float(np.nan)

    x_u = (x_clean - x_med) / (c * x_mad)
    y_u = (y_clean - y_med) / (c * y_mad)

    x_w = np.zeros_like(x_u)
    y_w = np.zeros_like(y_u)

    x_mask = np.abs(x_u) < 1.0
    y_mask = np.abs(y_u) < 1.0

    x_w[x_mask] = (1.0 - x_u[x_mask] ** 2) ** 2
    y_w[y_mask] = (1.0 - y_u[y_mask] ** 2) ** 2

    x_centered = (x_clean - x_med) * x_w
    y_centered = (y_clean - y_med) * y_w

    denominator = np.sqrt(np.sum(x_centered ** 2) * np.sum(y_centered ** 2))
    if denominator <= epsilon:
        return float(np.nan)

    corr = float(np.sum(x_centered * y_centered) / denominator)
    return float(np.clip(corr, -1.0, 1.0))


def _to_numeric_dataframe(
    data: Union[pd.DataFrame, np.ndarray, Sequence[Sequence[float]]]
) -> pd.DataFrame:
    """Normalize tabular numeric input into a DataFrame.

    Args:
        data: Tabular input as DataFrame, 2D ndarray, or sequence of rows.

    Returns:
        Numeric DataFrame.

    Raises:
        ValueError: If input is not tabular or has no numeric columns.
    """
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        array = np.asarray(data, dtype=float)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2:
            raise ValueError("data must be a 2D structure")
        columns = [f"var_{idx + 1}" for idx in range(array.shape[1])]
        df = pd.DataFrame(array, columns=columns)

    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if numeric_df.shape[1] == 0:
        raise ValueError("data must contain at least one numeric column")

    return numeric_df


def correlation_matrix(
    data: Union[pd.DataFrame, np.ndarray, Sequence[Sequence[float]]],
    method: str = "pearson",
    min_periods: int = 2,
) -> pd.DataFrame:
    """Compute a correlation matrix with geoscience-focused methods.

    Supported methods include:
    - ``pearson``
    - ``spearman``
    - ``kendall``
    - ``distance``
    - ``biweight`` or ``biweight_midcorrelation``

    Args:
        data: Tabular numeric data. If a NumPy array is provided, each
            column is treated as one variable.
        method: Correlation method name.
        min_periods: Minimum number of pairwise valid observations required
            to compute each matrix entry.

    Returns:
        A symmetric ``pandas.DataFrame`` correlation matrix.

    Raises:
        ValueError: If method is unsupported or input is invalid.

    Examples:
        >>> import pandas as pd
        >>> from minexpy.correlation import correlation_matrix
        >>> df = pd.DataFrame({'Zn': [45, 50, 47], 'Cu': [12, 14, 13]})
        >>> correlation_matrix(df, method='pearson')
                   Zn   Cu
        Zn  1.000000  1.0
        Cu  1.000000  1.0
    """
    numeric_df = _to_numeric_dataframe(data)
    method_lower = method.lower()

    if min_periods < 1:
        raise ValueError("min_periods must be at least 1")

    pandas_methods = {"pearson", "spearman", "kendall"}
    if method_lower in pandas_methods:
        return numeric_df.corr(method=method_lower, min_periods=min_periods)

    custom_methods: Mapping[str, object] = {
        "distance": distance_correlation,
        "biweight": biweight_midcorrelation,
        "biweight_midcorrelation": biweight_midcorrelation,
    }

    if method_lower not in custom_methods:
        supported = sorted(list(pandas_methods) + list(custom_methods.keys()))
        raise ValueError(
            f"Unsupported method '{method}'. Supported methods: {supported}"
        )

    method_func = custom_methods[method_lower]
    columns = list(numeric_df.columns)
    matrix = pd.DataFrame(
        np.nan,
        index=columns,
        columns=columns,
        dtype=float,
    )

    for i, col_i in enumerate(columns):
        x_values = numeric_df[col_i].to_numpy(dtype=float)

        for j in range(i, len(columns)):
            col_j = columns[j]
            y_values = numeric_df[col_j].to_numpy(dtype=float)

            mask = np.isfinite(x_values) & np.isfinite(y_values)
            n_valid = int(mask.sum())

            if n_valid < min_periods:
                value = float(np.nan)
            elif i == j:
                value = 1.0
            else:
                value = float(method_func(x_values[mask], y_values[mask]))

            matrix.iloc[i, j] = value
            matrix.iloc[j, i] = value

    return matrix
