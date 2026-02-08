"""Statistical visualization utilities for geoscience datasets.

This module provides practical plotting helpers for the most common
statistical diagnostic graphics used in exploration and geochemistry:

- Histogram (linear and log-x)
- Box plot
- Violin plot
- Empirical CDF (ECDF)
- Q-Q plot
- P-P plot
- Scatter plot with optional trend line

Every plotting function returns ``(figure, axes)`` so the caller can further
customize layout, styling, annotations, or export behavior.
"""

from typing import Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

ArrayLike1D = Union[np.ndarray, pd.Series, Sequence[float]]
CollectionLike = Union[
    pd.DataFrame,
    Mapping[str, ArrayLike1D],
    Sequence[ArrayLike1D],
    ArrayLike1D,
]
DistributionLike = Union[str, object]


__all__ = [
    "plot_histogram",
    "plot_box_violin",
    "plot_ecdf",
    "plot_qq",
    "plot_pp",
    "plot_scatter",
]


def _to_1d_float_array(data: ArrayLike1D, name: str = "data") -> np.ndarray:
    """Convert array-like data into a finite 1D float array.

    Args:
        data: Input numeric sequence.
        name: Parameter name for error messages.

    Returns:
        Clean one-dimensional ``numpy.ndarray`` of finite values.

    Raises:
        ValueError: If input is not 1D or contains no finite values.
    """
    if isinstance(data, pd.Series):
        values = data.to_numpy(dtype=float)
    else:
        values = np.asarray(data, dtype=float)

    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")

    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError(f"{name} has no finite numeric values")

    return values


def _resolve_axis(
    ax: Optional[plt.Axes],
    figsize: Tuple[float, float] = (8.0, 5.0),
) -> Tuple[plt.Figure, plt.Axes]:
    """Return a valid Matplotlib figure and axis tuple.

    Args:
        ax: Optional pre-existing axis.
        figsize: Figure size used when creating a new axis.

    Returns:
        ``(figure, axis)`` tuple.
    """
    if ax is None:
        fig, axis = plt.subplots(figsize=figsize)
    else:
        axis = ax
        fig = axis.figure
    return fig, axis


def _parse_collection(
    data: CollectionLike,
    labels: Optional[Sequence[str]] = None,
) -> Tuple[Sequence[str], Sequence[np.ndarray]]:
    """Normalize single or multiple datasets to a labeled list.

    Args:
        data: One dataset or multiple datasets.
        labels: Optional labels corresponding to datasets.

    Returns:
        Tuple of ``(names, arrays)`` where each array is finite and 1D.

    Raises:
        ValueError: If input cannot be interpreted as one or more datasets.
    """
    names = []
    arrays = []

    if isinstance(data, pd.DataFrame):
        if data.shape[1] == 0:
            raise ValueError("DataFrame must contain at least one column")
        for column in data.columns:
            names.append(str(column))
            arrays.append(_to_1d_float_array(data[column], name=f"column '{column}'"))
    elif isinstance(data, Mapping):
        if len(data) == 0:
            raise ValueError("Mapping input must contain at least one dataset")
        for key, values in data.items():
            names.append(str(key))
            arrays.append(_to_1d_float_array(values, name=f"dataset '{key}'"))
    elif isinstance(data, np.ndarray):
        if data.ndim == 1:
            names = ["Series 1"]
            arrays = [_to_1d_float_array(data)]
        elif data.ndim == 2:
            for idx in range(data.shape[1]):
                names.append(f"Series {idx + 1}")
                arrays.append(_to_1d_float_array(data[:, idx], name=f"column {idx + 1}"))
        else:
            raise ValueError("NumPy array input must be 1D or 2D")
    elif isinstance(data, (list, tuple)):
        if len(data) == 0:
            raise ValueError("Input list/tuple must not be empty")

        first = data[0]
        if isinstance(first, (list, tuple, np.ndarray, pd.Series)):
            for idx, values in enumerate(data):
                names.append(f"Series {idx + 1}")
                arrays.append(_to_1d_float_array(values, name=f"series {idx + 1}"))
        else:
            names = ["Series 1"]
            arrays = [_to_1d_float_array(data)]
    else:
        names = ["Series 1"]
        arrays = [_to_1d_float_array(data)]

    if labels is not None:
        if len(labels) != len(arrays):
            raise ValueError("labels length must match number of datasets")
        names = [str(label) for label in labels]

    return names, arrays


def _resolve_distribution(distribution: DistributionLike) -> object:
    """Resolve a SciPy distribution identifier.

    Args:
        distribution: Distribution object or a name in ``scipy.stats``.

    Returns:
        A SciPy continuous distribution object.

    Raises:
        ValueError: If distribution is not found or lacks required methods.
    """
    if isinstance(distribution, str):
        if not hasattr(scipy_stats, distribution):
            raise ValueError(
                f"Unknown distribution '{distribution}'. Provide a valid scipy.stats name."
            )
        dist_obj = getattr(scipy_stats, distribution)
    else:
        dist_obj = distribution

    required_methods = ("cdf", "ppf")
    if not all(hasattr(dist_obj, method) for method in required_methods):
        raise ValueError("distribution must provide both cdf and ppf methods")

    return dist_obj


def plot_histogram(
    data: ArrayLike1D,
    bins: Union[int, Sequence[float]] = 30,
    scale: str = "linear",
    density: bool = False,
    ax: Optional[plt.Axes] = None,
    color: str = "tab:blue",
    alpha: float = 0.75,
    label: Optional[str] = None,
    xlabel: str = "Value",
    ylabel: Optional[str] = None,
    title: str = "Histogram",
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a histogram with linear or logarithmic x-axis scaling.

    Args:
        data: One-dimensional numeric dataset.
        bins: Number of bins or explicit bin edges.
        scale: Histogram x-axis scale. Supported values are ``"linear"``
            and ``"log"``.
        density: If ``True``, normalize bars to form a density.
        ax: Optional existing axis for drawing.
        color: Bar color.
        alpha: Bar opacity.
        label: Optional legend label.
        xlabel: X-axis label.
        ylabel: Y-axis label. If not provided, ``"Frequency"`` or
            ``"Density"`` is selected automatically.
        title: Plot title.

    Returns:
        Tuple of ``(figure, axis)``.

    Raises:
        ValueError: If ``scale='log'`` and data contain non-positive values.

    Examples:
        >>> from minexpy.statviz import plot_histogram
        >>> fig, ax = plot_histogram([1, 2, 2, 3, 5], bins=5)
    """
    values = _to_1d_float_array(data)
    fig, axis = _resolve_axis(ax)

    scale_lower = scale.lower()
    if scale_lower not in {"linear", "log"}:
        raise ValueError("scale must be either 'linear' or 'log'")

    bins_to_use: Union[int, np.ndarray, Sequence[float]] = bins
    if scale_lower == "log":
        if np.any(values <= 0):
            raise ValueError("Log-scale histogram requires strictly positive values")

        if isinstance(bins, int):
            min_value = float(np.min(values))
            max_value = float(np.max(values))
            bins_to_use = np.logspace(np.log10(min_value), np.log10(max_value), bins + 1)

    axis.hist(
        values,
        bins=bins_to_use,
        density=density,
        color=color,
        alpha=alpha,
        edgecolor="black",
        linewidth=0.8,
        label=label,
    )

    if scale_lower == "log":
        axis.set_xscale("log")

    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel if ylabel is not None else ("Density" if density else "Frequency"))
    axis.set_title(title)

    if label:
        axis.legend()

    return fig, axis


def plot_box_violin(
    data: CollectionLike,
    kind: str = "box",
    labels: Optional[Sequence[str]] = None,
    ax: Optional[plt.Axes] = None,
    show_means: bool = True,
    color: str = "tab:blue",
    xlabel: str = "Variables",
    ylabel: str = "Value",
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot box plot or violin plot for one or multiple datasets.

    Args:
        data: One dataset or a collection of datasets. Supported inputs
            include 1D arrays, list of arrays, mapping, and DataFrame.
        kind: Plot type: ``"box"`` or ``"violin"``.
        labels: Optional labels replacing default dataset names.
        ax: Optional existing axis for drawing.
        show_means: If ``True``, mean markers/lines are displayed.
        color: Primary face color used for glyphs.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        title: Optional title. A default title is used if omitted.

    Returns:
        Tuple of ``(figure, axis)``.

    Raises:
        ValueError: If ``kind`` is not supported.

    Examples:
        >>> from minexpy.statviz import plot_box_violin
        >>> fig, ax = plot_box_violin({'Zn': [1, 2, 3], 'Cu': [2, 3, 4]}, kind='violin')
    """
    names, arrays = _parse_collection(data, labels=labels)
    fig, axis = _resolve_axis(ax)

    kind_lower = kind.lower()
    if kind_lower == "box":
        box = axis.boxplot(arrays, labels=names, patch_artist=True, showmeans=show_means)
        for patch in box["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.45)
        default_title = "Box Plot"
    elif kind_lower == "violin":
        violin = axis.violinplot(arrays, showmeans=show_means, showmedians=True)
        for body in violin["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor("black")
            body.set_alpha(0.45)
        axis.set_xticks(np.arange(1, len(names) + 1))
        axis.set_xticklabels(names)
        default_title = "Violin Plot"
    else:
        raise ValueError("kind must be either 'box' or 'violin'")

    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(default_title if title is None else title)

    return fig, axis


def plot_ecdf(
    data: CollectionLike,
    labels: Optional[Sequence[str]] = None,
    ax: Optional[plt.Axes] = None,
    xlabel: str = "Value",
    ylabel: str = "Empirical Cumulative Probability",
    title: str = "ECDF",
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot empirical cumulative distribution function (ECDF).

    Args:
        data: One dataset or collection of datasets.
        labels: Optional labels replacing default dataset names.
        ax: Optional existing axis for drawing.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        title: Plot title.

    Returns:
        Tuple of ``(figure, axis)``.

    Examples:
        >>> from minexpy.statviz import plot_ecdf
        >>> fig, ax = plot_ecdf([1, 3, 2, 4, 8])
    """
    names, arrays = _parse_collection(data, labels=labels)
    fig, axis = _resolve_axis(ax)

    for name, values in zip(names, arrays):
        sorted_values = np.sort(values)
        empirical_prob = np.arange(1, sorted_values.size + 1) / sorted_values.size
        axis.step(sorted_values, empirical_prob, where="post", label=name)

    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)

    if len(arrays) > 1 or labels is not None:
        axis.legend()

    return fig, axis


def plot_qq(
    data: ArrayLike1D,
    distribution: DistributionLike = "norm",
    distribution_parameters: Optional[Sequence[float]] = None,
    fit_distribution: bool = True,
    show_fit_line: bool = True,
    ax: Optional[plt.Axes] = None,
    marker: str = "o",
    color: str = "tab:blue",
    xlabel: str = "Theoretical Quantiles",
    ylabel: str = "Sample Quantiles",
    title: str = "Q-Q Plot",
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot quantile-quantile (Q-Q) diagnostic against a theoretical distribution.

    Args:
        data: One-dimensional numeric dataset.
        distribution: SciPy distribution name or object. Default is normal.
        distribution_parameters: Optional fixed parameters for the chosen
            distribution. If omitted and ``fit_distribution=True``, parameters
            are estimated from data using ``distribution.fit`` when available.
        fit_distribution: If ``True`` and parameters are not provided,
            estimate distribution parameters from data.
        show_fit_line: If ``True``, draw least-squares fit line.
        ax: Optional existing axis for drawing.
        marker: Marker style used for sample quantiles.
        color: Marker color.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        title: Plot title.

    Returns:
        Tuple of ``(figure, axis)``.

    Examples:
        >>> from minexpy.statviz import plot_qq
        >>> fig, ax = plot_qq([1.2, 1.4, 1.8, 2.0, 2.1])
    """
    values = _to_1d_float_array(data)
    dist_obj = _resolve_distribution(distribution)

    params: Tuple[float, ...]
    if distribution_parameters is not None:
        params = tuple(float(p) for p in distribution_parameters)
    elif fit_distribution and hasattr(dist_obj, "fit"):
        params = tuple(float(p) for p in dist_obj.fit(values))
    else:
        params = tuple()

    theoretical_quantiles, sample_quantiles = scipy_stats.probplot(
        values,
        dist=dist_obj,
        sparams=params,
        fit=False,
    )

    fig, axis = _resolve_axis(ax)
    axis.scatter(
        theoretical_quantiles,
        sample_quantiles,
        marker=marker,
        color=color,
        alpha=0.8,
        label="Sample quantiles",
    )

    line_min = min(float(np.min(theoretical_quantiles)), float(np.min(sample_quantiles)))
    line_max = max(float(np.max(theoretical_quantiles)), float(np.max(sample_quantiles)))
    line_x = np.linspace(line_min, line_max, 200)

    axis.plot(line_x, line_x, linestyle="--", color="black", linewidth=1.2, label="1:1")

    if show_fit_line:
        fit = scipy_stats.linregress(theoretical_quantiles, sample_quantiles)
        axis.plot(
            line_x,
            fit.slope * line_x + fit.intercept,
            color="tab:red",
            linewidth=1.4,
            label=f"Fit line (r={fit.rvalue:.3f})",
        )

    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.legend()

    return fig, axis


def plot_pp(
    data: ArrayLike1D,
    distribution: DistributionLike = "norm",
    distribution_parameters: Optional[Sequence[float]] = None,
    fit_distribution: bool = True,
    ax: Optional[plt.Axes] = None,
    color: str = "tab:blue",
    xlabel: str = "Theoretical Cumulative Probability",
    ylabel: str = "Empirical Cumulative Probability",
    title: str = "P-P Plot",
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot probability-probability (P-P) diagnostic against a distribution.

    Args:
        data: One-dimensional numeric dataset.
        distribution: SciPy distribution name or object. Default is normal.
        distribution_parameters: Optional fixed parameters for the chosen
            distribution. If omitted and ``fit_distribution=True``, parameters
            are estimated from data using ``distribution.fit`` when available.
        fit_distribution: If ``True`` and parameters are not provided,
            estimate distribution parameters from data.
        ax: Optional existing axis for drawing.
        color: Point color.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        title: Plot title.

    Returns:
        Tuple of ``(figure, axis)``.

    Examples:
        >>> from minexpy.statviz import plot_pp
        >>> fig, ax = plot_pp([1.2, 1.4, 1.8, 2.0, 2.1])
    """
    values = np.sort(_to_1d_float_array(data))
    dist_obj = _resolve_distribution(distribution)

    params: Tuple[float, ...]
    if distribution_parameters is not None:
        params = tuple(float(p) for p in distribution_parameters)
    elif fit_distribution and hasattr(dist_obj, "fit"):
        params = tuple(float(p) for p in dist_obj.fit(values))
    else:
        params = tuple()

    n = values.size
    empirical_prob = (np.arange(1, n + 1) - 0.5) / n
    theoretical_prob = dist_obj.cdf(values, *params)
    theoretical_prob = np.clip(theoretical_prob, 0.0, 1.0)

    fig, axis = _resolve_axis(ax)
    axis.scatter(theoretical_prob, empirical_prob, color=color, alpha=0.8, label="Observed")
    axis.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="black", linewidth=1.2, label="1:1")

    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.legend()

    return fig, axis


def plot_scatter(
    x: ArrayLike1D,
    y: ArrayLike1D,
    ax: Optional[plt.Axes] = None,
    color: str = "tab:blue",
    alpha: float = 0.8,
    marker: str = "o",
    label: Optional[str] = None,
    add_trendline: bool = False,
    trendline_color: str = "tab:red",
    xlabel: str = "X",
    ylabel: str = "Y",
    title: str = "Scatter Plot",
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot scatter data with optional linear trend line.

    Args:
        x: X-axis values.
        y: Y-axis values.
        ax: Optional existing axis for drawing.
        color: Marker color.
        alpha: Marker opacity.
        marker: Marker style.
        label: Optional label for data points.
        add_trendline: If ``True``, add least-squares regression line and
            annotate Pearson ``r`` in the legend.
        trendline_color: Trend line color.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        title: Plot title.

    Returns:
        Tuple of ``(figure, axis)``.

    Raises:
        ValueError: If paired valid observations are fewer than two.

    Examples:
        >>> from minexpy.statviz import plot_scatter
        >>> fig, ax = plot_scatter([1, 2, 3], [2, 4, 6], add_trendline=True)
    """
    x_values = _to_1d_float_array(x, name="x")
    y_values = _to_1d_float_array(y, name="y")

    if x_values.size != y_values.size:
        raise ValueError("x and y must have the same length")

    mask = np.isfinite(x_values) & np.isfinite(y_values)
    x_clean = x_values[mask]
    y_clean = y_values[mask]

    if x_clean.size < 2:
        raise ValueError("At least two valid paired observations are required")

    fig, axis = _resolve_axis(ax)
    point_label = "Samples" if label is None else label

    axis.scatter(
        x_clean,
        y_clean,
        color=color,
        alpha=alpha,
        marker=marker,
        label=point_label,
    )

    if add_trendline:
        fit = scipy_stats.linregress(x_clean, y_clean)
        line_x = np.linspace(float(np.min(x_clean)), float(np.max(x_clean)), 200)
        line_y = fit.slope * line_x + fit.intercept
        axis.plot(
            line_x,
            line_y,
            color=trendline_color,
            linewidth=1.6,
            label=f"Trend line (r={fit.rvalue:.3f})",
        )

    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)

    if label is not None or add_trendline:
        axis.legend()

    return fig, axis
