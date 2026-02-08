"""
Statistical visualization module for geoscience datasets.

This module provides practical plotting helpers for common statistical
visual diagnostics used during geochemical and environmental data analysis:

- Histogram (linear and log-scale)
- Box plot / violin plot
- ECDF (empirical cumulative distribution function)
- Q-Q plot
- P-P plot
- Scatter plot with optional trend line

All public plotting functions return ``(figure, axis)`` so users can apply
additional Matplotlib customization (annotations, styling, export settings)
after MinexPy constructs the base diagnostic plot.

Examples
--------
Basic histogram:

    >>> import numpy as np
    >>> from minexpy.statviz import plot_histogram
    >>> values = np.random.lognormal(mean=2.2, sigma=0.4, size=200)
    >>> fig, ax = plot_histogram(values, bins=30, scale='log')

Scatter with trend line:

    >>> from minexpy.statviz import plot_scatter
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([2, 4, 6, 8, 10])
    >>> fig, ax = plot_scatter(x, y, add_trendline=True)
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
    """
    Convert an array-like input into a finite 1D float array.

    Parameters
    ----------
    data : array-like
        Input numeric sequence.
    name : str, default 'data'
        Parameter name used in validation error messages.

    Returns
    -------
    numpy.ndarray
        Clean one-dimensional float array with finite values only.

    Raises
    ------
    ValueError
        If input is not one-dimensional or contains no finite values.
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
    """
    Resolve or create a Matplotlib figure/axis pair.

    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        Existing axis to draw on. If ``None``, a new figure and axis are
        created.
    figsize : tuple of float, default (8.0, 5.0)
        Figure size used when creating a new axis.

    Returns
    -------
    tuple
        ``(figure, axis)``.
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
    """
    Normalize one or many datasets into labeled arrays.

    Parameters
    ----------
    data : array-like, mapping, sequence of arrays, or DataFrame
        Input data structure.
    labels : sequence of str, optional
        Optional custom labels for resulting datasets.

    Returns
    -------
    tuple
        Tuple ``(names, arrays)`` where ``names`` is a list of labels and
        ``arrays`` is a list of finite 1D arrays.

    Raises
    ------
    ValueError
        If input cannot be interpreted as one or more valid datasets.
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
    """
    Resolve a SciPy distribution specification to an object.

    Parameters
    ----------
    distribution : str or object
        Distribution name from ``scipy.stats`` or distribution object exposing
        ``cdf`` and ``ppf`` methods.

    Returns
    -------
    object
        SciPy distribution object.

    Raises
    ------
    ValueError
        If the distribution cannot be resolved or lacks required methods.
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
    """
    Plot a histogram with linear or logarithmic x-axis scaling.

    Histograms are often the first diagnostic for distribution shape,
    outlier concentration, and modal behavior. ``scale='log'`` is especially
    useful for right-skewed concentration data with multiplicative structure.

    Parameters
    ----------
    data : array-like
        One-dimensional numeric dataset.
    bins : int or sequence, default 30
        Number of bins or explicit bin edges.
    scale : {'linear', 'log'}, default 'linear'
        X-axis scale and binning mode.
    density : bool, default False
        If ``True``, normalize bar heights to represent density.
    ax : matplotlib.axes.Axes, optional
        Existing axis to draw on.
    color : str, default 'tab:blue'
        Bar face color.
    alpha : float, default 0.75
        Bar opacity.
    label : str, optional
        Legend label for plotted dataset.
    xlabel : str, default 'Value'
        X-axis label.
    ylabel : str, optional
        Y-axis label. If ``None``, set automatically to ``Frequency`` or
        ``Density`` based on ``density``.
    title : str, default 'Histogram'
        Plot title.

    Returns
    -------
    tuple
        ``(figure, axis)`` for additional customization.

    Raises
    ------
    ValueError
        If ``scale`` is invalid or if ``scale='log'`` and data contain
        non-positive values.

    Examples
    --------
    >>> from minexpy.statviz import plot_histogram
    >>> fig, ax = plot_histogram([1, 2, 2, 3, 5], bins=5)
    >>> fig, ax = plot_histogram([1, 2, 3, 4], scale='log')

    Notes
    -----
    For ``scale='log'`` and integer bins, logarithmically spaced bin edges are
    constructed from the finite min/max of the data.
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
    """
    Plot box plot or violin plot for one or multiple datasets.

    Box and violin plots are complementary diagnostics for comparing spread
    and shape across variables, lithological domains, or spatial groups.

    Parameters
    ----------
    data : array-like, mapping, sequence of arrays, or DataFrame
        One dataset or multiple datasets.
    kind : {'box', 'violin'}, default 'box'
        Plot type to generate.
    labels : sequence of str, optional
        Custom labels replacing detected dataset names.
    ax : matplotlib.axes.Axes, optional
        Existing axis to draw on.
    show_means : bool, default True
        If ``True``, display mean marker/line.
    color : str, default 'tab:blue'
        Primary face color for box/violin glyphs.
    xlabel : str, default 'Variables'
        X-axis label.
    ylabel : str, default 'Value'
        Y-axis label.
    title : str, optional
        Plot title. If omitted, a default title is used.

    Returns
    -------
    tuple
        ``(figure, axis)``.

    Raises
    ------
    ValueError
        If ``kind`` is not ``'box'`` or ``'violin'``.

    Examples
    --------
    >>> from minexpy.statviz import plot_box_violin
    >>> fig, ax = plot_box_violin({'Zn': [1, 2, 3], 'Cu': [2, 3, 4]}, kind='violin')

    Notes
    -----
    For DataFrame input, each numeric column is treated as one distribution.
    Non-finite values are removed independently per dataset.
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
    """
    Plot empirical cumulative distribution function (ECDF).

    ECDF plots avoid histogram binning artifacts and are useful for direct
    comparison of distribution shifts and tail behavior across groups.

    Parameters
    ----------
    data : array-like, mapping, sequence of arrays, or DataFrame
        One dataset or multiple datasets.
    labels : sequence of str, optional
        Custom labels replacing detected dataset names.
    ax : matplotlib.axes.Axes, optional
        Existing axis to draw on.
    xlabel : str, default 'Value'
        X-axis label.
    ylabel : str, default 'Empirical Cumulative Probability'
        Y-axis label.
    title : str, default 'ECDF'
        Plot title.

    Returns
    -------
    tuple
        ``(figure, axis)``.

    Examples
    --------
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
    """
    Plot quantile-quantile (Q-Q) diagnostic against a theoretical distribution.

    Q-Q plots assess distributional fit by comparing sample quantiles to
    theoretical quantiles. Tail deviations are especially informative for
    heavy-tailed geochemical variables.

    Parameters
    ----------
    data : array-like
        One-dimensional numeric dataset.
    distribution : str or scipy.stats distribution, default 'norm'
        Reference distribution name or object.
    distribution_parameters : sequence of float, optional
        Fixed distribution parameters. If omitted and ``fit_distribution=True``,
        parameters are estimated from the sample via ``distribution.fit`` when
        available.
    fit_distribution : bool, default True
        Fit distribution parameters from data when not provided.
    show_fit_line : bool, default True
        If ``True``, overlay least-squares line through plotted points.
    ax : matplotlib.axes.Axes, optional
        Existing axis to draw on.
    marker : str, default 'o'
        Marker style for sample quantiles.
    color : str, default 'tab:blue'
        Marker color.
    xlabel : str, default 'Theoretical Quantiles'
        X-axis label.
    ylabel : str, default 'Sample Quantiles'
        Y-axis label.
    title : str, default 'Q-Q Plot'
        Plot title.

    Returns
    -------
    tuple
        ``(figure, axis)``.

    Examples
    --------
    >>> from minexpy.statviz import plot_qq
    >>> fig, ax = plot_qq([1.2, 1.4, 1.8, 2.0, 2.1])

    Notes
    -----
    The dashed 1:1 line shows ideal agreement. Systematic curvature indicates
    mismatch between empirical and theoretical distributions.
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
    """
    Plot probability-probability (P-P) diagnostic against a distribution.

    P-P plots compare empirical cumulative probabilities against theoretical
    CDF values. They are often more sensitive in the central part of the
    distribution than Q-Q plots.

    Parameters
    ----------
    data : array-like
        One-dimensional numeric dataset.
    distribution : str or scipy.stats distribution, default 'norm'
        Reference distribution name or object.
    distribution_parameters : sequence of float, optional
        Fixed distribution parameters. If omitted and ``fit_distribution=True``,
        parameters are estimated from the sample via ``distribution.fit`` when
        available.
    fit_distribution : bool, default True
        Fit distribution parameters from data when not provided.
    ax : matplotlib.axes.Axes, optional
        Existing axis to draw on.
    color : str, default 'tab:blue'
        Marker color.
    xlabel : str, default 'Theoretical Cumulative Probability'
        X-axis label.
    ylabel : str, default 'Empirical Cumulative Probability'
        Y-axis label.
    title : str, default 'P-P Plot'
        Plot title.

    Returns
    -------
    tuple
        ``(figure, axis)``.

    Examples
    --------
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
    """
    Plot scatter data with optional least-squares trend line.

    Scatter plots are central to geoscience exploratory analysis because they
    reveal linear/nonlinear patterns, heteroscedasticity, clustering, and
    potential outliers before formal modeling.

    Parameters
    ----------
    x : array-like
        X-variable values.
    y : array-like
        Y-variable values.
    ax : matplotlib.axes.Axes, optional
        Existing axis to draw on.
    color : str, default 'tab:blue'
        Marker color.
    alpha : float, default 0.8
        Marker opacity.
    marker : str, default 'o'
        Marker style.
    label : str, optional
        Legend label for plotted points.
    add_trendline : bool, default False
        If ``True``, overlay least-squares regression line and annotate
        Pearson ``r`` in legend.
    trendline_color : str, default 'tab:red'
        Trend line color.
    xlabel : str, default 'X'
        X-axis label.
    ylabel : str, default 'Y'
        Y-axis label.
    title : str, default 'Scatter Plot'
        Plot title.

    Returns
    -------
    tuple
        ``(figure, axis)``.

    Raises
    ------
    ValueError
        If vector lengths differ or fewer than two valid paired values exist.

    Examples
    --------
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
