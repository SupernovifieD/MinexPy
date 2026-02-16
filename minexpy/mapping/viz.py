"""
Final map composition utilities for geochemical mapping workflows.

This module implements Step 4 of MinexPy's mapping workflow by composing
preparation, gridding, interpolation, and cartographic layout into one map.

Examples
--------
Generate a map in one call:

    >>> import pandas as pd
    >>> from minexpy.mapping import plot_map
    >>> df = pd.DataFrame(
    ...     {
    ...         "x": [0, 20, 40, 0, 40],
    ...         "y": [0, 0, 0, 40, 40],
    ...         "Zn": [10.0, 15.0, 23.0, 18.0, 30.0],
    ...     }
    ... )
    >>> fig, ax = plot_map(
    ...     data=df,
    ...     x_col="x",
    ...     y_col="y",
    ...     value_col="Zn",
    ...     cell_size=5.0,
    ...     title_parts={"what": "Zn (ppm)", "where": "Area X", "when": "2026"},
    ... )
"""

from typing import Dict, List, Optional, Sequence, Tuple
import warnings

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd

from .dataloader import GeochemPrepareMetadata, invert_values_for_display, prepare
from .gridding import GridDefinition, create_grid
from .interpolation import InterpolationResult, interpolate


def _warn_ignored(message: str) -> None:
    """Emit a standardized warning for ignored arguments in mixed mode."""
    warnings.warn(message, UserWarning)


def _run_or_resolve_pipeline(
    data: Optional[pd.DataFrame],
    prepared: Optional[pd.DataFrame],
    prepare_metadata: Optional[GeochemPrepareMetadata],
    grid: Optional[GridDefinition],
    interpolation_result: Optional[InterpolationResult],
    x_col: str,
    y_col: str,
    value_col: str,
    source_crs: str,
    target_crs: str,
    coordinate_transform: Optional[object],
    value_transform: Optional[object],
    drop_duplicate_coordinates: bool,
    cell_size: Optional[float],
    padding_ratio: float,
    method: str,
    interpolation_kwargs: Optional[Dict[str, object]],
) -> Tuple[Optional[pd.DataFrame], Optional[GeochemPrepareMetadata], GridDefinition, InterpolationResult]:
    """Resolve mixed raw/precomputed inputs into a concrete interpolation result."""
    interp_kwargs = dict(interpolation_kwargs or {})

    prepared_used = prepared
    metadata_used = prepare_metadata
    grid_used = grid

    if interpolation_result is not None:
        if data is not None or prepared is not None or grid is not None:
            _warn_ignored(
                "interpolation_result was provided; upstream inputs were ignored where conflicting."
            )
        grid_used = interpolation_result.grid
        return prepared_used, metadata_used, grid_used, interpolation_result

    if grid_used is not None:
        if x_col != grid_used.x_col or y_col != grid_used.y_col:
            _warn_ignored("x_col/y_col were ignored because precomputed grid was provided.")
        if cell_size is not None or padding_ratio != 0.05:
            _warn_ignored("cell_size/padding_ratio were ignored because precomputed grid was provided.")

        if prepared_used is None:
            if data is None:
                raise ValueError("Provide prepared data or raw data when using a precomputed grid.")
            prepared_used, meta = prepare(
                data=data,
                x_col=x_col,
                y_col=y_col,
                value_col=value_col,
                source_crs=source_crs,
                target_crs=target_crs,
                coordinate_transform=coordinate_transform,
                value_transform=value_transform,
                drop_duplicate_coordinates=drop_duplicate_coordinates,
            )
            if metadata_used is None:
                metadata_used = meta
        elif data is not None:
            _warn_ignored("raw data input was ignored because prepared data was provided.")

        interp_result = interpolate(
            data=prepared_used,
            grid=grid_used,
            method=method,
            value_col=value_col,
            **interp_kwargs,
        )
        return prepared_used, metadata_used, grid_used, interp_result

    if prepared_used is not None:
        if data is not None:
            _warn_ignored("raw data input was ignored because prepared data was provided.")
        if cell_size is None:
            raise ValueError("cell_size is required when grid is not provided.")

        grid_used = create_grid(
            data=prepared_used,
            cell_size=cell_size,
            x_col=x_col,
            y_col=y_col,
            padding_ratio=padding_ratio,
        )
        interp_result = interpolate(
            data=prepared_used,
            grid=grid_used,
            method=method,
            value_col=value_col,
            **interp_kwargs,
        )
        return prepared_used, metadata_used, grid_used, interp_result

    if data is None:
        raise ValueError(
            "Insufficient inputs. Provide one of: interpolation_result, grid+data/prepared, "
            "prepared (+cell_size), or raw data (+cell_size)."
        )
    if cell_size is None:
        raise ValueError("cell_size is required when using raw data.")

    prepared_used, meta = prepare(
        data=data,
        x_col=x_col,
        y_col=y_col,
        value_col=value_col,
        source_crs=source_crs,
        target_crs=target_crs,
        coordinate_transform=coordinate_transform,
        value_transform=value_transform,
        drop_duplicate_coordinates=drop_duplicate_coordinates,
    )
    if metadata_used is None:
        metadata_used = meta

    grid_used = create_grid(
        data=prepared_used,
        cell_size=cell_size,
        x_col=x_col,
        y_col=y_col,
        padding_ratio=padding_ratio,
    )
    interp_result = interpolate(
        data=prepared_used,
        grid=grid_used,
        method=method,
        value_col=value_col,
        **interp_kwargs,
    )
    return prepared_used, metadata_used, grid_used, interp_result


def _resolve_display_surface(
    interpolation_result: InterpolationResult,
    prepare_metadata: Optional[GeochemPrepareMetadata],
) -> np.ndarray:
    """Resolve displayed surface values, inverting transforms when possible."""
    Z = interpolation_result.Z
    if interpolation_result.value_col == "value_raw":
        return Z
    if interpolation_result.value_col != "value":
        return Z

    if prepare_metadata is None:
        _warn_ignored(
            "prepare_metadata is unavailable; transformed interpolation values are displayed as-is."
        )
        return Z
    if not prepare_metadata.can_invert_for_display:
        _warn_ignored(
            "prepare_metadata does not define an inverse transform; transformed values are displayed as-is."
        )
        return Z

    flattened = Z.ravel()
    restored = invert_values_for_display(flattened, prepare_metadata)
    return restored.reshape(Z.shape)


def _compose_title(title: Optional[str], title_parts: Optional[Dict[str, str]]) -> str:
    """Compose map title from explicit title or structured parts."""
    if title is not None and title.strip():
        return title.strip()
    if title_parts:
        what = title_parts.get("what", "").strip()
        where = title_parts.get("where", "").strip()
        when = title_parts.get("when", "").strip()
        core = " – ".join([part for part in (what, where) if part])
        if core and when:
            return f"{core} ({when})"
        if core:
            return core
    return "Geochemical Interpolation Map"


def _draw_surface_and_colorbar(
    ax: Axes,
    grid: GridDefinition,
    Z_display: np.ndarray,
    cmap: str,
    show_contours: bool,
    contour_levels: int,
    colorbar_label: str,
) -> Colorbar:
    """Draw interpolated surface, optional contours, and colorbar."""
    mesh = ax.pcolormesh(grid.Xi, grid.Yi, Z_display, cmap=cmap, shading="auto")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.12)
    colorbar = ax.figure.colorbar(mesh, cax=cax)
    colorbar.set_label(colorbar_label)

    if show_contours and np.isfinite(Z_display).any():
        ax.contour(
            grid.Xi,
            grid.Yi,
            Z_display,
            levels=contour_levels,
            colors="k",
            linewidths=0.6,
            alpha=0.65,
        )
    return colorbar


def _draw_north_arrow(ax: Axes) -> None:
    """Draw a simple north arrow in axes coordinates."""
    ax.annotate(
        "N",
        xy=(0.95, 0.93),
        xytext=(0.95, 0.80),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        arrowprops={"arrowstyle": "-|>", "lw": 1.5, "color": "black"},
    )


def _nice_number(value: float) -> float:
    """Round value to 1/2/5 x 10^n for readable scale lengths."""
    if value <= 0:
        return 0.0
    exponent = np.floor(np.log10(value))
    fraction = value / (10.0 ** exponent)
    if fraction < 1.5:
        nice_fraction = 1.0
    elif fraction < 3.5:
        nice_fraction = 2.0
    elif fraction < 7.5:
        nice_fraction = 5.0
    else:
        nice_fraction = 10.0
    return nice_fraction * (10.0 ** exponent)


def _is_metric_unit(unit: Optional[str]) -> bool:
    """Check whether unit text indicates metric length units."""
    if unit is None:
        return False
    normalized = unit.strip().lower()
    return normalized in {"m", "meter", "meters", "metre", "metres", "km", "kilometer", "kilometers"}


def _infer_units_from_metadata(
    crs_info: Optional[Dict[str, str]],
    prepare_metadata: Optional[GeochemPrepareMetadata],
) -> Optional[str]:
    """Infer map units from explicit CRS info or fallback metadata."""
    if crs_info and "units" in crs_info and str(crs_info["units"]).strip():
        return str(crs_info["units"]).strip()
    if prepare_metadata is not None:
        crs_name = prepare_metadata.target_crs.upper()
        if "3857" in crs_name:
            return "m"
        if "4326" in crs_name:
            return "deg"
    return None


def _draw_scale_bar(
    ax: Axes,
    grid: GridDefinition,
    units: Optional[str],
) -> None:
    """Draw map scale bar inside the map frame."""
    xmin, xmax, ymin, ymax = grid.padded_extent
    x_span = xmax - xmin
    y_span = ymax - ymin

    if x_span <= 0 or y_span <= 0:
        return

    target_length = x_span * 0.2
    bar_length = _nice_number(target_length)
    if bar_length <= 0:
        return

    x0 = xmin + 0.05 * x_span
    y0 = ymin + 0.06 * y_span
    x1 = x0 + bar_length
    tick_h = 0.02 * y_span

    ax.plot([x0, x1], [y0, y0], color="black", lw=2.0, zorder=5)
    ax.plot([x0, x0], [y0 - tick_h / 2.0, y0 + tick_h / 2.0], color="black", lw=1.2, zorder=5)
    ax.plot([x1, x1], [y0 - tick_h / 2.0, y0 + tick_h / 2.0], color="black", lw=1.2, zorder=5)

    unit_text = units if units is not None else "units"
    ax.text(
        x0 + bar_length / 2.0,
        y0 + 1.2 * tick_h,
        f"{bar_length:g} {unit_text}",
        ha="center",
        va="bottom",
        fontsize=8,
        zorder=5,
    )


def _compute_numeric_scale_text(
    ax: Axes,
    grid: GridDefinition,
    units: Optional[str],
    show_numeric_scale: bool,
) -> Optional[str]:
    """Compute numeric 1:n scale text for external metadata panel."""
    if not show_numeric_scale:
        return None

    if not _is_metric_unit(units):
        _warn_ignored(
            "Numeric scale (1:n) was skipped because map units are not metric or not provided."
        )
        return None

    xmin, xmax, _, _ = grid.padded_extent
    x_span = xmax - xmin
    if x_span <= 0:
        return None

    ax.figure.canvas.draw()
    bbox = ax.get_window_extent()
    ax_width_m = (bbox.width / ax.figure.dpi) * 0.0254
    if ax_width_m <= 0:
        return None

    denominator = x_span / ax_width_m
    if denominator <= 0:
        return None
    return f"1:{int(round(denominator)):,}"


def _collect_crs_info_lines(
    crs_info: Optional[Dict[str, str]],
    prepare_metadata: Optional[GeochemPrepareMetadata],
) -> Tuple[Optional[str], List[str]]:
    """Collect CRS/projection lines and inferred units."""
    info: Dict[str, str] = {}
    if prepare_metadata is not None:
        info["crs"] = prepare_metadata.target_crs
        info["source_crs"] = prepare_metadata.source_crs

    if crs_info:
        for key, value in crs_info.items():
            if value is not None:
                info[key.lower()] = str(value)

    units = _infer_units_from_metadata(crs_info=crs_info, prepare_metadata=prepare_metadata)
    if units is not None and "units" not in info:
        info["units"] = units

    lines = []
    key_map = [
        ("crs", "CRS"),
        ("projection", "Projection"),
        ("datum", "Datum"),
        ("zone", "Zone"),
        ("units", "Units"),
        ("source_crs", "Source CRS"),
    ]
    for key, label in key_map:
        if key in info and str(info[key]).strip():
            lines.append(f"{label}: {info[key]}")
    return units, lines


def _draw_external_info_panel(
    fig: Figure,
    colorbar: Colorbar,
    info_lines: List[str],
) -> Optional[Axes]:
    """Draw metadata panel outside the map, adjacent to the colorbar."""
    if not info_lines:
        return None

    cbar_pos = colorbar.ax.get_position()
    panel_gap = 0.03
    right_space = 0.98 - (cbar_pos.x1 + panel_gap)
    panel_width = min(0.22, right_space)
    if panel_width < 0.10:
        _warn_ignored(
            "Could not allocate enough horizontal space for external info panel; metadata text was skipped."
        )
        return None

    # Keep panel at bottom-right of figure, aligned with the map x-axis.
    line_count = max(1, len(info_lines))
    base_height = 0.11 + 0.026 * line_count
    panel_height = min(cbar_pos.height * 0.55, max(cbar_pos.height * 0.22, base_height))
    panel_rect = [cbar_pos.x1 + panel_gap, cbar_pos.y0, panel_width, panel_height]

    panel_ax = fig.add_axes(panel_rect)
    panel_ax.set_axis_off()
    panel_ax.text(
        0.02,
        0.04,
        "\n".join(info_lines),
        ha="left",
        va="bottom",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.6", "boxstyle": "round,pad=0.25"},
    )
    return panel_ax


def _apply_coordinate_grid_and_neatline(
    ax: Axes,
    show_coordinate_grid: bool,
    show_neatline: bool,
) -> None:
    """Apply coordinate grid lines and neatline styling."""
    if show_coordinate_grid:
        ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.45)
    if show_neatline:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)
            spine.set_color("black")


def _draw_locator_inset(
    fig: Figure,
    grid: GridDefinition,
    locator_config: Optional[Dict[str, object]],
) -> Optional[Axes]:
    """Draw optional locator inset map using user-provided locator configuration."""
    if locator_config is None:
        return None
    if not bool(locator_config.get("enabled", False)):
        return None

    extent = locator_config.get("extent")
    if (
        not isinstance(extent, Sequence)
        or len(extent) != 4
        or not np.isfinite(np.asarray(extent, dtype=float)).all()
    ):
        _warn_ignored(
            "locator_config is enabled but has invalid/missing 'extent'; locator inset was skipped."
        )
        return None

    position = locator_config.get("position", [0.68, 0.68, 0.26, 0.26])
    if not isinstance(position, Sequence) or len(position) != 4:
        _warn_ignored("locator_config 'position' is invalid; default inset position was used.")
        position = [0.68, 0.68, 0.26, 0.26]

    inset_ax = fig.add_axes(position)
    xmin, xmax, ymin, ymax = [float(v) for v in extent]
    inset_ax.set_xlim(xmin, xmax)
    inset_ax.set_ylim(ymin, ymax)
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])

    if bool(locator_config.get("show_main_bbox", True)):
        gxmin, gxmax, gymin, gymax = grid.padded_extent
        bbox = Rectangle(
            (gxmin, gymin),
            gxmax - gxmin,
            gymax - gymin,
            fill=False,
            edgecolor="red",
            linewidth=1.2,
        )
        inset_ax.add_patch(bbox)

    frame_label = locator_config.get("frame_label")
    if frame_label is not None and str(frame_label).strip():
        inset_ax.set_title(str(frame_label), fontsize=8)

    for spine in inset_ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("black")

    return inset_ax


def _draw_footer(fig: Figure, footer: Optional[str]) -> None:
    """Draw free-text footer content at the bottom-left of the figure."""
    if footer is None or not footer.strip():
        return
    fig.text(0.01, 0.01, footer.strip(), ha="left", va="bottom", fontsize=8)


def plot_map(
    data: Optional[pd.DataFrame] = None,
    prepared: Optional[pd.DataFrame] = None,
    prepare_metadata: Optional[GeochemPrepareMetadata] = None,
    grid: Optional[GridDefinition] = None,
    interpolation_result: Optional[InterpolationResult] = None,
    x_col: str = "x",
    y_col: str = "y",
    value_col: str = "value",
    source_crs: str = "EPSG:4326",
    target_crs: str = "EPSG:4326",
    coordinate_transform: Optional[object] = None,
    value_transform: Optional[object] = None,
    drop_duplicate_coordinates: bool = True,
    cell_size: Optional[float] = None,
    padding_ratio: float = 0.05,
    method: str = "idw",
    interpolation_kwargs: Optional[Dict[str, object]] = None,
    title: Optional[str] = None,
    title_parts: Optional[Dict[str, str]] = None,
    cmap: str = "viridis",
    show_contours: bool = False,
    contour_levels: int = 10,
    show_points: bool = True,
    point_size: float = 12.0,
    point_alpha: float = 0.7,
    point_color: str = "black",
    show_north_arrow: bool = True,
    show_scale_bar: bool = True,
    show_numeric_scale: bool = True,
    show_coordinate_grid: bool = True,
    show_neatline: bool = True,
    locator_config: Optional[Dict[str, object]] = None,
    crs_info: Optional[Dict[str, str]] = None,
    footer: Optional[str] = None,
    figsize: Tuple[float, float] = (10.0, 8.0),
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """
    Compose a final geochemical interpolation map with cartographic elements.

    This function can run the full mapping pipeline (`prepare` -> `create_grid`
    -> `interpolate`) or consume precomputed outputs from any stage.

    Parameters
    ----------
    data : pandas.DataFrame, optional
        Raw input table for full-pipeline mode.
    prepared : pandas.DataFrame, optional
        Preprocessed table from Step 1.
    prepare_metadata : GeochemPrepareMetadata, optional
        Step 1 metadata, used for display inversion and CRS annotations.
    grid : GridDefinition, optional
        Precomputed grid from Step 2.
    interpolation_result : InterpolationResult, optional
        Precomputed interpolation from Step 3.
    x_col, y_col, value_col : str
        Coordinate/value column names used in raw or prepared modes.
    source_crs, target_crs : str
        CRS arguments for raw `prepare` calls when needed.
    coordinate_transform : callable, optional
        Optional coordinate transform hook for raw mode.
    value_transform : {None, 'log10', callable}, optional
        Optional value transform for raw mode.
    drop_duplicate_coordinates : bool, default True
        Duplicate handling rule for raw mode.
    cell_size : float, optional
        Grid spacing for modes requiring grid construction.
    padding_ratio : float, default 0.05
        Grid padding ratio for modes requiring grid construction.
    method : str, default 'idw'
        Interpolation method for modes requiring interpolation construction.
    interpolation_kwargs : dict, optional
        Additional keyword arguments for interpolation.
    title : str, optional
        Explicit map title.
    title_parts : dict, optional
        Structured title parts with optional keys `what`, `where`, `when`.
    cmap : str, default 'viridis'
        Colormap name for interpolated surface.
    show_contours : bool, default False
        If True, overlay contours.
    contour_levels : int, default 10
        Number of contour levels when contours are enabled.
    show_points : bool, default True
        If True, overlay sample points when available.
    point_size : float, default 12.0
        Point marker size.
    point_alpha : float, default 0.7
        Point transparency.
    point_color : str, default 'black'
        Point color.
    show_north_arrow : bool, default True
        Draw north arrow.
    show_scale_bar : bool, default True
        Draw scale bar.
    show_numeric_scale : bool, default True
        Draw numeric scale annotation (1:n) when metric units are available.
    show_coordinate_grid : bool, default True
        Draw coordinate grid lines.
    show_neatline : bool, default True
        Draw map frame (neatline).
    locator_config : dict, optional
        Locator inset configuration with keys:
        `enabled`, `position`, `extent`, `show_main_bbox`, `frame_label`.
    crs_info : dict, optional
        CRS metadata dictionary for annotation block.
    footer : str, optional
        Free-text map credits/footer content.
    figsize : tuple of float, default (10.0, 8.0)
        Figure size when creating a new figure.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.

    Returns
    -------
    matplotlib.figure.Figure, matplotlib.axes.Axes
        Final map figure and primary axes.

    Examples
    --------
    >>> import pandas as pd
    >>> from minexpy.mapping import plot_map
    >>> df = pd.DataFrame(
    ...     {"x": [0, 20, 40], "y": [0, 20, 40], "Zn": [12.0, 17.0, 24.0]}
    ... )
    >>> fig, ax = plot_map(df, x_col="x", y_col="y", value_col="Zn", cell_size=5.0)

    Notes
    -----
    Mixed-mode input is allowed. When both upstream and downstream stage
    objects are provided, downstream precomputed objects take precedence and
    ignored upstream inputs emit warnings.
    """
    prepared_used, metadata_used, grid_used, interpolation_used = _run_or_resolve_pipeline(
        data=data,
        prepared=prepared,
        prepare_metadata=prepare_metadata,
        grid=grid,
        interpolation_result=interpolation_result,
        x_col=x_col,
        y_col=y_col,
        value_col=value_col,
        source_crs=source_crs,
        target_crs=target_crs,
        coordinate_transform=coordinate_transform,
        value_transform=value_transform,
        drop_duplicate_coordinates=drop_duplicate_coordinates,
        cell_size=cell_size,
        padding_ratio=padding_ratio,
        method=method,
        interpolation_kwargs=interpolation_kwargs,
    )

    Z_display = _resolve_display_surface(
        interpolation_result=interpolation_used,
        prepare_metadata=metadata_used,
    )

    if ax is None:
        fig, map_ax = plt.subplots(figsize=figsize)
        fig.subplots_adjust(right=0.78)
    else:
        map_ax = ax
        fig = map_ax.figure

    colorbar_label = (
        title_parts.get("what", interpolation_used.value_col)
        if title_parts is not None
        else interpolation_used.value_col
    )
    colorbar = _draw_surface_and_colorbar(
        ax=map_ax,
        grid=grid_used,
        Z_display=Z_display,
        cmap=cmap,
        show_contours=show_contours,
        contour_levels=contour_levels,
        colorbar_label=colorbar_label,
    )

    if show_points:
        if prepared_used is None:
            _warn_ignored(
                "Sample point overlay was requested but prepared data was unavailable; points were skipped."
            )
        elif grid_used.x_col not in prepared_used.columns or grid_used.y_col not in prepared_used.columns:
            _warn_ignored(
                "Sample point overlay was requested but coordinate columns were unavailable; points were skipped."
            )
        else:
            map_ax.scatter(
                prepared_used[grid_used.x_col].to_numpy(dtype=float),
                prepared_used[grid_used.y_col].to_numpy(dtype=float),
                s=point_size,
                alpha=point_alpha,
                c=point_color,
                edgecolors="white",
                linewidths=0.3,
                zorder=4,
            )

    map_ax.set_title(_compose_title(title=title, title_parts=title_parts))
    map_ax.set_xlabel(grid_used.x_col)
    map_ax.set_ylabel(grid_used.y_col)
    map_ax.set_aspect("equal", adjustable="box")

    if show_north_arrow:
        _draw_north_arrow(map_ax)

    units, info_lines = _collect_crs_info_lines(
        crs_info=crs_info,
        prepare_metadata=metadata_used,
    )

    if show_scale_bar:
        _draw_scale_bar(
            ax=map_ax,
            grid=grid_used,
            units=units,
        )
    numeric_scale = _compute_numeric_scale_text(
        ax=map_ax,
        grid=grid_used,
        units=units,
        show_numeric_scale=show_numeric_scale,
    )
    if numeric_scale is not None:
        info_lines.insert(0, f"Scale: {numeric_scale}")

    _draw_external_info_panel(fig=fig, colorbar=colorbar, info_lines=info_lines)

    _apply_coordinate_grid_and_neatline(
        ax=map_ax,
        show_coordinate_grid=show_coordinate_grid,
        show_neatline=show_neatline,
    )

    _draw_locator_inset(fig=fig, grid=grid_used, locator_config=locator_config)
    _draw_footer(fig=fig, footer=footer)

    return fig, map_ax


def viz(*args: object, **kwargs: object) -> Tuple[Figure, Axes]:
    """
    Alias for :func:`plot_map`.

    Returns
    -------
    matplotlib.figure.Figure, matplotlib.axes.Axes
        Final map figure and primary axes.
    """
    return plot_map(*args, **kwargs)


__all__ = [
    "plot_map",
    "viz",
]
