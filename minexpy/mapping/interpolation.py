"""
Interpolation utilities for geochemical mapping workflows.

This module implements Step 3 of MinexPy's mapping workflow. It interpolates
prepared point values onto a regular grid using multiple methods.

Examples
--------
Run interpolation with the dispatcher:

    >>> import pandas as pd
    >>> from minexpy.mapping import create_grid, interpolate
    >>> prepared = pd.DataFrame(
    ...     {
    ...         "x": [0.0, 50.0, 100.0, 100.0],
    ...         "y": [0.0, 100.0, 0.0, 100.0],
    ...         "value": [10.0, 15.0, 20.0, 30.0],
    ...     }
    ... )
    >>> grid = create_grid(prepared, cell_size=10.0)
    >>> result = interpolate(prepared, grid, method="idw")
    >>> result.Z.shape == (grid.ny, grid.nx)
    True
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

from .gridding import GridDefinition


@dataclass(frozen=True)
class InterpolationResult:
    """
    Container for interpolation outputs and diagnostics.

    Parameters
    ----------
    grid : GridDefinition
        Grid metadata and geometry used for interpolation.
    Z : numpy.ndarray
        Interpolated values on grid with shape ``(grid.ny, grid.nx)``.
    method : str
        Interpolation method identifier.
    value_col : str
        Name of the source value column used from input data.
    valid_mask : numpy.ndarray
        Boolean mask where interpolated values are finite.
    parameters : dict
        Method parameters used during interpolation.
    converged : bool, optional
        Convergence status for iterative methods (minimum curvature).
    iterations : int, optional
        Number of iterations performed by iterative methods.
    max_change : float, optional
        Final maximum absolute iteration update for iterative methods.
    """

    grid: GridDefinition
    Z: np.ndarray
    method: str
    value_col: str
    valid_mask: np.ndarray
    parameters: Dict[str, object]
    converged: Optional[bool] = None
    iterations: Optional[int] = None
    max_change: Optional[float] = None


def _validate_grid(grid: GridDefinition) -> None:
    """Validate grid structure and numeric consistency."""
    if not isinstance(grid, GridDefinition):
        raise ValueError("grid must be a GridDefinition instance")

    if grid.Xi.shape != grid.Yi.shape:
        raise ValueError("grid.Xi and grid.Yi must have identical shapes")
    if grid.Xi.shape != (grid.ny, grid.nx):
        raise ValueError("grid shape metadata (ny, nx) is inconsistent with Xi/Yi")
    if grid.grid_points.shape != (grid.n_nodes, 2):
        raise ValueError("grid.grid_points shape must be (n_nodes, 2)")
    if grid.n_nodes != grid.nx * grid.ny:
        raise ValueError("grid.n_nodes must equal grid.nx * grid.ny")

    grid_is_finite = (
        np.isfinite(grid.xi).all()
        and np.isfinite(grid.yi).all()
        and np.isfinite(grid.Xi).all()
        and np.isfinite(grid.Yi).all()
        and np.isfinite(grid.grid_points).all()
    )
    if not grid_is_finite:
        raise ValueError("grid contains non-finite values")


def _prepare_interpolation_inputs(
    data: pd.DataFrame,
    grid: GridDefinition,
    value_col: str,
    min_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and prepare points/values arrays for interpolation."""
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame")
    if data.empty:
        raise ValueError("data must not be empty")

    _validate_grid(grid)

    required_columns = [grid.x_col, grid.y_col, value_col]
    missing_columns = [name for name in required_columns if name not in data.columns]
    if missing_columns:
        raise KeyError(f"Missing interpolation columns: {missing_columns}")

    frame = data.copy()
    for column in required_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    x_values = frame[grid.x_col].to_numpy(dtype=float)
    y_values = frame[grid.y_col].to_numpy(dtype=float)
    values = frame[value_col].to_numpy(dtype=float)

    finite_mask = np.isfinite(x_values) & np.isfinite(y_values) & np.isfinite(values)
    points = np.c_[x_values[finite_mask], y_values[finite_mask]]
    clean_values = values[finite_mask]

    if points.shape[0] < min_points:
        raise ValueError(
            f"At least {min_points} finite points are required; received {points.shape[0]}"
        )

    return points, clean_values


def _build_result(
    grid: GridDefinition,
    Z: np.ndarray,
    method: str,
    value_col: str,
    parameters: Dict[str, object],
    converged: Optional[bool] = None,
    iterations: Optional[int] = None,
    max_change: Optional[float] = None,
) -> InterpolationResult:
    """Construct a standardized interpolation result object."""
    if Z.shape != (grid.ny, grid.nx):
        raise ValueError("Interpolated surface shape must be (grid.ny, grid.nx)")

    valid_mask = np.isfinite(Z)
    return InterpolationResult(
        grid=grid,
        Z=Z,
        method=method,
        value_col=value_col,
        valid_mask=valid_mask,
        parameters=parameters,
        converged=converged,
        iterations=iterations,
        max_change=max_change,
    )


def _outside_hull_mask(points: np.ndarray, grid: GridDefinition) -> np.ndarray:
    """Compute mask for nodes outside convex hull using linear triangulation."""
    hull_values = griddata(
        points=points,
        values=np.ones(points.shape[0], dtype=float),
        xi=grid.grid_points,
        method="linear",
        fill_value=np.nan,
    )
    return ~np.isfinite(hull_values.reshape(grid.ny, grid.nx))


def interpolate(
    data: pd.DataFrame,
    grid: GridDefinition,
    method: str = "triangulation",
    value_col: str = "value",
    **kwargs: object,
) -> InterpolationResult:
    """
    Dispatch interpolation to one of the supported methods.

    Parameters
    ----------
    data : pandas.DataFrame
        Prepared point data, typically returned by ``minexpy.mapping.prepare``.
    grid : GridDefinition
        Grid definition returned by ``minexpy.mapping.create_grid``.
    method : {'nearest', 'triangulation', 'idw', 'minimum_curvature'}, default 'triangulation'
        Interpolation method name.
    value_col : str, default 'value'
        Value column name from ``data``.
    **kwargs : dict
        Method-specific keyword arguments forwarded to the selected
        interpolation function.

    Returns
    -------
    InterpolationResult
        Interpolation result object containing surface and diagnostics.

    Raises
    ------
    ValueError
        If ``method`` is not supported.
    """
    method_name = method.strip().lower()

    if method_name == "nearest":
        return interpolate_nearest(data, grid, value_col=value_col, **kwargs)
    if method_name == "triangulation":
        return interpolate_triangulation(data, grid, value_col=value_col, **kwargs)
    if method_name == "idw":
        return interpolate_idw(data, grid, value_col=value_col, **kwargs)
    if method_name == "minimum_curvature":
        return interpolate_minimum_curvature(data, grid, value_col=value_col, **kwargs)

    raise ValueError(
        "Unknown interpolation method. Supported methods are: "
        "'nearest', 'triangulation', 'idw', 'minimum_curvature'."
    )


def interpolate_nearest(
    data: pd.DataFrame,
    grid: GridDefinition,
    value_col: str = "value",
) -> InterpolationResult:
    """
    Interpolate values to grid nodes using nearest neighbor assignment.

    Parameters
    ----------
    data : pandas.DataFrame
        Prepared point data.
    grid : GridDefinition
        Grid definition.
    value_col : str, default 'value'
        Value column name from ``data``.

    Returns
    -------
    InterpolationResult
        Result with nearest-neighbor interpolated surface.

    Examples
    --------
    >>> import pandas as pd
    >>> from minexpy.mapping import create_grid, interpolate_nearest
    >>> d = pd.DataFrame({"x": [0, 10], "y": [0, 10], "value": [1.0, 2.0]})
    >>> g = create_grid(d, cell_size=5.0)
    >>> out = interpolate_nearest(d, g)
    >>> out.Z.shape == (g.ny, g.nx)
    True

    Notes
    -----
    This method is local and piecewise-constant. It does not smooth between
    sample locations.
    """
    points, values = _prepare_interpolation_inputs(
        data=data, grid=grid, value_col=value_col, min_points=1
    )

    tree = cKDTree(points)
    _, neighbor_indices = tree.query(grid.grid_points, k=1)
    Z = values[neighbor_indices].reshape(grid.ny, grid.nx)

    return _build_result(
        grid=grid,
        Z=Z,
        method="nearest",
        value_col=value_col,
        parameters={},
    )


def interpolate_triangulation(
    data: pd.DataFrame,
    grid: GridDefinition,
    value_col: str = "value",
    kind: str = "linear",
) -> InterpolationResult:
    """
    Interpolate values using triangulation-based griddata interpolation.

    Parameters
    ----------
    data : pandas.DataFrame
        Prepared point data.
    grid : GridDefinition
        Grid definition.
    value_col : str, default 'value'
        Value column name from ``data``.
    kind : {'linear', 'cubic'}, default 'linear'
        Triangulation interpolation mode passed to ``scipy.interpolate.griddata``.

    Returns
    -------
    InterpolationResult
        Result with triangulation-based interpolated surface.

    Raises
    ------
    ValueError
        If ``kind`` is unsupported.

    Notes
    -----
    Grid nodes outside the convex hull of input points are returned as NaN.
    """
    kind_normalized = kind.strip().lower()
    if kind_normalized not in {"linear", "cubic"}:
        raise ValueError("kind must be either 'linear' or 'cubic'")

    points, values = _prepare_interpolation_inputs(
        data=data, grid=grid, value_col=value_col, min_points=3
    )

    z_flat = griddata(
        points=points,
        values=values,
        xi=grid.grid_points,
        method=kind_normalized,
        fill_value=np.nan,
    )
    Z = z_flat.reshape(grid.ny, grid.nx)

    return _build_result(
        grid=grid,
        Z=Z,
        method="triangulation",
        value_col=value_col,
        parameters={"kind": kind_normalized},
    )


def interpolate_idw(
    data: pd.DataFrame,
    grid: GridDefinition,
    value_col: str = "value",
    power: float = 2.0,
    k: int = 12,
    radius: Optional[float] = None,
    eps: float = 1e-12,
) -> InterpolationResult:
    """
    Interpolate values using inverse distance weighting (IDW).

    Parameters
    ----------
    data : pandas.DataFrame
        Prepared point data.
    grid : GridDefinition
        Grid definition.
    value_col : str, default 'value'
        Value column name from ``data``.
    power : float, default 2.0
        IDW distance exponent.
    k : int, default 12
        Maximum number of nearest neighbors considered per grid node.
    radius : float, optional
        Optional maximum neighbor distance. If provided, neighbors farther than
        ``radius`` are ignored.
    eps : float, default 1e-12
        Small positive value used to stabilize weight computation near zero
        distance.

    Returns
    -------
    InterpolationResult
        Result with IDW interpolated surface.

    Notes
    -----
    If a grid node coincides with one or more samples, exact sample value
    matching is used instead of weighted averaging.
    """
    if not np.isfinite(power) or power <= 0:
        raise ValueError("power must be a finite float greater than zero")
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    if radius is not None and (not np.isfinite(radius) or radius <= 0):
        raise ValueError("radius must be a finite float greater than zero when provided")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be a finite float greater than zero")

    points, values = _prepare_interpolation_inputs(
        data=data, grid=grid, value_col=value_col, min_points=1
    )

    neighbor_count = min(k, points.shape[0])
    tree = cKDTree(points)

    query_kwargs: Dict[str, object] = {"k": neighbor_count}
    if radius is not None:
        query_kwargs["distance_upper_bound"] = float(radius)

    distances, indices = tree.query(grid.grid_points, **query_kwargs)
    distances = np.asarray(distances, dtype=float)
    indices = np.asarray(indices, dtype=int)

    if distances.ndim == 1:
        distances = distances[:, np.newaxis]
        indices = indices[:, np.newaxis]

    z_flat = np.full(grid.n_nodes, np.nan, dtype=float)
    n_samples = points.shape[0]

    for node_idx in range(grid.n_nodes):
        d_row = distances[node_idx]
        i_row = indices[node_idx]

        valid = np.isfinite(d_row) & (i_row >= 0) & (i_row < n_samples)
        if not np.any(valid):
            continue

        d_valid = d_row[valid]
        i_valid = i_row[valid]
        v_valid = values[i_valid]

        zero_mask = d_valid <= eps
        if np.any(zero_mask):
            z_flat[node_idx] = float(np.mean(v_valid[zero_mask]))
            continue

        weights = 1.0 / np.power(d_valid + eps, power)
        z_flat[node_idx] = float(np.sum(weights * v_valid) / np.sum(weights))

    Z = z_flat.reshape(grid.ny, grid.nx)

    return _build_result(
        grid=grid,
        Z=Z,
        method="idw",
        value_col=value_col,
        parameters={
            "power": float(power),
            "k": int(k),
            "radius": None if radius is None else float(radius),
            "eps": float(eps),
        },
    )


def interpolate_minimum_curvature(
    data: pd.DataFrame,
    grid: GridDefinition,
    value_col: str = "value",
    max_iter: int = 2000,
    tolerance: float = 1e-4,
    relaxation: float = 1.0,
    mask_outside_hull: bool = False,
) -> InterpolationResult:
    """
    Interpolate values using iterative grid-based minimum curvature.

    Parameters
    ----------
    data : pandas.DataFrame
        Prepared point data.
    grid : GridDefinition
        Grid definition.
    value_col : str, default 'value'
        Value column name from ``data``.
    max_iter : int, default 2000
        Maximum number of solver iterations.
    tolerance : float, default 1e-4
        Convergence threshold on maximum absolute node update.
    relaxation : float, default 1.0
        Relaxation factor applied to each node update.
    mask_outside_hull : bool, default False
        If True, nodes outside convex hull are masked to NaN after solving.

    Returns
    -------
    InterpolationResult
        Result with minimum-curvature interpolated surface and convergence
        diagnostics.

    Notes
    -----
    The solver enforces sample constraints on nearest grid nodes at every
    iteration and minimizes surface roughness via a discrete biharmonic
    condition in free nodes.

    References
    ----------
    .. [1] Briggs, I. C. (1974). Machine contouring using minimum curvature.
           Geophysics, 39(1), 39-48.
    """
    if not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be a finite float greater than zero")
    if not np.isfinite(relaxation) or relaxation <= 0:
        raise ValueError("relaxation must be a finite float greater than zero")

    points, values = _prepare_interpolation_inputs(
        data=data, grid=grid, value_col=value_col, min_points=3
    )

    # Initialize with nearest-neighbor field for stable starting surface.
    init_result = interpolate_nearest(data=data, grid=grid, value_col=value_col)
    Z = init_result.Z.copy()

    xi = grid.xi
    yi = grid.yi

    x_indices = np.abs(points[:, 0][:, np.newaxis] - xi[np.newaxis, :]).argmin(axis=1)
    y_indices = np.abs(points[:, 1][:, np.newaxis] - yi[np.newaxis, :]).argmin(axis=1)

    fixed_mask = np.zeros((grid.ny, grid.nx), dtype=bool)
    fixed_values = np.full((grid.ny, grid.nx), np.nan, dtype=float)
    accum_sum = np.zeros((grid.ny, grid.nx), dtype=float)
    accum_count = np.zeros((grid.ny, grid.nx), dtype=int)

    for sample_idx in range(points.shape[0]):
        row = int(y_indices[sample_idx])
        col = int(x_indices[sample_idx])
        accum_sum[row, col] += float(values[sample_idx])
        accum_count[row, col] += 1
        fixed_mask[row, col] = True

    positive_count = accum_count > 0
    fixed_values[positive_count] = accum_sum[positive_count] / accum_count[positive_count]
    Z[fixed_mask] = fixed_values[fixed_mask]

    converged = False
    max_change = np.inf
    iterations = 0

    use_biharmonic = grid.nx >= 5 and grid.ny >= 5

    for iteration in range(1, max_iter + 1):
        max_change_iter = 0.0
        Z_prev = Z.copy()

        if use_biharmonic:
            for row in range(2, grid.ny - 2):
                for col in range(2, grid.nx - 2):
                    if fixed_mask[row, col]:
                        continue

                    z_new = (
                        8.0
                        * (
                            Z_prev[row + 1, col]
                            + Z_prev[row - 1, col]
                            + Z_prev[row, col + 1]
                            + Z_prev[row, col - 1]
                        )
                        - 2.0
                        * (
                            Z_prev[row + 1, col + 1]
                            + Z_prev[row + 1, col - 1]
                            + Z_prev[row - 1, col + 1]
                            + Z_prev[row - 1, col - 1]
                        )
                        - (
                            Z_prev[row + 2, col]
                            + Z_prev[row - 2, col]
                            + Z_prev[row, col + 2]
                            + Z_prev[row, col - 2]
                        )
                    ) / 20.0

                    updated = Z_prev[row, col] + relaxation * (z_new - Z_prev[row, col])
                    delta = abs(updated - Z_prev[row, col])
                    if delta > max_change_iter:
                        max_change_iter = delta
                    Z[row, col] = updated
        else:
            for row in range(1, grid.ny - 1):
                for col in range(1, grid.nx - 1):
                    if fixed_mask[row, col]:
                        continue

                    z_new = 0.25 * (
                        Z_prev[row + 1, col]
                        + Z_prev[row - 1, col]
                        + Z_prev[row, col + 1]
                        + Z_prev[row, col - 1]
                    )
                    updated = Z_prev[row, col] + relaxation * (z_new - Z_prev[row, col])
                    delta = abs(updated - Z_prev[row, col])
                    if delta > max_change_iter:
                        max_change_iter = delta
                    Z[row, col] = updated

        Z[fixed_mask] = fixed_values[fixed_mask]
        iterations = iteration
        max_change = max_change_iter

        if max_change_iter < tolerance:
            converged = True
            break

    if mask_outside_hull:
        outside_mask = _outside_hull_mask(points, grid)
        Z = Z.copy()
        Z[outside_mask] = np.nan

    return _build_result(
        grid=grid,
        Z=Z,
        method="minimum_curvature",
        value_col=value_col,
        parameters={
            "max_iter": int(max_iter),
            "tolerance": float(tolerance),
            "relaxation": float(relaxation),
            "mask_outside_hull": bool(mask_outside_hull),
        },
        converged=bool(converged),
        iterations=int(iterations),
        max_change=float(max_change),
    )


__all__ = [
    "InterpolationResult",
    "interpolate",
    "interpolate_nearest",
    "interpolate_triangulation",
    "interpolate_idw",
    "interpolate_minimum_curvature",
]
