"""
Grid creation utilities for geochemical mapping workflows.

This module implements Step 2 of MinexPy's mapping workflow. It builds a
regular 2D grid (base layer) from prepared point coordinates so later steps
can perform interpolation and map rendering.

Examples
--------
Create a mesh from prepared point data:

    >>> import pandas as pd
    >>> from minexpy.mapping import create_grid
    >>> data = pd.DataFrame(
    ...     {
    ...         "x": [100.0, 130.0, 140.0, 170.0],
    ...         "y": [200.0, 210.0, 240.0, 260.0],
    ...     }
    ... )
    >>> grid = create_grid(data, cell_size=10.0)
    >>> grid.Xi.shape
    (8, 10)
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


Extent = Tuple[float, float, float, float]


@dataclass(frozen=True)
class GridDefinition:
    """
    Container for mesh grid geometry and metadata.

    Parameters
    ----------
    x_col : str
        Name of the x-coordinate column used for grid creation.
    y_col : str
        Name of the y-coordinate column used for grid creation.
    raw_extent : tuple of float
        Unpadded data extent as ``(xmin, xmax, ymin, ymax)``.
    padded_extent : tuple of float
        Padded grid extent as ``(xmin, xmax, ymin, ymax)``.
    cell_size : float
        Uniform grid spacing used for both x and y axes.
    padding_ratio : float
        Relative padding applied to each axis range.
    xi : numpy.ndarray
        One-dimensional x-axis coordinates of the grid.
    yi : numpy.ndarray
        One-dimensional y-axis coordinates of the grid.
    Xi : numpy.ndarray
        Two-dimensional x-coordinate mesh from ``numpy.meshgrid``.
    Yi : numpy.ndarray
        Two-dimensional y-coordinate mesh from ``numpy.meshgrid``.
    grid_points : numpy.ndarray
        Flattened grid nodes of shape ``(n_nodes, 2)`` for interpolation.
    nx : int
        Number of nodes along x-axis.
    ny : int
        Number of nodes along y-axis.
    n_nodes : int
        Total node count (``nx * ny``).
    """

    x_col: str
    y_col: str
    raw_extent: Extent
    padded_extent: Extent
    cell_size: float
    padding_ratio: float
    xi: np.ndarray
    yi: np.ndarray
    Xi: np.ndarray
    Yi: np.ndarray
    grid_points: np.ndarray
    nx: int
    ny: int
    n_nodes: int


def create_grid(
    data: pd.DataFrame,
    cell_size: float,
    x_col: str = "x",
    y_col: str = "y",
    padding_ratio: float = 0.05,
) -> GridDefinition:
    """
    Build a regular mesh grid from prepared geochemical point coordinates.

    The function computes data extent, applies relative padding, creates
    one-dimensional grid axes, builds a 2D mesh using ``numpy.meshgrid``, and
    provides a flattened ``(x, y)`` node array for interpolation routines.

    Parameters
    ----------
    data : pandas.DataFrame
        Prepared point dataset, typically returned by
        ``minexpy.mapping.prepare``. Must contain finite numeric coordinate
        columns.
    cell_size : float
        Uniform grid spacing for x and y axes. Must be finite and greater
        than zero.
    x_col : str, default 'x'
        Name of x-coordinate column.
    y_col : str, default 'y'
        Name of y-coordinate column.
    padding_ratio : float, default 0.05
        Relative padding added to each side of the data extent. Must be
        finite and non-negative.

    Returns
    -------
    GridDefinition
        Dataclass containing extents, axes, mesh arrays, flattened points,
        and node counts.

    Raises
    ------
    ValueError
        If input data is invalid, coordinate values are non-finite, axis
        ranges are zero, or numeric parameters are invalid.
    KeyError
        If ``x_col`` or ``y_col`` does not exist in ``data``.

    Examples
    --------
    >>> import pandas as pd
    >>> from minexpy.mapping import create_grid
    >>> prepared = pd.DataFrame(
    ...     {
    ...         "x": [0.0, 100.0, 200.0],
    ...         "y": [0.0, 50.0, 100.0],
    ...     }
    ... )
    >>> grid = create_grid(prepared, cell_size=25.0, padding_ratio=0.10)
    >>> grid.grid_points.shape[1]
    2

    Notes
    -----
    This step only creates mesh geometry. It does not interpolate
    concentration values. Interpolation is handled in a later mapping step.

    References
    ----------
    .. [1] Burrough, P. A., & McDonnell, R. A. (1998). Principles of
           Geographical Information Systems. Oxford University Press.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame")
    if data.empty:
        raise ValueError("data must not be empty")
    if x_col not in data.columns or y_col not in data.columns:
        missing = [name for name in (x_col, y_col) if name not in data.columns]
        raise KeyError(f"Missing coordinate columns: {missing}")

    if not np.isfinite(cell_size) or cell_size <= 0:
        raise ValueError("cell_size must be a finite float greater than zero")
    if not np.isfinite(padding_ratio) or padding_ratio < 0:
        raise ValueError("padding_ratio must be a finite float greater than or equal to zero")

    x_values = pd.to_numeric(data[x_col], errors="coerce").to_numpy(dtype=float)
    y_values = pd.to_numeric(data[y_col], errors="coerce").to_numpy(dtype=float)

    finite_mask = np.isfinite(x_values) & np.isfinite(y_values)
    if not finite_mask.all():
        invalid_count = int((~finite_mask).sum())
        raise ValueError(
            f"Coordinate columns must contain only finite numeric values; "
            f"found {invalid_count} invalid rows"
        )

    xmin = float(np.min(x_values))
    xmax = float(np.max(x_values))
    ymin = float(np.min(y_values))
    ymax = float(np.max(y_values))
    raw_extent: Extent = (xmin, xmax, ymin, ymax)

    x_range = xmax - xmin
    y_range = ymax - ymin
    if x_range == 0:
        raise ValueError("x range is zero; cannot create 2D grid from identical x values")
    if y_range == 0:
        raise ValueError("y range is zero; cannot create 2D grid from identical y values")

    x_pad = x_range * float(padding_ratio)
    y_pad = y_range * float(padding_ratio)

    xmin_pad = xmin - x_pad
    xmax_pad = xmax + x_pad
    ymin_pad = ymin - y_pad
    ymax_pad = ymax + y_pad
    padded_extent: Extent = (xmin_pad, xmax_pad, ymin_pad, ymax_pad)

    xi = np.arange(xmin_pad, xmax_pad + cell_size, cell_size, dtype=float)
    yi = np.arange(ymin_pad, ymax_pad + cell_size, cell_size, dtype=float)

    Xi, Yi = np.meshgrid(xi, yi)
    grid_points = np.c_[Xi.ravel(), Yi.ravel()]

    nx = int(xi.size)
    ny = int(yi.size)
    n_nodes = int(grid_points.shape[0])

    return GridDefinition(
        x_col=x_col,
        y_col=y_col,
        raw_extent=raw_extent,
        padded_extent=padded_extent,
        cell_size=float(cell_size),
        padding_ratio=float(padding_ratio),
        xi=xi,
        yi=yi,
        Xi=Xi,
        Yi=Yi,
        grid_points=grid_points,
        nx=nx,
        ny=ny,
        n_nodes=n_nodes,
    )


__all__ = [
    "GridDefinition",
    "create_grid",
]
