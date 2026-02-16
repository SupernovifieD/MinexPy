"""
Data loading and preprocessing utilities for geochemical point mapping.

This module implements Step 1 of MinexPy's mapping workflow. It focuses on
loading point datasets, validating required fields, handling missing/non-numeric
values, projecting coordinates, and applying optional concentration transforms.

Examples
--------
Prepare point data from a DataFrame:

    >>> import pandas as pd
    >>> from minexpy.mapping.dataloader import prepare
    >>>
    >>> df = pd.DataFrame(
    ...     {
    ...         "lon": [44.1, 44.2, 44.3],
    ...         "lat": [36.5, 36.6, 36.7],
    ...         "Cu_ppm": [12.5, 25.0, 18.2],
    ...     }
    ... )
    >>> prepared, meta = prepare(
    ...     data=df,
    ...     x_col="lon",
    ...     y_col="lat",
    ...     value_col="Cu_ppm",
    ...     source_crs="EPSG:4326",
    ...     target_crs="EPSG:3857",
    ...     value_transform="log10",
    ... )
    >>> prepared[["x", "y", "value", "value_raw"]].head()

Invert transformed values for display:

    >>> from minexpy.mapping.dataloader import invert_values_for_display
    >>> restored = invert_values_for_display(prepared["value"], meta)
"""

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple, Union
import warnings

import numpy as np
import pandas as pd

ArrayLike1D = Union[np.ndarray, pd.Series, Sequence[float]]
TabularInput = Union[pd.DataFrame, str, PathLike]
CoordinateTransform = Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
ValueTransform = Callable[[np.ndarray], np.ndarray]

EARTH_RADIUS_METERS = 6378137.0


class GeochemDataWarning(UserWarning):
    """Warning category for non-fatal data quality issues during preparation."""


@dataclass(frozen=True)
class GeochemPrepareMetadata:
    """
    Metadata describing cleaning and transformation actions applied to a dataset.

    Parameters
    ----------
    source_crs : str
        Coordinate reference system identifier for input coordinates.
    target_crs : str
        Coordinate reference system identifier for output coordinates.
    x_column : str
        Name of the input x-coordinate column.
    y_column : str
        Name of the input y-coordinate column.
    value_column : str
        Name of the input concentration column.
    n_input_rows : int
        Number of rows in input data before cleaning.
    n_dropped_nan_or_non_numeric : int
        Number of rows dropped due to missing/non-numeric required fields.
    n_dropped_projection_invalid : int
        Number of rows dropped due to non-finite projected coordinates.
    n_dropped_value_transform_invalid : int
        Number of rows dropped due to invalid transformed concentration values.
    n_dropped_duplicates : int
        Number of duplicate coordinate rows dropped.
    value_transform_applied : bool
        Whether a value transformation was applied.
    value_transform_name : str
        Name of the value transformation ('none', 'log10', or callable name).
    can_invert_for_display : bool
        Whether transformed values can be inverted back for display.
    inverse_transform_name : str, optional
        Name of inverse transform, when available.
    """

    source_crs: str
    target_crs: str
    x_column: str
    y_column: str
    value_column: str
    n_input_rows: int
    n_dropped_nan_or_non_numeric: int
    n_dropped_projection_invalid: int
    n_dropped_value_transform_invalid: int
    n_dropped_duplicates: int
    value_transform_applied: bool
    value_transform_name: str
    can_invert_for_display: bool
    inverse_transform_name: Optional[str] = None


def _normalize_crs(crs: str, name: str) -> str:
    """Normalize and validate CRS string inputs."""
    if not isinstance(crs, str) or not crs.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return crs.strip().upper()


def _load_tabular_input(data: TabularInput) -> pd.DataFrame:
    """Load tabular data from DataFrame, CSV, or Excel."""
    if isinstance(data, pd.DataFrame):
        return data.copy()

    if not isinstance(data, (str, PathLike)):
        raise ValueError(
            "data must be a pandas DataFrame or a file path to CSV/Excel input"
        )

    path = Path(data)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xls", ".xlsx"}:
        try:
            return pd.read_excel(path)
        except ImportError as exc:
            raise ImportError(
                "Reading Excel files requires openpyxl>=3.1.0. "
                "Install it with 'pip install openpyxl>=3.1.0'."
            ) from exc

    raise ValueError("Unsupported file type. Supported extensions are: .csv, .xls, .xlsx")


def _project_4326_to_3857(x_values: np.ndarray, y_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Project longitude/latitude degrees to Web Mercator meters."""
    lon_rad = np.deg2rad(x_values)
    lat_rad = np.deg2rad(y_values)
    projected_x = EARTH_RADIUS_METERS * lon_rad
    projected_y = EARTH_RADIUS_METERS * np.log(np.tan((np.pi / 4.0) + (lat_rad / 2.0)))
    return projected_x, projected_y


def _project_3857_to_4326(x_values: np.ndarray, y_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Project Web Mercator meters to longitude/latitude degrees."""
    lon_deg = np.rad2deg(x_values / EARTH_RADIUS_METERS)
    lat_deg = np.rad2deg((2.0 * np.arctan(np.exp(y_values / EARTH_RADIUS_METERS))) - (np.pi / 2.0))
    return lon_deg, lat_deg


def _resolve_coordinate_transform(
    source_crs: str,
    target_crs: str,
    coordinate_transform: Optional[CoordinateTransform],
) -> CoordinateTransform:
    """Resolve which coordinate transform function should be used."""
    if coordinate_transform is not None:
        return coordinate_transform

    if source_crs == target_crs:
        return lambda x, y: (x, y)
    if source_crs == "EPSG:4326" and target_crs == "EPSG:3857":
        return _project_4326_to_3857
    if source_crs == "EPSG:3857" and target_crs == "EPSG:4326":
        return _project_3857_to_4326

    raise ValueError(
        f"Unsupported CRS transform: {source_crs} -> {target_crs}. "
        "Use coordinate_transform hook for custom CRS pairs."
    )


def _apply_value_transform(
    values: np.ndarray,
    value_transform: Optional[Union[str, ValueTransform]],
) -> Tuple[np.ndarray, str, bool, Optional[str]]:
    """Apply optional value transformation and report inversion metadata."""
    if value_transform is None:
        return values.copy(), "none", True, "identity"

    if isinstance(value_transform, str):
        transform_name = value_transform.strip().lower()
        if transform_name != "log10":
            raise ValueError("Only 'log10' is supported as a string value_transform")
        with np.errstate(divide="ignore", invalid="ignore"):
            transformed = np.log10(values)
        return transformed, "log10", True, "power10"

    if callable(value_transform):
        transformed = np.asarray(value_transform(values.copy()), dtype=float)
        if transformed.shape != values.shape:
            raise ValueError("Custom value_transform must return an array with the same shape")
        custom_name = getattr(value_transform, "__name__", "custom_transform")
        return transformed, str(custom_name), False, None

    raise ValueError("value_transform must be None, 'log10', or a callable")


def prepare(
    data: TabularInput,
    x_col: str,
    y_col: str,
    value_col: str,
    source_crs: str = "EPSG:4326",
    target_crs: str = "EPSG:4326",
    coordinate_transform: Optional[CoordinateTransform] = None,
    value_transform: Optional[Union[str, ValueTransform]] = None,
    drop_duplicate_coordinates: bool = True,
) -> Tuple[pd.DataFrame, GeochemPrepareMetadata]:
    """
    Load and prepare geochemical point data for mapping workflows.

    The function validates required columns, enforces numeric types, removes
    rows with missing values, projects coordinates, applies optional value
    transformation, and optionally drops duplicated coordinates.

    Parameters
    ----------
    data : DataFrame or path-like
        Input geochemical point table, or path to `.csv`, `.xls`, or `.xlsx`.
    x_col : str
        Column name for x coordinate (longitude/easting).
    y_col : str
        Column name for y coordinate (latitude/northing).
    value_col : str
        Column name containing element concentration values.
    source_crs : str, default 'EPSG:4326'
        Input coordinate reference system.
    target_crs : str, default 'EPSG:4326'
        Output coordinate reference system.
    coordinate_transform : callable, optional
        Optional coordinate hook with signature `(x, y) -> (x_new, y_new)`.
        When provided, it takes precedence over built-in CRS transforms.
    value_transform : {None, 'log10', callable}, default None
        Optional concentration transform. `'log10'` applies base-10 logarithm.
        Custom callables must return an array of the same shape.
    drop_duplicate_coordinates : bool, default True
        If True, duplicate `(x, y)` rows are dropped after warning.

    Returns
    -------
    DataFrame, GeochemPrepareMetadata
        Prepared table and metadata describing all cleaning/transformation
        actions.

    Examples
    --------
    >>> import pandas as pd
    >>> from minexpy.mapping.dataloader import prepare
    >>> df = pd.DataFrame(
    ...     {"x": [44.1, 44.1, 44.2], "y": [36.5, 36.5, 36.6], "Zn": [11.0, 11.0, 15.5]}
    ... )
    >>> prepared, meta = prepare(df, "x", "y", "Zn", value_transform="log10")
    >>> prepared[["x", "y", "value", "value_raw"]].head()

    Notes
    -----
    Built-in projection support is intentionally limited to:

    - `EPSG:4326` to `EPSG:3857`
    - `EPSG:3857` to `EPSG:4326`
    - identity when source and target CRS are equal

    For other CRS pairs, pass a custom `coordinate_transform` callable.

    References
    ----------
    .. [1] Snyder, J. P. (1987). Map Projections: A Working Manual.
           U.S. Geological Survey Professional Paper 1395.
    """
    source_crs_normalized = _normalize_crs(source_crs, "source_crs")
    target_crs_normalized = _normalize_crs(target_crs, "target_crs")

    frame = _load_tabular_input(data)
    required_columns = [x_col, y_col, value_col]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    n_input_rows = int(len(frame))
    frame = frame.copy()
    for column in required_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    valid_required_mask = frame[required_columns].notna().all(axis=1).to_numpy()
    n_dropped_nan_or_non_numeric = int((~valid_required_mask).sum())
    if n_dropped_nan_or_non_numeric > 0:
        warnings.warn(
            f"Dropped {n_dropped_nan_or_non_numeric} rows with missing or non-numeric "
            "required fields.",
            GeochemDataWarning,
        )

    frame = frame.loc[valid_required_mask].copy()
    if frame.empty:
        raise ValueError("No valid rows remain after removing missing/non-numeric required fields")

    x_values = frame[x_col].to_numpy(dtype=float)
    y_values = frame[y_col].to_numpy(dtype=float)
    raw_values = frame[value_col].to_numpy(dtype=float)

    transform_coordinates = _resolve_coordinate_transform(
        source_crs=source_crs_normalized,
        target_crs=target_crs_normalized,
        coordinate_transform=coordinate_transform,
    )
    projected = transform_coordinates(x_values.copy(), y_values.copy())
    if not isinstance(projected, tuple) or len(projected) != 2:
        raise ValueError("coordinate_transform must return a tuple: (x_array, y_array)")

    projected_x = np.asarray(projected[0], dtype=float)
    projected_y = np.asarray(projected[1], dtype=float)
    if projected_x.shape != x_values.shape or projected_y.shape != y_values.shape:
        raise ValueError(
            "coordinate_transform must return arrays with the same shape as input coordinates"
        )

    finite_projection_mask = np.isfinite(projected_x) & np.isfinite(projected_y)
    n_dropped_projection_invalid = int((~finite_projection_mask).sum())
    if n_dropped_projection_invalid > 0:
        warnings.warn(
            f"Dropped {n_dropped_projection_invalid} rows with invalid projected coordinates.",
            GeochemDataWarning,
        )
        frame = frame.loc[finite_projection_mask].copy()
        projected_x = projected_x[finite_projection_mask]
        projected_y = projected_y[finite_projection_mask]
        raw_values = raw_values[finite_projection_mask]

    if frame.empty:
        raise ValueError("No valid rows remain after coordinate projection filtering")

    transformed_values, transform_name, can_invert, inverse_name = _apply_value_transform(
        raw_values, value_transform
    )
    finite_value_mask = np.isfinite(transformed_values)
    n_dropped_value_transform_invalid = int((~finite_value_mask).sum())
    if n_dropped_value_transform_invalid > 0:
        warnings.warn(
            f"Dropped {n_dropped_value_transform_invalid} rows with invalid transformed values.",
            GeochemDataWarning,
        )
        frame = frame.loc[finite_value_mask].copy()
        projected_x = projected_x[finite_value_mask]
        projected_y = projected_y[finite_value_mask]
        raw_values = raw_values[finite_value_mask]
        transformed_values = transformed_values[finite_value_mask]

    if frame.empty:
        raise ValueError("No valid rows remain after value transformation filtering")

    frame["x"] = projected_x
    frame["y"] = projected_y
    frame["value_raw"] = raw_values
    frame["value"] = transformed_values

    duplicate_mask = frame.duplicated(subset=["x", "y"], keep="first").to_numpy()
    n_duplicates = int(duplicate_mask.sum())
    n_dropped_duplicates = 0
    if n_duplicates > 0:
        action = "dropped (keeping first occurrence)" if drop_duplicate_coordinates else "retained"
        warnings.warn(
            f"Detected {n_duplicates} duplicate coordinate rows; duplicates were {action}.",
            GeochemDataWarning,
        )
        if drop_duplicate_coordinates:
            frame = frame.loc[~duplicate_mask].copy()
            n_dropped_duplicates = n_duplicates

    if frame.empty:
        raise ValueError("No valid rows remain after duplicate-coordinate filtering")

    metadata = GeochemPrepareMetadata(
        source_crs=source_crs_normalized,
        target_crs=target_crs_normalized,
        x_column=x_col,
        y_column=y_col,
        value_column=value_col,
        n_input_rows=n_input_rows,
        n_dropped_nan_or_non_numeric=n_dropped_nan_or_non_numeric,
        n_dropped_projection_invalid=n_dropped_projection_invalid,
        n_dropped_value_transform_invalid=n_dropped_value_transform_invalid,
        n_dropped_duplicates=n_dropped_duplicates,
        value_transform_applied=(transform_name != "none"),
        value_transform_name=transform_name,
        can_invert_for_display=can_invert,
        inverse_transform_name=inverse_name,
    )

    return frame.reset_index(drop=True), metadata


def invert_values_for_display(
    values: ArrayLike1D,
    metadata: GeochemPrepareMetadata,
) -> np.ndarray:
    """
    Invert transformed concentration values for display when possible.

    Parameters
    ----------
    values : array-like
        Transformed or untransformed concentration values.
    metadata : GeochemPrepareMetadata
        Metadata returned by `prepare`.

    Returns
    -------
    numpy.ndarray
        Values mapped back to display scale when inversion is available.

    Raises
    ------
    ValueError
        If inversion is not available for the applied value transform.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from minexpy.mapping.dataloader import prepare, invert_values_for_display
    >>> df = pd.DataFrame({"x": [1, 2], "y": [3, 4], "Cu": [10.0, 100.0]})
    >>> prepared, meta = prepare(df, "x", "y", "Cu", value_transform="log10")
    >>> np.allclose(invert_values_for_display(prepared["value"], meta), prepared["value_raw"])
    True
    """
    array = np.asarray(values, dtype=float)

    if not metadata.value_transform_applied or metadata.value_transform_name in {"none", "identity"}:
        return array.copy()

    if metadata.value_transform_name == "log10" and metadata.can_invert_for_display:
        return np.power(10.0, array)

    raise ValueError(
        "Cannot invert values for display because the applied transform does not "
        "define a known inverse in metadata."
    )


__all__ = [
    "GeochemDataWarning",
    "GeochemPrepareMetadata",
    "prepare",
    "invert_values_for_display",
]
