# Mapping Workflow Notes (Draft)

This document stores draft notes for the 4-step mapping workflow and will be moved into the official `docs/` tree after all steps are complete.

## Step 1: Data Loading and Projection Preparation

### Scope

Step 1 adds foundational preprocessing for geochemical point datasets:

- check required columns (`x`, `y`, concentration value column)
- drop rows with missing or non-numeric required values
- warn on duplicate coordinates and drop duplicates (keep first)
- apply coordinate projection (`EPSG:4326 <-> EPSG:3857`)
- support optional custom coordinate transform hook
- support optional value transform (`log10` or callable)
- keep metadata so transformed values can be inverted later for display

### Public API (Step 1)

Implemented in `minexpy/mapping/dataloader.py`:

- `prepare(...)`
- `invert_values_for_display(...)`
- `GeochemPrepareMetadata`
- `GeochemDataWarning`

### Input Support

- In-memory `pandas.DataFrame`
- CSV file path (`.csv`)
- Excel file path (`.xls`, `.xlsx`) via pandas + `openpyxl`

### Output Contract

`prepare(...)` returns:

1. cleaned/prepared `DataFrame`
2. `GeochemPrepareMetadata` with:
   - row-drop counts by reason
   - CRS details
   - transform name
   - inversion capability flags

Prepared table includes canonical columns:

- `x`: projected x coordinate
- `y`: projected y coordinate
- `value_raw`: original numeric concentration
- `value`: working/transformed concentration

### Projection Strategy

Built-in transforms:

- `EPSG:4326 -> EPSG:3857`
- `EPSG:3857 -> EPSG:4326`
- identity when CRS are equal

For all other CRS pairs, pass `coordinate_transform`.

### Value Transform Strategy

- `None`: no transform
- `"log10"`: apply `np.log10`
- callable: custom transform

Rows that become invalid after transform (non-finite values) are warned and dropped.

### Inversion Strategy

`invert_values_for_display` uses metadata:

- no transform: passthrough
- log10: inverse with power-10
- custom transform: raises because inverse is unknown

## Step 2: Grid Creation and Mesh

### Scope

Step 2 builds the base map grid geometry from prepared coordinates:

- compute raw extent (`xmin`, `xmax`, `ymin`, `ymax`)
- apply relative padding (`padding_ratio`, default `0.05`)
- build 1D axes with `np.arange`
- create mesh arrays with `np.meshgrid`
- flatten nodes to `grid_points` for interpolation input

### Public API (Step 2)

Implemented in `minexpy/mapping/gridding.py`:

- `create_grid(...)`
- `GridDefinition`

### API Contract

```python
create_grid(
    data: pd.DataFrame,
    cell_size: float,
    x_col: str = "x",
    y_col: str = "y",
    padding_ratio: float = 0.05,
) -> GridDefinition
```

### Validation Rules

- `data` must be a non-empty DataFrame
- `x_col` and `y_col` must exist
- `cell_size` must be finite and greater than zero
- `padding_ratio` must be finite and non-negative
- coordinate columns must be fully finite numeric values (fail fast if invalid)
- zero range on x or y raises `ValueError`

### GridDefinition Output Fields

- `x_col`, `y_col`
- `raw_extent`: `(xmin, xmax, ymin, ymax)`
- `padded_extent`: `(xmin, xmax, ymin, ymax)`
- `cell_size`, `padding_ratio`
- `xi`, `yi`
- `Xi`, `Yi`
- `grid_points` with shape `(n_nodes, 2)`
- `nx`, `ny`, `n_nodes`

### Notes

Step 2 only constructs grid geometry. Interpolation of geochemical values is
deferred to Step 3.

## Step 3: Interpolation Surface Generation

### Scope

Step 3 interpolates prepared geochemical values onto Step 2 mesh nodes:

- nearest neighbor interpolation
- triangulation interpolation using `scipy.interpolate.griddata`
- inverse distance weighting (IDW)
- true grid-based minimum curvature (iterative biharmonic solver)

### Public API (Step 3)

Implemented in `minexpy/mapping/interpolation.py`:

- `interpolate(...)` dispatcher
- `interpolate_nearest(...)`
- `interpolate_triangulation(...)`
- `interpolate_idw(...)`
- `interpolate_minimum_curvature(...)`
- `InterpolationResult`

### Input Contract

All Step 3 methods accept:

- `data`: prepared `pandas.DataFrame` (typically from `prepare`)
- `grid`: `GridDefinition` (from `create_grid`)
- `value_col`: default `"value"`

### Dispatcher

```python
interpolate(
    data,
    grid,
    method="triangulation",
    value_col="value",
    **kwargs,
)
```

Supported method names:

- `"nearest"`
- `"triangulation"`
- `"idw"`
- `"minimum_curvature"`

### Method Defaults

- Triangulation: `kind="linear"` (optional `kind="cubic"`)
- IDW: `power=2.0`, `k=12`, `radius=None`, `eps=1e-12`
- Minimum curvature:
  - `max_iter=2000`
  - `tolerance=1e-4`
  - `relaxation=1.0`
  - `mask_outside_hull=False`

### Key Behaviors

- Triangulation keeps `NaN` outside convex hull.
- IDW uses exact values for zero-distance matches.
- Minimum curvature enforces mapped sample nodes as fixed constraints and
  reports convergence diagnostics.

### Output

All Step 3 methods return `InterpolationResult`, containing:

- `grid` (for x/y mesh context)
- `Z` interpolated surface (`ny x nx`)
- `valid_mask`
- method name, parameters, and optional convergence diagnostics
