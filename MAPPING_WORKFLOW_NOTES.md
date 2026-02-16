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

