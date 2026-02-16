## [0.1.2] - 2026-02-16

### Added
- New `minexpy.mapping` subpackage with end-to-end geochemical mapping workflow:
  - `prepare`, `invert_values_for_display`
  - `create_grid`, `GridDefinition`
  - `interpolate` dispatcher and method functions:
    - `interpolate_nearest`
    - `interpolate_triangulation`
    - `interpolate_idw`
    - `interpolate_minimum_curvature`
  - `plot_map` and `viz` final map composition APIs
- New mapping data/metadata types:
  - `GeochemPrepareMetadata`
  - `GeochemDataWarning`
  - `InterpolationResult`
- Excel input support for mapping data preparation (`.xls`/`.xlsx`) via `openpyxl`.
- New mapping documentation pages:
  - `docs/user-guide/geochemical-mapping.md`
  - `docs/api/mapping.md`
  - `docs/api/statistics.md`
  - `docs/research-notes/mapping-foundations.md`
- New downloadable mapping example dataset and generated map assets:
  - `docs/user-guide/data/demo_data.csv`
  - `docs/user-guide/images/mapping_zn_idw_demo.png`
  - `docs/user-guide/images/mapping_au_minimum_curvature_demo.png`
- `AGENTS.md` for a standardized use of AI development tools

### Changed
- Expanded top-level exports and mapping package exports to include all mapping APIs.
- Updated CLI help/info output with mapping functionality.
- Reorganized API documentation into categorical pages:
  - Overview
  - Statistics API
  - Mapping API
- Updated docs navigation to include:
  - User Guide mapping page
  - Mapping and statistics API pages
  - Mapping research note

### Removed
- Removed unused `docs/getting-started/images/` directory.

## [0.1.1] - 2026-02-08

### Added
- New `minexpy.correlation` module:
  - `pearson_correlation`, `spearman_correlation`, `kendall_correlation`
  - `distance_correlation`, `biweight_midcorrelation`, `partial_correlation`
  - `correlation_matrix`
- New `minexpy.statviz` module:
  - histogram (linear/log), box/violin, ECDF, Q-Q, P-P, scatter plots
- New CLI interface.
- New documentation plot galleries in:
  - `docs/research-notes/examples/`
  - `docs/examples/images/`
- Added research note for the statistics.

### Changed
- Switched mkdocstrings parsing to NumPy style in `mkdocs.yml`.
