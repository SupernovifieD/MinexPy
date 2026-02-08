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
