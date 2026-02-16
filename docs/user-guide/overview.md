# User Guide Overview

This section explains how to apply MinexPy's core analysis modules in practical geoscience workflows.

## Core Modules

- `minexpy.stats`: Descriptive statistics, spread metrics, and distribution diagnostics.
- `minexpy.correlation`: Pairwise and matrix-based correlation analysis (linear, rank-based, robust, and nonlinear).
- `minexpy.statviz`: Statistical visualization tools for distribution checks and bivariate analysis.
- `minexpy.mapping`: Geochemical point preparation, grid creation, interpolation, and map composition.

## Recommended Workflow

1. Start with `describe` or `StatisticalAnalyzer.summary` to inspect scale, spread, and distribution shape.
2. Use `correlation_matrix` and pairwise correlation functions to test relationships between variables.
3. Confirm assumptions with visual diagnostics such as histogram, ECDF, Q-Q, and scatter plots.

## Guides in This Section

- [Statistical Analysis](statistical-analysis.md)
- [Correlation Analysis](correlation-analysis.md)
- [Statistical Visualization](statistical-visualization.md)
- [Geochemical Mapping](geochemical-mapping.md)
