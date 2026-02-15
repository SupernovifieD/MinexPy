# MinexPy

A comprehensive toolkit for geoscience researchers providing tools for geochemical analysis, statistical processing, and visualization of geological sample data.

## Features

- **Statistical Analysis**: Comprehensive statistical tools for geochemical data
- **Correlation Analysis**: Pearson, Spearman, Kendall, robust, nonlinear, and partial correlation tools
- **Statistical Visualization**: Histogram, box/violin, ECDF, Q-Q, P-P, and scatter diagnostics
- **Data Processing**: Efficient data preprocessing and cleaning utilities

## Installation

```bash
pip install minexpy
```

## Quick Start

```python
import minexpy

from minexpy import describe, pearson_correlation

# Descriptive statistics
summary = describe([12.5, 15.3, 18.7, 22.1, 19.4, 16.8])
print(summary["mean"])

# Correlation
result = pearson_correlation([10, 20, 30], [2, 4, 6])
print(result["correlation"])
```

For detailed installation instructions, see the [Installation Guide](getting-started/installation.md).

## Documentation

Browse the full documentation to learn more about using MinexPy.
