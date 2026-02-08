# Quick Start Guide

Get started with MinexPy in minutes. This guide will walk you through basic statistical analysis of geochemical data.

## Installation

First, install MinexPy:

```bash
pip install minexpy
```

Or install from source:

```bash
git clone https://github.com/yourusername/MinexPy.git
cd MinexPy
pip install -e .
```

## Basic Usage

### Using Functions

The simplest way to use MinexPy is through its functional API:

```python
import numpy as np
import minexpy.stats as mstats

# Example geochemical data (element concentrations in ppm)
data = np.array([12.5, 15.3, 18.7, 22.1, 19.4, 16.8, 25.0, 14.2])

# Calculate individual metrics
skew = mstats.skewness(data)
kurt = mstats.kurtosis(data)
cv = mstats.coefficient_of_variation(data)

print(f"Skewness: {skew:.3f}")
print(f"Kurtosis: {kurt:.3f}")
print(f"Coefficient of Variation: {cv:.3f}")
```

### Comprehensive Summary

Get all statistics at once:

```python
# Get comprehensive statistical summary
summary = mstats.describe(data)

# Print all metrics
for key, value in summary.items():
    print(f"{key}: {value:.3f}")
```

### Using the StatisticalAnalyzer Class

For more advanced analysis, use the `StatisticalAnalyzer` class:

```python
from minexpy.stats import StatisticalAnalyzer
import pandas as pd

# Single array analysis
data = np.array([12.5, 15.3, 18.7, 22.1, 19.4, 16.8])
analyzer = StatisticalAnalyzer(data)
summary = analyzer.summary()
print(summary)
```

### Analyzing Multiple Elements

When working with multiple geochemical elements:

```python
import pandas as pd
from minexpy.stats import StatisticalAnalyzer

# Load your geochemical data
df = pd.read_csv('geochemical_data.csv')

# Analyze multiple elements
analyzer = StatisticalAnalyzer(df[['Zn', 'Cu', 'Pb', 'Ag']])
summary_df = analyzer.summary()

# The summary is a DataFrame with elements as rows and statistics as columns
print(summary_df)
```

### Accessing Individual Metrics

You can also access individual metrics:

```python
analyzer = StatisticalAnalyzer(data)

# Get specific metrics
skew = analyzer.skewness()
kurt = analyzer.kurtosis()
std_dev = analyzer.std()
mean_val = analyzer.mean()
median_val = analyzer.median()
cv = analyzer.coefficient_of_variation()
iqr_val = analyzer.iqr()
```

### Z-Scores for Outlier Detection

Identify outliers using z-scores:

```python
from minexpy.stats import z_score

# Calculate z-scores for all data points
z_scores = z_score(data)

# Values with |z| > 2 are typically considered outliers
outliers = data[np.abs(z_scores) > 2]
print(f"Outliers: {outliers}")

# Or check a specific value
z_specific = z_score(data, value=30.0)
print(f"Z-score for 30.0: {z_specific:.3f}")
```

### Correlation Analysis

Use the correlation module to quantify relationships between elements:

```python
import numpy as np
from minexpy.correlation import (
    pearson_correlation,
    spearman_correlation,
    kendall_correlation,
    correlation_matrix,
)

zn = np.array([45.2, 52.3, 38.7, 61.2, 49.8, 55.1, 42.3, 58.9])
cu = np.array([12.5, 15.3, 11.2, 18.4, 14.1, 16.0, 12.8, 17.2])

print("Pearson:", pearson_correlation(zn, cu))
print("Spearman:", spearman_correlation(zn, cu))
print("Kendall:", kendall_correlation(zn, cu))

# Matrix form for multiple elements
import pandas as pd
df = pd.DataFrame({"Zn": zn, "Cu": cu})
print(correlation_matrix(df, method="pearson"))
```

### Statistical Visualization

Create diagnostic plots for distribution checks and bivariate trends:

```python
import matplotlib.pyplot as plt
import numpy as np
from minexpy.statviz import (
    plot_histogram,
    plot_box_violin,
    plot_ecdf,
    plot_qq,
    plot_pp,
    plot_scatter,
)

rng = np.random.default_rng(42)
zn = rng.lognormal(mean=2.2, sigma=0.35, size=250)
cu = 0.3 * zn + rng.normal(0, 1.5, size=250)

plot_histogram(zn, bins=30, scale="log", xlabel="Zn (ppm)")
plot_box_violin({"Zn": zn, "Cu": cu}, kind="box", ylabel="Concentration (ppm)")
plot_ecdf({"Zn": zn, "Cu": cu}, xlabel="Concentration (ppm)")
plot_qq(zn, distribution="norm")
plot_pp(zn, distribution="norm")
plot_scatter(zn, cu, add_trendline=True, xlabel="Zn (ppm)", ylabel="Cu (ppm)")

plt.show()
```

## Next Steps

- Read the [User Guide](../user-guide/overview.md) for detailed explanations
- Explore the [API Reference](../api/index.md) for all available functions
- Check out [Examples](../examples/index.md) for more complex use cases
