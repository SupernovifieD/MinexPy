# Statistical Analysis

MinexPy provides descriptive statistics workflows for single-element and multi-element geochemical datasets. This page consolidates practical statistical analysis examples, starting with the original core workflows first.

## Available Statistical Tools

- `describe`: One-call summary statistics and percentiles.
- `StatisticalAnalyzer`: Object-oriented API for arrays and DataFrames.
- `z_score`: Standard score calculation for anomaly/outlier screening.
- `percentile`: Percentile calculation for threshold-based interpretation.

## Basic Statistical Analysis

Analyze a single geochemical element:

```python
import numpy as np
from minexpy.stats import StatisticalAnalyzer

# Sample zinc (Zn) concentration data (ppm)
zn_data = np.array([45.2, 52.3, 38.7, 61.2, 49.8, 55.1, 42.3, 58.9, 47.6, 51.4])

# Create analyzer
analyzer = StatisticalAnalyzer(zn_data)

# Get comprehensive summary
summary = analyzer.summary()
print(summary)
```

## Multi-Element Analysis

Compare statistics across multiple elements:

```python
import pandas as pd
from minexpy.stats import StatisticalAnalyzer

# Create sample geochemical data
data = {
    "Zn": [45.2, 52.3, 38.7, 61.2, 49.8],
    "Cu": [12.5, 15.3, 18.7, 22.1, 19.4],
    "Pb": [8.3, 10.1, 9.5, 11.2, 9.8],
    "Ag": [0.5, 0.7, 0.6, 0.9, 0.8],
}
df = pd.DataFrame(data)

# Analyze all elements
analyzer = StatisticalAnalyzer(df)
summary_df = analyzer.summary()

print(summary_df)
```

## Outlier Detection with Z-Scores

Identify anomalous values using z-scores:

```python
import numpy as np
from minexpy.stats import z_score

# Geochemical data with potential outliers
data = np.array([12.5, 15.3, 18.7, 22.1, 19.4, 16.8, 45.2, 14.2, 17.9])

# Calculate z-scores
z_scores = z_score(data)

# Identify outliers (|z| > 2)
outlier_mask = np.abs(z_scores) > 2
outliers = data[outlier_mask]
outlier_indices = np.where(outlier_mask)[0]

print(f"Outliers found: {outliers}")
print(f"At indices: {outlier_indices}")
print(f"Z-scores: {z_scores[outlier_mask]}")
```

## Distribution Shape Diagnostics

Interpret skewness, kurtosis, and coefficient of variation:

```python
import numpy as np
from minexpy.stats import StatisticalAnalyzer

data = np.array([12.5, 15.3, 18.7, 22.1, 19.4, 16.8, 25.0, 14.2])

analyzer = StatisticalAnalyzer(data)

# Get distribution metrics
skew = analyzer.skewness()
kurt = analyzer.kurtosis()
cv = analyzer.coefficient_of_variation()

print(f"Skewness: {skew:.3f}")
if skew > 0.5:
    print("  → Right-skewed distribution (positive tail)")
elif skew < -0.5:
    print("  → Left-skewed distribution (negative tail)")
else:
    print("  → Approximately symmetric distribution")

print(f"\nKurtosis: {kurt:.3f}")
if kurt > 0:
    print("  → Heavy tails (more outliers than normal)")
else:
    print("  → Light tails (fewer outliers than normal)")

print(f"\nCoefficient of Variation: {cv:.3f} ({cv * 100:.1f}%)")
if cv < 0.15:
    print("  → Low variability")
elif cv > 0.5:
    print("  → High variability")
else:
    print("  → Moderate variability")
```

## Percentile Analysis

Inspect tail behavior and thresholds with custom percentiles:

```python
import numpy as np
from minexpy.stats import describe

data = np.array([12.5, 15.3, 18.7, 22.1, 19.4, 16.8, 25.0, 14.2, 20.5, 17.6])

# Get summary with custom percentiles
summary = describe(data, percentiles=[10, 25, 50, 75, 90, 95, 99])

# Display key percentiles
print(f"10th percentile: {summary['percentile_10']:.2f}")
print(f"25th percentile (Q1): {summary['percentile_25']:.2f}")
print(f"50th percentile (Median): {summary['percentile_50']:.2f}")
print(f"75th percentile (Q3): {summary['percentile_75']:.2f}")
print(f"90th percentile: {summary['percentile_90']:.2f}")
print(f"95th percentile: {summary['percentile_95']:.2f}")
print(f"99th percentile: {summary['percentile_99']:.2f}")
print(f"\nIQR: {summary['iqr']:.2f}")
```

## Working with CSV Data

Load and analyze selected elements from a CSV file:

```python
import pandas as pd
from minexpy.stats import StatisticalAnalyzer

# Load geochemical data
df = pd.read_csv("geochemical_data.csv")

# Select elements of interest
elements = ["Zn", "Cu", "Pb", "Ag", "Mo"]

# Analyze each element
for element in elements:
    if element in df.columns:
        analyzer = StatisticalAnalyzer(df[element])
        summary = analyzer.summary()

        print(f"\n=== {element} Statistics ===")
        print(f"Mean: {summary['mean']:.2f}")
        print(f"Median: {summary['median']:.2f}")
        print(f"Std Dev: {summary['std']:.2f}")
        print(f"Skewness: {summary['skewness']:.3f}")
        print(f"Kurtosis: {summary['kurtosis']:.3f}")
        print(f"CV: {summary['coefficient_of_variation']:.3f}")
```

## Comparing Variability Across Elements

Compare coefficient of variation (CV) across multiple elements:

```python
import pandas as pd
from minexpy.stats import StatisticalAnalyzer

df = pd.read_csv("geochemical_data.csv")
elements = ["Zn", "Cu", "Pb"]

analyzer = StatisticalAnalyzer(df[elements])
summary = analyzer.summary()

# Compare coefficient of variation
print("Variability Comparison (CV):")
for element in elements:
    cv = summary.loc[element, "coefficient_of_variation"]
    print(f"{element}: {cv:.3f} ({cv * 100:.1f}%)")

# Find most variable element
most_variable = summary["coefficient_of_variation"].idxmax()
print(f"\nMost variable element: {most_variable}")
```
