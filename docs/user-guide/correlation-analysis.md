# Correlation Analysis

MinexPy provides a dedicated correlation module for geoscience datasets with missing values, skewed distributions, and occasional outliers.

## Available Correlation Methods

- `pearson_correlation`: Linear association.
- `spearman_correlation`: Rank-based monotonic association.
- `kendall_correlation`: Rank concordance (robust in smaller samples).
- `distance_correlation`: Nonlinear dependence measure.
- `biweight_midcorrelation`: Outlier-resistant robust correlation.
- `partial_correlation`: Correlation between two variables while controlling for confounders.

## Pairwise Correlation Example

```python
import numpy as np
from minexpy.correlation import (
    pearson_correlation,
    spearman_correlation,
    kendall_correlation,
)

zn = np.array([45.2, 52.3, 38.7, 61.2, 49.8, 55.1, 42.3, 58.9])
cu = np.array([12.5, 15.3, 11.2, 18.4, 14.1, 16.0, 12.8, 17.2])

print("Pearson:", pearson_correlation(zn, cu))
print("Spearman:", spearman_correlation(zn, cu))
print("Kendall:", kendall_correlation(zn, cu))
```

## Robust and Nonlinear Example

```python
import numpy as np
from minexpy.correlation import distance_correlation, biweight_midcorrelation

x = np.linspace(-3, 3, 100)
y = x ** 2 + 0.1 * np.random.randn(100)  # nonlinear relation

print("Distance correlation:", distance_correlation(x, y))
print("Biweight midcorrelation:", biweight_midcorrelation(x, y))
```

## Partial Correlation Example

```python
import numpy as np
from minexpy.correlation import partial_correlation

rng = np.random.default_rng(42)
depth = np.linspace(10, 200, 80)
zn = 0.12 * depth + rng.normal(0, 3, size=80)
cu = 0.10 * depth + 0.5 * zn + rng.normal(0, 3, size=80)

# Control for depth while checking Zn-Cu relation
result = partial_correlation(zn, cu, controls=depth)
print(result)
```

## Correlation Matrix Example

```python
import pandas as pd
from minexpy.correlation import correlation_matrix

df = pd.DataFrame(
    {
        "Zn": [45.2, 52.3, 38.7, 61.2, 49.8, 55.1],
        "Cu": [12.5, 15.3, 11.2, 18.4, 14.1, 16.0],
        "Pb": [8.9, 9.7, 8.1, 10.5, 9.1, 9.9],
    }
)

print(correlation_matrix(df, method="pearson"))
print(correlation_matrix(df, method="spearman"))
print(correlation_matrix(df, method="distance"))
```
