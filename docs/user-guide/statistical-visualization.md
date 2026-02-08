# Statistical Visualization

MinexPy includes a `minexpy.statviz` module for common statistical plots used in geoscience QA/QC and exploratory analysis.

## Available Plot Functions

- `plot_histogram`: Histogram with `linear` or `log` x-axis.
- `plot_box_violin`: Box or violin representation for one or multiple variables.
- `plot_ecdf`: Empirical cumulative distribution function.
- `plot_qq`: Quantile-quantile diagnostic against a theoretical distribution.
- `plot_pp`: Probability-probability diagnostic against a theoretical distribution.
- `plot_scatter`: Scatter plot with optional linear trend line.

All plot functions set axis labels and return `(figure, axis)` for further customization.

## Histogram (Linear / Log)

```python
import matplotlib.pyplot as plt
import numpy as np
from minexpy.statviz import plot_histogram

values = np.random.lognormal(mean=2.2, sigma=0.5, size=300)

fig, ax = plot_histogram(values, bins=30, scale="linear", xlabel="Zn (ppm)")
fig.savefig("hist_linear.png", dpi=150, bbox_inches="tight")

fig, ax = plot_histogram(values, bins=30, scale="log", xlabel="Zn (ppm, log scale)")
fig.savefig("hist_log.png", dpi=150, bbox_inches="tight")
plt.close("all")
```

## Box / Violin

```python
import matplotlib.pyplot as plt
import pandas as pd
from minexpy.statviz import plot_box_violin

df = pd.DataFrame(
    {
        "Zn": [45.2, 52.3, 38.7, 61.2, 49.8, 55.1],
        "Cu": [12.5, 15.3, 11.2, 18.4, 14.1, 16.0],
        "Pb": [8.9, 9.7, 8.1, 10.5, 9.1, 9.9],
    }
)

fig, ax = plot_box_violin(df, kind="box", ylabel="Concentration (ppm)")
fig.savefig("box_plot.png", dpi=150, bbox_inches="tight")

fig, ax = plot_box_violin(df, kind="violin", ylabel="Concentration (ppm)")
fig.savefig("violin_plot.png", dpi=150, bbox_inches="tight")
plt.close("all")
```

## ECDF, Q-Q, P-P, Scatter

```python
import matplotlib.pyplot as plt
import numpy as np
from minexpy.statviz import plot_ecdf, plot_qq, plot_pp, plot_scatter

rng = np.random.default_rng(7)
zn = rng.lognormal(mean=2.2, sigma=0.35, size=250)
cu = 0.3 * zn + rng.normal(0, 1.5, size=250)

fig, ax = plot_ecdf({"Zn": zn, "Cu": cu}, xlabel="Concentration (ppm)")
fig.savefig("ecdf.png", dpi=150, bbox_inches="tight")

fig, ax = plot_qq(zn, distribution="norm", ylabel="Zn Quantiles")
fig.savefig("qq_plot.png", dpi=150, bbox_inches="tight")

fig, ax = plot_pp(zn, distribution="norm", ylabel="Empirical Probability")
fig.savefig("pp_plot.png", dpi=150, bbox_inches="tight")

fig, ax = plot_scatter(
    zn,
    cu,
    add_trendline=True,
    xlabel="Zn (ppm)",
    ylabel="Cu (ppm)",
    title="Zn vs Cu",
)
fig.savefig("scatter_plot.png", dpi=150, bbox_inches="tight")
plt.close("all")
```
