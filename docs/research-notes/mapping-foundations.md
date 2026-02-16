# Geochemical Mapping Foundations

<!-- Enable MathJax on this page so LaTeX renders when browsing the local MkDocs site -->
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

This note documents the scientific and numerical foundations behind MinexPy's
four-step mapping workflow:

1. point-data preparation (`prepare`)
2. grid construction (`create_grid`)
3. interpolation (`interpolate*`)
4. final cartographic composition (`plot_map` / `viz`)

The goal is to explain why each step is required, what equations are used,
and what assumptions affect interpretation.

## Scope and notation

Given point observations with coordinates and one measured element:

$$
\{(x_i, y_i, z_i)\}_{i=1}^{N}
$$

where:

- \(x_i\), \(y_i\): coordinate pair for sample \(i\)
- \(z_i\): concentration value for sample \(i\)
- \(N\): number of valid samples after cleaning

The mapping objective is to estimate a continuous surface:

$$
z = f(x, y)
$$

over a regular grid for visualization and interpretation.

## Step 1: Data preparation and projection

Step 1 ensures the interpolation problem is numerically valid before any
surface modeling is attempted.

### Why this step is necessary

- Interpolation requires finite numeric coordinates and values.
- Coordinate reference systems (CRS) must be consistent.
- Duplicate points can overweight local influence.
- Transforming skewed concentrations can stabilize interpolation behavior.

### Core operations in MinexPy

- Required-column validation.
- Numeric coercion and removal of invalid rows.
- Optional duplicate-coordinate removal (keep-first).
- CRS transformation (built-in for `EPSG:4326 <-> EPSG:3857` or custom hook).
- Optional value transform:
  - none
  - base-10 logarithm
  - custom callable
- Metadata capture for optional inverse display transformation.

### Projection math used in built-in CRS conversion

For geographic coordinates (longitude \(\lambda\), latitude \(\phi\), in radians)
to Web Mercator with Earth radius \(R\):

$$
x = R \lambda
$$

$$
y = R \ln\left(\tan\left(\frac{\pi}{4} + \frac{\phi}{2}\right)\right)
$$

Inverse mapping:

$$
\lambda = \frac{x}{R}
$$

$$
\phi = 2\arctan\left(e^{y/R}\right) - \frac{\pi}{2}
$$

### Value-transform math

Optional log transform:

$$
z_i' = \log_{10}(z_i)
$$

If inversion is available (for log10):

$$
z_i = 10^{z_i'}
$$

This is important because interpolation may be computed on transformed values
while maps are often interpreted in original concentration units.

## Step 2: Grid creation (mesh base layer)

Interpolation routines estimate values at target nodes, so a regular grid must
be defined first.

### Why this step is necessary

- Produces deterministic target geometry.
- Controls map resolution through `cell_size`.
- Avoids clipping to sample convex support by adding optional padding.

### Extent and padding

Raw extent:

$$
x_{\min} = \min_i(x_i), \quad x_{\max} = \max_i(x_i)
$$

$$
y_{\min} = \min_i(y_i), \quad y_{\max} = \max_i(y_i)
$$

Ranges:

$$
\Delta x = x_{\max} - x_{\min}, \quad \Delta y = y_{\max} - y_{\min}
$$

Padding with ratio \(p \ge 0\):

$$
x_{\min}^{p} = x_{\min} - p\Delta x, \quad x_{\max}^{p} = x_{\max} + p\Delta x
$$

$$
y_{\min}^{p} = y_{\min} - p\Delta y, \quad y_{\max}^{p} = y_{\max} + p\Delta y
$$

### Regular axes and mesh

For grid spacing \(h\):

$$
\xi = \{x_{\min}^{p}, x_{\min}^{p}+h, \dots, x_{\max}^{p}\}
$$

$$
\eta = \{y_{\min}^{p}, y_{\min}^{p}+h, \dots, y_{\max}^{p}\}
$$

Mesh:

$$
X, Y = \operatorname{meshgrid}(\xi, \eta)
$$

Flattened node list for interpolation:

$$
G = \{(X_j, Y_j)\}_{j=1}^{M}
$$

where \(M = n_x n_y\).

## Step 3: Interpolation methods

Step 3 estimates surface values at each grid node.

## Common prerequisite

All methods operate on finite cleaned samples:

$$
\{(x_i, y_i, z_i)\}_{i=1}^{N}
$$

and return:

$$
Z \in \mathbb{R}^{n_y \times n_x}
$$

### 3.1 Nearest neighbor

Each grid node takes the value of the closest sample in Euclidean distance:

$$
i^*(g) = \arg\min_i \sqrt{(x_g-x_i)^2 + (y_g-y_i)^2}
$$

$$
\hat{z}(g) = z_{i^*(g)}
$$

Properties:

- exact at sample locations
- piecewise-constant surface
- no smoothing

### 3.2 Triangulation (`griddata` linear/cubic)

Samples are triangulated (Delaunay-based), then interpolation is done inside
each triangle.

For linear interpolation in one triangle with vertices \((v_1,v_2,v_3)\),
using barycentric coordinates \((w_1,w_2,w_3)\):

$$
w_1 + w_2 + w_3 = 1
$$

$$
\hat{z}(g) = w_1 z_1 + w_2 z_2 + w_3 z_3
$$

In MinexPy, nodes outside the convex hull are returned as `NaN` by design.

### 3.3 Inverse distance weighting (IDW)

For neighbors \(i \in \mathcal{N}(g)\) around node \(g\):

$$
w_i(g) = \frac{1}{(d_i(g) + \varepsilon)^p}
$$

$$
\hat{z}(g) = \frac{\sum_{i \in \mathcal{N}(g)} w_i(g) z_i}
{\sum_{i \in \mathcal{N}(g)} w_i(g)}
$$

where:

- \(d_i(g)\): distance from node \(g\) to sample \(i\)
- \(p\): power parameter (default 2)
- \(\varepsilon\): small stabilizer

Special case for exact coordinate match:

$$
d_i(g) \approx 0 \Rightarrow \hat{z}(g) = z_i
$$

or mean of coincident samples.

### 3.4 Minimum curvature (iterative biharmonic solver)

Minimum-curvature gridding seeks a smooth surface satisfying a biharmonic
condition away from fixed data constraints:

$$
\nabla^4 z = 0
$$

In MinexPy:

1. samples are anchored to nearest grid nodes (fixed nodes)
2. initial surface is nearest-neighbor
3. free nodes are iteratively updated using a discrete biharmonic stencil
4. fixed nodes are re-imposed each iteration

Interior-node update target (five-point-distance stencil form):

$$
z_{i,j}^{*} =
\frac{
8(z_{i+1,j}+z_{i-1,j}+z_{i,j+1}+z_{i,j-1})
-2(z_{i+1,j+1}+z_{i+1,j-1}+z_{i-1,j+1}+z_{i-1,j-1})
-(z_{i+2,j}+z_{i-2,j}+z_{i,j+2}+z_{i,j-2})
}{20}
$$

Relaxed iteration:

$$
z_{i,j}^{(k+1)} = z_{i,j}^{(k)} + \omega\left(z_{i,j}^{*}-z_{i,j}^{(k)}\right)
$$

Convergence check:

$$
\max_{i,j} \left| z_{i,j}^{(k+1)} - z_{i,j}^{(k)} \right| < \tau
$$

where:

- \(\omega\): relaxation factor
- \(\tau\): tolerance

This method yields smooth contours but can be sensitive to parameter choices
and grid resolution.

## Step 4: Final map composition

Step 4 assembles interpolation output into a publication-style map with context.

### Why this step is necessary

- Surface alone is insufficient for interpretation.
- Cartographic context is needed for communication and reproducibility.
- Metadata display helps prevent CRS/unit misinterpretation.

### Core cartographic elements

- title
- colorbar legend
- north arrow
- scale bar and optional numeric scale
- CRS/datum/units metadata block
- coordinate grid and neatline
- optional locator inset
- footer credits

### Scale ratio concept

Numeric scale \(1:n\) is derived from map span vs rendered axis width
(for metric units):

$$
n = \frac{\text{map span in ground units}}{\text{axis width in meters on page/screen}}
$$

### Display-value policy for transformed interpolation

If interpolation used transformed values and inversion metadata exists, MinexPy
can invert to display scale before rendering:

$$
\hat{z}_{display} = T^{-1}(\hat{z}_{work})
$$

For log10:

$$
\hat{z}_{display} = 10^{\hat{z}_{work}}
$$

## Practical interpretation notes

- **CRS first**: interpolation geometry only makes sense in a consistent CRS.
- **Grid size trade-off**: smaller cell size increases detail and cost.
- **Method choice**:
  - nearest: crisp domains, no smoothing
  - triangulation: piecewise surface, hull-limited support
  - IDW: local weighted smoothing, parameter-sensitive
  - minimum curvature: very smooth surfaces, iterative stability considerations
- **Transform awareness**: use log transforms for strongly right-skewed elements,
  then clearly label whether displayed values were inverted.

## References

1. Shepard, D. (1968). A two-dimensional interpolation function for irregularly-spaced data. *Proceedings of the 1968 ACM National Conference*, 517-524. DOI: [https://doi.org/10.1145/800186.810616](https://doi.org/10.1145/800186.810616)
2. Lee, D. T., & Schachter, B. J. (1980). Two algorithms for constructing a Delaunay triangulation. *International Journal of Computer & Information Sciences*, 9, 219-242. DOI: [https://doi.org/10.1007/BF00977785](https://doi.org/10.1007/BF00977785)
3. Briggs, I. C. (1974). Machine contouring using minimum curvature. *Geophysics*, 39(1), 39-48. DOI: [https://doi.org/10.1190/1.1440410](https://doi.org/10.1190/1.1440410)
4. Li, J., & Heap, A. D. (2014). Spatial interpolation methods applied in the environmental sciences: A review. *Environmental Modelling & Software*, 53, 173-189. DOI: [https://doi.org/10.1016/j.envsoft.2013.12.008](https://doi.org/10.1016/j.envsoft.2013.12.008)
5. Virtanen, P., et al. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. *Nature Methods*, 17, 261-272. DOI: [https://doi.org/10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2)
