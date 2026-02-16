"""
MinexPy - A toolkit for geoscience researchers.

This package provides tools for geochemical analysis, statistical processing,
and visualization for geological sample data.
"""

__version__ = "0.1.1"

# Import main modules for easy access
from . import stats
from . import correlation
from . import statviz
from . import mapping

# Export commonly used classes and functions
from .stats import (
    StatisticalAnalyzer,
    describe,
    skewness,
    kurtosis,
    std,
    variance,
    mean,
    median,
    coefficient_of_variation,
    iqr,
    percentile,
    z_score,
)
from .correlation import (
    pearson_correlation,
    spearman_correlation,
    kendall_correlation,
    distance_correlation,
    biweight_midcorrelation,
    partial_correlation,
    correlation_matrix,
)
from .statviz import (
    plot_histogram,
    plot_box_violin,
    plot_ecdf,
    plot_qq,
    plot_pp,
    plot_scatter,
)
from .mapping import (
    GeochemDataWarning,
    GeochemPrepareMetadata,
    GridDefinition,
    InterpolationResult,
    prepare,
    create_grid,
    interpolate,
    interpolate_nearest,
    interpolate_triangulation,
    interpolate_idw,
    interpolate_minimum_curvature,
    invert_values_for_display,
)

__all__ = [
    'stats',
    'correlation',
    'statviz',
    'mapping',
    'StatisticalAnalyzer',
    'describe',
    'skewness',
    'kurtosis',
    'std',
    'variance',
    'mean',
    'median',
    'coefficient_of_variation',
    'iqr',
    'percentile',
    'z_score',
    'pearson_correlation',
    'spearman_correlation',
    'kendall_correlation',
    'distance_correlation',
    'biweight_midcorrelation',
    'partial_correlation',
    'correlation_matrix',
    'plot_histogram',
    'plot_box_violin',
    'plot_ecdf',
    'plot_qq',
    'plot_pp',
    'plot_scatter',
    'GeochemDataWarning',
    'GeochemPrepareMetadata',
    'GridDefinition',
    'InterpolationResult',
    'prepare',
    'create_grid',
    'interpolate',
    'interpolate_nearest',
    'interpolate_triangulation',
    'interpolate_idw',
    'interpolate_minimum_curvature',
    'invert_values_for_display',
]
