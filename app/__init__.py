"""
MinexPy - A toolkit for geoscience researchers.

This package provides tools for geochemical analysis, statistical processing,
and visualization for geological sample data.
"""

__version__ = "0.1.0"

# Import main modules for easy access
from . import stats

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

__all__ = [
    'stats',
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
]

