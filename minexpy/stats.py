"""
Statistical analysis module for geoscience data.

This module provides comprehensive descriptive statistical metrics for analyzing
geochemical and geological sample data. It includes both functional
and class-based APIs for maximum flexibility.

Examples
--------
Basic usage with functions:

    >>> import numpy as np
    >>> import minexpy.stats as mstats
    >>> 
    >>> data = np.array([12.5, 15.3, 18.7, 22.1, 19.4, 16.8])
    >>> skew = mstats.skewness(data)
    >>> kurt = mstats.kurtosis(data)
    >>> summary = mstats.describe(data)

Class-based usage:

    >>> from minexpy.stats import StatisticalAnalyzer
    >>> import pandas as pd
    >>> 
    >>> df = pd.read_csv('geochemical_data.csv')
    >>> analyzer = StatisticalAnalyzer(df[['Zn', 'Cu', 'Pb']])
    >>> results = analyzer.summary()
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, List, Tuple
from scipy import stats as scipy_stats


def skewness(data: Union[np.ndarray, pd.Series, List[float]]) -> float:
    """
    Calculate the skewness of the data.
    
    Skewness measures the asymmetry of the data distribution around the mean.
    It indicates whether the data is skewed to the left or right.
    
    - Positive skewness: Right tail is longer; mass is concentrated on the left
    - Negative skewness: Left tail is longer; mass is concentrated on the right
    - Zero skewness: Data is symmetric (normal distribution has zero skewness)
    
    For geochemical data, skewness is particularly useful for identifying
    log-normal distributions, which are common in geochemical datasets.
    
    Parameters
    ----------
    data : array-like
        Input data array. Can be numpy array, pandas Series, or list.
        NaN values are automatically excluded.
        
    Returns
    -------
    float
        Skewness value. Typically ranges from -3 to +3, though extreme
        values can occur with small sample sizes.
        
    Examples
    --------
    >>> import numpy as np
    >>> from minexpy.stats import skewness
    >>> 
    >>> data = np.array([1.2, 3.4, 5.6, 7.8, 9.0, 11.2, 13.4])
    >>> skew = skewness(data)
    >>> print(f"Skewness: {skew:.3f}")
    
    Notes
    -----
    Uses Fisher's definition of skewness (third standardized moment).
    The calculation uses scipy.stats.skew with nan_policy='omit' to
    handle missing values.
    
    References
    ----------
    .. [1] Joanes, D. N., & Gill, C. A. (1998). Comparing measures of
           sample skewness and kurtosis. Journal of the Royal Statistical
           Society: Series D, 47(1), 183-189.
    """
    data = np.asarray(data)
    return float(scipy_stats.skew(data, nan_policy='omit'))


def kurtosis(data: Union[np.ndarray, pd.Series, List[float]], 
             fisher: bool = True) -> float:
    """
    Calculate the kurtosis of the data.
    
    Kurtosis measures the "tailedness" of the probability distribution.
    It indicates the presence of outliers and the shape of the distribution
    tails compared to a normal distribution.
    
    - High kurtosis (>0 with Fisher's definition): Heavy tails, more outliers
    - Low kurtosis (<0 with Fisher's definition): Light tails, fewer outliers
    - Normal distribution: Kurtosis = 0 (Fisher's) or 3 (Pearson's)
    
    In geochemical analysis, high kurtosis often indicates the presence of
    anomalous values or multiple populations in the dataset.
    
    Parameters
    ----------
    data : array-like
        Input data array. Can be numpy array, pandas Series, or list.
        NaN values are automatically excluded.
    fisher : bool, default True
        If True, uses Fisher's definition (excess kurtosis).
        Normal distribution has kurtosis = 0.
        If False, uses Pearson's definition.
        Normal distribution has kurtosis = 3.
        
    Returns
    -------
    float
        Kurtosis value. With Fisher's definition, typically ranges from
        -2 to +10, though extreme values are possible.
        
    Examples
    --------
    >>> import numpy as np
    >>> from minexpy.stats import kurtosis
    >>> 
    >>> data = np.array([1.2, 3.4, 5.6, 7.8, 9.0, 11.2, 13.4])
    >>> kurt = kurtosis(data)
    >>> print(f"Kurtosis (Fisher's): {kurt:.3f}")
    >>> 
    >>> kurt_pearson = kurtosis(data, fisher=False)
    >>> print(f"Kurtosis (Pearson's): {kurt_pearson:.3f}")
    
    Notes
    -----
    Fisher's definition (excess kurtosis) is more commonly used in
    modern statistics and is the default. It subtracts 3 from Pearson's
    definition so that a normal distribution has kurtosis = 0.
    
    References
    ----------
    .. [1] DeCarlo, L. T. (1997). On the meaning and use of kurtosis.
           Psychological methods, 2(3), 292.
    """
    data = np.asarray(data)
    return float(scipy_stats.kurtosis(data, fisher=fisher, nan_policy='omit'))


def std(data: Union[np.ndarray, pd.Series, List[float]], 
        ddof: int = 1) -> float:
    """
    Calculate the standard deviation of the data.
    
    Standard deviation measures the amount of variation or dispersion
    in the dataset. It is the square root of the variance.
    
    For geochemical data, standard deviation helps quantify the variability
    of element concentrations, which is crucial for understanding
    geochemical processes and identifying anomalies.
    
    Parameters
    ----------
    data : array-like
        Input data array. Can be numpy array, pandas Series, or list.
        NaN values are automatically excluded.
    ddof : int, default 1
        Delta degrees of freedom. The divisor used in calculations is
        N - ddof, where N is the number of observations.
        - ddof=1: Sample standard deviation (default, unbiased estimator)
        - ddof=0: Population standard deviation
        
    Returns
    -------
    float
        Standard deviation value. Same units as the input data.
        
    Examples
    --------
    >>> import numpy as np
    >>> from minexpy.stats import std
    >>> 
    >>> data = np.array([12.5, 15.3, 18.7, 22.1, 19.4, 16.8])
    >>> sample_std = std(data, ddof=1)
    >>> print(f"Sample std: {sample_std:.3f}")
    >>> 
    >>> pop_std = std(data, ddof=0)
    >>> print(f"Population std: {pop_std:.3f}")
    
    Notes
    -----
    For sample data (most common case), use ddof=1. For population data,
    use ddof=0. The default (ddof=1) provides an unbiased estimate of
    the population standard deviation from a sample.
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    return float(np.std(data, ddof=ddof, axis=None))


def variance(data: Union[np.ndarray, pd.Series, List[float]], 
             ddof: int = 1) -> float:
    """
    Calculate the variance of the data.
    
    Variance measures the average squared deviation from the mean.
    It quantifies the spread or dispersion of the data.
    
    Parameters
    ----------
    data : array-like
        Input data array. Can be numpy array, pandas Series, or list.
        NaN values are automatically excluded.
    ddof : int, default 1
        Delta degrees of freedom. The divisor used in calculations is
        N - ddof, where N is the number of observations.
        - ddof=1: Sample variance (default, unbiased estimator)
        - ddof=0: Population variance
        
    Returns
    -------
    float
        Variance value. Units are the square of the input data units.
        
    Examples
    --------
    >>> import numpy as np
    >>> from minexpy.stats import variance
    >>> 
    >>> data = np.array([12.5, 15.3, 18.7, 22.1, 19.4, 16.8])
    >>> var = variance(data)
    >>> print(f"Variance: {var:.3f}")
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    return float(np.var(data, ddof=ddof, axis=None))


def coefficient_of_variation(data: Union[np.ndarray, pd.Series, List[float]]) -> float:
    """
    Calculate the coefficient of variation (CV).
    
    The coefficient of variation is the ratio of the standard deviation
    to the mean, expressed as a percentage or decimal. It is a normalized
    measure of dispersion that allows comparison of variability across
    different scales and units.
    
    CV is particularly useful in geochemistry for:
    - Comparing variability of different elements with different
      concentration ranges
    - Identifying elements with high relative variability
    - Assessing data quality and homogeneity
    
    Parameters
    ----------
    data : array-like
        Input data array. Can be numpy array, pandas Series, or list.
        NaN values are automatically excluded.
        
    Returns
    -------
    float
        Coefficient of variation (std/mean). Returns NaN if mean is zero.
        Typically expressed as a decimal (e.g., 0.25 = 25%).
        
    Examples
    --------
    >>> import numpy as np
    >>> from minexpy.stats import coefficient_of_variation
    >>> 
    >>> data = np.array([12.5, 15.3, 18.7, 22.1, 19.4, 16.8])
    >>> cv = coefficient_of_variation(data)
    >>> print(f"CV: {cv:.3f} ({cv*100:.1f}%)")
    
    Notes
    -----
    CV is dimensionless and unitless, making it ideal for comparing
    variability across different elements or datasets with different
    measurement units. A CV < 0.15 is often considered low variability,
    while CV > 0.5 indicates high variability.
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    mean = np.mean(data)
    if mean == 0:
        return np.nan
    return float(np.std(data, ddof=1) / mean)


def mean(data: Union[np.ndarray, pd.Series, List[float]]) -> float:
    """
    Calculate the arithmetic mean of the data.
    
    The mean is the sum of all values divided by the number of values.
    It is the most common measure of central tendency.
    
    Parameters
    ----------
    data : array-like
        Input data array. Can be numpy array, pandas Series, or list.
        NaN values are automatically excluded.
        
    Returns
    -------
    float
        Arithmetic mean value.
        
    Examples
    --------
    >>> import numpy as np
    >>> from minexpy.stats import mean
    >>> 
    >>> data = np.array([12.5, 15.3, 18.7, 22.1, 19.4, 16.8])
    >>> avg = mean(data)
    >>> print(f"Mean: {avg:.3f}")
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    return float(np.mean(data))


def median(data: Union[np.ndarray, pd.Series, List[float]]) -> float:
    """
    Calculate the median of the data.
    
    The median is the middle value when data is sorted. It is a robust
    measure of central tendency that is less affected by outliers than
    the mean.
    
    For geochemical data, the median is often preferred when dealing
    with skewed distributions or datasets containing outliers.
    
    Parameters
    ----------
    data : array-like
        Input data array. Can be numpy array, pandas Series, or list.
        NaN values are automatically excluded.
        
    Returns
    -------
    float
        Median value.
        
    Examples
    --------
    >>> import numpy as np
    >>> from minexpy.stats import median
    >>> 
    >>> data = np.array([12.5, 15.3, 18.7, 22.1, 19.4, 16.8])
    >>> med = median(data)
    >>> print(f"Median: {med:.3f}")
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    return float(np.median(data))


def mode(data: Union[np.ndarray, pd.Series, List[float]]) -> Tuple[float, int]:
    """
    Calculate the mode (most frequent value) of the data.
    
    The mode is the value that appears most frequently in the dataset.
    For continuous data, this function returns the most common value
    after rounding to a reasonable precision.
    
    Parameters
    ----------
    data : array-like
        Input data array. Can be numpy array, pandas Series, or list.
        NaN values are automatically excluded.
        
    Returns
    -------
    tuple
        A tuple containing (mode_value, count). The mode_value is the
        most frequent value, and count is the number of times it appears.
        If multiple modes exist, returns the first one encountered.
        
    Examples
    --------
    >>> import numpy as np
    >>> from minexpy.stats import mode
    >>> 
    >>> data = np.array([1, 2, 2, 3, 3, 3, 4, 4])
    >>> mode_val, count = mode(data)
    >>> print(f"Mode: {mode_val} (appears {count} times)")
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    values, counts = np.unique(data, return_counts=True)
    max_count_idx = np.argmax(counts)
    return (float(values[max_count_idx]), int(counts[max_count_idx]))


def iqr(data: Union[np.ndarray, pd.Series, List[float]]) -> float:
    """
    Calculate the interquartile range (IQR).
    
    The IQR is the difference between the 75th percentile (Q3) and
    the 25th percentile (Q1). It is a robust measure of spread that
    is not affected by outliers.
    
    IQR is commonly used in geochemistry for:
    - Identifying outliers (values beyond Q1 - 1.5*IQR or Q3 + 1.5*IQR)
    - Describing data spread in skewed distributions
    - Box plot construction
    
    Parameters
    ----------
    data : array-like
        Input data array. Can be numpy array, pandas Series, or list.
        NaN values are automatically excluded.
        
    Returns
    -------
    float
        Interquartile range (Q3 - Q1).
        
    Examples
    --------
    >>> import numpy as np
    >>> from minexpy.stats import iqr
    >>> 
    >>> data = np.array([12.5, 15.3, 18.7, 22.1, 19.4, 16.8, 25.0])
    >>> interquartile_range = iqr(data)
    >>> print(f"IQR: {interquartile_range:.3f}")
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    return float(q3 - q1)


def percentile(data: Union[np.ndarray, pd.Series, List[float]], 
               p: float) -> float:
    """
    Calculate a specific percentile of the data.
    
    The percentile is the value below which a given percentage of
    observations fall. For example, the 90th percentile is the value
    below which 90% of the data points lie.
    
    Parameters
    ----------
    data : array-like
        Input data array. Can be numpy array, pandas Series, or list.
        NaN values are automatically excluded.
    p : float
        Percentile to calculate, between 0 and 100.
        
    Returns
    -------
    float
        The p-th percentile value.
        
    Examples
    --------
    >>> import numpy as np
    >>> from minexpy.stats import percentile
    >>> 
    >>> data = np.array([12.5, 15.3, 18.7, 22.1, 19.4, 16.8])
    >>> p90 = percentile(data, 90)
    >>> print(f"90th percentile: {p90:.3f}")
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    return float(np.percentile(data, p))


def z_score(data: Union[np.ndarray, pd.Series, List[float]], 
            value: Optional[float] = None) -> Union[float, np.ndarray]:
    """
    Calculate z-scores (standardized values).
    
    Z-scores measure how many standard deviations a value is from the mean.
    They are useful for:
    - Identifying outliers (typically |z| > 2 or 3)
    - Standardizing data for comparison
    - Detecting anomalies in geochemical data
    
    Parameters
    ----------
    data : array-like
        Input data array. Can be numpy array, pandas Series, or list.
        NaN values are automatically excluded.
    value : float, optional
        If provided, calculates the z-score for this specific value.
        If None, returns z-scores for all values in the data.
        
    Returns
    -------
    float or array
        If value is provided, returns the z-score for that value.
        Otherwise, returns an array of z-scores for all data points.
        
    Examples
    --------
    >>> import numpy as np
    >>> from minexpy.stats import z_score
    >>> 
    >>> data = np.array([12.5, 15.3, 18.7, 22.1, 19.4, 16.8])
    >>> z_scores = z_score(data)
    >>> print(f"Z-scores: {z_scores}")
    >>> 
    >>> z_specific = z_score(data, value=25.0)
    >>> print(f"Z-score for 25.0: {z_specific:.3f}")
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    mean_val = np.mean(data)
    std_val = np.std(data, ddof=1)
    
    if value is not None:
        return float((value - mean_val) / std_val)
    else:
        return (data - mean_val) / std_val


def describe(data: Union[np.ndarray, pd.Series, List[float]], 
             percentiles: Optional[List[float]] = None) -> Dict[str, float]:
    """
    Generate a comprehensive statistical summary of the data.
    
    This function calculates all major descriptive statistics in a single
    call, providing a complete overview of the data distribution. It is
    particularly useful for initial data exploration in geochemical analysis.
    
    Parameters
    ----------
    data : array-like
        Input data array. Can be numpy array, pandas Series, or list.
        NaN values are automatically excluded.
    percentiles : list of float, optional
        Additional percentiles to calculate beyond the default set.
        Default percentiles include: [25, 50, 75, 90, 95, 99].
        Values should be between 0 and 100.
        
    Returns
    -------
    dict
        Dictionary containing all statistical metrics with the following keys:
        - 'count': Number of observations
        - 'mean': Arithmetic mean
        - 'median': Median (50th percentile)
        - 'std': Standard deviation (sample)
        - 'variance': Variance (sample)
        - 'min': Minimum value
        - 'max': Maximum value
        - 'range': Range (max - min)
        - 'skewness': Skewness
        - 'kurtosis': Kurtosis (Fisher's definition)
        - 'coefficient_of_variation': CV (std/mean)
        - 'q1': First quartile (25th percentile)
        - 'q3': Third quartile (75th percentile)
        - 'iqr': Interquartile range
        - 'percentile_X': Additional percentiles as specified
        
    Examples
    --------
    >>> import numpy as np
    >>> from minexpy.stats import describe
    >>> 
    >>> data = np.array([12.5, 15.3, 18.7, 22.1, 19.4, 16.8, 25.0, 14.2])
    >>> summary = describe(data)
    >>> for key, value in summary.items():
    ...     print(f"{key}: {value:.3f}")
    >>> 
    >>> # With custom percentiles
    >>> summary_custom = describe(data, percentiles=[10, 50, 90, 95])
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    
    if len(data) == 0:
        raise ValueError("Input data is empty or contains only NaN values")
    
    if percentiles is None:
        percentiles = [25, 50, 75, 90, 95, 99]
    
    results = {
        'count': len(data),
        'mean': mean(data),
        'median': median(data),
        'std': std(data),
        'variance': variance(data),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'range': float(np.max(data) - np.min(data)),
        'skewness': skewness(data),
        'kurtosis': kurtosis(data),
        'coefficient_of_variation': coefficient_of_variation(data),
        'q1': percentile(data, 25),
        'q3': percentile(data, 75),
        'iqr': iqr(data),
    }
    
    # Add custom percentiles
    for p in percentiles:
        if 0 <= p <= 100:
            results[f'percentile_{p}'] = percentile(data, p)
    
    return results


class StatisticalAnalyzer:
    """
    Comprehensive statistical analyzer for geoscience data.
    
    This class provides a convenient, object-oriented interface for
    calculating multiple statistical metrics on geochemical data. It
    supports both single arrays and pandas DataFrames for batch analysis.
    
    The class is designed to be intuitive and flexible, allowing researchers
    to quickly obtain comprehensive statistical summaries of their data
    with minimal code.
    
    Attributes
    ----------
    data : numpy.ndarray or pandas.DataFrame
        The input data being analyzed.
    is_dataframe : bool
        True if input is a DataFrame, False if it's an array.
        
    Examples
    --------
    Single array analysis:
    
        >>> import numpy as np
        >>> from minexpy.stats import StatisticalAnalyzer
        >>> 
        >>> data = np.array([12.5, 15.3, 18.7, 22.1, 19.4, 16.8])
        >>> analyzer = StatisticalAnalyzer(data)
        >>> summary = analyzer.summary()
        >>> print(summary)
    
    DataFrame analysis (multiple columns):
    
        >>> import pandas as pd
        >>> from minexpy.stats import StatisticalAnalyzer
        >>> 
        >>> df = pd.read_csv('geochemical_data.csv')
        >>> analyzer = StatisticalAnalyzer(df[['Zn', 'Cu', 'Pb']])
        >>> summary_df = analyzer.summary()
        >>> print(summary_df)
    
    Individual metric access:
    
        >>> analyzer = StatisticalAnalyzer(data)
        >>> skew = analyzer.skewness()
        >>> kurt = analyzer.kurtosis()
        >>> cv = analyzer.coefficient_of_variation()
    """
    
    def __init__(self, data: Union[np.ndarray, pd.Series, pd.DataFrame, List[float]]):
        """
        Initialize the statistical analyzer.
        
        Parameters
        ----------
        data : array-like or DataFrame
            Input data to analyze. Can be:
            - 1D numpy array
            - pandas Series
            - pandas DataFrame (for multi-column analysis)
            - Python list
            
        Raises
        ------
        ValueError
            If input data is empty or contains only NaN values.
            
        Examples
        --------
        >>> import numpy as np
        >>> from minexpy.stats import StatisticalAnalyzer
        >>> 
        >>> # Array input
        >>> data = np.array([1, 2, 3, 4, 5])
        >>> analyzer = StatisticalAnalyzer(data)
        >>> 
        >>> # Series input
        >>> import pandas as pd
        >>> series = pd.Series([1, 2, 3, 4, 5])
        >>> analyzer = StatisticalAnalyzer(series)
        """
        if isinstance(data, pd.DataFrame):
            self.data = data
            self.is_dataframe = True
        elif isinstance(data, pd.Series):
            self.data = data.values
            self.is_dataframe = False
        else:
            self.data = np.asarray(data)
            self.is_dataframe = False
        
        # Validate data
        if self.is_dataframe:
            if self.data.empty:
                raise ValueError("Input DataFrame is empty")
        else:
            valid_data = self.data[~np.isnan(self.data)]
            if len(valid_data) == 0:
                raise ValueError("Input data is empty or contains only NaN values")
    
    def summary(self, as_dataframe: bool = True, 
                percentiles: Optional[List[float]] = None) -> Union[pd.DataFrame, Dict]:
        """
        Calculate comprehensive statistical summary.
        
        This method computes all major descriptive statistics for the data.
        For DataFrames, it calculates statistics for each column separately.
        
        Parameters
        ----------
        as_dataframe : bool, default True
            If True, returns results as a pandas DataFrame (for DataFrames)
            or Series (for arrays). If False, returns a dictionary.
        percentiles : list of float, optional
            Additional percentiles to include in the summary beyond defaults.
            
        Returns
        -------
        DataFrame, Series, or dict
            Statistical summary. For DataFrames, returns a DataFrame with
            columns as rows and statistics as columns. For arrays, returns
            a Series or dict with statistics as values.
            
        Examples
        --------
        >>> import numpy as np
        >>> from minexpy.stats import StatisticalAnalyzer
        >>> 
        >>> data = np.array([12.5, 15.3, 18.7, 22.1, 19.4, 16.8])
        >>> analyzer = StatisticalAnalyzer(data)
        >>> summary = analyzer.summary()
        >>> print(summary)
        """
        if self.is_dataframe:
            results = {}
            for col in self.data.columns:
                results[col] = describe(self.data[col].values, percentiles=percentiles)
            
            if as_dataframe:
                return pd.DataFrame(results).T
            return results
        else:
            results = describe(self.data, percentiles=percentiles)
            if as_dataframe:
                return pd.Series(results)
            return results
    
    def skewness(self) -> Union[float, pd.Series]:
        """
        Calculate skewness.
        
        Returns
        -------
        float or Series
            Skewness value(s). For DataFrames, returns a Series with
            skewness for each column.
        """
        if self.is_dataframe:
            return self.data.apply(lambda x: skewness(x))
        return skewness(self.data)
    
    def kurtosis(self, fisher: bool = True) -> Union[float, pd.Series]:
        """
        Calculate kurtosis.
        
        Parameters
        ----------
        fisher : bool, default True
            Use Fisher's definition (excess kurtosis) if True.
            
        Returns
        -------
        float or Series
            Kurtosis value(s). For DataFrames, returns a Series with
            kurtosis for each column.
        """
        if self.is_dataframe:
            return self.data.apply(lambda x: kurtosis(x, fisher=fisher))
        return kurtosis(self.data, fisher=fisher)
    
    def std(self, ddof: int = 1) -> Union[float, pd.Series]:
        """
        Calculate standard deviation.
        
        Parameters
        ----------
        ddof : int, default 1
            Delta degrees of freedom.
            
        Returns
        -------
        float or Series
            Standard deviation value(s). For DataFrames, returns a Series.
        """
        if self.is_dataframe:
            return self.data.std(ddof=ddof)
        return std(self.data, ddof=ddof)
    
    def variance(self, ddof: int = 1) -> Union[float, pd.Series]:
        """
        Calculate variance.
        
        Parameters
        ----------
        ddof : int, default 1
            Delta degrees of freedom.
            
        Returns
        -------
        float or Series
            Variance value(s). For DataFrames, returns a Series.
        """
        if self.is_dataframe:
            return self.data.var(ddof=ddof)
        return variance(self.data, ddof=ddof)
    
    def mean(self) -> Union[float, pd.Series]:
        """
        Calculate mean.
        
        Returns
        -------
        float or Series
            Mean value(s). For DataFrames, returns a Series.
        """
        if self.is_dataframe:
            return self.data.mean()
        return mean(self.data)
    
    def median(self) -> Union[float, pd.Series]:
        """
        Calculate median.
        
        Returns
        -------
        float or Series
            Median value(s). For DataFrames, returns a Series.
        """
        if self.is_dataframe:
            return self.data.median()
        return median(self.data)
    
    def coefficient_of_variation(self) -> Union[float, pd.Series]:
        """
        Calculate coefficient of variation.
        
        Returns
        -------
        float or Series
            CV value(s). For DataFrames, returns a Series.
            Returns NaN for columns/arrays with zero mean.
        """
        if self.is_dataframe:
            return self.data.apply(lambda x: coefficient_of_variation(x))
        return coefficient_of_variation(self.data)
    
    def iqr(self) -> Union[float, pd.Series]:
        """
        Calculate interquartile range.
        
        Returns
        -------
        float or Series
            IQR value(s). For DataFrames, returns a Series.
        """
        if self.is_dataframe:
            return self.data.apply(lambda x: iqr(x))
        return iqr(self.data)
    
    def z_score(self, value: Optional[float] = None) -> Union[float, np.ndarray, pd.DataFrame]:
        """
        Calculate z-scores.
        
        Parameters
        ----------
        value : float, optional
            If provided, calculates z-score for this specific value.
            Otherwise, returns z-scores for all data points.
            
        Returns
        -------
        float, array, or DataFrame
            Z-score(s). For DataFrames, returns a DataFrame with z-scores
            for each column.
        """
        if self.is_dataframe:
            if value is not None:
                return self.data.apply(lambda x: z_score(x, value=value))
            else:
                return self.data.apply(lambda x: z_score(x))
        return z_score(self.data, value=value)

