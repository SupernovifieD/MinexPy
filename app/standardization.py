import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class Standardizer:
    """
    A class for standardizing specified columns of a pandas DataFrame.

    ...

    Attributes
    ----------
    data : pd.DataFrame
        A pandas DataFrame with the data to be standardized.
    columns : str
        A string of column names to be standardized, separated by spaces.

    Methods
    -------
    standardize():
        Standardizes the specified columns of the DataFrame.

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [5, 4, 3, 2, 1], 'C': ['a', 'b', 'c', 'd', 'e']})
    >>> standardizer = Standardizer(df, 'A B')
    >>> standardizer.standardize()
    """
    def __init__(self, data, columns):
        """
        Constructs all the necessary attributes for the Standardizer object.

        Parameters
        ----------
        data : pd.DataFrame
            pandas DataFrame to be standardized.
        columns : str
            String of column names to be standardized, separated by spaces.

        Examples
        --------
        >>> df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [5, 4, 3, 2, 1], 'C': ['a', 'b', 'c', 'd', 'e']})
        >>> standardizer = Standardizer(df, 'A B')
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")
        if not isinstance(columns, str):
            raise ValueError("Columns input must be a string.")
        self.data = data
        self.columns = columns

    def standardize(self):
        """
        Standardizes the specified columns of the DataFrame using StandardScaler.

        This method parses the input string to get the specified column names and applies
        standardization to transform these columns to have a mean of zero and a
        standard deviation of one.

        Returns
        -------
        pd.DataFrame
            DataFrame with standardized specified columns.

        Examples
        --------
        >>> df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [5, 4, 3, 2, 1], 'C': ['a', 'b', 'c', 'd', 'e']})
        >>> standardizer = Standardizer(df, 'A B')
        >>> standardizer.standardize()
        """
        # Split the input string to get the list of column names
        cols_to_standardize = self.columns.split()

        # Check if the specified columns are in the DataFrame
        missing_cols = [col for col in cols_to_standardize if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

        # Check if the specified columns are numeric
        non_numeric_cols = [col for col in cols_to_standardize if not np.issubdtype(self.data[col].dtype, np.number)]
        if non_numeric_cols:
            raise ValueError(f"Non-numeric columns specified: {non_numeric_cols}")

        scaler = StandardScaler()
        self.data[cols_to_standardize] = scaler.fit_transform(self.data[cols_to_standardize])
        return self.data
