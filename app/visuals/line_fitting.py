
import numpy as np

class LineFitting:
    """
    This class provides methods for fitting lines to data points.
    
    Methods
    -------
    best_fit(X, Y)
        Computes the best fit line for the given X and Y values using the least squares method.
    """

    def best_fit(self, X, Y):
        """
        Computes the best fit line (y = ax + b) for the given X and Y values using the least squares method.

        Parameters
        ----------
        X : list or array-like
            The x-coordinates of the data points.
        Y : list or array-like
            The y-coordinates of the data points.

        Returns
        -------
        tuple
            A tuple (a, b) where 'a' is the intercept and 'b' is the slope of the best fit line.

        Examples
        --------
        >>> line_fitting = LineFitting()
        >>> X = [1, 2, 3, 4, 5]
        >>> Y = [2, 4, 6, 8, 10]
        >>> line_fitting.best_fit(X, Y)
        (0.0, 2.0)
        """

        X = np.array(X)
        Y = np.array(Y)
        
        n = len(X)
        xbar = X.mean()
        ybar = Y.mean()

        numer = np.sum(X * Y) - n * xbar * ybar
        denum = np.sum(X**2) - n * xbar**2
        b = numer / denum
        a = ybar - b * xbar

        return a, b
