
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
        # Convert to numpy arrays for efficient computation
        X = np.array(X)
        Y = np.array(Y)
        
        # Calculate means of X and Y
        xbar = X.mean()
        ybar = Y.mean()

        # Calculate the numerator and denominator for the slope (b)
        numer = np.sum(X * Y) - len(X) * xbar * ybar
        denum = np.sum(X**2) - len(X) * xbar**2

        # Check if the denominator is zero (to avoid division by zero)
        if denum == 0:
            # Handle zero variance situation, maybe return None or raise an exception
            return None, None

        # Calculate slope (b) and intercept (a)
        b = numer / denum
        a = ybar - b * xbar

        return a, b

