# Import the libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Define a class for standardizing values
class Standardizer:
    """
    This class standardizes the values of a DataFrame that contains coordinates and concentration values of different elements.
    It transforms each element column such that it has a mean of zero and a standard deviation of one.
    It returns a dictionary of standardized columns as an attribute of the object.

    Example:

    # Create a DataFrame with some sample data
    df = pd.DataFrame({
        "X": [1, 2, 3, 4, 5],
        "Y": [1, 2, 3, 4, 5],
        "Zn": [10, 20, 30, 40, 50],
        "Pb": [50, 40, 30, 20, 10]
    })

    # Create a Standardizer object with the DataFrame
    s = Standardizer(df)

    # Access the attribute of the object
    print(s.column_zs) # {"Zn": array([-1.41421356 -0.70710678 0.         0.70710678 1.41421356]), "Pb": array([1.41421356   0.70710678   -0.         -0.70710678 -1.41421356])}

    """

    # Initialize the object with the dataframe
    def __init__(self, cleaned_data):

        # Store the dataframe as an attribute
        self.cleaned_data = cleaned_data

        # Initialize an empty dictionary to store the standardized columns
        self.column_zs = {}

        # Standardize all element columns in the dataframe
        self.standardize_all()

    # Define a method to standardize the values of a column
    def standardize_column(self, column):
        scaler = StandardScaler()
        column_std = scaler.fit_transform(column.reshape(-1, 1)).reshape(column.shape[0], )
        return column_std

    # Define a method to standardize all element columns in the dataframe
    def standardize_all(self):

        # Loop through all the columns except the coordinates and standardize their values
        for column in self.cleaned_data.columns:
            if column not in ["X", "Y"]:
                self.column_zs[column] = self.standardize_column(self.cleaned_data.loc[:, f'{column}'].values)
