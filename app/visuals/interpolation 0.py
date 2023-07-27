import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from sklearn.preprocessing import StandardScaler

# Define a class for gridding
class Gridding:
    """
    This class performs gridding on a DataFrame that contains coordinates and concentration values of different elements.
    It standardizes the values of each element column, creates a square grid of points, and interpolates the values on the grid using a specified method.
    It returns the interpolated grids and the original columns as attributes of the object.
    
    Example:
    
    # Create a DataFrame with some sample data
    df = pd.DataFrame({
        "X": [1, 2, 3, 4, 5],
        "Y": [1, 2, 3, 4, 5],
        "Zn": [10, 20, 30, 40, 50],
        "Pb": [50, 40, 30, 20, 10]
    })
    
    # Create a Gridding object with the DataFrame and the parameters
    g = Gridding(df, "X", "Y", "linear")
    
    # Access the attributes of the object
    print(g.column_x) # array([1, 2, 3, 4, 5])
    print(g.column_y) # array([1, 2, 3, 4, 5])
    print(g.column_z) # {"Zn": array([10, 20, 30, 40, 50]), "Pb": array([50, 40, 30, 20, 10])}
    print(g.column_zs) # {"Zn": array([-1.41421356 -0.70710678 0.         0.70710678 1.41421356]), "Pb": array([1.41421356   0.70710678   -0.         -0.70710678 -1.41421356])}
    print(g.grid_points.shape) # (1000, 1000, 2)
    print(g.grid["Zn"].shape) # (1000, 1000)
    print(g.grid["Pb"].shape) # (1000, 1000)
    print(g.grids["Zn"].shape) # (1000, 1000)
    print(g.grids["Pb"].shape) # (1000, 1000)
    
    """
    
    # Initialize the object with the raw data and the parameters
    def __init__(self, raw_data, input_X, input_Y, input_interpol):
        
        # Extract the columns from the DataFrame
        self.column_x = raw_data.loc[:, f'{input_X}'].values
        self.column_y = raw_data.loc[:, f'{input_Y}'].values
        
        # Create a square grid of points
        self.grid_points = self.create_grid(self.column_x, self.column_y)
        
        # Initialize an empty dictionary to store the concentration columns and their standardized values
        self.column_z = {}
        self.column_zs = {}
        
        # Initialize an empty dictionary to store the interpolated grids and their standardized values
        self.grid = {}
        self.grids = {}
        
        # Store the interpolation method
        self.interpolation_method = input_interpol
        
        # Loop through all the columns except the coordinates and perform gridding on them
        for column in raw_data.columns:
            self.perform_gridding(raw_data, column)
    
    # Define a method to standardize the values of a column
    def standardize_column(self, column):
        scaler = StandardScaler()
        column_std = scaler.fit_transform(column.reshape(-1, 1)).reshape(column.shape[0], )
        return column_std

    # Define a method to create a square grid of points
    def create_grid(self, x, y, n_points=1000):
        x_axis = np.linspace(min(x), max(x), n_points)
        y_axis = np.linspace(min(y), max(y), n_points)
        grid_points = np.meshgrid(x_axis, y_axis)
        grid_points = np.stack(grid_points, axis=-1)
        return grid_points

    # Define a method to interpolate the values of a column on a grid
    def interpolate_column(self, x, y, z, grid_points):
        transposed_mat = np.array([x,
                                   y]).T
        grid = griddata(transposed_mat,
                        z,
                        grid_points,
                        method=self.interpolation_method,
                        fill_value=0)
        return grid
    
    # Define a method to perform gridding on a single element column
    def perform_gridding(self, raw_data, element):
        
        # Extract the column from the DataFrame
        self.column_z[element] = raw_data.loc[:, f'{element}'].values
        
        # Standardize the values of the concentration column
        self.column_zs[element] = self.standardize_column(self.column_z[element])
        
        # Interpolate the values of the concentration column on the grid
        self.grid[element] = self.interpolate_column(self.column_x, self.column_y, self.column_z[element], self.grid_points)
        
        # Interpolate the standardized values of the concentration column on the grid
        self.grids[element] = self.interpolate_column(self.column_x, self.column_y, self.column_zs[element], self.grid_points)

    
