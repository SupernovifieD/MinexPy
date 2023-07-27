# Import the libraries
import numpy as np
from scipy.interpolate import griddata
from standardization import Standardizer # Import the Standardizer class from the standardization module

# Define a class for gridding
class Gridding:
    """
    This class performs gridding on a DataFrame that contains coordinates and concentration values of a single element.
    It creates a square grid of points, and interpolates the values on the grid using a specified method.
    It returns the interpolated grid and the original column as attributes of the object.
    
    Example:
    
    # Create a DataFrame with some sample data
    df = pd.DataFrame({
        "X": [1, 2, 3, 4, 5],
        "Y": [1, 2, 3, 4, 5],
        "Zn": [10, 20, 30, 40, 50]
    })
    
    # Create a Gridding object for the Zn column with linear interpolation
    g = Gridding(df, "X", "Y", "Zn", "linear")
    
    # Access the attributes of the object
    print(g.column_x) # array([1, 2, 3, 4, 5])
    print(g.column_y) # array([1, 2, 3, 4, 5])
    print(g.column_z) # array([10, 20, 30, 40, 50])
    print(g.grid_points.shape) # (1000, 1000, 2)
    print(g.grid.shape) # (1000, 1000)
    
    """
    
    # Initialize the object with the raw data and the parameters
    def __init__(self, raw_data, input_X, input_Y, element_concentration, input_interpol):
        
        # Extract the columns from the DataFrame
        self.column_x = raw_data.loc[:, f'{input_X}'].values
        self.column_y = raw_data.loc[:, f'{input_Y}'].values
        self.column_z = raw_data.loc[:, f'{element_concentration}'].values
        
        # Create a square grid of points
        self.grid_points = self.create_grid(self.column_x, self.column_y)
        
        # Interpolate the values of the concentration column on the grid using the specified method
        self.grid = self.interpolate_column(self.column_x,
                                            self.column_y,
                                            self.column_z,
                                            self.grid_points,
                                            input_interpol)

    # Define a method to create a square grid of points
    def create_grid(self, x, y, n_points=1000):
        x_axis = np.linspace(min(x), max(x), n_points)
        y_axis = np.linspace(min(y), max(y), n_points)
        grid_points = np.meshgrid(x_axis,
                                  y_axis)
        grid_points = np.stack(grid_points,
                               axis=-1)
        return grid_points

    # Define a method to interpolate the values of a column on a grid
    def interpolate_column(self,
                           x,
                           y,
                           z,
                           grid_points,
                           interpolation_method,
                           fill_value=0):
        transposed_mat = np.array([x,
                                   y]).T
        grid = griddata(transposed_mat,
                        z,
                        grid_points,
                        method=interpolation_method,
                        fill_value=fill_value)
        return grid

