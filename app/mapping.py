# Import the libraries
import os
import datetime
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from matplotlib import pyplot as plt
from interpolation import Gridding # Import the Gridding class from the interpolation module

# Define a class for mapping
class Mapping:
    # Initialize the object with the data and the parameters
    def __init__(self,
                 input_elements,
                 data,
                 input_X,
                 input_Y,
                 input_interpol,
                 cmap,
                 title):

        # Store the data and the parameters as attributes
        self.input_elements = input_elements
        self.data = data
        self.input_X = input_X
        self.input_Y = input_Y
        self.input_interpol = input_interpol
        self.cmap = cmap
        self.title = title

        # Create a folder on the desktop to store the maps if it does not exist
        self.folder = os.path.join(os.path.expanduser("~"),
                                   "Desktop",
                                   "maps generated with MinexPy")
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    # Define a method to create a single map for a given element column
    def create_map(self,
                   element):

        # Create a Gridding object for the element column
        gridder = Gridding(self.data,
                           self.input_X,
                           self.input_Y,
                           element,
                           self.input_interpol)

        # Get the interpolated grid for the element column
        grid = gridder.grid

        # Create a figure and an axes for plotting
        fig, ax = plt.subplots(figsize=(12,
                                        9))

        # Plot the grid as an image with a colorbar
        plot = ax.imshow(grid,
                         origin='lower',
                         cmap=plt.get_cmap(self.cmap),
                         extent=[min(gridder.column_x),
                                 max(gridder.column_x),
                                 min(gridder.column_y),
                                 max(gridder.column_y)]) # Use the original coordinates as the extent of the image
        cbar = fig.colorbar(plot,
                            ax=ax)
        cbar.set_label(element,
                       labelpad=+2)

        # Set the title of the plot with a larger font size and a higher position
        ax.set_title(f"{self.title} {element}",
                     fontsize=24,
                     pad=20)

        # Add a north arrow to the plot outside the axes beside the y axis on the left hand side and near the top of the figure
        self.add_north_arrow(ax)

        # Format the y-axis ticks to show the coordinates without scientific notation
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))

        # Save the figure as a png file in the folder with the element name and the date and time
        filename = os.path.join(self.folder,
                                f"{element}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
        fig.savefig(filename)

        # Close the figure to free up memory
        plt.close(fig)

    # Define a method to add a north arrow to an axes
    def add_north_arrow(self, ax):

        # Define the coordinates and dimensions of the arrow in display space
        x_tail = 0.95 * ax.figure.bbox.width  # 95% of the figure width from the left
        y_tail = 0.05 * ax.figure.bbox.height  # 5% of the figure height from the bottom
        dx = 0  # No horizontal displacement
        dy = 0.05 * ax.figure.bbox.height  # 5% of the figure height vertical displacement

        # Draw an arrow with a text "N" above it using annotation
        ax.annotate("N", xy=(x_tail + dx/2, y_tail + dy + dy/10), xycoords="figure pixels",
                    ha="center", va="center", fontsize=20)
        ax.annotate("", xy=(x_tail + dx, y_tail + dy), xycoords="figure pixels",
                    xytext=(x_tail, y_tail), textcoords="figure pixels",
                    arrowprops=dict(arrowstyle="-|>", facecolor="black"))
        # Define a method to create maps for selected element columns using a for loop
    def create_maps(self):

        # Loop through the input elements and create a map for each one
        for element in self.input_elements.split():
            self.create_map(element)
