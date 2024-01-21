
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from interpolation import Gridding
from line_fitting import LineFitting

class FractalModelling:
    def get_area(self, element_concentration, raw_data, num_slice, interpolation_method):
        """
        Calculate the area for each slice of concentration.

        Parameters
        ----------
        element_concentration : str
            The name of the element concentration column.
        raw_data : DataFrame
            The raw data containing the element concentrations.
        num_slice : int
            The number of slices to divide the concentration range into.
        interpolation_method : str
            The interpolation method to use.

        Returns
        -------
        list, list
            Returns two lists, one with the area of each slice and another with the mean concentration value of each slice.
        """
       # Instantiate Gridding class
        gridder = Gridding(raw_data, 'X', 'Y', element_concentration, interpolation_method)

        # Access the necessary attributes from the Gridding object
        grid1 = gridder.grid
        col_x = gridder.column_x
        col_y = gridder.column_y
        col_z = gridder.column_z

        # Calculate threshold and slices
        thr = round((max(col_z) - min(col_z)) / num_slice, 2)
        slices = [np.min(grid1)] + [np.min(grid1) + i * thr for i in range(1, num_slice)] + [np.max(grid1)]
        real_area = (max(col_x) - min(col_x)) * (max(col_y) - min(col_y))

        areas = []
        X_values = []

        for i, value in enumerate(slices[:-1]):
            helper = np.where((grid1 < slices[i+1]) & (grid1 >= value))
            sub_area = helper[0].shape[0]
            sub_area = round(sub_area * real_area / 1000000, 2)

            mean_concentration = np.mean(grid1[helper])
            if mean_concentration > 0.1:
                if not areas or sub_area > areas[-1]:
                    areas.append(sub_area)
                    X_values.append(mean_concentration)

        return areas[-20:], X_values[-20:]

    def ca(self, data, elements, output_folder):
        """
        Create concentration-area (C-A) diagrams for each element and save them as files.

        Parameters
        ----------
        data : DataFrame
            The data containing the element concentrations.
        elements : str
            Space-separated string of element names.
        output_folder : str
            The folder where the output diagrams will be saved.
        """
        # Initialize the lines dictionary
        lines = {}

        # Create the output folder on the desktop
        desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
        folder_name = "Concentration-Area Diagrams Generated with MinexPy"
        output_folder = os.path.join(desktop_path, folder_name)
        os.makedirs(output_folder, exist_ok=True)

        # Determine the number of rows and columns for the subplot grid
        num_elements = len(data.loc[:, elements.split()].columns)
        rows = int(np.ceil(num_elements / 3))
        cols = 3 if num_elements > 1 else 1

        # Create a grid of subplots
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
        if num_elements == 1:
            axes = [axes]  # Ensure axes is iterable for a single subplot

        for counter, column in enumerate(data.loc[:, elements.split()].columns):
            # Select the correct subplot
            ax = axes[counter // cols][counter % cols] if num_elements > 1 else axes[0]

            lines[column] = []
            scaler = MinMaxScaler(feature_range=(0.01, 0.99))
            data[column] = scaler.fit_transform(data[column].values.reshape(-1, 1)).reshape(data.shape[0],)
            areas, X_values = self.get_area(column, data, 24, 'linear')
            X = np.log10(X_values)
            Y = np.log10(areas)
            points = [np.array_split(X, 5), np.array_split(Y, 5)]

            line_fitting = LineFitting()

            for i in range(5):
                x = list(points[0][i])
                y = list(points[1][i])
                a, b = line_fitting.best_fit(x, y)
                lines[column] += [[a, b]]

                if a is not None and b is not None:
                    lines[column] += [[a, b]]

                    if i != 0:
                        x = list(points[0][i-1])[:] + x

                    try:
                        x = x + list(points[0][i+1])[:]
                    except:
                        pass

                    y = [a + b*xi for xi in x]
                    ax.plot(x, y, 'r')

                else:
                    # Skip drawing the line if best_fit returns None
                    pass

            ax.scatter(X, Y)
            ax.set_title(f"C-A diagram of {column}")
            ax.set_xlabel(f"Log transformed value of {column}")
            ax.set_ylabel(f"Log(Area) of {column}")
            ax.grid()

            plot_filename = f"{column}_CA_diagram.png"
            plt.savefig(os.path.join(output_folder, plot_filename))

            plt.close(fig)  # Close the figure after saving

        return lines

# Example usage
fm = FractalModelling()
data = pd.read_csv('~/MinexPy/Data.csv')
fm.ca(data, 'Zn Pb Ag Cu Mo Cr Ni Co Ba', 'output_folder')
