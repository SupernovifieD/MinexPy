import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from scipy.interpolate import griddata
from scipy.stats import describe, probplot, stats
from scipy import interpolate
from math import *


class MinexPy:
    """
    A class for preprocessing and analyzing geological sample data.

    Attributes:
        data_path (str): The file path for the data CSV file.
        input_elements (str): A string of element abbreviations to analyze.
        input_X (str): The column name in the data representing the X-coordinate.
        input_Y (str): The column name in the data representing the Y-coordinate.
        input_interpol (str): The interpolation method (e.g., 'linear', 'cubic', 'nearest').

    Methods:
        preprocess(location): Reads and cleans data from a given location.
        basic_stats(element_concentration): Calculates basic statistics for a given element.
        TODO: add all methods here
    """

    def __init__(self, data_path, input_elements, input_X, input_Y, input_interpol):
        """
        Initializes the MinexPy object with data path, elements, coordinates, and interpolation method.

        Parameters:
            data_path (str): The path to the data file.
            input_elements (str): Space-separated string of element abbreviations to include in the analysis.
            input_X (str): The column name for X-coordinate values.
            input_Y (str): The column name for Y-coordinate values.
            input_interpol (str): The interpolation method to use ('linear', 'cubic', 'nearest').
        """
        self.data_path = data_path
        self.input_elements = input_elements
        self.input_X = input_X
        self.input_Y = input_Y
        self.input_interpol = input_interpol

    def preprocess(self, location):
        """
        Preprocesses the data by reading from a location, handling missing values, and removing duplicates.

        Parameters:
            location (str): The file path for the data CSV file.

        Returns:
            DataFrame: A pandas DataFrame with cleaned data, starting from column 'X' onwards.
        """
        data = pd.read_csv(location)
        data = data.replace("", np.nan)
        data = data.dropna()
        data = data.drop_duplicates().loc[:, "X":]

        return data

    def basic_stats(self, element_concentration):
        """
        Computes basic statistics for a specified element concentration in the dataset.

        Parameters:
            element_concentration (str): The column name for the element concentration to analyze.

        Returns:
            list: A list containing number of observations, min value, max value, mean, variance, skewness, and kurtosis.
        """

        raw_data = self.preprocess(self.data_path)
        selected_values = describe(raw_data.loc[:, f"{element_concentration}"].values)

        s1 = str(selected_values.nobs)
        s2 = str(selected_values.minmax[0])
        s3 = str(selected_values.minmax[1])
        s4 = str(round(selected_values.mean, 2))
        s5 = str(round(selected_values.variance, 2))
        s6 = str(round(selected_values.skewness, 2))
        s7 = str(round(selected_values.kurtosis, 2))

        return [s1, s2, s3, s4, s5, s6, s7]

    def basic_stats_df(self, element_concentration):
        """
        Generates a DataFrame with basic statistics for specified element concentrations.

        This method preprocesses the raw data from the initialized data path and calculates
        statistical measures including the number of observations, minimum and maximum values,
        mean, variance, skewness, and kurtosis for each element specified in the
        `element_concentration` parameter.

        Parameters:
            element_concentration (str): A space-separated string of element abbreviations
                                        for which to calculate statistics. Each abbreviation
                                        should correspond to a column name in the dataset.

        Returns:
            DataFrame: A pandas DataFrame indexed by element abbreviations with columns for
                    each of the statistical measures.

        Example:
            # Assuming `test` is an instance of MinexPy initialized with a data path and relevant parameters.
            element_stats_df = test.basic_stats_df("Zn Pb Ag")
            print(element_stats_df)

            This will output a DataFrame with the basic statistics for the Zinc (Zn),
            Lead (Pb), and Silver (Ag) concentrations from the dataset.
        """
        raw_data = self.preprocess(self.data_path)
        columns = [
            "No. of Observations",
            "Min",
            "Max",
            "Mean",
            "Variance",
            "Skewness",
            "Kurtosis",
        ]
        index_names = element_concentration.split()
        df = pd.DataFrame(columns=columns)

        data = []
        selected_columns = raw_data.loc[:, index_names].columns
        for counter, column in enumerate(selected_columns):
            values = self.basic_stats(element_concentration=column)
            zipped = zip(columns, values)
            data.append(dict(zipped))

        # df = df.append(data, True)
        df = pd.DataFrame(data, columns=columns, index=index_names)

        index = pd.Index(index_names)
        df = df.set_index(index)

        return df

    def correlation(self, first_element, second_element):
        """
        Calculates the Pearson and Spearman correlation coefficients between two specified elements.

        This method first preprocesses the data to ensure it's clean and then computes both the
        Pearson and Spearman correlation coefficients for the concentrations of two specified elements.
        The Pearson coefficient measures the linear correlation, while the Spearman coefficient assesses
        the monotonic relationship.

        Parameters:
            first_element (str): The column name for the first element's concentration.
            second_element (str): The column name for the second element's concentration.

        Returns:
            tuple: A tuple containing the Pearson correlation coefficient followed by the Spearman
                correlation coefficient, both rounded to three decimal places.

        Example:
            # Assuming `test` is an instance of MinexPy initialized with a data path and relevant parameters.
            pearson_corr, spearman_corr = test.correlation("Zn", "Cu")
            print(f"Pearson Correlation: {pearson_corr}, Spearman Correlation: {spearman_corr}")

            This will output the Pearson and Spearman correlation coefficients for the Zinc (Zn) and
            Copper (Cu) concentrations from the dataset, providing insight into their linear and
            monotonic relationships, respectively.
        """
        raw_data = self.preprocess(self.data_path)

        # Extract the specified elements' data
        first_elem_data, second_elem_data = (
            raw_data.loc[:, first_element],
            raw_data.loc[:, second_element],
        )

        # Calculate Spearman and Pearson correlations
        spearcorr = stats.spearmanr(first_elem_data, second_elem_data)
        pearcorr = stats.pearsonr(first_elem_data, second_elem_data)

        # Round the correlation coefficients to 3 decimal places
        spearcorr_coefficient = round(spearcorr[0], 3)
        pearcorr_coefficient = round(pearcorr[0], 3)

        return pearcorr_coefficient, spearcorr_coefficient

    def correlation_iterated(self, element_concentration):
        """
        Calculates and prints the Pearson and Spearman correlation matrices for specified elements.

        This method iterates over each pair of specified elements, computes both Pearson and Spearman
        correlation coefficients using the `correlation` method, and organizes the results into two
        DataFrames. One DataFrame contains the Pearson correlation coefficients, and the other contains
        the Spearman correlation coefficients, providing a comprehensive view of both linear and
        monotonic relationships among the elements.

        Parameters:
            element_concentration (str): A space-separated string of element abbreviations
                                        for which to calculate correlations. Each abbreviation
                                        should correspond to a column name in the dataset.

        Returns:
            None. This method prints the correlation matrices directly.

        Example:
            # Assuming `test` is an instance of MinexPy initialized with a data path and relevant parameters.
            test.correlation_iterated("Zn Pb Ag Cu")

            This will print the Pearson and Spearman correlation matrices for Zinc (Zn), Lead (Pb),
            Silver (Ag), and Copper (Cu) concentrations, offering insights into their interrelationships.
        """
        raw_data = self.preprocess(self.data_path)

        columns = element_concentration.split()

        data1 = {}  # For Pearson correlations
        data2 = {}  # For Spearman correlations
        selected_columns = raw_data.loc[:, columns].columns
        for i in selected_columns:
            temp1 = []  # Temporary list to hold Pearson correlations for element i
            temp2 = []  # Temporary list to hold Spearman correlations for element i
            for j in selected_columns:
                pearson_corr, spearman_corr = self.correlation(first_element=i, second_element=j)
                temp1.append(pearson_corr)
                temp2.append(spearman_corr)

            data1[i] = temp1
            data2[i] = temp2

        df1 = pd.DataFrame(data1, index=columns)
        df2 = pd.DataFrame(data2, index=columns)

        print("Here is the Pearson correlation coefficient of the elements:\n", df1, "\n")
        print("Here is the Spearman correlation coefficient of the elements:\n", df2)

    def statistical_plots(self, desired_column):
        """
        Generates a series of statistical plots for a specified column in the dataset.

        This method preprocesses the dataset to ensure it's clean and then generates four types
        of plots in a 2x2 subplot arrangement: a histogram, a boxplot, a Q-Q (quantile-quantile) plot,
        and a P-P (probability-probability) plot. These plots are useful for analyzing the distribution,
        variability, and normality of the data for the selected column.

        Parameters:
            desired_column (str): The column name for which to generate statistical plots.

        Returns:
            None. This method directly displays the plots.

        Example:
            # Assuming `test` is an instance of MinexPy initialized with a data path and relevant parameters.
            test.statistical_plots("Zn")

            This will display the histogram, boxplot, Q-Q plot, and P-P plot for the Zinc (Zn)
            concentration values from the dataset, providing a visual summary of its statistical properties.
        """
        raw_data = self.preprocess(self.data_path)
        values = raw_data.loc[:, desired_column].values
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=[8, 8])

        # Histogram
        axes[0, 0].hist(values, bins=50, color="c", edgecolor="k", alpha=0.65)
        axes[0, 0].axvline(values.mean(), color="k", linestyle="--", linewidth=1)
        min_ylim, max_ylim = axes[0, 0].set_ylim()
        axes[0, 0].text(values.mean() * 1.4, max_ylim * 0.9, f"Mean: {values.mean():.2f}")
        axes[0, 0].set(title=f"Histogram chart for {desired_column}")

        # Boxplot
        axes[0, 1].boxplot(values, vert=False, patch_artist=False, sym="", showmeans=True)
        axes[0, 1].set(title=f"Box plot for {desired_column}")
        axes[0, 1].grid(axis="x")

        # Q-Q plot
        probplot(values, plot=axes[1, 0])
        axes[1, 0].set(title=f"Q-Q plot of {desired_column}")
        axes[1, 0].grid(True)

        # P-P plot
        # Custom P-P plot code as described
        norm = np.random.normal(0, 1, len(values))
        norm = norm - (min(norm) - min(values))
        min_value = min(norm)
        scale = (max(values) - min_value) / (max(norm) - min_value)
        norm = np.apply_along_axis(
            lambda x: min_value + (x - min_value) * scale, axis=0, arr=norm
        )
        norm.sort()
        values.sort()
        bins = np.percentile(norm, np.linspace(0, 100, 1000))
        data_hist, _ = np.histogram(values, bins=bins)
        cumsum_data = np.cumsum(data_hist)
        cumsum_data = np.array(cumsum_data) / max(cumsum_data)
        norm_hist, _ = np.histogram(norm, bins=bins)
        cumsum_norm = np.cumsum(norm_hist)
        cumsum_norm = np.array(cumsum_norm) / max(cumsum_norm)

        plot4 = axes[1, 1].plot(cumsum_norm, cumsum_data, "o")
        plot4 = axes[1, 1].plot([0, 1], [0, 1], color="r")
        plot4 = axes[1, 1].set(title=f"P-P plot of {desired_column}")
        plot4 = axes[1, 1].grid(True)

        plt.suptitle(f"Basic Statistical Plots of {desired_column}")
        fig.tight_layout()
        plt.show()

    def show_statistical_plots(self, input_elements):
        """
        Displays statistical plots for a list of specified elements in the dataset.

        This method iterates over the given elements, preprocessing the dataset for each and then
        utilizing the `statistical_plots` method to generate and display a series of statistical plots
        (histogram, boxplot, Q-Q plot, and P-P plot) for each element. It's an efficient way to visually
        analyze the distribution, central tendency, variability, and normality of multiple data columns.

        Parameters:
            input_elements (str): A space-separated string of element abbreviations for which to generate and display statistical plots. Each abbreviation should correspond to a column name in the dataset.

        Returns:
            None. This method directly displays the plots for each specified element.

        Example:
            # Assuming `test` is an instance of MinexPy initialized with a data path and relevant parameters.
            test.show_statistical_plots("Zn Pb Ag Cu")

            This will iterate over the elements Zinc (Zn), Lead (Pb), Silver (Ag), and Copper (Cu),
            displaying the set of statistical plots for each element's concentration values from the dataset.
        """
        raw_data = self.preprocess(self.data_path)
        elements = input_elements
        for column in raw_data.loc[:, elements.split()].columns:
            self.statistical_plots(column)

    def gridding(
        self,
        raw_data,
        x_coordinates,
        y_coordinates,
        element_concentration,
        interpolation_method,
        feature_range=(0.001, 0.999),
        fill_value=0.0,
    ):
        """
        Performs spatial interpolation on the specified element concentration data to create a gridded representation.

        This method applies a MinMax scaling to the concentration values and then uses the specified
        interpolation method to generate a continuous spatial grid. The grid can represent either the
        original or scaled concentration values over a defined two-dimensional space determined by the
        x and y coordinates.

        Parameters:
            raw_data (DataFrame): The pandas DataFrame containing the dataset.
            x_coordinates (str): The column name in `raw_data` representing the x-coordinates.
            y_coordinates (str): The column name in `raw_data` representing the y-coordinates.
            element_concentration (str): The column name in `raw_data` for the element concentration values.
            interpolation_method (str): The method of interpolation (e.g., 'linear', 'nearest', 'cubic').
            feature_range (tuple, optional): The range (min, max) for scaling the concentration values. Defaults to (0.001, 0.999).
            fill_value (float, optional): The value used to fill in for missing data in the interpolation. Defaults to 0.0.

        Returns:
            tuple: A tuple containing the grid of original concentration values, the grid of scaled concentration values, 
                arrays of original x and y coordinates, and arrays of original and scaled concentration values.

        Example:
            # Assuming `test` is an instance of MinexPy initialized with a data path and relevant parameters,
            # and `raw_data` is a pandas DataFrame loaded with the necessary columns.
            grid, scaled_grid, x_coords, y_coords, z_values, scaled_z_values = test.gridding(
                raw_data=raw_data,
                x_coordinates="X",
                y_coordinates="Y",
                element_concentration="Zn",
                interpolation_method="linear"
            )

            This will return the grids (original and scaled) for the Zinc (Zn) concentration
            across the specified spatial domain, using linear interpolation.
        """
        column_x = raw_data.loc[:, x_coordinates].values
        column_y = raw_data.loc[:, y_coordinates].values
        column_z = raw_data.loc[:, element_concentration].values
        scaler = MinMaxScaler(feature_range=feature_range)
        column_zs = scaler.fit_transform(column_z.reshape(-1, 1)).reshape(-1)

        x_axis = np.linspace(min(column_x), max(column_x), 1000)
        y_axis = np.linspace(min(column_y), max(column_y), 1000)

        grid_points = np.meshgrid(x_axis, y_axis)
        grid_points = np.stack(grid_points, axis=-1)

        transposed_mat = np.array([column_x, column_y]).T
        grid = griddata(
            transposed_mat,
            column_z,
            grid_points,
            method=interpolation_method,
            fill_value=fill_value,
        )
        grids = griddata(
            transposed_mat,
            column_zs,
            grid_points,
            method=interpolation_method,
            fill_value=fill_value,
        )

        return grid, grids, column_x, column_y, column_z, column_zs    

    def mapping(self, data, element_concentration, cmap, title):
        """
        Creates a visual map of the spatial distribution for specified element concentrations.

        Utilizes the `gridding` method to interpolate element concentration data onto a regular grid
        and then visualizes these grids for each specified element. The plots are arranged in a 3x3
        subplot matrix, allowing for up to 9 element concentrations to be visualized simultaneously.

        Parameters:
            data (DataFrame): The pandas DataFrame containing the dataset.
            element_concentration (str): A space-separated string of element abbreviations for which to generate maps. Each abbreviation should correspond to a column name in the dataset.
            cmap (str): The colormap name to use for the visualizations.
            title (str): The base title for the plots, which will be appended with the element abbreviation.

        Returns:
            None. This method directly displays the visual maps.

        Example:
            # Assuming `test` is an instance of MinexPy initialized with a data path and relevant parameters,
            # and `data` is a pandas DataFrame loaded with the necessary columns.
            test.mapping(
                data=data,
                element_concentration="Zn Pb Ag",
                cmap="viridis",
                title="Spatial distribution of "
            )

            This will create and display visual maps for the spatial distribution of Zinc (Zn),
            Lead (Pb), and Silver (Ag) concentrations, using the 'viridis' colormap.
        """
        fig, ax = plt.subplots(3, 3, figsize=(36, 28))

        for counter, column in enumerate(data.loc[:, element_concentration.split()].columns):
            x_index, y_index = divmod(counter, 3)
            grid = self.gridding(data, self.input_X, self.input_Y, column, self.input_interpol)[0]
            plot = ax[x_index, y_index].imshow(grid, origin="lower", cmap=plt.get_cmap(cmap))
            cbar = fig.colorbar(plot, ax=ax[x_index, y_index])
            cbar.set_label("", labelpad=+2)
            ax[x_index, y_index].set_title(f"{title} {column}")

        # plt.show()

    def pca(self, element_concentration):
        """
        Performs Principal Component Analysis (PCA) on specified element concentrations.

        This method standardizes the selected element concentration data before applying PCA to
        identify the principal components. It returns the PCA loadings, the transformed dataset
        with components as features, and a matrix of loadings for visualization or further analysis.

        Parameters:
            element_concentration (str): A space-separated string of element abbreviations
                                        for which PCA is to be performed. Each abbreviation
                                        should correspond to a column name in the dataset.

        Returns:
            tuple: A tuple containing:
                - A DataFrame of PCA loadings for each element.
                - A transformed DataFrame with the original data adjusted by the PCA loadings.
                - A DataFrame of PCA loading matrices for visualization or further analysis.

        Example:
            # Assuming `test` is an instance of MinexPy initialized with a data path and relevant parameters.
            loadings, transformed_df, loading_matrix = test.pca("Zn Pb Ag Cu")

            This will perform PCA on the concentrations of Zinc (Zn), Lead (Pb), Silver (Ag),
            and Copper (Cu), returning the PCA loadings, the dataset transformed by these loadings,
            and the PCA loading matrices.
        """
        raw_data = self.preprocess(self.data_path)
        splitted = element_concentration.split()
        ele = raw_data.loc[:, splitted].values
        ele = StandardScaler().fit_transform(ele)

        pca = PCA(n_components=len(splitted))
        principalComponents = pca.fit_transform(ele)

        # Creating PCA loading labels
        column_names_PCA = [f"PCA{i}" for i in range(1, len(splitted) + 1)]
        loadings1 = pd.DataFrame(
            np.round(pca.components_.T, 3), columns=column_names_PCA, index=splitted
        )

        # Loadings for PCA matrix
        loadings2 = pca.components_.T * np.sqrt(pca.explained_variance_)
        column_names_PCA_matrix = [f"PCAM{i}" for i in range(1, len(splitted) + 1)]
        loadings_matrix = pd.DataFrame(
            np.round(loadings2, 3), columns=column_names_PCA_matrix, index=splitted
        )

        # Adjusting the original data based on PCA loadings
        for i, element in enumerate(splitted):
            raw_data[element] = raw_data[element] * max(abs(loadings1.loc[element, :]))

        processed_df = pd.DataFrame(raw_data)

        return loadings1, processed_df, loadings_matrix

    def best_fit(self, X, Y):
        """
        Calculates the slope and intercept of the best fit line for given X and Y values.

        This method computes the parameters of a linear regression line (y = a + bx) that best fits the data. 
        It uses the least squares method to calculate the slope (b) and intercept (a) of the line.

        Parameters:
            X (list or numpy array): The independent variable values.
            Y (list or numpy array): The dependent variable values.

        Returns:
            tuple: A tuple containing the slope (b) and intercept (a) of the best fit line.

        Example:
            # Assuming `test` is an instance of MinexPy.
            X = [1, 2, 3, 4, 5]
            Y = [2, 3, 5, 7, 11]
            slope, intercept = test.best_fit(X, Y)
            print(f"Slope: {slope}, Intercept: {intercept}")

            This will calculate and output the slope and intercept of the best fit line for the provided X and Y data.
        """
        n = len(X)
        xbar = sum(X) / n

        m = len(Y)
        ybar = sum(Y) / m

        # Ensure that both X and Y have the same number of data points
        if n != m:
            raise ValueError("X and Y must have the same number of data points.")

        numer = sum([xi * yi for xi, yi in zip(X, Y)]) - n * xbar * ybar
        denum = sum([xi**2 for xi in X]) - n * xbar**2
        b = numer / denum
        a = ybar - b * xbar

        return b, a

    def get_area(self, element_concentration, raw_data, num_slice, interpolation_method):
        """
        Calculates the areas within defined concentration slices for a specified element.

        This method uses gridded interpolation of element concentrations to divide the study
        area into slices based on concentration thresholds. It calculates the area of each slice,
        allowing for an analysis of how the element's concentration is spatially distributed
        across the entire study area.

        Parameters:
            element_concentration (str): The column name for the element's concentration values.
            raw_data (DataFrame): The pandas DataFrame containing the dataset.
            num_slice (int): The number of slices to divide the concentration range into.
            interpolation_method (str): The method of interpolation to use for gridding the data.

        Returns:
            tuple: A tuple containing two lists:
                - The first list contains the areas of each slice (in square kilometers, assuming
                    input coordinates are in meters).
                - The second list contains the average concentration value for each slice.

        Example:
            # Assuming `test` is an instance of MinexPy initialized with a data path and relevant parameters.
            areas, avg_concentrations = test.get_area(
                element_concentration="Zn",
                raw_data=raw_data,
                num_slice=10,
                interpolation_method="linear"
            )

            This will calculate the areas and average Zinc (Zn) concentrations for 10 concentration
            slices across the study area, using linear interpolation.
        """
        # Perform gridding based on the specified parameters
        grid, _, col_x, col_y, _, col_z = self.gridding(
            raw_data,
            self.input_X,
            self.input_Y,
            element_concentration,
            interpolation_method=self.input_interpol,
            fill_value=0.1,
        )

        # Determine thresholds for slices
        thr = round((max(col_z) - min(col_z)) / num_slice, 2)
        slices = [np.min(grid)] + [np.min(grid) + i * thr for i in range(1, num_slice)] + [np.max(grid)]
        real_area = (max(col_x) - min(col_x)) * (max(col_y) - min(col_y))

        areas, X_values = [], []
        for i, value in enumerate(slices[:-1]):
            helper = np.where((grid >= value) & (grid < slices[i + 1]))
            sub_area = helper[0].shape[0] * real_area / (1000 * 1000)  # Convert from m^2 to km^2
            sub_area = round(sub_area, 2)
            avg_concentration = np.mean(grid[helper])

            if avg_concentration > 0.1:  # Filter based on avg. concentration
                X_values.append(avg_concentration)
                areas.append(sub_area)

        # Handle potential edge cases or errors
        filtered_areas, filtered_X_values = [], []
        for i in range(len(areas)):
            if X_values[i] > 0.1:
                filtered_areas.append(areas[i])
                filtered_X_values.append(X_values[i])

        return filtered_areas[-20:], filtered_X_values[-20:]  # Adjust as needed based on specific requirements

    def ca(self, data, elements):
        """
        Generates concentration-area (C-A) diagrams for specified elements.

        This method scales the concentration data for each element, divides the scaled concentrations
        into slices, calculates the area for each slice, and plots the log-transformed concentration
        values against the log-transformed area values. It then fits a line to segments of these points
        to analyze the concentration-area relationship, commonly used in geochemical exploration.

        Parameters:
            data (DataFrame): The pandas DataFrame containing the dataset with element concentrations.
            elements (str): A space-separated string of element abbreviations to generate C-A diagrams for.

        Returns:
            dict: A dictionary where keys are element names and values are lists of line parameters (slope and intercept) fitted to segments of the C-A relationship.

        Example:
            # Assuming `test` is an instance of MinexPy initialized with a data path and relevant parameters.
            lines = test.ca(data, "Zn Cu Pb")
            print(lines)

            This will generate C-A diagrams for Zinc (Zn), Copper (Cu), and Lead (Pb), and return the fitted line parameters for each.
        """
        fig, ax = plt.subplots(3, 3, figsize=(36, 28))  # Adjust subplot size as needed

        points = {}
        lines = {}

        for counter, column in enumerate(data.loc[:, elements.split()].columns):

            lines[column] = []
            scaler = MinMaxScaler(feature_range=(0.01, 0.99))
            data[column] = scaler.fit_transform(data[column].values.reshape(-1, 1)).flatten()
            areas, X_values = self.get_area(column, data, 24, "linear")
            X, Y = [], []

            for i, area in enumerate(areas):
                X.append(np.log10(X_values[i]))
                Y.append(np.log10(area))

            points[column] = [np.array_split(np.array(X), 5), np.array_split(np.array(Y), 5)]

            for i in range(5):
                x = list(points[column][0][i])
                y = list(points[column][1][i])
                a, b = self.best_fit(x, y)
                lines[column].append([a, b])

                # Extend the x range for drawing lines
                if i != 0:
                    x = list(points[column][0][i - 1])[:] + x
                if i < 4:  # Ensure i+1 doesn't exceed bounds
                    x = x + list(points[column][0][i + 1])[:]

                y = [a + b * xi for xi in x]
                ax[int(counter / 3), counter % 3].plot(x, y, "r")
            ax[int(counter / 3), counter % 3].scatter(X, Y)
            ax[int(counter / 3), counter % 3].set_title(f"C-A diagram of {column}")
            ax[int(counter / 3), counter % 3].set_xlabel(f"Log transformed value of {column}")
            ax[int(counter / 3), counter % 3].set_ylabel("Log(Area)")
            ax[int(counter / 3), counter % 3].grid()

        plt.tight_layout()
        plt.show()

        return lines

    def known_minerals(self, element_concentration, raw_data, interpolation_method):
        """
        Maps known mineral occurrences onto a spatial grid based on UTM coordinates.

        This method reads a file containing UTM coordinates of known mineral occurrences and
        calculates their positions on the spatial grid generated for the specified element
        concentration. This allows for the analysis of the spatial correlation between known
        mineral deposits and the element concentration data.

        Parameters:
            element_concentration (str): The column name for the element's concentration values.
            raw_data (DataFrame): The pandas DataFrame containing the dataset with element concentrations.
            interpolation_method (str): The method of interpolation to use for gridding the data.
                                        This should match the method used in the `gridding` process.

        Returns:
            tuple: Two lists containing the x and y grid coordinates of known mineral occurrences,
                respectively.

        Example:
            # Assuming `test` is an instance of MinexPy initialized with a data path and relevant parameters,
            # and `raw_data` is a pandas DataFrame loaded with the necessary columns.
            known_mineral_x, known_mineral_y = test.known_minerals(
                element_concentration="Zn",
                raw_data=raw_data,
                interpolation_method="linear"
            )
            print(known_mineral_x, known_mineral_y)

            This will return the grid coordinates of known mineral occurrences based on the
            Zn concentration grid and the specified linear interpolation method.
        """
        # Load the file containing known mineral occurrences
        # Replace the placeholder path with the actual file path
        file_path = "Known_mineral_deposits-970729.xls"
        known_minerals_data = pd.read_excel(file_path)[["XUTM", "YUTM"]]

        # Perform gridding based on the specified parameters
        _, _, col_x, col_y, _, _ = self.gridding(
            raw_data,
            self.input_X,
            self.input_Y,
            element_concentration,
            interpolation_method=self.input_interpol,
            fill_value=0.1,
        )

        # Calculate grid coordinates for known mineral occurrences
        known_mineral_x, known_mineral_y = [], []
        for index, row in known_minerals_data.iterrows():
            grid_x = int((row["XUTM"] - min(col_x)) * 1000 / (max(col_x) - min(col_x)))
            grid_y = int((row["YUTM"] - min(col_y)) * 1000 / (max(col_y) - min(col_y)))
            known_mineral_x.append(grid_x)
            known_mineral_y.append(grid_y)

        return known_mineral_x, known_mineral_y

    def discrete_mapping(self, element_concentration, raw_data, interpolation_method, slices, ax, counter, fig):
        """
        Creates a discretized spatial map for a specified element concentration, overlaying known mineral occurrences.

        This method discretizes the spatial distribution of the specified element's concentration into defined slices,
        applying a unique color to each slice. It then overlays markers for known mineral occurrences to highlight
        their locations relative to the concentration distribution.

        Parameters:
            element_concentration (str): The column name for the element's concentration values.
            raw_data (DataFrame): The pandas DataFrame containing the dataset with element concentrations.
            interpolation_method (str): The method of interpolation to use for gridding the data.
            slices (list): A list of concentration thresholds defining the slices for discretization.
            ax (matplotlib.axes.Axes): The axes object on which to draw the map.
            counter (int): The index of the subplot within a figure, used to position the axes.
            fig (matplotlib.figure.Figure): The figure object that contains the subplot.

        Example:
            # Assuming `test` is an instance of MinexPy initialized with a data path and relevant parameters,
            # and `fig, ax = plt.subplots(3, 3, figsize=(36, 28))` has been defined.
            slices = [0, 0.2, 0.4, 0.6, 0.8, 1]
            test.discrete_mapping(
                "Zn",
                raw_data,
                "linear",
                slices,
                ax,
                0,  # Position of the subplot
                fig
            )

            This will create a discretized map of Zinc (Zn) concentrations, with known mineral occurrences marked,
            and add it to the first subplot of the figure.
        """
        grid = self.gridding(
            raw_data,
            self.input_X,
            self.input_Y,
            element_concentration,
            interpolation_method=interpolation_method
        )[0]

        # Define the colormap and normalization based on the given slices
        cmap = mpl.colors.ListedColormap(['green', 'cyan', 'yellow', 'orange', 'blue', 'linen', 'saddlebrown', 'salmon', 'darkkhaki'][:len(slices) - 1])
        norm = mpl.colors.BoundaryNorm(slices, cmap.N, clip=True)

        # Retrieve known mineral locations
        known_mineral_x, known_mineral_y = self.known_minerals(element_concentration, raw_data, interpolation_method)

        # Plot the discretized map and known mineral occurrences
        ax[int(counter / 3), counter % 3].imshow(grid, origin='lower', cmap=cmap, norm=norm)
        ax[int(counter / 3), counter % 3].scatter(known_mineral_x, known_mineral_y, c='black', marker='^', label='Known Minerals')

        # Add a colorbar to the plot
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), extendfrac='auto', ticks=slices, spacing='uniform', ax=ax[int(counter / 3), counter % 3])
        cbar.set_label('Concentration Scale')

        # Set the title and adjust the layout
        ax[int(counter / 3), counter % 3].set_title(f"Discretized map of {element_concentration}")
        ax[int(counter / 3), counter % 3].legend()

        plt.tight_layout()

    def discrete_mapping_iterated(self, raw_data, element_concentration, lines):
        """
        Automatically generates discretized spatial maps for specified elements based on line segments.

        This method iterates over specified elements, uses the intersections of provided line segments
        to determine concentration slices for discretization, and creates spatial maps to visualize
        the distribution of each element's concentration across the study area.

        Parameters:
            raw_data (DataFrame): The pandas DataFrame containing the dataset with element concentrations.
            element_concentration (str): A space-separated string of element abbreviations for which to generate maps.
            lines (dict): A dictionary where keys are element names and values are lists of line parameters (slope, intercept)
                        defining segments for concentration slicing.

        Example:
            # Assuming `test` is an instance of MinexPy initialized with relevant parameters.
            lines = {
                "Zn": [[slope1, intercept1], [slope2, intercept2]],
                "Cu": [[slope1, intercept1], [slope2, intercept2]],
                # Add more elements and their line segments as needed.
            }
            test.discrete_mapping_iterated(raw_data, "Zn Cu Pb", lines)

            This will generate discretized maps for Zinc (Zn), Copper (Cu), and Lead (Pb),
            using the specified line segments to define concentration slices.
        """
        fig, ax = plt.subplots(3, 3, figsize=(36, 28))  # Adjust subplot layout as needed

        for counter, column in enumerate(raw_data.loc[:, element_concentration.split()].columns):

            # Update raw_data[column] with scaled concentration values for better visualization
            # Assuming the 5th returned value from `gridding` method is the scaled concentration
            raw_data[column] = self.gridding(
                raw_data,
                self.input_X,
                self.input_Y,
                column,
                self.input_interpol,
                feature_range=(0.1, 0.99),
            )[4]

            # Calculate slices based on intersections of lines
            slices = [0.001]
            for i, line in enumerate(lines[column]):
                if i < len(lines[column]) - 1:
                    a = np.array([[-1 * line[1], 1], [-1 * lines[column][i + 1][1], 1]])
                    b = np.array([line[0], lines[column][i + 1][0]])
                    x = np.linalg.solve(a, b)
                    intersection = 10 ** x[0]
                    if 0.01 <= intersection <= 0.99:
                        slices.append(intersection)

            slices.append(0.999)
            self.discrete_mapping(column, raw_data, self.input_interpol, slices, ax, counter, fig)

        plt.tight_layout()
        plt.show()

    def area_percentage(self, element_concentration, raw_data, interpolation_method, slices):
        """
        Calculates the area and known mineral occurrences percentages within specified concentration slices.

        This method divides the spatial distribution of a given element's concentration into predefined slices
        and calculates the percentage of the total study area and the percentage of known mineral occurrences
        within each slice. It's useful for assessing the spatial correlation between element concentrations and
        mineral distributions.

        Parameters:
            element_concentration (str): The column name for the element's concentration values.
            raw_data (DataFrame): The pandas DataFrame containing the dataset with element concentrations.
            interpolation_method (str): The method of interpolation to use for gridding the data.
            slices (list): A list of concentration thresholds defining the slices for analysis.

        Returns:
            tuple: A tuple containing three lists:
                - The first list contains the percentages of the total area within each concentration slice.
                - The second list contains the percentages of known mineral occurrences within each slice.
                - The third list is the input slices list for reference.

        Example:
            # Assuming `test` is an instance of MinexPy initialized with a data path and relevant parameters.
            slices = [0, 0.2, 0.4, 0.6, 0.8, 1]
            area_percents, mineral_percents, slices = test.area_percentage(
                "Zn",
                raw_data,
                "linear",
                slices
            )
            print(area_percents, mineral_percents, slices)

            This will output the percentages of the total area and known mineral occurrences for Zinc (Zn)
            within the defined concentration slices.
        """
        grid, _, col_x, col_y, col_z, _ = self.gridding(
            raw_data,
            self.input_X,
            self.input_Y,
            element_concentration,
            interpolation_method=self.input_interpol,
            fill_value=0.1,
        )

        known_mineral_x, known_mineral_y = self.known_minerals(
            element_concentration,
            raw_data,
            interpolation_method,
        )

        total_area = (max(col_x) - min(col_x)) * (max(col_y) - min(col_y))
        known_mineral_percent = []
        area_percent = []

        for i in range(len(slices) - 1):
            slice_filter = (grid >= slices[i]) & (grid < slices[i + 1])
            area = np.sum(slice_filter) * total_area / grid.size  # Convert to percentage of the total area
            area_percent.append(round(area * 100 / total_area, 2))

            # Filter known minerals within the current slice
            minerals_in_slice = slice_filter[known_mineral_x, known_mineral_y]
            known_mineral_percent.append(round(np.sum(minerals_in_slice) * 100 / len(known_mineral_x), 2))

        return area_percent, known_mineral_percent, slices

    def find_nearest(self, array, value):
        """
        Finds the nearest value in an array to a given target value.

        This method calculates the absolute difference between each element in the array and the target value,
        returning the element with the smallest difference, i.e., the nearest value to the target.

        Parameters:
            array (numpy.ndarray or list): The array of values to search through. If a list is provided, it will
                                        be converted to a numpy array.
            value (float or int): The target value to find the nearest element for in the array.

        Returns:
            The value from the array that is nearest to the specified target value.

        Example:
            array = [0, 2, 4, 6, 8, 10]
            target_value = 5
            nearest_value = self.find_nearest(array, target_value)
            print(f"The nearest value to {target_value} is {nearest_value}")

            This will output: The nearest value to 5 is 4
        """
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx] 

    def pa_diagram(self, raw_data, element_concentration, lines):
        """
        Generates Prediction-Area (P-A) diagrams for specified elements based on their concentration data.

        This method scales the concentration data, calculates concentration slices based on intersections
        of provided line segments, and uses linear interpolation to estimate the distribution of known mineral
        occurrences and the study area percentage within each slice. It visualizes these distributions and
        calculates prediction rates, occupied area percentages, and normalized densities for each element.

        Parameters:
            raw_data (DataFrame): The pandas DataFrame containing the dataset with element concentrations.
            element_concentration (str): A space-separated string of element abbreviations for which to generate P-A diagrams.
            lines (dict): A dictionary where keys are element names and values are lists of line parameters (slope, intercept)
                        defining segments for concentration slicing.

        Returns:
            DataFrame: A pandas DataFrame containing the prediction rate, occupied area percentage, normalized density,
                    and weight for each specified element based on the generated P-A diagrams.

        Example:
            # Assuming `test` is an instance of MinexPy initialized with a data path and relevant parameters,
            # and lines have been defined for each element.
            df_results = test.pa_diagram(raw_data, "Zn Cu Pb", lines)
            print(df_results)

            This will output a DataFrame with the calculated metrics for Zinc (Zn), Copper (Cu), and Lead (Pb)
            based on their P-A diagrams.
        """
        fig, ax = plt.subplots(3, 3, figsize=(36, 28))
        intersections = {}

        # Iterate through each specified element
        for counter, column in enumerate(raw_data.loc[:, element_concentration.split()].columns):
            # Scale the element concentrations
            scaler = MinMaxScaler(feature_range=(0.01, 0.99))
            raw_data[column] = scaler.fit_transform(raw_data[column].values.reshape(-1, 1)).flatten()

            # Determine concentration slices from line intersections
            slices = [0.1]
            for i, line in enumerate(lines[column]):
                if i < len(lines[column]) - 1:
                    intersection = self._calculate_line_intersection(line, lines[column][i + 1])
                    if intersection and 0.2 <= intersection <= 0.99:
                        slices.append(intersection)
            slices.append(0.99)

            # Calculate area percentages and known mineral occurrence percentages
            percents, known_mineral_percent, _ = self.area_percentage(column, raw_data, self.input_interpol, slices)

            # Interpolate percentages for more detailed plotting
            xnew, known_mineral_percent, percents = self._interpolate_percentages(slices, percents, known_mineral_percent)

            # Plotting
            self._plot_pa_diagram(ax, counter, xnew, known_mineral_percent, percents, column)

            # Calculating intersections (if needed for further analysis)

        # Constructing the results DataFrame
        df = self._construct_pa_results(intersections)

        return df

    def index_overlay(self, raw_data, element_concentration, weights):
        """
        Generates an index overlay from specified element concentrations, applying weights to each element.

        This method creates a composite index by weighting individual element concentrations according to
        specified weights. This index can serve as a tool for highlighting areas of interest within a study
        area based on the combined geochemical or geophysical characteristics.

        Parameters:
            raw_data (DataFrame): The pandas DataFrame containing the dataset with element concentrations.
            element_concentration (str): A space-separated string of element abbreviations to be included in the index.
            weights (DataFrame): A pandas DataFrame containing the weights for each element. This DataFrame
                                should have an 'Index' column with element names and a 'Weight' column with
                                corresponding weights.

        Returns:
            DataFrame: A pandas DataFrame with original X and Y coordinates and a new 'Index Overlay' column
                    representing the weighted sum of the specified elements' concentrations.

        Example:
            # Assuming `test` is an instance of MinexPy initialized with relevant parameters,
            # `raw_data` contains the geochemical data, and `weights` contains weights derived from a PA diagram.
            result_df = test.index_overlay(raw_data, "Zn Cu Pb", weights)
            print(result_df.head())

            This will print the first few rows of the resulting DataFrame, including the 'Index Overlay' column.
        """
        # Split element_concentration into a list of elements
        elements = element_concentration.split()

        # Ensure weights DataFrame has correct column names for merging
        weights.set_index('Index', inplace=True)

        # Create a new DataFrame for calculations
        new_df = raw_data[elements].copy()

        # Apply weights to each element's concentration
        for element in elements:
            if element in weights.index:
                new_df[element] *= weights.loc[element, 'Weight']

        # Calculate the index overlay
        final_df = pd.DataFrame(new_df.sum(axis=1) / weights['Weight'].sum(), columns=['Index Overlay'])

        # Combine index overlay with original coordinates
        result = pd.concat([raw_data[[self.input_X, self.input_Y]], final_df], axis=1)

        return result

    def io_mapping(self, data):
        """
        Visualizes the 'Index Overlay' scores on a spatial grid as a heatmap.

        This method uses the 'Index Overlay' scores calculated for a study area to produce a heatmap
        visualization. These scores represent the weighted combination of specified element concentrations,
        providing insights into areas of potential interest based on geochemical or geophysical characteristics.

        Parameters:
            data (DataFrame): The pandas DataFrame containing the 'Index Overlay' scores along with X and Y coordinates.

        Returns:
            matplotlib.axes.Axes: The Axes object with the heatmap of the 'Index Overlay' scores.

        Example:
            # Assuming `test` is an instance of MinexPy initialized with relevant parameters,
            # and `data` contains the 'Index Overlay' scores and spatial coordinates.
            ax = test.io_mapping(data)
            plt.show()

            This will display a heatmap of the 'Index Overlay' scores across the study area, using a 'seismic' colormap
            to differentiate between high and low scores.
        """
        # Ensure 'Index Overlay' is available in the data for gridding
        if 'Index Overlay' not in data.columns:
            raise ValueError("Data must include an 'Index Overlay' column for mapping.")

        # Interpolate the 'Index Overlay' scores onto a spatial grid
        io_grid, _, _, _, _, _ = self.gridding(
            data,
            self.input_X,
            self.input_Y,
            "Index Overlay",
            interpolation_method=self.input_interpol,
            fill_value=0.1,
        )

        # Create the heatmap visualization
        fig, ax = plt.subplots()
        plot = ax.imshow(io_grid, origin='lower', cmap=plt.get_cmap("seismic"), interpolation='nearest')
        cbar = fig.colorbar(plot, ax=ax)
        cbar.set_label('Index Overlay Scores')

        ax.set_title("Map of Objective Multiclass Index Overlay Scores")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")

        # Adjust the figure size if necessary
        fig.set_size_inches(8, 8)

        return ax

    def get_area_modified(self, raw_data, num_slice, interpolation_method):
        """
        Calculates the area for specified slices based on 'Index Overlay' values in a gridded dataset.

        This method divides the 'Index Overlay' values into slices according to the specified number
        of slices and calculates the area that each slice occupies within the study area. It's useful
        for understanding the spatial distribution of combined geochemical or geophysical characteristics.

        Parameters:
            raw_data (DataFrame): The pandas DataFrame containing the dataset with 'Index Overlay' values
                                and spatial coordinates (X, Y).
            num_slice (int): The number of slices to divide the 'Index Overlay' values into.
            interpolation_method (str): The method of interpolation to use for gridding the data.

        Returns:
            tuple: A tuple containing two lists:
                - The first list contains the areas of each slice (in square kilometers, assuming
                    input coordinates are in meters).
                - The second list contains the average 'Index Overlay' value for each slice.

        Example:
            # Assuming `test` is an instance of MinexPy initialized with a data path and relevant parameters,
            # `raw_data` contains the geochemical data with 'Index Overlay' values.
            areas, avg_values = test.get_area_modified(raw_data, 10, "linear")
            print("Areas (sq km):", areas)
            print("Average 'Index Overlay' values:", avg_values)

            This will output the areas occupied by each of the 10 slices and their average 'Index Overlay' values.
        """
        # Grid the 'Index Overlay' values
        grid, _, col_x, col_y, _, col_z = self.gridding(
            raw_data,
            self.input_X,
            self.input_Y,
            "Index Overlay",
            interpolation_method=self.input_interpol,
            fill_value=0.1,
        )

        # Calculate threshold for slices and define slices
        thr = round((max(col_z) - min(col_z)) / num_slice, 2)
        slices = [np.min(grid1)] + [np.min(grid1) + i * thr for i in range(1, num_slice)] + [np.max(grid1)]
        real_area = (max(col_x) - min(col_x)) * (max(col_y) - min(col_y)) / 1e6  # Convert from m^2 to km^2

        areas, X_values = [], []
        for i, value in enumerate(slices[:-1]):
            helper = np.where((grid >= value) & (grid < slices[i + 1]))
            sub_area = helper[0].size * real_area / grid.size  # Normalize by grid size to get area in km^2
            areas.append(round(sub_area, 2))
            X_values.append(np.mean(grid[helper]) if helper[0].size > 0 else 0)

        # Optionally, filter or process the areas and X_values as needed

        return areas, X_values

    def ca_modified(self, data, elements):
        """
        Generates a Concentration-Area (C-A) diagram for 'Index Overlay' scores, applying linear fitting to log-transformed data points.

        This method scales the 'Index Overlay' scores within the specified data, divides them into slices,
        and fits linear segments to the log-transformed values of area and 'Index Overlay' scores. The
        resulting C-A diagram visualizes the relationship between area and concentration, highlighting
        potential patterns or anomalies.

        Parameters:
            data (DataFrame): The pandas DataFrame containing the dataset with 'Index Overlay' scores.
            elements (str): Unused parameter in this version of the method. Intended for future use or customization.

        Returns:
            list: A list of line parameters (slope and intercept) for each fitted segment.

        Example:
            # Assuming `test` is an instance of MinexPy initialized with relevant parameters,
            # and `data` contains the 'Index Overlay' scores.
            lines = test.ca_modified(data, "Index Overlay")
            plt.show()

            This will display the C-A diagram for the 'Index Overlay' scores and return the line parameters for each segment.
        """
        # Initialize lists for storing points and lines
        points, lines = [], []

        # Scale the 'Index Overlay' values
        scaler = MinMaxScaler(feature_range=(0.01, 0.99))
        data["Index Overlay"] = scaler.fit_transform(data["Index Overlay"].values.reshape(-1, 1)).flatten()

        # Calculate areas and corresponding 'Index Overlay' values
        areas, X_values = self.get_area_modified(data, 24, "linear")
        X, Y = [], []

        # Log-transform area and 'Index Overlay' values
        for i, area in enumerate(areas):
            X.append(np.log10(X_values[i]))
            Y.append(np.log10(area))

        # Split data for fitting
        points = [np.array_split(np.array(X), 5), np.array_split(np.array(Y), 5)]

        # Fit lines to the log-transformed data
        for i in range(5):
            x = list(points[0][i])
            y = list(points[1][i])
            a, b = self.best_fit(x, y)
            lines.append([a, b])

            # Extend the fitting range by including adjacent points
            if i != 0:
                x = list(points[0][i - 1])[:] + x
            if i < 4:  # Ensure i+1 doesn't exceed bounds
                x = x + list(points[0][i + 1])[:]

            # Plot the fitted line segments
            y_fit = [a + b * xi for xi in x]
            plt.plot(x, y_fit, "r")
        
        # Plot the original log-transformed data points
        plt.scatter(X, Y)
        plt.title("C-A diagram of 'Index Overlay'")
        plt.xlabel("Log transformed value of 'Index Overlay'")
        plt.ylabel("Log(Area) of 'Index Overlay'")
        plt.grid()

        return lines

    def discrete_mapping_modified_new(self, data, lines):
        """
        Creates a discretized spatial map for 'Index Overlay' scores, based on specified line segments,
        and plots known mineral occurrences.

        This method generates a map that visualizes different ranges of 'Index Overlay' scores as discrete
        color-coded regions. The boundaries of these regions are determined by solving for the intersections
        of provided line segments, representing changes in geochemical or geophysical characteristics.

        Parameters:
            data (DataFrame): The pandas DataFrame containing the dataset with 'Index Overlay' scores and spatial coordinates (X, Y).
            lines (list of lists): Each inner list contains parameters (slope, intercept) of a line segment used to define slice boundaries.

        Example:
            # Assuming `test` is an instance of MinexPy initialized with relevant parameters,
            # and `data` contains the 'Index Overlay' scores.
            lines = [[slope1, intercept1], [slope2, intercept2], ...]  # Define line segments
            test.discrete_mapping_modified_new(data, lines)

            This will display a discretized map of 'Index Overlay' scores, with known mineral occurrences marked.
        """
        # Grid the 'Index Overlay' values
        grid, _, _, _, _, _ = self.gridding(data, "X", "Y", "Index Overlay", self.input_interpol)[0]

        # Calculate slice boundaries from line intersections
        slices = [0.001]
        for i, line in enumerate(lines):
            if i < len(lines) - 1:
                a = np.array([[-1 * line[1], 1], [-1 * lines[i + 1][1], 1]])
                b = np.array([line[0], lines[i + 1][0]])
                x = np.linalg.solve(a, b)
                intersection = 10 ** x[0]
                if 0.1 <= intersection <= 1.1:
                    slices.append(intersection)
        slices.append(0.999)

        # Define the colormap and normalization
        cmap = mpl.colors.ListedColormap(['green', 'cyan', 'yellow', 'orange', 'blue', 'linen', 'saddlebrown', 'salmon', 'darkkhaki'][:len(slices) - 1])
        norm = mpl.colors.BoundaryNorm(slices, cmap.N, clip=True)

        # Fetch known mineral locations for plotting
        known_mineral_x, known_mineral_y = self.known_minerals("Index Overlay", data, self.input_interpol)

        # Create the discretized map and plot known mineral occurrences
        plt.figure(figsize=(8, 8))
        plt.imshow(grid, origin='lower', cmap=cmap, norm=norm)
        plt.scatter(known_mineral_x, known_mineral_y, c='black', marker='^', label='Known Minerals')
        cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), extendfrac='auto', ticks=slices, spacing='uniform')
        cbar.set_label('Index Overlay Score')

        plt.title("Discretized Map of 'Index Overlay'")
        plt.legend()
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.show()

    def pa_diagram_modified(self, raw_data, element_concentration, lines):
        """
        Generates a Prediction-Area (P-A) diagram for 'Index Overlay' scores, highlighting the distribution
        of known mineral occurrences and the study area percentages within specified concentration slices.

        Parameters:
            raw_data (DataFrame): The DataFrame containing the 'Index Overlay' scores and spatial coordinates.
            element_concentration (str): Unused in this method; intended for future customization.
            lines (list of lists): Line segments defined by slope and intercept for calculating slice thresholds.

        Returns:
            DataFrame: A DataFrame summarizing the prediction rate, occupied area percentage, normalized density,
                    and weight for the 'Index Overlay' analysis.

        Example:
            # Assuming `test` is an instance of MinexPy initialized with relevant parameters,
            # and `lines` are defined for 'Index Overlay'.
            df_results = test.pa_diagram_modified(raw_data, "Index Overlay", lines)
            print(df_results)

            This will output a DataFrame with metrics derived from the P-A diagram analysis.
        """
        # Calculate slice thresholds from line intersections
        slices = [0.001]
        for i, line in enumerate(lines):
            if i < len(lines) - 1:
                intersection = self._calculate_line_intersection(line, lines[i + 1])
                if intersection and 0.2 <= intersection <= 1.1:
                    slices.append(intersection)
        slices.append(0.999)

        # Calculate area and known mineral occurrence percentages
        percents, known_mineral_percent, slices = self.area_percentage(raw_data, "Index Overlay", self.input_interpol, slices)

        # Interpolate percentages for plotting
        xnew, known_mineral_percent_interp, percents_interp = self._interpolate_percentages(slices, percents, known_mineral_percent)

        # Plotting the P-A diagram
        self._plot_pa_diagram(xnew, known_mineral_percent_interp, percents_interp)

        # Calculate intersections and metrics for the P-A diagram
        df = self._calculate_pa_metrics(known_mineral_percent_interp, percents_interp, xnew)

        return df

    def model_evaluation(self, io_data, input_elements):
        """
        Visualizes various aspects of the geochemical model evaluation including geochemical maps,
        fractal C-A models, discretized maps, and P-A diagrams.

        Parameters:
            io_data (DataFrame): The dataset used for generating the geochemical maps and analyses.
            input_elements (str or list): The elements or indices used for the analysis.

        Note:
        This function assumes that `io_mapping`, `ca_modified`, `discrete_mapping_modified_new`, 
        and `pa_diagram_modified` are methods within the same class that produce matplotlib figures
        or axes objects, and not just images. Adjustments may be needed based on their implementations.
        """
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=[16, 16])
        plt.suptitle("Model Evaluation")

        # Assuming each method modifies the passed axes object or returns an axes for plotting
        # Geochemical map
        self.io_mapping(data=io_data, ax=axes[0, 0])

        # Fractal concentration-area model
        self.ca_modified(data=io_data, elements=input_elements, ax=axes[0, 1])

        # Discretized map - This might need adjustments based on what `discrete_mapping_modified_new` actually returns
        self.discrete_mapping_modified_new(data=io_data, lines=input_elements, ax=axes[1, 0])

        # Prediction-area diagram - Adjust based on actual functionality
        self.pa_diagram_modified(data=io_data, element_concentration="Index Overlay", lines=input_elements, ax=axes[1, 1])

        plt.show()
    

########################################################################
# input_elements = input(
#     "\n For what columns should the geochemical processes be done? "
#     "\n Please use 'space' as a separator. \n"
# )
# # Zn Pb Ag Cu Mo Cr Ni Co Ba

# input_X = input(" Which column corresponds with X axis of the Catesian system?\n")
# # X

# input_Y = input("\n Which column corresponds with Y axis of the Catesian system?\n")
# # Y

# input_interpol = input(
#     "\n Which interpolation method should be used? cubic, linear or nearest? "
#     "Please write the desired one.\n"
# )
# # linear


data_path = "C:\\Users\\thc\\Desktop\\MinexPy\\examples\\Data.csv"
input_elements = "Zn Pb Ag Cu Mo Cr Ni Co Ba"

test = MinexPy(
    data_path=data_path,
    input_elements=input_elements,
    input_X="X",
    input_Y="Y",
    input_interpol="linear",  # cubic, linear, or nearest
)


# test.basic_stats_df(input_elements)
# test.correlation_iterated(input_elements)
# test.show_statistical_plots(input_elements)

# raw_data = test.preprocess(data_path)
# test.mapping(raw_data, input_elements, "seismic", title="Geological map of ")


# a, b, c = test.pca(input_elements)

# print(" Here is the Principal Component Analysis for each element:\n")
# print(a)
# print("\n And, here is the Principal Component Analysis Matrix of each element:\n")
# print(c)
# print("\n Finally, here is the PCA map of each element:\n")

# test.mapping(b, input_elements, "seismic", title="PCA map of ")

raw_data = test.preprocess(data_path)
lines_from_ca = test.ca(raw_data, input_elements)

# raw_data = test.preprocess(data_path)
## BUG: file path is not defined for 'Known_mineral_deposits-970729.xls'
# test.discrete_mapping_iterated(raw_data, input_elements, lines=lines_from_ca)

## BUG: file path is not defined for 'Known_mineral_deposits-970729.xls'
weights = test.pa_diagram(raw_data, input_elements, lines=lines_from_ca)
print(weights)


io_data = test.index_overlay(raw_data, input_elements, weights)
print(io_data)

test.io_mapping(io_data)

new_lines_from_ca = test.ca_modified(io_data, input_elements)
## TODO: Not defined discrete_mapping_iterated_modified()
# test.discrete_mapping_iterated_modified(data=io_data, lines=new_lines_from_ca)


test.pa_diagram_modified(
    raw_data=io_data, element_concentration="Index Overlay", lines=new_lines_from_ca
)


test.model_evaluation()
