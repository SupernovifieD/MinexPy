import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.graphics.gofplots as smg

class StatPlots:
    """
    A class to create and save basic statistical plots for a given data set and columns.

    Attributes
    ----------
    raw_data : pandas.DataFrame
        The cleaned data set as a pandas DataFrame object.
    columns : list of str
        The list of column names to plot.

    Methods
    -------
    plot_stats():
        Plots the statistical plots for all the columns and saves them to a folder on the user's desktop.
    statistical_plots(desired_column):
        Plots the histogram, violin plot, Q-Q plot, and P-P plot for a single column and saves them to a folder on the user's desktop.
    plot_histogram(values, desired_column, ax):
        Plots the histogram for a single column on a given axis.
    plot_boxplot(values, desired_column, ax):
        Plots the violin plot for a single column on a given axis.
    plot_qqplot(values, desired_column, ax):
        Plots the Q-Q plot for a single column on a given axis.
    plot_ppplot(values, desired_column, ax):
        Plots the P-P plot for a single column on a given axis.
    save_figure(desired_column, fig):
        Saves the figure to a folder on the user's desktop with a file name based on the column name.
    
    Example
    -------
    To use this class, you need to create an object of it by passing in your cleaned data and columns. For example:

    >>> sp = StatPlots(cleaned_data, ["age", "weight", "height"])

    Then, you can use the object's methods to plot the statistical plots for your columns. For example:

    >>> sp.plot_stats()

    This will create and save four plots for each column in a folder called "stats graphs made with MinexPy" on your desktop.
    """

    def __init__(self, data, columns):
        """
        Constructs all the necessary attributes for the StatPlots object.

        Parameters
        ----------
        cleaned_data : pandas.DataFrame
            The cleaned data set as a pandas DataFrame object.
        columns : list of str
            The list of column names to plot.
        """

        self.raw_data = data # use the cleaned data as an attribute
        self.columns = columns # use the columns as an attribute
    
    def plot_stats(self): # change the name of the method to plot_stats
        """
        Plots the statistical plots for all the columns and saves them to a folder on the user's desktop.

        This method uses list comprehension to iterate over the columns attribute and call the statistical_plots method for each column.
        """

        [self.statistical_plots(column) for column in self.columns] # use list comprehension to plot all columns
    
    def statistical_plots(self, desired_column):
        """
        Plots the histogram, violin plot, Q-Q plot, and P-P plot for a single column and saves them to a folder on the user's desktop.

        This method uses statsmodels graphics functions to create various types of plots and matplotlib.pyplot functions to customize and save the figure.

        Parameters
        ----------
        desired_column : str
            The name of the column to plot.
        """

        values = self.raw_data.loc[:,f'{desired_column}'].values # get the values from the desired column
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=[8,8]) # create a figure with four subplots
        
        #plot1 => histogram
        self.plot_histogram(values, desired_column, axes[0, 0])
        
        #plot2 => violin plot
        self.plot_violinplot(values, desired_column, axes[0, 1])
        
        #plot3 => Q-Q plot
        self.plot_qqplot(values, desired_column, axes[1, 0])
        
        #plot4 => P-P plot
        self.plot_ppplot(values, desired_column, axes[1, 1])
        
        plt.suptitle(f'Basic Statistical Plots of {desired_column}')
        fig.tight_layout()
        
        # save the figure to a folder on the user's desktop
        self.save_figure(desired_column, fig)
    
    def plot_histogram(self, values, desired_column, ax):
        """
        Plots the histogram for a single column on a given axis.

        This method uses matplotlib.pyplot functions to create and customize the histogram.

        Parameters
        ----------
        values : numpy.ndarray
            The array of values from the column to plot.
        desired_column : str
            The name of the column to plot.
        ax : matplotlib.axes.Axes
            The axis object to plot on.
        
       """

        ax.hist(values, bins=50,color='c', edgecolor='k', alpha=0.65)
        ax.axvline(values.mean(), color='k', linestyle='--', linewidth=1)
        min_ylim, max_ylim = ax.set_ylim() 
        ax.text(values.mean()*1.4, max_ylim*0.9, 'Mean: {:.2f}'.format(values.mean()))
        ax.set(title = f'Histogram chart for {desired_column}')
    
    def plot_boxplot(self, values, desired_column, ax):
        """
        Plots the box plot for a single column on a given axis.

        This method uses matplotlib boxplot function to create the box plot.

        Parameters
        ----------
        values : numpy.ndarray
            The array of values from the column to plot.
        desired_column : str
            The name of the column to plot.
        ax : matplotlib.axes.Axes
            The axis object to plot on.
        
       """

        plt.boxplot(values, vert=False, patch_artist=True, sym='r+', showmeans=True, meanline=True) # use the matplotlib boxplot function
        ax.set(title = f'Box plot for {desired_column}')
        ax.grid(axis = 'x')
    
    def plot_qqplot(self, values, desired_column, ax):
        """
        Plots the Q-Q plot for a single column on a given axis.

        This method uses statsmodels graphics gofplots function to create the Q-Q plot.

        Parameters
        ----------
        values : numpy.ndarray
            The array of values from the column to plot.
        desired_column : str
            The name of the column to plot.
        ax : matplotlib.axes.Axes
            The axis object to plot on.
        
       """

        smg.qqplot(values, line='45', ax=ax) # use statsmodels qqplot function
        ax.set(title = f'Q-Q plot of {desired_column}')
        ax.grid(True)
    
    def plot_ppplot(self, values, desired_column, ax):
        """
        Plots the P-P plot for a single column on a given axis.

        This method uses statsmodels ProbPlot class and ppplot method to create the P-P plot.

        Parameters
        ----------
        values : numpy.ndarray
            The array of values from the column to plot.
        desired_column : str
            The name of the column to plot.
        ax : matplotlib.axes.Axes
            The axis object to plot on.
        
       """

        sm.ProbPlot(values).ppplot(line='45', ax=ax) # use statsmodels ProbPlot class and ppplot method
        ax.set(title = f'P-P plot of {desired_column}')
        ax.grid(True)
    
    def save_figure(self, desired_column, fig):
        """
        Saves the figure to a folder on the user's desktop with a file name based on the column name.

        This method uses os module functions to handle the file and folder operations and matplotlib.pyplot function to save the figure.

        Parameters
        ----------
        desired_column : str
            The name of the column to save.
        fig : matplotlib.figure.Figure
            The figure object to save.
        
       """

        folder_name = "stats graphs made with MinexPy"
        folder_path = os.path.join(os.path.expanduser("~"), "Desktop", folder_name) # get the path to the folder
        if not os.path.exists(folder_path): # check if the folder exists
            os.makedirs(folder_path) # if not, create the folder
        file_name = f"{desired_column}_plots.png" # name the file with the column name and plots suffix
        file_path = os.path.join(folder_path, file_name) # get the path to the file
        fig.savefig(file_path) # save the figure to the file

