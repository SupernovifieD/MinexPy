import utils
from utils.config import input_elements

data_importer = utils.DataImporter('/Users/yasin/Desktop/ProjectX/Data.csv')
cleaned_data = data_importer.get_cleaned_data()

# Access the input_elements variable
columns = input_elements.split()

class DescriptiveStatistics:
    """
    A class that calculates descriptive statistics for a given dataframe.

    Parameters:
    -----------
    data : pandas.DataFrame
        The dataframe to calculate the statistics on.

    columns : list
        The columns to calculate the statistics on.

    Returns:
    --------
    pandas.DataFrame
        A dataframe containing the descriptive statistics for each column.

    Examples:
    ---------
    >>> data = pd.read_csv('Data.csv')
    >>> stats = DescriptiveStatistics(data=data, columns=['column1', 'column2'])
    >>> stats.basic_stats_df()

               No. of Observations   Min   Max  Mean  Variance  Skewness  Kurtosis
    column1                    1000  0.01  0.99  0.50      0.08      0.00     -1.20
    column2                    1000 -1.00  1.00 -0.00      1.00      0.00     -1.20

    """

    def __init__(self, data, columns):
        self.data = data
        self.columns = columns

    def basic_stats(self, element_concentration):
        """
        Calculates basic statistics for a given column in the dataframe.

        Parameters:
        -----------
        element_concentration : str
            The name of the column to calculate the statistics on.

        Returns:
        --------
        list
            A list containing the calculated statistics.

        """

        selected_values = self.data[element_concentration].describe().to_dict()

        s1 = str(selected_values['count'])
        s2 = str(selected_values['min'])
        s3 = str(selected_values['max'])
        s4 = str(round(selected_values['mean'], 2))
        s5 = str(round(selected_values['std'], 2))
        s6 = str(round(selected_values['skew'], 2))
        s7 = str(round(selected_values['kurt'], 2))

        return [s1,s2,s3,s4,s5,s6,s7]

    def basic_stats_df(self):
        """
        Calculates basic statistics for all columns in the dataframe.

        Returns:
        --------
        pandas.DataFrame
            A dataframe containing the calculated statistics for each column.

        """

        columns = ['No. of Observations', 'Min', 'Max', 'Mean', 'Variance', 'Skewness', 'Kurtosis']

        df = self.data[self.columns].agg(['count', 'min', 'max', 'mean', 'var', 'skew', 'kurt']).T

        df.columns = columns

        return df






#stats = DescriptiveStatistics(data=cleaned_data ,columns=columns)
#print(stats.basic_stats_df())