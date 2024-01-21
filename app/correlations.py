import pandas as pd
from scipy import stats

class Correlation:
    """
    A class used to calculate the Pearson and Spearman correlation coefficients for a dataframe.

    ...

    Attributes
    ----------
    data : pandas.DataFrame
        a dataframe containing the data to be analyzed

    Methods
    -------
    correlation(first_element, second_element)
        Calculates the Pearson and Spearman correlation coefficients for two columns in a dataframe.
    correlation_iterated(element_concentration)
        Iterates over each column and returns the results for all columns in a single dataframe for each of the coefficients.
    """

    def __init__(self, data, element_concentration):
        """
        Constructs all the necessary attributes for the Correlation object.

        Parameters
        ----------
            data : pandas.DataFrame
                a dataframe containing the data to be analyzed
            element_concentration : str
                names of columns to be analyzed separated by spaces
        """

        self.data = data
        self.element_concentration = element_concentration

    def correlation(self, first_element, second_element):
        """
        Calculates the Pearson and Spearman correlation coefficients for two columns in a dataframe.

        Parameters
        ----------
            first_element : str
                name of the first column to be analyzed
            second_element : str
                name of the second column to be analyzed

        Returns
        -------
            tuple
                a tuple containing the Pearson and Spearman correlation coefficients for the two columns
        """

        first_elem, second_elem = self.data.loc[:, f'{first_element}'], self.data.loc[:, f'{second_element}']
        spearcorr = list(stats.spearmanr(first_elem, second_elem))
        pearcorr = list(stats.pearsonr(first_elem, second_elem))

        spearcorr = round(spearcorr[0], 3)
        pearcorr = round(pearcorr[0], 3)

        return pearcorr, spearcorr

    def correlation_iterated(self):
        """
        Iterates over each column and returns the results for all columns in a single dataframe for each of the coefficients.

        Returns
        -------
            None
                prints out two dataframes containing the Pearson and Spearman correlation coefficients for all columns in element_concentration
        """

        columns = index_names = self.element_concentration

        data1 = {}
        data2 = {}
        selected_columns = self.data.loc[:,columns].columns   
        
        for i in selected_columns:
            df = self.data.loc[:, [i] + columns]
            df_corr = df.corr(method='pearson')
            df_corr_spearman = df.corr(method='spearman')
            temp1 = list(df_corr[i].values[1:])
            temp2 = list(df_corr_spearman[i].values[1:])
            data1[i] = temp1
            data2[i] = temp2

        df1 = pd.DataFrame(data1,index_names)
        df2 = pd.DataFrame(data2,index_names)

        print('Here is the Pearson correlation coefficient of the elements:\n', df1 , '\n')
        print('Here is the Spearman correlation coefficient of the elements:\n', df2)

