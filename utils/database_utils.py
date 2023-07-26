import pandas as pd
import numpy as np

class DataImporter:
    """
    A class for importing and cleaning data from a CSV file.
    
    Attributes:
        file_path (str): The path to the CSV file.
    
    Methods:
        import_csv(): Reads the CSV file at the specified location and returns a DataFrame.
        clean_data(data): Cleans up the DataFrame by removing empty rows and duplicates.
        get_cleaned_data(): Imports the data and cleans it up in one step.
    """
    
    def __init__(self, file_path):
        """
        The constructor for the DataImporter class.
        
        Parameters:
            file_path (str): The path to the CSV file.
        """
        self.file_path = file_path
    
    def import_csv(self):
        """
        Reads the CSV file at the specified location and returns a DataFrame.
        
        Returns:
            pandas.DataFrame: The imported data as a DataFrame.
        """
        data = pd.read_csv(self.file_path)
        return data
    
    def clean_data(self, data):
        """
        Cleans up the DataFrame by removing empty rows and duplicates.
        
        Parameters:
            data (pandas.DataFrame): The DataFrame to clean up.
        
        Returns:
            pandas.DataFrame: The cleaned up DataFrame.
        """
        data = data.replace("", np.nan)
        data = data.dropna()
        data = data.drop_duplicates().loc[:, 'X':]
        return data
    
    def get_cleaned_data(self):
        """
        Imports the data and cleans it up in one step.
        
        Returns:
            pandas.DataFrame: The cleaned up DataFrame.
        """
        data = self.import_csv()
        cleaned_data = self.clean_data(data)
        return cleaned_data
    

# data_importer = DataImporter('/Users/yasin/Desktop/ProjectX/Data.csv')
# cleaned_data = data_importer.get_cleaned_data()
