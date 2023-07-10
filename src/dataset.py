from .processing import Processor
from .utils import (
    read_data,
    create_datetime_range,
    check_evaluation_option,
    get_ground_truth
)
import pandas as pd
import numpy as np

class Dataset():

    """ Class for Dataset creation and definition """

    def __init__(self, data_dir, device):

        """
        Set necessary attributes for Dataset class

        Args:
        data_dir (str): File path of sensors dataset
        device (str): device ID

        """

        self.raw_data = read_data(data_dir)
        self.processor = Processor(self.raw_data)
        self.device = device

    
    def get_complete_dataset(self):

        """ Get complete dataset filtering by device. """

        self.dataset = self.processor.define_complete_dataset(self.device)
        self.dataset.rename(columns={'time': 'ds', 'device_activated': 'y'})


    def generate_train_test_dataset(self, current_time: str):

        """
        Generate the training and testing dataset. Testing dataset will be only the dates.

        Args:
            current_time (str): Timestamp to get next 24 hours
        
        Returns:
            train_data (pandas.DataFrame): Training data
            test_data_dates (pandas.DataFrame): Testing data dates

        """

        self.get_complete_dataset()
        self.test_data_dates = create_datetime_range(current_time)
        last_date_complete_dataset = self.dataset['ds'][len(self.dataset)-1]
        first_date_test_data = self.test_data_dates['ds'][0]
        if first_date_test_data <= last_date_complete_dataset:
            self.train_data = self.dataset[self.dataset['ds'] < first_date_test_data]
        else:
            self.train_data = self.dataset
        return self.train_data, self.test_data_dates
        
    
    def check_evaluation_option(self):
        """ Check if it is possible carry out a evaluation of classification """
        return check_evaluation_option(self.dataset, self.test_data_dates)
    

    def get_ground_truth_test_dates(self):

        """ Get ground truth labels if it is possible. To perform evaluation"""

        try:
            labels = get_ground_truth(self.dataset, self.test_data_dates)
            return labels
        except Exception as e:
            return np.array([])
   


