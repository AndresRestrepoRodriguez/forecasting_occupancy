import pandas as pd
from datetime import datetime, time, timedelta
from .utils import (
    filter_data_by_device,
    create_dummy_date_data
)



class Processor():

    """ Class for Processing pipeline """

    def __init__(self, raw_data):

        """
        Set necessary attributes for Processor class

        Args:
        raw_data (pandas.DataFrame): data from csv file

        """

        self.raw_data = raw_data
        self.process_init_data()
        self.starting_date, self.finishing_date = self.get_min_max_time()


    def process_init_data(self):

        """ Convert time columns from object type to datetime """

        self.raw_data['time'] = pd.to_datetime(self.raw_data['time'])


    def get_min_max_time(self):

        """ Get max and min date in whole dataset """

        starting_date = datetime.combine(self.raw_data['time'].min(), datetime.min.time())
        finishing_date = datetime.combine(self.raw_data['time'].max(), time(0, 0)) + timedelta(1)
        return starting_date, finishing_date


    def define_complete_dataset(self, device):

        """
        Filter dataset by device, sum data by hour and add dummy dates to complete max and min dates.

        Args:
            device (str): Device ID

        Returns:
            complete_dataset_device: Processed data for the device 

        """

        filter_data_device = filter_data_by_device(self.raw_data, device)
        grouped_data_hour = self.group_by_hour(filter_data_device)
        dummy_data = create_dummy_date_data(self.starting_date, self.finishing_date)
        complete_dataset_device = pd.merge(dummy_data, grouped_data_hour, on='ds', how='left').fillna(0)
        return complete_dataset_device
    
    
    @staticmethod
    def group_by_hour(raw_device_data):

        """
        Group (Sum) data by hour

        Args:
            raw_device_data (pandas.DataFrame): Dataset filtered by device

        Returns:
            grouped_data (pandas.DataFrame): Sum activations by hour

        """

        grouped_data = raw_device_data.set_index(pd.DatetimeIndex(raw_device_data['time'])).resample("H").sum().reset_index().rename(
        columns={'time': 'ds', 'device_activated': 'y'})
        return grouped_data