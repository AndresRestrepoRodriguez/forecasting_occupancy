from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
from .utils import (
    labelling_predictions,

)
import os
import time 

class Model():

    """ Class for Model definition """

    def __init__(self, device_id):

        """
        Set necessary attributes for Model class

        Args:
        device_id (str): device ID

        """

        self.model = Prophet()
        self.device_id = device_id

    
    def fit_predict(self, train_data, test_data):

        """
        Train and predict with Prophet model.

        Args:
            train_data (pandas.DataFrame): Training dataset
            test_data (pandas.DataFrame): Testing dates dataset

        Returns:
            predicted_labels (pandas.DataFrame): Occupancy predicted labels
        
        """

        self.model.fit(train_data)
        predicted_data = self.model.predict(test_data)
        predicted_data = predicted_data[['ds', 'yhat']].rename(columns={'ds': 'time', 'yhat': 'pred'})
        predicted_labels = labelling_predictions(predicted_data)
        return predicted_labels
    

    def save_model(self, folder):

        """ Save trained model """

        file_path = os.path.join(folder, f"model_prophet_occupancy_{self.device_id}.json")
        with open(file_path, 'w') as fout:
            fout.write(model_to_json(self.model))

    
    def load_model(self, folder):

        """ Load saved model """

        file_path = os.path.join(folder, f"model_prophet_occupancy_{self.device_id}.json")
        print(file_path)
        with open(file_path, 'r') as fin:
            self.model = model_from_json(fin.read())

    
    def predict_service(self, test_data):

        """
        Predict as of test dataset

        Args:
            test_data (pandas.DataFrame): Testing dates dataset

        Return:
            predicted_labels (pandas.DataFrame): Occupancy predicted labels

        """

        predicted_data = self.model.predict(test_data)
        predicted_data = predicted_data[['ds', 'yhat']].rename(columns={'ds': 'time', 'yhat': 'pred'})
        predicted_labels = labelling_predictions(predicted_data)
        return predicted_labels



    
