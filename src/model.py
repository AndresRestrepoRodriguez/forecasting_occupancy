from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
from .utils import (
    labelling_predictions,

)
import os
import time 

class Model():
    def __init__(self, device_id):
        self.model = Prophet()
        self.device_id = device_id

    
    def fit_predict(self, train_data, test_data):
        self.model.fit(train_data)
        predicted_data = self.model.predict(test_data)
        predicted_data = predicted_data[['ds', 'yhat']].rename(columns={'ds': 'time', 'yhat': 'pred'})
        predicted_labels = labelling_predictions(predicted_data)
        return predicted_labels
    

    def save_model(self, folder):
        file_path = os.path.join(folder, f"model_prophet_occupancy_{self.device_id}.json")
        with open(file_path, 'w') as fout:
            fout.write(model_to_json(self.model))

    
    def load_model(self, folder):
        file_path = os.path.join(folder, f"model_prophet_occupancy_{self.device_id}.json")
        print(file_path)
        with open(file_path, 'r') as fin:
            self.model = model_from_json(fin.read())

    
    def predict_service(self, test_data):
        predicted_data = self.model.predict(test_data)
        predicted_data = predicted_data[['ds', 'yhat']].rename(columns={'ds': 'time', 'yhat': 'pred'})
        predicted_labels = labelling_predictions(predicted_data)
        return predicted_labels



    
