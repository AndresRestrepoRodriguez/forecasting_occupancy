from datetime import datetime


def generate_timestamp():
    dt = datetime.now()
    return dt.strftime(format='%Y-%m-%d %H:%M:%S')


def format_predictions(predictions, device):
    prediction_dict = predictions.to_dict()
    time_values = prediction_dict['time']
    occupancy_values = prediction_dict['activation_predicted']
    results = {'device': device}
    list_predictions = [{'timestamp': value.to_pydatetime().strftime(format='%Y-%m-%d %H:%M:%S'), 'prediction': occupancy_values[key]} for (key, value) in time_values.items()]
    results['predictions'] = list_predictions
    return results
