import pandas as pd
from datetime import datetime, time, timedelta
import numpy as np
from sklearn.metrics import classification_report


def read_data(input_path):
    raw_data = pd.read_csv(input_path)
    return raw_data


def filter_data_by_device(raw_data, device):
    return raw_data[raw_data['device'] == device][['time', 'device_activated']].reset_index(drop=True)


def creating_date_range(start_date, end_date, time_frame):

    delta = timedelta(minutes=time_frame)
    while start_date < end_date:
        yield start_date
        start_date += delta


def create_dummy_date_data(starting_date, finishing_date):
    dates = []
    dates_df = pd.DataFrame(columns=['ds'])

    for single_date in creating_date_range(start_date=starting_date, end_date=finishing_date, time_frame=60):
        dates.append(single_date.strftime('%Y-%m-%d %H:%M:%S'))

    dates_df['ds'] = dates
    dates_df['ds'] = pd.to_datetime(dates_df['ds'])
    return dates_df


def create_datetime_range(current_time):
        next_24_hours = pd.date_range(current_time, periods=24, freq='H').ceil('H')

        return pd.DataFrame(next_24_hours, columns=['ds'])


def check_evaluation_option(complete_dataset, test_dates):
    complete_dataset_dates_array = complete_dataset['ds'].values
    test_dates_array = test_dates['ds'].values
    return np.in1d(test_dates_array, complete_dataset_dates_array).all()


def get_ground_truth(complete_dataset, test_dates):
    test_dates_array = test_dates['ds'].values
    filter_data = complete_dataset[(complete_dataset['ds']>=test_dates_array[0]) & (complete_dataset['ds'] <= test_dates_array[-1])]
    filter_data['label'] = filter_data['y'].apply(lambda x: 0 if x == 0 else 1)
    return filter_data['label'].values


def labelling_predictions(pred_data):
    pred_mean = pred_data['pred'].mean()
    pred_std = pred_data['pred'].std()
    pred_data['z_score'] = pred_data['pred'].apply(lambda row: round((row - pred_mean)/pred_std, 2))
    pred_data['activation_predicted'] = pred_data['z_score'].apply(lambda row: 1 if row > 0 else 0)
    return pred_data[['time', 'activation_predicted']]


def evaluation_classification(true_labels, predicted_labels):
    return classification_report(true_labels, predicted_labels)


def export_results_output_file(output_file_path, list_dataframes):
    pd.concat(list_dataframes)[['time', 'device', 'activation_predicted']].to_csv(output_file_path, index=False)

