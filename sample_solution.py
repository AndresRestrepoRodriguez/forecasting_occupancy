from src.dataset import Dataset
from src.model import Model
from src.utils import (
    evaluation_classification,
    export_results_output_file
)
import warnings
import argparse
warnings.filterwarnings('ignore')


DEVICES = [f'device_{num}' for num in range(1,8)]
METRICS_RESULTS_ = []

def predict_future_activation(data_dir, current_time, output_file, save_models=False, save_folder='models/'):
    predictions_results_stack = []
    for device in DEVICES:
        print(f"Processing device: {device}")
        dataset = Dataset(data_dir, device)
        train, test = dataset.generate_train_test_dataset(current_time)
        status_evaluation = dataset.check_evaluation_option()

        prophet_model = Model(device)
        predicted_results = prophet_model.fit_predict(train, test)
        predicted_results['device'] = device
        labels = predicted_results['activation_predicted'].values
        predictions_results_stack.append(predicted_results)
        
        if save_models:
            prophet_model.save_model(save_folder)

        if status_evaluation:
            ground_truth_labels = dataset.get_ground_truth_test_dates()
            if len(ground_truth_labels) > 0:
                METRICS_RESULTS_.append((device, evaluation_classification(ground_truth_labels, labels)))
        
    export_results_output_file(output_file, predictions_results_stack)

    if METRICS_RESULTS_ and len(METRICS_RESULTS_) == len(DEVICES):
        for metric_result in METRICS_RESULTS_:
            print(f'Device: {metric_result[0]}')
            print('Metrics')
            print(metric_result[-1])
            print('-'*10)
    else:
        message = "It is not possible to show the classification metrics since,\nfor the selected timestamp the next 24 hours are not found \nin the original dataset. But you can check your output file."""
        print(message)

if __name__ == "__main__":

    parser = argparse.ArgumentParser('Substitution Process')
    parser.add_argument('-i', '--input_file', type=str, required=True)
    parser.add_argument('-t', '--timestamp', type=str, required=True)
    parser.add_argument('-o', '--output_file', type=str, required=True)
    parser.add_argument('-s', '--save_models',  type=str, required=False, default='false')

    args = parser.parse_args()
    input_file = str(args.input_file)
    current_time = str(args.timestamp)
    output_file = str(args.output_file)
    save_models = str(args.save_models).lower()

    save_models_bool = True if save_models == 'true' else False

    if save_models_bool:
        predict_future_activation(input_file, current_time, output_file, save_models_bool)
    
    else:
        predict_future_activation(input_file, current_time, output_file)
    




