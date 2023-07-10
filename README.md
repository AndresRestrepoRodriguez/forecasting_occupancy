# Occupancy forecasting

For the development of the solution from a research stage, the following papers and repositories were taken as a starting point:
- [A review on occupancy prediction through machine learning for enhancing energy efficiency, air quality and thermal comfort in the built environment](https://www.sciencedirect.com/science/article/pii/S1364032122005937?ref=pdf_download&fr=RR-2&rr=7e47db137c775902)
- [Prophet model for forecasting occupancy presence in indoor
spaces using non-intrusive sensors](https://agile-giss.copernicus.org/articles/2/9/2021/agile-giss-2-9-2021.pdf)
- https://github.com/oguzhanyediel/occupancy-prediction

From this, [Prophet](https://facebook.github.io/prophet/) is taken as a forecasting model and the code is created to solve the problem posed.

<br>

# 1) Occupancy prediction model

## Description

To give a solution to predict occupancy for the next 24h after a given timestamp. There is the code in the src folder and the sample_solution.py script. From this, we have the following steps for the solution:

- Read the data. The history of all sensor readings up to the input time.
- Data processing:
    - Data Type Conversion
    - Generation of training and test data for the Prophet model
- Evaluate if it is possible to obtain classification metrics
- Train the model
- Generate predictions with the trained model from the test dataset
- Save the model if required.
- Obtain evaluation metrics if possible.
- Repeat the above steps for each device.
- Export the predictions obtained in a file with a CSV extension
- Whether the evaluation metrics were obtained are displayed in the output of the script.

## Parameters

The sample_solution.py script receives the following parameters:

- '-i' or '--input_file': Input file path (Required)
- '-t' or '--timestamp': Input timestamp in format '%Y-%m-%d %H:%M:%S' (Required)
- '-o' or '--output_file': Output file path (Required)
- '-s' or '--save_models': To save or not the trained models. They will be stored in the models folder. Default is false. (true or false) (Not required)

## Usage
These instructions are specified for Linux OS.

Create a virtual python environment
```
python3 -m venv env
```

Activate the created virtual environment
```
source env/bin/activate
```

Clone the repository
```
git clone https://github.com/AndresRestrepoRodriguez/forecasting_occupancy.git
```

Go to the repository folder
```
cd forecasting_occupancy
```

Install the necessary packages
```
pip install -r requirements.txt
```

Run the script sample_solution.py
```
python3 sample_solution.py -t '2023-08-31 23:59:59' -i data/device_activations.csv -o data/predictions.csv
```

### Examples
1) <b>Without performance evaluation</b>

    Es decir que seleccionamos 24 horas que no están en el dataset por lo tanto no se tiene el ground truth para obtener métricas.
    ```
    python3 sample_solution.py -t '2022-08-31 23:59:59' -i data/device_activations.csv -o data/predictions.csv
    ```
    console output
    ```
    12:21:51 - cmdstanpy - INFO - Chain [1] done processing
    Processing device: device_5
    12:21:51 - cmdstanpy - INFO - Chain [1] start processing
    12:21:51 - cmdstanpy - INFO - Chain [1] done processing
    Processing device: device_6
    12:21:51 - cmdstanpy - INFO - Chain [1] start processing
    12:21:52 - cmdstanpy - INFO - Chain [1] done processing
    Processing device: device_7
    12:21:52 - cmdstanpy - INFO - Chain [1] start processing
    12:21:52 - cmdstanpy - INFO - Chain [1] done processing
    It is not possible to show the classification metrics since,
    for the selected timestamp the next 24 hours are not found
    in the original dataset. But you can check your output file.
    ```

2) <b>With performance evaluation</b>

    Es decir que seleccionamos 24 horas que si están en el dataset por lo tanto se tiene el ground truth para obtener métricas.
    ```
    python3 sample_solution.py -t '2022-08-29 23:59:59' -i data/device_activations.csv -o data/predictions.csv
    ```
    console output
    ```
    Processing device: device_1
    12:22:39 - cmdstanpy - INFO - Chain [1] start processing
    12:22:39 - cmdstanpy - INFO - Chain [1] done processing
    Processing device: device_2
    12:22:39 - cmdstanpy - INFO - Chain [1] start processing
    12:22:39 - cmdstanpy - INFO - Chain [1] done processing
    Processing device: device_3
    12:22:39 - cmdstanpy - INFO - Chain [1] start processing
    12:22:40 - cmdstanpy - INFO - Chain [1] done processing
    Processing device: device_4
    12:22:40 - cmdstanpy - INFO - Chain [1] start processing
    12:22:40 - cmdstanpy - INFO - Chain [1] done processing
    Processing device: device_5
    12:22:40 - cmdstanpy - INFO - Chain [1] start processing
    12:22:40 - cmdstanpy - INFO - Chain [1] done processing
    Processing device: device_6
    12:22:40 - cmdstanpy - INFO - Chain [1] start processing
    12:22:40 - cmdstanpy - INFO - Chain [1] done processing
    Processing device: device_7
    12:22:40 - cmdstanpy - INFO - Chain [1] start processing
    12:22:40 - cmdstanpy - INFO - Chain [1] done processing
    Device: device_1
    Metrics
                precision    recall  f1-score   support

            0       1.00      0.93      0.97        15
            1       0.90      1.00      0.95         9

        accuracy                           0.96        24
    macro avg       0.95      0.97      0.96        24
    weighted avg       0.96      0.96      0.96        24

    ----------
    Device: device_2
    Metrics
                precision    recall  f1-score   support

            0       0.93      1.00      0.96        13
            1       1.00      0.91      0.95        11

        accuracy                           0.96        24
    macro avg       0.96      0.95      0.96        24
    weighted avg       0.96      0.96      0.96        24

    ----------
    Device: device_3
    Metrics
                precision    recall  f1-score   support

            0       1.00      1.00      1.00        15
            1       1.00      1.00      1.00         9

        accuracy                           1.00        24
    macro avg       1.00      1.00      1.00        24
    weighted avg       1.00      1.00      1.00        24

    ----------
    Device: device_4
    Metrics
                precision    recall  f1-score   support

            0       0.93      1.00      0.96        13
            1       1.00      0.91      0.95        11

        accuracy                           0.96        24
    macro avg       0.96      0.95      0.96        24
    weighted avg       0.96      0.96      0.96        24

    ----------
    Device: device_5
    Metrics
                precision    recall  f1-score   support

            0       1.00      0.87      0.93        15
            1       0.82      1.00      0.90         9

        accuracy                           0.92        24
    macro avg       0.91      0.93      0.91        24
    weighted avg       0.93      0.92      0.92        24

    ----------
    Device: device_6
    Metrics
                precision    recall  f1-score   support

            0       0.86      1.00      0.92        12
            1       1.00      0.83      0.91        12

        accuracy                           0.92        24
    macro avg       0.93      0.92      0.92        24
    weighted avg       0.93      0.92      0.92        24

    ----------
    Device: device_7
    Metrics
                precision    recall  f1-score   support

            0       1.00      0.62      0.77        24
            1       0.00      0.00      0.00         0

        accuracy                           0.62        24
    macro avg       0.50      0.31      0.38        24
    weighted avg       1.00      0.62      0.77        24

    ----------
    ```

3) <b>Saving the models</b>

    ```
    python3 sample_solution.py -t '2023-08-31 23:59:59' -i data/device_activations.csv -o data/predictions.csv -s true
    ```

    These models that were saved in the models folder will be used in the prediction from an API.

<br>

# 2) REST API

The REST API was built with FastAPI by consuming the trained models in the *Saving the models* example. Additionally, the REST API was Dockerized and it can be deployed as follows:

Create the image
```
docker build -t forecasting_occupancy:latest .
```

Run the image
```
docker run -d -p 5000:5000 -t forecasting_occupancy:latest
```

Test the REST API
- Health check endpoint: Check if the api is working.
    ```
    curl --location 'http://<DOCKER-IP>:5000/health'
    ```

- Prediction endpoint: Predict the next 24 hours from the time the request is made.
    ```
    curl --location 'http://<DOCKER-IP>:5000/predict'
    ```


