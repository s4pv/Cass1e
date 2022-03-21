from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import math
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error
from helper import Helper

import numpy

from preprocessing import Preprocessing
from modelplot import ModelPlot
from modelparameters import ModelParameters

warnings.filterwarnings("ignore")

# Configuration and class variables
parsed_config = Helper.load_config('config.yml')

NO_DAYS = parsed_config['model_options']['NO_DAYS']

TRAIN_SIZE = parsed_config['ml_options']['TRAIN_SIZE']
NEURONS = parsed_config['ml_options']['NEURONS']
DROPOUT = parsed_config['ml_options']['DROPOUT']
N_STEPS_IN = parsed_config['ml_options']['N_STEPS_IN']
N_STEPS_OUT = parsed_config['ml_options']['N_STEPS_OUT']
EPOCH = parsed_config['ml_options']['EPOCH']
N_FEATURES = parsed_config['ml_options']['N_FEATURES']
VERBOSE = parsed_config['ml_options']['VERBOSE']

class ModelForecast:
    def Predict_LSTM(dataset, coin):
        try:
            # load previously fitted model
            lstm = ModelParameters.Load_Model('LSTM', coin)
            # evaluate loaded model on test data
            lstm.compile(loss='mean_squared_error', optimizer='adam')
            ds = Preprocessing.Reshape_Float(dataset)
            ds = Preprocessing.Minmax_Scaler(ds, 'LSTM', coin)
            print(len(ds))
            train, test = Preprocessing.Dataset_Split(ds)
            trainX, trainY, testX, testY = Preprocessing.Split_Sequences(train, test)
            # fix random seed for reproducibility
            numpy.random.seed(7)
            # evaluate model score
            score = lstm.evaluate(trainX, trainY, batch_size=N_STEPS_IN, verbose=VERBOSE)
            print("%s: %.2f%%" % ('loss is: ', score * 100))


            # Mean Absolute Error (MAE)
            MAE = mean_absolute_error(y_test_unscaled, y_pred)
            print(f'Median Absolute Error (MAE): {np.round(MAE, 2)}')

            # Mean Absolute Percentage Error (MAPE)
            MAPE = np.mean((np.abs(np.subtract(y_test_unscaled, y_pred) / y_test_unscaled))) * 100
            print(f'Mean Absolute Percentage Error (MAPE): {np.round(MAPE, 2)} %')

            # Median Absolute Percentage Error (MDAPE)
            MDAPE = np.median((np.abs(np.subtract(y_test_unscaled, y_pred) / y_test_unscaled))) * 100
            print(f'Median Absolute Percentage Error (MDAPE): {np.round(MDAPE, 2)} %')

        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True