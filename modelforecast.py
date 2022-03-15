from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers import Dropout
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import math
import warnings
from sklearn.metrics import mean_squared_error
from helper import Helper

import numpy

from preprocessing import Preprocessing
from modelplot import ModelPlot
from modelparameters import ModelParameters

warnings.filterwarnings("ignore")

# Configuration and class variables
parsed_config = Helper.load_config('config.yml')

TO_FORECAST = parsed_config['forecast_options']['TO_FORECAST']

NO_DAYS = parsed_config['model_options']['NO_DAYS']

LOOK_BACK = parsed_config['ml_options']['LOOK_BACK']
TRAIN_SIZE = parsed_config['ml_options']['TRAIN_SIZE']
NEURONS = parsed_config['ml_options']['NEURONS']
DROPOUT = parsed_config['ml_options']['DROPOUT']
BATCH_SIZE = parsed_config['ml_options']['BATCH_SIZE']
OUTPUT_DIM = parsed_config['ml_options']['OUTPUT_DIM']
EPOCH = parsed_config['ml_options']['EPOCH']
N_FEATURES = parsed_config['ml_options']['N_FEATURES']
HIDDEN_UNITS = parsed_config['ml_options']['HIDDEN_UNITS']
TIME_STEPS = parsed_config['ml_options']['TIME_STEPS']
DENSE_UNITS = parsed_config['ml_options']['DENSE_UNITS']
FILTERS = parsed_config['ml_options']['FILTERS']
KERNEL_SIZE = parsed_config['ml_options']['KERNEL_SIZE']
POOL_SIZE = parsed_config['ml_options']['POOL_SIZE']
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
            trainX, trainY, testX, testY = Preprocessing.Reshape_Data(train, test)
            # fix random seed for reproducibility
            numpy.random.seed(7)
            # evaluate model score
            score = lstm.evaluate(trainX, trainY, batch_size=BATCH_SIZE, verbose=VERBOSE)
            print("%s: %.2f%%" % ('loss is: ', score * 100))
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True