from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Bidirectional

import math
import warnings
from sklearn.metrics import mean_squared_error
from helper import Helper
import os

import numpy

from preprocessing import Preprocessing
from modelplot import ModelPlot
from modelparameters import ModelParameters

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

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


class MachineLearning:
    def LSTM(dataset, coin):
        try:
            # preprocessing data
            ds_f_c, ds_f_ohlv, ds_f_ohlcv = Preprocessing.Prepare_Data(dataset)
            # scaling data
            ohlcv_scaled = Preprocessing.Minmax_Scaler(ds_f_ohlcv, 'OHLCV_LSTM', coin)
            ohlv_scaled = Preprocessing.Minmax_Scaler(ds_f_ohlv, 'OHLV_LSTM', coin)
            c_scaled = Preprocessing.Minmax_Scaler(ds_f_c, 'C_LSTM', coin)
            print(ohlcv_scaled.shape)
            print(ohlv_scaled.shape)
            print(c_scaled.shape)
            # Prediction Index
            train, test = Preprocessing.Dataset_Split(ohlcv_scaled)
            # for multivariate multi step predict purposes
            trainX, trainY = Preprocessing.Split_Sequences(train, N_STEPS_IN, N_STEPS_OUT)
            testX, testY = Preprocessing.Split_Sequences(test, N_STEPS_IN, N_STEPS_OUT)
            # Print the shapes: the result is: (rows, training_sequence, features) (prediction value, )
            print(trainX.shape, trainY.shape)
            print(testX.shape, testY.shape)
            # fix random seed for reproducibility
            numpy.random.seed(7)
            # Create and fit the Long short Term Memory network.
            lstm = Sequential()
            # Not stacked LSTM bidirectional multi step output
            lstm.add(Bidirectional(LSTM(NEURONS, input_shape=(N_STEPS_IN, N_FEATURES), return_sequences=True)))
            lstm.add(Dropout(DROPOUT))
            lstm.add(Dense(N_STEPS_OUT))
            lstm.compile(loss='mean_squared_error', optimizer='adam')
            # No memory between batches
            lstm.fit(trainX, trainY, epochs=EPOCH, batch_size=N_STEPS_IN, verbose=VERBOSE)
            print(lstm.summary())
            lstm.reset_states()
            # make predictions
            trainPredict = lstm.predict(trainX, batch_size=N_STEPS_IN)
            lstm.reset_states()
            print(trainPredict.shape)
            testPredict = lstm.predict(testX, batch_size=N_STEPS_IN)
            print(testPredict.shape)
            # reshape sets
            trainX = numpy.reshape(trainX, (869, 4))
            testX = numpy.reshape(testX, (69, 4))
            trainY = numpy.reshape(trainY, (869, 32))
            testY = numpy.reshape(testY, (69, 32))
            # invert sets
            trainPredict = Preprocessing.Invert_Transform(trainPredict[868], coin, 'C_LSTM', 'MINMAXSCALER')
            testPredict = Preprocessing.Invert_Transform(testPredict[68], coin, 'C_LSTM', 'MINMAXSCALER')
            trainX = Preprocessing.Invert_Transform(trainX, coin, 'OHLV_LSTM', 'MINMAXSCALER')
            testX = Preprocessing.Invert_Transform(testX, coin, 'OHLV_LSTM', 'MINMAXSCALER')
            trainY = Preprocessing.Invert_Transform(trainY, coin, 'C_LSTM', 'MINMAXSCALER')
            testY = Preprocessing.Invert_Transform(testY, coin, 'C_LSTM', 'MINMAXSCALER')
            ds = Preprocessing.Invert_Transform(ohlcv_scaled, coin, 'OHLCV_LSTM', 'MINMAXSCALER')
            print(trainPredict.shape)
            print(testPredict.shape)
            print(trainX.shape)
            print(testX.shape)
            print(trainY.shape)
            print(testY.shape)
            trainPredict = pd.DataFrame(trainPredict)
            trainPredict.to_csv("trainPredict.csv")
            testPredict = pd.DataFrame(testPredict)
            testPredict.to_csv("testPredict.csv")
            trainX = pd.DataFrame(trainX)
            trainX.to_csv("trainX.csv")
            trainY = pd.DataFrame(trainY)
            trainY.to_csv("trainY.csv")
            testX = pd.DataFrame(testX)
            testX.to_csv("testX.csv")
            testY = pd.DataFrame(testY)
            testY.to_csv("testY.csv")
            ds = pd.DataFrame(ds)
            ds.to_csv("ds.csv")
            # Estimate model performance
            #trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
            #print('Train Score: %.2f RMSE' % (trainScore))
            #testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
            #print('Test Score: %.2f RMSE' % (testScore))

            # shift train predictions for plotting
            #trainPredictPlot, testPredictPlot = ModelPlot.Shift_Plot(ds, trainPredict, testPredict)
            # plot baseline and predictions
            #ModelPlot.Plot_Actual(ds, trainPredictPlot, testPredictPlot, coin, 'LSTM')
            # Saving model to disk
            #ModelParameters.Save_Model(lstm, coin, 'LSTM')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True
