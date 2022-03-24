from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import math
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy

from helper import Helper
from scalerparameters import ScalerParameters
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
            # preprocess data
            ds_f_c, ds_f_ohlv, ds_f_ohlcv = Preprocessing.Prepare_Data(dataset)
            # scaling data
            scaler_ohlcv = ScalerParameters.Load(coin, 'OHLCV_LSTM', 'MINMAXSCALER')
            ohlcv_scaled = scaler_ohlcv.fit_transform(ds_f_ohlcv)
            scaler_ohlv = ScalerParameters.Load(coin, 'OHLV_LSTM', 'MINMAXSCALER')
            ohlv_scaled = scaler_ohlv.fit_transform(ds_f_ohlv)
            scaler_c = ScalerParameters.Load(coin, 'C_LSTM', 'MINMAXSCALER')
            c_scaled = scaler_c.fit_transform(ds_f_c)
            print(ohlcv_scaled.shape)
            print(ohlv_scaled.shape)
            print(c_scaled.shape)
            # for multivariate multi step predict purposes
            ohlv, c = Preprocessing.Split_Sequences(ohlcv_scaled, N_STEPS_IN, N_STEPS_OUT)
            # load previously fitted model
            lstm = ModelParameters.Load_Model('LSTM2', coin)
            # evaluate loaded model on test data
            lstm.compile(loss='mean_squared_error', optimizer='adam')
            # make predictions
            forecast = lstm.predict(ohlv, batch_size=N_STEPS_IN)
            print(forecast.shape)
            # invert sets
            forecast = forecast[len(forecast)-1]
            print(forecast.shape)
            ohlv = ohlv[len(ohlv) - 1]
            forecast = Preprocessing.Invert_Transform(forecast, coin, 'C_LSTM', 'MINMAXSCALER')
            ohlcv = Preprocessing.Invert_Transform(ohlcv_scaled, coin, 'OHLCV_LSTM', 'MINMAXSCALER')
            ohlv_ = Preprocessing.Invert_Transform(ohlv_scaled, coin, 'OHLV_LSTM', 'MINMAXSCALER')
            c_ = Preprocessing.Invert_Transform(c_scaled, coin, 'C_LSTM', 'MINMAXSCALER')
            ohlv = Preprocessing.Invert_Transform(ohlv, coin, 'OHLV_LSTM', 'MINMAXSCALER')
            c = Preprocessing.Invert_Transform(c, coin, 'C_LSTM', 'MINMAXSCALER')
            # shift train predictions for plotting
            forecastPlot, cPlot = ModelPlot.Shift_Forecast_Plot(c_, forecast)
            # plot baseline and predictions
            ModelPlot.Plot_Forecast(cPlot, forecastPlot, coin, 'LSTM')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True