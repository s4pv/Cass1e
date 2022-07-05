import warnings

from helper import Helper
from sklearn import preprocessing
from datapreparation import Datapreparation
from modelplot import ModelPlot
from modelparameters import ModelParameters
from stats import Stats
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# Configuration and class variables
parsed_config = Helper.load_config('config.yml')

TRAIN_SIZE = parsed_config['ml_options']['TRAIN_SIZE']
NEURONS = parsed_config['ml_options']['NEURONS']
DROPOUT = parsed_config['ml_options']['DROPOUT']
N_STEPS_IN = parsed_config['ml_options']['N_STEPS_IN']
N_STEPS_OUT = parsed_config['ml_options']['N_STEPS_OUT']
EPOCH = parsed_config['ml_options']['EPOCH']
N_FEATURES = parsed_config['ml_options']['N_FEATURES']
VERBOSE = parsed_config['ml_options']['VERBOSE']

ML_MODEL = parsed_config['model_options']['ML_MODEL']
WEIGHT = parsed_config['model_options']['WEIGHT']

class ModelForecast:
    def Predict_LSTM(dataset, coin):
        try:
            # statistical tests
            print('Transforming the data to returns to make the data stationary')
            #return_dataset = Datapreparation.Returns_Transformation(dataset)
            return_dataset = Datapreparation.Fractional_Differentiation(dataset, WEIGHT)
            print('Testing if the data is stationary on close data')
            statsDF, pDF = Stats.Dickey_Fuller(return_dataset['close'])
            # preprocessing data
            r_c_unscaled, r_ohlv_unscaled, r_ohlcv_unscaled = Datapreparation.Prepare_Data(return_dataset)
            # scaling data without saving it
            print('Starting to normalize the set with the quantile gaussian transformation (robust to outliers)')
            scaler_ohlcv = preprocessing.QuantileTransformer(output_distribution="normal")
            #scaler_ohlcv = preprocessing.PowerTransformer(method="yeo-johnson")
            r_ohlcv_scaled = scaler_ohlcv.fit_transform(r_ohlcv_unscaled)
            scaler_c = preprocessing.QuantileTransformer(output_distribution="normal")
            #scaler_c = preprocessing.PowerTransformer(method="yeo-johnson")
            r_c_scaled = scaler_c.fit_transform(r_c_unscaled)
            # reshaping data
            r_ohlcv_scaled_2 = pd.DataFrame(r_ohlcv_scaled)
            r_ohlcv_scaled_2.columns = ['volume', 'open', 'high', 'low', 'close']
            # plot histogram, probplot and qqplot
            #print('Starting to plot histogram, PPplot and QQplot on close data')
            #Stats.Plots(r_ohlcv_scaled_2['close'])
            # testing for normality
            print('Testing for normality on close data sets.')
            statsSW, pSW = Stats.Shapiro_Wilk(r_ohlcv_scaled_2['close'])
            # Split sequences
            r_ohlv_scaled_2, r_c_scaled_2 = Datapreparation.Split_Sequences(r_ohlcv_scaled, N_STEPS_IN, N_STEPS_OUT)
            # load previously fitted model
            lstm = ModelParameters.Load_Model(ML_MODEL, 'BTCUSDT')
            # compile loaded model on test data
            lstm.compile(loss='mean_squared_error', optimizer='adam')
            # evaluate loaded model
            lstm.evaluate( r_ohlv_scaled_2, r_c_scaled_2, batch_size=N_STEPS_IN, verbose=VERBOSE)
            # make predictions
            r_forecast_scaled = lstm.predict(r_ohlv_scaled_2, batch_size=N_STEPS_IN)
            # reshape sets
            r_forecast_scaled = r_forecast_scaled[len(r_forecast_scaled)-1].reshape(N_STEPS_OUT, -1)
            #print(r_forecast_scaled.shape)
            #print(r_ohlcv_scaled.shape)
            #print(r_ohlv_scaled_2.shape)
            #print(r_c_scaled_2.shape)
            # invert sets
            r_forecast_unscaled = scaler_c.inverse_transform(r_forecast_scaled)
            # convert back from returns to price
            #p_forecast_unscaled = Datapreparation.Price_Transformation(r_forecast_unscaled, dataset, 'forecast')
            #p_c_unscaled = Datapreparation.Price_Transformation(r_c_unscaled, dataset, 'forecast')
            # convert back from fractional differentiation to price
            p_forecast_unscaled = Datapreparation.Fractional_Integration(r_forecast_unscaled, dataset, WEIGHT, 'forecast')
            p_c_unscaled = Datapreparation.Fractional_Integration(r_c_unscaled, dataset, WEIGHT, 'forecast')
            # reshape dataset to plot correctly close data
            p_c_unscaled_2 = dataset['close'].to_numpy()
            p_c_unscaled_2 = np.reshape(p_c_unscaled_2, (len(p_c_unscaled_2), 1))
            # shift train predictions for plotting
            forecastPlot, cPlot = ModelPlot.Shift_Forecast_Plot(p_c_unscaled_2, p_forecast_unscaled)
            # plot baseline and predictions
            ModelPlot.Plot_Forecast(cPlot, forecastPlot, coin, ML_MODEL)
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return p_forecast_unscaled