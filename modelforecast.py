import warnings

from helper import Helper
from sklearn import preprocessing
from scalerparameters import ScalerParameters
from datapreparation import Datapreparation
from modelplot import ModelPlot
from modelparameters import ModelParameters

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

class ModelForecast:
    def Predict_LSTM(dataset, coin):
        try:
            # preprocess data
            ds_f_c, ds_f_ohlv, ds_f_ohlcv = Datapreparation.Prepare_Data(dataset)
            # scaling data without saving it
            #print('Starting to normalize the set with Min Max Scaler')
            scaler_ohlcv = preprocessing.MinMaxScaler(feature_range=(0, 1))
            ohlcv_scaled = scaler_ohlcv.fit_transform(ds_f_ohlcv)
            scaler_ohlv = preprocessing.MinMaxScaler(feature_range=(0, 1))
            ohlv_scaled = scaler_ohlv.fit_transform(ds_f_ohlv)
            scaler_c = preprocessing.MinMaxScaler(feature_range=(0, 1))
            c_scaled = scaler_c.fit_transform(ds_f_c)
            #print(ohlcv_scaled.shape)
            #print(ohlv_scaled.shape)
            #print(c_scaled.shape)
            # for multivariate multi step predict purposes
            ohlv, c = Datapreparation.Split_Sequences(ohlcv_scaled, N_STEPS_IN, N_STEPS_OUT)
            # load previously fitted model
            lstm = ModelParameters.Load_Model('LSTM2', 'BTCUSDT')
            # compile loaded model on test data
            lstm.compile(loss='mean_squared_error', optimizer='adam')
            # evaluate loaded model
            lstm.evaluate(ohlv, c, batch_size=N_STEPS_IN, verbose=VERBOSE)
            # make predictions
            forecast = lstm.predict(ohlv, batch_size=N_STEPS_IN)
            #print(forecast.shape)
            # invert sets
            forecast = forecast[len(forecast)-1]
            #print(forecast.shape)
            ohlv = ohlv[len(ohlv) - 1]
            forecast = scaler_c.inverse_transform(forecast)
            ohlcv = scaler_ohlcv.inverse_transform(ohlcv_scaled)
            ohlv_ = scaler_ohlv.inverse_transform(ohlv_scaled)
            c_ = scaler_c.inverse_transform(c_scaled)
            ohlv = scaler_ohlv.inverse_transform(ohlv)
            c = scaler_c.inverse_transform(c)
            # shift train predictions for plotting
            forecastPlot, cPlot = ModelPlot.Shift_Forecast_Plot(c_, forecast)
            # plot baseline and predictions
            ModelPlot.Plot_Forecast(cPlot, forecastPlot, coin, 'LSTM2')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return forecast