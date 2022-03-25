import numpy
from datetime import datetime
import pandas as pd
from pandas import DataFrame as Dataframe
from helper import Helper
from sklearn import preprocessing
import warnings
from scalerparameters import ScalerParameters

warnings.filterwarnings("ignore")

parsed_config = Helper.load_config('config.yml')

TRAIN_SIZE = parsed_config['ml_options']['TRAIN_SIZE']
N_FEATURES = parsed_config['ml_options']['N_FEATURES']
N_STEPS_IN = parsed_config['ml_options']['N_STEPS_IN']
N_STEPS_OUT = parsed_config['ml_options']['N_STEPS_OUT']


class Datapreparation:

    def OHLCV_DataFrame(dataset):
        try:
            print('Creating an OHLCV dataframe from Binance')
            candles_dataframe = Dataframe(dataset)
            candles_dataframe_date = candles_dataframe[0]
            final_date = []
            for time in candles_dataframe_date.unique():
                readable = datetime.fromtimestamp(int(time / 1000))
                final_date.append(readable)
            candles_dataframe.pop(0)
            candles_dataframe.pop(6)
            candles_dataframe.pop(7)
            candles_dataframe.pop(8)
            candles_dataframe.pop(9)
            candles_dataframe.pop(10)
            candles_dataframe.pop(11)
            dataframe_final_date = Dataframe(final_date)
            dataframe_final_date.columns = ['date']
            final_dataframe = candles_dataframe.join(dataframe_final_date)
            final_dataframe.set_index('date', inplace=True)
            final_dataframe.columns = ['open', 'high', 'low', 'close', 'volume']
            final_dataframe = final_dataframe.reindex(['volume', 'open', 'high', 'low', 'close'], axis=1)
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return final_dataframe


    # split a multivariate sequence into samples
    def Split_Sequences(sequences, N_STEPS_IN, N_STEPS_OUT):
        try:
            X, y = list(), list()
            for i in range(len(sequences)):
                # find the end of this pattern
                end_ix = i + N_STEPS_IN
                out_end_ix = end_ix + N_STEPS_OUT - 1
                # check if we are beyond the dataset
                if out_end_ix > len(sequences):
                    break
                # gather input and output parts of the pattern
                seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1:out_end_ix, -1]
                X.append(seq_x)
                y.append(seq_y)
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return numpy.array(X), numpy.array(y)


    def Reshape_Float(dataset):
        try:
            # Convert an array of values into a dataset matrix
            # Load the dataset
            #print('Reshaping the data to float')
            data = dataset.values
            #print('setting float')
            data = data.astype('float32')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return data

    def Minmax_Scaler(dataset, model, coin):
        try:
            #print('Starting to normalize the set with Min Max Scaler')
            scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
            ds = scaler.fit_transform(dataset)
            ScalerParameters.Save(coin, scaler, 'MINMAXSCALER', model)
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return ds

    def Dataset_Split(dataset):
        try:
            #print('Splitting the dataset into training and test sets')
            train_size = int(len(dataset) * TRAIN_SIZE)
            #print(train_size)
            test_size = len(dataset) - train_size
            #print(test_size)
            train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
            #print(train.shape)
            #print(test.shape)
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return train, test

    def Prepare_Data(dataset):
        try:
            # preprocess data into 2 sets to scale and invert without problems
            ds_c = dataset['close']
            ds_ohlcv = pd.DataFrame(dataset)
            ds_ohlv = ds_ohlcv.drop(columns=['close'])
            ds_f_ohlcv = Datapreparation.Reshape_Float(ds_ohlcv)
            ds_f_ohlv = Datapreparation.Reshape_Float(ds_ohlv)
            ds_f_c = Datapreparation.Reshape_Float(ds_c)
            # data to array
            ds_f_ohlcv = numpy.array(ds_f_ohlcv)
            ds_f_ohlv = numpy.array(ds_f_ohlv)
            ds_f_c = numpy.array(ds_f_c).reshape(-1, 1)
            #print(ds_f_ohlv.shape)
            #print(ds_f_c.shape)
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return ds_f_c, ds_f_ohlv, ds_f_ohlcv


    def Invert_Transform(ds, coin, modelname, method):
        try:
            # Invert predictions. Added variables for predictions
            #print('Inverting the scale back to normal')
            scaler = ScalerParameters.Load(coin, modelname, method)
            db = scaler.inverse_transform(ds)
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return db
