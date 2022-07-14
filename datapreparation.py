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

NO_DATA = parsed_config['model_options']['NO_DATA']


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

    def Fractional_Differentiation(dataset, weight):
        try:
            dataframe = pd.DataFrame(dataset)
            dataframe = Datapreparation.Reshape_Float(dataframe)
            dataframe = pd.DataFrame(dataframe)
            # ratio returns
            # dataframe[coin['symbol']] = dataframe / dataframe.shift(1) - 1
            # log returns (not for volume)
            # dataframe = numpy.log(dataframe/dataframe.shift(1))
            dataframe[1] = dataframe[1] - dataframe[1].shift(1) * weight
            dataframe[2] = dataframe[2] - dataframe[2].shift(1) * weight
            dataframe[3] = dataframe[3] - dataframe[3].shift(1) * weight
            dataframe[4] = dataframe[4] - dataframe[4].shift(1) * weight
            dataframe = dataframe.drop(index=0)
            dataframe.columns = ['volume', 'open', 'high', 'low', 'close']
        except Exception as e:
            print("An exception ocurred - {}".format(e))
            return False
        return dataframe

    def Fractional_Integration(returns, price_dataset, weight, type):
        try:
            prices = numpy.empty_like(returns)
            prices = pd.DataFrame(prices)
            df = pd.DataFrame(price_dataset)
            df = Datapreparation.Reshape_Float(df)
            df = pd.DataFrame(df)
            df.columns = ['volume', 'open', 'high', 'low', 'close']
            initial_price = df['close'][0]
            if type == 'forecast':
                final_price = df['close'][len(price_dataset)-1]
            elif type == 'model':
                final_price = df['close'][len(price_dataset) - N_STEPS_OUT]
            returns_df = pd.DataFrame(returns)
            returns_df = Datapreparation.Reshape_Float(returns_df)
            returns_df = pd.DataFrame(returns_df)
            # prices
            if len(returns) > N_STEPS_OUT:
                prices.iloc[0] = initial_price
            else:
                prices.iloc[0] = final_price
            for x in range(len(prices) - 1):
                prices.iloc[x + 1] = prices.iloc[x] * weight + returns_df.iloc[x + 1]
            prices.columns = ['close']
        except Exception as e:
            print("An exception ocurred - {}".format(e))
            return False
        return prices

    def Returns_Transformation(dataset):
        try:
            dataframe = pd.DataFrame(dataset)
            dataframe = Datapreparation.Reshape_Float(dataframe)
            dataframe = pd.DataFrame(dataframe)
            # ratio returns
            # dataframe[coin['symbol']] = dataframe / dataframe.shift(1) - 1
            # log returns (not for volume)
            #dataframe = numpy.log(dataframe/dataframe.shift(1))
            dataframe[1] = numpy.log(dataframe[1]/dataframe[1].shift(1))
            dataframe[2] = numpy.log(dataframe[2]/dataframe[2].shift(1))
            dataframe[3] = numpy.log(dataframe[3]/dataframe[3].shift(1))
            dataframe[4] = numpy.log(dataframe[4]/dataframe[4].shift(1))
            dataframe = dataframe.drop(index=0)
            dataframe.columns = ['volume', 'open', 'high', 'low', 'close']
        except Exception as e:
            print("An exception ocurred - {}".format(e))
            return False
        return dataframe

    def Price_Transformation(returns, price_dataset, type):
        try:
            prices = numpy.empty_like(returns)
            prices = pd.DataFrame(prices)
            df = pd.DataFrame(price_dataset)
            df = Datapreparation.Reshape_Float(df)
            df = pd.DataFrame(df)
            df.columns = ['volume', 'open', 'high', 'low', 'close']
            initial_price = df['close'][0]
            if type == 'forecast':
                final_price = df['close'][len(price_dataset)-1]
            elif type == 'model':
                final_price = df['close'][len(price_dataset)-1-N_STEPS_OUT]
            returns_df = pd.DataFrame(returns)
            returns_df = Datapreparation.Reshape_Float(returns_df)
            returns_df = pd.DataFrame(returns_df)
            # prices
            if len(returns) > N_STEPS_OUT:
                prices.iloc[0] = initial_price
            else:
                prices.iloc[0] = final_price
            for x in range(len(prices)-1):
                prices.iloc[x+1] = prices.iloc[x] * (1 + returns_df.iloc[x+1])
            prices.columns = ['close']
        except Exception as e:
            print("An exception ocurred - {}".format(e))
            return False
        return prices

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

    def Minmax_Scaler(dataset, model, coin, date):
        try:
            #print('Starting to scale the set with Min Max Scaler')
            scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
            ds = scaler.fit_transform(dataset)
            ScalerParameters.Save(coin, scaler, 'MINMAXSCALER', model, date)
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return ds

    def Standard_Scaler(dataset, model, coin, date):
        try:
            #print('Starting to scale the set with Standard Scaler')
            scaler = preprocessing.StandardScaler()
            ds = scaler.fit_transform(dataset)
            ScalerParameters.Save(coin, scaler, 'STANDARDSCALER', model, date)
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return ds

    def Maxabs_Scaler(dataset, model, coin, date):
        try:
            #print('Starting to scale the set with Maxabs Scaler')
            scaler = preprocessing.MaxAbsScaler()
            ds = scaler.fit_transform(dataset)
            ScalerParameters.Save(coin, scaler, 'MAXABSSCALER', model, date)
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return ds

    def Robust_Scaler(dataset, model, coin, date):
        try:
            #print('Starting to scale the set with Robust Scaler')
            scaler = preprocessing.RobustScaler(quantile_range=(25, 75))
            ds = scaler.fit_transform(dataset)
            ScalerParameters.Save(coin, scaler, 'ROBUSTSCALER', model, date)
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return ds

    def Yeo_Johnson_Scaler(dataset, model, coin, date):
        try:
            #print('Starting to scale the set with Yeo-Johnson Power Scaler')
            scaler = preprocessing.PowerTransformer(method="yeo-johnson")
            ds = scaler.fit_transform(dataset)
            ScalerParameters.Save(coin, scaler, 'YEOJOHNSONSCALER', model, date)
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return ds

    def Box_Cox_Scaler(dataset, model, coin, date):
        try:
            # print('Starting to scale the set with Box-Cox Power Scaler(data must be entirely positive)')
            scaler = preprocessing.PowerTransformer(method="box-cox")
            ds = scaler.fit_transform(dataset)
            ScalerParameters.Save(coin, scaler, 'BOXCOXSCALER', model, date)
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return ds

    def Uniform_Scaler(dataset, model, coin, date):
        try:
            # print('Starting to scale the set with Uniform Quantile Scaler')
            scaler = preprocessing.QuantileTransformer(output_distribution="uniform")
            ds = scaler.fit_transform(dataset)
            ScalerParameters.Save(coin, scaler, 'UNIFORMSCALER', model, date)
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return ds

    def Gaussian_Scaler(dataset, model, coin, date):
        try:
            # print('Starting to scale the set with Gaussian Quantile Scaler')
            scaler = preprocessing.QuantileTransformer(output_distribution="normal")
            ds = scaler.fit_transform(dataset)
            ScalerParameters.Save(coin, scaler, 'GAUSSIANSCALER', model, date)
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return ds

    def Normal_Scaler(dataset, model, coin, date):
        try:
           # print('Starting to scale the set with Normal Scaler')
           scaler = preprocessing.Normalizer()
           ds = scaler.fit_transform(dataset)
           ScalerParameters.Save(coin, scaler, 'NORMALSCALER', model, date)
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
            #print(ds_f_ohlcv.shape)
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
