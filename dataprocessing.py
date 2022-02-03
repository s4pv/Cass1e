import numpy
from datetime import datetime
from pandas import DataFrame as Dataframe
from helper import Helper
from sklearn import preprocessing
import warnings
import pickle
from scalerparameters import ScalerParameters


warnings.filterwarnings("ignore")

parsed_config = Helper.load_config('config.yml')

NO_DAYS = parsed_config['model_options']['NO_DAYS']
TRAIN_SIZE = parsed_config['ml_options']['TRAIN_SIZE']
LOOK_BACK = parsed_config['ml_options']['LOOK_BACK']
N_FEATURES = parsed_config['ml_options']['N_FEATURES']

class DataProcessing:

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
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return final_dataframe

    # Look_back are timesteps

    def Create_Dataset(dataset, LOOK_BACK):
        dataX, dataY = [], []
        for i in range(len(dataset) - LOOK_BACK - 1):
            a = dataset[i:(i + LOOK_BACK), 0]
            dataX.append(a)
            dataY.append(dataset[i + LOOK_BACK, 0])
        return numpy.array(dataX), numpy.array(dataY)

    # fix random seed for reproducibility
    numpy.random.seed(7)

    def Reshape_Data(dataset, model):
        try:
            # Convert an array of values into a dataset matrix
            # Load the dataset
            print('Reshaping the data')
            dataframe = dataset['close']
            data = dataframe.values
            if model == 'GARCH' or model == 'ARCH':
                print('setting double')
                data = data.astype('double')
            elif model == 'LR' or model == 'KNN' or model == 'CART' or model == 'SVC' or model == 'NB' or model == 'PN'\
                    or model == 'SGD' or model == 'RF':
                print('setting int')
                lab_enc = preprocessing.LabelEncoder()
                data = lab_enc.fit_transform(data)
            else:
                print('setting float')
                data = data.astype('float32')
            print('reshaping')
            data = numpy.reshape(data, (len(data), 1))
            # data = numpy.reshape(data, (NO_DAYS, 1))
            # data = numpy.reshape(data, len(data))
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return data

    def Preprocess_Data(dataset, model, coin):
        try:
            # Normalize the dataset
            df = DataProcessing.Reshape_Data(dataset, model)
            if model == 'LR' or model == 'KNN' or model == 'CART' or model == 'SVC' or model == 'NB' or model == 'PN'\
                    or model == 'SGD' or model == 'RF':
                ds = df
            elif model == 'ARCH' or model == 'GARCH':
                ds, modelname = ScalerParameters.save(df, coin, 'SCALERSTANDARD', model)
            else:
                ds, modelname = ScalerParameters.save(df, coin, 'MINMAXSCALER', model)
            # Split into train and test sets. 1000 rows. 67% = training sample and 33% = test sample
            print('Splitting the dataset into training and test sets')
            train_size = int(len(ds) * TRAIN_SIZE)
            print(train_size)
            test_size = len(ds) - train_size
            print(test_size)

            train, test = ds[0:train_size, :], ds[train_size:len(ds), :len(ds)]
            print(train.shape)
            print(test.shape)

            # reshape into X=t and Y=t+1. Added variables for predictions
            print('Rearranging the datasets')
            trainX, trainY = DataProcessing.Create_Dataset(train, LOOK_BACK)
            testX, testY = DataProcessing.Create_Dataset(test, LOOK_BACK)
            print(trainX.shape)
            print(trainY.shape)
            print(testX.shape)
            print(testY.shape)

            # Reshape input to be [samples, time steps, features].
            # Features could be 5 if we want the whole candle for train and/or test: OHLCV

            # Samples
            #print('Samples')
            train_samples = trainX.shape[0]
            #print(train_samples)
            test_samples = testX.shape[0]

            # Timesteps
            #print('Timesteps')
            train_timesteps = trainX.shape[1]
            #print(train_timesteps)
            test_timesteps = testX.shape[1]

            # Memory between batches -> batch_size = n_features = 1.
            #print('final dataset shapes')
            #trainX = numpy.reshape(trainX, (train_samples, train_timesteps, N_FEATURES))
            trainX = numpy.reshape(trainX, (train_samples, N_FEATURES))
            testX = numpy.reshape(testX, (test_samples, N_FEATURES))
            #testX = numpy.reshape(testX, (test_samples, test_timesteps, N_FEATURES))
            #print('asdf')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return trainX, trainY, testX, testY, ds

    def Invert_Data(trainX, trainY, testX, testY, ds, coin, method, modelname):
        try:
            # Invert predictions. Added variables for predictions
            print('Inverting the scale back to normal')
            scaler = ScalerParameters.load(coin, method, modelname)
            trainX = scaler.inverse_transform(trainX)
            trainY = scaler.inverse_transform([trainY])
            testX = scaler.inverse_transform(testX)
            testY = scaler.inverse_transform([testY])
            df = scaler.inverse_transform(ds)
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return trainX, trainY, testX, testY, df
