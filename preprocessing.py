import numpy
from datetime import datetime
from pandas import DataFrame as Dataframe
from helper import Helper
from sklearn import preprocessing
import warnings
from scalerparameters import ScalerParameters


warnings.filterwarnings("ignore")

parsed_config = Helper.load_config('config.yml')

NO_DAYS = parsed_config['model_options']['NO_DAYS']
TRAIN_SIZE = parsed_config['ml_options']['TRAIN_SIZE']
LOOK_BACK = parsed_config['ml_options']['LOOK_BACK']
N_FEATURES = parsed_config['ml_options']['N_FEATURES']

class Preprocessing:

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
        try:
            dataX, dataY = [], []
            for i in range(len(dataset) - LOOK_BACK - 1):
                a = dataset[i:(i + LOOK_BACK), 0]
                dataX.append(a)
                dataY.append(dataset[i + LOOK_BACK, 0])
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return numpy.array(dataX), numpy.array(dataY)


    # For the models: LR, KNN, CART, SVC, NB, PN, SGD, RF
    def Reshape_Int(dataset):
        try:
            # Convert an array of values into a dataset matrix
            # Load the dataset
            print('Reshaping the data to integer')
            dataframe = dataset['close']
            data = dataframe.values
            print('setting int')
            lab_enc = preprocessing.LabelEncoder()
            data = lab_enc.fit_transform(data)
            data = numpy.reshape(data, (len(data), 1))
            # data = numpy.reshape(data, (NO_DAYS, 1))
            # data = numpy.reshape(data, len(data))
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return data

    # For the arch and garch models only
    def Reshape_Double(dataset):
        try:
            # Convert an array of values into a dataset matrix
            # Load the dataset
            print('Reshaping the data to double')
            dataframe = dataset['close']
            data = dataframe.values
            print('setting double')
            data = data.astype('double')
            data = numpy.reshape(data, (len(data), 1))
            # data = numpy.reshape(data, (NO_DAYS, 1))
            # data = numpy.reshape(data, len(data))
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return data

    # For other models use
    def Reshape_Float(dataset):
        try:
            # Convert an array of values into a dataset matrix
            # Load the dataset
            print('Reshaping the data to float')
            dataframe = dataset['close']
            data = dataframe.values
            print('setting float')
            data = data.astype('float32')
            data = numpy.reshape(data, (len(data), 1))
            # data = numpy.reshape(data, (NO_DAYS, 1))
            # data = numpy.reshape(data, len(data))
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return data

    # For others models only
    def Minmax_Scaler(dataset, model, coin):
        try:
            print('Starting to normalize the set with Min Max Scaler')
            scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
            ds = scaler.fit_transform(dataset)
            ScalerParameters.Save(coin, scaler, 'MINMAXSCALER', model)
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return ds

    # For the models: LR, KNN, CART, SVC, NB, PN, SGD, RF do nothing

    # For the arch and garch models only
    def Scaler_Standard(dataset, model, coin):
        try:
            print('Starting to normalize the set with Scaler Standard')
            scaler = preprocessing.StandardScaler()
            ds = scaler.fit_transform(dataset)
            ScalerParameters.Save(coin, scaler, 'SCALERSTANDARD', model)
            print('Model saved')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return ds

    def Robust_Scaler(dataset, model, coin):
        try:
            print('Starting to normalize the set with Robust Scaler')
            scaler = preprocessing.RobustScaler()
            ds = scaler.fit_transform(dataset)
            ScalerParameters.Save(coin, scaler, 'ROBUSTSCALER', model)
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return ds

    def Maxabs_Scaler(dataset, model, coin):
        try:
            print('Starting to normalize the set with Max Abs Scaler')
            scaler = preprocessing.MaxAbsScaler()
            ds = scaler.fit_transform(dataset)
            ScalerParameters.Save(coin, scaler, 'MAXABSSCALER', model)
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return ds

    def Normalizer_Scaler(dataset, model, coin):
        try:
            print('Starting to normalize the set with Normalizer Scaler')
            scaler = preprocessing.Normalizer()
            ds = scaler.fit_transform(dataset)
            ScalerParameters.Save(coin, scaler, 'NORMALIZER', model)
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return ds

    def Dataset_Split(dataset):
        try:
            print('Splitting the dataset into training and test sets')
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


    def Reshape_Data(train, test):
        try:

            # reshape into X=t and Y=t+1. Added variables for predictions
            print('Rearranging the datasets')
            trainX, trainY = Preprocessing.Create_Dataset(train, LOOK_BACK)
            testX, testY = Preprocessing.Create_Dataset(test, LOOK_BACK)
            #print(trainX.shape)
            #print(trainY.shape)
            print(testX.shape)
            print(testY.shape)

            # Reshape input to be [samples, time steps, features].
            # Features could be 5 if we want the whole candle for train and/or test: OHLCV

            # Samples
            print('Samples')
            train_samples = trainX.shape[0]
            print(train_samples)
            test_samples = testX.shape[0]
            print(test_samples)

            # Timesteps
            print('Timesteps')
            train_timesteps = trainX.shape[1]
            print(train_timesteps)
            test_timesteps = testX.shape[1]
            print(test_timesteps)

            # Memory between batches -> batch_size = n_features = 1.
            print('final dataset shapes')
            trainX = numpy.reshape(trainX, (train_samples, train_timesteps, N_FEATURES))
            #trainX = numpy.reshape(trainX, (train_samples, train_timesteps))
            print(trainX.shape)
            #testX = numpy.reshape(testX, (test_samples, train_timesteps))
            testX = numpy.reshape(testX, (test_samples, test_timesteps, N_FEATURES))
            print(testX.shape)
            print(trainY.shape)
            print(testY.shape)
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return trainX, trainY, testX, testY

    def Reshape_Data_ARCH(train, test):
        try:

            # reshape into X=t and Y=t+1. Added variables for predictions
            print('Rearranging the datasets')
            trainX, trainY = Preprocessing.Create_Dataset(train, LOOK_BACK)
            testX, testY = Preprocessing.Create_Dataset(test, LOOK_BACK)
            #print(trainX.shape)
            #print(trainY.shape)
            print(testX.shape)
            print(testY.shape)

            # Reshape input to be [samples, time steps, features].
            # Features could be 5 if we want the whole candle for train and/or test: OHLCV

            # Samples
            print('Samples')
            train_samples = trainX.shape[0]
            print(train_samples)
            test_samples = testX.shape[0]
            print(test_samples)

            # Timesteps
            print('Timesteps')
            train_timesteps = trainX.shape[1]
            print(train_timesteps)
            test_timesteps = testX.shape[1]
            print(test_timesteps)

            # Memory between batches -> batch_size = n_features = 1.
            print('final dataset shapes')
            #trainX = numpy.reshape(trainX, (train_samples, train_timesteps, N_FEATURES))
            trainX = numpy.reshape(trainX, (train_samples, train_timesteps))
            print(trainX.shape)
            testX = numpy.reshape(testX, (test_samples, train_timesteps))
            #testX = numpy.reshape(testX, (test_samples, test_timesteps, N_FEATURES))
            print(testX.shape)
            print(trainY.shape)
            print(testY.shape)
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return trainX, trainY, testX, testY

    def Invert_Transform(ds, coin, modelname, method):
        try:
            # Invert predictions. Added variables for predictions
            print('Inverting the scale back to normal')
            scaler = ScalerParameters.Load(coin, modelname, method)
            db = scaler.inverse_transform(ds)
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return db
