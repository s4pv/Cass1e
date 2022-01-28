import os
import pickle
from sklearn import preprocessing

class ScalerParameters:

    def save(dataset, coin, method, modelname):
        try:
            # Normalize the dataset
            if method == 'MINMAXSCALER':
                print('Starting to normalize the set with MinMaxScaler')
                scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
            elif method == 'SCALERSTANDARD':
                print('Starting to standardize the set with StandardScaler')
                scaler = preprocessing.StandardScaler()
            elif method == 'ROBUSTSCALER':
                scaler = preprocessing.RobustScaler()
            elif method == 'MAXABSSCALER':
                scaler = preprocessing.MaxAbsScaler()
            elif method == 'NORMALIZER':
                scaler = preprocessing.Normalizer()
            ds = scaler.fit_transform(dataset)
            # Save the scaler to disk
            print('Saving the scaler to the disk')
            filename = os.path.join('C:/Users/pablo/PycharmProjects/Cass1e/scaler_parameters',
                                    str(coin['symbol']) + '_' + str(modelname) + '_' + method + '_scaler.pkl')
            pickle.dump(scaler, open(filename, 'wb'))
            print("Saved scaler to disk")
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return ds, modelname

    def load(coin, method, modelname):
        try:
            # Load scaler into new model
            if modelname == 'LR' or modelname == 'KNN' or modelname == 'CART' or modelname == 'SVC'\
                    or modelname == 'NB' or modelname == 'PN' or modelname == 'SGD' or modelname == 'RF':
                scaler = 0
            else:
                scaler = pickle.load(open(os.path.join('C:/Users/pablo/PycharmProjects/Cass1e/scaler_parameters',
                                                   str(coin['symbol']) + '_' + str(modelname) + '_' + method +
                                                      '_scaler.pkl'), 'rb'))
            print('Scaler loaded correctly')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return scaler
