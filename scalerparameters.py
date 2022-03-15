import os
import pickle

class ScalerParameters:

    # For the models: LR, KNN, CART, SVC, NB, PN, SGD, RF do nothing
    # For arch and garch use min max scaler
    # For other models use standard scaler
    def Save(coin, scaler, method, modelname):
        try:
            # Save the scaler to disk
            print('Saving the scaler to the disk')
            filename = os.path.join('C:/Users/pablo/PycharmProjects/Cass1e/scaler_parameters',
                                    str(coin['symbol']) + '_' + str(modelname) + '_' + method + '_scaler.pkl')
            pickle.dump(scaler, open(filename, 'wb'))
            print("Saved scaler to disk")
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return

    def Load(coin, modelname, method):
        try:
            scaler = pickle.load(open(os.path.join('C:/Users/pablo/PycharmProjects/Cass1e/scaler_parameters',
                                                   str(coin['symbol']) + '_' + str(modelname) + '_' + method +
                                                      '_scaler.pkl'), 'rb'))
            print('Scaler loaded correctly')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return scaler
