import os
import pickle
from helper import Helper

# Configuration and class variables
parsed_config = Helper.load_config('config.yml')

SCALER_DATE = parsed_config['scaler_parameters_options']['SCALER_DATE']

class ScalerParameters:

    def Save(coin, scaler, method, modelname, date):
        try:
            # Save the scaler to disk
            #print('Saving the scaler to the disk')
            filedir = 'scaler_parameters/' + str(date) + '/'
            filename = os.path.join(filedir, str(coin['symbol']) + '_' + str(modelname) + '_' + method + '_.pkl')
            os.makedirs(filedir, exist_ok=True)
            pickle.dump(scaler, open(filename, 'wb'))
            #print("Saved scaler to disk")
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return

    def Load(coin, modelname, method):
        try:
            scaler = pickle.load(open(os.path.join('scaler_parameters/' + SCALER_DATE + '/',
                                                   str(coin['symbol']) + '_' + str(modelname) + '_' + method +
                                                      '_.pkl'), 'rb'))
            #print('Scaler loaded correctly')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return scaler
