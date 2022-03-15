import os
from keras.models import model_from_json

class ModelParameters:

    def Save_Model(model, coin, modelname):
        try:

            # Save the model to disk with YAML
            print('Saving the model to the disk')
            filename = os.path.join('C:/Users/pablo/PycharmProjects/Cass1e/model_parameters',
                                    str(coin['symbol']) + '_model_' + str(modelname) + '.json')
            model_json = model.to_json()
            with open(filename, "w") as json_file:
                json_file.write(model_json)
            # Serialize weights to HDF5
            filename2 = os.path.join('C:/Users/pablo/PycharmProjects/Cass1e/model_parameters',
                               str(coin['symbol']) + '_model_' + str(modelname) + '.h5')
            model.save_weights(filename2)
            print("Saved model to disk")
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def Load_Model(modelname, coin):
        try:
            # Load the model and scaler from disk
            filename = os.path.join('C:/Users/pablo/PycharmProjects/Cass1e/model_parameters',
                                    str(coin['symbol']) + '_model_' + str(modelname) + '.json')
            json_file = open(filename, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            # Load weights into new model
            filename2 = os.path.join('C:/Users/pablo/PycharmProjects/Cass1e/model_parameters',
                               str(coin['symbol']) + '_model_' + str(modelname) + '.h5')
            model.load_weights(filename2)
            print("Loaded model from disk")
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return model
