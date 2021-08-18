import os

# Disabling TF Debugging logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import load_model


def load_mdl(path):
    """
    A simple function to load Keras compatible .h5 model from the disk.
    :param path: Absolute path to the .h5 file
    :return: Keras model object
    """
    try:
        mdl = load_model(path, compile=False)
        return mdl
    except IOError:
        raise IOError("Incorrect model file path!")
