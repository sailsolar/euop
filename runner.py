import sys
import json
import argparse

import numpy as np

import utils
import blackSholes as bS
from montecarlo import pricing


def nn_predict(path, sample):
    """
    A method handling the loading of Keras saved model and getting data processed from the loaded network

    Parameters
    ----------
    path : str
        absolute path to the saved .h5 file
    data : dict
        json object as an input sample to check the result
    """
    try:
        mdl = utils.load_mdl(path)

        # Loading data points
        a = sample["asset_price"]
        s = sample["strike"]
        v = sample["volatility"]
        rfr = sample["riskfree_interest_rate"]
        texp = sample["time_expiration"]

        # Formatting input as an numpy array
        X = np.array([[s, a, texp, v, rfr]])

        # Making prediction
        print(f'Neural network prediction price for this European Call option = {mdl.predict(X)[0][0]}')

    except Exception as e:
        print("Exception occurred while making prediction: {}".format(e))


def main(mode, data):
    """
    A function to return the option price depending upon the selected mode of prediction.

    Parameters
    ----------
    mode : str
        either of the three modes i.e.
        'bs' for Black-Scholes pricing
        'dl' for Neural Network based pricing
        'nm' for Numerical Method based pricing
    data : dict
        a valid json object

    Returns
    -------
        unit price according to the selected mode
    """

    if mode == 'bs':
        sample = data["bs"]
        try:
            value = bS.EuropeanCall(asset_price=sample["asset_price"],
                                    strike_price=sample["strike"],
                                    volatility=sample["volatility"],
                                    riskfree_factor=sample["riskfree_interest_rate"],
                                    time_to_expiry=sample["time_expiration"])
            print(f'Black-Scholes price for this European Call option = {value.price}')
        except KeyError:
            raise ValueError('Please check keys of input json data.')

    elif mode == 'dl':
        sample = data["neural_network"]
        path = sample["model_path"]
        nn_predict(path, sample)

    elif mode == 'nm':
        sample = data["numerical_method"]
        value = pricing(asset_price=sample["asset_price"],
                        strike=sample["strike"],
                        volatility=sample["volatility"],
                        time_to_expire=sample["time_expiration"])
        print(f'Monte-Carlo price for this European Call option = {value * 100}')

    else:
        print(f'Please specify an appropriate mode for pricing.')
        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', required=True, default='bs', type=str, dest='mode',
                        help="Mode of calculating the option price")
    parser.add_argument('--data', '-d', required=True, dest='data',
                        type=argparse.FileType(mode="r", encoding="utf-8"),
                        help="Path to json data file. Use '-' to read input from stdin")

    args = vars(parser.parse_args())

    mode = args['mode']
    data = args['data']

    d = json.loads(data.read())

    try:
        main(mode, d)
    except (BrokenPipeError, Exception) as p:
        print("Exception occurred in main: {}".format(p))
