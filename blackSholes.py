import math
from scipy.stats import norm


class EuropeanCall:
    """A simple class to implement the Black-Scholes Formula for valuing an European Call Option given the underlying
    asset is operating with the Black-Scholes assumptions.
    """

    @staticmethod
    def call_price(s, v, k, t_exp, rf):
        """
         Returns
         -------
            a float indicating the price of an European Basket Call option according to BS model

        Exception
        ------
        ValueError
        """
        try:
            b = math.exp(-rf * t_exp)
            x1 = math.log(s / (b * k)) + .5 * (v * v) * t_exp
            x1 = x1 / (v * (t_exp ** .5))
            z1 = norm.cdf(x1)
            z1 = z1 * s

            x2 = math.log(s / (b * k)) - .5 * (v * v) * t_exp
            x2 = x2 / (v * (t_exp ** .5))
            z2 = norm.cdf(x2)
            z2 = b * k * z2
            return float(z1 - z2)
        except Exception as e:
            print(f"Caught exception while evaluating call price: {e}")

    def __init__(self, asset_price, strike_price, volatility, time_to_expiry, riskfree_factor):
        """Calculate the Black-Scholes price of an European Basket Call Option

        Parameters
        ----------
        asset_price : float
            stock price at the moment
        strike_price : float
            strike price attached to the option
        volatility : float
            volatility or implied volatility of the under lying asset
        time_to_expiry : int
            time left to exercise the option
        riskfree_factor : float
            risk-free interest rate
        """
        self.s = asset_price
        self.v = volatility
        self.k = strike_price
        self.t_exp = time_to_expiry
        self.rf = riskfree_factor

        self.price = self.call_price(self.s, self.v, self.k, self.t_exp, self.rf)
