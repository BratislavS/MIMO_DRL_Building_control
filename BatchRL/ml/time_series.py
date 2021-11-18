import numpy as np

from statsmodels.tsa.arima_model import ARIMA


class AR_Model(object):
    """
    AR Time Series Model.
    """

    def __init__(self, lag=1):
        self.lag = lag

        self.model = None
        self.model_fit = None

    def fit(self, data):
        self.model = ARIMA(data, order=(self.lag, 0, 0))
        self.model_fit = self.model.fit(trend='nc', disp=False)

    def predict(self, data):
        ar_coeff = self.model_fit.arparams
        sig = np.sqrt(self.model_fit.sigma2)
        y_hat = 0.0
        for i in range(1, len(ar_coeff) + 1):
            y_hat += ar_coeff[i - 1] * data[-i]
        return y_hat + np.random.normal(0, sig)
