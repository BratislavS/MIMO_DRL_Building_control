import numpy as np

from dynamics.base_model import BaseDynamicsModel
from util.numerics import make_periodic
from data_processing.dataset import Dataset


class Periodic1DayModel(BaseDynamicsModel):
    """
    The 1-day periodic model that predicts the values as the ones
    from the last day.
    """

    # Member variables
    hist: np.ndarray
    pred_t: int = 0
    tot_t: int = 0

    def __init__(self, d: Dataset, exo_inds: np.ndarray = None, alpha: float = 0.1):
        """
        Initializes model.

        :param d: Dataset to use.
        :param exo_inds: The indices of the series that will be predicted.
        :param alpha: The decay parameter alpha.
        """
        name = "1DayPeriodic_Alpha" + str(alpha)
        super(Periodic1DayModel, self).__init__(d, name, exo_inds, exo_inds)

        # Save parameters
        self.data = d
        self.n: int = 60 * 24 // d.dt
        self.alpha: float = alpha

    def init_1day(self, day_data: np.ndarray) -> None:
        """
        Sets the history and resets time. Makes the history periodic
        for predictions longer than one day.

        :param day_data: New history data to use.
        """
        self.pred_t = 0
        self.tot_t = 0
        dat_copy = np.copy(day_data)
        n_feat = dat_copy.shape[-1]
        n_hists = dat_copy.shape[0]
        for k in range(n_feat):
            for i in range(n_hists):
                dat_copy[i, :, k] = make_periodic(dat_copy[i, :, k])
        self.hist = dat_copy

    def fit(self, verbose: int = 0, train_data: str = "train") -> None:
        """No need to fit anything.
        """
        pass

    def curr_alpha(self, t: int) -> float:
        """
        The decay function:
        :math:`\\alpha_t = e^{-\\alpha t} \\in [0, 1]`

        :param t: Time variable.
        :return: Current weight of the input.
        """
        return np.exp(-self.alpha * t)

    def predict(self, in_data: np.ndarray) -> np.ndarray:
        """
        Make predictions converging to the data in self.hist
        when predicting for multiple times in a row.
        Starts with predicting the last input.

        :param in_data: Prepared data.
        :return: New prediction.
        """

        # Update time
        self.pred_t = (self.pred_t + 1) % self.n
        self.tot_t += 1

        # Make prediction
        in_data = in_data[:, :, self.p_in_indices]
        curr_in = in_data[:, -1, :]
        curr_h = self.hist[:, self.pred_t, self.p_in_indices]
        curr_a = self.curr_alpha(self.tot_t)
        curr_out = curr_a * curr_in + (1.0 - curr_a) * curr_h
        return curr_out

    def disturb(self):
        """
        Returns a sample of noise of length n.
        """
        raise NotImplementedError("Disturbance for naive model not implemented!")
