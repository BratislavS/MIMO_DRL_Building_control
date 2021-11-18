"""The Battery Model is defined here.

In the class :class:`BatteryModel`.
"""
import os
from typing import Callable, Tuple

import numpy as np

from data_processing.dataset import Dataset
from dynamics.base_model import BaseDynamicsModel
from util.numerics import fit_linear_bf_1d, npf32, trf_mean_and_std
from util.util import print_if_verb, yeet, Num
from util.visualize import scatter_plot, OVERLEAF_IMG_DIR, basic_plot, LONG_FIG_SIZE


def clean_battery_dataset(ds: Dataset) -> None:
    """Removes the ugly data in the battery dataset.

    TODO: Do this by considering the data specifying when
        the experiments were...
    """
    n = ds.data.shape[0]
    up_lim = int(0.6 * n)
    low_lim = int(0.3 * n)
    ds.data[low_lim: up_lim] = np.nan


# Fit pw. linear model: $y = \alpha_1 + \alpha_2 * x * \alpha_3 * max(0, x)$
def pw_lin_fun_factory(cont_at_zero: bool = True, through_zero: bool = False) -> Callable:
    fun_list = [
        lambda x: x,
        lambda x: np.maximum(0.0, x),
    ]

    # Add more functions depending on arguments
    if not cont_at_zero:
        fun_list += [lambda x: np.where(x > 0, 1.0, 0.0)]
    if not through_zero:
        fun_list = [lambda x: 1.0] + fun_list

    # Count the number of functions
    n_fun = len(fun_list)

    # Define and return feature function
    def feat_fun(x: float):
        if np.isscalar(x):
            x = np.array([x])
        n_x = len(x)
        res = npf32((n_x, n_fun))
        for ct, f in enumerate(fun_list):
            res[:, ct] = f(x)
        return res

    return feat_fun


class BatteryModel(BaseDynamicsModel):
    """Dynamics model of the battery.

    The model of the battery:
    :math:`s_t`: SoC at time t
    Model:
    :math:`s_{t+1} = s_t + \\eta(s_t) p_{t+1}`,
    :math:`p_{t+1}`:
    average charging power from time t to t+1 (control input)
    """

    hard_soc_limits: Tuple[Num, Num] = (10.0, 90.0)  #: SoC limit for prediction.
    _scaled_soc_limits: Tuple[Num, Num]

    _feat_fun: Callable = None  #: The function specifying the features.
    params: np.ndarray = None  #: Parameters of the pw linear model.

    # The data to plot after fitting
    p: np.ndarray = None
    ds: np.ndarray = None
    soc: np.ndarray = None

    masked_p: np.ndarray = None
    masked_ds: np.ndarray = None
    masked_soc: np.ndarray = None
    init_data: Tuple[np.ndarray, ...]
    nan_mask: np.ndarray
    exp_mask: np.ndarray

    _scatter_plot_size = LONG_FIG_SIZE

    def __init__(self, dataset: Dataset, base_ind: int = None):
        """Initializes the battery model with the specified dataset.

        Args:
            dataset: Dataset with data to fit modes.
        """
        in_inds, out_inds = None, None
        if base_ind is not None:
            in_inds = np.array([base_ind, base_ind + 1], dtype=np.int32)
            out_inds = np.array([base_ind], dtype=np.int32)
        super().__init__(dataset, dataset.name,
                         out_inds=out_inds,
                         in_inds=in_inds)

        # Define feature function
        self._feat_fun = pw_lin_fun_factory(cont_at_zero=True)

        # Scale soc limits
        if dataset.fully_scaled:
            scale = np.copy(dataset.scaling[self.in_inds[0]])
            self._scaled_soc_limits = tuple(trf_mean_and_std(np.array(self.hard_soc_limits),
                                                             scale, remove=True))
        else:
            self._scaled_soc_limits = self.hard_soc_limits

    def fit(self, verbose: int = 0, train_data: str = "train") -> None:
        """Fits the battery model.

        Does nothing if it has already been fitted.
        `predict` throws an error if the model wasn't fitted
        before calling it.

        Args:
            verbose: Verbosity, 0: silent.
            train_data: Training data set.
        """
        self.fit_data = train_data
        if self.params is not None:
            print_if_verb(verbose, "Battery model already fitted!")
            return
        else:
            print_if_verb(verbose, "Fitting battery model...")

        # Get data
        d = self.data
        dat = d.data  # split_dict[train_data].get_rel_data()

        # Extract data
        s_ind, p_ind = self.in_inds
        p = np.copy(dat[1:, p_ind])
        soc = np.copy(dat[:-1, s_ind])
        ds = np.copy(dat[1:, s_ind] - soc)
        self.init_data = p, ds, soc

        # Remove nans
        not_nans = np.logical_not(np.logical_or(np.isnan(p), np.isnan(ds)))
        self.nan_mask = not_nans
        p, ds, soc = (dat[not_nans] for dat in [p, ds, soc])
        self.p, self.ds, self.soc = p, ds, soc

        # Reduce data to exclude strange part
        exclude: bool = False
        if exclude:
            n_p = len(p)
            n = n_p // 3
            m = np.zeros((n_p,), dtype=np.bool)
            m[:n] = True
            m[-n:] = True
            self.masked_p, self.masked_ds = p[m], ds[m]
            self.masked_soc = soc[m]
            self.exp_mask = m
        else:
            self.masked_p, self.masked_ds = p, ds
            self.masked_soc, self.exp_mask = soc, np.ones(self.masked_p.shape, dtype=np.bool)

        # Fit parameters
        params = fit_linear_bf_1d(self.masked_p, self.masked_ds, self._feat_fun)
        self.params = params

        # Remove outliers based on fit
        fitted_ds = self._eval_at(self.masked_p)
        errs = np.abs(fitted_ds - self.masked_ds)
        thresh = np.max(errs) / 3.4  # This is heuristic
        self.masked_p = self.masked_p[errs < thresh]
        self.masked_ds = self.masked_ds[errs < thresh]

        # Filter the full data
        full_p = self.data.data[1:, 1]
        full_ds = self.data.data[1:, 0] - self.data.data[:-1, 0]
        errs = np.abs(self._eval_at(full_p) - full_ds)
        data_m1 = self.data.data[:-1, :]
        data_m1[errs > thresh] = np.nan
        self.data.data = data_m1
        self.data.split_data()

        # Update params
        params = fit_linear_bf_1d(self.masked_p, self.masked_ds, self._feat_fun)
        self.params = params

    def predict(self, in_data: np.ndarray) -> np.ndarray:
        """Make predictions using the fitted model on the provided data.

        Args:
            in_data: Prepared data.

        Returns:
            Predictions
        """
        p = np.copy(in_data[:, -1, 1])
        s_t = np.copy(in_data[:, -1, 0])

        # Evaluate model and clip to limits
        s_tp1 = s_t + self._eval_at(p)
        s_tp1 = np.clip(s_tp1, *self._scaled_soc_limits)
        return s_tp1.reshape((-1, 1))

    def disturb(self):
        """Returns a sample of noise.
        """
        return 0

    def _eval_at(self, p):
        """Evaluates the model for a given active power `p`."""
        if self.params is None:
            yeet("Need to fit battery model first!")
        res = 0
        f_eval = self._feat_fun(p)
        for ct, p in enumerate(self.params):
            res += p * f_eval[:, ct]
        return res

    def _get_plot_name(self, base: str, put_on_ol: bool = False):
        ol_name = os.path.join(OVERLEAF_IMG_DIR, base)
        return ol_name if put_on_ol else self.get_plt_path(base)

    @staticmethod
    def _get_labs(set_title: bool = False):
        return {'title': 'Battery model' if set_title else None,
                'xlab': 'Active power [kW]',
                'ylab': r'$\Delta$ SoC [%]'}

    def plot_all_data(self, put_on_ol: bool = False,
                      overwrite: bool = False, set_title: bool = False):
        """Plots all the data."""
        # Define plot labels
        labs = self._get_labs(set_title)

        # Get scaling
        d = self.data
        scale = np.copy(d.scaling[self.in_inds])
        scale[0, 0] = 0.0

        # Check if file exists
        before_plt_path = self._get_plot_name("WithOutliers", put_on_ol)
        if not os.path.isfile(before_plt_path + ".pdf") or overwrite:
            # Plot data
            scatter_plot(self.p, self.ds, lab_dict=labs,
                         show=False,
                         m_and_std_x=scale[1],
                         m_and_std_y=scale[0],
                         add_line=True,
                         save_name=before_plt_path,
                         fig_size=self._scatter_plot_size)

    def analyze_bat_model(self, put_on_ol: bool = False,
                          overwrite: bool = False) -> None:
        """This is basically the fit method, but it also
        does some data analysis and makes some battery data specific plots.
        """
        if self.fit_data is None:
            self.fit()

        # Get scaling
        d = self.data
        scale = np.copy(d.scaling[self.in_inds])
        scale[0, 0] = 0.0

        # Define plot labels
        labs = {'title': '',
                'xlab': 'Active power [kW]',
                'ylab': r'$\Delta$ SoC [%]'}

        # Plot residuals vs. SoC
        res_plt_path = self._get_plot_name("ResVsSoc", put_on_ol)
        res_time_plt_path = self._get_plot_name("ResVsTime", put_on_ol)
        if not os.path.isfile(res_plt_path + ".pdf") or overwrite or \
                not os.path.isfile(res_time_plt_path + ".pdf"):
            labs_res = {"title": "", "xlab": "State of charge [%]",
                        "ylab": r"Residuals ($\Delta$ SoC [%])"}
            p_orig, _, soc_orig = self.init_data
            res = soc_orig[:-1] + self._eval_at(p_orig[:-1]) - soc_orig[1:]
            last_bool = self.nan_mask[-1]
            n_used = len(self.exp_mask) - last_bool
            filtered_res = res[self.nan_mask[:-1]][self.exp_mask[:n_used]]

            # Plot data
            n_soc = len(self.masked_soc) - last_bool
            scatter_plot(self.masked_soc[:n_soc], filtered_res, lab_dict=labs_res,
                         show=False,
                         m_and_std_x=d.scaling[self.in_inds[0]],
                         m_and_std_y=scale[0],
                         add_line=True,
                         save_name=res_plt_path,
                         fig_size=self._scatter_plot_size)

            # Plot residuals vs. time
            time_labs = ("Timestep", "Residuals")
            basic_plot(None, filtered_res, res_time_plt_path,
                       time_labs,
                       fig_size=self._scatter_plot_size)

        # Check if plot already exists
        after_plt_path = self._get_plot_name("Cleaned", put_on_ol)
        if os.path.isfile(after_plt_path + ".pdf") and not overwrite:
            return

        # Eval for pw linear line
        x_pw_line = np.array([np.min(self.p), -0.001, 0, 0.001, np.max(self.p)], dtype=np.float32)
        y_pw_line = self._eval_at(x_pw_line)

        # Plot model
        scatter_plot(self.masked_p, self.masked_ds, lab_dict=labs,
                     show=False,
                     add_line=False,
                     m_and_std_x=scale[1],
                     m_and_std_y=scale[0],
                     custom_line=[x_pw_line, y_pw_line],
                     custom_label='Piece-wise linear fit',
                     save_name=after_plt_path,
                     fig_size=self._scatter_plot_size)
