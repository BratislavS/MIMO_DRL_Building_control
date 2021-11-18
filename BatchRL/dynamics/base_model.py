"""Defines the interface for ML models.

As an abstract class: :class:`BaseDynamicsModel`.
"""
import os
import warnings
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple, List, Type

import numpy as np

from data_processing.dataset import Dataset
from ml.keras_util import KerasBase
from ml.time_series import AR_Model
from util.numerics import add_mean_and_std, rem_mean_and_std, copy_arr_list, get_shape1, npf32, \
    save_performance_extended, get_metrics_eval_save_name_list, ErrMetric, MSE, find_inds
from util.util import create_dir, mins_to_str, Arr, tot_size, yeet, DEFAULT_TRAIN_SET, DEFAULT_EVAL_SET
from util.visualize import plot_dataset, model_plot_path, plot_residuals_acf, \
    OVERLEAF_IMG_DIR, plot_visual_all_in_one, LONG_FIG_SIZE

#: Plot title definition
CONT_TITLE: str = "One week continuous predictions"


def check_train_str(train_data: str) -> None:
    """Check if training data string is valid."""
    if train_data not in ["train", "train_val", "all"]:
        yeet(f"String specifying training data: {train_data} is not valid!")


def _get_title(n: int, dt: int = 15) -> str:
    """Plot title generating helper function."""
    return CONT_TITLE if n == 0 else mins_to_str(dt * n) + " ahead predictions"


def get_plot_ds(s, tr: Optional[np.ndarray], d: Dataset, orig_p_ind: np.ndarray,
                n_offs: int = 0) -> Dataset:
    """
    Creates a dataset with truth time series tr and parameters
    from the original dataset d. Intended for plotting after
    the first column of the data is set to the predicted series.

    Args:
        n_offs: Number of time steps to offset t_init in the new dataset.
        s: Shape of data of dataset.
        tr: Ground truth time series.
        d: Original dataset.
        orig_p_ind: Prediction index relative to the original dataset.

    Returns:
        Dataset with ground truth series as second series.
    """
    plot_data = np.empty((s[0], 2), dtype=np.float32)
    if tr is not None:
        plot_data[:, 1] = tr
    scaling, is_scd = d.get_scaling_mul(orig_p_ind[0], 2)
    actual_dt = d.t_init if n_offs == 0 else d.get_shifted_t_init(n_offs + d.seq_len - 1)
    analysis_ds = Dataset(plot_data,
                          d.dt,
                          actual_dt,
                          scaling,
                          is_scd,
                          ['Prediction', 'Ground truth'])
    return analysis_ds


def _get_inds_str(indices: np.ndarray, pre: str = "In") -> str:
    inds_str = ""
    if indices is not None:
        inds_str = "_" + pre + '-'.join(map(str, indices))
    return inds_str


def check_model_compatibility(model_list: List['BaseDynamicsModel'],
                              raise_error: bool = True) -> bool:
    """Checks if models in list are compatible.

    I.e. they need to predict the same series.

    Args:
        model_list: List with models.
        raise_error: Whether to raise an error instead of returning a bool.

    Returns:
        Bool indicating whether models are compatible.

    Raises:
        ValueError: If models are not compatible and `raise_error` is True
        AssertionError: If the `model_list` has length 0.
    """
    n_mods = len(model_list)
    assert n_mods > 0, "No models provided!"
    inds = model_list[0].out_inds
    compat = True
    for m in model_list:
        curr_inds = m.out_inds
        if not np.array_equal(inds, curr_inds):
            compat = False
            if raise_error:
                raise ValueError(f"Indices {curr_inds} of model {m} not compatible "
                                 f"with those of first model: {inds}")
    return compat


class BaseDynamicsModel(KerasBase, ABC):
    """This class describes the interface of a ML-based (partial) dynamics model.
    """

    # Constants
    N_LAG: int = 6
    debug: bool = True
    use_AR: bool = True

    # Parameters
    verbose: int = 1  #: Verbosity level
    name: str  #: Name of the model
    plot_path: str  #: Path to the plot folder

    #: Dataset containing all the data
    data: Dataset  #: Reference to the underlying dataset.
    out_inds: np.ndarray
    p_out_inds: np.ndarray
    in_inds: np.ndarray
    p_in_indices: np.ndarray
    n_pred_full: int  #: Number of non-control variables in dataset.
    n_pred: int  #: Number of dimensions of the prediction

    fit_data: str = None  #: Subset of data the model was trained on, set by fit()

    # Disturbance variables
    modeled_disturbance: bool = False
    res_std: Arr = None
    dist_mod = None
    init_pred: np.ndarray = None

    def __init__(self, ds: Dataset, name: str,
                 out_inds: np.ndarray = None,
                 in_inds: np.ndarray = None,
                 verbose: int = None):
        """Constructor for the base of every dynamics model.

        If `out_inds` is None, all series are predicted.
        If `in_inds` is None, all series are used as input to the model.

        Args:
            ds: Dataset containing all the data.
            name: Name of the model.
            out_inds: Indices specifying the series in the data that the model predicts.
            in_inds: Indices specifying the series in the data that the model takes as input.
            verbose: The verbosity level.
        """

        # Set dataset
        self.data = ds

        # Verbosity
        if verbose is not None:
            self.verbose = verbose

        # Set up indices
        out_inds = self._get_inds(out_inds, ds, False)
        self.out_inds, self.p_out_inds = out_inds
        for k in ds.c_inds:
            if k in self.out_inds:
                raise IndexError("You cannot predict control indices!")
        in_inds = self._get_inds(in_inds, ds, True)
        self.in_inds, self.p_in_indices = in_inds
        self.p_out_in_indices = find_inds(self.p_in_indices, self.p_out_inds)

        # Set name
        self.name = self._get_full_name(name)

        self.n_pred = len(self.out_inds)
        self.n_pred_full = ds.d - ds.n_c

        self.plot_path = os.path.join(model_plot_path, self.name)

    @staticmethod
    def _get_inds(indices: Optional[np.ndarray], ds: Dataset, in_inds: bool = True):
        """Prepares the indices."""
        if indices is None:
            indices = np.arange(ds.d)
            if not in_inds:
                indices = ds.from_prepared(indices[:-ds.n_c])
        p_indices = ds.to_prepared(indices)
        ds.check_inds(indices, True)
        return indices, p_indices

    def _extract_output(self, input_arr: np.ndarray) -> np.ndarray:
        in_sh = input_arr.shape
        assert len(in_sh) >= 2, f"Invalid shape {in_sh}"
        assert np.max(self.p_out_in_indices) < in_sh[-1], \
            f"Last shape too small: {in_sh}, indices: {self.p_out_in_indices}!"
        return input_arr[..., -1, self.p_out_in_indices]

    def _get_full_name(self, base_name: str):
        return self._get_full_name_static(self.data, self.out_inds, self.in_inds, base_name)

    @staticmethod
    def _get_full_name_static(data: Dataset, out_inds: np.ndarray,
                              in_inds: np.ndarray,
                              b_name: str,
                              no_data: bool = False):
        """Argghhh, duplicate code here... But where is the duplicate part?"""
        out_inds, _ = BaseDynamicsModel._get_inds(out_inds, data, False)
        in_inds, _ = BaseDynamicsModel._get_inds(in_inds, data, True)
        ind_str = _get_inds_str(out_inds, "Out") + _get_inds_str(in_inds)
        str_out = f"{ind_str}_MODEL_{b_name}"
        if not no_data:
            str_out = data.name + str_out
        return str_out

    @abstractmethod
    def fit(self, verbose: int = 0, train_data: str = DEFAULT_TRAIN_SET) -> None:
        pass

    @abstractmethod
    def predict(self, in_data):
        pass

    def init_1day(self, day_data: np.ndarray) -> None:
        """Initializer for models that need more previous data than `seq_len` time steps.

        Deprecated!

        Args:
            day_data: The data of one day to initialize model.
        """
        pass

    def get_fit_data(self, data_name: str = "train", *, seq_out: bool = False,
                     residual_output: bool = False):
        """Returns the required data for fitting the model
        taking care of choosing the right series by indexing.

        Args:
            data_name: The string specifying which portion of the data to use.
            seq_out: Whether to return the full output sequences.
            residual_output: Whether to subtract the previous state from the output.

        Returns:
            The input and output data for supervised learning.
        """
        in_dat, out_dat, n = self.data.get_split(data_name, seq_out)
        res_in_dat = in_dat[:, :, self.p_in_indices]
        res_out_dat_out = out_dat[..., self.p_out_inds]
        if residual_output:
            res_out_dat_out -= in_dat[:, -1, self.p_out_inds]
        return res_in_dat, res_out_dat_out

    def rescale_output(self, arr: np.ndarray,
                       out_put: bool = True,
                       whiten: bool = False) -> np.ndarray:
        """Transforms an array back to having / not having original mean and std.

        If `out_put` is True, then the data in `arr` is assumed to
        lie in the output space of the model, else it should lie in the
        input space. If `whiten` is true, the mean and the std, as computed
        in the Dataset, is removed, else added.

        Args:
            arr: The array with the data to transform.
            out_put: Whether `arr` is in the output space of the model.
            whiten: Whether to remove the mean and std, else add it.

        Returns:
            Array with transformed data.

        Raises:
            ValueError: If the last dimension does not have the right size.
        """
        # Determine transform function and indices
        trf_fun = rem_mean_and_std if whiten else add_mean_and_std
        inds = self.out_inds if out_put else self.in_inds

        # Check dimension
        n_feat = len(inds)
        if n_feat != arr.shape[-1]:
            raise ValueError(f"Last dimension must be {n_feat}!")

        # Scale the data
        arr_scaled = np.copy(arr)
        for ct, ind in enumerate(inds):
            if self.data.is_scaled[ind]:
                mas = self.data.scaling[ind]
                arr_scaled[..., ct] = trf_fun(arr_scaled[..., ct], mean_and_std=mas)

        return arr_scaled

    def model_disturbance(self, data_str: str = 'train') -> None:
        """Models the uncertainties in the model.

        It is done by matching the distribution of the residuals.
        Either use the std of the residuals and use Gaussian noise
        with the same std or fit an AR process to each series.

        Args:
            data_str: The string determining the part of the data for fitting.
        """

        # Compute residuals
        residuals = self.get_residuals(data_str)
        self.modeled_disturbance = True

        if self.use_AR:
            # Fit an AR process for each output dimension
            self.dist_mod = [AR_Model(lag=self.N_LAG) for _ in range(self.n_pred)]
            for k, d in enumerate(self.dist_mod):
                d.fit(residuals[:, k])
            self.reset_disturbance()

        self.res_std = np.std(residuals, axis=0)

    def disturb(self) -> np.ndarray:
        """Returns a sample of noise.

        Returns:
            Numpy array of disturbances.

        Raises:
            AttributeError: If the disturbance model was not fitted before.
        """

        # Check if disturbance model was fitted
        if not self.modeled_disturbance:
            raise AttributeError('Need to model the disturbance first!')

        # Compute next noise
        if self.use_AR:
            next_noise = np.empty((self.n_pred,), dtype=np.float32)
            for k in range(self.n_pred):
                next_noise[k] = self.dist_mod[k].predict(self.init_pred[:, k])

            self.init_pred[:-1, :] = self.init_pred[1:, :]
            self.init_pred[-1, :] = next_noise
            return next_noise
        return np.random.normal(0, 1, self.n_pred) * self.res_std

    def reset_disturbance(self) -> None:
        """Resets the disturbance to zero.

        Returns:
            None
        """
        self.init_pred = np.zeros((self.N_LAG, self.n_pred), dtype=np.float32)

    def n_step_predict(self, prepared_data: Sequence, n: int, *,
                       pred_ind: int = None,
                       return_all_predictions: bool = False,
                       disturb_pred: bool = False) -> np.ndarray:
        """Applies the model n times and returns the predictions.

        TODO: Make it work with any prediction indices!?

        Args:
            prepared_data: Data to predict.
            n: Number of timesteps to predict.
            pred_ind: Which series to predict, all if None.
            return_all_predictions: Whether to return intermediate predictions.
            disturb_pred: Whether to apply a disturbance to the prediction.

        Returns:
            The predictions.
        Raises:
            ValueError: If `n` < 0 or `n` too large or if the prepared data
                does not have the right shape.
        """

        in_data, out_data = copy_arr_list(prepared_data)

        # Get shapes
        n_pred = len(self.out_inds)
        n_tot = self.data.d
        n_samples = in_data.shape[0]
        n_feat = n_tot - self.data.n_c
        n_out = n_samples - n + 1

        # Prepare indices
        if pred_ind is None:
            # Predict all series in out_inds
            orig_pred_inds = np.copy(self.out_inds)
            out_inds = np.arange(n_pred)
        else:
            warnings.warn("Deprecated!")
            # Predict pred_ind'th series only
            mod_pred_ind = self.out_inds[pred_ind]
            orig_pred_inds = np.array([mod_pred_ind], dtype=np.int32)
            out_inds = np.array([pred_ind], dtype=np.int32)
        prep_pred_inds = self.data.to_prepared(orig_pred_inds)

        # Do checks
        if n < 1:
            raise ValueError(f"n: ({n}) has to be larger than 0!")
        if n_out <= 0:
            raise ValueError(f"n: ({n}) too large")
        if in_data.shape[0] != out_data.shape[0]:
            raise ValueError("Shape mismatch of prepared data.")
        if in_data.shape[-1] != n_tot or out_data.shape[-1] != n_feat:
            raise ValueError("Not the right number of dimensions in prepared data!")

        # Initialize values and reserve output array
        all_pred = None
        if return_all_predictions:
            all_pred = np.empty((n_out, n, n_pred))
        curr_in_data = np.copy(in_data[:n_out])
        curr_pred = None

        # Predict continuously
        for k in range(n):
            # Predict
            rel_in_dat, _ = self.get_rel_part(np.copy(curr_in_data))
            curr_pred = self.predict(rel_in_dat)
            if disturb_pred:
                curr_pred += self.disturb()
            if return_all_predictions:
                all_pred[:, k] = np.copy(curr_pred)

            # Construct next data
            curr_in_data[:, :-1, :] = np.copy(curr_in_data[:, 1:, :])
            curr_in_data[:, -1, :n_feat] = np.copy(out_data[k:(n_out + k), :])
            if k != n - 1:
                curr_in_data[:, -1, n_feat:] = np.copy(in_data[(k + 1):(n_out + k + 1), -1, n_feat:])
            else:
                curr_in_data[:, -1, n_feat:] = 0
            curr_in_data[:, -1, prep_pred_inds] = np.copy(curr_pred[:, out_inds])

        # Return
        if return_all_predictions:
            return all_pred
        return curr_pred

    def get_plt_path(self, name: str) -> str:
        """Specifies the path of the plot with name 'name' where it should be saved.

        If there is not a directory
        for the current model, it is created.

        Args:
            name: Name of the plot.

        Returns:
            Full path of the plot file.
        """
        dir_name = self.plot_path
        create_dir(dir_name)
        return os.path.join(dir_name, name)

    def get_predicted_plt_data(self, predict_data: Tuple, name_list: List[str], title: str,
                               const_steps: int = 0, n_ts_off: int = 0) -> List[Tuple]:

        # Whether to predict continuously for one week
        one_week = const_steps <= 0

        # Get data
        d = self.data
        in_d, out_d = predict_data
        s = in_d.shape

        # Continuous prediction
        n_ts = s[0] if one_week else const_steps
        full_pred = self.n_step_predict([in_d, out_d], n_ts,
                                        pred_ind=None,
                                        return_all_predictions=one_week)

        n_pred = len(self.out_inds)
        all_plt_dat = []
        for k in range(n_pred):
            # Construct dataset
            k_orig = self.out_inds[k]
            k_orig_arr = np.array([k_orig])
            k_prep = self.data.to_prepared(k_orig_arr)[0]
            new_ds = get_plot_ds(s, np.copy(out_d[:, k_prep]), d, k_orig_arr, n_ts_off)

            # Set the predicted data
            if one_week:
                new_ds.data[:, 0] = np.copy(full_pred[0, :, k])
            else:
                new_ds.data[(n_ts - 1):, 0] = np.copy(full_pred[:, k])
                new_ds.data[:(n_ts - 1), 0] = np.nan

            # Find descriptions and add data
            desc = d.descriptions[k_orig]
            title_and_ylab = [title, desc]
            all_plt_dat += [(new_ds, title_and_ylab, name_list[k])]

        return all_plt_dat

    def plot_one_week_analysis(self, predict_data,
                               const_steps: int = 0, *,
                               ext: str = None,
                               overwrite: bool = False,
                               base: str = None,
                               combine_plots: bool = False,
                               put_on_ol: bool = False,
                               add_errors: bool = False,
                               n_ts_off: int = 0,
                               series_mask: List[int] = None,
                               ):
        """Plots predictions of the model over one week.

        If `const_steps` = 0, the predictions are done in a continuous way,
        else, a fixed number of steps, i.e. `const_steps`, is predicted.

        Args:
            predict_data: Data to use for making plots.
            const_steps: The number of steps to predict if fixed number,
                continuous predictions if `const_steps` = 0.
            ext: String extension for the filename.
            n_ts_off: Time step offset of data used.
            overwrite: Whether to overwrite existing plot files.
            combine_plots: Whether to combine the plots of all series in one file.
            base: The base name of the output file.
            put_on_ol: Whether to save the plots on Overleaf instead.
            add_errors: Whether to add errors as additional legends.
            series_mask: Used to select subset of series to plot., only
                used if `combine_plots` is True.
        """
        continuous_pred = const_steps <= 0

        dt = self.data.dt
        time_str = mins_to_str(dt * const_steps)

        # Define title
        # title = _get_title(const_steps, self.data.dt)
        title = ""

        # Setup base name
        if base is None:
            base = "OneWeek" if continuous_pred else time_str + 'Ahead'
            if combine_plots:
                base += "Combined"

        # Check if plot file already exists
        _, ex = self._get_one_week_plot_name(base, ext, 0, put_on_ol)
        if not overwrite and ex:
            return

        name_list = [self._get_one_week_plot_name(base, ext, k, put_on_ol)[0]
                     for k in range(len(self.out_inds))]
        all_plt_dat = self.get_predicted_plt_data(predict_data, name_list,
                                                  const_steps=const_steps,
                                                  title=title, n_ts_off=n_ts_off)

        # Plot all the things
        if not combine_plots:
            for ds, t, cn in all_plt_dat:
                plot_dataset(ds,
                             show=False,
                             title_and_ylab=t,
                             save_name=cn,
                             fig_size=LONG_FIG_SIZE)
        else:
            tot_save_name, _ = self._get_one_week_plot_name(base, ext, 0, put_on_ol)
            plot_visual_all_in_one(all_plt_dat, tot_save_name, add_errors,
                                   series_mask=series_mask,
                                   fig_size=LONG_FIG_SIZE)

        return all_plt_dat

    def _get_plt_or_ol_path(self, full_b_name: str, put_on_ol: bool = False):
        if put_on_ol:
            curr_name = os.path.join(OVERLEAF_IMG_DIR, full_b_name)
        else:
            curr_name = self.get_plt_path(full_b_name)
        return curr_name

    def _get_one_week_plot_name(self, base: str, ext: str = None,
                                ind: int = None, put_on_ol: bool = False):
        ext = "" if ext is None else ext
        full_b_name = f"{base}_{ind}_{ext}"
        curr_name = self._get_plt_or_ol_path(full_b_name, put_on_ol)
        exists = os.path.isfile(curr_name + ".pdf")
        return curr_name, exists

    def get_fit_data_ext(self):
        data_str = self._get_fit_data_set()
        return f"_TS_{data_str}" if data_str != DEFAULT_TRAIN_SET else ""

    def _acf_plot_path(self, i: int = 0,
                       add_ext: bool = True, partial: bool = False):
        """Creates the plot path for the acf plot file."""
        p = "P" if partial else ""
        dat_ext = self.get_fit_data_ext()
        ext = ".pdf" if add_ext else ""
        res_str = f"Res{p}ACF_{i}{dat_ext}{ext}"
        return self.get_plt_path(res_str)

    def _get_fit_data_set(self, assert_set: bool = True) -> str:
        f_dat = self.fit_data
        if assert_set:
            assert f_dat is not None, f"Model {self.name} of class " \
                                      f"{self.__class__.__name__} not yet fitted!"
        return f_dat

    def analyze_visually(self, plot_acf: bool = True,
                         n_steps: Sequence = (1, 24),
                         overwrite: bool = False,
                         verbose: bool = True,
                         base_name: str = None,
                         save_to_ol: bool = False,
                         one_file: bool = False,
                         add_errors: bool = False,
                         eval_parts: List[str] = None,
                         series_mask: List[int] = None,
                         use_other_plot_function=False):
        """Analyzes the trained model.

        Makes some plots using the fitted model and the streak data.
        Also plots the acf and the partial acf of the residuals.

        Args:
            plot_acf: Whether to plot the acf of the residuals.
            n_steps: The list with the number of steps for `const_nts_plot`.
            overwrite: Whether to overwrite existing plot files.
            verbose: Whether to print info to console.
            base_name: The base name to give to the plots.
            save_to_ol: Whether to save the prediction plots to Overleaf.
            one_file: Whether to plot all series in one file.
            add_errors: Whether to add errors in a box. Do not do this!
            eval_parts: Evaluation dataset parts.
            series_mask: Used to select subset of series to plot, only used
                if `one_file` is True.
        """
        if verbose:
            print(f"Analyzing model {self.name}")
        d = self.data
        data_str = self._get_fit_data_set()

        # Check input
        assert np.all(np.array(n_steps) >= 0), f"Negative timestep found in: {n_steps}"

        # Get residuals and plot autocorrelation
        if plot_acf:
            # Check if file already exist.
            first_acf_name = self._acf_plot_path()

            if overwrite or not os.path.isfile(first_acf_name):
                res = self.get_residuals(data_str)
                for k in range(get_shape1(res)):
                    acf_name = self._acf_plot_path(i=k, add_ext=False)
                    plot_residuals_acf(res[:, k], name=acf_name)
                    pacf_name = self._acf_plot_path(i=k, add_ext=False, partial=True)
                    plot_residuals_acf(res[:, k], name=pacf_name, partial=True)

        # Define the extension string lists for naming
        dat_ext = self.get_fit_data_ext()
        if eval_parts is None:
            eval_parts = ["train", "val"]
        ext_list = [f"EV_" + e.capitalize() + dat_ext for e in eval_parts]

        # Do the same for train and validation set
        all_data = []
        for ct, p_str in enumerate(eval_parts):
            dat_1, dat_2, n = d.get_streak(p_str)
            dat = [dat_1, dat_2]
            dat_copy = copy_arr_list(dat)

            # Plot for fixed number of time-steps
            base_kwargs = {
                'overwrite': overwrite,
                'put_on_ol': save_to_ol,
                'combine_plots': one_file,
                'add_errors': add_errors,
                'n_ts_off': n,
                'ext': ext_list[ct],
                'series_mask': series_mask,
            }
            # Plot fixed number of step predictions
            for n_s in n_steps:
                curr_b_name = None if base_name is None else base_name + str(n_s)
                all_data += [self.plot_one_week_analysis(dat_copy, n_s,
                                                         base=curr_b_name,
                                                         **base_kwargs)]

            # Plot for continuous predictions
            if use_other_plot_function:
                pass
            else:
                all_data += [self.plot_one_week_analysis(copy_arr_list(dat), 0,
                                                         base=base_name,
                                                         **base_kwargs)]

        return all_data

    def analyze_performance(self, n_steps: Sequence = (1, 4, 20),
                            verbose: int = 0,
                            overwrite: bool = False,
                            metrics: Sequence[Type[ErrMetric]] = (MSE,),
                            n_days: int = 14,
                            parts: List = None) -> None:
        """Analyzes the multistep prediction performance of the model.

        Uses the metrics provided by `metrics`.

        Args:
            n_steps: The list with the timesteps to predict.
            verbose: Whether to output to console.
            overwrite: Whether to overwrite existing files.
            metrics: A sequence of metric functions that can be applied to two arrays.
            n_days: Length of sequence to perform analysis.
            parts:
        """
        # Print to console
        if verbose:
            print(f"Analyzing performance of model {self.name}.")

        # Specify the parts of the data to use
        if parts is None:
            parts = ["train", "val"]

        # Get the data
        d = self.data
        data_str = self._get_fit_data_set()

        # Create file names
        save_names = get_metrics_eval_save_name_list(parts, d.dt, data_str)
        save_names = [self.get_plt_path(s) for s in save_names]

        # Check if file already exists
        if not overwrite:
            found_all = True
            for f in save_names:
                if not os.path.isfile(f):
                    found_all = False
            if found_all:
                if verbose:
                    print("Performance evaluation already done.")
                return

        # Performance values
        n_sets = len(parts)
        n_pred = len(self.out_inds)
        n_n_steps = len(n_steps)
        n_metrics = len(metrics)
        perf_values = npf32((n_sets, n_pred, n_metrics, n_n_steps))

        for part_ind, p_str in enumerate(parts):

            # Get relevant data
            # dat_1, dat_2, n = d.get_streak(p_str, use_max_len=True)
            dat_1, dat_2, n = d.get_streak(p_str, n_days=n_days)
            in_d, out_d = np.copy(dat_1), np.copy(dat_2)

            # Compute n-step predictions
            for step_ct, n_ts in enumerate(n_steps):

                # Predict
                full_pred = self.n_step_predict(copy_arr_list([in_d, out_d]), n_ts, pred_ind=None)

                # Plot all
                for series_ind in range(n_pred):
                    # Handle indices
                    k_orig = self.out_inds[series_ind]
                    k_orig_arr = np.array([k_orig])
                    k_prep = self.data.to_prepared(k_orig_arr)[0]

                    # Extract prediction and ground truth
                    gt = out_d[(n_ts - 1):, k_prep]
                    pred = full_pred[:, series_ind]

                    # Compute performance metrics
                    for m_id, m in enumerate(metrics):
                        perf = m.err_fun(gt, pred)
                        perf_values[part_ind, series_ind, m_id, step_ct] = perf

        # Save performances
        met_names = [m.__name__ for m in metrics]
        save_performance_extended(perf_values, n_steps, save_names, met_names)

    def analyze_disturbed(self,
                          ext: str = None,
                          data_str: str = "val",
                          n_trials: int = 25) -> None:
        """Analyses the model using noisy predictions.

        Makes a plot by continuously predicting with
        the fitted model and comparing it to the ground
        truth. If predict_ind is None, all series that can be
        predicted are predicted simultaneously and each predicted
        series is plotted individually.

        Args:
            data_str: The string specifying the data to use.
            n_trials: Number of predictions with noise to average.
            ext: String extension for the filename.

        Returns: None
        """

        # Model the disturbance
        self.model_disturbance("train")

        # Get the data
        d = self.data
        dat_val = d.get_streak(data_str)
        in_dat_test, out_dat_test, n_ts_off = dat_val

        # Predict without noise
        s = in_dat_test.shape
        full_pred = self.n_step_predict([in_dat_test, out_dat_test], s[0],
                                        pred_ind=None,
                                        return_all_predictions=True)

        # Predict with noise
        s_pred = full_pred.shape
        all_noise_preds = np.empty((n_trials, s_pred[1], s_pred[2]), dtype=full_pred.dtype)
        for k in range(n_trials):
            all_noise_preds[k] = self.n_step_predict([in_dat_test, out_dat_test], s[0],
                                                     pred_ind=None,
                                                     return_all_predictions=True,
                                                     disturb_pred=True)
        mean_noise_preds = np.mean(all_noise_preds, axis=0)
        std_noise_preds = np.std(all_noise_preds, axis=0)

        # Build dataset
        plot_data = np.empty((s_pred[1], 5), dtype=np.float32)
        actual_dt = d.get_shifted_t_init(n_ts_off)
        descs = ['Prediction',
                 'Ground Truth',
                 'Mean Noisy Prediction',
                 'Noisy Prediction +2 STD.',
                 'Noisy Prediction -2 STD.']
        ext = "_" if ext is None else "_" + ext

        for k in range(self.n_pred):
            # Construct dataset and plot
            k_orig = self.out_inds[k]
            k_prep = self.data.to_prepared(np.array([k_orig]))[0]
            scaling, is_scd = d.get_scaling_mul(k_orig, 5)
            plot_data[:, 0] = np.copy(full_pred[0, :, k])
            plot_data[:, 1] = np.copy(out_dat_test[:, k_prep])
            mean_pred = mean_noise_preds[:, k]
            std_pred = std_noise_preds[:, k]
            plot_data[:, 2] = np.copy(mean_pred)
            plot_data[:, 3] = np.copy(mean_pred + 2 * std_pred)
            plot_data[:, 4] = np.copy(mean_pred - 2 * std_pred)

            desc = d.descriptions[k_orig]
            title_and_ylab = ['One week predictions', desc]
            analysis_ds = Dataset(plot_data,
                                  d.dt,
                                  actual_dt,
                                  scaling,
                                  is_scd,
                                  descs)
            plot_dataset(analysis_ds,
                         show=False,
                         title_and_ylab=title_and_ylab,
                         save_name=self.get_plt_path(f"OneWeek_WithNoise_{k}_{ext}"))

    def hyper_obj(self, n_ts: int = 24, series_ind: Arr = None,
                  eval_data: str = DEFAULT_EVAL_SET) -> float:
        """Defines the objective for the hyperparameter optimization.

        Uses multistep prediction to define the performance for
        the hyperparameter optimization. Objective needs to be minimized.

        Args:
            n_ts: Number of timesteps to predict.
            series_ind: The indices of the series to predict.
            eval_data: Evaluation set for the optimization.

        Returns:
            The numerical value of the objective.
        """
        d = self.data

        # Transform indices and get data
        if isinstance(series_ind, (int, float)):
            series_ind = d.to_prepared(np.array([series_ind]))
        elif series_ind is None:
            series_ind = self.p_out_inds
        in_d, out_d, _ = d.get_streak(eval_data, use_max_len=True)
        tr = np.copy(out_d[:, series_ind])

        # Predict and compute residuals
        one_h_pred = self.n_step_predict(copy_arr_list([in_d, out_d]), n_ts,
                                         pred_ind=None)
        residuals = tr[(n_ts - 1):] - one_h_pred
        tot_s = tot_size(one_h_pred.shape)
        return float(np.sum(residuals * residuals)) / tot_s

    def get_rel_part(self, in_dat: np.ndarray,
                     out_dat: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the relevant data series from input and output data.

        Args:
            in_dat: Full input data.
            out_dat: Full output data.

        Returns:
            Reduced input and output data.
        """
        rel_in_dat = in_dat[..., self.p_in_indices]
        if out_dat is not None:
            rel_out_dat = out_dat[..., self.p_out_inds]
        else:
            rel_out_dat = None
        return rel_in_dat, rel_out_dat

    def get_residuals(self, data_str: str):
        """Computes the residuals using the fitted model.

        Args:
            data_str: String defining which part of the data to use.

        Returns:
            Residuals.
        """
        input_data, output_data, n = self.data.get_split(data_str)
        rel_in_dat, rel_out_dat = self.get_rel_part(input_data, output_data)
        residuals = self.predict(rel_in_dat) - rel_out_dat
        return residuals

    def deb(self, *args) -> None:
        """Prints Debug Info to console.

        Args:
            args: Arguments as for print() function.
        """
        if self.debug:
            print(*args)


def compare_models(model_list: List[BaseDynamicsModel],
                   save_name: str, *,
                   n_steps: Tuple[int, ...] = (1,),
                   part_spec: str = "val",
                   model_names: List[str] = None,
                   overwrite: bool = False) -> None:
    """Compares all the models in the list visually.

    All steps in `n_steps` need to be positive, if 0,
    continuous prediction over one week is evaluated.

    Args:
        model_list: List with all models.
        save_name: Name of the plot file.
        n_steps: The sequence with the numbers of prediction steps to evaluate.
        part_spec: The string specifying the part of the data, e.g. "val" or "train".
        model_names: The simplified model name showing up in the plot.
        overwrite: Whether to overwrite existing files.
    """
    # Count models
    n_models = len(model_list)
    n_plot_series = n_models + 1
    assert n_models > 0, "No models provided!"
    if model_names is not None:
        assert len(model_names) == n_models, "Incorrect number of model names!"

    # Get and check data
    m_0 = model_list[0]
    d, n_pred = m_0.data, len(m_0.out_inds)
    for m in model_list:
        assert m.data.name == d.name, "Model data is not compatible!"

    # Extract data
    dat_1, dat_2, n = d.get_streak(part_spec)
    dat = (dat_1, dat_2)

    # Plot fixed number of step predictions
    title = ""
    names = ["" for _ in range(n_pred)]

    for n_s in n_steps:
        curr_save_name = save_name + "_" + part_spec.capitalize()
        if os.path.isfile(curr_save_name + ".pdf") and not overwrite:
            continue

        # Get data
        data_lst = []
        m_names = ['Ground truth']
        for ct, m in enumerate(model_list):
            m_names += [model_list[ct].name if model_names is None else model_names[ct]]
            data_lst += [m.get_predicted_plt_data(dat, name_list=names,
                                                  const_steps=n_s,
                                                  title=title, n_ts_off=n)]

        # Aggregate data from all models for same series
        new_data_list = []
        for k in range(n_pred):

            # Prepare plot data
            first_plot_ds = data_lst[0][k][0]
            data_len = first_plot_ds.data.shape[0]
            plot_data = np.empty((data_len, n_plot_series), dtype=np.float32)

            # Set ground truth
            plot_data[:, 0] = first_plot_ds.data[:, 1]

            # Add predictions of all models
            tal = None
            for ct, p_data in enumerate(data_lst):
                ds, tal, _ = p_data[k]
                plot_data[:, 1 + ct] = ds.data[:, 0]

            # Define dataset for plotting
            scaling, is_scd = first_plot_ds.get_scaling_mul(0, n_plot_series)
            actual_dt = d.get_shifted_t_init(n + d.seq_len - 1)
            analysis_ds = Dataset(plot_data, d.dt,
                                  actual_dt, scaling,
                                  is_scd, m_names)

            new_data_list += [(analysis_ds, tal, None)]

        # Plot dataset
        if n_s > 0:
            curr_save_name += f"_{n_s}"
        plot_visual_all_in_one(new_data_list, curr_save_name)
