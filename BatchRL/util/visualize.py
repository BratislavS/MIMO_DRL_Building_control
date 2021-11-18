"""This file contains all plotting stuff.

Plotting is based on matplotlib.
If we are not on a Windows platform, the backend 'Agg' is
used, since this works also on Euler.

Most functions plot some kind of time series, but there
are also more specialized plot functions. E.g. plotting
the training of a keras neural network.
"""
import os
import warnings
from contextlib import contextmanager
from typing import Dict, Sequence, Tuple, List, Any, Type

import matplotlib as mpl
import numpy as np
from matplotlib.dates import DateFormatter
from mpl_toolkits.axes_grid1 import host_subplot
from pandas.plotting import register_matplotlib_converters
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from opcua_empa.opcuaclient_subscription import MAX_TEMP, MIN_TEMP
from util.numerics import fit_linear_1d, load_performance, check_shape, ErrMetric, MaxAbsEer, MAE
from util.util import EULER, datetime_to_np_datetime, string_to_dt, get_if_not_none, clean_desc, split_desc_units, \
    create_dir, Num, yeet, tot_size, mins_to_str, IndT, BASE_DIR, DEFAULT_TRAIN_SET

if EULER:
    # Do not use GUI based backend.
    mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors

# Set pdf output to be editable
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

register_matplotlib_converters()

DEFAULT_FONT_SIZE: int = 18
LONG_FIG_SIZE = (12, 4)

font = {'family': 'serif',
        # 'weight': 'bold',
        'size': DEFAULT_FONT_SIZE}

plt.rc('font', **font)
# plt.rc('text', usetex=True)  # Makes Problems with the Celsius sign :(

# Plotting colors
colors = mpl_colors.TABLEAU_COLORS
names = list(colors)
clr_map = [colors[name] for name in names]
clr_map[0], clr_map[1] = clr_map[1], clr_map[0]  # For Bratislav
n_cols: int = len(clr_map)  #: Number of colors in colormap.

# Plotting styles
styles = [
    ("--", 'o'),
    ("-", 's'),
    ("-.", 'v'),
    (":", '*'),
]
joint_styles = [j + i for i, j in styles]

# Saving
PLOT_DIR = os.path.join(BASE_DIR, "Plots")  #: Base plot folder.
preprocess_plot_path = os.path.join(PLOT_DIR, "Preprocessing")  #: Data processing plot folder.
model_plot_path = os.path.join(PLOT_DIR, "Models")  #: Dynamics modeling plot folder.
rl_plot_path = os.path.join(PLOT_DIR, "RL")
EVAL_MODEL_PLOT_DIR = os.path.join(model_plot_path, "EvalTables")
OVERLEAF_DIR = os.path.join(BASE_DIR, "Overleaf")  #: Overleaf base folder.

# Create folders if they do not exist
create_dir(preprocess_plot_path)
create_dir(model_plot_path)
create_dir(rl_plot_path)
create_dir(EVAL_MODEL_PLOT_DIR)


class ModifyDir:
    """Class wrapper for directory that can be changed across modules.

    Initially:

    >>> CONST_DIR_OLD = "Some/Constant/Path"

    Then if you import the variable and change it locally,
    the other modules do not see the change. Now this changed to:

    >>> CONST_DIR = ModifyDir("Some/Constant/Path")

    Which allows you to modify it:

    >>> CONST_DIR.set_folder("New")
    >>> CONST_DIR
    "Some/Constant/Path/New"

    And now this can be seen from all modules! The :func:`__str__`, :func:`__repr__`
    and the :func:`__fspath__` still allow the usage as string / path.
    """

    def __init__(self, orig_dir: str):
        """Initializer

        Args:
            orig_dir: The directory that is wrapped and may be modified.
        """
        self.ol_dir = orig_dir
        self.ret_dir = orig_dir
        create_dir(orig_dir)

    def set_folder(self, new_dir) -> None:
        """Adds the given directory to the current path.

        Args:
            new_dir: The new directory.
        """
        if new_dir is None:
            self.ret_dir = self.ol_dir
        else:
            self.ret_dir = os.path.join(self.ol_dir, new_dir)
            create_dir(self.ret_dir)

    def __str__(self):
        return self.ret_dir

    def __repr__(self):
        return self.ret_dir

    def __fspath__(self):
        # Allows usage as file path
        return self.ret_dir


# Here it is
OVERLEAF_IMG_DIR = ModifyDir(os.path.join(OVERLEAF_DIR, "Imgs"))
OVERLEAF_DATA_DIR = ModifyDir(os.path.join(OVERLEAF_DIR, "Data"))


@contextmanager
def change_dir_name(new_dir_name: str,
                    dir_to_modify: ModifyDir = OVERLEAF_IMG_DIR):
    """This Context Manager allows you to change the
    overleaf images folder temporarily."""
    dir_to_modify.set_folder(new_dir_name)
    yield None
    dir_to_modify.set_folder(None)


def save_figure(save_name, show: bool = False,
                vector_format: bool = True,
                size: Tuple[Num, Num] = None,
                font_size: int = None,
                auto_fmt_time: bool = True) -> None:
    """Saves the current figure.

    Args:
        size: The size in inches, if None, (16, 9) is used.
        save_name: Path where to save the plot.
        show: If true, does nothing.
        vector_format: Whether to save image in vector format.
        font_size: The desired font size.
        auto_fmt_time: Nice stuff, just set to True!
    """
    if auto_fmt_time:
        fig = plt.gcf()
        fig.autofmt_xdate()

    if font_size is not None:
        plt.rc('font', size=font_size)

    if save_name is not None and not show:
        # Set figure size
        fig = plt.gcf()
        sz = size if size is not None else (16, 9)
        fig.set_size_inches(*sz)

        # Save and clear
        save_format = '.pdf' if vector_format else '.png'
        # save_kwargs = {'bbox_inches': 'tight', 'dpi': 500}
        save_kwargs = {'bbox_inches': 'tight'}
        fig.tight_layout()
        plt.savefig(save_name + save_format, **save_kwargs)
        plt.close()

    # Set font back to original
    if font_size is not None:
        plt.rc('font', size=DEFAULT_FONT_SIZE)


def _plot_helper(x, y, m_col='blue', label: str = None,
                 dates: bool = False, steps: bool = False, ax=plt,
                 grid: bool = True):
    """Defining basic plot style for all plots.

    TODO: Make x optional. (Except for `dates` == True case!)

    Args:
        x: X values
        y: Y values
        m_col: Marker and line color.
        label: The label of the current series.
        dates: Whether to use datetimes in x-axis.
        steps: Whether to plot piecewise constant series.
        ax: The axis to plot the series on.
    """
    # Determine style
    ls = ':'
    marker = '^'
    ms = 4
    kwargs = {'marker': marker, 'c': m_col, 'linestyle': ls, 'label': label, 'markersize': ms, 'mfc': m_col,
              'mec': m_col}

    # Choose plotting method
    plot_method = ax.plot
    if dates:
        plot_method = ax.plot_date
        if steps:
            kwargs["drawstyle"] = "steps"
    elif steps:
        plot_method = ax.step

    if grid:
        plt.grid(b=True)

    # Finally plot and return
    return plot_method(x, y, **kwargs)


def basic_plot(x, y, save_name: str,
               xy_lab: Tuple[str, str] = (None, None),
               title: str = None,
               fig_size=None):
    """Simple plot function that plots one data series."""

    if x is None:
        x = range(len(y))
    _plot_helper(x, y)

    # Set labels and title
    plt.ylabel(xy_lab[1])
    plt.xlabel(xy_lab[0])
    if title is not None:
        plt.title(title)

    # Save
    save_figure(save_name, size=fig_size)


def plot_time_series(x, y, m: Dict, show: bool = True,
                     series_index: int = 0,
                     title: str = None,
                     save_name: str = None):
    """Plots a raw time-series where x are the dates and y are the values.
    """

    # Define plot
    lab = clean_desc(m['description'])
    if series_index == 0:
        # Init new plot
        plt.subplots()
    _plot_helper(x, y, clr_map[series_index], label=lab, dates=True)
    if title:
        plt.title(title)
    plt.ylabel(m['unit'])
    plt.xlabel('Time')
    plt.legend()

    # Show plot
    if show:
        plt.show()

    # Sate to raster image since vector image would be too large
    save_figure(save_name, show, vector_format=False)


def plot_multiple_time_series(x_list, y_list, m_list, *,
                              show: bool = True,
                              title_and_ylab: Sequence = None,
                              save_name: str = None):
    """Plots multiple raw time series.
    """
    n = len(x_list)
    for ct, x in enumerate(x_list):
        plot_time_series(x, y_list[ct], m_list[ct], show=show and ct == n - 1, series_index=ct)

    # Set title
    if title_and_ylab is not None:
        plt.title(title_and_ylab[0])
        plt.ylabel(title_and_ylab[1])
    plt.legend()

    # Sate to raster image since vector image would be too large
    save_figure(save_name, show, vector_format=False)


def plot_ip_time_series(y, lab=None, m=None, show=True, init=None, mean_and_stds=None, use_time=False):
    """
    Plots an interpolated time series
    where x is assumed to be uniform.
    DEPRECATED: DO NOT USE!!!!!
    """
    warnings.warn("This is deprecated!!!!")

    # Define plot
    plt.subplots()
    if isinstance(y, list):
        n = y[0].shape[0]
        n_init = 0 if init is None else init.shape[0]
        if init is not None:
            x_init = [15 * i for i in range(n_init)]
            _plot_helper(x_init, init, m_col='k')
            # plt.plot(x_init, init, linestyle=':', marker='^', color='red', markersize=5, mfc = 'k', mec = 'k')

        if use_time:
            mins = m[0]['dt']
            interval = np.timedelta64(mins, 'm')
            dt_init = datetime_to_np_datetime(string_to_dt(m[0]['t_init']))
            x = [dt_init + i * interval for i in range(n)]
        else:
            x = [15 * i for i in range(n_init, n_init + n)]

        for ct, ts in enumerate(y):
            if mean_and_stds is not None:
                ts = mean_and_stds[ct][1] * ts + mean_and_stds[ct][0]
            clr = clr_map[ct % n_cols]
            curr_lab = None if lab is None else lab[ct]
            _plot_helper(x, ts, m_col=clr, label=curr_lab, dates=use_time)
    else:
        y_curr = y
        if mean_and_stds is not None:
            y_curr = mean_and_stds[1] * y + mean_and_stds[0]
        x = range(len(y_curr))
        _plot_helper(x, y_curr, m_col='blue', label=lab, dates=use_time)

        if m is not None:
            plt.title(m['description'])
            plt.ylabel(m['unit'])

    plt.xlabel('Time [min.]')
    plt.legend()

    # Show plot
    if show:
        plt.show()
    raise NotImplementedError("This was deprecated as of 17.01.20.")


def plot_single_ip_ts(y,
                      lab=None,
                      show=True,
                      *,
                      mean_and_std=None,
                      use_time=False,
                      title_and_ylab=None,
                      dt_mins=15,
                      dt_init_str=None):
    """
    Wrapper function with fewer arguments for single
    time series plotting.
    """
    plot_ip_ts(y,
               lab=lab,
               show=show,
               mean_and_std=mean_and_std,
               title_and_ylab=title_and_ylab,
               dt_mins=dt_mins,
               dt_init_str=dt_init_str,
               use_time=use_time,
               last_series=True,
               series_index=0,
               timestep_offset=0)


def plot_ip_ts(y,
               lab=None,
               show=True,
               mean_and_std=None,
               use_time=False,
               series_index=0,
               last_series=True,
               title_and_ylab=None,
               dt_mins=15,
               dt_init_str=None,
               timestep_offset=0,
               new_plot: bool = True):
    """
    Plots an interpolated time series
    where x is assumed to be uniform.
    """

    if series_index == 0 and new_plot:
        # Define new plot
        plt.subplots()

    n = len(y)
    y_curr = np.copy(y)

    # Add std and mean back
    if mean_and_std is not None:
        y_curr = mean_and_std[1] * y + mean_and_std[0]

    # Use datetime for x values
    if use_time:
        if dt_init_str is None:
            raise ValueError("Need to know the initial time of the time series when plotting with dates!")

        mins = dt_mins
        interval = np.timedelta64(mins, 'm')
        dt_init = datetime_to_np_datetime(string_to_dt(dt_init_str))
        x = [dt_init + (timestep_offset + i) * interval for i in range(n)]
    else:
        x = range(n)

    _plot_helper(x, y_curr, m_col=clr_map[series_index], label=clean_desc(lab), dates=use_time)

    if last_series:
        if title_and_ylab is not None:
            tt, lab = title_and_ylab
            if tt is not None:
                plt.title(title_and_ylab[0])
            if lab is not None:
                plt.ylabel(title_and_ylab[1])

        x_lab = 'Time' if use_time else 'Time [' + str(dt_mins) + ' min.]'
        plt.xlabel(x_lab)
        plt.legend()

        # Show plot
        if show:
            plt.show()

    return


def plot_multiple_ip_ts(y_list,
                        lab_list=None,
                        mean_and_std_list=None,
                        use_time=False,
                        timestep_offset_list=None,
                        dt_init_str_list=None,
                        show_last=True,
                        title_and_ylab=None,
                        dt_mins=15,
                        new_plot: bool = True):
    """
    Plotting function for multiple time series.
    """
    n = len(y_list)
    for k, y in enumerate(y_list):
        ts_offset = get_if_not_none(timestep_offset_list, k, 0)
        lab = get_if_not_none(lab_list, k)
        m_a_s = get_if_not_none(mean_and_std_list, k)
        dt_init_str = get_if_not_none(dt_init_str_list, k)

        last_series = k == n - 1
        plot_ip_ts(y,
                   lab=lab,
                   show=last_series and show_last,
                   mean_and_std=m_a_s,
                   use_time=use_time,
                   series_index=k,
                   last_series=last_series,
                   title_and_ylab=title_and_ylab,
                   dt_mins=dt_mins,
                   dt_init_str=dt_init_str,
                   timestep_offset=ts_offset,
                   new_plot=new_plot)


def plot_single(time_series, m, use_time=True, show=True, title_and_ylab=None, scale_back=True, save_name=None):
    """
    Higher level plot function for single time series.
    """
    m_a_s = m.get('mean_and_std') if scale_back else None
    plot_single_ip_ts(time_series,
                      lab=m.get('description'),
                      show=show,
                      mean_and_std=m_a_s,
                      use_time=use_time,
                      title_and_ylab=title_and_ylab,
                      dt_mins=m.get('dt'),
                      dt_init_str=m.get('t_init'))
    save_figure(save_name, show)


def plot_all(all_data, m, use_time=True, show=True, title_and_ylab=None, scale_back=True, save_name=None):
    """
    Higher level plot function for multiple time series
    stored in the matrix 'all_data'
    as they are e.g. saved in processed form.
    """

    n_series = all_data.shape[1]
    all_series = [all_data[:, i] for i in range(n_series)]

    mean_and_std_list = [m[i].get('mean_and_std') for i in range(n_series)] if scale_back else None

    plot_multiple_ip_ts(all_series,
                        lab_list=[m[i].get('description') for i in range(n_series)],
                        mean_and_std_list=mean_and_std_list,
                        use_time=use_time,
                        timestep_offset_list=[0 for _ in range(n_series)],
                        dt_init_str_list=[m[i].get('t_init') for i in range(n_series)],
                        show_last=show,
                        title_and_ylab=title_and_ylab,
                        dt_mins=m[0].get('dt'))
    save_figure(save_name, show)


def plot_dataset(dataset, show: bool = True,
                 title_and_ylab=None,
                 save_name: str = None,
                 new_plot: bool = True,
                 fig_size: Tuple[int, int] = None) -> None:
    """Plots the unscaled series in a dataset.

    Args:
        dataset: The dataset to plot.
        show: Whether to show the plot.
        title_and_ylab: List with title and y-label.
        save_name: The file path for saving the plot.
        new_plot: Whether to open a new figure.
    """
    all_data = dataset.get_unscaled_data()
    n_series = all_data.shape[1]
    all_series = [np.copy(all_data[:, i]) for i in range(n_series)]
    labs = [d for d in dataset.descriptions]
    t_init = dataset.t_init

    plot_multiple_ip_ts(all_series,
                        lab_list=labs,
                        mean_and_std_list=None,
                        use_time=True,
                        timestep_offset_list=[0 for _ in range(n_series)],
                        dt_init_str_list=[t_init for _ in range(n_series)],
                        show_last=show,
                        title_and_ylab=title_and_ylab,
                        dt_mins=dataset.dt,
                        new_plot=new_plot)

    fig = plt.gcf()
    fig.autofmt_xdate()

    if save_name is not None:
        save_figure(save_name, show, size=fig_size)


def scatter_plot(x, y, *,
                 show=True,
                 lab_dict=None,
                 lab='Measurements',
                 m_and_std_x=None,
                 m_and_std_y=None,
                 add_line=False,
                 custom_line=None,
                 custom_label=None,
                 save_name=None,
                 fig_size=None) -> None:
    """Scatter Plot.

    Args:
        x: x-coordinates
        y: y-coordinates
        show:
        lab_dict:
        lab:
        m_and_std_x:
        m_and_std_y:
        add_line:
        custom_line:
        custom_label:
        save_name:
        fig_size:
    """

    plt.subplots()
    plt.grid(True)

    # Transform data back to original mean and std.
    x_curr = x
    if m_and_std_x is not None:
        x_curr = m_and_std_x[1] * x + m_and_std_x[0]
    y_curr = y
    if m_and_std_y is not None:
        y_curr = m_and_std_y[1] * y + m_and_std_y[0]

    if add_line:
        # Fit a line with Least Squares
        max_x = np.max(x_curr)
        min_x = np.min(x_curr)
        x = np.linspace(min_x, max_x, 5)
        y = fit_linear_1d(x_curr, y_curr, x)
        plt.plot(x, y, label='Linear fit')

    if custom_line:
        # Add a custom line to plot
        x, y = custom_line
        if m_and_std_y is not None:
            y = m_and_std_y[1] * y + m_and_std_y[0]
        if m_and_std_x is not None:
            x = m_and_std_x[1] * x + m_and_std_x[0]
        plt.plot(x, y, label=custom_label)

    # Plot
    plt.scatter(x_curr, y_curr, marker='^', c='red', label=lab)

    # Add Labels
    if lab_dict is not None:
        if lab_dict.get("title") is not None:
            plt.title(lab_dict['title'])
        plt.ylabel(lab_dict['ylab'])
        plt.xlabel(lab_dict['xlab'])
    plt.legend()

    # Show or save
    if show:
        plt.show()
    save_figure(save_name, show, size=fig_size)


def plot_train_history(hist, name: str = None, val: bool = True) -> None:
    """Visualizes the training of a keras model.

    Plot training & validation loss values of 
    a history object returned by keras.Model.fit().

    Args:
        hist: The history object returned by the fit method.
        name: The path of the plot file if saving it.
        val: Whether to include the validation curve.
    """
    plt.subplots()
    plt.plot(hist.history['loss'])
    if val:
        plt.plot(hist.history['val_loss'])
    plt.yscale('log')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    leg = ['Training']
    if val:
        leg += ['Validation']
    plt.legend(leg, loc='upper right')
    if name is not None:
        save_figure(name, False)
    else:
        plt.show()


def plot_rewards(hist, name: str = None) -> None:
    """Plots the rewards from RL training.

    Args:
        hist: The history object.
        name: The path where to save the figure.
    """
    plt.subplots()
    plt.plot(hist.history['episode_reward'])
    plt.title('Model reward')
    plt.ylabel('Reward')
    plt.xlabel('Episode')
    leg = ['Training']
    plt.legend(leg, loc='upper left')
    if name is not None:
        save_figure(name, False)
    else:
        plt.show()


def plot_simple_ts(y_list, title=None, name=None):
    """Plots the given aligned time series.
    """
    n = len(y_list[0])
    x = range(n)

    plt.subplots()
    for y in y_list:
        plt.plot(x, y)

    if title is not None:
        plt.title(title)
    if name is not None:
        save_figure(name, False)
    else:
        plt.show()


def stack_compare_plot(stack_y, y_compare, title=None, name=None):
    """Plots the given aligned time series.
    """
    n = len(y_compare[0])
    x = range(n)
    ys = [stack_y[:, i] for i in range(stack_y.shape[1])]

    fig, ax = plt.subplots()
    ax.stackplot(x, *ys)
    for y in y_compare:
        ax.plot(x, y)

    if title is not None:
        plt.title(title)
    if name is not None:
        save_figure(name, False)
    else:
        plt.show()


def plot_residuals_acf(residuals: np.ndarray,
                       name: str = None,
                       lags: int = 50,
                       partial: bool = False) -> None:
    """Plots the ACF of the residuals given.

    If `name` is None, the plot will be shown, else
    it will be saved to the path specified by `name`.

    Args:
        residuals: The array with the residuals for one series.
        name: The filename for saving the plot.
        lags: The number of lags to consider.
        partial: Whether to use the partial ACF.

    Returns:
        None

    Raises:
        ValueError: If residuals do not have the right shape.
    """
    if len(residuals.shape) != 1:
        yeet("Residuals needs to be a vector!")

    # Initialize and plot data
    plt.subplots()
    plot_fun = plot_pacf if partial else plot_acf
    plot_fun(residuals, lags=lags)

    # Set title and labels
    title = "Autocorrelation of residuals"
    if partial:
        title = "Partial " + title.lower()
    plt.title(title)
    plt.ylabel("Correlation")
    plt.xlabel("Lag")

    # Save or show
    if name is not None:
        save_figure(name, False, size=LONG_FIG_SIZE)
    else:
        plt.show()


def _setup_axis(ax, base_title: str, desc: str, title: bool = True):
    """Helper function for `plot_env_evaluation`.

    Adds label and title to axis."""
    t, u = split_desc_units(desc)
    ax.set_ylabel(u)
    if title:
        if base_title != "":
            ax.set_title(f"{base_title}: {t}")
        else:
            ax.set_title(f"{t}")


def _full_setup_axis(ax_list: List, desc_list: List, title: str = None,
                     hide: bool = True, hide_last: bool = True):
    # Check input
    assert len(ax_list) == len(desc_list), f"Incompatible lists: {ax_list} and {desc_list}!" \
                                           f" with title: {title}."

    # Set title if it is not None or an empty string.
    set_title = title is not None

    # Set axes
    for ct, ax in enumerate(ax_list):
        _setup_axis(ax, title, desc_list[ct], title=set_title)

        # if hide:
        #     if hide_last or ct < len(ax_list) - 1:
        #         ax.get_xaxis().set_visible(False)
        #         ax.get_xaxis().set_ticklabels([])


def _get_ds_descs(ds, series_mask=None, series_merging_list=None):
    """Extracts the descriptions for the control and the non-control series.

    Args:
        ds: Dataset

    Returns:
        The two list of descriptions.
    """
    # Get descriptions from dataset
    n_tot_vars = ds.d
    c_inds = ds.c_inds
    control_descs = [ds.descriptions[c] for c in c_inds]
    state_descs = [ds.descriptions[c] for c in range(n_tot_vars) if c not in c_inds]

    # Extract descriptions for merged plots
    if series_merging_list is not None:
        lst = []
        for inds, *_ in series_merging_list:
            inner_l = [state_descs[i] for i in inds]
            lst += [inner_l]
        merge_descs = lst
    else:
        merge_descs = None

    # Apply mask to state descriptions
    if series_mask is not None:
        state_descs = [state_descs[i] for i in series_mask]

    # Return
    return control_descs, state_descs, merge_descs


def _handle_merging(n_feats_init, series_mask=None, series_merging_list=None) -> Tuple[int, Any]:
    masking = series_mask is not None
    merging = series_merging_list is not None
    s_mask = series_mask if masking else np.arange(n_feats_init)
    n_masked = len(s_mask)
    if not merging:
        return n_masked, s_mask

    # Now merge!
    inds = np.ones((n_masked,), dtype=np.bool)
    for el, *_ in series_merging_list:
        for e in el:
            # Find e in mask
            w = np.argwhere(s_mask == e)
            assert len(w) == 1, f"Series {e} cannot be combined!!"
            pos = w[0][0]
            assert inds[pos], f"Series {e} combined at least twice!!"
            inds[pos] = False

    # Extract final indices
    new_inds = s_mask[inds]
    return len(new_inds), new_inds


def _extract_states(states, series_mask=None, series_merging_list=None):
    masked_state = states if series_mask is None else states[:, :, series_mask]
    merged_state_list = []
    if series_merging_list is not None:
        for inds, *_ in series_merging_list:
            merged_state_list += [states[:, :, inds]]

    return masked_state, merged_state_list


def _remove_tick_label(ax):
    # labels = [item.get_text() for item in ax.get_xticklabels()]
    # empty_string_labels = [''] * len(labels)
    # ax.set_xticklabels(empty_string_labels)
    [label.set_visible(False) for label in ax.get_xticklabels()]


# Type for series merge lists
MergeListT = List[Tuple[IndT, str]]


def plot_env_evaluation(actions: np.ndarray,
                        states: np.ndarray,
                        rewards: np.ndarray,
                        ds,
                        agent_names: Sequence[str],
                        save_path: str = None,
                        extra_actions: np.ndarray = None,
                        series_mask: np.ndarray = None, *,
                        title_ext: str = None,
                        show_rewards: bool = True,
                        np_dt_init: Any = None,
                        rew_save_path: str = None,
                        series_merging_list: MergeListT = None,
                        bounds: List[Tuple[int, Tuple[Num, Num]]] = None,
                        reward_descs: List[str] = None,
                        disconnect_data: Tuple[int, Tuple[int, int], Any] = None,
                        ex_ext: bool = True,
                        tot_reward_only: bool = False,
                        color_offset: int = 0,
                        fig_size: Any = None) -> None:
    """Plots the evaluation of multiple agents on an environment.

    TODO: Refactor this shit more!
    TODO: Add grid????

    Only for one specific initial condition.
    """
    assert len(agent_names) == actions.shape[0], "Incorrect number of names!"
    if series_mask is not None:
        check_shape(series_mask, (-1,))
        assert len(series_mask) < states.shape[2]

    # Check fallback actions
    plot_extra = extra_actions is not None
    use_time = np_dt_init is not None

    # Extract and check shapes
    n_agents, episode_len, n_feats = states.shape
    check_shape(actions, (n_agents, episode_len, -1))
    n_rewards = 1
    if tot_reward_only:
        if len(rewards.shape) == 3:
            rewards = rewards[:, :, 0]
        if len(rewards.shape) == 2:
            check_shape(rewards, (n_agents, episode_len))
            rewards = np.expand_dims(rewards, axis=-1)
        rew_s = rewards.shape
        assert len(rew_s) == 3 and rew_s[2] == 1, f"Fuck: {rewards.shape}"
        if reward_descs is not None:
            reward_descs = ["Total reward"]
    elif len(rewards.shape) == 2:
        check_shape(rewards, (n_agents, episode_len))
        rewards = np.expand_dims(rewards, axis=-1)
    else:
        check_shape(rewards, (n_agents, episode_len, -1))
        n_rewards = rewards.shape[2]
        if reward_descs is not None:
            reward_descs = ["Total reward"] + reward_descs
            assert n_rewards == len(reward_descs), f"Fuck this: {reward_descs}"
    if not show_rewards:
        n_rewards = 0

    n_feats, series_mask = _handle_merging(n_feats, series_mask, series_merging_list)
    if series_merging_list is None:
        series_merging_list = []
    n_merged_series = len(series_merging_list)
    n_actions = actions.shape[-1]
    tot_n_plots = n_actions + n_feats + n_merged_series + \
                  n_rewards * show_rewards + plot_extra * n_actions

    # We'll use a separate GridSpecs for controls, states and rewards
    fig = plt.figure()
    h_s = 0.4
    margin = 0.4 / tot_n_plots
    t = 1 - margin  # 0.95
    gs_con = plt.GridSpec(tot_n_plots, 1, hspace=h_s, top=t, bottom=0.0, figure=fig)
    gs_state = plt.GridSpec(tot_n_plots, 1, hspace=h_s, top=t, bottom=0.0, figure=fig)
    gs_state_merged = plt.GridSpec(tot_n_plots, 1, hspace=h_s, top=t, bottom=0.0, figure=fig)
    gs_rew = plt.GridSpec(tot_n_plots, 1, hspace=h_s, top=t, bottom=0.0, figure=fig)
    gs_con_fb = plt.GridSpec(tot_n_plots, 1, hspace=h_s, top=t, bottom=0.0, figure=fig)

    # Define axes
    n_act_plots = n_actions * (1 + plot_extra)
    rew_ax = fig.add_subplot(gs_rew[-1, :])
    rew_axs = [fig.add_subplot(gs_rew[(-n_rewards + i), :], sharex=rew_ax) for i in range(n_rewards - 1)]
    rew_axs += [rew_ax]
    con_axs = [fig.add_subplot(gs_con[i, :], sharex=rew_ax) for i in range(n_actions)]
    state_axs = [fig.add_subplot(gs_state[i, :], sharex=rew_ax)
                 for i in range(n_act_plots + n_merged_series, tot_n_plots - n_rewards)]
    state_mrg_axs = [fig.add_subplot(gs_state_merged[i, :], sharex=rew_ax)
                     for i in range(n_act_plots, n_act_plots + n_merged_series)]
    con_fb_axs = [fig.add_subplot(gs_con_fb[i, :], sharex=rew_ax) for i in range(n_actions, n_act_plots)]
    ax_list = con_axs + con_fb_axs + state_mrg_axs + state_axs + rew_axs
    assert plot_extra or con_fb_axs == [], "Something is wrong!"

    # Find legends
    control_descs, state_descs, merge_descs = _get_ds_descs(ds, series_mask, series_merging_list)

    # Reduce series
    states, merged_states_list = _extract_states(states, series_mask, series_merging_list)

    # Set titles and setup axes
    if show_rewards:
        if reward_descs is None:
            if n_rewards == 1:
                reward_descs = [f"Rewards"]
            else:
                reward_descs = [f"Rewards {ct}" for ct, r_ax in enumerate(rew_axs)]
        [r_ax.set_title(reward_descs[ct]) for ct, r_ax in enumerate(rew_axs)]

        _full_setup_axis(rew_axs[:-1], reward_descs[:-1], "Reward", hide=True)

    c_title = "Control inputs"
    _full_setup_axis(con_axs, control_descs, "Original " + c_title.lower() if plot_extra else c_title)
    if plot_extra:
        _full_setup_axis(con_fb_axs, control_descs, "Constrained " + c_title.lower())
    _full_setup_axis(state_axs, state_descs, "States", hide_last=show_rewards)
    for ct, m in enumerate(series_merging_list):
        _full_setup_axis([state_mrg_axs[ct]], [m[1]], "Exogenous states" if ex_ext else "")

    # Define legend fontsize
    sz = 12
    leg_kwargs = {'prop': {'size': sz},
                  'loc': 1}
    l_kws_reduced = {'prop': {'size': sz}, }

    # Take care of the x-axis
    if use_time:
        interval = np.timedelta64(ds.dt, 'm')
        x = [np_dt_init + i * interval for i in range(episode_len)]
    else:
        x = range(len(rewards[0]))
    x_array = np.array(x)
    auto_fmt = False
    if use_time:
        eval_5_15h = np.timedelta64(5, 'h') <= x_array[-1] - x_array[0] <= np.timedelta64(15, 'h')
        eval_4_7d = np.timedelta64(4, 'D') <= x_array[-1] - x_array[0] <= np.timedelta64(7, 'D')
        auto_fmt = not (eval_5_15h or eval_4_7d)

    ph_kwargs = {"dates": use_time}

    # Define helper function
    def _plot_helper_helper(data: np.ndarray, axis_list: List, ag_names: Sequence[str],
                            steps: bool = False, merged: bool = False, col_off: int = 0) -> None:
        for j, a_name in enumerate(ag_names):
            for i, ax in enumerate(axis_list):
                curr_dat = data[i, :, j] if merged else data[j, :, i]
                _plot_helper(x, curr_dat, m_col=clr_map[j + col_off],
                             label=a_name, ax=ax, steps=steps, **ph_kwargs)
                if use_time:
                    if x_array[-1] - x_array[0] < np.timedelta64(3, 'D'):
                        formatter = DateFormatter("%H:%M")
                        ax.xaxis.set_major_formatter(formatter)
                    elif x_array[-1] - x_array[0] < np.timedelta64(6, 'D'):
                        if fig_size is not None and fig_size[0] < 10:
                            formatter = DateFormatter("%d.%m")
                            ax.xaxis.set_major_formatter(formatter)

                    pass
                    # formatter = DateFormatter("%m/%d, %H:%M")
                    # ax.xaxis.set_major_formatter(formatter)
                    # ax.xaxis.set_tick_params(rotation=30)

    # Plot all the series
    _plot_helper_helper(actions, con_axs, agent_names, steps=True, col_off=color_offset)
    if plot_extra:
        _plot_helper_helper(extra_actions, con_fb_axs, agent_names, steps=True, col_off=color_offset)
    _plot_helper_helper(states, state_axs, agent_names, steps=False, col_off=color_offset)
    if show_rewards:
        _plot_helper_helper(rewards, rew_axs, agent_names, steps=False, col_off=color_offset)
    for ct, m in enumerate(series_merging_list):
        if len(m) == 2:
            # Add left and right y-label
            axs = state_mrg_axs[ct]
            all_axs = [axs, axs.twinx()]
            for ct_x, curr_ax in enumerate(all_axs):
                curr_desc = merge_descs[ct][ct_x]
                _, u = split_desc_units(curr_desc)
                _plot_helper_helper(merged_states_list[ct][:, :, ct_x:(ct_x + 1)],
                                    [curr_ax], [curr_desc],
                                    steps=False, merged=True, col_off=ct_x)
                curr_ax.set_ylabel(u)

            # See https://github.com/matplotlib/matplotlib/issues/3706
            legs = [curr_ax.legend(**l_kws_reduced, loc=2 - ct_x)
                    for ct_x, curr_ax in enumerate(all_axs)]
            legs[0].set_zorder(999999)
            legs[0].remove()
            all_axs[-1].add_artist(legs[0])

        else:
            # Series might be scaled badly!
            _, u = split_desc_units(m[2])
            _plot_helper_helper(merged_states_list[ct], [state_mrg_axs[ct]], merge_descs[ct],
                                steps=False, merged=True)
            state_mrg_axs[ct].set_ylabel(u)
            state_mrg_axs[ct].legend(**leg_kwargs)

    # Plot bounds
    if bounds is not None:
        al = 0.13
        for i, bd in bounds:
            low, up = bd
            if low == up:
                mid = up
                low -= 0.05
                up += 0.05
                al = 0.6
                # dx = x[-1] - x[0]
                # state_axs[i].annotate(f'Comfort setpoint',
                #                       xy=(x[0] + dx / 2, mid),
                #                       xytext=(0, 0),  # 3 points vertical offset
                #                       bbox={'facecolor': 'white', 'alpha': 0.2,
                #                             'edgecolor': 'none'},
                #                       textcoords="offset points",
                #                       ha='center', va='bottom',
                #                       fontsize=14)
                state_axs[i].plot([x[0], x[-1]], [mid, mid], "--o",
                                  ms=4, lw=4, c=(0.0, 1.0, 0.0, 0.6),
                                  label=f'Comfort setpoint')
            else:
                upper = [up for _ in range(episode_len)]
                lower = [low for _ in range(episode_len)]
                state_axs[i].fill_between(x, lower, upper, facecolor='green',
                                          interpolate=True, alpha=al)

    # Plot disconnect time
    if disconnect_data is not None:
        i, dis_t, ran = disconnect_data
        dt_h_low, dt_h_high = dis_t
        init_day = np.array(x_array[0], dtype='datetime64[D]')
        end_day = np.array(x_array[-1], dtype='datetime64[D]')
        time_passed_init = x_array[0] - init_day
        end_dt = np.timedelta64(ds.dt, 'm')
        fb_args = {"facecolor": 'black',
                   'interpolate': True,
                   'alpha': 1.0,
                   'zorder': 9999}
        dx_h = (x_array[1] - x_array[0]) / 2
        if dt_h_low <= time_passed_init < dt_h_high:
            state_axs[i].fill_between([x_array[0] - dx_h, init_day + dt_h_high - end_dt],
                                      ran[0], ran[1], **fb_args)

        if dt_h_low > time_passed_init > dt_h_low - np.timedelta64(2, 'h'):
            state_axs[i].fill_between([init_day + dt_h_low, init_day + dt_h_high - end_dt],
                                      ran[0], ran[1], **fb_args)

        time_passed_end = x_array[-1] - end_day
        if dt_h_low < time_passed_end <= dt_h_high:
            state_axs[i].fill_between([end_day + dt_h_low, x_array[-1] + dx_h],
                                      ran[0], ran[1], **fb_args)

    # Add legends
    con_axs[0].legend(**leg_kwargs)
    state_axs[0].legend(**leg_kwargs)
    x_label = "Time" if use_time else f"Timestep [{ds.dt}min]"
    if not show_rewards:
        rew_axs[-1].yaxis.set_visible(False)
        # rew_axs[-1].xaxis.set_visible(True)
    if show_rewards:
        rew_axs[-1].set_xlabel(x_label)
        rew_axs[-1].legend(**leg_kwargs)
    else:
        state_axs[-1].set_xlabel(x_label)

    # Remove tick labels
    for a in ax_list:
        a.grid(b=True)
    for c_ax in con_axs:
        _remove_tick_label(c_ax)
    for c_ax in state_mrg_axs:
        _remove_tick_label(c_ax)
    for c_ax in con_fb_axs:
        _remove_tick_label(c_ax)
    if show_rewards:
        for s_ax in state_axs:
            _remove_tick_label(s_ax)
        for r_ax in rew_axs[:-1]:
            _remove_tick_label(r_ax)
    else:
        for s_ax in state_axs[:-1]:
            _remove_tick_label(s_ax)

    # Super title
    if title_ext is not None:
        sup_t = 'Visual Analysis' if title_ext is None else title_ext
        con_axs[0].annotate(sup_t, (0.5, 1 - margin / 3),
                            xycoords='figure fraction', ha='center',
                            fontsize=24)
        # plt.suptitle('Main title')
        # plt.subplots_adjust(top=1.85)

    # Save
    s = (16, tot_n_plots * 1.8) if fig_size is None else fig_size
    if save_path is not None:
        save_figure(save_path, size=s, auto_fmt_time=auto_fmt)

    # Make a plot of the rewards
    if n_rewards == 1:
        if rew_save_path is not None:
            n_rewards = rewards.shape[1]
            r_res = rewards.reshape((n_agents, n_rewards, 1))
            plot_reward_details(agent_names, r_res, rew_save_path, [],
                                dt=ds.dt, n_eval_steps=n_rewards)


def plot_heat_cool_rew_det(*args, **kwargs):
    """Plotting reward details separately for heating and cooling.

    Args:
        args: Args for :func:`plot_reward_details`.
        kwargs: Kwargs for :func:`plot_reward_details`.
    """
    # Extract states and rewards
    ep_marks = kwargs['ep_marks']
    labels, rewards, path_name = args[:3]
    reward_copy = np.copy(rewards)

    plot_reward_details(*args, **kwargs)
    n_eval_steps = kwargs["n_eval_steps"]

    assert np.allclose(np.max(ep_marks, axis=0), np.min(ep_marks, axis=0))
    ep_marks = ep_marks[0]

    assert reward_copy.shape[1] == n_eval_steps, "WTF?"
    ep_marks = np.array(ep_marks, dtype=np.bool)
    heat_rewards = np.copy(reward_copy[:, ep_marks])
    cool_rewards = np.copy(reward_copy[:, np.logical_not(ep_marks)])
    n_eval_heat = heat_rewards.shape[1]
    n_eval_cool = cool_rewards.shape[1]

    heat_path = f"{path_name}_Heat"
    cool_path = f"{path_name}_Cool"
    assert n_eval_heat + n_eval_cool == n_eval_steps, "Hmm, fuck!"

    if heat_rewards.size != 0:
        kwargs["n_eval_steps"] = n_eval_heat
        plot_reward_details(labels, heat_rewards, heat_path, *args[3:], **kwargs)
    if cool_rewards.size != 0:
        kwargs["n_eval_steps"] = n_eval_cool
        plot_reward_details(labels, cool_rewards, cool_path, *args[3:], **kwargs)


def plot_reward_details(labels: Sequence[str],
                        rewards: np.ndarray,
                        path_name: str,
                        rew_descs: List[str], *,
                        dt: int = 15,
                        n_eval_steps: int = 2000,
                        title_ext: str = None,
                        scale_tot_rew: bool = True,
                        sum_reward: bool = False,
                        tol: float = 0.0005,
                        verbose: int = 0,
                        add_base_title: bool = True,
                        **kwargs) -> None:
    """Creates a bar plot with the different rewards of the different agents.

    Args:
        labels: List with agent names.
        rewards: Array with all rewards for all agents.
        path_name: Path of the plot to save.
        rew_descs: Descriptions of the parts of the reward.
        dt: Number of minutes in a timestep.
        n_eval_steps: Number of evaluation steps.
        title_ext: Extension to add to the title.
        scale_tot_rew: Whether to scale the total reward to a nice range.
        sum_reward: Whether to sum the rewards instead of averaging it.
        tol: Values smaller than `tol` will be set to 0.0 to avoid very small numbers in plot.
        verbose: Verbosity.
        add_base_title:
    """

    if verbose:
        if len(kwargs) > 0:
            print(f"Unused kwargs: {kwargs}")
    n_agents, _, n_rewards = rewards.shape
    assert n_rewards == len(rew_descs) + 1, "Incorrect number of descriptions!"
    assert n_agents == len(labels)
    red_fun = np.sum if sum_reward else np.mean
    mean_rewards = red_fun(rewards, axis=1)
    all_descs = ["Total reward"] + rew_descs

    agg_rew = "Total reward" if sum_reward else "Mean rewards per hour"
    title = f"{agg_rew} for {n_eval_steps} steps. "
    if not add_base_title:
        title = ""
    if title_ext is not None:
        title += title_ext

    if not sum_reward:
        fac = 60 / dt
        mean_rewards *= fac

    if scale_tot_rew and n_rewards > 1:
        # Scale the maximum reward to the same magnitude of any of the other parts
        max_tot = np.max(np.abs(mean_rewards[:, 0]))
        max_not_tot = np.max(np.abs(mean_rewards[:, 1:]))
        mean_rewards[:, 0] *= max_not_tot / max_tot

    if tol > 0:
        mean_rewards[np.abs(mean_rewards) < tol] = 0.0

    fig, ax = plt.subplots()

    # Define the bars
    x = np.arange(len(labels))  # the label locations
    width = 0.8 / n_rewards  # the width of the bars
    offs = (n_rewards - 1) * width / 2
    rects = [ax.bar(x - offs + i * width,
                    mean_rewards[:, i],
                    width,
                    label=all_descs[i])
             for i in range(n_rewards)]

    min_rew, max_rew = np.min(mean_rewards), np.max(mean_rewards)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Reward')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fac = 1.5
    plt.ylim((min_rew * fac, max_rew * fac))

    # Label the rectangles with the values.
    def auto_label(rect_list):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rect_list:
            height = rect.get_height()
            pos = height > 0
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3 if pos else -3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom' if pos else 'top',
                        fontsize=18 if n_rewards < 4 else 11)

    for r in rects:
        auto_label(r)

    # Shrink current axis's height by 10% on the bottom
    leg_h = 0.3
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * leg_h,
                     box.width, box.height * (1.0 - leg_h)])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -leg_h / 2),
              fancybox=True, shadow=True, ncol=1)

    # Set layout and save
    # fig.tight_layout()
    # s = ((1.5 - n_rewards / 5) * 24 * n_rewards / 5, 9 * n_rewards / 5)
    s = (11, 5.5)
    save_figure(save_name=path_name, size=s, auto_fmt_time=False)


def _load_all_model_data(model_list: List,
                         parts: List[str],
                         metric_list: List[str],
                         series_mask=None,
                         fit_data: str = DEFAULT_TRAIN_SET) -> Tuple[np.ndarray, np.ndarray]:
    """Helper function for `plot_performance_table` to load the performance data of all models.

    Args:
        model_list:
        parts:
        metric_list:

    Returns:
        The loaded data.
    """
    # Get sizes and check them
    n_models, n_parts, n_metrics = len(model_list), len(parts), len(metric_list)
    assert n_models > 0 and n_parts > 0 and n_metrics > 0, "Nothing to do here!"
    dt = model_list[0].data.dt

    # Load data of all models
    data_list = []
    inds = None
    for ct, m in enumerate(model_list):

        # # Define function generating path.
        # def gen_fun(name):
        #     return m.get_plt_path(name)

        # Load data of current model and check shape
        data, inds_curr = load_performance(m.get_plt_path, parts, dt, n_metrics,
                                           fit_data=fit_data)
        if ct == 0:
            inds = inds_curr
        else:
            assert np.array_equal(inds, inds_curr), f"Wrong indices: {inds_curr} != {inds}"
            check_shape(data_list[0], data.shape)
        data_list += [data]

    # Check shape and stack
    check_shape(data_list[0], (n_parts, -1, n_metrics, len(inds)))
    data_array = np.stack(data_list)

    # Reduce data
    if series_mask is not None:
        d = model_list[0].data
        prep_mask = d.to_prepared(series_mask)
        data_array = data_array[:, :, prep_mask]

    return data_array, inds


def _get_descs(model_list: List, remove_units: bool = True,
               series_mask=None, short_mod_names: List = None):
    # Edit series descriptions
    d = model_list[0].data
    all_desc = np.ones((d.d,), dtype=np.bool)
    all_desc[d.c_inds] = False
    if series_mask is None:
        series_descs = d.descriptions[all_desc]
        rel_scaling = d.scaling[all_desc]
    else:
        series_descs = d.descriptions[series_mask]
        rel_scaling = d.scaling[series_mask]
    if remove_units:
        series_descs = [sd.split("[")[0] for sd in series_descs]

    # Edit model names
    mod_names = [m.name for m in model_list]
    if short_mod_names is not None:
        assert len(short_mod_names) == len(mod_names), "Incorrect number of model names!"
        mod_names = short_mod_names

    return series_descs, mod_names, rel_scaling


def plot_performance_table(model_list: List, parts: List[str], metric_list: List[str],
                           name: str = "Test", short_mod_names: List = None,
                           remove_units: bool = True,
                           series_mask=None) -> None:
    # Define the ordering of the rows.
    order = (0, 1, 4, 2, 3)

    # Prepare the labels
    series_descs, mod_names, _ = _get_descs(model_list, remove_units, series_mask, short_mod_names)

    # Construct the path of the plot
    plot_path = os.path.join(EVAL_MODEL_PLOT_DIR, name)

    # Load data
    data_array, inds = _load_all_model_data(model_list, parts, metric_list, series_mask)

    # Handle indices and shapes
    sec_order = np.argsort(order)
    last_ind = order[-1]
    dat_shape = data_array.shape
    n_dim = len(dat_shape)
    n_models, n_parts, n_series, n_metrics, n_steps = dat_shape
    tot_s = tot_size(dat_shape)
    n_last = dat_shape[last_ind]
    tot_n_rows = tot_s // n_last

    # Compute indices to convert 5d array to 2d
    n_sub = []
    curr_sz = tot_n_rows
    for k in range(n_dim - 1):
        curr_sh = dat_shape[order[k]]
        curr_sz //= curr_sh
        n_sub += [(curr_sz, curr_sh)]

    # Initialize empty string array
    table_array = np.empty((tot_n_rows, n_dim - 1 + n_last), dtype="<U50")

    for k in range(tot_n_rows):

        all_inds = [(k // n) % m for n, m in n_sub]

        # Fill header cols
        table_array[k, 0] = mod_names[all_inds[0]]
        table_array[k, 1] = parts[all_inds[1]]
        table_array[k, 2] = f"{int(inds[all_inds[2]])}"
        table_array[k, 3] = series_descs[all_inds[3]]

        for i in range(n_last):
            ind_list = np.array(all_inds + [i])[sec_order]
            table_array[k, 4 + i] = f"{data_array[tuple(ind_list)]:.4g}"

    # Define column labels
    col_labels = ["Model", "Set", "Steps", "Series"]
    for k in range(n_last):
        col_labels += [metric_list[k]]

    base_w = 0.07
    col_w = [base_w for _ in col_labels]
    col_w[0] = col_w[3] = base_w * 4

    fig, ax = plt.subplots()

    # hide axes
    ax.axis('off')
    ax.axis('tight')

    ax.table(cellText=table_array, colLabels=col_labels,
             loc='center', colWidths=col_w)

    fig.tight_layout()

    save_figure(plot_path)


def _trf_desc_units(curr_desc: str, m: Type[ErrMetric], add_fac: Num = None) -> str:
    split_d = curr_desc.split("[")
    assert len(split_d) == 2, "Invalid description!"
    base, rest = split_d
    unit = rest.split("]")[0]
    trf_unit = m.unit_trf(unit)
    use_fac = add_fac is not None and not np.allclose(add_fac, 1.0)
    f_add = f"{add_fac:.3g} " if use_fac else ""
    return f"{base}[{f_add}{trf_unit}]"


def plot_p_g_2(model_list: List,
               parts: List[str],
               metric_list: Sequence[Type[ErrMetric]],
               short_mod_names: List = None,
               remove_units: bool = True,
               series_mask=None,
               scale_back: bool = False,
               compare_models: bool = False,
               overwrite: bool = True,
               fit_data: str = DEFAULT_TRAIN_SET,
               plot_folder: str = None,
               fig_size: Tuple[float, float] = None,
               **kwargs):
    """Similar as the one below."""

    plt_name = "mas_max_error_comp"
    save_path = os.path.join(plot_folder, plt_name)
    if os.path.isfile(save_path + ".pdf") and not overwrite:
        return

    metric_names = [m.name for m in metric_list]

    # Check if models are compatible
    from dynamics.base_model import check_model_compatibility
    check_model_compatibility(model_list)
    if not np.any(model_list[0].data.is_scaled):
        scale_back = False

    # Prepare the labels
    series_descs, mod_names, scaling = _get_descs(model_list, remove_units,
                                                  series_mask, short_mod_names)

    # Load data
    data_array, inds = _load_all_model_data(model_list,
                                            parts, metric_names,
                                            series_mask, fit_data=fit_data)
    n_models, n_parts, n_series, n_metrics, n_steps = data_array.shape

    # Switch lists
    if compare_models:
        parts, mod_names = mod_names, parts

    assert n_models == 1 and n_series == 1 and scale_back

    # Series scaling
    dt = model_list[0].data.dt
    ind = model_list[0].p_out_inds
    if scale_back:
        for k in range(n_series):
            for i, m in enumerate(metric_list):
                dat = data_array[:, :, k, i, :]
                # Scale the errors
                if scale_back:
                    m_and_sd = scaling[ind[k]]
                    data_array[:, :, k, i, :] = m.scaling_fun(dat, m_and_sd[1])

    s_desc = series_descs[ind[0]]
    s_desc = s_desc.replace("temperature", "temp.")

    fig = plt.figure()
    ax = plt.subplot(111)

    met_inds = [1, 2]
    for m_ct, m_ind in enumerate(met_inds):
        for p_ind in range(n_parts):
            dat = data_array[0, p_ind, 0, m_ind, :]
            ax.plot(inds, dat, joint_styles[p_ind],
                    c=clr_map[m_ind],
                    label=f"{parts[p_ind].capitalize()}: {metric_names[m_ind]}")

    tick_label = [mins_to_str(dt * int(i)) +
                  f"\n{int(i)} Step{'s' if i > 1 else ''}" for i in inds]
    plt.xticks(inds, tick_label)
    plt.ylabel(s_desc)
    ax.tick_params(axis='x', labelsize=14)

    # Shrink current axis by 20%
    shrink_fac = 0.8
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * shrink_fac, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
              fancybox=True, shadow=True)
    ax.grid(b=True)

    if fig_size is None:
        fig_size = (14, 3.5)
    save_figure(save_path, size=fig_size)


def plot_performance_graph(model_list: List, parts: List[str],
                           metric_list: Sequence[Type[ErrMetric]],
                           name: str = "Test",
                           short_mod_names: List = None,
                           remove_units: bool = True,
                           series_mask=None,
                           scale_back: bool = False,
                           put_on_ol: bool = False,
                           compare_models: bool = False,
                           overwrite: bool = True,
                           scale_over_series: bool = False,
                           fit_data: str = DEFAULT_TRAIN_SET,
                           titles: List = None,
                           plot_folder: str = None,
                           fig_size: Tuple[float, float] = None,
                           set_title: bool = True) -> None:
    """Plots the evaluated performance for multiple models.

    `series_mask` can be used to select subset of series.

    Args:
        model_list: List with all models
        parts: The list of strings specifying the parts of the data.
        metric_list: List with error metrics.
        name: The plot file name.
        short_mod_names: The simplified model names to put in plot.
        remove_units: Whether to remove units in labels.
            (Use when 'scale_back' is False and data is still scaled)
        series_mask: Mask specifying which series to use.
        scale_back: Whether to scale the errors to original values.
        put_on_ol: Whether to put the file into Overleaf folder.
        compare_models: Whether to compare different models instead of different
            parts.
        overwrite: Whether to overwrite an existing file.
        scale_over_series: Scale the errors to have a similar maximum. Especially
            useful if `scale_back` is True.
        fit_data: The portion of the data the models were fit on.
        titles: List with titles for the plots.
        plot_folder: The folder to put the plot into. Overwrites `put_on_ol`!
        fig_size:
        set_title:
    """
    metric_names = [m.name for m in metric_list]

    # Check if models are compatible
    from dynamics.base_model import check_model_compatibility
    check_model_compatibility(model_list)
    if not np.any(model_list[0].data.is_scaled):
        scale_back = False

    # Prepare the labels
    series_descs, mod_names, scaling = _get_descs(model_list, remove_units,
                                                  series_mask, short_mod_names)

    # Load data
    data_array, inds = _load_all_model_data(model_list,
                                            parts, metric_names,
                                            series_mask, fit_data=fit_data)
    n_models, n_parts, n_series, n_metrics, n_steps = data_array.shape

    # Switch lists
    if compare_models:
        parts, mod_names = mod_names, parts

    # Check model compatibility
    ind = model_list[0].p_out_inds
    for m in model_list:
        assert np.array_equal(ind, m.p_out_inds), "Fuck"

    # Series scaling
    if scale_back:
        for k in range(n_series):
            for i, m in enumerate(metric_list):
                dat = data_array[:, :, k, i, :]
                # Scale the errors
                if scale_back:
                    m_and_sd = scaling[ind[k]]
                    data_array[:, :, k, i, :] = m.scaling_fun(dat, m_and_sd[1])

    # Scale errors of each series to have same maximum
    scaling_fac_arr = None
    if scale_over_series and n_series > 1:
        max_arr = np.amax(data_array, axis=(0, 1, 4))
        min_arr = np.amin(max_arr, axis=0)
        scaling_fac_arr = np.empty_like(max_arr)
        check_shape(max_arr, (n_series, n_metrics))
        for k in range(n_series):
            for i, m in enumerate(metric_list):
                inv_scaling_fac = max_arr[k, i] / min_arr[i]
                scaling_fac_arr[k, i] = inv_scaling_fac
                data_array[:, :, k, i, :] = data_array[:, :, k, i, :] / inv_scaling_fac
    if scale_over_series and n_series == 1:
        print("Set scale_over_series to False!")
    if plot_folder is None:
        plot_folder = OVERLEAF_IMG_DIR if put_on_ol else EVAL_MODEL_PLOT_DIR
    for model_ind, m_name in enumerate(mod_names):

        # Skip loop if file exists
        f_name = f"{name}_{m_name}" if name != "" else m_name
        plot_path = os.path.join(plot_folder, f_name)
        if os.path.isfile(plot_path + ".pdf") and not overwrite:
            continue

        share_y = False
        dt = model_list[0].data.dt

        if len(parts) > len(joint_styles):
            warnings.warn("Duplicate plot styles!")

        ax1 = None
        for ct_m, m in enumerate(metric_list):
            last_met = ct_m == len(metric_list) - 1
            subplot_ind = 311 + ct_m
            ax1 = plt.subplot(subplot_ind, sharex=ax1, sharey=ax1 if share_y else None)

            # Set ticks
            tick_label = [mins_to_str(dt * int(i)) + f"\n{int(i)} Step{'s' if i > 1 else ''}" for i in inds]
            if ct_m != len(metric_list) - 1:
                tick_label = ["" for _ in inds]
            if not last_met:
                plt.setp(ax1.get_xticklabels(), visible=False)
            plt.xticks(inds, tick_label)
            plt.ylabel(m.name)

            # Plot all series
            for set_id, set_name in enumerate(parts):
                for series_id in range(n_series):
                    # Get labels and errors
                    s_desc = series_descs[ind[series_id]]
                    if scale_back:
                        fac = scaling_fac_arr[series_id, ct_m] if scale_over_series else None
                        s_desc = _trf_desc_units(s_desc, m, fac)
                    lab = f"{set_name}: {s_desc}"
                    if compare_models:
                        si = data_array[set_id, model_ind, series_id, ct_m]
                    else:
                        si = data_array[model_ind, set_id, series_id, ct_m]

                    # Plot
                    plt.plot(inds, si, joint_styles[set_id],
                             c=clr_map[series_id], label=lab)

            # Add title, legend and x-label
            if ct_m == 0 or scale_back:
                if ct_m == 0:
                    t = titles[model_ind] if titles is not None else m_name
                    if set_title:
                        plt.title(t)
                plt.legend()
            # if ct_m == len(metric_names) - 1:
            #     plt.xlabel(f"Steps [{mins_to_str(dt)}]")

        # Construct the path of the plot
        save_figure(plot_path, size=fig_size)


def plot_visual_all_in_one(all_plt_dat: List[Tuple], save_name: str,
                           add_errors: bool = False,
                           series_mask: List[int] = None,
                           fig_size: Tuple[int, int] = None) -> None:
    """Stacks multiple dataset plots on top of each other.

    Args:
        all_plt_dat: List with tuples (Dataset, title_and_ylab, Any)
        save_name: The path name of the generated plot.
        add_errors: Whether to add errors in a box.
        series_mask: Used to select subset of series to plot.
        fig_size:
    """
    if series_mask is not None:
        assert max(series_mask) < len(all_plt_dat) and min(series_mask) >= 0, "Fuck"
        all_plt_dat = [all_plt_dat[k] for k in series_mask]

    n_series = len(all_plt_dat)
    assert n_series > 0, "Fuck this!"

    ax1 = None
    for ct_m, tup in enumerate(all_plt_dat):
        subplot_ind = 11 + ct_m + n_series * 100
        ax1 = plt.subplot(subplot_ind, sharex=ax1)

        # Plot all series
        ds, t, cn = tup
        if ct_m > 0:
            t[0] = None
        plot_dataset(ds,
                     show=False,
                     title_and_ylab=t,
                     save_name=None,
                     new_plot=False)

        # Add a box with errors
        if add_errors:
            metrics = [MAE, MaxAbsEer]
            unscaled_dat = ds.get_unscaled_data()
            s1, s2 = unscaled_dat[:, 0], unscaled_dat[:, 1]
            e = [m.err_fun(s1, s2) for m in metrics]
            text_str = f"{metrics[0].name}: {e[0]:.2f}\n{metrics[1].name}: {e[1]:.2f}"
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            plt.text(0.05, 0.95, text_str, transform=ax1.transAxes, fontsize=14,
                     verticalalignment='top', bbox=props)
    save_figure(save_name, size=fig_size)


def plot_valve_opening(timestamps: np.ndarray, valves: np.ndarray, save_name: str,
                       t_timestamps: np.ndarray = None, t_setpoints: np.ndarray = None,
                       t_setpoints_meas: np.ndarray = None):
    # Check and extract shape of valves
    check_shape(valves, (-1, -1))
    n_data = len(timestamps)
    assert n_data == valves.shape[0], f"Incompatible shape: {valves} and {timestamps}"
    n_valves = valves.shape[1]

    ax = host_subplot(111)
    ax.set_xlabel("Time")
    ax.set_ylabel("Valve closed / open (0 / 1)")

    # Plot all valves
    for k in range(n_valves):
        _plot_helper(timestamps, valves[:, k], grid=True, dates=True,
                     m_col=clr_map[k], label=f"Valve {k + 1}", ax=ax)

    # Plot temperature setpoint
    if t_setpoints is not None:

        # Check input
        n_t_sp = len(t_setpoints)
        if t_timestamps is None:
            t_timestamps = timestamps
        assert n_t_sp == t_timestamps.shape[0], \
            f"Incompatible shape: {t_timestamps} and {t_setpoints}"

        # Plot written temperature setpoints
        ax_twin = ax.twinx()
        ax_twin.set_ylim(MIN_TEMP - 2, MAX_TEMP + 2)
        ax_twin.set_ylabel("Temperature [°C]")
        _plot_helper(t_timestamps, t_setpoints, grid=True, dates=True,
                     m_col=clr_map[n_valves], label=f"Temp. setpoint", ax=ax_twin)

        # Plot feedback temperature setpoints
        if t_setpoints_meas is not None:
            _plot_helper(timestamps, t_setpoints_meas, grid=True, dates=True,
                         m_col=clr_map[n_valves + 1],
                         label=f"Temp. setpoint feedback", ax=ax_twin)

    # Save
    plt.legend(loc=7)
    save_figure(save_name, size=LONG_FIG_SIZE)


def make_experiment_table(arr_list: List[np.ndarray], name_list, series_list,
                          f_name: str,
                          caption: str = "Test",
                          lab: str = "exp_tab",
                          metric_eval: Any = None,
                          metrics_names: Any = None,
                          tot_width: float = 0.8,
                          daily_averages: bool = True,
                          content_only: bool = False,
                          cell_colors: Any = None):
    assert len(name_list) == len(arr_list) > 0
    first_arr = arr_list[0]
    n_days = first_arr.shape[1]
    assert first_arr.shape[0] == len(series_list)
    n_metrics = len(metrics_names) if metrics_names is not None else 0
    if not daily_averages:
        n_days = 0
    n_columns = n_days + n_metrics
    init_str = ""
    if cell_colors is None:
        cell_colors = []

    if not content_only:
        init_str += "\\begin{table}[ht]\n"
        init_str += "\\centering\n"

        day_w = 0.33
        other_w = (tot_width - day_w)
        s = "|".join([f"p{{{day_w / n_columns}\\textwidth}}"] * n_columns)
        init_str += f"\\begin{{tabular}}{{|p{{{(1.0 - tot_width) * other_w}" \
                    f"\\textwidth}}|p{{{tot_width * other_w}\\textwidth}}|{s}|}}\n"
        init_str += "\\hline\n"

    # Add titles
    init_str += "Agent & Data"
    if daily_averages:
        init_str += " & " + " & ".join([f'Day {i + 1}' for i in range(n_days)])
    if metric_eval is not None:
        init_str += " & " + " & ".join([i for i in metrics_names])
    init_str += f"\\\\\n\\hline\n"

    def to_str(f_val, col=""):
        str_base = f"\\cellcolor{{{col}!25}}" if col != "" else ""
        if f_val != f_val:
            return str_base
        else:
            return str_base + f"{f_val:.2f}"

    # Add rows
    for ct, a in enumerate(name_list):
        init_str += f"\\hline\n"
        for ct_s, s in enumerate(series_list):
            s_cols = [(i, c) for (i, k, c) in cell_colors if k == ct_s]
            if ct_s == 0:
                init_str += a
            init_str += f" & {s}"
            if daily_averages:
                init_str += " & " + " & ".join([to_str(v) for v in arr_list[ct][ct_s]])
            if metric_eval is not None:
                met_str_list = [to_str(metric_eval[ct][ct_s, i])
                                                for i in range(n_metrics)]
                for (i, c) in s_cols:
                    met_str_list[i] = f"\\cellcolor{{{c}!25}}" + met_str_list[i]
                init_str += " & " + " & ".join(met_str_list)
            init_str += "\\\\\n"

    # Remaining part
    init_str += "\\hline\n"

    if not content_only:
        init_str += "\\end{tabular}\n"
        init_str += f"\\caption{{{caption}}}\n\\label{{tab:exp_{lab}}}\n"
        init_str += "\\end{table}\n"

    print(init_str)

    # Save
    f_path = os.path.join(OVERLEAF_DATA_DIR, f_name + ".tex")
    if not os.path.isfile(f_path):
        with open(f_path, "w") as f:
            f.write(init_str)


def plot_hist(vals, save_path: str, fig_size: Any = None, x_lab: str = None,
              title: str = None, tol: float = 0.0,
              bin_size: float = 1.0) -> None:
    """Plots a histogram of `vals`.

    Args:
        vals:
        save_path:
        fig_size:
        x_lab:
        title:
        tol:
        bin_size:
    """
    max_val = np.nanmax(vals)
    n_bins = int(np.ceil(max_val / bin_size)) + 1
    bins = [-bin_size + tol + i * bin_size for i in range(n_bins + 1)]
    plt.hist(vals, bins, density=True)
    plt.grid(True)
    plt.ylabel(f"Probability" if bin_size == 1.0 else f"Probability density")
    if title is not None:
        plt.title(title)
    if x_lab is not None:
        plt.xlabel(x_lab)
    if fig_size is None:
        fig_size = (8, 4.5)
    save_figure(save_path, size=fig_size, auto_fmt_time=False)


def eval_env_evaluation(all_rewards, all_states, ep_marks,
                        episode_len: int, plt_base_name: str,
                        bin_size: float = 0.25):

    print("Evaluating environment quality...")

    n_agents, n_steps, n_tot_rewards = all_rewards.shape

    n_eps = n_steps // episode_len

    res = np.empty((n_eps, ), dtype=np.float32)
    case_ind = np.empty((n_eps, ), dtype=np.int32)
    tol = 0.0001

    for k in range(n_eps):
        k0, k1 = k * episode_len, (k + 1) * episode_len
        assert np.max(ep_marks[:, k0:k1]) == np.min(ep_marks[:, k0:k1])
        mark = ep_marks[0, k0:k1]

        curr_states = all_states[:, k0:k1]
        heating = np.all(curr_states[:, :, 2] > curr_states[:, :, 4])
        cooling = np.all(curr_states[:, :, 2] < curr_states[:, :, 4])
        # print("Mean", np.mean(curr_states[:, :, 4]))

        open_r_temp = curr_states[0, :, 4]
        closed_r_temp = curr_states[1, :, 4]

        if heating and np.all(mark):
            case_ind[k] = 1
            res[k] = np.maximum(0, np.max(closed_r_temp - open_r_temp))
        elif cooling and np.all(np.logical_not(mark)):
            case_ind[k] = -1
            res[k] = np.maximum(0, np.max(-closed_r_temp + open_r_temp))
        else:
            case_ind[k] = 0
            res[k] = np.nan
            # print(f"Special case found, mark: {mark[0]}")

    n_tot = np.sum(case_ind != 0)
    plot_hist(res, plt_base_name + f"_{n_tot}",
              x_lab="Temperature deviation [°C]",
              bin_size=bin_size, tol=tol)
    n_heat = np.sum(case_ind == 1)
    if n_heat > 1:
        plot_hist(res[case_ind == 1], plt_base_name + f"_Heat_{n_heat}",
                  x_lab="Temperature deviation [°C]", bin_size=bin_size / 5, tol=tol)
    if np.sum(case_ind == -1) > 1:
        plot_hist(res[case_ind == -1], plt_base_name + f"_Cool_{n_tot - n_heat}",
                  x_lab="Temperature deviation [°C]", bin_size=bin_size, tol=tol)
