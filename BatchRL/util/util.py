"""A few general functions with multiple use cases.

Includes a few general python functions,
a lot of numpy transformations and also some tools
to handle the datetime of python and numpy. Also some
tests of these functions are included.
"""
import builtins as __builtin__
import datetime
import os
import pickle
import random
import shutil
import socket
import subprocess
import sys
import time
import warnings
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Union, List, Tuple, Any, Sequence, TypeVar, Dict, Callable, Optional, Type

import numpy as np

# BASE_DIR = ".."
BASE_DIR = Path(os.path.dirname(os.path.realpath(__file__))).parent.parent
TEMP_DIR = os.path.join(BASE_DIR, "Temp")

#######################################################################################################
# Platform specific stuff

# Determine platform, assuming we are on Euler if it is not a windows platform.
EULER: bool = os.name != 'nt'

DEFAULT_END_DATE: str = "2019-12-31"
DEFAULT_TRAIN_SET: str = "train"  #: Default training set
DEFAULT_EVAL_SET: str = "val"  #: Default evaluation / validation set
DEFAULT_ROOM_NR: int = 43
DEFAULT_SEQ_LEN: int = 20


def get_rl_steps(eul: bool = EULER):
    return 2000000 if eul else 1000


#######################################################################################################
# Relative paths, relative to folder "BatchRL"

# Define paths
MODEL_DIR = os.path.join(BASE_DIR, "Models")
dynamic_model_dir = os.path.join(MODEL_DIR, "Dynamics")

#######################################################################################################
# Random seed

SEED: int = 42  #: Default seed value.


def fix_seed(seed: int = SEED) -> None:
    """Fixes the random seed."""
    np.random.seed(seed)
    random.seed(seed)


#######################################################################################################
# Typing

# Type for general number
Num = Union[int, float]

# Type for general array, including 0-D ones, i.e. single numbers.
Arr = Union[Num, np.ndarray]

# Type for list or single element of specified type.
T = TypeVar('T')
LOrEl = Union[Sequence[T], T]

# Indices
IndArr = np.ndarray
IndT = Union[Sequence[int], IndArr]


#######################################################################################################
# Python stuff

def execute_powershell(script_path: str, args: str):
    """Executes a powershell script."""
    p = subprocess.Popen(f"powershell.exe {script_path} {args}",
                         stdout=sys.stdout)
    p.communicate()


def data_ext(date_str: str, room_nr: int, eval_set: str = DEFAULT_EVAL_SET) -> str:
    """Creates an extension string to differentiate different datasets."""
    data_ext_date = f"_D_{date_str}" if date_str != DEFAULT_END_DATE else ""
    data_ext_room_nr = f"_RN_{room_nr}" if room_nr != DEFAULT_ROOM_NR else ""
    data_ext_eval = f"_EVAL_{eval_set}" if eval_set != DEFAULT_EVAL_SET else ""
    return data_ext_date + data_ext_room_nr + data_ext_eval


def force_decorator_factory(wait: float = 0.5, verbose: bool = True):
    """Decorator factory for force executing a function.

    Calls the same function again and again recursively until
    it terminates without an error. May be used when an error
    based on the current internet connection might happen. Might lead
    to stack overflow if the waiting time before calling gain (`wait`)
    is very small.

    Args:
        wait: Waiting time in seconds until next call.
        verbose: Whether to print the error that happened.

    Returns:
        The decorator.
    """

    def force_decorator(fun):
        """The decorator to be returned."""

        @wraps(fun)
        def forced_fun(*args, **kwargs):
            """The decorated function."""
            try:
                fun(*args, **kwargs)
            except Exception as e:
                if verbose:
                    print(f"Exception: {e} happened, retrying...")
                time.sleep(wait)
                forced_fun(*args, **kwargs)

        return forced_fun

    return force_decorator


def stdout_redirection_test():
    """Some experiment with redirecting console output."""

    import ctypes
    import io
    import tempfile

    ##############################################################
    if sys.version_info < (3, 5):
        libc = ctypes.CDLL(ctypes.util.find_library('c'))
    else:
        if hasattr(sys, 'gettotalrefcount'):  # debug build
            libc = ctypes.CDLL('ucrtbased')
        else:
            libc = ctypes.CDLL('api-ms-win-crt-stdio-l1-1-0')

    ##############################################################

    class StdoutRedirector(object):
        """
        """
        s_str: str
        p_count: int = 0

        def __init__(self, stream, s_name: str = "stdout"):
            self.s_str = s_name
            self.stream = stream
            self.s = getattr(sys, self.s_str)
            self.original_stdout_fd = self.s.fileno()
            self.saved_stdout_fd = os.dup(self.original_stdout_fd)

        def _redirect_stdout(self, to_fd, original_fd):
            """Redirect stdout to the given file descriptor."""
            # Flush the C-level buffer stdout
            libc.fflush(None)
            # Flush and close sys.stdout - also closes the file descriptor (fd)
            getattr(sys, self.s_str).close()
            # Make original_stdout_fd point to the same file as to_fd
            os.dup2(to_fd, original_fd)
            # Create a new sys.stdout that points to the redirected fd
            setattr(sys, self.s_str, io.TextIOWrapper(os.fdopen(original_fd, 'wb')))

        def _enter_helper(self):
            self.t_file = tempfile.TemporaryFile(mode='w+b', buffering=0)
            self._redirect_stdout(self.t_file.fileno(), self.original_stdout_fd)

        def __enter__(self):
            self._enter_helper()

            self.old_print = __builtin__.print

            def new_print(*args, **kwargs):
                self.old_print(*args, **kwargs)

                # Exit
                self.stream.write(b"Printed:\n")
                self._exit_helper()
                self.stream.write(b"End\n")

                # Enter
                self.saved_stdout_fd = os.dup(self.original_stdout_fd)
                self._enter_helper()

            __builtin__.print = new_print
            return self

        def _exit_helper(self):
            self._redirect_stdout(self.saved_stdout_fd, self.original_stdout_fd)
            # Copy contents of temporary file to the given stream
            self.t_file.flush()
            self.t_file.seek(0, io.SEEK_SET)
            t_file_read = self.t_file.read()
            self.stream.write(t_file_read)
            self.t_file.close()
            os.close(self.saved_stdout_fd)

        def __exit__(self, exc_type, exc_value, exc_traceback):
            self._exit_helper()
            __builtin__.print = self.old_print

    f = io.BytesIO()

    with StdoutRedirector(f):

        print('foobar')
        print(12)
        warnings.warn("this is a warning")
        print("after warning")
        libc.puts(b'this comes from C')
        print("after puts")
        os.system('echo and this is from echo')
        print("after echo")
    std_out_val = f.getvalue().decode("utf-8")
    print(f'Got stdout: "{std_out_val}"')


def str2bool(v) -> bool:
    """Converts a string to a boolean.

    Raises:
        ValueError: If it cannot be converted.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', '1.0'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', '0.0'):
        return False
    else:
        raise ValueError(f"Boolean value expected, got {v}")


class FunClassBase:

    def __call__(self, *args, **kwargs):
        pass


def fun_to_class(fun: Callable):
    class FunClass(FunClassBase):

        def __call__(self, *args, **kwargs):
            return fun(*args, **kwargs)

    return FunClass()


def prog_verb(verbose: int) -> int:
    """Propagates verbosity.

    Can be used if a function taking the `verbose` argument
    calls another function using it.

    Args:
        verbose: The verbosity level.

    Returns:
        The verbosity level for next level function calls.
    """
    return max(0, verbose - 1)


S = TypeVar('S')


def cast_to_subclass(class_inst, sub_class: Type[S]) -> S:
    """Cast an instance of a class to the given subclass.

    Args:
        class_inst: The instance to cast.
        sub_class: The class type to cast the instance to.

    Raises:
        AssertionError: If `class_inst` is not a subclass of type
            `sub_class`!
    """
    assert isinstance(class_inst, sub_class), "Casting not possible!"
    ret_val: sub_class = class_inst
    return ret_val


def print_decorator(print_fun: Callable):
    def print_fun_dec(*args, **kwargs):
        print_fun("   ", *args, **kwargs)

    return print_fun_dec


def stdout_decorator(print_fun: Callable):
    def print_fun_dec(text, *args, **kwargs):
        print_fun(f"    {text}", *args, **kwargs)

    return print_fun_dec


class ProgWrap(object):
    """Context manager that wraps the body with output to the console.

    If `verbose` is False, this does absolutely nothing.
    Allows for nesting, since it is not using the carriage return.
    """

    def __init__(self, init_str: str = "Starting...", verbose: bool = True):
        self.init_str = init_str
        self.v = verbose
        self.orig_print = None
        self.std_out = None

    def __enter__(self):
        if self.v:
            print(self.init_str)
            self.orig_print = sys.stdout.write
            sys.stdout.write = stdout_decorator(sys.stdout.write)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.v:
            sys.stdout.write = self.orig_print
            ext = " Done." if exc_type is None else " Failed!"
            print(self.init_str + ext)


class ProgWrapV2(object):
    """Context manager that wraps the body with output to the console.

    If `verbose` is False, this does absolutely nothing.
    Allows for nesting, since it is not using the carriage return.
    """

    def __init__(self, init_str: str = "Starting...", verbose: bool = True):
        self.init_str = init_str
        self.v = verbose
        self.orig_print = None
        self.std_out = None

    def __enter__(self):
        if self.v:
            print(self.init_str)
            self.orig_print = __builtin__.print
            __builtin__.print = print_decorator(print)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.v:
            __builtin__.print = self.orig_print
            print(self.init_str + " Done.")


def print_if_verb(verb: Union[bool, int] = True, *args, **kwargs):
    """Prints the given stuff if `verb` is True."""
    if verb:
        print(*args, **kwargs)


def now_str() -> str:
    """Returns the current time as a string.

    Example: '2020_01_08T08_45_03'
    E.g. for usage in filenames.
    """
    n = str(np.datetime_as_string(np.datetime64('now')))
    return n.replace(":", "_").replace("-", "_")


def yeet(msg: str = "YEET") -> None:
    """Raises an exception."""
    raise ValueError(msg)


def to_list(lore: LOrEl) -> List:
    if not isinstance(lore, list):
        return [lore]
    return lore


def param_dict_to_name(d: Dict) -> str:
    """Turns a dict of parameters to an extension string."""
    i = [i for i in d.items()]
    return make_param_ext(i)


def make_param_ext(l: List[Tuple[str, Any]]) -> str:
    """Converts a list of parameters to a string.

    Can be used as an extension of a file name to differentiate
    files generated with different parameters.

    Args:
        l: List of (Name, parameter) tuples.

    Returns:
        String combining all parameters.
    """
    s = ""
    for t in l:
        pre_str, el = t
        if (type(el) is bool and not el) or el is None:
            pass
        else:
            s += "_" + pre_str
        if type(el) is bool or el is None:
            continue
        elif hasattr(el, "name"):
            s += el.name
        elif type(el) in [int, str]:
            s += str(el)
        elif type(el) is float:
            s += f"{el:.4g}"
        elif type(el) in [list, tuple]:
            s += '-'.join(map(str, el))
        else:
            raise ValueError(f"Type: {type(el)} of {el} not supported!")
    return s


def tot_size(t: Tuple[int, ...]) -> int:
    """Computes the product of all numbers in `t`.

    Returns 0 for empty `t`.
    """
    res = 1 if len(t) > 0 else 0
    for k in t:
        res *= k
    return res


def scale_to_range(x: Num, tot_len: Num, ran: Sequence[Num]) -> float:
    """Interval transformation.

    Assumes `x` is in [0, `tot_len`] and scales it affine linearly
    to the interval `ran`.

    Args:
        x: Point to transform.
        tot_len: Length of initial interval.
        ran: Interval to transform into.

    Returns:
        New point in requested interval.
    """
    # Check input
    assert tot_len >= np.nanmax(x) and np.nanmin(x) >= 0.0, "Invalid values!"
    assert len(ran) == 2, "Range must have length 2!"
    assert ran[1] > ran[0], "Interval must have positive length!"

    # Compute output
    d_ran = ran[1] - ran[0]
    return ran[0] + x / tot_len * d_ran


def check_and_scale(action: Num, tot_n_actions: int, interval: Sequence[Num]):
    """Checks if `action` is in the right range and scales it.

    Works for an array of actions. Ignores nans.

    Args:
        action: The action to scale.
        tot_n_actions: The total number of possible actions.
        interval: The range to scale `action` into.

    Returns:
        The scaled action.
    """
    if not 0 <= np.nanmin(action) or not np.nanmax(action) <= tot_n_actions:
        raise ValueError(f"Action: {action} not in correct range!")
    cont_action = scale_to_range(action, tot_n_actions - 1, interval)
    return cont_action


def linear_oob(x: Arr, bounds: Sequence[Num]) -> Arr:
    """Same as :func:`linear_oob_penalty` but compatible for numpy arrays."""
    assert bounds[0] <= bounds[1], "Invalid bounds!"
    low = np.where(x < bounds[0], bounds[0] - x, 0)
    high = np.where(x > bounds[1], x - bounds[1], 0)
    return low + high


def linear_oob_penalty(x: Num, bounds: Sequence[Num]) -> float:
    """Computes the linear penalty for `x` not lying within `bounds`.

    Args:
        x: Value that should be within bounds.
        bounds: The specified bounds.

    Returns:
        Penalty value.
    """
    assert bounds[0] <= bounds[1], "Invalid bounds!"
    if x < bounds[0]:
        return bounds[0] - x
    elif x > bounds[1]:
        return x - bounds[1]
    return 0


def rem_first(t: Tuple) -> Tuple:
    """Removes first element from tuple.

    Args:
        t: Original tuple.

    Returns:
        New tuple without first value.
    """
    assert len(t) >= 1, "Tuple must have at least one element!"
    lis = [i for i in t]
    return tuple(lis[1:])


def get_if_not_none(lst: Sequence, indx: int, default=None):
    """Returns a list element if list is not None, else the default value.

    Args:
        lst: List of elements or None
        indx: List index.
        default: Default return value

    Returns:
        List element at position indx if lst is not None, else default.
    """
    return default if lst is None else lst[indx]


def apply(list_or_el, fun):
    """Applies the function fun to each element of `list_or_el`.

    If it is a list, else it is applied directly to `list_or_el`.

    Args:
        list_or_el: List of elements or single element.
        fun: Function to apply to elements.

    Returns:
        List or element with function applied.
    """
    if isinstance(list_or_el, list):
        return [fun(k) for k in list_or_el]
    else:
        return fun(list_or_el)


def extract_args(args: Optional[List], *def_vals,
                 raise_too_many_error: bool = False) -> List:
    """Extracts single arguments from parsed argument list.

    Default values need to be specified.
    """
    n = len(def_vals)
    assert n > 0, f"No default values provided!"
    ret = list(def_vals)
    if args is not None:
        for ct, a in enumerate(args):
            if ct < n:
                ret[ct] = a
            else:
                if raise_too_many_error:
                    yeet("Too many arguments specified!")
                else:
                    warnings.warn(f"Ignored argument: {args[ct:]} and following...")
                    break
    return ret


def remove_files_in_sub_folders(base_dir: str, bool_fun: Callable,
                                remove_empty_dirs: bool = True,
                                verbose: bool = True) -> None:
    """Removes files in sub-folder according to a pattern.

    Args:
        base_dir:
        bool_fun: Function taking filename and returning bool deciding
            whether to delete file.
        remove_empty_dirs:
        verbose: Verbosity.
    """
    for sub_dir in os.listdir(base_dir):
        # Get full path
        full_sub_path = os.path.join(base_dir, sub_dir)

        # Check if it is a file instead of a folder
        if os.path.isfile(full_sub_path):
            if verbose:
                print(f"Found unexpected file: {full_sub_path}")
            continue

        # Find sub files (and folders)
        sub_files = os.listdir(full_sub_path)

        # Delete folder if empty
        if len(sub_files) == 0 and remove_empty_dirs:
            print(f"Removing folder: {sub_dir}")
            os.rmdir(full_sub_path)

        # Iterate over files in sub-folder
        for f in sub_files:
            f_path = os.path.join(full_sub_path, f)

            # Check if it is actually a folder
            if os.path.isdir(f):
                if verbose:
                    print(f"Found unexpected folder: {f} in {full_sub_path}")
                continue

            # Remove
            if bool_fun(f):
                if verbose:
                    print(f"Removing file: {f}")
                os.remove(f_path)


def repl(el, n: int) -> List:
    """Constructs a list with `n` equal elements 'el'.

    If el is not a primitive type, then it might
    give a list with views on el.

    Args:
        el: Element to repeat.
        n: Number of times.

    Returns:
        New list with `n` elements.
    """
    return [el for _ in range(n)]


def b_cast(l_or_el, n: int) -> List:
    """Returns a list with `n` repeated elements `l_or_el`.

    Checks if `l_or_el` is a list or not, if it is and
    it already has length `n`, it is returned.

    Args:
        l_or_el: List of elements or element.
        n: Length of list.

    Returns:
        list

    Raises:
        ValueError: If `l_or_el` is a list and does not have `n` elements.
    """
    if isinstance(l_or_el, list):
        if len(l_or_el) == n:
            return l_or_el
        raise ValueError(f"Broadcast failed, len({l_or_el}) != {n}!")
    return repl(l_or_el, n)


class CacheDecoratorFactory(object):
    """Decorator for caching results of a function.

    Function output and function input is stored in a list
    and returned if the same input is given to the decorated function.

    TODO: Make it work for non-member functions!!
    """

    n: List  #: List of function arguments.
    d: List  #: List of function outputs.

    def __init__(self, n_list: List = None, data_list: List = None):
        """Initialize the decorator.

        If no lists are provided, the
        results are stored in this class.

        Args:
            n_list: List where the input is stored.
            data_list: List where the function output is stored.
        """
        self.n = [] if n_list is None else n_list
        self.d = [] if data_list is None else data_list
        # print("Init decorator!!!")

    def __call__(self, f):
        """Decorates the function `f`.

        Args:
            f: The function to be decorated.

        Returns:
            The decorated function.
        """

        @wraps(f)
        def decorated(s, n: Union[Tuple, int], *args, **kwargs):
            """The actual decorator.

            Args:
                s: Self of the class whose member function is decorated.
                n: The unique input to the function.
                *args: Arguments.
                **kwargs: Keyword arguments.

            Returns:
                The decorated function.
            """
            if n in self.n:
                i = self.n.index(n)
                return self.d[i]
            else:
                dat = f(s, n, *args, **kwargs)
                self.n += [n]
                self.d += [dat]
                return dat

        return decorated


class TestDecoratorFactory(object):
    """Testing decorator.

    Prints different messages for AssertionErrors
    and other errors.
    This sucks because it fucks up the debugging.

    TODO: Solve this! (Class decorator?)
    TODO: Find out how or decide to remove!
    """

    def __init__(self, msg: str = "Test failed!"):
        """Initialize the decorator.

        Args:
            msg: Error message .
        """
        self.m = msg

    def __call__(self, f):
        """Decorates the function `f`.

        Args:
            f: The function to be decorated.

        Returns:
            The decorated function.
        """

        def decorated(*args, **kwargs):

            try:
                f(*args, **kwargs)
            except AssertionError as a:
                print(f"{self.m}-test failed!")
                raise a
            except Exception as e:
                print(f"Exception: {e}")
                raise AssertionError(f"Unexpected error happened in test {self.m}")

        return decorated


def train_decorator():
    """Decorates trains :)

    Decorator factory for fit method of ML model.

    It tries loading the model from disk,
    if that fails the original fitting method is called.
    If the model can be loaded from disk, no actual fitting
    is done.

    Assumes `self` has a model `m` as member
    variable and a name `name`, further the methods
    `self.load_if_exists(self.m, self.name, train_data=train_data)`
    and `self.save_model(self.m, self.name, train_data=train_data)` that
    load and save the model respectively need to be implemented.
    """

    def decorator(fit):
        """The actual decorator."""

        @wraps(fit)
        def decorated(self, verbose: int = 0,
                      train_data: str = DEFAULT_TRAIN_SET, **kwargs):
            """The decorated function."""
            self.fit_data = train_data

            loaded = self.load_if_exists(self.m, self.name, train_data=train_data)
            if not loaded:
                # Set seed for reproducibility
                np.random.seed(SEED)

                # Fit and save
                if verbose:
                    print("Fitting Model...")
                fit(self, verbose, train_data, **kwargs)
                self.save_model(self.m, self.name, train_data=train_data)
            elif verbose:
                print("Restored trained model")

        return decorated

    return decorator


#######################################################################################################
# NEST stuff

def clean_desc(nest_desc: str) -> str:
    """Cleans a description string of the NEST database.

    Removes the measurement code from the string containing
    the description of the measurement series.

    Args:
        nest_desc: The description from the database.

    Returns:
        The clean description.
    """
    if nest_desc[:4] == "65NT":
        return nest_desc.split(" ", 1)[1]
    return nest_desc


def add_dt_and_t_init(m: Sequence, dt_mins: int, dt_init: np.datetime64) -> None:
    """Adds dt and t_init to each metadata dictionary in `m`.

    Args:
        m: List with all the metadata dictionaries.
        dt_mins: Number of minutes in a timestep.
        dt_init: Time of first timestep.
    """
    for ct, e in enumerate(m):
        m[ct]['t_init'] = dt_to_string(np_datetime_to_datetime(dt_init))
        m[ct]['dt'] = dt_mins


def split_desc_units(desc: str) -> Tuple[str, str]:
    """Splits a description into a title and a unit part.

    Unit needs to be in square brackets, e.g.: [unit]."""
    parts = desc.split("[")
    if len(parts) > 2:
        raise ValueError("String cannot be split.")
    if len(parts) == 1:
        return parts[0], ""
    p1, p2 = parts
    return p1, f"[{p2}"


def w_temp_str(h_in_and_out) -> str:
    """Constructs a string for title based on water temperatures.

    Args:
        h_in_and_out: Water temperatures in and out.

    Returns:
        String
    """
    assert len(h_in_and_out) == 2
    h_in, h_out = h_in_and_out
    h = h_in > h_out
    title_ext = f"In / Out temp: {h_in:.3g} / {h_out:.3g} C"
    suf = "Heating: " if h else "Cooling: "
    title_ext = suf + title_ext
    return title_ext


#######################################################################################################
# Os functions

def create_dir(dirname: str) -> None:
    """Creates directory if it doesn't exist already.

    Args:
        dirname: The directory to create.
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def rem_files_and_dirs(base_dir: str, pat: str, anywhere: bool = False) -> None:
    """Removes all files / folders in the directory `base_dir` based on `pat`.

    If `anywhere` is True, then all files and directories that contain
    the `pat` anywhere are removed. Otherwise only if they contain
    `pat` at the beginning of their name.

    Args:
        base_dir: Base directory.
        pat: String specifying the pattern of stuff to delete.
        anywhere: Whether the pattern can be contained anywhere in the name
            and not just at the start.
    """
    pat_len = len(pat)

    # Define function to choose files to delete.
    def cond(f_name: str) -> bool:
        if anywhere:
            return pat in f_name
        else:
            return f_name[:pat_len] == pat

    # Iterate over files / dirs in `EVAL_MODEL_PLOT_DIR`.
    for f in os.listdir(base_dir):
        if cond(f):
            fol = os.path.join(base_dir, f)
            if os.path.isdir(fol):
                shutil.rmtree(fol)
            else:
                os.remove(fol)


#######################################################################################################
# Datetime conversions

def np_datetime_to_datetime(np_dt: np.datetime64) -> datetime:
    """Convert from numpy datetime to datetime.

    Args:
        np_dt: Numpy datetime.

    Returns:
        Python datetime.
    """
    ts = (np_dt - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
    dt = datetime.utcfromtimestamp(ts)
    return dt


def datetime_to_np_datetime(dt: datetime) -> np.datetime64:
    """Convert from datetime to numpy datetime.

    Args:
        dt: Python datetime.

    Returns:
        Numpy datetime object.
    """
    return np.datetime64(dt)


def dt_to_string(dt: datetime) -> str:
    """Convert datetime to string.
    """
    return str(dt)


def string_to_dt(s: str) -> datetime:
    """Convert string to datetime.

    Assumes smallest unit of time in string are seconds.
    """
    return datetime.strptime(s, '%Y-%m-%d %H:%M:%S')


def str_to_np_dt(s: str) -> np.datetime64:
    """Convert string to numpy datetime64.

    Args:
        s: Date string.

    Returns:
        np.datetime64
    """
    dt = string_to_dt(s)
    return datetime_to_np_datetime(dt)


def np_dt_to_str(np_dt: np.datetime64) -> str:
    """
    Converts a single datetime64 to a string.

    Args:
        np_dt: np.datetime64

    Returns:
        String
    """

    dt = np_datetime_to_datetime(np_dt)
    return dt_to_string(dt)


def mins_to_str(mins: int) -> str:
    """Converts the integer `mins` to a string.

    Args:
        mins: Number of minutes.

    Returns:
        String representation.
    """
    return str(mins) + 'min' if mins < 60 else str(mins / 60) + 'h'


def floor_datetime_to_min(dt, mt: int) -> np.ndarray:
    """Rounds date- / deltatime64 `dt` down to `mt` minutes.

    In a really fucking cumbersome way!

    Args:
        dt: Original deltatime.
        mt: Number of minutes.

    Returns:
        Floored deltatime.
    """
    # Check if mt is valid.
    assert mt > 0, f"Timestep: {mt} must be positive!"
    if mt >= 60:
        assert mt % 60 == 0, f"Timestep: {mt} must be divisible by 60 if it is more than 1h.!"
    else:
        assert 60 % mt == 0, f"Timestep: {mt} must divide 60 if it is smaller!"

    # Convert to python datetime
    dt = np.array(dt, dtype='datetime64[s]')
    dt64 = np.datetime64(dt)
    ts = (dt64 - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
    pdt = datetime.utcfromtimestamp(ts)

    # Subtract remainder minutes (, remainder hours, minutes) and seconds
    h = pdt.hour
    minutes = pdt.minute
    if mt > 60:
        # Subtract remainder hours if mt > 60
        h = h % (mt // 60)
        dt -= np.timedelta64(h, 'h')
    else:
        # Only subtract remainder minutes if mt < 60
        minutes = minutes % mt
    dt -= np.timedelta64(pdt.second, 's')
    dt -= np.timedelta64(minutes, 'm')
    return dt


def n_mins_to_np_dt(mins: int) -> np.timedelta64:
    """Converts an int (assuming number of minutes) to a numpy deltatime object."""
    return np.timedelta64(mins, 'm')


def ts_per_day(n_min: int) -> int:
    """Computes the number of timesteps in a day.

    Returns the number of time steps in a day when
    one timestep is `n_min` minutes.

    Args:
        n_min: Length of timestep in minutes.

    Returns:
        Number of timesteps in a day.

    Raises:
        ValueError: If the result would be a float.
    """
    if (24 * 60) % n_min != 0:
        raise ValueError(f"Number of mins in a day not divisible by n_min: {n_min}")
    return 24 * 60 // n_min


def day_offset_ts(t_init: Union[str, np.datetime64], mins: int = 15,
                  remaining: bool = True) -> int:
    """Computes the number of timesteps of length `mins` minutes until the next day starts.

    Args:
        t_init: The reference time.
        mins: The number of minutes in a timestep.
        remaining: Whether to return the number of remaining timesteps or
            the number of passed timesteps (False).

    Returns:
        Number of timesteps until next day.
    """
    if type(t_init) == str:
        np_t_init = str_to_np_dt(t_init)
    else:
        np_t_init = t_init
    t_0 = np.datetime64(np_t_init, 'D')
    dt_int = np.timedelta64(mins, 'm')
    n_ts_passed = int((np_t_init - t_0) / dt_int)
    tot_n_ts = int(np.timedelta64(1, 'D') / dt_int)
    return tot_n_ts - n_ts_passed if remaining else n_ts_passed


def get_min_diff(t1: datetime, t2: datetime = None) -> float:
    """Computes the time difference in minutes between two datetimes.

    `t2` must be a later datetime than `t1`.
    If `t2` is None, the current datetime will be used.
    """
    if t2 is None:
        t2 = datetime.now()
    d1_ts = time.mktime(t1.timetuple())
    d2_ts = time.mktime(t2.timetuple())
    return (d2_ts - d1_ts) / 60


# Create paths
create_dir(MODEL_DIR)
create_dir(dynamic_model_dir)


def internet(host="8.8.8.8", port=53, timeout=0.5):
    """Checks if internet connection is available.

    FROM: https://stackoverflow.com/questions/3764291/checking-network-connection
    Host: 8.8.8.8 (google-public-dns-a.google.com)
    OpenPort: 53/tcp
    Service: domain (DNS/TCP)
    """
    try:
        socket.setdefaulttimeout(timeout)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))
        return True
    except socket.error as ex:
        print(f"No internet connection available: {ex}")
        return False


def skip_if_no_internet(f):
    """TestCase method decorator.

    Skips test method if there is no internet connection.
    """

    def wrapper(self, *args, **kwargs):
        if not internet():
            self.skipTest("No internet connection available!")
        else:
            f(self, *args, **kwargs)

    return wrapper


def save_pickle(name: str, data: Any) -> None:
    """Save data using pickle."""
    with open(name, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(name: str) -> Any:
    """Load pickled data."""
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data
