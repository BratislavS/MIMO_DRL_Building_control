"""The hyperparameter optimization module.

Defines a class that extends the base model class `BaseDynamicsModel`
for hyperparameter optimization.
"""
import os
import pickle
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from hyperopt import fmin, tpe

from dynamics.base_model import BaseDynamicsModel
from util.share_data import upload_folder_zipped, download_and_extract_zipped_folder
from util.util import create_dir, EULER, MODEL_DIR, DEFAULT_EVAL_SET, yeet

# Define path for optimization results.
hop_path = os.path.join(MODEL_DIR, "Hop")  #: The path to all hyperopt data.
create_dir(hop_path)

OptHP = Tuple[Dict, float]  #: The type of the stored info.


def upload_hop_pars():
    print("Uploading hyperopt parameters to Google Drive.")
    upload_folder_zipped(hop_path)


def download_hop_pars():
    print("Downloading hyperopt parameters from Google Drive.")
    download_and_extract_zipped_folder("Hop", hop_path)


def check_eval_data(eval_data: str):
    if eval_data not in ["test", "val"]:
        yeet(f"Invalid evaluation set for hyperopt: {eval_data}")


def save_hp(name_hp: str, opt_hp: OptHP) -> None:
    """Save hyperparameters."""
    with open(name_hp, 'wb') as f:
        pickle.dump(opt_hp, f)


def load_hp(name_hp) -> OptHP:
    """Load hyperparameters."""
    with open(name_hp, 'rb') as f:
        opt_hp = pickle.load(f)
    return opt_hp


class HyperOptimizableModel(BaseDynamicsModel, ABC):
    """The abstract base class for models using hyperopt.

    Need to override the abstract methods and set `base_name`
    in constructor.
    """
    param_list: List[Dict] = []  #: List of tried parameters.
    base_name: str  #: Base name independent of hyperparameters.
    curr_val: float = 10e100  #: Start value for optimization.

    @abstractmethod
    def get_space(self) -> Dict:
        """Defines the hyperopt space with the hyper parameters
        to be optimized for a given model.

        Returns:
            hyperopt space definition.
        """
        pass

    @classmethod
    @abstractmethod
    def get_base_name(cls, **kwargs) -> str:
        """Returns the unique name given all the non-hyperparameter parameters."""
        pass

    @abstractmethod
    def conf_model(self, hp_sample: Dict) -> 'HyperOptimizableModel':
        """Configure new model with given parameters.

        Initializes another model with the parameters as
        specified by the sample, which is a sample of the specified
        hyperopt space.

        Args:
            hp_sample: Sample of hyperopt space.

        Returns:
            Another model with the same type as self, initialized
            with the parameters in the sample.
        """
        pass

    @abstractmethod
    def hyper_objective(self, eval_data: str = DEFAULT_EVAL_SET) -> float:
        """
        Defines the objective to be used for hyperopt.
        It will be minimized, i.e. it has to be some kind of
        loss, e.g. validation loss.
        Model assumed to be fitted first.

        Returns:
            Numerical value from evaluation of the objective.
        """
        pass

    def optimize(self, n: int = 100, verbose: int = 1,
                 eval_data: str = DEFAULT_EVAL_SET,
                 data_ext: str = "") -> Dict:
        """Does the full hyper parameter optimization with
        the given objective and space.

        Args:
            n: Number of model initializations, fits and objective
                computations.
            verbose: The verbosity level for fmin.
            eval_data: Evaluation set for the optimization.
            data_ext: Extension to differentiate used data.

        Returns:
            The optimized hyper parameters.
        """
        fit_data = "train" if eval_data == "val" else "train_val"

        hp_space = self.get_space()
        self.param_list = []

        # Load the previously optimum if exists
        save_path = self._get_opt_hp_f_name(self.base_name, ext=data_ext)
        try:
            _, self.curr_val = load_hp(save_path)
            if verbose:
                print("Found previous hyperparameters!")
        except FileNotFoundError:
            if verbose:
                print("No previous hyperparameters found!")

        # Define final objective function
        def f(hp_sample: Dict) -> float:
            """Fits model and evaluates it.

            Args:
                hp_sample: Model parameters.

            Returns:
                Value of the objective.
            """
            mod = self.conf_model(hp_sample)
            self.param_list += [hp_sample]
            mod.fit(train_data=fit_data)
            curr_obj = mod.hyper_objective(eval_data=eval_data)

            # Save if new skl_mod are better
            if curr_obj < self.curr_val:
                self.curr_val = curr_obj
                save_hp(save_path, (hp_sample, self.curr_val))
            return curr_obj

        # Do parameter search
        best = fmin(
            fn=f,
            space=hp_space,
            algo=tpe.suggest,
            max_evals=n,
            verbose=verbose > 0,
            show_progressbar=verbose > 0,
        )

        return best

    @classmethod
    def _get_opt_hp_f_name(cls, b_name: str, ext: str = ""):
        """Determines the file path given the model name."""
        return os.path.join(hop_path, f"{b_name}_OPT_HP{ext}.pkl")

    @classmethod
    def from_best_hp(cls, verbose: int = 0, ext: str = "", **kwargs):
        """Initialize a model with the best previously found hyperparameters.

        Returns:
             An instance of the same class initialized with the optimal
             hyperparameters.
        """
        base_name = cls.get_base_name(include_data_name=False, **kwargs)
        name_hp = cls._get_opt_hp_f_name(base_name, ext=ext)
        try:
            if verbose:
                print("Loading model from hyperparameters.")
            opt_hp = load_hp(name_hp)
        except FileNotFoundError:
            print(name_hp)
            raise FileNotFoundError("No hyperparameters found, need to run optimize() first!")
        hp_params, val = opt_hp
        init_params = cls._hp_sample_to_kwargs(hp_params)
        return cls(**kwargs, **init_params)

    @classmethod
    def _hp_sample_to_kwargs(cls, hp_sample: Dict) -> Dict:
        """Converts the sample from the hyperopt space to kwargs for initialization.

        Needs to be overridden if a general `hp_sample` cannot be
        passed to `__init__` as kwargs.

        Returns:
            Dict with kwargs for initialization.
        """
        return hp_sample


def optimize_model(mod: HyperOptimizableModel, verbose: bool = True,
                   n_restarts: int = None,
                   eval_data: str = DEFAULT_EVAL_SET,
                   data_ext: str = "") -> None:
    """Executes the hyperparameter optimization of a model.

    Uses `n_restarts` calls to fit, if it is None,
    uses reduced number of model trainings if not on Euler.

    Args:
        mod: Model whose hyperparameters are to be optimized.
        verbose: Whether to print the result to the console.
        n_restarts: How many models should be fitted during the
            optimization. If None, uses different default values depending
            on whether `EULER` is True or not.
        eval_data: Evaluation set for the optimization.
        data_ext: Extension to differentiate used data.
    """
    n_opt = 50 if EULER else 2
    if n_restarts is not None:
        n_opt = n_restarts
    opt_params = mod.optimize(n_opt, verbose=verbose, eval_data=eval_data,
                              data_ext=data_ext)

    if verbose:
        print(f"Optimal parameters: {opt_params}.")
