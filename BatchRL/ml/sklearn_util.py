import pickle

from typing import TYPE_CHECKING

from util.util import param_dict_to_name

if TYPE_CHECKING:
    from dynamics.classical import SKLearnModel


def get_skl_model_name(skl_model) -> str:
    """Defines a name for the skl model for saving."""
    params = skl_model.get_params()
    ext = param_dict_to_name(params)
    name = skl_model.__class__.__name__ + ext
    return name


class SKLoader:
    """Wrapper class for sklearn models to be used as a Keras model
    in terms of saving and loading parameters.

    Enables the use of `train_decorator` with the fit() method.
    """

    def __init__(self, skl_mod, parent: 'SKLearnModel'):
        self.skl_mod = skl_mod
        self.p = parent

    def load_weights(self, full_path: str):
        with open(full_path, "rb") as f:
            mod = pickle.load(f)
        self.skl_mod = mod
        self.p.skl_mod = mod
        self.p.is_fitted = True

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.skl_mod, f)
