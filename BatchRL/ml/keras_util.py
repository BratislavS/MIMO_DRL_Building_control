import os
from typing import Sequence, Union, Any

from keras import Model
from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Sequential
from keras.regularizers import l2

from ml.sklearn_util import SKLoader
from util.util import dynamic_model_dir, create_dir, DEFAULT_TRAIN_SET, DEFAULT_EVAL_SET

KerasModel = Union[Sequential, Model, SKLoader]


def soft_update_params(model_to_update, other, lam: float = 1.0) -> None:
    """
    Soft parameter update:
    :math:`\\theta' = \\lambda * \\theta + (1 - \\lambda) * \\theta'`

    :param model_to_update: Model where the parameters will be updated.
    :param other: Model where the parameters of the model should be updated to.
    :param lam: Factor determining how much the parameters are updated.
    :return: None
    """
    params = other.get_weights()

    if lam == 1.0:
        model_to_update.set_weights(params)
        return
    else:
        orig_params = model_to_update.get_weights()
        for ct, el in enumerate(orig_params):
            orig_params[ct] = (1.0 - lam) * el + lam * params[ct]
        model_to_update.set_weights(orig_params)


def max_loss(y_true, y_pred):
    """
    Loss independent of the true labels for
    optimization without it, i.e. maximize y_pred
    directly.

    :param y_true: True labels, not used here.
    :param y_pred: Output for maximization.
    :return: -y_pred since this will be minimized.
    """
    return -y_pred


def getMLPModel(mlp_layers: Sequence = (20, 20), out_dim: int = 1,
                trainable: bool = True,
                dropout: bool = False,
                bn: bool = False,
                ker_reg: float = 0.01):
    """Returns a sequential MLP keras model.

    Args:
        mlp_layers: The numbers of neurons per layer.
        out_dim: The output dimension.
        trainable: Whether the parameters should be trainable.
        dropout: Whether to use dropout.
        bn: Whether to use batch normalization.
        ker_reg: Kernel regularization weight.

    Returns:
        Sequential keras MLP model.
    """
    model = Sequential()
    if bn:
        model.add(BatchNormalization(trainable=trainable, name="bn0"))

    # Add layers
    n_fc_layers = len(mlp_layers)
    for i in range(n_fc_layers):
        next_layer = Dense(mlp_layers[i],
                           activation='relu',
                           trainable=trainable,
                           kernel_regularizer=l2(ker_reg),
                           name=f"dense{i}")
        model.add(next_layer)
        if bn:
            model.add(BatchNormalization(trainable=trainable, name=f"bn{i + 1}"))
        if dropout:
            model.add(Dropout(0.2))

    # Reduce to 1D
    last = Dense(out_dim, activation=None, trainable=trainable, name="last_dense")
    model.add(last)
    return model


class KerasBase:
    """Base class for keras models.

    Provides an interface for saving and loading models.
    """

    model_path: str = dynamic_model_dir
    m: KerasModel

    def _model_path_name(self, name, train_data: str):
        ext = f"_TDP_{train_data}" if train_data != DEFAULT_TRAIN_SET else ""
        return self.get_path(f"{name}{ext}")

    def save_model(self, m, name: str,
                   train_data: str = DEFAULT_TRAIN_SET) -> None:
        """Saves a keras model.

        Args:
            m: Keras model.
            name: Name of the model.
            train_data: Train data specifier.
        """
        m.save(self._model_path_name(name, train_data))

    def load_if_exists(self, m, name: str,
                       train_data: str = DEFAULT_TRAIN_SET) -> bool:
        """Loads the keras model if it exists.

        Returns true if it could be loaded, else False.

        Args:
            m: Keras model to be loaded.
            name: Name of model.
            train_data: Train data specifier.

        Returns:
             True if model could be loaded else False.
        """
        full_path = self._model_path_name(name, train_data)
        found = os.path.isfile(full_path)
        # print(f"Model: {full_path}, found? {found}")
        if found:
            m.load_weights(full_path)
        return found

    def get_path(self, name: str, ext: str = ".h5",
                 env: Any = None,
                 hop_eval_set: str = DEFAULT_EVAL_SET) -> str:
        """
        Returns the path where the model parameters
        are stored.

        Args:
            name: Model name.
            ext: Filename extension.
            env: Environment with a name attribute.
            hop_eval_set: Hyperparameter opt. evaluation set.

        Returns:
            Model parameter file path.
        """
        res_folder = self.model_path
        if env is not None and hasattr(env, "name"):
            hop_ext = f"_HEV_{hop_eval_set}" if hop_eval_set != DEFAULT_EVAL_SET else ""
            res_folder = os.path.join(res_folder, env.name + hop_ext)
            create_dir(res_folder)
        return os.path.join(res_folder, name + ext)
