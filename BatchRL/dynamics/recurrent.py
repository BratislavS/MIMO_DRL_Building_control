"""The recurrent models to be used for modeling the dynamical system.

The models are derived from `HyperOptimizableModel` or at least
from `BaseDynamicsModel`.
"""
import os
from functools import partial
from typing import Dict, Optional, Sequence, Any, Tuple, List

import numpy as np
from hyperopt import hp
from hyperopt.pyll import scope as ho_scope
from keras import backend as K
from keras.layers import GRU, LSTM, Dense, Input, Add, Concatenate, Reshape, Lambda, Activation
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import plot_model

from data_processing.dataset import SeriesConstraint, Dataset
from dynamics.base_hyperopt import HyperOptimizableModel
from ml.keras_layers import ConstrainedNoise, FeatureSlice, ExtractInput, IdRecurrent, IdDense
from tests.test_data import get_test_ds
from tests.test_keras import get_multi_input_layer_output
from util.util import EULER, create_dir, rem_first, train_decorator, DEFAULT_TRAIN_SET
from util.visualize import plot_train_history, OVERLEAF_DATA_DIR


def weighted_loss(y_true, y_pred, weights):
    """Returns the weighted MSE between y_true and y_pred.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        weights: Weights.

    Returns:
        Weighted MSE.
    """
    return K.mean((y_true - y_pred) * (y_true - y_pred) * weights)


def _constr_hp_name(hidden_sizes: Sequence,
                    n_iter_max: int,
                    lr: float,
                    gru: bool = False,
                    input_noise_std: float = None) -> str:
    """Constructs the part of the name of the network associated with the hyperparameters.

    TODO: Use make_param_ext()

    Args:
        hidden_sizes: Layer size list or tuple
        n_iter_max: Number of iterations
        lr: Learning rate
        gru: Whether to use GRU units
        input_noise_std: Standard deviation of input noise.

    Returns:
        String combining all these parameters.
    """
    arch = '_L' + '-'.join(map(str, hidden_sizes))
    ep_s = f"_E{n_iter_max}"
    lrs = f"_LR{lr:.4g}"
    n_str = '' if input_noise_std is None else f"_N{input_noise_std:.4g}"
    gru_str = '' if not gru else '_GRU'
    return ep_s + arch + lrs + n_str + gru_str


def _constr_base_name(name: str,
                      residual_learning: bool = True,
                      weight_vec: np.ndarray = None,
                      constraint_list: Sequence[SeriesConstraint] = None,
                      train_seq: bool = False,
                      ) -> str:
    """Constructs the part of the name of the network not associated with hyperparameters.

    TODO: Use make_param_ext()

     Args:
        name: Base name
        residual_learning: Whether to use residual learning.
        weight_vec: Weight vector for weighted loss.
        constraint_list: List with constraints.
        train_seq: Whether to train with sequence output.

    Returns:
        String combining all these parameters.
    """
    res_str = '' if not residual_learning else '_RESL'
    w_str = '' if weight_vec is None else '_W' + '-'.join(map(str, weight_vec))
    con_str = '' if constraint_list is None else '_CON'
    seq_str = '' if not train_seq else '_SEQ'
    return name + res_str + w_str + con_str + seq_str


def constr_name(name: str,
                hidden_sizes: Sequence,
                n_iter_max: int,
                lr: float,
                gru: bool = False,
                residual_learning: bool = True,
                weight_vec: np.ndarray = None,
                input_noise_std: float = None,
                constraint_list: Sequence[SeriesConstraint] = None,
                train_seq: bool = False,
                ) -> str:
    """
    Constructs the name of the network.

    Args:
        name: Base name
        hidden_sizes: Layer size list or tuple
        n_iter_max: Number of iterations
        lr: Learning rate
        gru: Whether to use GRU units
        residual_learning: Whether to use residual learning
        weight_vec: Weight vector for weighted loss
        input_noise_std: Standard deviation of input noise.
        constraint_list: List with constraints.
        train_seq: Whether to train with sequence output.

    Returns:
        String combining all these parameters.
    """
    hp_part = _constr_hp_name(hidden_sizes, n_iter_max, lr,
                              gru, input_noise_std)
    base_part = _constr_base_name(name, residual_learning, weight_vec, constraint_list, train_seq)
    return base_part + hp_part


def _extract_kwargs(hp_sample: Dict):
    """Turns a hyperopt parameter sample into a kwargs dict.

    Args:
        hp_sample: Dict with hyperopt parameters.

    Returns:
        Kwargs dict for initialization.
    """
    n_layers = hp_sample['n_layers']
    n_units = hp_sample['n_neurons']
    hidden_sizes = [n_units for _ in range(n_layers)]
    init_kwargs = {key: hp_sample[key] for key in hp_sample if key not in ['n_layers', 'n_neurons']}
    init_kwargs['hidden_sizes'] = hidden_sizes
    return init_kwargs


def construct_rnn(hidden_sizes: Sequence[int],
                  use_gru: bool = False,
                  debug: bool = False,
                  input_shape_dict: Dict = None,
                  model=None,
                  input_tensor=None,
                  n_pred: int = None,
                  cl_and_oi: Tuple = None,
                  name_ext: str = None) -> Optional[Any]:
    # Define layers
    if input_shape_dict is None:
        input_shape_dict = {}
    n_lstm = len(hidden_sizes)
    assert n_lstm > 0, "Retard!"

    rnn = GRU if use_gru else LSTM
    layer_list = []
    for k in range(n_lstm):
        ret_seq = k != n_lstm - 1
        if debug:
            lay = IdRecurrent(return_sequences=ret_seq)
        else:
            lay = rnn(int(hidden_sizes[k]),
                      return_sequences=ret_seq,
                      name=f"rnn_layer_{k}{name_ext}",
                      **input_shape_dict)
        layer_list += [lay]
        input_shape_dict = {}

    # Add last dense layer
    if n_pred is not None:
        assert n_pred > 0, "No predictions is not possible!"
        last_layer = IdDense(n=n_pred) if debug else Dense(n_pred,
                                                           activation=None,
                                                           name=f"dense_reduce{name_ext}")
        layer_list += [last_layer]

    # Output layer
    if cl_and_oi is not None:
        assert len(cl_and_oi) == 2, f"WTF is this shit: {cl_and_oi}?"
        c_list, out_inds = cl_and_oi
        if c_list is not None:
            out_constraints = [c_list[i] for i in out_inds]
            out_const_layer = ConstrainedNoise(0, consts=out_constraints,
                                               is_input=False,
                                               name=f"constrain_output{name_ext}")
            layer_list += [out_const_layer]

    # Apply layers
    for lay in layer_list:
        if model is not None:
            model.add(lay)
        if input_tensor is not None:
            input_tensor = lay(input_tensor)
    if input_tensor is not None:
        return input_tensor


class RNNDynamicModel(HyperOptimizableModel):
    """Simple RNN used for training a dynamics model.

    All parameters are defined in `__init__`. Implements
    the `HyperOptimizableModel` for hyperparameter optimization.
    """

    train_seq_len: int  #: Length of sequences used for training.
    n_feats: int  #: Number of input features in train data.

    @classmethod
    def get_base_name(cls, include_data_name: bool = True, **kwargs):
        super_keys = [
            'data',
            'out_inds',
            'in_inds'
        ]
        super_kwargs = {k: kwargs[k] for k in kwargs if k in super_keys}
        base_kwargs = {k: kwargs[k] for k in kwargs if k not in super_keys}
        # Handle None default arguments
        if base_kwargs.get('name') is None:
            base_kwargs['name'] = cls.def_name()
        if super_kwargs.get('in_inds') is None:
            super_kwargs['in_inds'], _ = cls._get_inds(None, super_kwargs['data'], True)
        if super_kwargs.get('out_inds') is None:
            super_kwargs['out_inds'], _ = cls._get_inds(None, super_kwargs['data'], False)
        b_n = _constr_base_name(**base_kwargs)
        return cls._get_full_name_static(b_name=b_n, no_data=not include_data_name, **super_kwargs)

    @classmethod
    def _hp_sample_to_kwargs(cls, hp_sample: Dict) -> Dict:
        """Converts the sample from the hyperopt space to kwargs for initialization.

        Returns:
            Dict with kwargs for initialization.
        """
        return _extract_kwargs(hp_sample)

    def get_space(self) -> Dict:
        """Defines the hyper parameter space.

        Space is larger when on Euler (`EULER` == True).

        Returns:
            Dict specifying the hyper parameter space.
        """
        n_high = 80 if EULER else 20
        hp_space = {
            'n_layers': ho_scope.int(hp.quniform('n_layers', low=1, high=4, q=1)),
            'n_neurons': ho_scope.int(hp.quniform('n_neurons', low=5, high=n_high, q=5)),
            'n_iter_max': ho_scope.int(hp.quniform('n_iter_max', low=5, high=n_high, q=5)),
            'gru': hp.choice('gru', [False, True]),
            'lr': hp.loguniform('lr', low=-5 * np.log(10), high=1 * np.log(10)),
            'input_noise_std': hp.loguniform('input_noise_std', low=-6 * np.log(10), high=-1 * np.log(10)),
        }
        return hp_space

    def conf_model(self, hp_sample: Dict) -> 'HyperOptimizableModel':
        """Configures a new model.

        Returns a model of the same type with the same output and input
        indices and the same constraints, but different hyper parameters.

        Args:
            hp_sample: Sample of hyper parameters to initialize the model with.

        Returns:
            New model.
        """
        init_kwargs = _extract_kwargs(hp_sample)
        new_mod = RNNDynamicModel(self.data,
                                  in_inds=self.in_inds,
                                  out_inds=self.out_inds,
                                  constraint_list=self.constraint_list,
                                  verbose=0,
                                  **init_kwargs)
        return new_mod

    def hyper_objective(self, *args, **kwargs) -> float:
        """Defines the objective of the hyperparameter optimization.

        Uses the hyperparameter objective from the base class.

        Returns:
            Objective loss.
        """
        return self.hyper_obj(*args, **kwargs)

    @staticmethod
    def def_name() -> str:
        """Returns the base name of this model."""
        return "basicRNN"

    def __init__(self,
                 data: Dataset,
                 name: str = None,
                 hidden_sizes: Sequence[int] = (20, 20),
                 n_iter_max: int = 10000,
                 *,
                 in_inds: np.ndarray = None,
                 out_inds: np.ndarray = None,
                 weight_vec: Optional[np.ndarray] = None,
                 gru: bool = False,
                 input_noise_std: Optional[float] = None,
                 residual_learning: bool = True,
                 lr: float = 0.001,
                 constraint_list: Sequence[SeriesConstraint] = None,
                 train_seq: bool = False,
                 verbose: int = 0):

        """Constructor, defines all the network parameters.

        Args:
            name: Base name
            data: Dataset
            hidden_sizes: Layer size list or tuple
            out_inds: Prediction indices
            in_inds: Input indices
            n_iter_max: Number of iterations
            lr: Base learning rate
            gru: Whether to use GRU units
            residual_learning: Whether to use residual learning
            weight_vec: Weight vector for weighted loss
            input_noise_std: Standard deviation of input noise.
            constraint_list: The constraints on the data series.
            verbose: The verbosity level, 0, 1 or 2.
        """
        if name is None:
            name = self.def_name()
        name_orig = name
        name = constr_name(name, hidden_sizes, n_iter_max, lr, gru,
                           residual_learning, weight_vec, input_noise_std,
                           constraint_list)
        super(RNNDynamicModel, self).__init__(data, name, out_inds, in_inds, verbose)

        # Store data
        self.train_seq_len = self.data.seq_len - 1
        self.n_feats = len(self.in_inds)

        # Store name for hyperopt
        all_kwargs = {
            'name': name_orig,
            'data': data,
            'out_inds': out_inds,
            'in_inds': in_inds,
            'residual_learning': residual_learning,
            'weight_vec': weight_vec,
            'constraint_list': constraint_list,
            'train_seq': train_seq,
        }

        # Construct base name for hop
        self.base_name = self.get_base_name(include_data_name=False, **all_kwargs)

        # Store parameters
        self.constraint_list = constraint_list
        self.hidden_sizes = np.array(hidden_sizes, dtype=np.int32)
        self.n_iter_max = n_iter_max
        self.gru = gru
        self.input_noise_std = input_noise_std
        self.weight_vec = weight_vec
        self.residual_learning = residual_learning
        self.lr = lr
        self.train_seq = train_seq

        # Build model
        self.m = self._build_model()

    @property
    def n_layers(self):
        return len(self.hidden_sizes)

    @property
    def layer_size(self):
        return self.hidden_sizes[0]

    def _build_model(self, debug: bool = False) -> Any:
        """Builds the keras RNN model and returns it.

        The parameters how to build it were passed to `__init__`.

        Returns:
             The built keras model.
        """

        # Initialize
        model = Sequential(name="rnn")
        input_shape_dict = {'input_shape': (self.train_seq_len, self.n_feats)}

        # Add noise layer
        if self.input_noise_std is not None:
            model.add(ConstrainedNoise(self.input_noise_std, consts=self.constraint_list,
                                       name="constrain_input",
                                       **input_shape_dict))
            input_shape_dict = {}

        # Build the rnn part
        construct_rnn(hidden_sizes=self.hidden_sizes,
                      use_gru=self.gru,
                      debug=debug,
                      input_shape_dict=input_shape_dict,
                      model=model,
                      input_tensor=None,
                      n_pred=self.n_pred,
                      cl_and_oi=None)

        # Define the constraint layer
        if self.constraint_list is not None:
            out_constraints = [self.constraint_list[i] for i in self.out_inds]
        else:
            out_constraints = None
        out_const_layer = ConstrainedNoise(0, consts=out_constraints,
                                           is_input=False,
                                           name="constrain_output")

        # Do residual learning
        if self.residual_learning:
            if self.train_seq:
                return NotImplementedError("Cannot combine residual and sequence output learning!")
            # Add last non-control input to output
            seq_input = Input(shape=(self.train_seq_len, self.n_feats),
                              name="input_sequences")
            m_out = model(seq_input)
            slicer = FeatureSlice(self.p_out_inds, name="get_previous_output")
            last_input = slicer(seq_input)
            final_out = Add(name="add_previous_state")([m_out, last_input])
            final_out = out_const_layer(final_out)
            model = Model(inputs=seq_input, outputs=final_out)
        else:
            model.add(out_const_layer)

        # Define loss
        if self.weight_vec is not None:
            k_constants = K.constant(self.weight_vec)
            fixed_input = Input(tensor=k_constants)
            seq_input = Input(shape=(self.train_seq_len, self.n_feats))
            model = Model(inputs=seq_input, outputs=model(seq_input))
            self.loss = partial(weighted_loss, weights=fixed_input)
        else:
            self.loss = 'mse'

        return model

    def _plot_model(self, model: Any, name: str = "Model.png",
                    expand: bool = True) -> None:
        """Plots keras model."""
        pth = self.get_plt_path(name)
        plot_model(model, to_file=pth,
                   show_shapes=True,
                   expand_nested=expand,
                   dpi=500)

    @train_decorator()
    def fit(self, verbose: int = 0, train_data: str = DEFAULT_TRAIN_SET) -> None:
        """Fit the model if it hasn't been fitted before.

        Else it loads the trained model.

        TODO: Compute more accurate val_percent for fit method!
        """
        self.fit_data = train_data

        # Define optimizer and compile
        opt = Adam(lr=self.lr)
        self.m.compile(loss=self.loss, optimizer=opt)
        if self.verbose:
            self.m.summary()
        if not EULER:
            # No model plot without GraphViz
            try:
                self._plot_model(self.m)
            except OSError:
                print("Cannot create model plot, GraphViz not installed!")

        # Prepare the data
        monitor_val_loss = train_data != "all"
        used_data = "train_val" if train_data == DEFAULT_TRAIN_SET else "all"
        input_data, output_data = self.get_fit_data(used_data)

        # Fit and save model
        val_per = 0 if not monitor_val_loss else self.data.val_percent
        h = self.m.fit(input_data, output_data,
                       epochs=self.n_iter_max,
                       initial_epoch=0,
                       batch_size=128,
                       validation_split=val_per,
                       verbose=self.verbose)

        # Add extension specifying training set
        ext = "" if train_data == DEFAULT_TRAIN_SET else f"_Data_{train_data}"
        pth = self.get_plt_path(f"TrainHist{ext}")
        plot_train_history(h, pth, val=monitor_val_loss)
        create_dir(self.model_path)

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Predicts a batch of sequences using the fitted model.

        Args:
            input_data: 3D numpy array with input data.

        Returns:
            2D numpy array with the predictions.
        """

        n = input_data.shape[0]

        # Predict
        predictions = self.m.predict(input_data)
        predictions = predictions.reshape((n, -1))
        return predictions


class PhysicallyConsistentRNN(RNNDynamicModel):

    def _build_model(self, debug: bool = False) -> Any:
        assert self.n_pred == 1, f"Invalid number of predictions: {self.n_pred}!"

        # Get scaling parameters
        do_scaling = self.data.fully_scaled
        water_in_scaling, room_t_scaling, valve_scaling = None, None, None
        if do_scaling:
            scale = self.data.scaling
            water_in_scaling = scale[2]
            room_t_scaling = scale[4]
            valve_scaling = scale[5]

        # Initialize
        seq_input = Input(shape=(self.train_seq_len, self.n_feats),
                          name="input_sequences")
        first_layer = seq_input

        # Add noise layer
        if self.input_noise_std is not None:
            first_layer = ConstrainedNoise(self.input_noise_std, consts=self.constraint_list,
                                           name="constrain_input")(seq_input)

        # Extract weather, room temperature and time
        weather_input = Lambda(lambda x: x[:, :, :2], name="extract_weather")(first_layer)
        control_input = Lambda(lambda x: x[:, :, -1:], name="extract_control")(first_layer)
        temp_time_input = Lambda(lambda x: x[:, :, 4:7], name="extract_time_r_temp")(first_layer)
        last_control_input = Lambda(lambda x: x[:, -1], name="extract_last_control")(control_input)
        water_in_temp = Lambda(lambda x: x[:, -1, 2:3], name="extract_water_in")(first_layer)
        prev_room_temp = Lambda(lambda x: x[:, -1, -2:-1], name="extract_room_temp")(first_layer)
        scaled_room_temp = prev_room_temp

        # Scale back
        if do_scaling:
            def lam(sc):
                return lambda x: sc[1] * x + sc[0]

            water_in_temp = Lambda(lam(water_in_scaling), name="scale_water_in")(water_in_temp)
            last_control_input = Lambda(lam(valve_scaling), name="scale_last_control")(last_control_input)
            scaled_room_temp = Lambda(lam(room_t_scaling), name="scale_room_temp")(scaled_room_temp)

        # Concatenate input
        base_input = Lambda(lambda x: K.concatenate(x, axis=-1),
                            name="concat_base_input")([weather_input, temp_time_input])
        correction_input = Lambda(lambda x: K.concatenate(x, axis=-1),
                                  name="concat_correction_input")([weather_input, control_input])

        # Construct base prediction (independent of control input)
        rnn_pred = construct_rnn(hidden_sizes=self.hidden_sizes,
                                 use_gru=self.gru,
                                 debug=debug,
                                 input_shape_dict=None,
                                 input_tensor=base_input,
                                 n_pred=self.n_pred,
                                 cl_and_oi=None)

        # Define the control dependent correction
        dT = Lambda(lambda x: x[0] - x[1], name="subtract")([water_in_temp, scaled_room_temp])
        update_pred = construct_rnn(hidden_sizes=self.hidden_sizes,
                                    use_gru=self.gru,
                                    debug=debug,
                                    input_shape_dict=None,
                                    input_tensor=correction_input,
                                    n_pred=self.n_pred,
                                    cl_and_oi=None,
                                    name_ext="_corr")
        act = Activation('relu')(update_pred)
        tot_correction = Lambda(lambda x: x[0] * x[1] * x[2],
                                name="compute_correction")([act, last_control_input, dT])

        # Compute the final output
        base_room_temp = Lambda(lambda x: x[0] + x[1],
                                name="base_room_temp")([prev_room_temp, rnn_pred])
        out = Lambda(lambda x: x[0] + x[1],
                     name="add_correction")([base_room_temp, tot_correction])

        model = Model(inputs=seq_input, outputs=out)
        self.loss = 'mse'

        return model

    @staticmethod
    def def_name() -> str:
        return "PC_RNN"


class RNNDynamicOvershootModel(RNNDynamicModel):
    m: Any = None  #: The base model used for prediction.
    overshoot_model: Any = None  #: The overshoot model used for training.
    n_overshoot: int  #: Number of timesteps to predict into the future.
    debug: bool = False

    def __init__(self, n_overshoot: int = 10, decay_rate: float = 1.0, debug: bool = False, **kwargs):
        """Initialize model

        Args:
            n_overshoot: The number of timesteps to consider in overshoot model.
            kwargs: Kwargs for super class.
        """
        super(RNNDynamicOvershootModel, self).__init__(**kwargs)

        self.decay_rate = decay_rate
        self.n_overshoot = n_overshoot
        self.pred_seq_len = self.train_seq_len
        self.tot_train_seq_len = n_overshoot + self.pred_seq_len
        self.debug = debug

        self._build()

    def _build(self) -> None:
        """Builds the keras model."""

        # Build base model.
        b_mod = self._build_model(self.debug)
        self.m = b_mod

        # Build train model.
        inds = self.p_out_inds
        tot_len = self.tot_train_seq_len

        def res_lay(k_ind: int):
            return Reshape((1, self.n_pred), name=f"reshape_{k_ind}")

        def copy_mod(k_ind: int):
            m = self.m
            ip = Input(rem_first(m.input_shape))
            out = m(ip)
            new_mod = Model(inputs=ip, outputs=out, name=f"base_model_{k_ind}")
            return new_mod

        # Define input
        full_input = Input((tot_len, self.n_feats))

        first_out = ExtractInput(inds, self.pred_seq_len, 0,
                                 name="first_extraction")(full_input)
        first_out = copy_mod(0)(first_out)
        all_out = [res_lay(0)(first_out)]
        for k in range(self.n_overshoot - 1):
            first_out = ExtractInput(inds, self.pred_seq_len, k + 1,
                                     name=f"extraction_{k + 1}")([full_input, first_out])
            first_out = copy_mod(k + 1)(first_out)
            all_out += [res_lay(k + 1)(first_out)]

        # Concatenate all outputs and make model
        full_out = Concatenate(axis=-2, name="final_concatenate")(all_out)
        train_mod = Model(inputs=full_input, outputs=full_out)

        # Define loss and compile
        if self.decay_rate != 1.0:
            ww = np.ones((self.n_overshoot, 1), dtype=np.float32)
            ww = np.repeat(ww, self.n_pred, axis=-1)
            for k in range(self.n_overshoot):
                ww[k] *= self.decay_rate ** k
            k_constants = K.constant(ww)
            fixed_input = Input(tensor=k_constants)
            loss = partial(weighted_loss, weights=fixed_input)
        else:
            loss = 'mse'
        opt = Adam(lr=self.lr)
        train_mod.compile(loss=loss, optimizer=opt)
        if self.verbose:
            train_mod.summary()

        # Plot and save
        if not EULER:
            self._plot_model(train_mod, "TrainModel.png", expand=False)
        self.train_mod = train_mod

    def fit(self, verbose: int = 0, train_data: str = "train") -> None:
        """Fits the model using the data from the dataset.

        TODO: Use decorator for 'loading if existing' to avoid duplicate
            code!

        Returns:
            None
        """
        self.fit_data = train_data
        if train_data != "train":
            raise NotImplementedError("Fuck")

        loaded = self.load_if_exists(self.m, self.name)
        if not loaded or self.debug:
            if self.verbose:
                self.deb("Fitting Model...")

            fit_data, _ = self.data.split_dict['train_val'].get_sequences(self.tot_train_seq_len)
            out_data = fit_data[:, self.pred_seq_len:, self.p_out_inds]
            h = self.train_mod.fit(x=fit_data, y=out_data,
                                   epochs=self.n_iter_max,
                                   initial_epoch=0,
                                   batch_size=128,
                                   validation_split=self.data.val_percent,
                                   verbose=self.verbose)

            pth = self.get_plt_path("TrainHist")
            plot_train_history(h, pth)
            create_dir(self.model_path)
            self.m.save_weights(self.get_path(self.name))
        else:
            self.deb("Restored trained model")


def _to_str(t: Any):
    """Helper function for :func:`make_latex_hop_table`."""
    if isinstance(t, float):
        return f"{t:.4g}"
    elif isinstance(t, str) or np.issubdtype(type(t), np.integer):
        return f"{t}"
    elif isinstance(t, bool):
        return "GRU" if t else "LSTM"
    else:
        print(f"Fuck off: {t}, type: {type(t)}!")


def _add_row(curr_str, rnn_model_list, s, attr: str):
    """Helper function for :func:`make_latex_hop_table`."""
    for ct, m in enumerate(rnn_model_list):
        end = "\\\\\n" if ct == len(rnn_model_list) - 1 else "&"
        at_val = _to_str(getattr(m, attr))
        curr_str += f" ${s} = {at_val}$ {end}"
    return curr_str


def make_latex_hop_table(rnn_model_list: List[RNNDynamicModel],
                         mod_names: List,
                         tot_w: float = 0.9,
                         f_name: str = None,
                         caption: str = None,
                         lab: str = None) -> str:
    """Creates a latex table as string with the values of the hyperparameters.

    Args:
        rnn_model_list: List with models.
        mod_names: Name of the models as title of columns.
        tot_w: Total fraction of textwidth to use.
        f_name: File saving name.
        caption: Caption of table.
        lab: Label of table.

    Returns:
        The latex table as string.
    """
    init_str = "\\begin{table}[ht]\n"
    init_str += "\\centering\n"

    n_mods = len(rnn_model_list)
    w = tot_w / n_mods
    s = "|".join([f"p{{{w}\\textwidth}}"] * n_mods)
    init_str += f"\\begin{{tabular}}{{|{s}|}}\n"

    # Add titles
    init_str += "\\hline\n" + " & ".join(mod_names) + "\\\\\n"
    init_str += f"\\hline\\hline\n"

    # Add rows
    init_str = _add_row(init_str, rnn_model_list, "n_l", "n_layers")
    init_str = _add_row(init_str, rnn_model_list, "n_c", "layer_size")
    init_str = _add_row(init_str, rnn_model_list, "n_{ep}", "n_iter_max")
    init_str = _add_row(init_str, rnn_model_list, "\\eta", "lr")
    init_str = _add_row(init_str, rnn_model_list, "\\sigma_i", "input_noise_std")
    init_str = _add_row(init_str, rnn_model_list, "Cell", "gru")

    # Remaining part
    init_str += "\\hline\n\\end{tabular}\n"
    init_str += f"\\caption{{{caption}}}\n\\label{{tab:hyp_{lab}}}\n"
    init_str += "\\end{table}\n"

    # Save or return
    if f_name is not None:
        f_path = os.path.join(OVERLEAF_DATA_DIR, f_name + ".tex")
        if not os.path.isfile(f_path):
            with open(f_path, "w") as f:
                f.write(init_str)
        print(init_str)
    else:
        return init_str


##########################################################################
# Testing stuff

RNN_TEST_DATA_NAME = "RNNTestDataset"


def test_rnn_models():
    """Tests the RNN model classes.

    Raises:
        AssertionError: If a test fails.
    """
    # Create dataset for testing
    n, n_feat = 200, 7
    dat = np.arange(n * n_feat).reshape((n, n_feat))
    c_inds = np.array([4, 6], dtype=np.int32)
    ds = get_test_ds(dat, c_inds, name=RNN_TEST_DATA_NAME, dt=4 * 60)
    ds.seq_len = 6
    ds.val_percent = 0.3
    train_s_len = ds.seq_len - 1
    ds.split_data()

    # Define model
    p_inds = np.array([0, 1, 3], dtype=np.int32)
    test_kwargs = {'hidden_sizes': (9,),
                   'n_iter_max': 2,
                   'input_noise_std': 0.001,
                   'lr': 0.01}
    fix_kwargs = {'data': ds,
                  'residual_learning': True,
                  'weight_vec': None,
                  'out_inds': p_inds,
                  'constraint_list': None}
    n_over = 3
    mod_test_overshoot = RNNDynamicOvershootModel(n_overshoot=n_over,
                                                  name="DebugOvershoot",
                                                  debug=True,
                                                  **test_kwargs,
                                                  **fix_kwargs)
    full_sam_seq_len = n_over + train_s_len

    # Try hyperopt
    mod_test_2 = RNNDynamicModel.from_best_hp(**fix_kwargs, name="TestRNN")
    mod_test_2.optimize(1)

    # Check sequence lengths
    assert full_sam_seq_len == mod_test_overshoot.tot_train_seq_len, "Train sequence length mismatch!"
    assert train_s_len == mod_test_overshoot.train_seq_len, "Seq len mismatch!"

    # Check model debug output
    lay_input = np.array([dat[:full_sam_seq_len]])
    test_mod = mod_test_overshoot.train_mod
    test_base_mod = mod_test_overshoot.m
    l_out = get_multi_input_layer_output(test_mod, lay_input, learning_phase=0)
    l_out_base = get_multi_input_layer_output(test_base_mod, lay_input[:, :train_s_len, :], learning_phase=0)
    assert np.allclose(l_out_base[0], l_out[0, 0]), "Model output not correct!"

    # Find intermediate layer output
    ex1_out = first_out = None
    for k in test_mod.layers:
        if k.name == 'extraction_1':
            ex1_out = k.output
        if k.name == 'first_extraction':
            first_out = k.output
    assert ex1_out is not None and first_out is not None, "Layers not found!"
    first_ex_out = Model(inputs=test_mod.inputs, outputs=first_out)
    l_out_first = get_multi_input_layer_output(first_ex_out, lay_input, learning_phase=0)
    assert np.allclose(lay_input[0, :train_s_len], l_out_first), "First extraction incorrect!"
    sec_ex_out = Model(inputs=test_mod.inputs, outputs=ex1_out)
    exp_out = np.copy(lay_input[:, 1:(1 + train_s_len), :])
    exp_out[0, -1, p_inds] = l_out[0, 0]
    l_out_sec = get_multi_input_layer_output(sec_ex_out, lay_input, learning_phase=0)
    assert np.allclose(exp_out, l_out_sec), "Second extraction incorrect!"
    # print(lay_input)
    # print(l_out_sec)
    # print(l_out)
    # raise NotImplementedError("Fuck")

    # Try fitting
    mod_test_overshoot.fit()

    print("RNN models test passed :)")
