"""Provides some high-level initialization function.

The function :func:`load_room_env` can be used to fully
load a RL env, :func:`load_room_models` may be used to initialize
multiple dynamics models and :func:`get_model` to load just
a single model.
"""

import os
from typing import List, Dict

import numpy as np
from sklearn.linear_model import MultiTaskLassoCV

from agents.base_agent import RL_MODEL_DIR
from data_processing.data import choose_dataset_and_constraints
from data_processing.dataset import Dataset, DatasetConstraints
from dynamics.base_model import BaseDynamicsModel
from dynamics.battery_model import BatteryModel
from dynamics.classical import SKLearnModel
from dynamics.composite import CompositeModel
from dynamics.const import ConstModel
from dynamics.recurrent import RNNDynamicModel, PhysicallyConsistentRNN, RNNDynamicOvershootModel
from dynamics.sin_cos_time import SCTimeModel
from envs.base_dynamics_env import DEFAULT_ENV_SAMPLE_DATA
from envs.dynamics_envs import RangeT, FullRoomEnv, RoomBatteryEnv, LowHighProfile, HeatSampler
from util.util import DEFAULT_TRAIN_SET, DEFAULT_END_DATE, DEFAULT_ROOM_NR, DEFAULT_EVAL_SET, prog_verb, data_ext, \
    ProgWrap, DEFAULT_SEQ_LEN, make_param_ext

# Define the models by name
#: List of partial models need to be combined to create complete model.
base_rnn_models = [
    "WeatherFromWeatherTime_RNN",
    # "Apartment_RNN",
    "RoomTempFromReduced_RNN",
    # "RoomTemp_RNN",
    # "WeatherFromWeatherTime_Linear",
    # "PhysConsModel",
    # "Full_RNN",
]
#: List of complete models that can be used for the RL env.
full_models = [
    "FullState_Comp_ReducedTempConstWaterWeather",
    "FullState_Comp_TempConstWaterWeather",
    "FullState_Comp_WeatherAptTime",
    "FullState_Naive",
    "FullState_Comp_Phys",
]
#: Short names for the models above.
full_models_short_names = [
    "Weather, Constant Water, Reduced Room Temp",
    "Weather, Constant Water, Room Temp.",
    "Weather, joint Room and Water",
    "Naive",
    "Weather, Constant Water, Consistent Room Temp",
]

DEFAULT_D_FAC = 0.3
used_d_fac = DEFAULT_D_FAC  # 0.001  # DEFAULT_D_FAC


def _convert_to_short(name: str):
    try:
        # Get env name
        env_name = name.split("_")[0]
        if env_name == "FullRoom":
            env_name = "Room"
        elif env_name != "RoomBattery":
            return None

        # Get room number
        r_ind = name.find("_Room")
        if r_ind == -1:
            r_num = DEFAULT_ROOM_NR
        else:
            r_num = int(name[(r_ind + 5):].split("_")[0])

        # Get date
        req_date = "2020-01-21"
        assert name.find(req_date) != -1, "Fuck"
        data_date = req_date

        # Get Hop Eval set
        h_ind = name.find("_HEV_")
        if h_ind == -1:
            h_set = DEFAULT_EVAL_SET
        else:
            h_set = name[(h_ind + 5):]

        # Find model data
        ind = name.find("_CON_")
        ext = name[(ind + 5):]
        if ext[0] != "H":
            m_dat = ext.split("_")[0]
        else:
            m_dat = DEFAULT_TRAIN_SET

        # Find alpha
        ind = name.find("_AL")
        if ind == -1:
            return None
        else:
            al = int(name[(ind + 3):].split("_")[0])

        # Temp bounds
        ind = name.find("_TBD")
        if ind == -1:
            tbd = "22.0-26.0"
        else:
            tbd = name[(ind + 4):].split("_")[0]

        # RL data
        ind = name.find("_SAM_")
        if ind == -1:
            rl_dat = DEFAULT_ENV_SAMPLE_DATA
        else:
            rl_dat = name[(ind + 5):].split("_")[0]

        # Heat sampling
        ind = name.find("_RejS_")
        if ind == -1:
            sam_ext = ""
        else:
            sam_ext = "_RS-" + name[(ind + 6):].split("_")[0]

        # Make new name
        new_name = f"{env_name}_R-{r_num}_DD-{data_date}_HD-{h_set}_MD-{m_dat}_A-{al}_TBD-{tbd}_RLD-{rl_dat}{sam_ext}"
    except Exception as e:
        print(f"Invalid fucking name: {name}!!! Generated exception: {e}")
        raise

    print(new_name)
    return new_name


def rename_rl_folder():
    for m in os.listdir(RL_MODEL_DIR):
        m_path = os.path.join(RL_MODEL_DIR, m)
        m_short = _convert_to_short(m)
        if m_short is not None:
            m_path_ren = os.path.join(RL_MODEL_DIR, m_short)
            os.rename(m_path, m_path_ren)
            print(m_short)


def load_room_env(m_name: str,
                  verbose: int = 1,
                  alpha: float = 5.0,
                  include_battery: bool = False,
                  date_str: str = DEFAULT_END_DATE,
                  temp_bds: RangeT = None,
                  train_data: str = DEFAULT_TRAIN_SET,
                  room_nr: int = DEFAULT_ROOM_NR,
                  hop_eval_set: str = DEFAULT_EVAL_SET,
                  dummy_use: bool = False,
                  sample_from: str = DEFAULT_ENV_SAMPLE_DATA,
                  use_heat_sampler: bool = False,
                  d_fac: float = DEFAULT_D_FAC,
                  ):
    """Loads the complete room environment.

    Args:
        m_name: Name of the underlying (complete) model.
        verbose: Verbosity.
        alpha: Comfort violation weighting factor.
        include_battery: See :func:`get_model`.
        date_str: See :func:`get_model`.
        temp_bds: Temperature comfort bounds.
        train_data: See :func:`get_model`.
        room_nr: See :func:`get_model`.
        hop_eval_set: See :func:`get_model`.
        dummy_use: If True, the underlying model is not fitted. To be
            used if agents are only needed for prediction / evaluation and
            were already fitted.
        sample_from: Sampling portion of data when resetting env.
        use_heat_sampler: Whether to use a heating sampler to increase
            number of heating cases.
        d_fac:

    Returns:
        The loaded env.
    """
    # Print warning
    if dummy_use and verbose:
        print("Using dummy environment, will raise an error if there "
              "is no fitted agent available!")

    # Propagate verbosity
    next_verb = prog_verb(verbose)

    # Get dataset and constraints
    with ProgWrap(f"Loading model...", verbose > 0):
        mod_dict = load_room_models([m_name],
                                    use_bat_data=include_battery,
                                    from_hop=True,
                                    fit=not dummy_use,
                                    date_str=date_str,
                                    room_nr=room_nr,
                                    hop_eval_set=hop_eval_set,
                                    train_data=train_data,
                                    verbose=next_verb)
        m = mod_dict[m_name]
        ds = m.data

    # Construct string with most important parameters
    p_list = [
        ("R-", room_nr),
        ("DD-", date_str),
        ("HD-", hop_eval_set),
        ("MD-", train_data),
        ("A-", alpha),
        ("TBD-", temp_bds),
        ("RLD-", sample_from),
        ("RS-", HeatSampler if use_heat_sampler else None),
        ("NF-", d_fac if d_fac != DEFAULT_D_FAC else None),
    ]
    short_param_ext = make_param_ext(p_list)

    # Load the model and init env
    with ProgWrap(f"Preparing environment...", verbose > 0):
        general_kwargs = {
            'cont_actions': True,
            'alpha': alpha,
            'temp_bounds': temp_bds,
            'disturb_fac': d_fac,
            'dummy_use': dummy_use,
            'sample_from': sample_from,
            'verbose': verbose,
        }
        if use_heat_sampler:
            general_kwargs["rejection_sampler"] = HeatSampler

        if include_battery:
            c_prof = LowHighProfile(ds.dt)
            assert isinstance(m, CompositeModel), \
                f"Invalid model: {m}, needs to be composite!"
            env = RoomBatteryEnv(m, p=c_prof,
                                 **general_kwargs)
            env.short_name = "RoomBattery" + short_param_ext
        else:
            env = FullRoomEnv(m, n_cont_actions=1,
                              **general_kwargs)
            env.short_name = "Room" + short_param_ext

    return env


def load_room_models(name_list: List[str], *,
                     use_bat_data: bool = False,
                     date_str: str = DEFAULT_END_DATE,
                     room_nr: int = DEFAULT_ROOM_NR,
                     verbose: int = 1,
                     seq_len=DEFAULT_SEQ_LEN,
                     **model_kwargs,
                     ) -> Dict[str, BaseDynamicsModel]:
    """Loads the models specified by name in `name_list`.

    Args:
        name_list: List with names of models to be loaded.
        use_bat_data: See :func:`data_processing.data.choose_dataset_and_constraints`.
        date_str: See :func:`get_model`.
        room_nr: See :func:`get_model`.
        verbose: Verbosity.
        seq_len: See :func:`data_processing.data.choose_dataset_and_constraints`.
        **model_kwargs: Kwargs for model loading, passed to :func:`get_model`.

    Returns:
        Dictionary mapping name to model.
    """
    # Propagate verbosity
    next_verb = prog_verb(verbose)

    # Get data and constraints
    with ProgWrap(f"Loading data...", verbose > 0):
        ds, rnn_consts = choose_dataset_and_constraints(seq_len=seq_len,
                                                        add_battery_data=use_bat_data,
                                                        date_str=date_str,
                                                        room_nr=room_nr)

    # Load and fit all models
    with ProgWrap(f"Loading models...", verbose > 0):
        all_mods = {nm: get_model(nm, Dataset.copy(ds), rnn_consts,
                                  verbose=next_verb,
                                  date_str=date_str,
                                  room_nr=room_nr,
                                  **model_kwargs
                                  ) for nm in name_list}

    return all_mods


def get_model(name: str,
              ds: Dataset,
              rnn_consts: DatasetConstraints = None, *,
              from_hop: bool = True,
              fit: bool = True,
              verbose: int = 0,
              train_data: str = DEFAULT_TRAIN_SET,
              date_str: str = DEFAULT_END_DATE,
              room_nr: int = DEFAULT_ROOM_NR,
              hop_eval_set: str = DEFAULT_EVAL_SET,
              ) -> BaseDynamicsModel:
    """Loads and optionally fits a model.

    Args:
        name: The name specifying the model.
        ds: The dataset to initialize the model with.
        rnn_consts: The constraints for the recurrent models.
        from_hop: Whether to initialize the model from optimal hyperparameters.
        fit: Whether to fit the model before returning it.
        verbose: Verbosity.
        train_data: String specifying the part of the data to train the model on.
        date_str: End date string specifying data.
        room_nr: Integer specifying the room number.
        hop_eval_set: Evaluation set for the hyperparameter optimization.

    Returns:
        The requested model.
    """
    # Check input
    assert (ds.d == 8 and ds.n_c == 1) or (ds.d == 10 and ds.n_c == 2)
    battery_used = ds.d == 10

    const_kwargs = {
        'ds': ds,
        'rnn_consts': rnn_consts,
        'fit': fit,
        'from_hop': from_hop,
        'verbose': verbose,
        'train_data': train_data,
        'date_str': date_str,
        'room_nr': room_nr,
        'hop_eval_set': hop_eval_set,
    }

    # Fit if required using one step recursion
    if fit:
        mod = get_model(name, ds, rnn_consts, from_hop=from_hop, fit=False,
                        verbose=prog_verb(verbose), train_data=train_data)
        mod.fit(verbose=prog_verb(verbose), train_data=train_data)
        if train_data != "train" and verbose:
            print(f"Trained on: {train_data}")
        return mod

    if verbose and not fit:
        print(f"Loading model {name}.")

    # Load battery model if required.
    battery_mod = None
    if battery_used:
        if verbose > 0:
            print("Dataset contains battery data.")
        battery_mod = BatteryModel(dataset=ds, base_ind=8)

    # Helper function to build composite models including the battery model.
    def _build_composite(model_list: List[BaseDynamicsModel], comp_name: str):

        # Load battery model.
        if battery_used:
            assert battery_mod is not None, "Need to rethink this!"
            if fit:
                battery_mod.fit()
            model_list = model_list + [battery_mod]

        # Adjust the name for full models.
        if comp_name.startswith("FullState_"):
            if battery_used:
                comp_name = comp_name + "_Battery"
            if rnn_consts is not None:
                comp_name += "_CON"
            if train_data != DEFAULT_TRAIN_SET:
                comp_name += f"_{train_data}"
        return CompositeModel(ds, model_list, new_name=comp_name)

    # Basic parameter set
    hop_kwargs = {
        'ext': data_ext(date_str, room_nr, hop_eval_set)
    }
    hop_pars = {
        'n_iter_max': 10,
        'hidden_sizes': (50, 50),
        'input_noise_std': 0.001,
        'lr': 0.01,
        'gru': False,
    }
    nec_pars = {
        'name': name,
        'data': ds,
    }
    fix_pars = {
        'residual_learning': True,
        'constraint_list': rnn_consts,
        'weight_vec': None,
    }
    fix_pars = dict(fix_pars, **nec_pars)
    all_out = {'out_inds': np.array([0, 1, 2, 3, 5], dtype=np.int32)}
    all_in = {'in_inds': np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32)}
    all_inds = dict(all_out, **all_in)
    base_params = dict(hop_pars, **fix_pars, **all_inds)
    base_params_no_inds = dict(hop_pars, **fix_pars)

    # Choose the model
    if name == "Time_Exact":
        # Time model: Predicts the deterministic time variable exactly.
        return SCTimeModel(ds, 6)
    elif name == "FullState_Naive":
        # The naive model that predicts all series as the last seen input.
        bat_inds = np.array([0, 1, 2, 3, 5, 6, 7, 8], dtype=np.int32)
        no_bat_inds = np.array([0, 1, 2, 3, 5, 6, 7], dtype=np.int32)
        inds = bat_inds if battery_used else no_bat_inds
        return ConstModel(ds, in_inds=inds, pred_inds=inds)
    elif name == "Battery":
        # Battery model.
        assert battery_mod is not None, "I fucked up somewhere!"
        return battery_mod
    elif name == "Full_RNN":
        # Full model: Predicting all series except for the controllable and the time
        # series. Weather predictions might depend on apartment data.
        if from_hop:
            return RNNDynamicModel.from_best_hp(**fix_pars, **all_inds, **hop_kwargs)
        return RNNDynamicModel(**base_params)
    elif name == "WeatherFromWeatherTime_RNN":
        # The weather model, predicting only the weather and the time, i.e. outside temperature and
        # irradiance from the past values and the time variable.
        inds = {
            'out_inds': np.array([0, 1], dtype=np.int32),
            'in_inds': np.array([0, 1, 6, 7], dtype=np.int32),
        }
        if from_hop:
            return RNNDynamicModel.from_best_hp(**fix_pars, **inds, **hop_kwargs)
        return RNNDynamicModel(**inds, **base_params_no_inds)
    elif name == "WeatherFromWeatherTime_Linear":
        # The weather model, predicting only the weather and the time, i.e. outside temperature and
        # irradiance from the past values and the time variable using a linear model.
        inds = {
            'out_inds': np.array([0, 1], dtype=np.int32),
            'in_inds': np.array([0, 1, 6, 7], dtype=np.int32),
        }
        skl_base_mod = MultiTaskLassoCV(max_iter=1000, cv=5)
        return SKLearnModel(skl_model=skl_base_mod, **nec_pars, **inds, clip_ind=1)
    elif name == "Apartment_RNN":
        # The apartment model, predicting only the apartment variables, i.e. water
        # temperatures and room temperature based on all input variables including the weather.
        out_inds = {'out_inds': np.array([2, 3, 5], dtype=np.int32)}
        out_inds = dict(out_inds, **all_in)
        if from_hop:
            return RNNDynamicModel.from_best_hp(**fix_pars, **out_inds, **hop_kwargs)
        return RNNDynamicModel(**out_inds, **base_params_no_inds)
    elif name == "RoomTemp_RNN":
        # The temperature only model, predicting only the room temperature from
        # all the variables in the dataset. Can e.g. be used with a constant water
        # temperature model.
        out_inds = {'out_inds': np.array([5], dtype=np.int32)}
        out_inds = dict(out_inds, **all_in)
        if from_hop:
            return RNNDynamicModel.from_best_hp(**fix_pars, **out_inds, **hop_kwargs)
        return RNNDynamicModel(**out_inds, **base_params_no_inds)
    elif name == "RoomTempFromReduced_RNN":
        # The temperature only model, predicting only the room temperature from
        # a reduced number of variables. Can e.g. be used with a constant water
        # temperature model.
        inds = {
            'out_inds': np.array([5], dtype=np.int32),
            'in_inds': np.array([0, 2, 4, 5, 6, 7], dtype=np.int32),
        }
        if from_hop:
            return RNNDynamicModel.from_best_hp(**fix_pars, **inds, **hop_kwargs)
        return RNNDynamicModel(**inds, **base_params_no_inds)
    elif name == "PhysConsModel":
        # The physically consistent temperature only model.
        inds = {
            'out_inds': np.array([5], dtype=np.int32),
            'in_inds': np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32),
        }
        if from_hop:
            return PhysicallyConsistentRNN.from_best_hp(**fix_pars, **inds, **hop_kwargs)
        return PhysicallyConsistentRNN(**inds, **base_params_no_inds)
    elif name == "WaterTemp_Const":
        # Constant model for water temperatures
        return ConstModel(ds, pred_inds=np.array([2, 3], dtype=np.int32))
    elif name == "Full_RNNOvershootDecay":
        # Similar to the model "FullModel", but trained with overshoot.
        return RNNDynamicOvershootModel(n_overshoot=5,
                                        decay_rate=0.8,
                                        **base_params)
    elif name == "Full_Comp_WeatherApt":
        # The full model combining the weather only model and the
        # apartment only model to predict all
        # variables except for the control and the time variables.
        mod_weather = get_model("WeatherFromWeather_RNN", **const_kwargs)
        mod_apt = get_model("Apartment_RNN", **const_kwargs)
        mod_list = [mod_weather, mod_apt]
        return _build_composite(mod_list, name)

    elif name == "FullState_Comp_WeatherAptTime":
        # The full state model combining the weather only model, the
        # apartment only model and the exact time model to predict all
        # variables except for the control variable.
        mod_weather = get_model("WeatherFromWeatherTime_RNN", **const_kwargs)
        mod_apt = get_model("Apartment_RNN", **const_kwargs)
        model_time_exact = get_model("Time_Exact", **const_kwargs)
        mod_list = [mod_weather, mod_apt, model_time_exact]
        return _build_composite(mod_list, name)

    elif name == "FullState_Comp_FullTime":
        # The full state model combining the combined weather and apartment model
        # and the exact time model to predict all
        # variables except for the control variable.
        mod_full = get_model("Full_RNN", **const_kwargs)
        model_time_exact = get_model("Time_Exact", **const_kwargs)
        mod_list = [mod_full, model_time_exact]
        return _build_composite(mod_list, name)

    elif name == "FullState_Comp_ReducedTempConstWaterWeather":
        # The full state model combining the weather, the constant water temperature,
        # the reduced room temperature and the exact time model to predict all
        # variables except for the control variable.
        mod_wt = get_model("WaterTemp_Const", **const_kwargs)
        model_reduced_temp = get_model("RoomTempFromReduced_RNN", **const_kwargs)
        model_time_exact = get_model("Time_Exact", **const_kwargs)
        model_weather = get_model("WeatherFromWeatherTime_RNN", **const_kwargs)
        mod_list = [model_weather, mod_wt, model_reduced_temp, model_time_exact]
        return _build_composite(mod_list, name)

    elif name == "FullState_Comp_Phys":
        # The full state model combining the weather, the constant water temperature,
        # the physically consistent room temperature and the exact time model to predict all
        # variables except for the control variable.
        mod_wt = get_model("WaterTemp_Const", **const_kwargs)
        model_reduced_temp = get_model("PhysConsModel", **const_kwargs)
        model_time_exact = get_model("Time_Exact", **const_kwargs)
        model_weather = get_model("WeatherFromWeatherTime_RNN", **const_kwargs)
        mod_list = [model_weather, mod_wt, model_reduced_temp, model_time_exact]
        return _build_composite(mod_list, name)

    elif name == "FullState_Comp_TempConstWaterWeather":
        # The full state model combining the weather, the constant water temperature,
        # the reduced room temperature and the exact time model to predict all
        # variables except for the control variable.
        mod_wt = get_model("WaterTemp_Const", **const_kwargs)
        model_reduced_temp = get_model("RoomTemp_RNN", **const_kwargs)
        model_time_exact = get_model("Time_Exact", **const_kwargs)
        model_weather = get_model("WeatherFromWeatherTime_RNN", **const_kwargs)
        mod_list = [model_weather, mod_wt, model_reduced_temp, model_time_exact]
        return _build_composite(mod_list, name)
    else:
        raise ValueError("No such model defined!")
