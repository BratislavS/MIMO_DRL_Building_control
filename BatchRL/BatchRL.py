"""
Command line tool for Reinforcement Learning at Empa.

.. moduleauthor:: Christian Baumann

.. argparse::
   :filename: ../MasterThesis/BatchRL/BatchRL.py
   :func: def_parser
   :prog: BatchRL.py

The `main` function runs all the necessary high-level
functions. The complicated stuff is hidden in the other
modules / packages.
"""
import argparse
import os
import warnings
from datetime import datetime
from functools import reduce
from typing import List, Tuple, Sequence, Type, Dict, Any

import numpy as np

from agents.agents_heuristic import RuleBasedAgent, get_const_agents
from agents.base_agent import upload_trained_agents, download_trained_agents
from agents.keras_agents import default_ddpg_agent, DEF_RL_LR
from data_processing.data import get_battery_data, \
    choose_dataset_and_constraints, update_data, unique_room_nr, load_room_data
from data_processing.dataset import Dataset, check_dataset_part
from dynamics.base_hyperopt import HyperOptimizableModel, optimize_model, check_eval_data, upload_hop_pars, \
    download_hop_pars
from dynamics.base_model import compare_models, check_train_str
from dynamics.battery_model import BatteryModel, clean_battery_dataset
from dynamics.load_models import base_rnn_models, full_models, full_models_short_names, get_model, load_room_models, \
    load_room_env, DEFAULT_D_FAC
from dynamics.recurrent import make_latex_hop_table, RNNDynamicModel
from envs.dynamics_envs import BatteryEnv, heat_marker, compute_room_rewards, RangeT, TEMP_BOUNDS, ROOM_ENG_FAC, \
    RoomBatteryEnv
from opcua_empa.opcua_util import analyze_valves_experiment, experiment_plot_path
from opcua_empa.run_opcua import run_rl_control, run_rule_based_control
from rest.client import check_date_str
from tests.test_util import cleanup_test_data, TEST_DIR, TestPlot
from util.numerics import MSE, MAE, MaxAbsEer, ErrMetric
from util.util import EULER, ProgWrap, prog_verb, str2bool, extract_args, DEFAULT_TRAIN_SET, \
    DEFAULT_ROOM_NR, DEFAULT_EVAL_SET, DEFAULT_END_DATE, data_ext, BASE_DIR, execute_powershell, cast_to_subclass, \
    str_to_np_dt, fix_seed
from util.visualize import plot_performance_table, plot_performance_graph, OVERLEAF_IMG_DIR, plot_dataset, \
    plot_heat_cool_rew_det, change_dir_name, plot_env_evaluation, make_experiment_table, LONG_FIG_SIZE, plot_p_g_2

# Model performance evaluation
N_PERFORMANCE_STEPS = (1, 4, 12, 24, 48)
METRICS: Tuple[Type[ErrMetric], ...] = (MSE, MAE, MaxAbsEer)
PARTS = ["Val", "Train"]


def run_tests(verbose: int = 1) -> None:
    """Runs a few rather time consuming tests.

    Args:
        verbose: Verbosity.

    Raises:
        AssertionError: If a test fails.
    """
    # Do all the tests.
    with ProgWrap(f"Running a few tests...", verbose > 0):
        # test_rnn_models()
        plot_test_case = TestPlot()
        plot_test_case.test_hist()
        # plot_test_case.test_reward_bar_plot()
        # plot_test_case.test_plot_env_evaluation()


def test_cleanup(verbose: int = 0) -> None:
    """Cleans the data that was generated for the tests.

    Args:
        verbose: Verbosity.
    """
    # Do some cleanup.
    with ProgWrap("Cleanup...", verbose=verbose > 0):
        cleanup_test_data(verbose=prog_verb(verbose))


def run_battery(do_rl: bool = True, overwrite: bool = False,
                verbose: int = 0, steps: Sequence = (24,),
                put_on_ol: bool = False,
                train_set: str = "train_val",
                date_str="2020-01-21"
                ) -> None:
    """Runs all battery related stuff.

    Loads and prepares the battery data, fits the
    battery model and evaluates some agents.

    Args:
        do_rl:
        overwrite:
        verbose: Verbosity.
        steps: Sequence of number of evaluation steps for analysis.
        put_on_ol:
        train_set: Set of data to train model on.
        date_str:
    """
    # date_str = "2020-01-21"

    # Print info to console
    next_verb = prog_verb(verbose)
    if verbose:
        print("Running battery modeling...")
        if put_on_ol:
            print("Putting images into Overleaf directory.")
        if overwrite:
            print("Overwriting existing images.")

    # Load and prepare battery data.
    with ProgWrap(f"Loading battery data...", verbose > 0):
        bat_ds = get_battery_data(date_str=date_str)
        bat_ds_ugly = Dataset.copy(bat_ds)
        clean_battery_dataset(bat_ds)
        bat_ds.standardize()
        bat_ds.split_data()
        bat_ds_ugly.standardize()
        bat_ds_ugly.split_data()

    # Initialize and fit battery model.
    with ProgWrap(f"Fitting and analyzing battery...", verbose > 0):
        bat_mod = BatteryModel(bat_ds)
        bat_mod.fit(verbose=prog_verb(verbose), train_data=train_set)
        bat_mod_bad = BatteryModel(bat_ds_ugly)
        bat_mod_bad.fit(verbose=prog_verb(verbose), train_data=train_set)
        bat_mod_bad.plot_all_data(put_on_ol=put_on_ol, overwrite=overwrite)
        bat_mod.analyze_bat_model(put_on_ol=put_on_ol, overwrite=overwrite)
        bat_mod.analyze_visually(save_to_ol=put_on_ol, base_name="Bat",
                                 overwrite=overwrite, n_steps=steps,
                                 verbose=verbose > 0)

        with ProgWrap(f"Analyzing model performance...", verbose > 0):
            parts = ["train", "val", "test"]
            bat_mod.analyze_performance(N_PERFORMANCE_STEPS, verbose=next_verb,
                                        overwrite=overwrite,
                                        metrics=METRICS,
                                        parts=parts)
            n_series = len(bat_mod.out_inds)
            for s_ind in range(n_series):
                curr_name = f"Series_{s_ind}{bat_mod.get_fit_data_ext()}"
                series_mask = [s_ind]
                plt_dir = bat_mod.get_plt_path("")[:-1]
                if put_on_ol:
                    plt_dir = OVERLEAF_IMG_DIR
                plot_performance_graph([bat_mod], parts, METRICS, "",
                                       short_mod_names=[curr_name],
                                       scale_back=True, remove_units=False,
                                       put_on_ol=put_on_ol,
                                       fit_data=train_set,
                                       series_mask=series_mask,
                                       overwrite=overwrite,
                                       titles=[""],
                                       plot_folder=plt_dir)
                plot_p_g_2([bat_mod], parts, METRICS,
                           short_mod_names=[curr_name],
                           scale_back=True, remove_units=False,
                           put_on_ol=put_on_ol,
                           fit_data=train_set,
                           series_mask=series_mask,
                           overwrite=overwrite,
                           titles=[""],
                           plot_folder=plt_dir)

    if not do_rl:
        if verbose:
            print("No RL this time.")
        return
    if verbose:
        print("Running battery RL.")

    with ProgWrap(f"Defining environment...", verbose > 0):

        # Define the environment
        bat_env = BatteryEnv(bat_mod,
                             disturb_fac=0.3,
                             cont_actions=True,
                             n_cont_actions=1)

    with ProgWrap(f"Analyzing the environment...", verbose > 0):

        # Define the agents
        const_ag_1, const_ag_2 = get_const_agents(bat_env)

        # Fit agents and evaluate.
        ag_list = [const_ag_1, const_ag_2]
        n_steps = 2 * bat_env.n_ts_per_eps
        bat_env.detailed_eval_agents(ag_list, use_noise=True, n_steps=n_steps,
                                     put_on_ol=put_on_ol, overwrite=overwrite,
                                     verbose=prog_verb(verbose),
                                     visual_eval=True,
                                     filter_good_cases=False,
                                     plot_tot_reward_cases=False,
                                     plot_constrained_actions=True,
                                     plot_tot_eval=False,
                                     eval_quality=True)

    # # Get numbers of steps
    # n_steps = get_rl_steps(True)
    # n_eval_steps = 10000 if EULER else 100
    # dqn_agent = DDPGBaseAgent(bat_env,
    #                           action_range=bat_env.action_range,
    #                           n_steps=n_steps,
    #                           gamma=0.99)
    # ag_list = [const_ag_1, const_ag_2, dqn_agent]
    # bat_env.detailed_eval_agents(ag_list,
    #                              use_noise=False,
    #                              n_steps=n_eval_steps)


def run_dynamic_model_hyperopt(use_bat_data: bool = True,
                               verbose: int = 1,
                               enforce_optimize: bool = False,
                               n_fit_calls: int = None,
                               hop_eval_set: str = DEFAULT_EVAL_SET,
                               date_str: str = DEFAULT_END_DATE,
                               room_nr: int = DEFAULT_ROOM_NR,
                               model_indices: List[int] = None) -> None:
    """Runs the hyperparameter optimization for all base RNN models.

    Does not much if not on Euler, except if `enforce_optimize`
    is True, then it will optimize anyways.

    Args:
        use_bat_data: Whether to include the battery data.
        verbose: Verbosity level.
        enforce_optimize: Whether to enforce the optimization.
        n_fit_calls: Number of fit evaluations, default if None.
        hop_eval_set: Evaluation set for the optimization.
        date_str: End date string specifying data.
        room_nr: Integer specifying the room number.
        model_indices: Indices of models for hyperparameter tuning referring
            to the list `base_rnn_models`.
    """
    assert hop_eval_set in ["val", "test"], f"Fuck: {hop_eval_set}"

    next_verb = prog_verb(verbose)
    if verbose:
        print(f"Doing hyperparameter optimization using "
              f"evaluation on {hop_eval_set} set. Using room {room_nr} "
              f"with data up to {date_str}.")

    # Check model indices and set model list
    if model_indices is not None:
        if len(model_indices) == 0:
            model_indices = None
        else:
            msg = f"Invalid indices: {model_indices}"
            assert max(model_indices) < len(base_rnn_models), msg
            assert min(model_indices) >= 0, msg
            if verbose:
                print("Not optimizing all models.")
    model_list = base_rnn_models
    if model_indices is not None:
        model_list = [model_list[i] for i in model_indices]

    # Load models
    model_dict = load_room_models(model_list,
                                  use_bat_data=use_bat_data,
                                  from_hop=False,
                                  fit=False,
                                  date_str=date_str,
                                  room_nr=room_nr,
                                  hop_eval_set=hop_eval_set,
                                  verbose=verbose)
    # Hyper-optimize model(s)
    with ProgWrap(f"Hyperoptimizing models...", verbose > 0):
        for name, mod in model_dict.items():

            # Optimize model
            if isinstance(mod, HyperOptimizableModel):
                # Create extension based on room number and data end date
                full_ext = data_ext(date_str, room_nr, hop_eval_set)

                if EULER or enforce_optimize:
                    with ProgWrap(f"Optimizing model: {name}...", next_verb > 0):
                        optimize_model(mod, verbose=next_verb > 0,
                                       n_restarts=n_fit_calls,
                                       eval_data=hop_eval_set,
                                       data_ext=full_ext)
                else:
                    print("Not optimizing!")
            else:
                warnings.warn(f"Model {name} not hyperparameter-optimizable!")
                # raise ValueError(f"Model {name} not hyperparameter-optimizable!")


def run_dynamic_model_fit_from_hop(use_bat_data: bool = False,
                                   verbose: int = 1,
                                   visual_analyze: bool = True,
                                   perf_analyze: bool = False,
                                   include_composite: bool = False,
                                   date_str: str = DEFAULT_END_DATE,
                                   train_data: str = DEFAULT_TRAIN_SET,
                                   room_nr: int = DEFAULT_ROOM_NR,
                                   hop_eval_set: str = DEFAULT_EVAL_SET,
                                   ) -> None:
    """Runs the hyperparameter optimization for all base RNN models.

    Args:
        use_bat_data: Whether to include the battery data.
        verbose: Verbosity level.
        visual_analyze: Whether to do the visual analysis.
        perf_analyze: Whether to do the performance analysis.
        include_composite: Whether to also do all the stuff for the composite models.
        date_str: End date string specifying data.
        train_data: String specifying the part of the data to train the model on.
        room_nr: Integer specifying the room number.
        hop_eval_set: Evaluation set for the hyperparameter optimization.
    """
    if verbose:
        print(f"Fitting dynamics ML models based on parameters "
              f"optimized by hyperparameter tuning. Using room {room_nr} "
              f"with data up to {date_str}. The models are fitted "
              f"using the {train_data} portion of the data. "
              f"Hyperparameter tuning used evaluation on {hop_eval_set} "
              f"set.")

    check_train_str(train_data)
    next_verb = prog_verb(verbose)

    # Create model list
    lst = base_rnn_models[:]
    if include_composite:
        lst += full_models

    # Load and fit all models
    with ProgWrap(f"Loading models...", verbose > 0):
        # Load models
        all_mods = load_room_models(lst,
                                    use_bat_data=use_bat_data,
                                    from_hop=True,
                                    fit=True,
                                    date_str=date_str,
                                    room_nr=room_nr,
                                    hop_eval_set=hop_eval_set,
                                    train_data=train_data,
                                    verbose=next_verb)

    # Analyze all initialized models
    with ProgWrap(f"Analyzing models...", verbose > 0):
        for name, m_to_use in all_mods.items():
            if verbose:
                print(f"Model: {name}")
            # Visual analysis
            if visual_analyze:
                with ProgWrap(f"Analyzing model visually...", verbose > 0):
                    m_to_use.analyze_visually(overwrite=False,
                                              verbose=next_verb > 0)

            # Do the performance analysis
            if perf_analyze:
                with ProgWrap(f"Analyzing model performance...", verbose > 0):
                    m_to_use.analyze_performance(N_PERFORMANCE_STEPS, verbose=next_verb,
                                                 overwrite=False,
                                                 metrics=METRICS)
                    n_series = len(m_to_use.out_inds)
                    for s_ind in range(n_series):
                        curr_name = f"Series_{s_ind}{m_to_use.get_fit_data_ext()}"
                        series_mask = [s_ind]
                        plt_dir = m_to_use.get_plt_path("")[:-1]
                        plot_performance_graph([m_to_use], PARTS, METRICS, "",
                                               short_mod_names=[curr_name],
                                               scale_back=True, remove_units=False,
                                               fit_data=train_data,
                                               series_mask=series_mask,
                                               titles=[""],
                                               plot_folder=plt_dir)

    # Create the performance table
    if include_composite:
        with ProgWrap("Creating performance table and plots...", verbose > 0):

            orig_mask = np.array([0, 1, 2, 3, 5])
            full_mods = [all_mods[n] for n in full_models]
            metric_names = [m.name for m in METRICS]

            # Plot the performance
            name = "EvalTable"
            if use_bat_data:
                name += "WithBat"
            plot_performance_table(full_mods, PARTS, metric_names, name,
                                   short_mod_names=full_models_short_names,
                                   series_mask=orig_mask)
            plot_name = "EvalPlot"
            plot_performance_graph(full_mods, PARTS, METRICS, plot_name + "_RTempOnly",
                                   short_mod_names=full_models_short_names,
                                   series_mask=np.array([5]), scale_back=True, remove_units=False)
            plot_performance_graph(full_mods, PARTS, METRICS, plot_name,
                                   short_mod_names=full_models_short_names,
                                   series_mask=orig_mask)


def run_room_models(verbose: int = 1,
                    put_on_ol: bool = False,
                    perf_eval: bool = False,
                    n_steps: int = None,
                    overwrite: bool = False,
                    include_battery: bool = False,
                    physically_consistent: bool = False,
                    hop_eval_set: str = DEFAULT_EVAL_SET,
                    visual_analysis: bool = True,
                    n_eval_steps: int = 10000,
                    agent_lr: float = DEF_RL_LR,
                    rbc_dt_inc: float = None,
                    eval_dict: Dict = None,
                    **env_kwargs
                    ) -> None:
    """Trains and evaluates the RL agent.

    Args:
        verbose:
        put_on_ol:
        perf_eval:
        n_steps: Number of steps to train the agent.
        overwrite:
        include_battery: Whether to use the combined env, room and battery.
        physically_consistent:
        hop_eval_set:
        visual_analysis: Whether to do the visual analysis.
        n_eval_steps: n_eval_steps: Total number of environment steps to perform
            for agent analysis.
        agent_lr: Learning rate of DDPG agent.
        rbc_dt_inc:
        eval_dict: Kwargs for evaluation.
        env_kwargs: Keyword arguments for environment.
    """
    if eval_dict is None:
        eval_dict = {}

    # Print what the code does
    if verbose:
        print("Running RL agents on learned room model.")
        if include_battery:
            print("Model includes battery.")
    next_verbose = prog_verb(verbose)

    # Select model
    m_name = "FullState_Comp_ReducedTempConstWaterWeather"
    if physically_consistent:
        m_name = "FullState_Comp_Phys"

    # if eval_list is None:
    #     eval_list = [2595, 8221, 0, 2042, 12067, None]

    # Load the model and init env
    with ProgWrap(f"Loading environment...", verbose > 0):
        env = load_room_env(m_name,
                            verbose=next_verbose,
                            include_battery=include_battery,
                            hop_eval_set=hop_eval_set,
                            **env_kwargs)

        # Run checks
        env.do_checks()
        fix_seed()

    # Define default agents and compare
    with ProgWrap(f"Initializing and fitting agents...", verbose > 0):
        closed_agent, open_agent = get_const_agents(env)
        ch_rate = 10.0 if include_battery else None
        rule_based_agent = RuleBasedAgent(env, env.temp_bounds,
                                          const_charge_rate=ch_rate,
                                          rbc_dt_inc=rbc_dt_inc)

        agent = default_ddpg_agent(env, n_steps, fitted=True,
                                   verbose=next_verbose,
                                   hop_eval_set=hop_eval_set,
                                   lr=agent_lr)

        agent_list = [open_agent, closed_agent, rule_based_agent, agent]

    # Get bounds for plotting
    b_ind = -2 if include_battery else -1
    bds = env.temp_bounds
    bounds = [(b_ind, bds)]

    # Do performance evaluation
    if perf_eval:
        with ProgWrap(f"Evaluating agents...", verbose > 0):
            disconnect_data = None
            if isinstance(env, RoomBatteryEnv):
                con_inds = env.connect_inds
                c_inds_np = [np.timedelta64(i // 4, 'h') for i in con_inds]
                disconnect_data = (1, c_inds_np, (18.0, 82.0))
            env.detailed_eval_agents(agent_list, use_noise=True, n_steps=n_eval_steps,
                                     put_on_ol=put_on_ol, overwrite=overwrite,
                                     verbose=prog_verb(verbose),
                                     plt_fun=plot_heat_cool_rew_det,
                                     episode_marker=heat_marker,
                                     visual_eval=visual_analysis,
                                     bounds=bounds,
                                     agent_filter_ind=3,
                                     disconnect_data=disconnect_data,
                                     eval_quality=not include_battery,
                                     **eval_dict)
    elif verbose > 0:
        print("No performance evaluation!")


def analyze_heating_period(start_dt, end_dt,
                           room_nr: int = DEFAULT_ROOM_NR,
                           name: str = "experiment_test",
                           verbose: int = 5,
                           overwrite: bool = False,
                           put_on_ol: bool = False,
                           agent_name: str = None,
                           metrics_eval_list: Any = False,
                           c_offs: int = 0,
                           **env_kwargs) -> np.ndarray:
    with ProgWrap(f"Analyzing experiments...", verbose > 0):
        dt = 15
        full_ds = load_room_data(start_dt=start_dt, end_dt=end_dt,
                                 room_nr=room_nr, exp_name=f"{name}_room_{room_nr}", dt=dt)

        actions = np.expand_dims(full_ds.data[:, -1:], axis=0)
        states = np.expand_dims(full_ds.data[:, :-1], axis=0)

        plt_dir = experiment_plot_path if not put_on_ol else OVERLEAF_IMG_DIR
        save_path = os.path.join(plt_dir, name)

        all_rewards = compute_room_rewards(full_ds.data[:, -1],
                                           full_ds.data[:, -2],
                                           full_ds.data[:, 2:4],
                                           **env_kwargs,
                                           dt=full_ds.dt)
        rewards = np.expand_dims(all_rewards[:, 0], axis=0)

        if verbose:
            print(f"Total rewards: {np.sum(all_rewards, axis=0)}")
            print(f"Number of steps: {full_ds.data.shape[0]}, dt = {full_ds.dt}")

        if not os.path.isfile(save_path + ".pdf") or overwrite:
            series_merging = [
                ([0, 1], "Weather"),
                ([2, 3], "Water temperatures", "[°C]")
            ]
            plot_env_evaluation(actions, states, rewards, full_ds,
                                [agent_name if agent_name is not None else "Test"],
                                save_path=save_path,
                                np_dt_init=str_to_np_dt(full_ds.t_init),
                                series_merging_list=series_merging,
                                reward_descs=["Reward"],
                                ex_ext=False,
                                color_offset=c_offs,
                                fig_size=(12, 7))
            plot_env_evaluation(actions, states, rewards, full_ds,
                                [agent_name if agent_name is not None else "Test"],
                                save_path=save_path + "_no_w_temp",
                                np_dt_init=str_to_np_dt(full_ds.t_init),
                                series_merging_list=[([0, 1], "Weather"), ],
                                series_mask=np.array([0, 1, 4]),
                                reward_descs=["Reward"],
                                show_rewards=True,
                                ex_ext=False,
                                color_offset=c_offs)
            plot_env_evaluation(actions, states, rewards, full_ds,
                                [agent_name if agent_name is not None else "Test"],
                                save_path=save_path + "_reduced",
                                np_dt_init=str_to_np_dt(full_ds.t_init),
                                series_merging_list=None,
                                series_mask=np.array([4]),
                                reward_descs=["Reward"],
                                show_rewards=False,
                                ex_ext=False,
                                color_offset=c_offs,
                                fig_size=(16, 3))
            plot_env_evaluation(actions, states, rewards, full_ds,
                                [agent_name if agent_name is not None else "Test"],
                                save_path=save_path + "_reduced_narrow",
                                np_dt_init=str_to_np_dt(full_ds.t_init),
                                series_merging_list=None,
                                series_mask=np.array([4]),
                                reward_descs=["Reward"],
                                show_rewards=False,
                                ex_ext=False,
                                color_offset=c_offs,
                                fig_size=(9, 3))
            plot_env_evaluation(actions, states, rewards, full_ds,
                                [agent_name if agent_name is not None else "Test"],
                                save_path=save_path + "_reduced_weather",
                                np_dt_init=str_to_np_dt(full_ds.t_init),
                                series_merging_list=[([0, 1], "Weather"), ],
                                series_mask=np.array([0, 1, 4]),
                                reward_descs=["Reward"],
                                show_rewards=False,
                                ex_ext=False,
                                color_offset=c_offs,
                                fig_size=(9, 4))
        elif verbose:
            print("Plot already exists!")

        # Return daily averages
        energy_used = all_rewards[:, 1]
        n_dt_per_day = 60 * 24 // dt
        n_days = energy_used.shape[0] // n_dt_per_day
        clip_at = n_dt_per_day * n_days

        out_arr = np.empty((3, n_days), dtype=np.float32)
        d_unscaled = full_ds.get_unscaled_data()
        out_arr[0, :] = np.mean(all_rewards[:clip_at, 1].reshape((n_days, n_dt_per_day)), axis=1)  # Energy
        out_arr[1, :] = np.mean(d_unscaled[:clip_at, 0].reshape((n_days, n_dt_per_day)), axis=1)  # Out. temp.
        out_arr[2, :] = np.mean(d_unscaled[:clip_at, 4].reshape((n_days, n_dt_per_day)), axis=1)  # Room temp.

        out_full = np.empty((3, clip_at), dtype=np.float32)
        out_full[0] = all_rewards[:clip_at, 1]
        out_full[1] = d_unscaled[:clip_at, 0]
        out_full[2] = d_unscaled[:clip_at, 4]

        if metrics_eval_list is not None:
            n_met = len(metrics_eval_list)

            met_eval = np.empty((3, n_met), dtype=np.float32)
            for ct_m, m in enumerate(metrics_eval_list):
                if type(m) is tuple:
                    met_eval[:, ct_m] = np.nan
                    for k in m[1]:
                        met_eval[k, ct_m] = m[0](out_full[k])
                    pass

                else:
                    for k in range(3):
                        met_eval[k, ct_m] = m(out_full[k])
            return out_arr, met_eval

        return out_arr


def analyze_experiments(room_nr: int = 41, verbose: int = 5,
                        put_on_ol: bool = False, overwrite: bool = False,
                        alpha: float = 50.0,
                        temp_bounds: RangeT = TEMP_BOUNDS):
    next_verb = prog_verb(verbose)

    # Analyze valve experiment
    with ProgWrap(f"Analyzing valve experiments...", verbose > 0):
        exp_name = "2020_01_15T21_14_51_R475_Experiment_15min_PT_0"
        analyze_valves_experiment(exp_name,
                                  compute_valve_delay=True,
                                  verbose=next_verb,
                                  put_on_ol=put_on_ol,
                                  exp_file_name="Valve_Delay_Experiment",
                                  overwrite=overwrite)
        exp_name = "2020_01_09T11_29_28_ValveToggle_Constant6min_PT_0"
        analyze_valves_experiment(exp_name,
                                  compute_valve_delay=True,
                                  verbose=next_verb,
                                  put_on_ol=put_on_ol,
                                  exp_file_name="Valve_Fast_Toggle",
                                  overwrite=overwrite)

    met_names = [
        "Min",
        "Max",
        "Mean",
        "MAE 22.0",
        "MAE 22.5",
    ]

    def mae_t(t_dev: float = 22.5):
        def mae_225(arr, *args, **kwargs) -> float:
            res = np.mean(np.abs(arr - t_dev), *args, **kwargs)
            return res.item()

        return mae_225

    met_list = [
        np.min,
        np.max,
        np.mean,
        (mae_t(22.0), [2]),
        (mae_t(22.5), [2]),
    ]

    # Define common kwargs
    kws = {'verbose': verbose,
           'alpha': alpha,
           'temp_bounds': temp_bounds,
           'put_on_ol': put_on_ol,
           'overwrite': overwrite,
           'metrics_eval_list': met_list,
           }

    # # Test data
    # if not put_on_ol:
    #     start_dt = datetime(2020, 2, 9, 12, 3, 12)
    #     end_dt = datetime(2020, 2, 11, 12, 6, 45)
    #     analyze_heating_period(start_dt, end_dt, room_nr, **kws)

    # DDPG experiment
    start_dt, end_dt = datetime(2020, 2, 5, 12, 0, 0), datetime(2020, 2, 10, 12, 0, 0)
    name = "DDPG_Exp_22_26"
    ddpg_res, ddpg_met = analyze_heating_period(start_dt, end_dt, room_nr, name,
                                                agent_name="DDPG", c_offs=3, **kws)

    # RBC experiment
    start_dt, end_dt = datetime(2020, 2, 19, 12, 0, 0), datetime(2020, 2, 24, 12, 0, 0)
    name = "RBC_Exp_22_5"
    rbc_res, rbc_met = analyze_heating_period(start_dt, end_dt, room_nr, name,
                                              agent_name="Rule-Based", c_offs=2, **kws)

    # DDPG experiment 2
    start_dt, end_dt = datetime(2020, 2, 26, 12, 0, 0), datetime(2020, 3, 2, 12, 0, 0)
    name = "DDPG_Exp_22_5"
    ddpg_2_res, ddpg_2_met = analyze_heating_period(start_dt, end_dt, room_nr, name,
                                                    agent_name="DDPG", c_offs=3, **kws)

    name_list = ["DDPG", "RBC", "DDPG 2"]
    name_list_short = name_list[:2]
    series_list = [
        f"Energy consumption [\\si{{\\watt\\hour}}]",
        "Average outside temp. [\\si{\\celsius}]",
        "Average room temp. [\\si{\\celsius}]"
    ]
    s_list_short = [
        f"Energy [\\si{{\\watt\\hour}}]",
        "Outside temp. [\\si{\\celsius}]",
        "Room temp. [\\si{\\celsius}]"
    ]
    all_res = [ddpg_res, rbc_res, ddpg_2_res]
    all_met = [ddpg_met, rbc_met, ddpg_2_met]
    for r in all_res:
        r[0, :] = ROOM_ENG_FAC * r[0, :]
    for r in all_met:
        r[0, :] = ROOM_ENG_FAC * r[0, :]
    make_experiment_table([ddpg_res, rbc_res, ddpg_2_res], name_list,
                          s_list_short, f_name="DDPG_RBC",
                          caption="Comparison of DDPG with Rule-Based controller (RBC)",
                          lab="com_ddpg_rbc",
                          metric_eval=[ddpg_met, rbc_met, ddpg_2_met],
                          metrics_names=met_names)
    cell_colors = [
        (2, 0, "red"),
        (2, 1, "blue"),
        (3, 2, "green"),
    ]
    make_experiment_table([rbc_res, ddpg_res], name_list_short[::-1],
                          s_list_short, f_name="DDPG_RBC_pres_colored",
                          caption="Comparison of DDPG with Rule-Based controller (RBC)",
                          metric_eval=[rbc_met[:, :-1], ddpg_met[:, :-1]],
                          metrics_names=met_names[:-1],
                          tot_width=0.7,
                          daily_averages=False,
                          content_only=True,
                          cell_colors=cell_colors)
    make_experiment_table([rbc_res, ddpg_res], name_list_short[::-1],
                          s_list_short, f_name="DDPG_RBC_days_only",
                          caption="Comparison of DDPG with Rule-Based controller (RBC)",
                          lab="com_ddpg_rbc_days",
                          content_only=True)
    cell_colors = [
        (0, 0, "red"),
        (0, 1, "blue"),
        (1, 2, "green"),
    ]
    make_experiment_table([rbc_res, ddpg_res], name_list_short[::-1],
                          s_list_short, f_name="DDPG_RBC_pres_colored_2",
                          caption="Comparison of DDPG with Rule-Based controller (RBC)",
                          metric_eval=[rbc_met[:, 2:-1], ddpg_met[:, 2:-1]],
                          metrics_names=met_names[2:-1],
                          tot_width=0.7,
                          daily_averages=False,
                          content_only=True,
                          cell_colors=cell_colors)


def update_overleaf_plots(verbose: int = 2, overwrite: bool = False,
                          debug: bool = False):
    # Constants
    date_str = "2020-01-21"
    room_nr = 43
    train_data, hop_eval_set = "train_val", "val"
    train_data_rl, hop_eval_set_rl = "all", "test"
    next_verb = prog_verb(verbose=verbose)

    # If debug is true, the plots are not saved to Overleaf.
    if verbose > 0 and debug:
        print("Running in debug mode!")

    # Generate the experiments plots
    with ProgWrap(f"Plotting experiments...", verbose > 0):
        with change_dir_name("Experiments"):
            analyze_experiments(put_on_ol=not debug, room_nr=41, verbose=next_verb,
                                overwrite=overwrite)

    # Load and fit all models
    with ProgWrap(f"Loading models...", verbose > 0):
        lst = ["RoomTempFromReduced_RNN",
               "WeatherFromWeatherTime_RNN",
               "WeatherFromWeatherTime_Linear"]
        w_mods = ["WeatherFromWeatherTime_RNN",
                  "WeatherFromWeatherTime_Linear"]
        c_mods = lst[:2]
        # Load models
        eval_mods = load_room_models(lst,
                                     use_bat_data=False,
                                     from_hop=True,
                                     fit=True,
                                     date_str=date_str,
                                     room_nr=room_nr,
                                     hop_eval_set=hop_eval_set,
                                     train_data=train_data,
                                     verbose=next_verb)
        weather_mods = [v for k, v in eval_mods.items() if k in w_mods]
        room_mod = eval_mods["RoomTempFromReduced_RNN"]
        assert isinstance(room_mod, RNNDynamicModel)
        comb_models: List[RNNDynamicModel] = [weather_mods[0], room_mod]
        rl_mods = load_room_models(lst,
                                   use_bat_data=False,
                                   from_hop=True,
                                   fit=True,
                                   date_str=date_str,
                                   room_nr=room_nr,
                                   hop_eval_set=hop_eval_set_rl,
                                   train_data=train_data_rl,
                                   verbose=next_verb)
        rl_comb_models: List[RNNDynamicModel] = [cast_to_subclass(rl_mods[k],
                                                                  RNNDynamicModel) for k in c_mods]

    # Battery model plots
    # with ProgWrap(f"Running battery...", verbose > 0):
    #     with change_dir_name("Battery"):
    #         run_battery(do_rl=True, overwrite=overwrite,
    #                     verbose=prog_verb(verbose), put_on_ol=not debug)

    # Get data and constraints
    date_str = "2020-01-21"
    with ProgWrap(f"Loading data...", verbose > 0):
        ds, rnn_consts = choose_dataset_and_constraints(seq_len=20,
                                                        date_str=date_str,
                                                        add_battery_data=False)

        # ds, rnn_consts = choose_dataset_and_constraints(seq_len=20,
        #                                                 date_str=date_str,
        #                                                 add_battery_data=True)

    # Weather model
    with ProgWrap(f"Analyzing weather model visually...", verbose > 0):
        with change_dir_name("WeatherLinearRNN"):
            train_data = "train_val"
            parts = ["val", "test"]

            # Define model indices and names
            model_names = ["RNN", "PW linear"]
            n_steps = (0, 24)

            # Compare the models for one week continuous and 6h predictions
            dir_to_use = OVERLEAF_IMG_DIR if not debug else TEST_DIR
            ol_file_name = os.path.join(dir_to_use, "WeatherComparison")

            compare_models(weather_mods, ol_file_name,
                           n_steps=n_steps,
                           model_names=model_names,
                           part_spec="test",
                           overwrite=overwrite)

            # Plot prediction performance
            try:
                # for m in mod_list:
                #     m.analyze_performance(metrics=METRICS)
                plot_performance_graph(weather_mods, parts, METRICS, "WeatherPerformance",
                                       short_mod_names=model_names,
                                       series_mask=None, scale_back=True,
                                       remove_units=False, put_on_ol=not debug,
                                       compare_models=True, overwrite=overwrite,
                                       fit_data=train_data,
                                       scale_over_series=True,
                                       fig_size=(18, 9),
                                       set_title=False)
            except OSError as e:
                if verbose:
                    print(f"{e}")
                    print(f"Need to analyze performance of model first!")
                    print(f"Need run again for completion!")
                for m in weather_mods:
                    m.analyze_performance(n_steps=N_PERFORMANCE_STEPS, verbose=prog_verb(verbose),
                                          overwrite=overwrite, metrics=METRICS, parts=parts)

    with change_dir_name("Data"):
        # Cooling water constant
        with ProgWrap(f"Plotting cooling water...", verbose > 0):
            ds_old, rnn_consts = choose_dataset_and_constraints(seq_len=20,
                                                                add_battery_data=False)

            s_name = os.path.join(OVERLEAF_IMG_DIR, "WaterTemp")
            if overwrite or not os.path.isfile(s_name + ".pdf"):
                ds_heat = ds_old[2:4]
                ds_heat.descriptions[0] = "Water temperature (in) [°C]"
                ds_heat.descriptions[1] = "Water temperature (out) [°C]"
                n_tot = ds_heat.data.shape[0]
                ds_heat_rel = ds_heat.slice_time(int(n_tot * 0.6), int(n_tot * 0.66))
                plot_dataset(ds_heat_rel, show=False,
                             title_and_ylab=["Cooling water temperatures", "Temperature [°C]"],
                             save_name=s_name, fig_size=LONG_FIG_SIZE)

        # Constant cooling
        with ProgWrap(f"Plotting cooling period...", verbose > 0):
            room_nr = 43
            ds_old, rnn_consts = choose_dataset_and_constraints(seq_len=20,
                                                                room_nr=room_nr,
                                                                add_battery_data=False)

            s_name = os.path.join(OVERLEAF_IMG_DIR, f"Cooling")
            if overwrite or not os.path.isfile(s_name + ".pdf"):
                ds_heat = ds_old[4]
                ds_heat.descriptions[0] = "Valve state"
                n_tot = ds_heat.data.shape[0]
                ds_heat_rel = ds_heat.slice_time(int(n_tot * 0.55), int(n_tot * 0.75))
                plot_dataset(ds_heat_rel, show=False,
                             title_and_ylab=[f"Cooling room {room_nr}", "Valve state"],
                             save_name=s_name, fig_size=LONG_FIG_SIZE)

    # Room temperature model
    with change_dir_name("RoomTempModel"):
        room_eval_plot_size = (18, 7.5)
        eval_parts = ["val", "test"]
        with ProgWrap(f"Analyzing room temperature model visually...", verbose > 0):
            room_mod = get_model("RoomTempFromReduced_RNN", Dataset.copy(ds), rnn_consts,
                                 train_data=train_data,
                                 date_str=date_str,
                                 hop_eval_set="val",
                                 from_hop=True, fit=True, verbose=False)
            all_data = room_mod.analyze_visually(n_steps=[24], overwrite=overwrite,
                                                 verbose=prog_verb(verbose) > 0, one_file=True,
                                                 save_to_ol=not debug, base_name="Room1W",
                                                 add_errors=False, eval_parts=eval_parts, use_other_plot_function=False)
            room_mod.analyze_performance(n_steps=N_PERFORMANCE_STEPS, verbose=prog_verb(verbose),
                                         overwrite=overwrite, metrics=METRICS, parts=parts)
            plot_performance_graph([room_mod], parts, METRICS, "RTempPerformance",
                                   short_mod_names=["RoomTemp"],
                                   series_mask=None, scale_back=True,
                                   remove_units=False, put_on_ol=not debug,
                                   compare_models=False, overwrite=overwrite,
                                   fit_data=train_data,
                                   scale_over_series=False,
                                   fig_size=room_eval_plot_size,
                                   set_title=False)
            plot_p_g_2([room_mod], parts, METRICS,
                       short_mod_names=["RoomTemp"],
                       series_mask=None, scale_back=True,
                       remove_units=False, put_on_ol=not debug,
                       compare_models=False, overwrite=overwrite,
                       fit_data=train_data,
                       scale_over_series=False,
                       set_title=False,
                       plot_folder=OVERLEAF_IMG_DIR)

    with ProgWrap(f"Creating latex table...", verbose > 0):
        mod_names = ["Room temperature model",
                     "Weather model"]
        make_latex_hop_table(comb_models, mod_names=mod_names,
                             f_name="HopPars_EvalVal",
                             caption="Test", lab=f"val_{room_nr}")
        make_latex_hop_table(rl_comb_models, mod_names=mod_names,
                             f_name="HopPars_EvalTest",
                             caption="Test", lab=f"test_{room_nr}")

    # Combined model evaluation
    with change_dir_name("FullRoomModel"):
        with ProgWrap(f"Analyzing full model performance...", verbose > 0):
            full_mod_name = full_models[0]
            full_mod = get_model(full_mod_name, Dataset.copy(ds), rnn_consts,
                                 train_data=train_data,
                                 date_str=date_str,
                                 hop_eval_set="val",
                                 from_hop=True, fit=True, verbose=False)
            all_data_full = full_mod.analyze_visually(n_steps=[24], overwrite=overwrite,
                                                      verbose=prog_verb(verbose) > 0, one_file=True,
                                                      save_to_ol=not debug, base_name="Room1W",
                                                      add_errors=False, eval_parts=eval_parts,
                                                      series_mask=[4])
            full_mod.analyze_performance(n_steps=N_PERFORMANCE_STEPS, verbose=prog_verb(verbose),
                                         overwrite=overwrite, metrics=METRICS, parts=parts)
            full_mods = [full_mod]
            plot_performance_graph(full_mods, parts, METRICS, "EvalPlot_RTempOnly",
                                   short_mod_names=["TempPredCombModel"],
                                   series_mask=np.array([5]), scale_back=True,
                                   remove_units=False,
                                   fit_data=train_data,
                                   overwrite=overwrite,
                                   put_on_ol=True,
                                   fig_size=room_eval_plot_size,
                                   set_title=False)
            plot_p_g_2(full_mods, parts, METRICS,
                       short_mod_names=["TempPredCombModel"],
                       series_mask=np.array([5]),
                       scale_back=True,
                       remove_units=False,
                       fit_data=train_data,
                       overwrite=overwrite,
                       put_on_ol=True,
                       set_title=False,
                       plot_folder=OVERLEAF_IMG_DIR)

    all_data = [all_data, all_data_full]
    un_dat = all_data[0][-1][0][0].get_unscaled_data()
    gt = un_dat[:, 1]
    pred_temp = un_dat[:, 0]
    pred_full = all_data[1][-1][4][0].get_unscaled_data()[:, 0]

    # DDPG Performance Evaluation
    with ProgWrap(f"Analyzing DDPG performance...", verbose > 0):
        temp_bds = (22.0, 26.0)
        eval_dict = {'filter_good_cases': False,
                     'heating_title_ext': True,
                     'indicate_bad_case': True,
                     'max_visual_evals': 40}

        with change_dir_name("RoomRL_R43_T22_26"):
            # Combined heating and cooling
            n_eval_steps = 10000 if not debug else 30000
            run_room_models(verbose=prog_verb(verbose),
                            n_steps=500000,
                            include_battery=False,
                            perf_eval=True,
                            visual_analysis=True,
                            overwrite=overwrite,
                            put_on_ol=not debug,
                            date_str=date_str,
                            physically_consistent=False,
                            hop_eval_set="test",
                            sample_from="all",
                            train_data="all",
                            temp_bds=temp_bds,
                            n_eval_steps=n_eval_steps,
                            eval_dict=eval_dict,
                            room_nr=43,
                            alpha=50.0)
        eval_dict['max_visual_evals'] = 20

        with change_dir_name("RoomRL_R41_T22_5_Heat"):
            run_room_models(verbose=prog_verb(verbose),
                            n_steps=20000,
                            include_battery=False,
                            perf_eval=True,
                            visual_analysis=True,
                            overwrite=overwrite,
                            put_on_ol=not debug,
                            date_str=date_str,
                            physically_consistent=False,
                            hop_eval_set="test",
                            sample_from="all",
                            train_data="all",
                            temp_bds=(22.5, 22.5),
                            n_eval_steps=n_eval_steps,
                            eval_dict=eval_dict,
                            room_nr=41,
                            alpha=50.0,
                            use_heat_sampler=True,
                            d_fac=0.001,
                            agent_lr=0.0001,
                            )

        with change_dir_name("BatteryRoomRL_R41_T22_5_Heat"):
            run_room_models(verbose=prog_verb(verbose),
                            n_steps=100000,
                            include_battery=True,
                            perf_eval=True,
                            visual_analysis=True,
                            overwrite=overwrite,
                            put_on_ol=not debug,
                            date_str=date_str,
                            physically_consistent=False,
                            hop_eval_set="test",
                            sample_from="all",
                            train_data="all",
                            temp_bds=(22.5, 22.5),
                            n_eval_steps=n_eval_steps,
                            eval_dict=eval_dict,
                            room_nr=41,
                            alpha=50.0,
                            use_heat_sampler=True,
                            d_fac=0.001,
                            agent_lr=0.0001,
                            )

    pass


def curr_tests() -> None:
    """The code that I am currently experimenting with."""

    print("Hi")

    mod_name = "PhysConsModel"

    all_mods = load_room_models([mod_name],
                                use_bat_data=False,
                                from_hop=True,
                                fit=True,
                                date_str="2020-01-21",
                                verbose=5)
    mod = all_mods[mod_name]

    mod.analyze_performance(N_PERFORMANCE_STEPS, verbose=5,
                            overwrite=False,
                            metrics=METRICS)
    return


arg_def_list = [
    # The following arguments can be provided.
    ("analyze_exp", "analyze experiments"),
    ("battery", "run the battery model"),
    ("cleanup", "cleanup all test files, including ones from unit tests"),
    ("data", "update the data from the nest database"),
    ("file_transfer", "transfer data"),
    ("mod_eval", "fit and evaluate the room ML models"),
    ("optimize", "optimize hyperparameters of ML models"),
    ("plot", "run overleaf plot creation"),
    ("room", "run the room simulation model to train and evaluate a rl agent"),
    ("sam_heat", "use heat sampling in RL environment"),
    ("test", "run a few integration tests, not running unit tests"),
    ("ua", "run opcua control"),
    ("verbose", "use verbose mode"),
    ("write_forced", "overwrite existing files"),
]
arg_def_no_short = [
    ("rule_based", "run rule-based controller at NEST"),
]
opt_param_l = [
    # Additional parameters, arbitrary number of them
    ("int", int, "additional integer parameter(s)"),
    ("float", float, "additional floating point parameter(s)"),
    ("str", str, "additional string parameter(s)"),
    ("bool", str2bool, "additional boolean parameter(s)"),
]
common_params = [
    # Parameters used for multiple tasks, format:
    # ("arg_name", type, "help string", default_value),
    ("train_data", str, "Data used for training the models, can be one of "
                        "'train', 'train_val' or 'all'.", "all"),
    ("eval_data", str, "Data used for evaluation of the models, can be one of "
                       "'train', 'val', 'train_val', 'test' or 'all'.",
     DEFAULT_EVAL_SET),
    ("hop_eval_data", str, "Data used for evaluation of the models in "
                           "hyperparameter optimization, can be one either "
                           "'val' or 'test'.", "test"),
    ("data_end_date", str, "String specifying the date when the data was "
                           "loaded from NEST database, e.g. 2020-01-21",
     "2020-01-21"),
    ("rl_sampling", str, "Sampling portion of data when resetting env.", "all"),
    ("room_nr", int, "Integer specifying the room number.", 43),
    ("agent_lr", float, "Learning rate of the DDPG agent.", DEF_RL_LR),
    ("env_noise", float, "Noise level of the RL environment.", DEFAULT_D_FAC),
]


def def_parser() -> argparse.ArgumentParser:
    """The argument parser factory.

    Defines the argument parser based on the lists defined
    above: `arg_def_list`, `opt_param_l` and `common_params`.

    Returns:
        An argument parser.
    """
    # Define argument parser
    parser = argparse.ArgumentParser()

    # Add boolean args
    for kw, h in arg_def_list:
        short_kw = f"-{kw[0]}"
        parser.add_argument(short_kw, f"--{kw}", action="store_true", help=h)
    for kw, h in arg_def_no_short:
        parser.add_argument(f"--{kw}", action="store_true", help=h)

    # Add parameters used for many tasks
    for kw, t, h, d in common_params:
        parser.add_argument(f"--{kw}", type=t, help=h, default=d)

    # Add general optional parameters
    for kw, t, h in opt_param_l:
        short_kw = f"-{kw[0:2]}"
        parser.add_argument(short_kw, f"--{kw}", nargs='+', type=t, help=h)

    return parser


def transfer_data(gd_upload: bool, gd_download: bool, data_to_euler: bool,
                  models_from_euler: bool, verbose: int = 5) -> None:
    """Transfers data to different computers.

    Args:
        gd_upload: Whether to upload files to Google Drive.
        gd_download: Whether to download files from Google Drive.
        data_to_euler:
        models_from_euler:
        verbose:
    """
    next_verb = prog_verb(verbose)

    # Upload to / download from Google Drive
    if gd_upload:
        with ProgWrap("Uploading data to Google Drive", verbose > 0):
            upload_trained_agents(verbose=next_verb)
            upload_hop_pars()
    if gd_download:
        with ProgWrap("Downloading data from Google Drive", verbose > 0):
            download_trained_agents(verbose=next_verb)
            download_hop_pars()

    # Upload to / download from Euler
    auto_script_path = os.path.join(BASE_DIR, "automate.ps1")
    if data_to_euler or models_from_euler:
        assert not EULER, "Cannot be executed on Euler"
        print("Make sure you have an active VPN connection to ETH.")

    if data_to_euler:
        execute_powershell(auto_script_path, "-cp_data")

    if models_from_euler:
        execute_powershell(auto_script_path, "-cp_hop")
        execute_powershell(auto_script_path, "-cp_rl")


def main() -> None:
    """The main function, here all the important, high-level stuff happens.

    Defines command line arguments that can be specified to run certain
    portions of the code. If no such flag is specified, the current
    experiments (defined in the function `curr_tests`) are run, especially
    this is the default in PyCharm.
    """

    # Parse arguments
    parser = def_parser()
    args = parser.parse_args()

    # Extract common bool args
    verbose = 5 if args.verbose else 0
    overwrite = args.write_forced
    sam_heat = args.sam_heat
    agent_lr = args.agent_lr
    env_noise = args.env_noise

    if args.verbose:
        print("Verbosity turned on.")

    # Extract arguments
    train_data, eval_data = args.train_data, args.eval_data
    hop_eval_data, rl_sampling = args.hop_eval_data, args.rl_sampling
    room_nr, date_str = args.room_nr, args.data_end_date

    # Check arguments
    check_date_str(date_str)
    check_train_str(train_data)
    check_dataset_part(eval_data)
    check_eval_data(hop_eval_data)
    room_nr = unique_room_nr(room_nr)
    if room_nr in [41, 51] and date_str == DEFAULT_END_DATE:
        raise ValueError(f"Room number and data end date combination "
                         f"not supported because of backwards compatibility "
                         f"reasons :(")

    # Run integration tests and optionally the cleanup after.
    if args.test:
        run_tests(verbose=verbose)
    if args.cleanup:
        test_cleanup(verbose=verbose)

    # Update stored data
    if args.data:
        update_data(date_str=date_str)

    # Transfer data
    if args.file_transfer:
        gd_upload, gd_download, data_to_euler, models_from_euler = \
            extract_args(args.bool, False, False, False, False)
        transfer_data(gd_upload=gd_upload, gd_download=gd_download,
                      data_to_euler=data_to_euler,
                      models_from_euler=models_from_euler,
                      verbose=verbose)

    # Run hyperparameter optimization
    if args.optimize:
        n_steps = extract_args(args.int, None, raise_too_many_error=False)[0]
        ind_list = []
        if n_steps is not None:
            _, *ind_list = args.int
        use_bat_data, enf_opt = extract_args(args.bool, False, False)
        run_dynamic_model_hyperopt(use_bat_data=use_bat_data,
                                   verbose=verbose,
                                   enforce_optimize=enf_opt,
                                   n_fit_calls=n_steps,
                                   hop_eval_set=hop_eval_data,
                                   date_str=date_str,
                                   room_nr=room_nr,
                                   model_indices=ind_list)

    # Fit and analyze all models
    if args.mod_eval:
        perf_analyze, visual_analyze, include_composite = extract_args(args.bool, True, True, True)
        run_dynamic_model_fit_from_hop(verbose=verbose, perf_analyze=perf_analyze,
                                       visual_analyze=visual_analyze,
                                       include_composite=include_composite,
                                       date_str=date_str, train_data=train_data,
                                       room_nr=room_nr, hop_eval_set=hop_eval_data)

    # Train and analyze the battery model
    if args.battery:
        ext_args = extract_args(args.bool, False, False)
        do_rl, put_on_ol = ext_args
        with change_dir_name("Battery"):
            run_battery(verbose=verbose, do_rl=do_rl, put_on_ol=put_on_ol,
                        overwrite=overwrite, date_str=date_str)

    # Evaluate room model
    if args.room:
        alpha, tb_low, tb_high = extract_args(args.float, 50.0, None, None)
        n_steps, n_eval_steps = extract_args(args.int, None, 10000)
        ext_args = extract_args(args.bool, False, True, False, True)
        add_bat, perf_eval, phys_cons, visual_analysis = ext_args
        temp_bds = None if tb_high is None else (tb_low, tb_high)
        run_room_models(verbose=verbose, alpha=alpha, n_steps=n_steps,
                        include_battery=add_bat, perf_eval=perf_eval,
                        physically_consistent=phys_cons, overwrite=overwrite,
                        date_str=date_str, temp_bds=temp_bds,
                        train_data=train_data, room_nr=room_nr,
                        hop_eval_set=hop_eval_data,
                        sample_from=rl_sampling,
                        visual_analysis=visual_analysis,
                        n_eval_steps=n_eval_steps,
                        agent_lr=agent_lr,
                        use_heat_sampler=sam_heat,
                        d_fac=env_noise)

    # Overleaf plots
    if args.plot:
        debug = extract_args(args.bool, False)[0]
        update_overleaf_plots(verbose, overwrite=overwrite, debug=debug)

    # Run rule-based control
    if args.rule_based:
        notify_debug = extract_args(args.bool, True)[0]
        name_ext = extract_args(args.str, "")[0]
        min_temp = extract_args(args.float, 21.0)[0]
        run_rule_based_control(room_nr=room_nr, notify_debug=notify_debug,
                               verbose=verbose, name_ext=name_ext,
                               min_temp=min_temp)

    # Opcua
    if args.ua:
        debug, notify_failure, phys_cons, notify_debug, dummy_use = \
            extract_args(args.bool, False, False, False, None, True)
        n_steps = extract_args(args.int, None)[0]
        alpha, tb_low, tb_high = extract_args(args.float, 50.0, None, None)
        temp_bds = None if tb_high is None else (tb_low, tb_high)
        run_rl_control(room_nr=room_nr, notify_failure=notify_failure,
                       debug=debug, alpha=alpha, n_steps=n_steps,
                       date_str=date_str, temp_bds=temp_bds,
                       train_data=train_data,
                       hop_eval_set=hop_eval_data,
                       notify_debug=notify_debug,
                       dummy_use=dummy_use,
                       sample_from=rl_sampling,
                       agent_lr=agent_lr,
                       use_heat_sampler=sam_heat,
                       d_fac=env_noise)

    # Analyze experiments
    if args.analyze_exp:
        analyze_experiments(room_nr=room_nr, verbose=verbose)

    # Check if any flag is set, if not, do current experiments.
    var_dict = vars(args)
    excluded = ["verbose"]
    var_dict = {k: val for k, val in var_dict.items() if k not in excluded}
    any_flag_set = reduce(lambda x, k: x or var_dict[k] is True, var_dict, 0)
    if not any_flag_set:
        print("No flags set")
        curr_tests()


# Execute main function
if __name__ == '__main__':
    main()
