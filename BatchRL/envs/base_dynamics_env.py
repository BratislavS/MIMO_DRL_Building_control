"""Dynamics model environment base class.

Use this class if you want to build an environment
based on a model of class `BaseDynamicsModel`.
"""
import os
from abc import ABC, abstractmethod
from typing import Dict, Union, List, Tuple, Optional, Callable, Any

import gym
import numpy as np

from agents import base_agent
from agents.base_agent import RL_MODEL_DIR
from dynamics.base_model import BaseDynamicsModel
from util.numerics import npf32
from util.util import Arr, create_dir, make_param_ext, str_to_np_dt, Num, day_offset_ts, ts_per_day, ProgWrap, \
    load_pickle, save_pickle
from util.visualize import rl_plot_path, plot_env_evaluation, plot_reward_details, OVERLEAF_IMG_DIR, MergeListT, \
    eval_env_evaluation

Agents = Union[List[base_agent.AgentBase], base_agent.AgentBase]

DEFAULT_ENV_SAMPLE_DATA: str = "train"

RejSamplerFun = Callable[[np.ndarray], bool]


class RejSampler:
    """Rejection sampling class.

    Can be used for unbalanced environment resets by
    rejecting unwanted states. Will impact performance if
    a lot are rejected.
    """

    name: str = "Rej"
    max_rej: int = 1

    def __init__(self, name: str, fun: RejSamplerFun = lambda x: True,
                 max_rej: int = 1):
        self.name = f"{name}{max_rej}"
        self.fun = fun
        self.max_rej = max_rej

    def __call__(self, hist: np.ndarray):
        return self.fun(hist)


class DynEnv(ABC, gym.Env):
    """The environment wrapper class for `BaseDynamicsModel`.

    Takes an instance of `BaseDynamicsModel` and adds
    all the functionality needed to turn it into an environment
    to be used for reinforcement learning.
    """
    # Fix data
    m: BaseDynamicsModel  #: Prediction model.
    act_dim: int  #: The dimension of the action space.
    state_dim: int  #: The dimension of the state space.
    n_ts_per_eps: int  #: The maximum number of timesteps per episode.
    n_ts_per_day: int  #: Number of time steps in a day.
    t_init_n: int  #: The index of the initial timestep in the dataset.

    short_name: str = None

    # State data, might change if `step` is called.
    n_ts: int = 0  #: The current number of timesteps.
    hist: np.ndarray  #: 2D array with current state.

    # The current data to sample initial conditions from.
    train_data: np.ndarray  #: The training data.
    train_indices: np.ndarray  #: The indices corresponding to `train_data`.
    n_start_data: int  #: The number of possible initializations using the training data.

    # Info about the current episode.
    use_noise: bool = False  #: Whether to add noise when simulating.
    curr_ind: int = None
    curr_n: int = None

    orig_actions: np.ndarray  #: Array with fallback actions

    # The one day data
    n_train_days: int  #: Number of training days
    train_days: List  #: The data of all days, where all data is available.
    day_inds: np.ndarray  #: Index vector storing the timestep offsets to the days
    day_ind: int = 0  #: The index of the day in `train_days`.

    # Scaling info
    info: Dict = None  #: A dict with info about the current agent.
    do_scaling: bool = False
    a_scaling_pars: Tuple[np.ndarray, np.ndarray] = None  #: Extra scaling parameters.

    # Plotting default values
    default_state_mask: np.ndarray = None
    default_series_merging: MergeListT = None

    _dummy_use: bool
    _plot_path: str
    _disturbance_is_modelled: bool = False

    def __init__(self, m: BaseDynamicsModel, name: str = None, max_eps: int = None,
                 disturb_fac: float = 1.0,
                 init_res: bool = True,
                 dummy_use: bool = False,
                 sample_from: str = DEFAULT_ENV_SAMPLE_DATA,
                 rejection_sampler: RejSampler = None,
                 verbose: int = 0):
        """Initialize the environment.

        Args:
            m: Full model predicting all the non-control features.
            max_eps: Number of continuous predictions in an episode.
        """
        self._dummy_use = dummy_use
        self.verbose = verbose
        if not self._dummy_use:
            # m.model_disturbance()
            pass
        self.m = m
        self.rej_sampler = rejection_sampler

        # Define name extensions
        sam_ext = f"_SAM_{sample_from}" if sample_from != DEFAULT_ENV_SAMPLE_DATA else ""
        h_sam_ext = f"_RejS_{rejection_sampler.name}" if rejection_sampler is not None else ""

        if name is not None:
            dist_ex = make_param_ext([("DF", disturb_fac)])
            self.name = f"{name}{dist_ex}{sam_ext}{h_sam_ext}_DATA_{m.name}"
        else:
            self.name = f"RLEnv_{m.name}"
        # self.plot_path = os.path.join(rl_plot_path, self.name)
        self._plot_path = os.path.join(rl_plot_path, self.name)

        # Set attributes.
        dat = m.data
        self.disturb_fac = disturb_fac
        self.act_dim = dat.n_c
        self.state_dim = dat.d

        # Time indices
        self.n_ts_per_eps = 100 if max_eps is None else max_eps
        self.t_init_n = day_offset_ts(dat.t_init, dat.dt,
                                      remaining=False)
        self.n_ts_per_day = ts_per_day(dat.dt)

        # Set data and initialize env.
        self._set_data(sample_from)
        if init_res:
            self.reset(use_noise=False)

    def __str__(self):
        """Generic string conversion."""
        n = self.short_name if self.short_name is not None else self.name
        return f"RL env of class {self.__class__.__name__} with name {n}"

    @property
    def plot_path(self) -> str:
        """Returns the path of the plot folder."""
        if self.short_name is not None:
            return os.path.join(rl_plot_path, self.short_name)
        return self._plot_path

    @property
    def model_path(self) -> str:
        """Returns the path of the model folder."""
        if self.short_name is not None:
            return os.path.join(RL_MODEL_DIR, self.short_name)
        else:
            raise ValueError("No short name specified!!!")

    def set_agent(self, a: base_agent.AgentBase) -> None:
        """Sets the given agent to the environment.

        Handles the extra scaling if it is required by the
        agent, the environment does not actually get access to the
        agent.

        Args:
            a: The agent to be used with the env.
        """
        # Get info about agent and check whether scaling is needed
        self.info = a.get_info()
        scaling = self.info.get('action_scaled_01')
        self.do_scaling = scaling is not None

        # Handle scaling
        if self.do_scaling:
            p1 = np.array([i[0] for i in scaling], dtype=np.float32)
            p2 = np.array([i[1] - i[0] for i in scaling], dtype=np.float32)
            self.a_scaling_pars = p1, p2

    def _set_data(self, part_str: str = "train") -> None:
        """Sets the data to use in the env.

        Args:
            part_str: The string specifying the part of the data.
        """
        self.m.data.check_part(part_str)
        self.train_data, _, self.train_indices = self.m.data.get_split(part_str)
        self.n_start_data = len(self.train_data)

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

    def get_model_path(self, mod_name: str) -> str:
        """Returns the model path."""
        dir_name = self.model_path
        create_dir(dir_name)
        return os.path.join(dir_name, mod_name)

    reward_descs: List = []  #: The description of the detailed reward.

    @abstractmethod
    def detailed_reward(self, curr_pred: np.ndarray, action: Arr) -> np.ndarray:
        """Computes the different components of the reward and returns them all.

        Args:
            curr_pred: The current predictions.
            action: The most recent action taken.

        Returns:
            The reward components in an 1d array.
        """
        pass

    def compute_reward(self, curr_pred: np.ndarray, action: Arr) -> float:
        """Computes the reward to be maximized during the RL training.

        Args:
            curr_pred: The current predictions.
            action: The most recent action taken.

        Returns:
            Value of the reward of that outcome.
        """
        # The base implementation just sums up the different rewards.
        return np.sum(self.detailed_reward(curr_pred, action)).item()

    @abstractmethod
    def episode_over(self, curr_pred: np.ndarray) -> bool:
        """Defines criterion for episode to be over.

        Args:
            curr_pred: The next predicted state.

        Returns:
            True if episode is over, else false
        """
        return False

    def scale_action_for_step(self, action: Arr) -> Arr:
        """Scales the action to be used by the underlying ML model.

        This does nothing, override if needed.

        Args:
            action: The action to be scaled.
        Returns:
            The scaled action, same shape as the input.
        """
        return action

    @property
    def dummy_use(self):
        return self._dummy_use

    def step(self, action: Arr) -> Tuple[np.ndarray, float, bool, Dict]:
        """Evolve the model with the given control input `action`.

        This function should not be overridden, instead override
        `scale_action_for_step`! If it is, it should be called at some
        point in the overriding method.

        Args:
            action: The control input (action), will be scaled for the underlying model
                of the environment.

        Returns:
            The next state, the reward of having chosen that action and a bool
            determining if the episode is over. (And an empty dict)
        """
        if self._dummy_use:
            raise ValueError("Stepping not supported for dummy usage!")

        # Scale the action
        action = self.scale_action_for_step(action)

        # Add the control to the history
        self.hist[-1, -self.act_dim:] = action

        # Predict using the model
        hist_res = np.copy(self.hist).reshape((1, -1, self.state_dim))
        curr_pred = self.m.predict(hist_res)[0]

        # Add noise
        if self.use_noise:
            assert self._disturbance_is_modelled, "Need to model disturbance first!"
            curr_pred += self.disturb_fac * self.m.disturb()

        # Save the chosen action
        self.orig_actions[self.n_ts, :] = action

        # Update history
        self.hist[:-1, :] = self.hist[1:, :]
        self.hist[-1, :-self.act_dim] = curr_pred
        self.n_ts += 1

        # Compute reward and episode termination
        r = np.array(self.compute_reward(curr_pred, action)).item()
        ep_over = self.n_ts == self.n_ts_per_eps or self.episode_over(curr_pred)
        return curr_pred, r, ep_over, {}

    @property
    def curr_state(self) -> np.ndarray:
        """Returns the current state.

        Returns:
            Current state of environment, 1d array.
        """
        return self.hist[-1, :-self.act_dim]

    def scale_state(self, state: np.ndarray,
                    remove_mean: bool = False) -> np.ndarray:
        """Scales the state using the Dataset."""
        return self.m.data.scale(state, remove_mean=remove_mean, state_only=True)

    def get_curr_day_n(self) -> int:
        """Returns the current timestep index."""
        return (self.curr_n + self.n_ts) % self.n_ts_per_day

    def _model_disturbance(self):
        """Models the disturbance of the underlying model."""
        with ProgWrap("Modeling disturbance...", self.verbose > 0):
            self.m.model_disturbance()
            self._disturbance_is_modelled = True

    def reset(self, start_ind: int = None, use_noise: bool = True) -> np.ndarray:
        """Resets the environment.

        Always needs to be called if the episode is over.
        Initializes the environment with a new initial state.
        Uses rejection sampling to favor some initial conditions over
        others, as defined by `self.rej_sampler`.

        Args:
            start_ind: The index specifying the initial condition.
            use_noise: Whether noise should be added when simulating.

        Returns:
            A new initial state.
        """
        # Reset time step and disturbance
        self.use_noise = use_noise
        self.n_ts = 0
        self.m.reset_disturbance()

        # Model disturbance if not yet done
        if use_noise and not self._disturbance_is_modelled:
            if not self._dummy_use:
                self._model_disturbance()

        # Reset original actions (necessary?)
        self.orig_actions = np.empty((self.n_ts_per_eps, self.act_dim), dtype=np.float32)

        accepted, ret_val = False, None
        ct = 0
        while not accepted:
            # Select new start data
            if start_ind is None or self.rej_sampler is not None:
                start_ind = np.random.randint(self.n_start_data)
            else:
                if self.n_start_data <= start_ind:
                    raise ValueError("start_ind is too fucking large!")

            self.hist = np.copy(self.train_data[start_ind])
            self.curr_ind = self.train_indices[start_ind] + self.m.data.seq_len - 1
            self.curr_n = (self.t_init_n + self.curr_ind) % self.n_ts_per_day
            ret_val = np.copy(self.hist[-1, :-self.act_dim])

            if self.rej_sampler is not None:
                accepted = self.rej_sampler(ret_val)
                if ct >= self.rej_sampler.max_rej:
                    accepted = True
            else:
                accepted = True

            if not accepted:
                ct += 1

        # print(f"Chose start ind: {start_ind}, rejected: {ct}")
        return ret_val

    def get_scaled_init_state(self, init_ind: int, heat_inds) -> np.ndarray:
        """Returns the initial states when reset with `init_ind`.

        Args:
            init_ind: The reset index.
            heat_inds: The indices of the series to return.

        Returns:
            The scaled initial values of the series.
        """
        # Find the current heating water temperatures
        self.reset(init_ind, use_noise=False)
        return self.scale_state(self.curr_state, remove_mean=False)[heat_inds]

    def render(self, mode='human'):
        """Render method for compatibility with OpenAI gym."""
        print("Rendering not implemented!")

    def _to_scaled(self, action: Arr, to_original: bool = False,
                   extra_scaling: bool = False) -> np.ndarray:
        """Converts actions to the right range.

        Needs to be overridden by subclasses.
        """
        raise NotImplementedError("Implement this shit!")

    def analyze_agents_visually(self, agents: Agents,
                                fitted: bool = True,
                                use_noise: bool = False,
                                start_ind: int = None,
                                max_steps: int = None,
                                state_mask: np.ndarray = None,
                                plot_constrain_actions: bool = True,
                                show_rewards: bool = True,
                                title_ext: str = "",
                                put_on_ol: bool = False,
                                plot_rewards: bool = False,
                                bounds: List[Tuple[int, Tuple[Num, Num]]] = None,
                                series_merging_list: MergeListT = None,
                                overwrite: bool = False,
                                plot_all_rewards: bool = True) -> None:
        """Analyzes and compares a set of agents / control strategies.

        Uses the same initial condition of the environment and evaluates
        all the given agents in `agents` starting from these initial conditions.
        Then makes a plot comparing how the environment behaves under all the
        agents.

        Args:
            agents: A list of agents or a single agent.
            fitted: Whether the agents are already fitted.
            use_noise: Whether to use noise in the predictions.
            start_ind: Index of initial configuration, random if None.
            max_steps: The maximum number of steps of an episode.
            state_mask: Mask defining which series to plot.
            plot_constrain_actions: Whether to plot the actions constrained by the env.
            show_rewards: Use default!
            title_ext: The title to put in the plot.
            put_on_ol: Whether to save in Overleaf plot dir.
            plot_rewards: Whether to plot the reward bar plot for this scenario.
            bounds: Optional state bounds to put in plot.
            series_merging_list: Defines series that will be merged. Should be
                independent of agent's actions.
            overwrite: Whether to overwrite existing plot files.
            plot_all_rewards:

        Raises:
            ValueError: If `start_ind` is too large or if an agent is not suited
                for this environment.
        """
        print("This is deprecated!")

        # Make function compatible for single agent input
        if not isinstance(agents, list):
            agents = [agents]

        if max_steps is None:
            max_steps = 100000

        if not show_rewards:
            raise NotImplementedError("Do not do this!")

        # Use (possibly overridden) defaults
        if state_mask is None:
            state_mask = self.default_state_mask
        if series_merging_list is None:
            series_merging_list = self.default_series_merging

        # Choose same random start for all agents
        if start_ind is None:
            start_ind = np.random.randint(self.n_start_data)
        elif start_ind >= self.n_start_data:
            raise ValueError(f"start_ind: {start_ind} cannot be larger than: {self.n_start_data}!")

        # Check if file already exists
        analysis_plot_path = self._construct_plot_name("AgentAnalysis", start_ind, agents, put_on_ol)
        if os.path.isfile(analysis_plot_path + ".pdf") and not overwrite:
            return

        # Define arrays to save trajectories
        n_non_c_states = self.state_dim - self.act_dim
        n_agents = len(agents)
        action_sequences = npf32((n_agents, self.n_ts_per_eps, self.act_dim), fill=np.nan)
        clipped_action_sequences = npf32((n_agents, self.n_ts_per_eps, self.act_dim), fill=np.nan)
        trajectories = npf32((n_agents, self.n_ts_per_eps, n_non_c_states), fill=np.nan)
        rewards = npf32((n_agents, self.n_ts_per_eps), fill=np.nan)
        n_tot_rewards = len(self.reward_descs) + 1
        all_rewards = npf32((n_agents, self.n_ts_per_eps, n_tot_rewards), fill=np.nan)

        for a_id, a in enumerate(agents):
            # Check that agent references this environment
            if not a.env == self:
                raise ValueError(f"Agent {a_id} was not assigned to this env!")

            # Set agent, required for the ones that need special scaling
            self.set_agent(a)

            # Fit agent if not already fitted
            if not fitted:
                a.fit()

            # Reset env
            curr_state = self.reset(start_ind=start_ind, use_noise=use_noise)
            episode_over = False
            count = 0

            # Evaluate agent and save states, actions and reward
            while not episode_over and count < max_steps:
                curr_action = a.get_action(curr_state)
                scaled_a = self.scale_action_for_step(curr_action)

                curr_state, rew, episode_over, extra = self.step(curr_action)
                det_rew = self.detailed_reward(curr_state, scaled_a)
                action_sequences[a_id, count, :] = curr_action

                # Save trajectories and reward
                trajectories[a_id, count, :] = np.copy(curr_state)
                rewards[a_id, count] = rew
                all_rewards[a_id, count, 0] = rew
                all_rewards[a_id, count, 1:] = det_rew
                count += 1

            # Get original actions
            clipped_action_sequences[a_id, :self.n_ts, :] = self.orig_actions[:self.n_ts, :]

            # Scale actions
            if self.do_scaling:
                action_sequences[a_id] = self.a_scaling_pars[0] + action_sequences[a_id] * self.a_scaling_pars[1]

        # Scale the data to the right values
        trajectories = self.m.rescale_output(trajectories, out_put=True)
        s_ac = clipped_action_sequences.shape
        for k in range(s_ac[0]):
            for i in range(s_ac[1]):
                clipped_action_sequences[k, i] = self._to_scaled(clipped_action_sequences[k, i],
                                                                 to_original=True)
        if not plot_constrain_actions or np.allclose(clipped_action_sequences, action_sequences):
            clipped_action_sequences = None

        # Time stuff
        shifted_t_init = self.m.data.get_shifted_t_init(self.curr_ind)
        np_st_init = str_to_np_dt(shifted_t_init)

        # Plot all the things
        name_list = [a.get_short_name() for a in agents]
        add_pth = self._construct_plot_name("AgentAnalysisReward", start_ind, agents, put_on_ol)
        add_pth = add_pth if plot_rewards else None
        rewards_for_eval = all_rewards if plot_all_rewards else rewards
        plot_env_evaluation(action_sequences, trajectories, rewards_for_eval, self.m.data,
                            name_list, analysis_plot_path, clipped_action_sequences,
                            state_mask, show_rewards=show_rewards, title_ext=title_ext,
                            np_dt_init=np_st_init, rew_save_path=add_pth, bounds=bounds,
                            series_merging_list=series_merging_list,
                            reward_descs=self.reward_descs)

        # Plot the rewards for this episode
        add_pth = self._construct_plot_name("AgentAnalysisRewardDetailed", start_ind, agents, put_on_ol)
        names = [a.get_short_name() for a in agents]
        plot_reward_details(names, all_rewards, add_pth,
                            self.reward_descs, dt=self.m.data.dt, n_eval_steps=self.n_ts_per_eps,
                            title_ext=title_ext)

    def _construct_plot_name(self, base_name: str, start_ind: Optional[int], agent_list: List,
                             put_on_ol: bool = False):
        name_list = [a.get_short_name() for a in agent_list]
        agent_names = '_'.join(name_list)
        s_ext = "" if start_ind is None else f"_{start_ind}"
        base = f"{base_name}{s_ext}_{agent_names}"
        orig_name = self.get_plt_path(base)
        if put_on_ol:
            print(f"Original: {orig_name}")
            return os.path.join(OVERLEAF_IMG_DIR, base)
        return orig_name

    def _get_detail_eval_title_ext(self):
        # For the water temperature info in the title, see overriding method.
        # This is so fucking ugly!
        return None

    def detailed_eval_agents(self, agent_list: Agents,
                             n_steps: int = 100,
                             use_noise: bool = False,
                             put_on_ol: bool = False,
                             overwrite: bool = False,
                             verbose: int = 0,
                             plt_fun: Callable = plot_reward_details,
                             episode_marker: Callable = None,
                             visual_eval: bool = False,
                             bounds: List[Tuple[int, Tuple[Num, Num]]] = None,
                             agent_filter_ind: int = None,
                             filter_good_cases: bool = True,
                             plot_tot_reward_cases: bool = True,
                             plot_tot_eval: bool = True,
                             plot_constrained_actions: bool = False,
                             max_visual_evals: int = 4,
                             heating_title_ext: bool = False,
                             indicate_bad_case: bool = False,
                             disconnect_data: Any = None,
                             eval_quality: bool = False
                             ) -> Optional[np.ndarray]:
        """Evaluates the given agents for this environment.

        Let's the agents act in the environment and observes the
        rewards. These are then averaged, plotted and returned.
        If `overwrite` is False and the plot already exists, nothing
        is returned.

        Args:
            agent_list: List of agents that are based on this env.
            n_steps: Number of evaluation steps.
            use_noise: Whether to use noise in the evaluation.
            put_on_ol: Whether to put the resulting plot into the Overleaf plot folder.
            overwrite: Whether to overwrite pre-existing plot files.
            verbose: Verbosity.
            plt_fun: The plotting function.
            episode_marker: Marker for the case if `plt_fun`
                is :func:`util.visualize.plot_heat_cool_rew_det`
            visual_eval: Whether to do visual analysis with the evaluated data.
            bounds: The temperature bounds for the visual analysis plot.
                Ignored if `visual_eval` is False.
            agent_filter_ind: Index of agent whose rewards need to be
                among the best to be plotted. Ignored if `visual_eval` is False.
            filter_good_cases: Whether to filter out the bad cases and only plot
                the good ones in the time series analysis.
            plot_tot_reward_cases: Whether to plot the rewards corresponding to
                 the time series cases.
            plot_tot_eval: Whether to plot the total rewards of the whole evaluation.
            plot_constrained_actions: Whether to plot the constrained actions in the time series
                plot.
            max_visual_evals: Max. number of visual evaluations.
            heating_title_ext:
            indicate_bad_case:
            disconnect_data:
            eval_quality:

        Returns:
            The rewards seen by all the agents, or None if `overwrite` is False
            and the plot already exists.
        """

        if agent_filter_ind is None:
            agent_filter_ind = 0

        # Make function compatible for single agent input.
        if not isinstance(agent_list, list):
            agent_list = [agent_list]

        # Handle overwriting
        p_name = self._construct_plot_name("DetailAnalysis", n_steps, agent_list, put_on_ol)
        if os.path.isfile(p_name + ".pdf") and not overwrite:
            if verbose > 0:
                print("Agent evaluation plot already exists!")
            return
        p_1 = self._construct_plot_name(f"AgentAnalysis_E0", None,
                                        agent_list, put_on_ol)
        p_2 = self._construct_plot_name(f"AgentAnalysis_E0_n", None,
                                        agent_list, put_on_ol)
        if (os.path.isfile(p_1 + ".pdf") or os.path.isfile(p_2 + ".pdf")) and not overwrite:
            if visual_eval and not plot_tot_eval:
                print("Agent time series plot already exists!")
                return

        pickle_data_path = self._construct_plot_name(f"pickled_data_{n_steps}", None,
                                                     agent_list, False) + ".pkl"

        # Init scores.
        n_agents = len(agent_list)
        n_extra_rewards = len(self.reward_descs)
        n_tot_rewards = n_extra_rewards + 1

        # Load data from pickle if exists
        if os.path.isfile(pickle_data_path):
            if verbose > 0:
                print("Loading from previous evaluation...")
            all_rewards, all_rewards, all_states, all_marks, \
                all_actions, all_scaled_actions, ret_inds = load_pickle(pickle_data_path)
        else:
            all_rewards = npf32((n_agents, n_steps, n_tot_rewards))
            all_actions = npf32((n_agents, n_steps, self.act_dim))
            all_scaled_actions = npf32((n_agents, n_steps, self.act_dim))
            all_states = npf32((n_agents, n_steps, self.state_dim - self.act_dim))
            all_marks = np.empty((n_agents, n_steps), dtype=np.int32)
            ret_inds = None

            do_scaling = self.m.data.fully_scaled

            if verbose > 0:
                print("Evaluating agents...")
            for a_id, a in enumerate(agent_list):

                # Set agent
                self.set_agent(a)

                # Fit agent if not already fitted
                if verbose:
                    print(f"Fitting agent: {a}")
                a.fit(verbose=verbose, train_data=a.fit_data)

                # Check that agent references this environment
                if not a.env == self:
                    raise ValueError(f"Agent {a_id} was not assigned to this env!")

                # Evaluate agent.
                if verbose:
                    print(f"Evaluating agent: {a}")
                eval_res = a.eval(n_steps, reset_seed=True, detailed=True,
                                  use_noise=use_noise, scale_states=do_scaling,
                                  episode_marker=episode_marker,
                                  return_inds=True,
                                  verbose=verbose)
                rew, ex_rew, states, ep_marks, acs, s_acs, ret_inds = eval_res

                all_rewards[a_id, :, 0] = rew
                all_rewards[a_id, :, 1:] = ex_rew
                all_states[a_id] = states
                all_marks[a_id] = ep_marks
                all_actions[a_id] = acs
                all_scaled_actions[a_id] = s_acs

                if verbose:
                    print(f"Saving evaluation data...")
                all_data = (all_rewards, all_rewards, all_states, all_marks,
                            all_actions, all_scaled_actions, ret_inds)
                save_pickle(pickle_data_path, all_data)

        # Plot total rewards for all steps
        title_ext = ""  # self._get_detail_eval_title_ext()
        if plot_tot_eval:
            if verbose > 0:
                print("Plotting evaluation...")
            names = [a.get_plot_name() for a in agent_list]
            plt_fun(names, all_rewards, p_name,
                    self.reward_descs, dt=self.m.data.dt, n_eval_steps=n_steps,
                    title_ext=title_ext, all_states=all_states,
                    verbose=0, ep_marks=all_marks,
                    add_base_title=False)

        if eval_quality:
            quality_plot_path = self._construct_plot_name(f"Env_Quality_{n_steps}", None,
                                                          agent_list, put_on_ol)
            eval_env_evaluation(all_rewards, all_states,
                                all_marks, self.n_ts_per_eps,
                                plt_base_name=quality_plot_path)

        print(f"Mean rewards: {np.mean(all_rewards[:, :, 0], axis=1)}")

        # Plot time series
        if visual_eval:
            ct = 0
            n_vis_eval = n_steps // self.n_ts_per_eps
            for k in range(n_vis_eval):
                k0 = k * self.n_ts_per_eps
                k1 = (k + 1) * self.n_ts_per_eps

                # Extract stuff
                action_sequences = all_actions[:, k0:k1]
                clipped_action_sequences = all_scaled_actions[:, k0:k1]
                curr_states = all_states[:, k0:k1]
                curr_rew = all_rewards[:, k0:k1]
                ep_ind = ret_inds[k]

                mean_rew = np.mean(curr_rew[:, :, 0], axis=1)
                winner = np.argwhere(mean_rew == np.amax(mean_rew))

                ext_ = "" if [agent_filter_ind] not in winner else "_n"
                if filter_good_cases:
                    if [agent_filter_ind] not in winner or ct > max_visual_evals:
                        # if [agent_filter_ind] in winner:
                        #     print(f"Len: {len(winner)}")
                        # else:
                        #     print(f"Bad case!")
                        continue
                    else:
                        # print(f"Found time series!!")
                        ct += 1
                else:
                    ct += 1
                if ct > max_visual_evals:
                    continue

                # Time stuff
                shifted_t_init = self.m.data.get_shifted_t_init(ep_ind)
                np_st_init = str_to_np_dt(shifted_t_init)

                state_mask = self.default_state_mask
                name_list = [a.get_plot_name() for a in agent_list]
                add_pth = None
                series_merging_list = self.default_series_merging

                if heating_title_ext:
                    unscaled_state = curr_states[0, 0]
                    w_in = unscaled_state[2]
                    heating = w_in > unscaled_state[4]
                    ext = f"{'Heating' if heating else 'Cooling'}, inflow water temp.: {w_in:.1f} °C"
                    curr_title_ext = f"{title_ext} {ext}"
                    ext_ += "_h" if heating else "_c"
                    if indicate_bad_case:
                        assert n_agents == 4, "Fuck you!"
                        r_temp_closed = curr_states[1, -1, 4]
                        r_temp_open = curr_states[0, -1, 4]
                        h_case = r_temp_open > r_temp_closed
                        if h_case != heating:
                            print(f"Fuck! k = {k}, {heating}")
                            ext_ += "_b"
                        else:
                            print(f"Nice! k = {k}, {heating}")
                else:
                    curr_title_ext = title_ext

                analysis_plot_path = self._construct_plot_name(f"AgentAnalysis_E{k}{ext_}", None,
                                                               agent_list, put_on_ol)

                # Plot time series for this episode
                ex_actions = clipped_action_sequences if plot_constrained_actions else None
                plot_env_evaluation(action_sequences, curr_states, curr_rew, self.m.data,
                                    name_list, analysis_plot_path, extra_actions=ex_actions,
                                    series_mask=state_mask, show_rewards=True, title_ext=curr_title_ext,
                                    np_dt_init=np_st_init, rew_save_path=add_pth,
                                    bounds=bounds,
                                    series_merging_list=series_merging_list,
                                    reward_descs=self.reward_descs,
                                    disconnect_data=disconnect_data,
                                    tot_reward_only=True)

                # Plot the rewards for this episode
                if not plot_tot_reward_cases:
                    continue
                add_pth = self._construct_plot_name(f"AgentAnalysis_E{k}_RewardDetailed", None,
                                                    agent_list, put_on_ol)
                plot_reward_details(name_list, curr_rew, add_pth,
                                    self.reward_descs, dt=self.m.data.dt,
                                    n_eval_steps=self.n_ts_per_eps,
                                    title_ext=title_ext,
                                    add_base_title=False)

        return all_rewards
