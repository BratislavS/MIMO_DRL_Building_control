"""Environments based on dynamics models.

Defines the three main RL environments, the
Battery env :class:`BatteryEnv`, the room env :class:`FullRoomEnv`
and the joint env :class:`RoomBatteryEnv`. Additionally, a common
base class (:class:`RLDynEnv`) is defined to avoid too much duplicate code.
"""
import warnings
from abc import ABC, abstractmethod
from typing import Sequence, Tuple, List, Dict

import numpy as np

from dynamics.base_model import BaseDynamicsModel
from dynamics.battery_model import BatteryModel
from dynamics.composite import CompositeModel
from envs.base_dynamics_env import DynEnv, RejSampler
from util.numerics import trf_mean_and_std, rem_mean_and_std, npf32, check_shape
from util.util import make_param_ext, Arr, linear_oob_penalty, LOrEl, Num, to_list, yeet, linear_oob

USE_LEFTOVER_REWARD = False

RangeT = Tuple[Num, Num]  #: Range for single state / action series.
InRangeT = LOrEl[RangeT]  #: The type of action ranges.
RangeListT = List[RangeT]

# General parameters for the environments

# For the room environment:
TEMP_BOUNDS: RangeT = (22.0, 26.0)  #: The requested temperature range.
HEAT_ACTION_BOUNDS: RangeT = (0.0, 1.0)  #: The action range for the valve opening time.
ROOM_ENG_FAC: float = 65.37 * 4.18 / 3.6
BAT_SCALING: float = 1000.0 / ROOM_ENG_FAC / 200

# For the battery environment:
BATTERY_ACTION_BOUNDS: RangeT = (-100.0, 100.0)  #: The action range for the active power.
SOC_BOUND: RangeT = (20.0, 80.0)  #: The desired state-of-charge range.
SOC_GOAL: Num = 60.0  #: Desired SoC at end of episode.

# Reward parts descriptions
BAT_ENERGY: str = f"Battery energy consumption [{ROOM_ENG_FAC:.4g} Wh]"
BAT_ENERGY_HIGH: str = f"Battery energy consumption [kWh]"
ROOM_ENERGY: str = f"Room energy consumption [{ROOM_ENG_FAC:.4g} Wh]"
TEMP_BOUND_PEN: str = "Comfort violation [Kh]"
ENG_COST: str = "Energy costs"


def heat_marker(state: np.ndarray) -> bool:
    """Determines whether it is a heating case or not."""
    return state[2] > state[4]


HeatSampler = RejSampler("HeatSam", heat_marker, 1000)


class RLDynEnv(DynEnv, ABC):
    """The base class for RL environments based on `BaseDynamicsModel`.

    Only working for one control input.
    """
    action_range: RangeListT  #: The range of the actions.
    action_range_scaled: np.ndarray  #: The range scaled to the whitened actions.
    scaling: np.ndarray = None  #: Whether the underlying `Dataset` was scaled.
    nb_actions: int  #: Number of actions if discrete else action space dim.

    def __init__(self, m: BaseDynamicsModel,
                 max_eps: int,
                 action_range: Sequence = (0.0, 1.0),
                 cont_actions: bool = True,
                 n_disc_actions: int = 11,
                 n_cont_actions: int = 1,
                 **kwargs):
        """Constructor."""
        super().__init__(m, max_eps=max_eps, **kwargs)

        if cont_actions and n_cont_actions is None:
            raise ValueError("Need to specify action space dimensionality!")
        if not cont_actions:
            raise NotImplementedError("Discrete actions are deprecated!")

        # Set parameters
        self.nb_actions = n_disc_actions if not cont_actions else n_cont_actions
        self.cont_actions = cont_actions

        # Save action range, original and scaled.
        self.action_range = to_list(action_range)
        assert len(self.action_range) == n_cont_actions, "False amount of action ranges!"
        action_range_scaled = npf32((n_cont_actions, 2))

        # Initialize fallback actions
        self.fb_actions = np.empty((max_eps, n_cont_actions), dtype=np.float32)

        d = m.data
        self.dt_h = d.dt / 60
        self.c_ind = d.c_inds
        if np.all(d.is_scaled):
            self.scaling = d.scaling

        for k in range(2):
            action_bd = np.array([ac[k] for ac in self.action_range])
            action_range_scaled[:, k] = self._to_scaled(action_bd, to_original=False)
        self.action_range_scaled = action_range_scaled

    def _to_scaled(self, action: Arr, to_original: bool = False,
                   extra_scaling: bool = False) -> np.ndarray:
        """Converts actions to the right range."""
        if np.array(action).shape == ():
            assert self.nb_actions == 1, "Ambiguous for more than one action!"
            action = np.array([action])
        else:
            assert len(action) == self.nb_actions, "Not the right amount of actions!"

        # This should not do anything
        cont_action = action

        # Do the extra scaling for rl agents with output in [0, 1]
        if extra_scaling and self.do_scaling:
            cont_action = self.a_scaling_pars[0] + cont_action * self.a_scaling_pars[1]

        # Revert standardization
        c_actions_scaled = cont_action
        if self.scaling is not None:
            c_actions_scaled = np.empty_like(cont_action)
            for k in range(self.nb_actions):
                c_actions_scaled[k] = trf_mean_and_std(cont_action[k], self.scaling[self.c_ind[k]], not to_original)

        # Return scaled actions
        return c_actions_scaled

    def scale_action_for_step(self, action: Arr):
        """Scales the action to be used with the model of the env.

        Overrides function in base class.
        """
        return self._to_scaled(action)

    def get_unscaled(self, curr_pred: np.ndarray, orig_ind: int) -> float:
        """Returns a part of the state scaled to original value.

        Args:
            curr_pred: Current (predicted) state.
            orig_ind: Index specifying the desired part of the state.
        """
        prep_ind = self.m.data.to_prepared(orig_ind)
        return self._state_to_scale(curr_pred[prep_ind], orig_ind=orig_ind,
                                    remove_mean=False).item()

    def _state_to_scale(self, original_state: Arr,
                        orig_ind: int,
                        remove_mean: bool = False) -> Arr:
        """Scales the state according to `orig_ind`."""
        if self.scaling is not None:
            return trf_mean_and_std(original_state, self.scaling[orig_ind], remove=remove_mean)
        return original_state


def temp_penalty(room_temp: Num, temp_bounds: RangeT, t_h: Num = 0.25):
    """Computes the penalty for temperatures out of bound.

    Needs the `room_temp` and the `temp_bounds` to be scaled
    to original scale, i.e. physically meaningful ones.
    Uses the absolute error.
    This is deprecated, use :func:`compute_room_rewards`!

    Args:
        room_temp: The actual room temperature.
        temp_bounds: The temperature bounds.
        t_h: The time step length in hours.

    Returns:
        The temperature bound violation penalty.
    """
    return t_h * linear_oob_penalty(room_temp, temp_bounds)


def room_energy_used(water_temps: Sequence, valve_action: Num, t_h: Num):
    """Computes the energy used from the water temperature and the action.

    Needs the actions and the water temperatures to be
    in the original (physical) space.
    This is deprecated, use :func:`compute_room_rewards`!

    Args:
        water_temps: The current water temperatures.
        valve_action: The action taken.
        t_h: The timestep length in hours.

    Returns:
        The amount of energy used.
    """
    d_temp = np.abs(water_temps[0] - water_temps[1])
    return np.clip(valve_action, 0.0, 1.0) * d_temp * t_h


def compute_room_rewards(actions: np.ndarray, room_temp: np.ndarray,
                         water_temp: np.ndarray, alpha: float = 50.0,
                         temp_bounds: RangeT = TEMP_BOUNDS,
                         dt: int = 15):
    """Computes all the parts that define the reward for the room env.

    Args:
        actions: The actions with shape (n,)
        room_temp: The room temperatures with shape (n,)
        water_temp: The water temperatures with shape (n, 2)
        alpha: The temperature penalty scaling factor.
        temp_bounds: The temperature bounds
        dt: The time step length in minutes.

    Returns:
        All rewards in an array with shape (n, 3)
    """
    # Check input
    n_actions = len(actions)
    check_shape(actions, (n_actions,))
    check_shape(room_temp, (n_actions,))
    check_shape(water_temp, (n_actions, 2))

    # Compute energy used
    clipped_actions = np.clip(actions, 0.0, 1.0)
    d_temp = np.abs(water_temp[:, 0] - water_temp[:, 1])
    eng_used = clipped_actions * d_temp * dt / 60.0

    # Compute temperature penalty
    temp_pen = linear_oob(room_temp, temp_bounds) * dt / 60.0

    # Compute total reward
    tot_rew = -eng_used - alpha * temp_pen

    # Put all into an array and return
    all_rew = np.empty((n_actions, 3))
    all_rew[:, 0] = tot_rew
    all_rew[:, 1] = eng_used
    all_rew[:, 2] = temp_pen

    return all_rew


class FullRoomEnv(RLDynEnv):
    """The environment modeling one room only."""
    alpha: float  #: Weight factor for temperature penalty in reward.
    temp_bounds: RangeT = TEMP_BOUNDS  #: The requested temperature range.

    # Plot defaults
    default_state_mask = np.array([0, 1, 4])
    default_series_merging = [((0, 1), "Weather")]

    def __init__(self, m: BaseDynamicsModel,
                 max_eps: int = 48,
                 temp_bounds: RangeT = None,
                 alpha: float = 2.5,
                 **kwargs):
        # Define name
        ext = make_param_ext([("NEP", max_eps), ("AL", alpha), ("TBD", temp_bounds)])
        name = "FullRoom" + ext

        # Initialize super class
        super(FullRoomEnv, self).__init__(m, max_eps, name=name, **kwargs)

        # Save parameters
        self.alpha = alpha if alpha is not None else 2.5
        d = m.data
        if temp_bounds is not None:
            self.temp_bounds = temp_bounds
        if np.all(d.is_scaled):
            self.scaling = d.scaling

        # Check model and dataset
        assert len(m.out_inds) == d.d - d.n_c, "Model not suited for this environment!!"
        assert d.d == 8 and d.n_c == 1, "Not the correct number of series in dataset!"

    def get_r_temp(self, curr_pred: np.ndarray) -> float:
        """Returns the originally scaled room temperature.

        Deprecated: Use `get_unscaled()` directly!

        Args:
            curr_pred: Current state.

        Returns:
            Room temperature
        """
        return self.get_unscaled(curr_pred, 5)

    def get_w_temp(self, curr_pred: np.ndarray) -> List[float]:
        # Deprecated as `get_r_temp`
        w_inds = 2, 3
        return [self._state_to_scale(curr_pred[i], orig_ind=i,
                                     remove_mean=False).item() for i in w_inds]

    reward_descs = [ROOM_ENERGY, TEMP_BOUND_PEN]

    def do_checks(self, verbose: int = 1):
        """Check a few properties."""

        # Check if room energy is 0 if valves closed.
        n_s = 7
        zero_ac = np.array([0.0])
        heat_ac = self._to_scaled(zero_ac, False)
        d_r = self.detailed_reward(np.ones((n_s,)), heat_ac)
        assert np.allclose(d_r[0], 0.0), f"Room energy computation incorrect: {d_r[0]}!"

        # Check same with scale_action_for_step
        h_ac2 = self.scale_action_for_step(zero_ac)
        d_r = self.detailed_reward(np.ones((n_s,)), h_ac2)
        assert np.allclose(d_r[0], 0.0), f"Second room energy computation incorrect: {d_r[0]}!"

        if verbose:
            print("Checks passed :)")

    def detailed_reward(self, curr_pred: np.ndarray, action: Arr) -> np.ndarray:

        # Compute energy used
        action_rescaled = self._to_scaled(action, True)[0]
        w_temps = self.get_w_temp(curr_pred)
        tot_energy_used = room_energy_used(w_temps, action_rescaled, self.dt_h)

        # Check for actions out of range
        action_penalty = linear_oob_penalty(action_rescaled, [0.0, 1.0])
        if action_penalty > 0:
            warnings.warn("Actions not in right range")

        # Penalty for constraint violation
        r_temp = self.get_r_temp(curr_pred)
        temp_pen = temp_penalty(r_temp, self.temp_bounds, self.dt_h)

        # Compute again with other method
        w_temps_r = np.reshape(w_temps, (1, 2))
        a_r = np.reshape(action_rescaled, (1,))
        r_temp_r = np.reshape(r_temp, (1,))
        all_rew = compute_room_rewards(a_r, r_temp_r, w_temps_r, self.alpha,
                                       self.temp_bounds, self.m.data.dt)
        assert np.allclose(all_rew[0, 1], tot_energy_used), f"{all_rew}, {tot_energy_used}"
        assert np.allclose(all_rew[0, 2], temp_pen), f"{all_rew}, {temp_pen}"

        return np.array([tot_energy_used, temp_pen])

    def compute_reward(self, curr_pred: np.ndarray, action: Arr) -> float:
        """Computes the total reward from the individual components."""
        det_rew = self.detailed_reward(curr_pred, action)
        return -det_rew[0] - self.alpha * det_rew[1]

    def episode_over(self, curr_pred: np.ndarray) -> bool:

        r_temp = self.get_r_temp(curr_pred)
        thresh = 10.0
        t_bounds = self.temp_bounds
        if r_temp > t_bounds[1] + thresh or r_temp < t_bounds[0] - thresh:
            return True
        return False

    def scale_action_for_step(self, action: Arr):
        # Get default min and max actions from bounds
        scaled_ac = self._to_scaled(np.array(self.action_range, dtype=np.float32))
        assert len(scaled_ac) == 1 and len(scaled_ac[0]) == 2, "Shape mismatch!"
        min_ac, max_ac = scaled_ac[0]

        scaled_action = self._to_scaled(action, extra_scaling=True)
        chosen_action = np.clip(scaled_action, min_ac, max_ac)
        if self.scaling is not None:
            assert not np.array_equal(action, chosen_action)

        return chosen_action

    def _get_detail_eval_title_ext(self):
        return f"Bound violation scaling factor: {self.alpha}"


class CProf(ABC):
    name: str

    @abstractmethod
    def __call__(self, t: int) -> float:
        """Returns the cost at timestep `t`."""
        pass


class ConstProfile(CProf):
    """Constant price profile."""

    def __init__(self, p: float):
        self.p = p
        self.name = f"Const_{p}"

    def __call__(self, t: int) -> float:
        return self.p


class PWProfile(CProf):
    """Some example price profile."""

    name = "PieceWise"

    def __call__(self, t: int) -> float:
        if t < 5:
            return 1.0
        elif t < 10:
            return 2.0
        elif t < 20:
            return 1.0
        elif t < 25:
            return 3.0
        elif t < 30:
            return 2.0
        else:
            return 1.0


class LowHighProfile(CProf):
    """Low- high price profile."""

    name = "LowHigh"

    low_start_ind: int
    high_start_ind: int

    def __init__(self, dt: int = 15, high_start: int = 6,
                 low_start: int = 20):

        n_ts_per_h = int(60 / dt)
        self.low_start_ind = low_start * n_ts_per_h + 1
        self.high_start_ind = high_start * n_ts_per_h + 1

    def __call__(self, t: int) -> float:
        if self.high_start_ind < t < self.low_start_ind:
            return 2.0
        else:
            return 1.0


def _get_minimum_soc(n_remain_steps: int,
                     bat_params,
                     scaled_action_range,
                     scaled_req_soc,
                     scaled_soc_bound) -> Num:
    s_min_scaled, s_max_scaled = scaled_soc_bound
    min_soc = s_min_scaled
    min_ac, max_ac = scaled_action_range
    b, c_min, gam = bat_params
    c_max = c_min + gam
    max_ds = b + max_ac * c_max
    min_soc_goal = scaled_req_soc - n_remain_steps * max_ds
    min_soc = np.maximum(min_soc, min_soc_goal)
    return min_soc


def clip_battery_action(scaled_action,
                        curr_scaled_soc,
                        scaled_soc_bound,
                        scaled_action_range,
                        bat_params,
                        use_goal_state_fallback: bool = True,
                        scaled_req_soc: Num = None,
                        time_step: int = None,
                        max_steps: int = None,
                        ):
    """Scales the actions s.t. the SoC will always be in a valid range.

    If `use_goal_state_fallback` is true, the SoC goal at the end of the
    episode will be reached by constraining the actions.

    Args:
        scaled_action: The standardized actions.
        curr_scaled_soc: The standardized current SoC.
        scaled_soc_bound: The standardized SoC bound.
        scaled_action_range: The standardized action range.
        bat_params: The parameters of the piecewise linear battery model.
        use_goal_state_fallback: Whether to assert reaching the goal SoC at end of episode.
        scaled_req_soc: The standardized SoC goal at end of episode.
        time_step: The current time step.
        max_steps: The total number of time steps until the episode is over.

    Returns:
        The clipped action that will not violate any bounds.
    """
    # Check input
    if use_goal_state_fallback:
        if time_step is None or max_steps is None or scaled_req_soc is None:
            yeet("Need to specify max_steps, scaled_req_soc and time_step")

    # Extract values from input
    min_ac, max_ac = scaled_action_range
    s_min_scaled, s_max_scaled = scaled_soc_bound
    b, c_min, gam = bat_params
    c_max = c_min + gam

    # Compute min and max action
    min_soc = s_min_scaled
    if use_goal_state_fallback:
        # Satisfy goal SoC requirements
        n_remain_steps = max_steps - time_step
        min_soc = _get_minimum_soc(n_remain_steps - 1,
                                   bat_params,
                                   scaled_action_range,
                                   scaled_req_soc,
                                   scaled_soc_bound)
    next_d_soc_min = min_soc - b - curr_scaled_soc
    if next_d_soc_min < 0:
        ac_min = np.maximum(next_d_soc_min / c_min, min_ac)
    else:
        ac_min = np.maximum(next_d_soc_min / c_max, min_ac)
    next_d_soc_max = s_max_scaled - b - curr_scaled_soc
    ac_max = np.minimum(next_d_soc_max / c_max, max_ac)

    # Clip and return
    return np.clip(scaled_action, ac_min, ac_max)


def compute_bat_energy(rescaled_action,
                       dt_h,
                       n_ts: int = None,
                       n_ts_per_eps: int = None,
                       scaled_soc=None,
                       soc_bound=None,
                       req_soc: Num = None,
                       action_range=None, *,
                       fail_violations: bool = True,
                       fail_actions: bool = True):
    if fail_actions:
        assert linear_oob_penalty(rescaled_action, action_range) <= 0.0001, "WTF"

    # Compute energy used
    energy_used = rescaled_action * dt_h

    # Check constraint violation
    if fail_violations:
        assert linear_oob_penalty(scaled_soc, soc_bound) <= 0.001, "WTF2"

        # Penalty for not having charged enough at the end of the episode.
        if n_ts > n_ts_per_eps - 1 and scaled_soc < req_soc - 0.001:
            assert linear_oob_penalty(scaled_soc, [req_soc, 100]) <= 0.0, "Model not working"

    # Total reward is minus the energy used.
    return np.array([energy_used])


class BatteryEnv(RLDynEnv):
    """The environment for the battery model.

    """

    alpha: float = 1.0  #: Reward scaling factor.
    action_range: RangeListT = [BATTERY_ACTION_BOUNDS]  #: The requested active power range.
    soc_bound: Sequence = SOC_BOUND  #: The requested state-of-charge range.
    scaled_soc_bd: np.ndarray = None  #: `soc_bound` scaled to the model space.
    req_soc: float = SOC_GOAL  #: Required SoC at end of episode.
    prev_pred: np.ndarray  #: The previous prediction.
    m: BatteryModel  #: The battery model.
    p: CProf = None  #: The cost profile.

    def __init__(self, m: BatteryModel, p: CProf = None, **kwargs):
        d = m.data

        # Add max predictions length to kwargs if not there yet.
        ep_key = 'max_eps'
        kwargs[ep_key] = kwargs.get(ep_key, 24 * 60 // d.dt // 6)

        # Define name
        name = "Battery"

        # Init base class.
        super().__init__(m, name=name, action_range=[(-100, 100)], **kwargs, init_res=False)

        assert p is None, "Cost profile does not make sense here!"
        # TODO: Remove cost profile from model!

        self.scaled_soc_bd = rem_mean_and_std(np.array(self.soc_bound), self.scaling[0])

        # Check model
        assert len(m.out_inds) == 1, "Model not suited for this environment!!"
        assert self.action_range == [(-100, 100)], "action_range value was overridden!"

        # Check underlying dataset
        assert d.d == 2 and d.n_c == 1, "Not the correct number of series in dataset!"
        assert d.c_inds[0] == 1, "Second series needs to be controllable!"

        self.reset()

    def _get_scaled_soc(self, unscaled_soc: Arr, remove_mean: bool = False) -> Arr:
        """Scales the state-of-charge."""
        return self._state_to_scale(unscaled_soc, orig_ind=0, remove_mean=remove_mean)

    reward_descs = [BAT_ENERGY_HIGH]  #: Description of the detailed reward.

    def detailed_reward(self, curr_pred: np.ndarray, action: Arr) -> np.ndarray:
        """Computes the energy used by dis- / charging the battery."""

        action_rescaled = self._to_scaled(action, True)[0]
        curr_pred = self._get_scaled_soc(curr_pred).item()

        bat_eng = compute_bat_energy(action_rescaled,
                                     self.dt_h,
                                     self.n_ts,
                                     self.n_ts_per_eps,
                                     curr_pred,
                                     self.soc_bound,
                                     self.req_soc,
                                     self.action_range[0],
                                     fail_violations=True,
                                     fail_actions=True)

        return bat_eng

    def compute_reward(self, curr_pred: np.ndarray, action: Arr) -> float:
        """Compute the reward for choosing action `action`.

        The reward takes into account the energy used.
        """
        # Return minus the energy used.
        e_used = self.detailed_reward(curr_pred, action)
        return -e_used.item() * self.alpha

    def episode_over(self, curr_pred: np.ndarray) -> bool:
        """Declare the episode as over if the SoC lies too far without bounds."""
        thresh = 10
        b = self.soc_bound
        scaled_soc = self._get_scaled_soc(curr_pred)
        if scaled_soc > b[1] + thresh or scaled_soc < b[0] - thresh:
            raise AssertionError("Battery model wrong!!")
            # return True
        return False

    def scale_action_for_step(self, action: Arr):
        # Get scaled actions and bound
        scaled_ac = self._to_scaled(np.array(self.action_range, dtype=np.float32))
        assert len(scaled_ac) == 1 and len(scaled_ac[0]) == 2, "Shape mismatch!"
        scaled_action = self._to_scaled(action, extra_scaling=True)

        # Extract and scale current soc, bound and goal
        curr_state = self.curr_state
        soc_bound_arr = np.array(self.soc_bound, copy=True)
        scaled_soc_bound = self._get_scaled_soc(soc_bound_arr, remove_mean=True)
        min_goal_soc = self._get_scaled_soc(self.req_soc, remove_mean=True)

        # Clip the actions.
        return clip_battery_action(scaled_action,
                                   curr_state,
                                   scaled_soc_bound,
                                   scaled_ac[0],
                                   self.m.params,
                                   True,
                                   min_goal_soc,
                                   self.n_ts,
                                   self.n_ts_per_eps)

    def reset(self, *args, **kwargs) -> np.ndarray:
        super().reset(*args, **kwargs)

        # Clip the values to the valid SoC range!
        self.hist[:, 0] = np.clip(self.hist[:, 0],
                                  self.scaled_soc_bd[0],
                                  self.scaled_soc_bd[1])
        return np.copy(self.hist[-1, :-self.act_dim])


def room_bat_name(p: CProf = None,
                  max_eps: int = 48,
                  temp_bounds: RangeT = None,
                  alpha: float = 2.5) -> str:
    ext = make_param_ext([("NEP", max_eps), ("AL", alpha),
                          ("TBD", temp_bounds), ("P", p)])
    return "RoomBattery" + ext


class RoomBatteryEnv(RLDynEnv):
    """The joint environment for the room model and the battery model.

    """

    alpha: float  #: Reward scaling factor.
    temp_bounds: RangeT = TEMP_BOUNDS  #: Temperature comfort bounds.
    soc_bound: Sequence = SOC_BOUND  #: The requested state-of-charge range.
    connect_inds: Tuple[int, int] = (7 * 4, 17 * 4)  #: The timestep index for disconnection and connection of the EV.
    p: CProf = None  #: The cost profile.
    m: CompositeModel
    bat_mod: BatteryModel

    reward_descs = list([ROOM_ENERGY, TEMP_BOUND_PEN, BAT_ENERGY])  #: Description of the detailed reward.

    # Indices specifying series
    inds: np.ndarray = np.array([2, 3, 5, 8], dtype=np.int32)  #: The indices of water temps, room temp and soc
    prep_inds: np.ndarray  #: The same indices but prepared.

    # Plot defaults
    default_state_mask = np.array([0, 1, 4, 7])
    default_series_merging = [((0, 1), "Weather")]

    @staticmethod
    def room_bat_name(p: CProf = None,
                      max_eps: int = 48,
                      temp_bounds: RangeT = None,
                      alpha: float = 2.5) -> str:

        ext = make_param_ext([("NEP", max_eps), ("AL", alpha),
                              ("TBD", temp_bounds), ("P", p)])
        return "RoomBattery" + ext

    @staticmethod
    def returning_soc() -> Num:
        """Defines the SoC of the EV when returning.

        You may override this and make it stochastic.

        Returns:
            The SoC of the EV.
        """
        return 30.0

    def __init__(self, m: CompositeModel,
                 p: CProf = None, *,
                 max_eps: int = 48,
                 temp_bounds: RangeT = None,
                 soc_bound: Sequence = None,
                 connect_hours: Tuple[int, int] = None,
                 alpha: float = 2.5,
                 **kwargs):

        # Define name
        name = self.room_bat_name(p, max_eps, temp_bounds, alpha)

        # Set prepared indices
        d = m.data
        self.prep_inds = d.to_prepared(self.inds)

        # Init base class.
        act_ranges = [HEAT_ACTION_BOUNDS, BATTERY_ACTION_BOUNDS]
        super().__init__(m, name=name, action_range=act_ranges,
                         n_cont_actions=2, max_eps=max_eps, **kwargs, init_res=False)

        # Save parameters
        self.alpha = 2.5 if alpha is None else alpha
        if temp_bounds is not None:
            self.temp_bounds = temp_bounds
        if soc_bound is not None:
            self.soc_bound = soc_bound
        if np.all(d.is_scaled):
            self.scaling = d.scaling

        # Battery (dis-) connection time
        if connect_hours is not None:
            n_ts_per_h = 60 // d.dt
            assert len(connect_hours) == 2, f"Invalid connect hours: {connect_hours}!!"
            c_d, c_c = connect_hours
            self.connect_inds = n_ts_per_h * c_d, n_ts_per_h * c_c

        # Set cost profile
        if p is not None:
            self.p = p
            self.reward_descs = list(self.reward_descs) + [ENG_COST]

        # Precompute scaled quantities
        soc_bound_arr = np.array(self.soc_bound, copy=True)
        bat_soc_ind = self.inds[-1]
        self.bat_soc_ind_prep = self.prep_inds[-1]
        self.scaled_soc_bound = self._state_to_scale(soc_bound_arr,
                                                     orig_ind=bat_soc_ind,
                                                     remove_mean=True)
        self.min_goal_soc_scaled = self._state_to_scale(np.array(SOC_GOAL),
                                                        orig_ind=bat_soc_ind,
                                                        remove_mean=True)

        # Set battery model
        bat_mod = self.m.model_list[-1]
        assert isinstance(bat_mod, BatteryModel), "Supposed to be a battery model!"
        self.bat_mod = bat_mod

        # Reset
        self.reset()

    def n_remain_connect(self) -> int:
        """Computes the number of timesteps left until disconnection of the EV.

        If it is already disconnected, returns 0.

        Returns:
            Number of timesteps remaining until disconnection.
        """
        disc_ind, con_ind = self.connect_inds
        curr_ind = self.get_curr_day_n()
        if disc_ind <= curr_ind < con_ind:
            return 0
        elif curr_ind < disc_ind:
            return disc_ind - curr_ind
        else:
            return disc_ind + self.n_ts_per_day - curr_ind

    def _disc_since(self) -> int:
        disc_ind, con_ind = self.connect_inds
        return self.get_curr_day_n() - disc_ind

    def get_water_temp(self, ind_pos: int, curr_pred: np.ndarray):
        """Retrieves the scaled and the original water temperatures."""
        pos_inds = [ind_pos, ind_pos + 1]
        prep_inds = self.prep_inds[pos_inds]
        inds = self.inds[pos_inds]
        scaled_water = curr_pred[prep_inds]
        orig_water = np.copy(scaled_water)
        for ct, el in enumerate(orig_water):
            orig_water[ct] = self._state_to_scale(el, orig_ind=inds[ct],
                                                  remove_mean=False).item()
        return scaled_water, orig_water

    def do_checks(self, verbose: int = 1):
        """Check a few properties."""

        # Check if room energy is 0 if valves closed.
        n_s = 8
        zero_ac = np.array([0.0, 0.0])
        heat_ac = self._to_scaled(zero_ac, False)
        d_r = self.detailed_reward(np.ones((n_s,)), heat_ac)
        assert np.allclose(d_r[0], 0.0), f"Room energy computation incorrect: {d_r[1]}!"

        # Check same with scale_action_for_step
        h_ac2 = self.scale_action_for_step(zero_ac)
        d_r = self.detailed_reward(np.ones((n_s,)), h_ac2)
        assert np.allclose(d_r[0], 0.0), f"Second room energy computation incorrect: {d_r[1]}!"

        if verbose:
            print("Checks passed :)")

    def detailed_reward(self, curr_pred: np.ndarray, action: Arr) -> np.ndarray:
        # Compute original actions
        orig_actions = self._to_scaled(action, True)

        # Compute penalty for room temperature
        r_inds = self.prep_inds[-2], self.inds[-2]
        room_temp_scaled = curr_pred[r_inds[0]]
        orig_room_temp = self._state_to_scale(room_temp_scaled,
                                              orig_ind=r_inds[1],
                                              remove_mean=False).item()
        temp_pen = temp_penalty(orig_room_temp, self.temp_bounds, self.dt_h)

        # Compute room energy
        scaled_water, orig_water = self.get_water_temp(0, curr_pred)
        room_eng = room_energy_used(orig_water, orig_actions[0], self.dt_h)

        # Compute battery energy
        bat_eng = orig_actions[1]

        # Check if the EV is connected, if not, no energy is used.
        ev_connected = self.n_remain_connect() > 0
        if not ev_connected:
            if self._disc_since() > 1:
                bat_eng = 0
            if self._disc_since() == 2 and USE_LEFTOVER_REWARD:
                # Return reward based on too high SoC
                ps = self.bat_mod.params
                bat_soc_ind_prep = self.prep_inds[-1]
                d_soc = self.min_goal_soc_scaled - curr_pred[bat_soc_ind_prep] - ps[0]
                bat_eng = 0.0
                if d_soc < 0:
                    _, bat_ac = self._to_scaled(np.array([0, d_soc / ps[1]]),
                                                extra_scaling=False,
                                                to_original=True)
                    bat_eng = bat_ac

        bat_eng *= BAT_SCALING

        # Return all parts
        all_rew = [room_eng, temp_pen, bat_eng]
        if self.p is not None:
            curr_n_ts = self.get_curr_day_n()
            all_rew += [(room_eng + bat_eng) * self.p(curr_n_ts)]
        return np.array(all_rew, dtype=np.float32)

    def compute_reward(self, curr_pred: np.ndarray, action: Arr) -> float:
        """Compute the reward for choosing action `action`.

        The reward takes into account the energy used.
        """
        det_rew = self.detailed_reward(curr_pred, action)
        if self.p is not None:
            return -det_rew[-1] - det_rew[1] * self.alpha

        # Return minus the energy used minus the temperature penalty.
        energy = det_rew[0] + det_rew[2]
        return -energy - det_rew[1] * self.alpha

    def scale_action_for_step(self, action: Arr):
        # Clips the fucking actions!
        assert len(action) == 2, f"Invalid action: {action}"
        scaled_action = self._to_scaled(action, extra_scaling=True)
        room_action, bat_action = scaled_action

        # Clip room action
        room_action_clipped = np.clip(room_action, *self.action_range_scaled[0])

        # Clip battery action
        scaled_ac = self.action_range_scaled[1]
        curr_state = self.curr_state
        bat_soc_ind_prep = self.prep_inds[-1]
        bat_action_clipped = clip_battery_action(bat_action,
                                                 curr_state[bat_soc_ind_prep],
                                                 self.scaled_soc_bound,
                                                 scaled_ac,
                                                 self.bat_mod.params,
                                                 True,
                                                 self.min_goal_soc_scaled,
                                                 self.n_ts_per_eps - self.n_remain_connect(),
                                                 max_steps=self.n_ts_per_eps + 1)

        return np.array([room_action_clipped, bat_action_clipped])

    def episode_over(self, curr_pred: np.ndarray) -> bool:
        # Let it diverge!
        return False

    def reset(self, *args, **kwargs) -> np.ndarray:
        super().reset(*args, **kwargs)

        # Handle invalid battery states
        n_remain = self.n_remain_connect()
        min_soc = _get_minimum_soc(n_remain,
                                   self.bat_mod.params,
                                   self.action_range_scaled[1],
                                   self.min_goal_soc_scaled,
                                   self.scaled_soc_bound)
        bat_soc_ind = self.inds[-1]
        min_soc_orig = self._state_to_scale(min_soc,
                                            orig_ind=bat_soc_ind,
                                            remove_mean=True)

        # Clip the values to the valid SoC range!
        bat_soc_ind_prep = self.prep_inds[-1]
        min_soc_tot = np.minimum(min_soc_orig, self.scaled_soc_bound[0])
        self.hist[:, bat_soc_ind_prep] = np.clip(self.hist[:, bat_soc_ind_prep],
                                                 min_soc_tot,
                                                 self.scaled_soc_bound[1])
        return np.copy(self.hist[-1, :-self.act_dim])

    def step(self, action: Arr) -> Tuple[np.ndarray, float, bool, Dict]:
        """Check if EV came back, if yes, set SoC as defined in `returning_soc()`.

        Calls the function of the base class for the actual stepping.

        Returns:
            See base method.
        """

        # Call base function and check if EV is connected
        n_before = self.n_remain_connect()
        curr_pred, r, ep_over, d = super().step(action)
        n_after = self.n_remain_connect()

        # If EV came back set SoC.
        if n_after > n_before:
            bat_soc_ind_prep = self.prep_inds[-1]
            bat_soc_ind = self.inds[-1]
            ret_soc = self._state_to_scale(self.returning_soc(),
                                           orig_ind=bat_soc_ind,
                                           remove_mean=True)
            self.hist[-1, bat_soc_ind_prep] = ret_soc
            curr_pred[bat_soc_ind_prep] = ret_soc

        # Return data
        return curr_pred, r, ep_over, d
