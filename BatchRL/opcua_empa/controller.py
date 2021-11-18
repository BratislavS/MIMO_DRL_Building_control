"""Controller interface for opcua client.

Defines controllers that can be used to
do control on the real system using the opcua client.
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Tuple
from typing import TYPE_CHECKING

import numpy as np

from agents.base_agent import AgentBase, AbstractAgent
from opcua_empa.opcuaclient_subscription import MAX_TEMP, MIN_TEMP
from util.numerics import int_to_sin_cos
from util.util import Num, get_min_diff, day_offset_ts, print_if_verb, ts_per_day, floor_datetime_to_min

if TYPE_CHECKING:
    # Avoiding cyclic imports for type checking
    from data_processing.dataset import Dataset
    from envs.dynamics_envs import RLDynEnv


class Controller(ABC):
    """Base controller interface.

    A controller needs to implement the __call__ function
    and optionally a termination criterion: `terminate()`.
    """

    state: np.ndarray = None

    @abstractmethod
    def __call__(self, values):
        """Returns the current control input."""
        pass

    def terminate(self) -> bool:
        return False

    def set_state(self, curr_state: np.ndarray) -> None:
        self.state = curr_state


class FixTimeController(Controller, ABC):
    """Fixed-time controller.

    Runs for a fixed number of timesteps.
    """
    max_n_minutes: int  #: The maximum allowed runtime in minutes.

    _start_time: datetime  #: The starting time.

    def __init__(self, max_n_minutes: int = None):
        self.max_n_minutes = max_n_minutes
        self._start_time = datetime.now()

    def terminate(self) -> bool:
        """Checks if the maximum time is reached.

        Returns:
            True if the max. runtime is reached, else False.
        """
        if self.max_n_minutes is None:
            return False
        h_diff = get_min_diff(self._start_time, t2=None)
        return h_diff > self.max_n_minutes


ControlT = List[Tuple[int, Controller]]  #: Room number to controller map type


class FixTimeConstController(FixTimeController):
    """Const Controller.

    Runs for a fixed amount of time if `max_n_minutes` is specified.
    Sets the value to be controlled to constant `val`.
    Control inputs do not depend on current time or on state!
    """

    val: Num  #: The numerical value to be set.

    def __init__(self, val: Num = MIN_TEMP, max_n_minutes: int = None):
        super().__init__(max_n_minutes)
        self.val = val

    def __call__(self, values=None) -> Num:
        return self.val


class ToggleController(FixTimeController):
    """Toggle controller.

    Toggles every `n_mins` between two values.
    Control inputs only depend on current time and not on state!
    """

    def __init__(self, val_low: Num = MIN_TEMP, val_high: Num = MAX_TEMP, n_mins: int = 2,
                 start_low: bool = True, max_n_minutes: int = None):
        """Controller that toggles every `n_mins` between two values.

        Args:
            val_low: The lower value.
            val_high: The higher value.
            n_mins: The number of minutes in an interval.
            start_low: Whether to start with `val_low`.
            max_n_minutes: The maximum number of minutes the controller should run.
        """
        super().__init__(max_n_minutes)
        self.v_low = val_low
        self.v_high = val_high
        self.dt = n_mins
        self.start_low = start_low

    def __call__(self, values=None) -> Num:
        """Computes the current value according to the current time."""
        min_diff = get_min_diff(self._start_time, t2=None)
        is_start_state = int(min_diff) % (2 * self.dt) < self.dt
        is_low = is_start_state if self.start_low else not is_start_state
        return self.v_low if is_low else self.v_high


class ValveToggler(FixTimeController):
    """Controller that toggles as soon as the valves have toggled."""

    n_delay: int  #: How many steps to wait with toggling back.
    TOL: float = 0.05

    _step_count: int = 0
    _curr_valve_state: bool = False

    def __init__(self, n_steps_delay: int = 10, n_steps_max: int = 60 * 60,
                 verbose: int = 0):
        super().__init__(n_steps_max)
        self.n_delay = n_steps_delay
        self.verbose = verbose

    def __call__(self, values=None):

        v = self.state[4]  # Extract valve state
        if v > 1.0 - self.TOL:
            if not self._curr_valve_state:
                # Valves just opened
                self._step_count = 0
                print_if_verb(self.verbose, "Valves opened!!!")
                self._curr_valve_state = True
        elif v < self.TOL:
            if self._curr_valve_state:
                # Valves just closed
                print_if_verb(self.verbose, "Valves closed!!!")
                self._step_count = 0
                self._curr_valve_state = False

        ret_min = self._curr_valve_state

        # If valves just switched, ignore change
        if self._step_count < self.n_delay:
            ret_min = not ret_min

        # Convert bool to temperature
        ret = MIN_TEMP if ret_min else MAX_TEMP

        # Increment and return
        self._step_count += 1
        return ret


class RuleBased(FixTimeController):
    """Rule based heating agent.

    Starts heating as soon as the temperature drops below `self.min_temp`.
    """
    def __call__(self, values=None):
        t = self.state[5]  # Extract room temp
        return MIN_TEMP if t >= self.min_temp else MAX_TEMP

    def __init__(self, min_temp: float = 21.0, n_steps_max: int = 60 * 60,
                 verbose: int = 0):
        super().__init__(n_steps_max)
        self.min_temp = min_temp
        self.verbose = verbose


def setpoint_toggle_frac(prev_state: bool, dt: int, action: Num, delay_open: Num,
                         delay_close: Num, tol: float = 0.05) -> Tuple[float, bool]:
    """Computes the time the setpoint needs to toggle.

    Since the opening and the closing of the valves are delayed,
    the setpoint needs to change earlier to ensure the correct valve behavior.

    This has to be computed once at every beginning of a timestep
    of length `dt`.

    Args:
        action: The action in [0, 1]
        prev_state: The previous valve state, open: True, closed: False
        delay_open: The time needed to open the valves in minutes.
        delay_close: The time needed to close the valves in minutes.
        dt: The number of minutes in a timestep.
        tol: Tolerance

    Returns:
        The setpoint toggle time in [0, 2].
    """
    # Check input
    assert tol >= 0 and 0.0 <= action <= 1.0
    assert delay_close >= 0 and delay_open >= 0, "Delays cannot be negative!"

    # Compute toggle time
    valve_tog = action if prev_state else 1.0 - action
    valve_tog_approx = 2.0 if valve_tog + tol >= 1.0 else valve_tog
    delay_needed = delay_close if prev_state else delay_open
    res = max(0.0, valve_tog_approx - delay_needed / dt)
    next_state = prev_state if res >= 1.0 else not prev_state
    return res, next_state


def setpoint_from_fraction(setpoint_frac: float, prev_state: bool,
                           next_state: bool, dt: int, start_time: np.datetime64 = None,
                           curr_time: np.datetime64 = None) -> bool:
    """Computes the current setpoint according to the current toggle fraction.

    Handles the current time.

    Args:
        setpoint_frac: The current setpoint fraction as computed
            by :func:`setpoint_toggle_frac`.
        prev_state: Previous valve state.
        next_state: Valve state at the end of the current timestep.
        dt: The number of minutes in a timestep.
        start_time: Start time of the step, automatically computed if None.
        curr_time: Current time, automatically computed if None.

    Returns:
        Whether the temperature setpoint should be set to high.
    """
    # Handle datetimes
    td = np.timedelta64(dt, 'm')
    if curr_time is None:
        curr_time = np.datetime64('now')
    if start_time is None:
        start_time = floor_datetime_to_min(curr_time, dt)
    time_fraction_passed = (curr_time - start_time) / td
    assert time_fraction_passed <= 1.0

    # Use setpoint_frac to determine output
    return prev_state if time_fraction_passed < setpoint_frac else next_state


class BaseRLController(FixTimeController):
    """Controller that uses a RL agent to do control."""

    default_val: Num = 21.0
    agent: AbstractAgent = None  #: RL agent
    dt: int = None

    #: The valve opening and closing delays in minutes.
    valve_delays: Tuple[float, float] = (0.5, 0.5)

    verbose: int
    const_debug: bool  #: Whether to output a constant value

    # Protected member variables
    _step_start_state: bool = None  #: Open: True, closed: False
    _next_start_state: bool = None
    _toggle_time_fraction: float = None
    _init_phase: bool = True

    _curr_ts_ind: int

    def __init__(self, rl_agent: AbstractAgent,
                 dt: int,
                 n_steps_max: int = 60 * 60,
                 const_debug: bool = False,
                 verbose: int = 3):
        super().__init__(n_steps_max)
        self.agent = rl_agent
        self.dt = dt
        self.verbose = verbose
        self.const_debug = const_debug

        self._curr_ts_ind = self.get_dt_ind()

    def get_dt_ind(self) -> int:
        """Computes the index of the current timestep."""
        t_now = np.datetime64('now')
        return day_offset_ts(t_now, mins=self.dt, remaining=False) - 1

    def prepared_state(self, next_ts_ind: int = None) -> np.ndarray:
        return self.state

    def __call__(self, values=None) -> float:

        if self._step_start_state is None:
            # __call__ is called for the first time,
            # set _step_start_state to valve state.
            valve_state = self.state[4]
            self._step_start_state = valve_state > 0.5

        # If next timestep started, compute next control input
        next_ts_ind = self.get_dt_ind()
        if next_ts_ind != self._curr_ts_ind:
            self._init_phase = False
            # Update start state
            self._step_start_state = self._next_start_state

            prepared_state = self.prepared_state(next_ts_ind)
            ac = self.agent.get_action(prepared_state)
            if self.verbose:
                print(f"Step {next_ts_ind}, Action: {ac}")

            # Compute toggle point
            tog_frac, next_state = setpoint_toggle_frac(self._step_start_state,
                                                        self.dt, ac,
                                                        *self.valve_delays)
            self._next_start_state = next_state
            self._toggle_time_fraction = tog_frac
            self._curr_ts_ind = next_ts_ind

        # If it is still in warm-up phase return default value
        if self._init_phase:
            return self.default_val

        # Find and return the actual temperature setpoint
        tog_state = setpoint_from_fraction(self._toggle_time_fraction,
                                           self._step_start_state,
                                           self._next_start_state,
                                           self.dt)

        if self.const_debug:
            return 21.0
        return MAX_TEMP if tog_state else MIN_TEMP


class RLController(BaseRLController):
    """Controller using an :class:`agents.base_agent.AgentBase` RL agent to do control."""

    data_ref: 'Dataset' = None  #: Dataset of model of env
    env: 'RLDynEnv' = None

    _scaling: np.ndarray = None

    def __init__(self, rl_agent: AgentBase, n_steps_max: int = 60 * 60,
                 const_debug: bool = False,
                 verbose: int = 3):

        self.data_ref = rl_agent.env.m.data
        dt = self.data_ref.dt
        super().__init__(rl_agent, dt, n_steps_max, verbose=verbose > 0,
                         const_debug=const_debug)

        assert isinstance(self.agent, AgentBase)
        env = self.agent.env

        # Check if model is a room model with or without battery.
        # Cannot directly check with isinstance because of cyclic imports.
        env_class_name = env.__class__.__name__
        if env_class_name == "RoomBatteryEnv":
            self.battery = True
            print_if_verb(self.verbose, "Full model including battery!")
        elif env_class_name == "FullRoomEnv":
            self.battery = False
            print_if_verb(self.verbose, "Room only model!")
        else:
            raise NotImplementedError(f"Env: {env} is not supported!")

        # Save scaling info
        assert not self.data_ref.partially_scaled, "Fuck this!"
        if self.data_ref.fully_scaled:
            self._scaling = self.data_ref.scaling

    def prepared_state(self, next_ts_ind: int = None) -> np.ndarray:
        """Prepares the current state to be fed to the agent."""
        time_state = self.add_time_to_state(self.state, next_ts_ind)
        if self.battery:
            # TODO: Implement this case
            raise NotImplementedError("Fuck")
        scaled_state = self.scale_for_agent(time_state)
        return scaled_state[self.data_ref.non_c_inds]

    def scale_for_agent(self, curr_state, remove_mean: bool = True) -> np.ndarray:
        """Scales the given state."""
        assert len(curr_state) == 8 + 2 * self.battery, "Shape mismatch!"
        if remove_mean:
            return (curr_state - self._scaling[:, 0]) / self._scaling[:, 1]
        else:
            return self._scaling[:, 1] * curr_state + self._scaling[:, 0]

    def add_time_to_state(self, curr_state: np.ndarray, t_ind: int = None) -> np.ndarray:
        """Appends the sin and cos of the daytime to the state."""
        assert len(curr_state) == 6, f"Invalid shape of state: {curr_state}"
        if t_ind is None:
            t_ind = self.get_dt_ind()
        n_ts_per_day = ts_per_day(self.dt)
        t = np.array(int_to_sin_cos(t_ind, n_ts_per_day))
        return np.concatenate((curr_state, t))


class ValveTest2Controller(BaseRLController):
    """Testing controller.

    Uses the RL agent setting, i.e. makes a decision
    at the beginning of each 15 minutes interval.
    Assumes no valve delay, therefore can be used to
    measure the valve delay.
    """

    class RandomAgent(AbstractAgent):
        """Helper agent class.

        Returns a random action in each step.
        """
        def __init__(self, verbose):
            self.verbose = verbose

        def get_action(self, state) -> float:
            rand_ac = np.random.uniform(0.0, 1.0)
            if self.verbose:
                print(f"Action: {rand_ac}")
            return rand_ac

    def __init__(self, n_hours: int = 3, verbose: int = 1):
        super().__init__(self.RandomAgent(verbose), dt=15, n_steps_max=n_hours * 60)
        self.valve_delays = (0.0, 0.0)
