import warnings
from typing import Sequence, Union, Tuple

import numpy as np

from agents.base_agent import AgentBase
from envs.dynamics_envs import FullRoomEnv, RoomBatteryEnv, BatteryEnv
from util.util import Arr, Num


def get_const_agents(env: Union[FullRoomEnv, RoomBatteryEnv, BatteryEnv]
                     ) -> Tuple['ConstActionAgent', 'ConstActionAgent']:
    """Defines two constant agents that can be used for analysis.

    Args:
        env: The environment.

    Returns:
        Tuple with two ConstActionAgent
    """
    n_agents = 2

    heat_pars = (0.0, 1.0)
    bat_pars = (-3.0, 6.0)

    # Define constant action based on env.
    if isinstance(env, FullRoomEnv):
        c = [np.array(heat_pars[i]) for i in range(n_agents)]
    elif isinstance(env, BatteryEnv):
        bat_pars = (10.0, -8.0)
        c = [np.array(bat_pars[i]) for i in range(n_agents)]
    elif isinstance(env, RoomBatteryEnv):
        c = [np.array([heat_pars[i], bat_pars[i]]) for i in range(n_agents)]
    else:
        raise TypeError(f"Env: {env} not supported!")

    a1, a2 = ConstActionAgent(env, c[0]), ConstActionAgent(env, c[1])

    # Set plot name
    if isinstance(env, FullRoomEnv):
        a1.plot_name = "Valves Closed"
        a2.plot_name = "Valves Open"
    elif isinstance(env, BatteryEnv):
        a1.plot_name = "Charging (10 kW)"
        a2.plot_name = "Discharging (8 kW)"
    elif isinstance(env, RoomBatteryEnv):
        a1.plot_name = "Closed, Discharge"
        a2.plot_name = "Open, Charge"
    return a1, a2


class RuleBasedAgent(AgentBase):
    """Agent applying rule-based heating control.

    """
    bounds: Sequence  #: The sequence specifying the rule for control.
    const_charge_rate: Num
    env: Union[FullRoomEnv, RoomBatteryEnv]

    w_inds_orig = [2, 3]
    r_temp_ind_orig = 5

    def __init__(self, env: Union[FullRoomEnv, RoomBatteryEnv],
                 rule: Sequence,
                 const_charge_rate: Num = None,
                 strict: bool = False,
                 rbc_dt_inc: float = None):
        """Initializer.

        Args:
            env: The RL environment.
            rule: The temperature bounds.
            const_charge_rate: The charging rate if the env includes the battery.
            strict: Whether to apply strict heating / cooling, start as soon as the
                room temperature deviates from the midpoint of the temperature bounds.
        """
        name = "RuleBasedControl"
        super().__init__(env, name=name)

        # Check input
        assert len(rule) == 2, "Rule needs to consist of two values!"
        if isinstance(env, RoomBatteryEnv):
            assert const_charge_rate is not None, "Need to specify charging rate!"
        elif const_charge_rate is not None:
            warnings.warn("Ignored charging rate!")
            const_charge_rate = None

        # Store parameters
        self.const_charge_rate = const_charge_rate
        if strict:
            mid = 0.5 * (rule[0] + rule[1])
            self.bounds = (mid, mid)
        else:
            if rbc_dt_inc is not None:
                rule = (rule[0] + rbc_dt_inc, rule[1])
            self.bounds = rule

        self.plot_name = "Rule-Based"

    def __str__(self):
        return f"Rule-Based Agent with bounds {self.bounds}"

    def get_action(self, state) -> Arr:
        """Defines the control strategy.

        Args:
            state: The current state.

        Returns:
            Next control action.
        """
        # Find water and room temperatures
        w_in_temp = self.env.get_unscaled(state, self.w_inds_orig[0])
        r_temp = self.env.get_unscaled(state, self.r_temp_ind_orig)

        # Determine if you want to do heating / cooling or not
        heat_action = 0.0
        if r_temp < self.bounds[0] and w_in_temp > r_temp:
            # Heating
            heat_action = 1.0
        if r_temp > self.bounds[1] and w_in_temp < r_temp:
            # Cooling
            heat_action = 1.0
        # Return
        final_action = heat_action
        if self.const_charge_rate is not None:
            final_action = np.array([heat_action, self.const_charge_rate])
        return final_action


class ConstActionAgent(AgentBase):
    """Applies a constant control input.

    Can be used for comparison, e.g. if you want
    to compare an agent to always heating or never heating.
    Does not really need the environment.
    """
    rule: Arr  #: The constant control input / action.
    out_num: int  #: The dimensionality of the action space.

    def __init__(self, env, rule: Arr):
        try:
            if len(rule) > 0:
                name = f"Const_{'_'.join(str(e) for e in rule)}"
            else:
                raise TypeError
        except TypeError:
            name = f"Const_{rule}"
        super().__init__(env, name=name)

        self.out_num = env.nb_actions
        self.rule = rule

        # Check rule
        if isinstance(rule, (np.ndarray, np.generic)):
            r_s, n_out = rule.shape, self.out_num
            if self.out_num > 1:
                assert r_s == (n_out,), f"Rule shape: {r_s} incompatible!"
            else:
                assert r_s == (n_out,) or r_s == (), f"Rule shape: {r_s} incompatible!"

    def __str__(self):
        return f"Constant Agent with value {self.rule}"

    def get_action(self, state) -> Arr:
        """Defines the control strategy.

        Using broadcasting it can handle numpy array rules
        of shape (`out_num`, )

        Args:
            state: The current state.

        Returns:
            Next control action.
        """
        return self.rule * np.ones((self.out_num,), dtype=np.float32)
