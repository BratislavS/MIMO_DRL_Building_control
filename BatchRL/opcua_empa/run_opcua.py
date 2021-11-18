"""Module for running the opcua client.

May be removed later and moved to BatchRL.py if
it is high-level enough.
"""
from typing import List

from agents.keras_agents import default_ddpg_agent, DEF_RL_LR
from dynamics.load_models import load_room_env
from opcua_empa.controller import ValveToggler, ValveTest2Controller, FixTimeConstController, RLController, RuleBased
from opcua_empa.opcua_util import check_room_list
from opcua_empa.opcuaclient_subscription import OpcuaClient
from opcua_empa.room_control_client import run_control
from tests.test_opcua import OfflineClient
from util.util import prog_verb, ProgWrap, DEFAULT_ROOM_NR, DEFAULT_EVAL_SET


def try_opcua(verbose: int = 1, room_list: List[int] = None, debug: bool = True):
    """Runs the opcua client."""

    if verbose:
        if debug:
            print("Running in debug mode!")

    # Choose experiment name
    exp_name = "Test"

    # Check list with room numbers
    check_room_list(room_list)

    # Define room and control
    # tc = ToggleController(n_mins=60 * 100, start_low=True, max_n_minutes=60 * 16)
    # tc = ValveToggler(n_steps_delay=30, n_steps_max=2 * 60)
    tc = ValveTest2Controller()
    room_list = [43] if room_list is None else room_list
    used_control = [(i, tc) for i in room_list]
    if debug:
        room_list = [41]
        used_control = [(r, ValveToggler(n_steps_delay=30))
                        for r in room_list]
        exp_name = "Offline_DebugValveToggle"

    # Use offline client in debug mode
    cl_class = OfflineClient if debug else OpcuaClient
    run_control(used_control=used_control,
                exp_name=exp_name,
                user=None,
                password=None,
                verbose=verbose,
                _client_class=cl_class)


def run_rl_control(room_nr: int = DEFAULT_ROOM_NR,
                   notify_failure: bool = False,
                   debug: bool = False,
                   verbose: int = 5,
                   n_steps: int = None,
                   hop_eval_set: str = DEFAULT_EVAL_SET,
                   notify_debug: bool = None,
                   agent_lr: float = DEF_RL_LR,
                   **env_kwargs,
                   ):
    """Runs the RL agent via the opcua client.

    Args:
        room_nr:
        notify_failure: Whether to send a mail upon failure.
        debug:
        verbose:
        n_steps:
        hop_eval_set:
        agent_lr:
        notify_debug: Whether to use debug mail address to send notifications,
            Ignored, if `notify_failure` is False.
        **env_kwargs: Keyword arguments for environment, see :func:`load_room_env`.
    """
    full_debug: bool = False

    assert room_nr in [41, 43], f"Invalid room number: {room_nr}"

    if notify_debug is None:
        notify_debug = debug
    msg = f"Using {'debug' if notify_debug else 'original'} " \
          f"mail address for notifications."
    if verbose:
        print(msg)

    next_verbose = prog_verb(verbose)
    m_name = "FullState_Comp_ReducedTempConstWaterWeather"
    n_hours = 24 * 3 if not debug else 3

    rl_cont = None
    if not full_debug:
        # Load the model and init env
        with ProgWrap(f"Loading environment...", verbose > 0):
            env = load_room_env(m_name,
                                verbose=next_verbose,
                                room_nr=room_nr,
                                hop_eval_set=hop_eval_set,
                                **env_kwargs)

        # Define default agents and compare
        with ProgWrap(f"Initializing agents...", verbose > 0):
            agent = default_ddpg_agent(env, n_steps, fitted=True,
                                       verbose=next_verbose,
                                       hop_eval_set=hop_eval_set,
                                       lr=agent_lr)
            if verbose:
                print(agent)

        # Choose controller
        rl_cont = RLController(agent, n_steps_max=3600 * n_hours,
                               const_debug=debug,
                               verbose=next_verbose)
    else:
        if verbose:
            print("Using constant model without an agent.")

    f_cont = FixTimeConstController(val=21.0, max_n_minutes=n_hours * 60)
    cont = f_cont if full_debug else rl_cont
    used_control = [(room_nr, cont)]

    exp_name = "DefaultExperiment"
    if debug:
        exp_name += "Debug"

    # Run control
    run_control(used_control=used_control,
                exp_name=exp_name,
                user=None,
                password=None,
                debug=notify_debug,
                verbose=verbose,
                _client_class=OpcuaClient,
                notify_failures=notify_failure)


def run_rule_based_control(room_nr: int = DEFAULT_ROOM_NR, *,
                           min_temp: float = 21.0,
                           name_ext: str = "",
                           notify_debug: bool = True,
                           verbose: int = 5,
                           ) -> None:
    """Runs rule-based controller for heating season.

    Args:
        room_nr:
        min_temp:
        name_ext:
        notify_debug:
        verbose:
    """

    exp_name = f"RuleBased_{min_temp}{name_ext}"
    controller = RuleBased(min_temp, n_steps_max=3600 * 24 * 31)
    used_control = [(room_nr, controller)]

    # Run control
    run_control(used_control=used_control,
                exp_name=exp_name,
                user=None,
                password=None,
                debug=notify_debug,
                verbose=verbose,
                _client_class=OpcuaClient,
                notify_failures=True)
