"""Defines the interface for RL agents.

Also provides a function to remove agents that
were not trained for long, e.g. for debugging or testing, which is
:func:`remove_agents`.

Further also functions to up- and download agents to
Google Drive, :func:`upload_trained_agents` and :func:`download_trained_agents`.
"""
import os
import warnings
from abc import ABC, abstractmethod
from typing import Dict, TYPE_CHECKING, Callable

import numpy as np

from util.numerics import npf32
from util.share_data import upload_folder_zipped, download_and_extract_zipped_folder
from util.util import Arr, fix_seed, MODEL_DIR, create_dir, remove_files_in_sub_folders
from util.visualize import rl_plot_path

if TYPE_CHECKING:
    from envs.base_dynamics_env import DynEnv

# Define directory for agent models
RL_MODEL_DIR = os.path.join(MODEL_DIR, "RL")  #: Folder for RL models.
create_dir(RL_MODEL_DIR)


def upload_trained_agents(verbose: int = 1):
    """Uploads all RL models to Google Drive.

    Uploads all data in folder `RL_MODEL_DIR`.
    """
    if verbose:
        print("Uploading agent neural network parameters to Google Drive.")
    upload_folder_zipped(RL_MODEL_DIR)


def download_trained_agents(verbose: int = 1):
    """Download trained agents from Google Drive.

    They need to be in a folder named `RL` and will
    be put into the folder `RL_MODEL_DIR`.
    """
    if verbose:
        print("Downloading agent neural network parameters from Google Drive.")
    download_and_extract_zipped_folder("RL", RL_MODEL_DIR)


def remove_agents(min_steps: int = 10000, verbose: int = 5) -> None:
    """Removes all agents that were trained for less than `min_steps` steps.

    For cleaning up agents that were produced when testing
    something or debugging. Also deletes empty folders, but not
    if the folder is empty only after removing the agents, so you may
    want to run it twice.

    Args:
        min_steps: Minimum number of training steps for an agent not to be
            deleted.
        verbose: Whether to print infos.
    """
    def remove_f(f):
        rem_file = False
        try:
            n_ep = int(f.split("_")[1][3:])
            rem_file = n_ep < min_steps
        except (IndexError, ValueError):
            if verbose:
                print(f"Invalid file name: {f}")
        return rem_file

    remove_files_in_sub_folders(RL_MODEL_DIR, remove_f,
                                True, verbose=verbose > 0)

    def remove_agent_eval(f):
        rem_file = False

        # Remove analysis plots

        i = f.find("_DDPG_")
        if i >= 0:
            num = f[(i + 6):].split(".")[0]
            try:
                n_eps = int(num)
                rem_file = n_eps < min_steps
            except ValueError:
                n_str = num.split("_")[0]
                if not n_str[:3] == "NEP":
                    n_eps = int(n_str)
                    rem_file = n_eps < min_steps
                else:
                    n_eps = int(n_str[3:])
                    rem_file = n_eps < min_steps

        # Remove train rewards plots
        try:
            if f.find("DDPG_NEP") >= 0:
                n_eps = int(f[8:].split("_")[0])
                rem_file = n_eps < min_steps
        except ValueError as e:
            pass
            # print(f"{e} happened.")

        # Remove evaluation plot
        try:
            if f.find("DetailAnalysis") >= 0:
                n_eps = int(f.split("_")[1])
                rem_file = n_eps < min_steps
        except ValueError as e:
            pass
            # print(f"{e} happened.")

        return rem_file

    remove_files_in_sub_folders(rl_plot_path, remove_agent_eval,
                                True, verbose=verbose > 0)


class AbstractAgent(ABC):
    """Base class for all agents."""

    @abstractmethod
    def get_action(self, state) -> Arr:
        """Defines the control strategy.

        Args:
            state: The current state.

        Returns:
            Next control action.
        """
        pass


class AgentBase(AbstractAgent, ABC):
    """Base class for an agent / control strategy.

    Might be specific for a certain environment accessible
    by attribute `env`.
    """
    env: 'DynEnv'  #: The corresponding environment
    name: str  #: The name of the Agent / control strategy
    fit_data: str = None
    plot_name: str = None

    def __init__(self, env: 'DynEnv', name: str = "Abstract Agent"):
        self.env = env
        self.name = name

    def fit(self, verbose: int = 0, train_data: str = "") -> None:
        """No fitting needed."""
        pass

    def get_plot_name(self) -> str:
        if self.plot_name is not None:
            return self.plot_name
        return self.get_short_name()

    def get_short_name(self) -> str:
        return self.name

    def get_info(self) -> Dict:
        return {}

    def __str__(self):
        """Generic string conversion."""
        return f"Agent of class {self.__class__.__name__} with name {self.name}"

    def eval(self, n_steps: int = 100, reset_seed: bool = False,
             detailed: bool = False,
             use_noise: bool = False, scale_states: bool = False,
             episode_marker: Callable = None,
             return_inds: bool = False,
             verbose: int = 0):
        """Evaluates the agent for a given number of steps.

        If the number is greater than the number of steps in an episode, the
        env is reset and a new episode is started.

        Args:
            n_steps: Number of steps.
            reset_seed: Whether to reset the seed at start.
            detailed: Whether to return all parts of the reward.
            use_noise: Whether to use noise during the evaluation.
            scale_states: Whether to scale the state trajectory to
                original values, only used if `detailed` is True.
            episode_marker: Function mapping from state to natural numbers.
            return_inds:
            verbose:

        Returns:
            The mean received reward if `detailed` is False, else
            all the rewards for all steps.
        """
        # Fix seed if needed.
        if reset_seed:
            fix_seed()

        if verbose:
            print(f"Evaluating agent: {self}")

        # Initialize env and reward.
        s_curr = self.env.reset(use_noise=use_noise)
        ep_mark = 0 if episode_marker is None else episode_marker(s_curr)
        all_rewards = npf32((n_steps,))

        ret_inds = None
        ret_ct = 1
        if return_inds:
            ret_inds = np.empty(((n_steps + 1) // self.env.n_ts_per_eps + 1,), dtype=np.int)
            ret_inds[0] = self.env.curr_ind

        # Detailed stuff
        det_rewards, state_t, ep_marks = None, None, None
        actions, scaled_actions = None, None
        if detailed:
            n_det = len(self.env.reward_descs)
            n_ac = self.env.act_dim
            n_states = self.env.state_dim - n_ac
            actions = npf32((n_steps, n_ac), fill=np.nan)
            scaled_actions = npf32((n_steps, n_ac), fill=np.nan)
            det_rewards = npf32((n_steps, n_det), fill=np.nan)
            state_t = npf32((n_steps, n_states), fill=np.nan)
            ep_marks = npf32((n_steps, ), fill=np.nan)
        elif scale_states:
            warnings.warn(f"Argument: {scale_states} ignored!")

        # Evaluate for `n_steps` steps.
        for k in range(n_steps):

            # Determine action
            a = self.get_action(s_curr)
            scaled_a = self.env.scale_action_for_step(a)

            # Save actions
            if actions is not None:
                actions[k, :] = a
                scaled_actions[k, :] = scaled_a

            # Execute step
            s_curr, r, fin, _ = self.env.step(a)

            # Store rewards
            all_rewards[k] = r
            if det_rewards is not None:
                det_rew = self.env.detailed_reward(s_curr, scaled_a)
                det_rewards[k, :] = det_rew
                state_t[k, :] = s_curr
                ep_marks[k] = ep_mark

            # Reset env if episode is over.
            if fin:
                s_curr = self.env.reset(use_noise=use_noise)
                ep_mark = 0 if episode_marker is None else episode_marker(s_curr)
                if return_inds:
                    ret_inds[ret_ct] = self.env.curr_ind
                    ret_ct += 1

        # This is an ugly hack, because I am running out of time!
        if hasattr(self, 'action_range') and self.action_range is not None:
            # assert hasattr(self, 'get_info'), f"(Go fuck your-) self: {self}"
            scaling = self.get_info().get('action_scaled_01')
            p1 = np.array([i[0] for i in scaling], dtype=np.float32)
            p2 = np.array([i[1] - i[0] for i in scaling], dtype=np.float32)
            a_scaling_pars = p1, p2
            actions = a_scaling_pars[0] + actions * a_scaling_pars[1]

        # Return all rewards
        if detailed:
            if scale_states:
                state_t = self.env.scale_state(state_t, remove_mean=False)
            if return_inds:
                return all_rewards, det_rewards, state_t, ep_marks, actions, scaled_actions, ret_inds
            return all_rewards, det_rewards, state_t, ep_marks, actions, scaled_actions

        # Return mean reward.
        return np.sum(all_rewards) / n_steps
