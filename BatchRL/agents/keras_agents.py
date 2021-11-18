"""A few keras RL agents.

Based on the agents of the keras-rl library, the agents
here are basically wrappers of those adding functionality
to work with the present framework.
"""
import os
from typing import Sequence, Dict

from keras import Input, Model, Sequential
from keras.layers import Flatten, Concatenate, Activation
from keras.optimizers import Adam
from rl.agents import DDPGAgent
from rl.agents.dqn import DQNAgent, NAFAgent
from rl.core import Agent
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy
from rl.random import OrnsteinUhlenbeckProcess

from agents.base_agent import AgentBase, RL_MODEL_DIR
from envs.dynamics_envs import FullRoomEnv, RLDynEnv, RangeListT
from ml.keras_util import getMLPModel, KerasBase
from util.util import make_param_ext, train_decorator, DEFAULT_TRAIN_SET, get_rl_steps, prog_verb, ProgWrap, \
    DEFAULT_EVAL_SET
from util.visualize import plot_rewards


# Constants, do not change!
DEF_RL_LR = 0.00001
DEF_GAMMA = 0.99

# Change values here
used_lr = 0.001
used_gamma = DEF_GAMMA


def ddpg_agent_name(n_steps: int, lr: float = DEF_RL_LR,
                    gam: float = DEF_GAMMA) -> str:
    lr_ext = "" if lr == DEF_RL_LR else f"_LR{lr}"
    g_ext = "" if gam == DEF_GAMMA else f"_G{gam}"
    return f"DDPG_NEP{n_steps}{lr_ext}{g_ext}"


class KerasBaseAgent(AgentBase, KerasBase):
    """The interface for all keras-rl agent wrappers."""

    m: Agent  #: The keras-rl agent.
    model_path: str = RL_MODEL_DIR  #: Where to store the model parameters.

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if kwargs.get('name') is None:
            print("Please provide a name for the agent!")

    def get_action(self, state):
        """Use the keras-rl model to get an action."""
        if self.m.training:
            # TODO: Is this OK?
            self.m.training = False
        assert not self.m.training, "Still in training mode!"
        return self.m.forward(state)


class DQNBaseAgent(KerasBaseAgent):

    def __init__(self, env: FullRoomEnv, n_train_steps: int = 10000):
        # Initialize super class
        name = "DQN"
        super().__init__(env=env, name=name)

        self.n_train_steps = n_train_steps

        # Build Q-function model.
        nb_actions = env.nb_actions
        n_state_vars = env.m.n_pred
        inputs = Input(shape=(1, n_state_vars))
        flat_inputs = Flatten()(inputs)
        model = getMLPModel(out_dim=nb_actions)
        model = Model(inputs=inputs, outputs=model(flat_inputs))
        # model.summary()

        # Configure and compile our agent.
        memory = SequentialMemory(limit=50000, window_length=1)
        policy = BoltzmannQPolicy()
        self.m = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
                          policy=policy,
                          gamma=0.9,
                          train_interval=100,
                          target_model_update=500)
        self.m.compile(Adam(lr=1e-5), metrics=['mae'])

        raise NotImplementedError("Deprecated!")

    @train_decorator()
    def fit(self) -> None:
        # Fit and plot rewards
        hist = self.m.fit(self.env, nb_steps=self.n_train_steps, visualize=False, verbose=1)
        train_plot = self.env.get_plt_path("test")
        plot_rewards(hist, train_plot)
        # dqn.test(env, nb_episodes=5, visualize=True)


class NAFBaseAgent(KerasBaseAgent):
    """This does not work!

    TODO: Fix this!
    """

    def __init__(self, env: FullRoomEnv):
        # Initialize super class
        name = "NAF"
        super().__init__(env=env, name=name)
        print("Why don't you work??????")

        # Build Q-function model.
        nb_actions = env.nb_actions
        n_state_vars = env.m.n_pred

        # V model
        inputs = Input(shape=(1, n_state_vars))
        flat_inputs = Flatten()(inputs)
        v_model = getMLPModel(out_dim=1)
        v_model = Model(inputs=inputs, outputs=v_model(flat_inputs))

        # Mu model
        inputs = Input(shape=(1, n_state_vars))
        flat_inputs = Flatten()(inputs)
        m_model = getMLPModel(out_dim=nb_actions)
        m_model = Model(inputs=inputs, outputs=m_model(flat_inputs))

        # L model
        n_out_l = (nb_actions * nb_actions + nb_actions) // 2
        action_input = Input(shape=(nb_actions,), name='action_input')
        state_inputs = Input(shape=(1, n_state_vars))
        flat_inputs = Flatten()(state_inputs)
        x = Concatenate()([action_input, flat_inputs])
        l_model = getMLPModel(out_dim=n_out_l)
        l_model = Model(inputs=[action_input, state_inputs], outputs=l_model(x))

        # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
        # even the metrics!
        memory = SequentialMemory(limit=100000, window_length=1)
        random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.3, size=nb_actions)
        self.m = NAFAgent(nb_actions=nb_actions, V_model=v_model, L_model=l_model, mu_model=m_model,
                          memory=memory, nb_steps_warmup=100, random_process=random_process,
                          gamma=.99, target_model_update=1e-3)
        self.m.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

        raise NotImplementedError("Deprecated!")

    def fit(self, verbose: int = 0, **kwargs) -> None:
        # Fit and plot rewards
        hist = self.m.fit(self.env, nb_steps=100000, visualize=False, verbose=1)
        train_plot = self.env.get_plt_path("test")
        plot_rewards(hist, train_plot)


class DDPGBaseAgent(KerasBaseAgent):
    """The wrapper of the keras-rl DDPG agent.

    Suited for continuous action and state space.
    Range of allowed actions can be specified.
    """

    def __init__(self, env: RLDynEnv,
                 n_steps: int = 50000,
                 lr: float = 0.001,
                 gamma: float = 0.9,
                 layers: Sequence[int] = (100, 100),
                 reg: float = 0.01,
                 action_range: RangeListT = None):
        """Constructor.

        Args:
            env: The underlying environment.
            n_steps: The number of steps to train.
            lr: The base learning rate.
            gamma: The discount factor.
            layers: The layer architecture of the MLP for the actor and the critic network.
            reg: The regularization factor for the networks.
            action_range: The range of the actions the actor can take.
        """
        # Find unique name based on parameters.
        param_ex_list = [("N", n_steps),
                         ("LR", lr),
                         ("GAM", gamma),
                         ("L", layers),
                         ("REG", reg),
                         ("AR", action_range)]
        # Set parameters
        self.n_steps = n_steps
        self.lr = lr
        self.gamma = gamma

        # Create name
        name = f"{self.get_short_name()}_{env.name}{make_param_ext(param_ex_list)}"

        # Initialize super class.
        super().__init__(env=env, name=name)

        # Save reference to env and extract relevant dimensions.
        self.env = env
        self.nb_actions = env.nb_actions
        self.n_state_vars = env.m.n_pred

        # Network parameters
        self.layers = layers
        self.reg = reg
        if action_range is not None:
            assert len(action_range) == env.nb_actions, "Wrong amount of ranges!"
        self.action_range = action_range

        # Build the model.
        self._build_agent_model()

        self.plot_name = "DDPG"

    def __str__(self) -> str:
        return f"DDPG Agent with layers {self.layers}."

    def _build_agent_model(self) -> None:
        """Builds the Keras model of the agent."""
        # Build actor model
        actor = Sequential()
        actor.add(Flatten(input_shape=(1, self.n_state_vars)))
        actor.add(getMLPModel(mlp_layers=self.layers,
                              out_dim=self.nb_actions,
                              ker_reg=self.reg))

        # Clip actions to desired interval
        if self.action_range is not None:
            actor.add(Activation('sigmoid'))
            # actor.add(get_constrain_layer(self.action_range))
            pass
            # actor.add(ConstrainOutput(self.action_range))

        # Build critic model
        action_input = Input(shape=(self.nb_actions,), name='action_input')
        observation_input = Input(shape=(1, self.n_state_vars), name='observation_input')
        flattened_observation = Flatten()(observation_input)
        x = Concatenate()([action_input, flattened_observation])
        x = getMLPModel(mlp_layers=self.layers, out_dim=1, ker_reg=self.reg)(x)
        critic = Model(inputs=[action_input, observation_input], outputs=x)

        # Configure and compile the agent.
        memory = SequentialMemory(limit=500000, window_length=1)
        random_process = OrnsteinUhlenbeckProcess(size=self.nb_actions, theta=.15, mu=0., sigma=.05)
        self.m = DDPGAgent(nb_actions=self.nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                           memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                           random_process=random_process, gamma=self.gamma, target_model_update=1e-3)
        opt = Adam(lr=self.lr, clipnorm=1.0)
        self.m.compile(opt, metrics=['mae'])

    def get_info(self) -> Dict:
        return {'action_scaled_01': self.action_range}

    def load_if_exists(self, m, name: str,
                       train_data: str = DEFAULT_EVAL_SET) -> bool:
        """Loads the keras model if it exists.

        Returns true if it could be loaded, else False.
        Overrides the function in `KerasBase`, but in this
        case there are two models to load.

        Args:
            m: Keras-rl agent model to be loaded.
            name: Name of model.
            train_data: Hyperparameter opt. evaluation set.

        Returns:
             True if model could be loaded else False.
        """
        full_path = self._save_path()
        # full_path = self.get_path(name, env=self.env,
        #                           hop_eval_set=train_data)
        path_actor = full_path[:-3] + "_actor.h5"
        path_critic = full_path[:-3] + "_critic.h5"

        if os.path.isfile(path_actor) and os.path.isfile(path_critic):
            m.load_weights(full_path)
            return True

        if self.env.dummy_use:
            raise ValueError(f"No trained model {full_path[:-3]} found!")
        return False

    def _save_path(self):
        return self.env.get_model_path(self.name + ".h5")

    def save_model(self, m, name: str, train_data: str = DEFAULT_EVAL_SET) -> None:
        """Saves a keras model.

        Needs to be overridden here since the keras-rl
        `DDPGAgent` class does not have a `save` method.

        Args:
            m: Keras-rl agent model.
            name: Name of the model.
            train_data: Hyperparameter opt. evaluation set.
        """
        w_path = self._save_path()
        m.save_weights(w_path)
        # m.save_weights(self.get_path(name, env=self.env,
        #                              hop_eval_set=train_data))

    @train_decorator()
    def fit(self, verbose: int = 1, train_data: str = DEFAULT_TRAIN_SET) -> None:
        """Fit the agent using the environment.

        Makes a plot of the rewards received during the training.
        """
        # Fit
        if verbose:
            print("Actually fitting...")

        self.env.use_noise = True
        hist = self.m.fit(self.env, nb_steps=self.n_steps,
                          visualize=False, verbose=min(verbose, 1), nb_max_episode_steps=200)

        # Check if fully trained
        n_steps_trained = hist.history['nb_steps'][-1]
        if n_steps_trained <= self.n_steps - self.env.n_ts_per_eps:
            if verbose:
                print(f"Training aborted after {n_steps_trained} steps, "
                      f"saving parameters anyway...")
            # Rename for parameter saving
            self.name = ddpg_agent_name(n_steps_trained)

        # Plot rewards
        train_plot = self.env.get_plt_path(self.name + "_train_rewards")
        plot_rewards(hist, train_plot)

    def get_short_name(self):
        return ddpg_agent_name(self.n_steps, self.lr, self.gamma)


def default_ddpg_agent(env: RLDynEnv,
                       n_steps: int = None,
                       fitted: bool = True,
                       verbose: int = 1,
                       hop_eval_set: str = DEFAULT_EVAL_SET,
                       lr: float = used_lr) -> DDPGBaseAgent:
    # Choose step number
    if n_steps is None:
        n_steps = get_rl_steps(eul=True)

    gam = used_gamma

    # Initialize agent
    with ProgWrap(f"Initializing DDPG agent...", verbose > 0):
        agent = DDPGBaseAgent(env,
                              action_range=env.action_range,
                              n_steps=n_steps,
                              gamma=gam,
                              lr=lr)
        agent.name = agent.get_short_name()
        # agent.name = ddpg_agent_name(n_steps, lr=lr, gam=gam)

    # Fit if requested
    if fitted:
        with ProgWrap(f"Fitting DDPG agent...", verbose > 0):
            agent.fit(verbose=prog_verb(verbose), train_data=hop_eval_set)

    return agent
