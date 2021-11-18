from unittest import TestCase

import numpy as np

from agents import agents_heuristic
from agents.agents_heuristic import get_const_agents, RuleBasedAgent
from agents.keras_agents import DDPGBaseAgent, KerasBaseAgent
from dynamics.base_model import BaseDynamicsModel
from dynamics.composite import CompositeModel
from envs.dynamics_envs import RLDynEnv, BatteryEnv, RangeListT, RoomBatteryEnv, PWProfile, FullRoomEnv, heat_marker
from tests.test_data import construct_test_ds
from tests.test_dynamics import TestModel, ConstTestModelControlled, get_test_battery_model, get_full_composite_model
from util.numerics import rem_mean_and_std, add_mean_and_std
from util.util import Arr, DEFAULT_TRAIN_SET
from util.visualize import plot_heat_cool_rew_det


def get_keras_test_agent(env: RLDynEnv) -> DDPGBaseAgent:
    """Defines a keras DDPG agent for testing.

    Args:
        env: The env the agent will be using.

    Returns:
        The agent.
    """
    ac_range_list = env.action_range
    ag = DDPGBaseAgent(env, n_steps=5,
                       layers=(5,),
                       action_range=ac_range_list)
    return ag


class TestDynEnv(RLDynEnv):
    """The test environment."""

    def __init__(self, m: BaseDynamicsModel, max_eps: int = None):
        super(TestDynEnv, self).__init__(m, max_eps, cont_actions=True,
                                         n_cont_actions=1,
                                         name="TestEnv")
        d = m.data
        self.n_pred = 3
        assert d.n_c == 1 and d.d == 4, "Dataset needs 4 series of which one is controllable!!"

    reward_descs = ["Action Dependent [unit1]",
                    "Constant 1 [unit2]"]

    def detailed_reward(self, curr_pred: np.ndarray, action: Arr) -> np.ndarray:
        self._assert_pred_shape(curr_pred)
        return np.array([curr_pred[2] * action, 1.0])

    def episode_over(self, curr_pred: np.ndarray) -> bool:
        self._assert_pred_shape(curr_pred)
        return False

    def _assert_pred_shape(self, curr_pred):
        assert curr_pred.shape == (self.n_pred,), "Shape of prediction not correct!"


class TestEnvs(TestCase):
    """Tests the RL environments.

    TODO: Remove files after testing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Define dataset
        self.n = 201
        self.test_ds = construct_test_ds(self.n)
        self.test_ds.standardize()
        self.test_ds.split_data()
        self.test_ds.c_inds = np.array([1])
        sh = self.test_ds.data.shape
        self.test_mod = TestModel(self.test_ds)
        self.n_ts_per_episode = 10
        self.test_env = TestDynEnv(self.test_mod, 10)

        # Another one
        self.test_ds2 = construct_test_ds(self.n)
        self.test_ds2.c_inds = np.array([1])
        self.test_ds2.data = np.arange(sh[0]).reshape((-1, 1)) * np.ones(sh, dtype=np.float32)
        self.test_ds2.split_data()
        self.model_2 = ConstTestModelControlled(self.test_ds2)
        self.test_env2 = TestDynEnv(self.model_2, 10)

    def test_shapes(self):
        # Test shapes
        for k in range(30):
            next_state, r, over, _ = self.test_env.step(np.array([0.0]))
            assert next_state.shape == (3,), "Prediction does not have the right shape!"
            if (k + 1) % self.n_ts_per_episode == 0 and not over:
                raise AssertionError("Episode should be over!!")
            if over:
                init_state = self.test_env.reset()
                self.assertEqual(init_state.shape, (3,), "Prediction does not have the right shape!")

    def test_step(self):
        for action in [np.array([1.0]), np.array([0.0])]:
            init_state = self.test_env2.reset(0)
            next_state, rew, ep_over, _ = self.test_env2.step(action)
            self.assertTrue(np.allclose(init_state + action, next_state), "Step contains a bug!")
            for k in range(3):
                prev_state = np.copy(next_state)
                next_state, rew, ep_over, _ = self.test_env2.step(action)
                self.assertTrue(np.allclose(prev_state + action, next_state), "Step contains a bug!")

    def test_agent_analysis(self):
        # Test agent analysis
        const_ag_1 = agents_heuristic.ConstActionAgent(self.test_env, 0.0)
        const_ag_2 = agents_heuristic.ConstActionAgent(self.test_env, 1.0)
        self.test_env.analyze_agents_visually([const_ag_1, const_ag_2])

    def test_reset(self):
        # Test deterministic reset
        const_control = np.array([0.0], dtype=np.float32)
        max_ind = self.test_env.n_start_data
        rand_int = np.random.randint(max_ind)
        self.test_env.reset(start_ind=rand_int, use_noise=False)
        first_out = self.test_env.step(const_control)
        for k in range(5):
            self.test_env.step(const_control)
        self.test_env.reset(start_ind=rand_int, use_noise=False)
        sec_first_out = self.test_env.step(const_control)
        assert np.allclose(first_out[0], sec_first_out[0]), "State output not correct!"
        assert first_out[1] == sec_first_out[1], "Rewards not correct!"
        assert first_out[2] == sec_first_out[2], "Episode termination not correct!"

    def test_scaling(self):
        my_c = 4.5
        self.assertTrue(self.test_env2.scaling is None)
        self.assertTrue(self.test_env.scaling is not None)

        # Test _to_scaled
        cont_ac_2 = self.test_env2._to_scaled(my_c)
        self.assertTrue(np.array_equal(np.array([my_c]), cont_ac_2), "_to_scaled not correct!")
        cont_ac = self.test_env._to_scaled(my_c)
        c_ind = self.test_env.c_ind[0]
        exp_ac = rem_mean_and_std(np.array([my_c]), self.test_env.scaling[c_ind])
        self.assertTrue(np.array_equal(cont_ac, exp_ac), "_to_scaled not correct!")
        cont_ac2 = self.test_env._to_scaled(my_c, True)
        exp_ac2 = add_mean_and_std(np.array([my_c]), self.test_env.scaling[c_ind])
        self.assertTrue(np.array_equal(cont_ac2, exp_ac2), "_to_scaled not correct!")


class TestBatteryEnv(TestCase):
    """Tests the battery RL environment.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.m = get_test_battery_model(linear=False)
        self.env = BatteryEnv(self.m, max_eps=8)
        self.ag_1 = agents_heuristic.ConstActionAgent(self.env, 3.0)
        self.ag_2 = agents_heuristic.ConstActionAgent(self.env, -7.0)

    def test_analyze_agents(self):
        self.env.analyze_agents_visually([self.ag_1, self.ag_2], start_ind=1)

    def test_agent_eval(self):
        n = 20
        r, r_other, _, _, _, _ = self.ag_1.eval(n, detailed=True)
        self.assertTrue(np.allclose(-r, r_other[:, 0]), "agent.eval not correct")
        self.ag_1.eval(n, reset_seed=True, detailed=True, scale_states=True)

    def test_detailed_analysis(self):
        self.env.detailed_eval_agents([self.ag_1, self.ag_2], n_steps=10)

    def test_scaled(self):
        self.assertTrue(self.env.scaling is not None, "Data should be scaled!!")

    def test_scaling(self):
        var = 5.0
        self.assertNotEqual(var, self.env._to_scaled(var))

    def test_get_const_agent(self):
        c1, c2 = get_const_agents(self.env)
        r, exp_r = c2.rule, np.array([-8.0])
        self.assertTrue(np.allclose(r, exp_r), msg=f"Expected: {exp_r}, got: {r}")


class TestFullRoomEnv(TestCase):
    """Tests the room RL environment.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mod = get_full_composite_model(add_battery=False)
        self.env = FullRoomEnv(mod, max_eps=5)
        self.pred = np.arange(7)

    def test_basics(self):
        self.env.reset()
        self.env.step(0.0)
        self.env.step(np.array([0.0]))

    def test_get_r_temp(self):
        r_temp = self.env.get_r_temp(self.pred)
        r_temp_2 = self.env.get_unscaled(self.pred, 5)
        self.assertEqual(r_temp, r_temp_2, msg=f"Series scaling failed!")

    def test_rule_based(self):
        rba = RuleBasedAgent(self.env, rule=[0.0, 2.0])
        a = rba.get_action(self.pred)
        self.assertAlmostEqual(a, 1.0)

    def test_strict_rule_based(self):
        r = [0.0, 5.0]
        rba_s = RuleBasedAgent(self.env, rule=r, strict=True)
        rba = RuleBasedAgent(self.env, rule=r)
        a = rba.get_action(self.pred)
        a_s = rba_s.get_action(self.pred)
        self.assertAlmostEqual(a_s, 1.0, msg="Rule based agent should be cooling!")
        self.assertAlmostEqual(a, 0.0, msg="Rule based agent should not be cooling!")


class KerasDDPGTest(DDPGBaseAgent):
    name = "DDPG_Test"

    def __init__(self, env: RLDynEnv, action_range: RangeListT = None, action: float = 0.6):
        KerasBaseAgent.__init__(self, env=env, name=self.name)

        self.action = action
        self.nb_actions = len(action_range)
        if action_range is not None:
            assert len(action_range) == env.nb_actions, "Wrong amount of ranges!"
        self.action_range = action_range

    def fit(self, verbose: int = 0, train_data: str = DEFAULT_TRAIN_SET) -> None:
        pass

    def load_if_exists(self, m, name: str, **kwargs) -> bool:
        pass

    def get_short_name(self):
        return self.name

    def get_action(self, state):
        return self.action * np.ones((self.nb_actions,))

    pass


class TestKerasAgent(TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.m = get_test_battery_model(linear=False)
        self.env = BatteryEnv(self.m, max_eps=8)
        self.env.name = "Test_Keras_Env"

        # Agents
        action_range = [(-100, 100)]
        self.const_action: float = 0.55
        self.true_action = self.const_action * 200 - 100
        self.test_agent = KerasDDPGTest(self.env,
                                        action_range=action_range,
                                        action=self.const_action)
        self.ag_1 = agents_heuristic.ConstActionAgent(self.env, 3.0)

    def test_detail_eval(self):
        self.env.detailed_eval_agents([self.test_agent, self.ag_1], n_steps=10)

    def test_analyze_agents(self):
        self.env.analyze_agents_visually([self.test_agent, self.ag_1], start_ind=1)

    def test_action(self):
        ac = self.test_agent.get_action(None)
        self.assertAlmostEqual(ac, self.const_action)

    def test_step(self):
        self.env.set_agent(self.test_agent)
        ac = self.test_agent.get_action(None)
        scaled_action = self.env.scale_action_for_step(ac)
        exp = self.env._to_scaled(self.true_action)
        self.assertAlmostEqual(scaled_action.item(), exp.item(), places=5,
                               msg="scale_action_for_step is wrong!")

    pass


class TestFullEnv(TestCase):
    """Tests the room RL environment.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        mod = get_full_composite_model()
        assert isinstance(mod, CompositeModel), "No composite model!"
        p = PWProfile()
        self.full_env = RoomBatteryEnv(mod, p, max_eps=5)

        mod2 = get_full_composite_model(standardized=True, dt_h=6)
        mod2.name = "FullTestModel2"
        self.full_env2 = RoomBatteryEnv(mod2, p, max_eps=5)
        self.full_env2.name = "TestFullBatteryRoomModelTest2"
        self.full_env2.connect_inds = (1, 3)

    def test_get_scaled_init_state(self):
        self.full_env2.get_scaled_init_state(0, [2, 3])
        # self.full_env2.get_scaled_init_state(0, [2, 3, 4, 5])

    def test_ev_connect(self):

        self.full_env2.reset(0, use_noise=False)
        d_ind = self.full_env2.get_curr_day_n()
        self.assertEqual(d_ind, 1, "Starting index wrong!")

        connect_n = self.full_env2.n_remain_connect()
        self.assertEqual(connect_n, 0, "Connection time computed incorrectly!")
        self.full_env2.step(np.array([0.0, 0.0]))
        self.full_env2.step(np.array([0.0, 0.0]))
        connect_n = self.full_env2.n_remain_connect()
        self.assertEqual(connect_n, 2, "Connection time computed incorrectly!")

        self.full_env2.step(np.array([0.0, 0.0]))
        self.assertEqual(self.full_env2.get_curr_day_n(), 0, "Starting index wrong!")

    def test_n_ts(self):
        self.assertEqual(self.full_env2.n_ts_per_day, 4, "ts_per_day incorrect!")
        self.assertEqual(self.full_env2.t_init_n, 2, "t_init_n incorrect!")
        self.assertEqual(self.full_env.t_init_n, 1, "t_init_n incorrect!")
        self.assertEqual(self.full_env.n_ts_per_day, 2, "ts_per_day incorrect!")

    def test_reset_and_step(self):
        init_state = self.full_env.reset(0)
        room_action = 1.0
        battery_action = 2.0
        action = np.array([room_action, battery_action])
        next_state, rew, over, _ = self.full_env.step(action)
        self.assertEqual(len(init_state), len(next_state), "Incompatible shapes!")
        control_working = np.allclose(next_state[2:4], init_state[2:4] + room_action)
        self.assertTrue(control_working, "Control not working as expected!")
        battery_working = np.allclose(next_state[-1], init_state[-1] + 2 * battery_action)
        self.assertTrue(battery_working, "Battery part not working as expected!")
        weather_working = np.allclose(next_state[:2], init_state[:2])
        self.assertTrue(weather_working, "Weather part not working as expected!")

    def test_visual_agent_analysis(self):
        ag1 = agents_heuristic.ConstActionAgent(self.full_env, 10.0)
        ag2 = agents_heuristic.ConstActionAgent(self.full_env, -10.0)
        self.full_env.analyze_agents_visually([ag1, ag2], start_ind=0,
                                              use_noise=False, fitted=True,
                                              title_ext="Test Title")

    def test_visual_keras_agent_analysis(self):
        action_range = self.full_env.action_range
        test_agent = KerasDDPGTest(self.full_env,
                                   action_range=action_range,
                                   action=0.5)
        ag1 = agents_heuristic.ConstActionAgent(self.full_env, 10.0)
        self.full_env.analyze_agents_visually([ag1, test_agent], start_ind=0,
                                              use_noise=False, fitted=True)
        self.full_env.analyze_agents_visually([ag1, test_agent], start_ind=0,
                                              use_noise=False, fitted=True, plot_rewards=True,
                                              title_ext="NewSupTitle")

    def test_agents_eval(self):
        ag1 = agents_heuristic.ConstActionAgent(self.full_env, 10.0)
        ag2 = agents_heuristic.ConstActionAgent(self.full_env, -10.0)
        self.assertEqual(len(self.full_env.reward_descs), 4,
                         "Reward descriptions incorrect!")
        self.full_env.detailed_eval_agents([ag1, ag2], n_steps=3, use_noise=False)

    def test_agents_eval_2(self):
        ag1 = agents_heuristic.ConstActionAgent(self.full_env2, 10.0)
        ag2 = agents_heuristic.ConstActionAgent(self.full_env2, -10.0)
        self.full_env2.detailed_eval_agents([ag1, ag2], n_steps=20, use_noise=False,
                                            plt_fun=plot_heat_cool_rew_det,
                                            episode_marker=heat_marker)

    def test_ddpg_test_agent(self):
        self.full_env.short_name = "FullTestModel"
        ddpg_ag = get_keras_test_agent(self.full_env)
        ddpg_ag.name = "TestAgent_FullEnv"
        ddpg_ag.fit(verbose=0)

    def test_get_const_agent(self):
        c1, c2 = get_const_agents(self.full_env)
        self.assertTrue(np.allclose(c1.rule, np.array([0.0, -3.0])))

    def test_rule_based_agent(self):
        c_rate = 6.0
        rba = RuleBasedAgent(self.full_env, rule=[0.0, 10.0],
                             const_charge_rate=c_rate)
        state = np.arange(8)
        a = rba.get_action(state)
        exp_shape = (2,)
        self.assertEqual(a.shape, exp_shape,
                         msg=f"Incorrect action shape: {a.shape}, expected: {exp_shape}!")
        self.assertEqual(c_rate, a[1], f"Unexpected charging action: {a[1]}, expected: {c_rate}")
        self.assertEqual(0.0, a[0], f"Unexpected heating action: {a[0]}, expected: {0.0}")

    pass
