import datetime
from typing import List
from unittest import TestCase

import pandas as pd
import numpy as np

import opcua_empa.opcua_util
from dynamics.composite import CompositeModel
from envs.dynamics_envs import RoomBatteryEnv, PWProfile, FullRoomEnv
from opcua_empa.controller import FixTimeConstController, ValveToggler, RLController, \
    setpoint_toggle_frac, setpoint_from_fraction
from opcua_empa.opcua_util import NodeAndValues, read_experiment_data
from opcua_empa.opcuaclient_subscription import OpcuaClient, MAX_TEMP, MIN_TEMP
from opcua_empa.room_control_client import ControlClient, run_control
from tests.test_dynamics import get_full_composite_model
from tests.test_rl import KerasDDPGTest
from util.util import get_min_diff


class OfflineClient(OpcuaClient):
    """Test client that works offline and returns arbitrary values.

    One room only, with three valves!
    Will run until `read_values` is called `N_STEPS_MAX` times, then,
    the read temperature will be set to out of bounds and the
    experiment will terminate.
    """

    N_STEPS_MAX = 10

    _step_ind: int = 0

    subscribed: bool = False

    node_strs: List[str]
    n_read_vals: int

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self.assert_connected()
        pass

    def read_values(self) -> pd.DataFrame:
        assert self.subscribed, "No subscription!"
        self._step_ind += 1
        r_temp = "22.0" if self._step_ind < self.N_STEPS_MAX else "35.0"
        r_vals = ["1", r_temp, "28.0", "1"]
        valves = ["1", "1", "1"]
        exo = ["5.0", "0.0", "26.0", "26.0"]
        vals = r_vals + valves + exo
        return pd.DataFrame({'node': self.node_strs,
                             'value': vals})

    def publish(self, df_write: pd.DataFrame,
                log_time: bool = False,
                sleep_after: float = None) -> None:
        assert len(df_write) == 3, f"Only one room supported! (df_write = {df_write})"
        self.assert_connected()

    def subscribe(self, df_read: pd.DataFrame,
                  sleep_after: float = None) -> None:
        self.assert_connected()
        self.subscribed = True
        pd_sub_df: pd.DataFrame = df_read.sort_index()
        self.node_strs = [opcua_empa.opcua_util._trf_node(i) for i in pd_sub_df['node']]
        self.n_read_vals = len(self.node_strs)
        assert self.n_read_vals == 11, f"Wrong number of read nodes: {self.n_read_vals}"

    def assert_connected(self):
        assert self._connected, "Not connected!"

    pass


class TestOpcua(TestCase):
    """Tests the opcua client and related stuff.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c_val = 10.0
        self.cont = [(41, FixTimeConstController(val=self.c_val, max_n_minutes=1))]

    def test_string_manipulation(self):
        inp = "Hoi_Du"
        exp = "strHoi.strDu"
        res = opcua_empa.opcua_util._th_string_to_node_name(inp)
        self.assertEqual(res[-len(exp):], exp)

    def test_min_diff(self):
        d1 = datetime.datetime(2005, 7, 14, 13, 30)
        d2 = datetime.datetime(2005, 7, 14, 12, 30)
        min_diff = get_min_diff(d2, d1)
        self.assertAlmostEqual(min_diff, 60.0)

    def test_node_and_values(self):
        nav = NodeAndValues(self.cont)
        nodes = nav.get_nodes()
        self.assertEqual(len(self.cont) * 3, len(nodes))
        vals = nav.compute_current_values()
        self.assertEqual(vals[0], self.c_val)
        self.assertEqual(vals[1], True)

    def test_offline_client(self):
        nav = NodeAndValues(self.cont)
        read_nodes = nav.get_read_nodes()
        write_nodes = nav.get_nodes()
        df_read = pd.DataFrame({'node': read_nodes})
        df_write = pd.DataFrame({'node': write_nodes, 'value': None})
        with OfflineClient() as client:
            client.subscribe(df_read)
            client.publish(df_write)
            r_vals = client.read_values()
            nav.extract_values(r_vals)

    def test_valve_toggler(self):
        class OCToggle(OfflineClient):
            t_state = False
            op = ["1" for _ in range(3)]
            cl = ["0" for _ in range(3)]

            def read_values(self):
                self.t_state = not self.t_state
                vals = super().read_values()
                vals['value'][4:7] = self.op if self.t_state else self.cl
                return vals

            def publish(self, df_write: pd.DataFrame,
                        log_time: bool = False,
                        sleep_after: float = None) -> None:
                super().publish(df_write, log_time, sleep_after)
                temp_set = df_write['value'][0]

                assert (self.t_state and temp_set == MIN_TEMP) or \
                       (not self.t_state and temp_set == MAX_TEMP)

        vt = [(41, ValveToggler(n_steps_delay=0))]
        run_control(vt,
                    exp_name="OfflineValveTogglerTest",
                    verbose=0,
                    no_data_saving=True,
                    debug=True,
                    _client_class=OCToggle,
                    notify_failures=False)

    def test_control_client(self):
        with ControlClient(self.cont,
                           exp_name="OfflineTest",
                           verbose=0,
                           no_data_saving=True,
                           _client_class=OfflineClient,
                           notify_failures=False) as cc:
            cc.read_publish_wait_check()
        pass

    def test_run_control(self):
        run_control(self.cont,
                    exp_name="OfflineRunControlTest",
                    verbose=0,
                    no_data_saving=True,
                    debug=True,
                    _client_class=OfflineClient)

    def test_node_and_val_saving(self):
        n = 5
        nav = NodeAndValues(self.cont, n_max=n,
                            exp_name="OfflineNAVSavingTest")
        for k in range(n):
            nav.compute_current_values()
        self.assertEqual(nav._curr_read_n, 0)
        self.assertEqual(nav._curr_write_n, 0)
        nav.compute_current_values()
        nav.save_cached_data(verbose=0)

    def test_setpoint_toggle_time(self):
        dt = 15
        delay_close, delay_open = 5.0, 3.0
        res1, b1 = setpoint_toggle_frac(True, dt, 0.5, delay_open, delay_close)
        res2, b2 = setpoint_toggle_frac(False, dt, 0.5, delay_open, delay_close)
        res3, b3 = setpoint_toggle_frac(True, dt, 1.0, delay_open, delay_close)
        res4, b4 = setpoint_toggle_frac(False, dt, 0.01, delay_open, delay_close)
        res5, b5 = setpoint_toggle_frac(True, dt, 0.1, delay_open, delay_close)
        self.assertAlmostEqual(res1, 0.5 - 1 / 3)
        self.assertAlmostEqual(res2, 0.5 - 1 / 5)
        self.assertTrue(res3 >= 1.0)
        self.assertTrue(res4 >= 1.0)
        self.assertAlmostEqual(res5, 0.0)
        self.assertTrue(not b1 and not b4)

    def test_compute_curr_setpoint(self):
        dt = 15
        t1 = np.datetime64('2019-12-31T00:33:29')
        t_start = np.datetime64('2019-12-31T00:30:00')
        res1 = setpoint_from_fraction(0.5, True, False, dt,
                                      start_time=t_start, curr_time=t1)
        res2 = setpoint_from_fraction(0.5, False, True, dt,
                                      start_time=t_start, curr_time=t1)
        self.assertTrue(res1 and not res2)
        self.assertTrue(not res2)


class TestOpcuaRL(TestCase):
    """Tests the opcua client and related stuff.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c_val = 10.0
        self.cont = [(41, FixTimeConstController(val=self.c_val, max_n_minutes=1))]

        # Setup keras test agent
        mod = get_full_composite_model(add_battery=True, standardized=True)
        assert isinstance(mod, CompositeModel), "No composite model!"
        p = PWProfile()
        self.full_env = RoomBatteryEnv(mod, p, max_eps=5)
        action_range = self.full_env.action_range
        self.test_agent = KerasDDPGTest(self.full_env,
                                        action_range=action_range,
                                        action=0.5)
        self.rl_cont = [(41, RLController(self.test_agent, verbose=0))]

        room_mod = get_full_composite_model(add_battery=False, standardized=True)
        self.room_env = FullRoomEnv(room_mod, max_eps=5)
        self.test_agent_room = KerasDDPGTest(self.room_env,
                                             action_range=self.room_env.action_range,
                                             action=0.5)

        self.rl_cont_room = [(41, RLController(self.test_agent_room, verbose=0))]

    @staticmethod
    def get_test_scaling():
        s = np.empty((10, 2))
        s.fill(1.0)
        s[:, 0] = 2.0
        return s

    def test_rl_controller(self):
        with ControlClient(self.rl_cont,
                           exp_name="OfflineRLControllerTest",
                           verbose=0,
                           no_data_saving=True,
                           _client_class=OfflineClient,
                           notify_failures=False) as cc:
            cc.read_publish_wait_check()

    def test_rl_controller_room_only(self):
        with ControlClient(self.rl_cont_room,
                           exp_name="OfflineRoomRLControllerTest",
                           verbose=0,
                           no_data_saving=True,
                           _client_class=OfflineClient,
                           notify_failures=False) as cc:
            cc.read_publish_wait_check()

    def test_controller_scaling(self):
        cont = self.rl_cont[0][1]
        self.assertTrue(cont._scaling.shape == (10, 2), "Wrong shape!!")
        cont._scaling = self.get_test_scaling()

        rand_in = np.random.normal(0.0, 1.0, (10,))
        scaled_in = cont.scale_for_agent(rand_in)
        scaled_in_mean_added = cont.scale_for_agent(rand_in, remove_mean=False)
        self.assertTrue(np.allclose(rand_in, scaled_in + 2.0))
        self.assertTrue(np.allclose(rand_in, scaled_in_mean_added - 2.0))

    def test_time_adding(self):
        cont = self.rl_cont[0][1]
        cont._scaling = self.get_test_scaling()
        t_ind = 0
        state = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        time_state = cont.add_time_to_state(state, t_ind)
        self.assertTrue(np.allclose(time_state[-2:], np.array([0.0, 1.0])))
        self.assertEqual(len(time_state), 8)

    def test_rl_controller_call(self):
        with ControlClient(self.rl_cont_room,
                           exp_name="OfflineRLControllerCallTest",
                           verbose=0,
                           no_data_saving=True,
                           _client_class=OfflineClient,
                           notify_failures=False,
                           ) as cc:
            cont = self.rl_cont_room[0][1]
            cont._curr_ts_ind = 0
            cc.read_publish_wait_check()
            cont._curr_ts_ind = 5
            cc.read_publish_wait_check()

    def test_experiment_read_data(self):
        with ControlClient(self.rl_cont,
                           exp_name="OfflineReadDataTest",
                           verbose=0,
                           _client_class=OfflineClient) as cc:
            cc.node_gen.n_max = 6
            cont = True
            while cont:
                cont = cc.read_publish_wait_check()

            exp_name = cc.node_gen.experiment_name + "_PT_0"

        dat = read_experiment_data(exp_name, verbose=0)
        assert len(dat) == 4
        d1, d2, d3, d4 = dat
        assert len(d1) == len(d2)
        assert len(d3) == len(d4)

    pass
