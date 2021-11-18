import os
import random
from unittest import TestCase

import mock
import numpy as np
from mock import call

import util.notify
from agents.agents_heuristic import ConstActionAgent
from agents.base_agent import RL_MODEL_DIR, remove_agents
from data_processing.dataset import dataset_data_path
from dynamics.base_hyperopt import hop_path
from dynamics.recurrent import RNN_TEST_DATA_NAME
from opcua_empa.opcua_util import experiment_data_path
from tests.test_data import SYNTH_DATA_NAME
from util.notify import send_mail
from util.numerics import has_duplicates, split_arr, move_inds_to_back, find_rows_with_nans, nan_array_equal, \
    extract_streak, cut_data, find_all_streaks, find_disjoint_streaks, prepare_supervised_control, npf32, align_ts, \
    num_nans, find_longest_streak, mse, mae, max_abs_err, check_shape, save_performance_extended, \
    get_metrics_eval_save_name_list, load_performance, MSE, find_inds, nan_avg_between, int_to_sin_cos, \
    find_sequence_inds, remove_nan_rows, contrary_indices
from util.util import rem_first, tot_size, scale_to_range, linear_oob_penalty, make_param_ext, CacheDecoratorFactory, \
    np_dt_to_str, str_to_np_dt, day_offset_ts, fix_seed, to_list, rem_files_and_dirs, split_desc_units, create_dir, \
    yeet, \
    dynamic_model_dir, param_dict_to_name, prog_verb, w_temp_str, floor_datetime_to_min, extract_args, fun_to_class, \
    skip_if_no_internet, ProgWrap, linear_oob
from util.visualize import PLOT_DIR, plot_reward_details, model_plot_path, rl_plot_path, plot_performance_table, \
    _trf_desc_units, plot_env_evaluation, plot_valve_opening, plot_hist

# Define and create directory for test files.
TEST_DIR = os.path.join(PLOT_DIR, "Test")  #: Directory for test output.
TEST_DATA_DIR = "./tests/data"
create_dir(TEST_DIR)


class TestNumerics(TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Define some index arrays
        self.ind_arr = np.array([1, 2, 3, 4, 2, 3, 0], dtype=np.int32)
        self.ind_arr_no_dup = np.array([1, 2, 4, 3, 0], dtype=np.int32)

        # Define data arrays
        self.data_array = np.array([
            [1.0, 1.0, 2.0],
            [2.0, 2.0, 5.0],
            [2.0, 2.0, 5.0],
            [3.0, -1.0, 2.0]])
        self.data_array_with_nans = np.array([
            [1.0, np.nan, 2.0],
            [2.0, 2.0, 5.0],
            [2.0, 2.0, 5.0],
            [2.0, 2.0, np.nan],
            [2.0, np.nan, np.nan],
            [2.0, 2.0, 5.0],
            [2.0, 2.0, 5.0],
            [3.0, -1.0, 2.0]])

        # Define bool vectors
        self.bool_vec = np.array([True, False, False, False, True, False, False, True])
        self.bool_vec_2 = np.array([True, False, False, False, False, True, False, False, False, False, True])
        self.bool_vec_3 = np.array([True, False, False, False])

        # Sequence data
        self.sequences = np.array([
            [[1, 2, 3],
             [1, 2, 3],
             [2, 3, 4]],
            [[3, 2, 3],
             [1, 2, 3],
             [4, 3, 4]],
        ])
        self.c_inds = np.array([1])

    def test_oob(self):
        a1 = np.array([0.0, 1.0, 2.0])
        b = [0.5, 1.0]
        exp = np.array([0.5, 0, 1.0])
        self.assertTrue(np.array_equal(exp, linear_oob(a1, b)))

    def test_mock_1(self):

        ret_val, rv2 = 5.0, 11.0

        def mp():
            assert print('hello') == ret_val
            return rv2

        with mock.patch('builtins.print') as mock_print:
            mock_print.return_value = ret_val
            assert mp() == rv2
            mock_print.assert_has_calls(
                [
                    call('hello')
                ]
            )

    def test_mock_2(self):

        def my_side_effect(arg1):
            if arg1 == "now":
                return "fuck"
            return "this"

        with mock.patch('numpy.datetime64') as mock_print:
            mock_print.side_effect = my_side_effect
            assert np.datetime64('now') == "fuck"
            assert np.datetime64('2019-12-31T00:37:29') == "this"

    def test_mock_3(self):
        orig_np_dt64 = np.datetime64

        def new_np_dt64(*args, **kwargs):
            if args[0] == "now":
                return "test"
            return orig_np_dt64(args, kwargs)

        np.datetime64 = new_np_dt64

        self.assertEqual(np.datetime64('now'), "test")

        np.datetime64 = orig_np_dt64

    def test_contrary_indices(self):
        inds = np.array([1, 3])
        c_inds = contrary_indices(inds, tot_len=5)
        exp_out = np.array([0, 2, 4])
        self.assertTrue(np.array_equal(c_inds, exp_out), "Wrong indices!")
        c_inds_2 = contrary_indices(np.array([], dtype=np.int32), tot_len=2)
        exp_out_2 = np.array([0, 1])
        self.assertTrue(np.array_equal(c_inds_2, exp_out_2), "Wrong indices!")

    def test_int_to_sin_cos(self):
        inds = np.array([0, 1, 2])
        tot_n = 4
        s, c = int_to_sin_cos(inds, tot_n)
        self.assertTrue(np.allclose(s, np.sin(2 * np.pi * inds / tot_n)))

    def test_nan_avg_bet(self):
        orig_np_dt64 = np.datetime64

        def new_np_dt64(*args, **kwargs):
            if args[0] == "now":
                return np.datetime64('2019-12-31T00:37:29')
            return orig_np_dt64(*args, **kwargs)

        ts = np.array([
            np.datetime64('2019-12-31T00:27:29'),
            np.datetime64('2019-12-31T00:29:29'),
            np.datetime64('2019-12-31T00:33:29'),
            np.datetime64('2019-12-31T00:35:29'),
            np.datetime64('2019-12-31T00:36:29'),
        ])
        vals = np.ones((len(ts), 2), dtype=np.float32)
        vals *= np.arange(len(ts)).reshape((len(ts), 1))

        try:
            np.datetime64 = new_np_dt64

            nan_avg = nan_avg_between(ts, vals, 7)
            exp_avf = np.array([3.0, 3.0])
            self.assertTrue(np.allclose(nan_avg, exp_avf))
        finally:
            np.datetime64 = orig_np_dt64

    def test_nan_removal(self):
        nan_ind_arr = np.array([1, np.nan, 3, 4, np.nan])
        other_arr = np.array([1, 3, 3, 4, 7])
        exp = np.array([1, 3, 4])
        r, [r2] = remove_nan_rows(nan_ind_arr, [other_arr])
        self.assertTrue(np.array_equal(exp, r))
        self.assertTrue(np.array_equal(exp, r2))

    def test_sequence_inds(self):
        arr = np.array([0, 0, 0, 1, 1, 0, 1])
        exp_inds = np.array([0, 3, 5, 6, 7])
        out = find_sequence_inds(arr)
        self.assertTrue(np.array_equal(exp_inds, out),
                        msg=f"Expected: {exp_inds}, got: {out}")

    def test_find_inds(self):
        # Define input
        in_i = np.array([1, 2, 4, 5])
        out_1 = np.array([2, 4, 5])
        out_2 = np.array([1, 4])

        # And output
        exp_out_1 = np.array([1, 2, 3])
        exp_out_2 = np.array([0, 2])
        act_out_1 = find_inds(in_i, out_1)
        act_out_2 = find_inds(in_i, out_2)
        self.assertTrue(np.array_equal(act_out_1, exp_out_1))
        self.assertTrue(np.array_equal(act_out_2, exp_out_2))

    def test_shape_check(self):
        self.assertEqual(check_shape(self.bool_vec, (-1,)), True)
        self.assertEqual(check_shape(self.sequences, (2, 3, 3)), True)
        with self.assertRaises(ValueError):
            check_shape(self.sequences, (2, 5, 3), "test")

    def test_has_duplicates(self):
        self.assertTrue(has_duplicates(self.ind_arr) and not has_duplicates(self.ind_arr_no_dup),
                        "Implementation of has_duplicates contains errors!")

    def test_array_splitting(self):
        # Test array splitting
        d1, d2, n = split_arr(self.data_array, 0.1)
        d1_exp = self.data_array[:3]
        self.assertTrue(np.array_equal(d1, d1_exp) and n == 3,
                        "split_arr not working correctly!!")

    def test_find_nans(self):
        # Test finding rows with nans
        nans_bool_arr = find_rows_with_nans(self.data_array_with_nans)
        nans_exp = np.array([True, False, False, True, True, False, False, False])
        self.assertTrue(np.array_equal(nans_exp, nans_bool_arr),
                        "find_rows_with_nans not working correctly!!")

    def test_streak_extract(self):
        # Test last streak extraction
        d1, d2, n = extract_streak(self.data_array_with_nans, 1, 1)
        d2_exp = self.data_array_with_nans[6:8]
        d1_exp = self.data_array_with_nans[:6]
        if not nan_array_equal(d2, d2_exp) or n != 7 or not nan_array_equal(d1, d1_exp):
            raise AssertionError("extract_streak not working correctly!!")

    def test_longest_sequence(self):
        # Test find_longest_streak
        ex_1 = (0, 1)
        ls_first = find_longest_streak(self.bool_vec, last=False)
        ls_last = find_longest_streak(self.bool_vec, last=True)
        self.assertEqual(ls_first, ex_1, "find_longest_streak incorrect!")
        self.assertEqual(ls_last, (7, 8), "find_longest_streak incorrect!")
        another_bool = np.array([0, 1, 1, 1, 0, 1, 0], dtype=np.bool)
        ls_last = find_longest_streak(another_bool, last=True)
        self.assertEqual(ls_last, (1, 4), "find_longest_streak incorrect!")
        one_only = np.array([1, 1, 1, 1])
        with self.assertRaises(ValueError):
            find_longest_streak(one_only, last=True, seq_val=0)

    def test_seq_cutting(self):
        # Test sequence cutting
        cut_dat_exp = np.array([
            self.data_array_with_nans[1:3],
            self.data_array_with_nans[5:7],
            self.data_array_with_nans[6:8],
        ])
        c_dat, inds = cut_data(self.data_array_with_nans, 2)
        inds_exp = np.array([1, 5, 6])
        if not np.array_equal(c_dat, cut_dat_exp) or not np.array_equal(inds_exp, inds):
            raise AssertionError("cut_data not working correctly!!")

    def test_streak_finding(self):
        streaks = find_all_streaks(self.bool_vec, 2)
        s_exp = np.array([1, 2, 5])
        if not np.array_equal(s_exp, streaks):
            raise AssertionError("find_all_streaks not working correctly!!")

    def test_disjoint_streak_finding(self):
        # Test find_disjoint_streaks
        s_exp = np.array([1, 2, 5])
        dis_s = find_disjoint_streaks(self.bool_vec, 2, 1)
        if not np.array_equal(dis_s, s_exp):
            raise AssertionError("find_disjoint_streaks not working correctly!!")
        dis_s = find_disjoint_streaks(self.bool_vec_2, 2, 2, 1)
        if not np.array_equal(dis_s, np.array([2, 6])):
            raise AssertionError("find_disjoint_streaks not working correctly!!")
        dis_s = find_disjoint_streaks(self.bool_vec_2, 2, 2, 0)
        if not np.array_equal(dis_s, np.array([1, 7])):
            raise AssertionError("find_disjoint_streaks not working correctly!!")
        dis_s = find_disjoint_streaks(self.bool_vec_3, 2, 2, 0)
        if not np.array_equal(dis_s, np.array([1])):
            raise AssertionError("find_disjoint_streaks not working correctly!!")

    def test_supervision_prep(self):
        # Test prepare_supervised_control
        in_arr_exp = np.array([
            [[1, 3, 2],
             [1, 3, 3]],
            [[3, 3, 2],
             [1, 3, 3]],
        ])
        out_arr_exp = np.array([
            [2, 4],
            [4, 4],
        ])
        in_arr, out_arr = prepare_supervised_control(self.sequences, self.c_inds, False)
        if not np.array_equal(in_arr, in_arr_exp) or not np.array_equal(out_arr, out_arr_exp):
            raise AssertionError("Problems encountered in prepare_supervised_control")

    def test_move_inds(self):
        # Test move_inds_to_back
        arr = np.arange(5)
        inds = [1, 3]
        exp_res = np.array([0, 2, 4, 1, 3])
        np.array_equal(move_inds_to_back(arr, inds), exp_res), "move_inds_to_back not working!"

    def test_fix_seed(self):
        # Tests the seed fixing.
        max_int = 1000
        fix_seed()
        a = random.randint(-max_int, max_int)
        arr = np.random.normal(0.0, 1.0, 10)
        fix_seed()
        self.assertEqual(a, random.randint(-max_int, max_int))
        self.assertTrue(np.array_equal(arr, np.random.normal(0.0, 1.0, 10)))

    def test_npf32(self):
        sh = (2, 3)
        arr = npf32(sh)
        self.assertEqual(arr.shape, sh)
        val = 3.0
        arr2 = npf32(sh, fill=val)
        self.assertTrue(np.all(arr2 == val))

    def test_align(self):
        # Test data
        t_i1 = '2019-01-01 00:00:00'
        t_i2 = '2019-01-01 00:30:00'
        dt = 15
        ts_1 = np.array([1, 2, 2, 2, 3, 3], dtype=np.float32)
        ts_2 = np.array([2, 3, 3], dtype=np.float32)
        msg = "align_ts not correct!!"

        # Do tests
        test1, t_init1 = align_ts(ts_1, ts_2, t_i1, t_i2, dt)
        exp1 = npf32((6, 2), fill=np.nan)
        exp1[:, 0] = ts_1
        exp1[2:5, 1] = ts_2
        self.assertEqual(t_init1, t_i1, msg)
        self.assertTrue(nan_array_equal(exp1, test1), msg=msg)

        sh_large = (8, 2)
        test2, _ = align_ts(ts_2, ts_1, t_i1, t_i2, dt)
        self.assertEqual(test2.shape, sh_large)

        test3, _ = align_ts(ts_1, ts_1, t_i1, t_i2, dt)
        self.assertEqual(test3.shape, sh_large)
        exp3 = npf32(sh_large, fill=np.nan)
        exp3[:6, 0] = ts_1
        exp3[2:, 1] = ts_1
        self.assertTrue(nan_array_equal(exp3, test3), msg=msg)

        test4, _ = align_ts(ts_1, ts_1, t_i2, t_i1, dt)
        self.assertEqual(test4.shape, sh_large)
        exp4 = npf32(sh_large, fill=np.nan)
        exp4[:6, 1] = ts_1
        exp4[2:, 0] = ts_1
        self.assertTrue(nan_array_equal(exp4, test4), msg=msg)

    def test_num_nans(self):
        sh = (2, 3)
        arr1 = npf32(sh, fill=np.nan)
        self.assertEqual(num_nans(arr1), tot_size(sh), "num_nans incorrect!")

    def test_file_name_generation(self):
        lst = ["test", "foo"]
        dt = 100
        name_list = get_metrics_eval_save_name_list(lst, dt)
        self.assertEqual(len(name_list), len(lst) + 1)

    def test_save_performance_extended(self):
        n, n_series = 4, 3
        dt_used = 30
        inds = range(n)
        met_list = ["met1", "met2"]
        p_list = ["first", "second"]
        n_f = len(p_list)
        n_metrics = len(met_list)
        sh = (n_f, n_series, n_metrics, n)
        tot_sz = tot_size(sh)
        np_arr = np.arange(tot_sz).reshape(sh)

        # Save data
        f_names = get_metrics_eval_save_name_list(p_list, dt_used)
        f_names = [os.path.join(TEST_DIR, i) for i in f_names]
        save_performance_extended(np_arr, inds, f_names, met_list)

        # Try loading again
        def path_gen(name):
            return os.path.join(TEST_DIR, name)

        perf_arrays, inds_loaded = load_performance(path_gen, p_list, dt_used, n_metrics)

        self.assertTrue(np.allclose(perf_arrays, np_arr))
        self.assertTrue(np.array_equal(inds, inds_loaded))

        class DatTest:
            descriptions = np.array([f"Series{i}" for i in range(n_series)])
            dt = dt_used
            d = n_series
            n_c = 0
            c_inds = np.array([], dtype=np.int32)
            scaling = 2 * np.ones((n_series, 2))

        # Test plotting
        class Mod:
            name = "TestMod"
            data = DatTest()

            @staticmethod
            def get_plt_path(name):
                return os.path.join(TEST_DIR, name)

        t_mod = Mod()

        plot_performance_table([t_mod], p_list, met_list)

    pass


class TestMetrics(TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Define some index arrays
        self.a1 = np.array([1, 2])
        self.a2 = np.array([1, 1])
        self.a3 = np.array([-1, -1])

    def test_mse(self):
        self.assertAlmostEqual(mse(self.a1, self.a2), 0.5, msg="mse incorrect")
        self.assertAlmostEqual(mse(self.a3, self.a2), 4.0, msg="mse incorrect")

    def test_max_abs_err(self):
        self.assertAlmostEqual(max_abs_err(self.a1, self.a2), 1.0, msg="max_abs_err incorrect")
        self.assertAlmostEqual(max_abs_err(self.a3, self.a2), 2.0, msg="mae incorrect")

    def test_mae(self):
        self.assertAlmostEqual(mae(self.a1, self.a2), 0.5, msg="mae incorrect")
        self.assertAlmostEqual(mae(self.a3, self.a2), 2.0, msg="mae incorrect")


class TestUtil(TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Define data
        self.dt1 = np.datetime64('2000-01-01T00:00', 'm')
        self.dt2 = np.datetime64('2000-01-01T01:15', 'm')
        self.dt3 = np.datetime64('2000-01-01T22:45', 'm')
        self.n_mins = 15
        self.t_init_str = np_dt_to_str(self.dt3)
        self.t_init_str_2 = np_dt_to_str(self.dt2)

    class Dummy:
        def __init__(self):
            # This would actually work, but I don't like it :(
            # self.mutable_fun = CacheDecoratorFactory()(self.mutable_fun)
            pass

        @CacheDecoratorFactory()
        def fun(self, n: int, k: int):
            return n + k * k

        @CacheDecoratorFactory()
        def mutable_fun(self, n: int, k: int):
            return [n, k]

    def test_cache_decorator(self):
        try:
            d = self.Dummy()
            assert d.fun(1, k=3) == 10
            assert d.fun(2, 3) == 11
            assert d.fun(1, k=4) == 10
            list_1_1 = d.mutable_fun(1, 1)
            assert d.mutable_fun(1, 2) == list_1_1
            list_1_1[0] = 0
            assert list_1_1 == d.mutable_fun(1, 5)
            # d2 = Dummy()
            # assert d2.mutable_fun(1, 2) == [1, 2]  # It fails here!
            assert d.fun(2, 7) == 11
            assert [4, 7] == d.mutable_fun(4, 7)
        except AssertionError as e:
            print("Cache Decorator Test failed!!")
            raise e
        except Exception as e:
            raise AssertionError(f"Some error happened: {e}")

    def test_fun_to_class(self):
        def f1():
            return

        def f2():
            return 0

        f_class_1 = fun_to_class(f1)
        f_class_2 = fun_to_class(f1)
        f_class_3 = fun_to_class(f2)
        self.assertEqual(f_class_1(), f_class_2())
        self.assertEqual(f_class_1.__class__.__name__, f_class_2.__class__.__name__)
        self.assertEqual(f_class_1.__class__.__name__, f_class_3.__class__.__name__)
        self.assertNotEqual(f_class_1.__class__, f_class_3.__class__)

    def test_floor_datetime_to_min(self):
        exp1 = np.datetime64('2000-01-01T01:00', 'm')
        out1 = floor_datetime_to_min(self.dt2, 60)
        out2 = floor_datetime_to_min(self.dt2, 120)
        self.assertEqual(exp1, out1)
        self.assertEqual(self.dt1, out2)

    def test_verbose_propagation(self):
        for i in [-5, -1, 0, 1, 5]:
            if i > 0:
                self.assertEqual(i, prog_verb(i) + 1)
            else:
                self.assertEqual(0, prog_verb(i))

    def test_w_temp_str(self):
        temps = [22.5, 22]
        res = w_temp_str(temps)
        self.assertEqual(res, "Heating: In / Out temp: 22.5 / 22 C")

    def test_extract_args(self):
        dum_args = [True]
        out = extract_args(dum_args, None)
        exp_out = [True]
        self.assertEqual(out[0], exp_out[0])

        use_bat_data, enf_opt = extract_args([False], True, None)
        self.assertEqual(use_bat_data, False)
        self.assertEqual(enf_opt, None)

    def test_yeet(self):
        self.assertRaises(ValueError, yeet)

    def test_dict_to_name(self):
        test_d = {"hello": 2, "star": "s"}
        exp = "_hello2_stars"
        out = param_dict_to_name(test_d)
        self.assertEqual(exp, out, f"Expected: {exp}, got: {out}")

    def test_to_list(self):
        self.assertEqual([1], to_list(1))
        self.assertEqual([1], to_list([1]))
        self.assertEqual(["1"], to_list("1"))

    def test_rem_first(self):
        # Test rem_first
        self.assertEqual(rem_first((1, 2, 3)), (2, 3),
                         "rem_first not working correctly!")
        self.assertEqual(rem_first((1, 2)), (2,),
                         "rem_first not working correctly!")

    def test_tot_size(self):
        # Test tot_size
        msg = "tot_size not working!"
        self.assertEqual(tot_size((1, 2, 3)), 6, msg)
        self.assertEqual(tot_size((0, 1)), 0, msg)
        self.assertEqual(tot_size(()), 0, msg)

    def test_scale_to_range(self):
        # Test scale_to_range
        assert np.allclose(scale_to_range(1.0, 2.0, [-1.0, 1.0]), 0.0), "scale_to_range not working correctly!"
        assert np.allclose(scale_to_range(1.0, 2.0, [0.0, 2.0]), 1.0), "scale_to_range not working correctly!"

    def test_lin_oob_penalty(self):
        # Test linear_oob_penalty
        assert np.allclose(linear_oob_penalty(1.0, [-1.0, 1.0]), 0.0), "linear_oob_penalty not working correctly!"
        assert np.allclose(linear_oob_penalty(5.0, [0.0, 2.0]), 3.0), "linear_oob_penalty not working correctly!"
        assert np.allclose(linear_oob_penalty(-5.0, [0.0, 2.0]), 5.0), "linear_oob_penalty not working correctly!"

    def test_make_param_ext(self):
        # Test make_param_ext
        res1 = make_param_ext([("a", 4), ("b", [1, 2])])
        assert res1 == "_a4_b1-2", f"make_param_ext not implemented correctly: {res1}"
        in2 = [("a", 4), ("b", None), ("c", False)]
        out2 = make_param_ext(in2)
        exp2 = "_a4"
        self.assertEqual(out2, exp2, msg=f"make_param_ext not correct, exp: {exp2}, got: {out2}!")
        res3 = make_param_ext([("a", 4.1111111), ("b", True)])
        assert res3 == "_a4.111_b", f"make_param_ext not implemented correctly: {res3}"

    def test_time_conversion(self):
        # Test time conversion
        assert str_to_np_dt(np_dt_to_str(self.dt1)) == self.dt1, "Time conversion not working"

    def test_day_offset_ts(self):
        # Test day_offset_ts
        n_ts = day_offset_ts(self.t_init_str, self.n_mins)
        n_ts_passed = day_offset_ts(self.t_init_str_2, self.n_mins, remaining=False)
        self.assertEqual(n_ts, 5, "Wrong number of remaining timesteps!")
        self.assertEqual(n_ts_passed, 5, "Wrong number of passed timesteps!")

    def test_file_and_dir_removal(self):
        id_str = "Test_Test_test_2519632984160348"

        # Create some files and dirs.
        f_name = id_str + ".txt"
        with open(f_name, "w") as f:
            f.write("Test")
        d_name1 = f"{id_str}_dir"
        d_name2 = f"Test_{d_name1}"
        os.mkdir(d_name1)
        os.mkdir(d_name2)

        # Test removal
        rem_files_and_dirs(".", pat=id_str)
        self.assertFalse(os.path.isfile(f_name))
        self.assertFalse(os.path.isdir(d_name1))
        self.assertTrue(os.path.isdir(d_name2))
        rem_files_and_dirs(".", pat=id_str, anywhere=True)
        self.assertFalse(os.path.isdir(d_name2))

    def test_desc_split(self):
        d1 = "desc [1]"
        p1, p2 = split_desc_units(d1)
        self.assertEqual(p1, "desc ")
        self.assertEqual(p2, "[1]")
        self.assertEqual("hoi", split_desc_units("hoi")[0])


class TestPlot(TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.test_plot_dir = os.path.join(PLOT_DIR, "Test")

        dates = [
            '2019-01-01 12:00:00',
            '2019-01-01 12:00:12',
            '2019-01-01 12:00:55',
        ]
        self.np_dt_vec = np.array([str_to_np_dt(d) for d in dates])

    def get_test_path(self, base_name: str):
        create_dir(self.test_plot_dir)
        return os.path.join(self.test_plot_dir, base_name)

    def test_valve_open_plot(self):
        valves = np.random.normal(0, 1, (3, 2))
        test_path = self.get_test_path(f"test_valve_plot")
        plot_valve_opening(self.np_dt_vec, valves, test_path)

    def test_valve_open_plot_with_ts(self):
        dates_2 = [
            '2019-01-01 12:00:05',
            '2019-01-01 12:00:32',
            '2019-01-01 12:01:15',
        ]
        ts_2 = np.array([str_to_np_dt(d) for d in dates_2])
        temp_sp = np.random.normal(2, 1, (3,))
        valves = np.random.normal(0, 1, (3, 2))
        test_path = self.get_test_path(f"test_valve_plot_extended")
        plot_valve_opening(self.np_dt_vec, valves, test_path,
                           ts_2, temp_sp)

    def make_bar_plot(self, n_ag, n_rew, n_steps):
        descs = [f"Rew_{i}" for i in range(n_rew - 1)]

        class Dummy:
            nb_actions = 1

        agents = [ConstActionAgent(Dummy(), 1.0 * i) for i in range(n_ag)]
        rewards = np.random.normal(0.0, 1.0, (n_ag, n_steps, n_rew))
        test_path = self.get_test_path(f"test_reward_bar_{n_ag}_{n_rew}_{n_steps}")
        lst = [a.get_short_name() for a in agents]

        # To have more representative cases!
        if len(lst) == 4 and n_rew == 3:
            lst = [
                "Valves Open",
                "Valves Closed",
                "Rule-Based",
                "DDPG",
            ]
            descs = ["room energy consumption[75.9 Wh]",
                     "temperature bound violation [Kh]"]
        elif len(lst) == 4 and n_rew == 5:
            lst = [
                "Open, Charge",
                "Closed, Discharge",
                "Rule-Based",
                "DDPG",
            ]
            descs = ["room energy consumption[75.9 Wh]",
                     "temperature bound violation [Kh]",
                     "battery energy consumption",
                     "total price"]

        plot_reward_details(lst, rewards, test_path, descs)

    def test_reward_bar_plot(self):
        self.make_bar_plot(4, 3, 5)
        self.make_bar_plot(4, 5, 10)
        self.make_bar_plot(1, 8, 3)

    def test_trf_unit_desc(self):
        init_d = "Test [5]"
        exp = "Test [(5)^2]"
        out = _trf_desc_units(init_d, MSE)
        self.assertEqual(exp, out)

    def test_hist(self):
        vals = np.array([np.nan, 0.3, 2.3, 1.2, np.nan, 0.0, 1.1, 5.1, 0.0, 0.0,
                         np.nan, 0.0, 0.2])
        pl_path = self.get_test_path(f"test_hist")
        plot_hist(vals, pl_path, tol=0.001)
        plot_hist(vals, pl_path + "_2", tol=0.001, bin_size=0.25)

    def test_plot_env_evaluation(self):

        # Variable parameters
        n_tot_series = 7
        n_agents = 4
        ep_len = 11

        # Fixed parameters (or at least unsave to change)
        n_c_series = 1
        n_weather = 2
        n_series = n_tot_series - n_c_series
        w_inds = np.arange(n_weather)
        s_mask = np.arange(n_series)
        s_mask = s_mask[np.logical_or(s_mask % 2, s_mask < n_weather)]
        n_reduced_series = len(s_mask)
        bounds = [(n_reduced_series - 1, (0.3, 0.6))]

        class MockDataset:
            dt = 30
            d = n_tot_series
            n_c = n_c_series
            c_inds = np.arange(n_c_series)
            descriptions = [f"Series {k}" for k in range(n_tot_series)]

        a_names = [f"Agent_{k}" for k in range(n_agents)]
        save_path = os.path.join(TEST_DIR, "TestPlotEnvEval")
        np_dt_init = str_to_np_dt('2019-01-01 12:00:00')

        # Define random data
        actions = np.random.rand(n_agents, ep_len, n_c_series)
        extra_actions = np.random.rand(n_agents, ep_len, n_c_series)
        states = np.random.rand(n_agents, ep_len, n_series)
        rewards = np.random.rand(n_agents, ep_len)
        for i in w_inds:
            for a in range(n_agents):
                states[a, :, i] = states[0, :, i]

        # Plot
        plot_env_evaluation(actions, states, rewards, MockDataset(), a_names, save_path,
                            extra_actions=extra_actions, title_ext="New Super Title", np_dt_init=np_dt_init)
        plot_env_evaluation(actions, states, rewards, MockDataset(), a_names, save_path + "_2",
                            series_mask=s_mask, bounds=[(-1, (0.4, 0.4))],
                            series_merging_list=[(w_inds, "Weather")], np_dt_init=np_dt_init)

        with self.assertRaises(AssertionError):
            plot_env_evaluation(actions, states, rewards, MockDataset(), a_names, save_path + "_2",
                                series_mask=s_mask, bounds=bounds,
                                series_merging_list=[(w_inds, "Weather"), ([s_mask[-1] - 1, 0], "Fail")])

    pass


class TestNotify(TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @skip_if_no_internet
    def test_send_mail(self):
        send_mail(subject="Running Tests", msg="Sali", debug=True)

    def test_load_login(self):
        test_login_file = os.path.join(TEST_DATA_DIR, "test_login.txt")
        user, pw = util.notify.login_from_file(test_login_file)
        self.assertEqual(pw, "test_pw")
        self.assertEqual(user, "test_user")

    def test_catching(self):
        with self.assertRaises(ValueError):
            with util.notify.FailureNotifier("test", verbose=0, debug=True):
                raise ValueError("Fuck")


class TestDataSharing(TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.local_test_file = os.path.join(TEST_DATA_DIR, "test_upload_file.txt")

    def test_data_upload(self):
        # This takes too long
        # upload_file(self.local_test_file)
        pass


def cleanup_test_data(verbose: int = 0):
    """Removes all test folders and files."""

    if verbose:
        print("Cleaning up some test files...")

    # Remove plots and graphs
    eval_tab_path = os.path.join(model_plot_path, "EvalTables")
    rem_files_and_dirs(model_plot_path, SYNTH_DATA_NAME)
    rem_files_and_dirs(model_plot_path, RNN_TEST_DATA_NAME)
    rem_files_and_dirs(eval_tab_path, "Test")
    rem_files_and_dirs(model_plot_path, "_basicRNN_", anywhere=True)

    # Cleanup models
    rem_files_and_dirs(dynamic_model_dir, RNN_TEST_DATA_NAME)
    rem_files_and_dirs(dynamic_model_dir, SYNTH_DATA_NAME)
    rem_files_and_dirs(dynamic_model_dir, "Test")
    rem_files_and_dirs(hop_path, "TestHop", anywhere=True)

    # Cleanup environments
    with ProgWrap(f"Cleaning up RL envs...", verbose > 0):
        for s in ["TestEnv", "BatteryTest", "FullTest"]:
            rem_files_and_dirs(rl_plot_path, s, anywhere=True)
            rem_files_and_dirs(RL_MODEL_DIR, s, anywhere=True)

        remove_agents()
        remove_agents()

    # Cleanup other test data
    rem_files_and_dirs(dataset_data_path, "Test", anywhere=True)
    rem_files_and_dirs(TEST_DIR, "")
    rem_files_and_dirs(experiment_data_path, "Offline", anywhere=True)
