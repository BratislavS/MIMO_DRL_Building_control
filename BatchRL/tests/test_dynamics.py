import os
from typing import Dict, Union, List
from unittest import TestCase

import numpy as np
from hyperopt import hp
from hyperopt.pyll import scope as ho_scope
from sklearn.linear_model import LinearRegression

from data_processing.dataset import Dataset
from dynamics.base_hyperopt import HyperOptimizableModel
from dynamics.base_model import BaseDynamicsModel, compare_models
from dynamics.battery_model import BatteryModel
from dynamics.classical import SKLearnModel
from dynamics.composite import CompositeModel
from dynamics.const import NoDisturbanceModel, ConstModel
from dynamics.recurrent import RNNDynamicModel, PhysicallyConsistentRNN
from dynamics.sin_cos_time import SCTimeModel
from tests.test_data import get_full_model_dataset, construct_test_ds
from tests.test_util import TEST_DIR
from util.numerics import copy_arr_list, MSE, MAE
from util.util import Num, tot_size, DEFAULT_EVAL_SET
from util.visualize import plot_performance_graph


def get_full_composite_model(standardized: bool = False,
                             dt_h: int = 12,
                             add_battery: bool = True) -> CompositeModel:
    # Get full dataset
    n = 100
    ds = get_full_model_dataset(n, dt=dt_h * 60, add_battery=add_battery)
    if standardized:
        ds.standardize()
    ds.split_data()

    # The weather model predicting constant weather.
    weather_mod_const = ConstTestModel(ds, pred_inds=np.array([0, 1]))

    # The room and heating model
    in_inds = np.array([2, 3, 4])
    out_inds = np.array([2, 3, 5])
    varying_mod = ConstTestModelControlled(ds, in_inds=in_inds,
                                           pred_inds=out_inds)

    # The exact time model
    time_mod = SCTimeModel(ds, time_ind=6)

    # Create full model and return
    mod_list = [weather_mod_const, varying_mod, time_mod]

    # Add battery model
    if add_battery:
        # Define and fit battery model
        bat_mod = BatteryModel(ds, base_ind=8)
        bat_mod.params = np.array([0, 1, 1])
        mod_list += [bat_mod]

    # Define full composite model
    full_mod = CompositeModel(ds, mod_list)
    full_mod.name = "FullTestModel" + "" if add_battery else "NoBattery"
    return full_mod


def test_testing():
    print("hey, test me, too!!")


class TestModel(BaseDynamicsModel):
    """Dummy dynamics model class for testing.

    Does not fit anything. Works only with datasets
    that have exactly 3 series to predict and one
    control variable series.
    """
    n_prediction: int = 0
    n_pred: int = 3

    def __init__(self,
                 ds: Dataset):
        name = "TestModel"
        super(TestModel, self).__init__(ds, name)
        self.use_AR = False

        if ds.n_c != 1:
            raise ValueError("Only one control variable allowed!!")
        if ds.d - 1 != self.n_pred:
            raise ValueError("Dataset needs exactly 3 non-controllable state variables!")

    def fit(self, verbose: int = 0, train_data: str = "train") -> None:
        self.fit_data = train_data
        if verbose:
            print(f"Not fitting anything, data part: '{train_data}'!")

    def predict(self, in_data: np.ndarray) -> np.ndarray:

        rel_out_dat = -0.9 * in_data[:, -1, :self.n_pred] + in_data[:, -1, -1].reshape(-1, 1)

        self.n_prediction += 1
        return rel_out_dat


class ConstSeriesTestModel(NoDisturbanceModel):
    """Test model that predicts a possibly different constant value for each series.

    Can take any number of input series and output series.
    """

    def __init__(self,
                 ds: Dataset,
                 pred_val_list: Union[Num, List[Num]],
                 predict_input_check: np.ndarray = None,
                 **kwargs):
        """Constructor

        If `pred_val_list` is a number, it will be used as prediction
        for all series that are predicted.

        Args:
            ds: The dataset.
            pred_val_list: The list with the values that will be predicted.
            predict_input_check: If not None, the input series have to have these values.
            **kwargs: Kwargs for the base class, e.g. `out_inds` or `in_inds`.
        """
        name = f"ConstModel"
        super().__init__(ds, name, **kwargs)

        # Check if the list contains the right number of elements if it is a list.
        if isinstance(pred_val_list, List):
            if not len(pred_val_list) == self.n_pred:
                raise ValueError("Not the right number of values specified!")

        self.predict_input_check = predict_input_check
        self.values = np.array(pred_val_list)
        self.use_AR = False

    def fit(self, verbose: int = 0, train_data: str = "train") -> None:
        self.fit_data = train_data
        if verbose:
            print(f"Not fitting anything, data part: '{train_data}'!")

    def predict(self, in_data: np.ndarray) -> np.ndarray:
        """Predicts a constant value for each series.

        Raises:
            AssertionError: If `predict_input_check` is not None and
                the input does not match.
        """
        in_sh = in_data.shape
        if self.predict_input_check is not None:
            pc = self.predict_input_check
            assert len(pc) == in_sh[-1], "Predictions do not have the right shape!"
            exp_inp = np.ones(in_sh) * pc
            assert np.array_equal(exp_inp, in_data), "Input incorrect!!!"
        preds = np.ones((in_sh[0], self.n_pred), dtype=np.float32) * self.values
        return preds


class ConstTestModel(ConstModel, NoDisturbanceModel):
    """Const Test model without a disturbance.

    Same as `ConstModel`, but no disturbance.
    """

    name: str = "NaiveNoDisturb"  #: Base name of model.


class ConstTestModelControlled(ConstTestModel):
    """Same as `ConstTestModel` but adds the control input to predictions."""

    def predict(self, in_data: np.ndarray) -> np.ndarray:
        action = in_data[:, -1, -1]
        return in_data[:, -1, :self.n_pred] + action


class TestBaseDynamics(TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Define dataset
        self.n = 111
        self.ds = construct_test_ds(self.n)
        self.ds_1 = construct_test_ds(self.n, 1)

        # Define models
        from dynamics.const import ConstModel
        self.test_mod = TestModel(self.ds)
        self.test_mod.fit()
        self.test_model_2 = ConstSeriesTestModel(self.ds_1,
                                                 pred_val_list=[0.0, 2.0],
                                                 out_inds=np.array([0, 2], dtype=np.int32),
                                                 in_inds=np.array([1, 2, 3], dtype=np.int32))
        self.test_model_2.fit()
        self.test_model_3 = ConstModel(self.ds_1)
        self.test_model_3.fit()

        # Compute sizes
        self.n_val = self.n - int((1.0 - self.ds.val_percent) * self.n)
        self.n_train = self.n - 2 * self.n_val
        self.n_train_seqs = self.n_train - self.ds.seq_len + 1
        self.n_streak = 7 * 24 * 60 // self.ds.dt
        self.n_streak_offset = self.n_train_seqs - self.n_streak

        self.dat_in_train, self.dat_out_train, _ = self.ds.get_split("train")

    def test_compare_analysis(self):
        test_model_4 = ConstSeriesTestModel(self.ds_1, pred_val_list=[0.0, 2.0, 3.0])
        save_name = os.path.join(TEST_DIR, "Test_Compare_Analysis")
        compare_models([test_model_4, self.test_model_3], n_steps=(2,),
                       model_names=["test_1", "test_2"], save_name=save_name)

    def test_compare_analysis_continuous(self):
        test_model_4 = ConstSeriesTestModel(self.ds_1, pred_val_list=[0.0, 2.0, 3.0])
        save_name = os.path.join(TEST_DIR, "Test_Compare_Analysis_Continuous")
        compare_models([test_model_4, self.test_model_3], n_steps=(0,),
                       model_names=["test_1", "test_2"], save_name=save_name)

    def test_sizes(self):
        # Check sizes
        self.assertEqual(len(self.dat_in_train), self.n_train_seqs, "Something baaaaad!")
        dat_in, dat_out, n_str = self.ds.get_streak("train")
        self.assertEqual(self.n_streak_offset, n_str, "Streak offset is wrong!!")

    def test_indices(self):
        # Check indices
        self.assertTrue(np.array_equal(np.array([0, 2, 3]), self.test_model_3.out_inds),
                        "Out indices incorrect!")
        self.assertTrue(np.array_equal(np.array([0, 1, 2]), self.test_model_3.p_out_inds),
                        "Prepared out indices incorrect!")
        self.assertTrue(np.array_equal(np.array([0, 1, 2, 3]), self.test_model_3.in_inds),
                        "In indices incorrect!")
        self.assertTrue(np.array_equal(np.array([0, 3, 1, 2]), self.test_model_3.p_in_indices),
                        "Prepared in indices incorrect!")

    def test_combined_plot(self):
        self.test_model_2.analyze_visually(plot_acf=False, n_steps=(3,), verbose=False,
                                           base_name="Test3Steps")

    def test_plot_with_errors(self):
        self.test_model_2.analyze_visually(plot_acf=False, n_steps=(3,), verbose=False, one_file=True, add_errors=True,
                                           base_name="TestWithErrors")

    def test_analysis(self):
        # Test model analysis
        base_data = np.copy(self.ds_1.data)
        streak = copy_arr_list(self.ds_1.get_streak("train"))
        assert self.test_model_2.fit_data is not None, "Fuuuuck"
        self.test_model_2.analyze_visually(plot_acf=False, n_steps=(2,),
                                           verbose=False, add_errors=True)
        streak_after = self.ds_1.get_streak("train")
        self.assertTrue(np.array_equal(base_data, self.ds_1.data),
                        "Data was changed during analysis!")
        for k in range(3):
            self.assertTrue(np.array_equal(streak_after[k], streak[k]),
                            "Streak data was changed during analysis!")

    def test_analyze_performance(self):
        met_list = (MSE, MAE)
        self.test_model_2.analyze_performance(n_steps=(1, 3), overwrite=True,
                                              verbose=0, metrics=met_list, n_days=7)

    def test_performance_plot(self):
        met_list = (MSE, MAE)
        parts = ["Val", "Train"]
        self.test_mod.analyze_performance(n_steps=(1, 3), overwrite=True,
                                          verbose=0, metrics=met_list, n_days=7)
        test_model_4 = ConstSeriesTestModel(self.ds, pred_val_list=[0.0, 2.0, 3.0])
        test_model_4.fit(verbose=0)
        test_model_4.analyze_performance(n_steps=(1, 3), overwrite=True,
                                         verbose=0, metrics=met_list, n_days=7)
        plot_name = "TestEvalPlot"
        mods = [test_model_4, self.test_mod]
        plot_performance_graph(mods, parts, met_list, plot_name,
                               short_mod_names=["Test4", "Test2"],
                               series_mask=np.array([0]),
                               scale_back=True,
                               remove_units=False,
                               put_on_ol=False)
        plot_performance_graph(mods, parts, met_list, plot_name,
                               short_mod_names=["Test4", "Test2"],
                               series_mask=np.array([0]),
                               scale_back=True,
                               remove_units=False,
                               put_on_ol=False,
                               compare_models=True)

    def test_const_series_test_model(self):
        # Test the test model
        n_feat = self.ds_1.d - self.ds_1.n_c
        preds = list(range(n_feat))
        c_mod = ConstSeriesTestModel(self.ds_1, preds)
        exp_out = np.copy(preds).reshape((1, -1))
        self.assertTrue(np.array_equal(exp_out, c_mod.predict(exp_out)),
                        "ConstSeriesTestModel implemented incorrectly!")

    def test_scaling(self):
        ds = self.ds
        ds_scaled = construct_test_ds(self.n)
        ds_scaled.standardize()
        test_mod_scaled = TestModel(ds_scaled)
        unscaled_out = test_mod_scaled.rescale_output(ds_scaled.data[:, :3], out_put=True,
                                                      whiten=False)
        unscaled_in = test_mod_scaled.rescale_output(ds_scaled.data, out_put=False,
                                                     whiten=False)
        scaled_out = test_mod_scaled.rescale_output(ds.data[:, :3], out_put=True,
                                                    whiten=True)
        assert np.allclose(unscaled_out, ds.data[:, :3]), "Rescaling not working!"
        assert np.allclose(unscaled_in, ds.data), "Rescaling not working!"
        assert np.allclose(scaled_out, ds_scaled.data[:, :3]), "Rescaling not working!"

    def test_pred_and_residuals(self):
        ds = construct_test_ds(self.n)
        dat_in_train, dat_out_train, _ = ds.get_split("train")
        test_mod = TestModel(ds)
        sample_pred_inp = ds.data[:(ds.seq_len - 1)]
        sample_pred_inp = sample_pred_inp.reshape((1, sample_pred_inp.shape[0], -1))
        test_mod.predict(sample_pred_inp)
        mod_out_test = test_mod.predict(dat_in_train)
        res = test_mod.get_residuals("train")
        if not np.allclose(mod_out_test - dat_out_train, res):
            raise AssertionError("Residual computation wrong!!")

    def test_n_step_predict(self):
        in_d, out_d, _ = self.ds.get_split('val')
        self.test_model_2.n_step_predict((in_d, out_d), 4)
        # TODO: Check return value!


class TestHopTable(HyperOptimizableModel):
    """Example hyperopt class that does not need fitting."""
    name: str = "TestHop"

    def __init__(self, ds: Dataset, base_param: int = 5, h_param_1: int = 0):
        super().__init__(ds, self.name)
        self.bp = base_param
        self.h_param = h_param_1

        self.base_name = self.get_base_name(base_param=base_param)

    def get_space(self) -> Dict:
        hp_space = {
            'h_param_1': ho_scope.int(hp.quniform('n_layers', low=0, high=20, q=1)),
        }
        return hp_space

    @classmethod
    def get_base_name(cls, **kwargs):
        return cls.name + "_" + str(kwargs['base_param'])

    def conf_model(self, hp_sample: Dict) -> 'HyperOptimizableModel':
        new_mod = TestHopTable(self.data, self.bp, **hp_sample)
        return new_mod

    def hyper_objective(self, eval_data: str = DEFAULT_EVAL_SET) -> float:
        x = self.h_param
        return -x * x + 12 * x + 15

    def fit(self, verbose: int = 0, train_data: str = "train") -> None:
        self.fit_data = train_data
        if verbose:
            print(f"Not fitting anything, data part: '{train_data}'!")

    def predict(self, in_data):
        pass


class TestHop(TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Define dataset
        self.ds = construct_test_ds(20)
        self.n = 2

    def test_save_and_load(self):
        # Init model
        test_hop_mod_4 = TestHopTable(self.ds, 4, 6)
        test_hop_mod_4.optimize(self.n, verbose=0)
        assert len(test_hop_mod_4.param_list) == self.n, "Parameter logging incorrect!"
        best_mod_4 = TestHopTable.from_best_hp(ds=self.ds, base_param=4)
        best_mod_4.optimize(self.n, verbose=0)


class TestClassical(TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Define datasets
        self.n = 200
        self.ds_1 = construct_test_ds(self.n, c_series=1)
        self.ds_1.data = np.ones((self.n, 4)) * np.arange(4)
        self.ds_1.split_data()

        self.model_name = "TestClassical"

    class DummyEst:
        """Dummy estimator for testing.

        More or less compliant with the sklearn API.
        Predicts what it gets as input.
        """
        x_pred = None
        x_fit = None
        y_fit = None
        n_out_feats: int = -1
        params: Dict = {"par": 10}

        def fit(self, x, y):
            y_sh = y.shape
            x_sh = x.shape
            self.n_out_feats = 1 if len(y_sh) == 1 else y_sh[-1]
            assert y_sh[0] == x_sh[0], "Incompatible shapes!"
            self.x_fit = x
            self.y_fit = y

        def predict(self, x):
            self.x_pred = x
            x_sh, x_train_sh = x.shape, self.x_fit.shape
            assert x_sh[1] == x_train_sh[1], f"Shape of x: {x_sh}, shape of train x: {x_train_sh}"
            return x[..., :self.n_out_feats]

        def get_params(self):
            return self.params

    def test_skl_dummy_mod(self):
        est = self.DummyEst()
        in_inds = np.array([0, 3])
        out_inds = np.array([3])
        skl_mod_res = SKLearnModel(self.ds_1, est, in_inds=in_inds, out_inds=out_inds)
        skl_mod_res.fit()
        self.assertTrue(np.array_equal(skl_mod_res.p_out_in_indices, np.array([1])))

        s_len = self.ds_1.seq_len - 1
        pred_ip = np.ones((1, s_len, 1)) * np.array([1, 2]).reshape((1, 1, -1))
        p = skl_mod_res.predict(pred_ip)
        self.assertTrue(np.array_equal(p, np.array([[3]])))

    def test_skl(self):
        skl_mod = LinearRegression()
        skl_mod_res = SKLearnModel(self.ds_1, skl_mod)
        exp_inds = np.array([0, 2, 3])
        msg = f"Indices: {skl_mod_res.p_out_in_indices} incorrect! (!= {exp_inds})"
        self.assertTrue(np.array_equal(skl_mod_res.p_out_in_indices, exp_inds), msg)
        skl_mod_res.fit()
        s_len = self.ds_1.seq_len - 1
        pred_ip = np.ones((1, s_len, 1)) * np.array([0, 1, 2, 3]).reshape((1, 1, -1))
        skl_mod_res.predict(pred_ip)

    def test_skl_save_and_load(self):
        # Test Keras model saving compatibility
        skl_mod = LinearRegression()
        skl_mod_res = SKLearnModel(self.ds_1, skl_mod)
        test_name = "TestSKLearnModel"
        skl_mod_res.save_model(skl_mod_res.m, test_name)
        skl_mod_res.load_if_exists(skl_mod_res.m, test_name)

    def test_loading(self):
        self.test_skl()
        self.test_skl_dummy_mod()


class TestComposite(TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Define datasets
        self.n = 200
        self.dataset = construct_test_ds(self.n)
        self.dataset.data = np.ones((self.n, 4)) * np.arange(4)
        self.dataset.split_data()
        self.ds_1 = construct_test_ds(self.n, c_series=1)
        self.ds_1.data = np.ones((self.n, 4)) * np.arange(4)
        self.ds_1.split_data()

        # Define indices
        self.inds_1 = np.array([1], dtype=np.int32)
        self.inds_2 = np.array([2], dtype=np.int32)
        self.inds_02 = np.array([0, 2], dtype=np.int32)
        self.inds_123 = np.array([1, 2, 3], dtype=np.int32)

        # Define output data
        self.sh = (2, 5, 4)
        self.sh_out = (2, 3)
        self.sh_tot = tot_size(self.sh)
        self.in_data = np.reshape(np.arange(self.sh_tot), self.sh)
        self.in_data_2 = np.ones(self.sh, dtype=np.float32) * np.arange(4)

    def test_first(self):
        m1 = ConstSeriesTestModel(self.dataset, pred_val_list=1.0,
                                  out_inds=self.inds_02)
        m2 = ConstSeriesTestModel(self.dataset, pred_val_list=2.0, out_inds=self.inds_1)
        mc1 = CompositeModel(self.dataset, [m1, m2], new_name="CompositeTest1")
        exp_out_data = np.ones((2, 3), dtype=np.float32)
        exp_out_data[:, 1] = 2.0
        assert np.array_equal(exp_out_data, mc1.predict(self.in_data)), "Composite model prediction wrong!"

    def test_second(self):
        m3 = ConstSeriesTestModel(self.dataset,
                                  pred_val_list=[1.0],
                                  out_inds=self.inds_1,
                                  in_inds=self.inds_1)
        m4 = ConstSeriesTestModel(self.dataset,
                                  pred_val_list=[0.0, 2.0],
                                  out_inds=self.inds_02,
                                  in_inds=self.inds_123)
        mc2 = CompositeModel(self.dataset, [m3, m4], new_name="CompositeTest2")

        exp_out_2 = np.ones((2, 3), dtype=np.float32) * np.arange(3.0)
        assert np.array_equal(exp_out_2, mc2.predict(self.in_data)), "Composite model prediction wrong!"
        assert np.array_equal(exp_out_2, mc2.predict(self.in_data_2)), "Composite model prediction wrong!"

    def test_third(self):
        m5 = ConstTestModel(self.dataset, pred_inds=self.inds_1)
        m6 = ConstTestModel(self.dataset,
                            pred_inds=self.inds_02,
                            in_inds=self.inds_123)
        mc3 = CompositeModel(self.dataset, [m5, m6], new_name="CompositeTest3")

        exp_out_3 = np.ones((2, 3), dtype=np.float32) * np.arange(3.0)
        exp_out_3[:, 0] = 1.0
        assert np.array_equal(exp_out_3, mc3.predict(self.in_data_2)), "Composite model prediction wrong!"

    def test_fourth(self):
        m7 = ConstTestModel(self.ds_1,
                            pred_inds=self.inds_02,
                            in_inds=np.array([1, 2], dtype=np.int32))
        m8 = ConstTestModel(self.ds_1,
                            pred_inds=np.array([3], dtype=np.int32),
                            in_inds=self.inds_1)
        mc4 = CompositeModel(self.ds_1, [m7, m8], new_name="CompositeTest4")

        exp_out_4 = np.ones(self.sh_out) * np.array([3, 1, 3])
        act_out = mc4.predict(self.in_data_2)
        assert np.array_equal(exp_out_4, act_out), "Composite model prediction wrong!"

    def test_analyze(self):
        # Test analyze_visually()
        m9 = ConstSeriesTestModel(self.ds_1,
                                  pred_val_list=[2.0],
                                  predict_input_check=self.inds_2,
                                  out_inds=self.inds_2,
                                  in_inds=self.inds_2)
        m10 = ConstSeriesTestModel(self.ds_1,
                                   pred_val_list=[0.0, 3.0],
                                   predict_input_check=self.inds_123,
                                   out_inds=np.array([0, 3], dtype=np.int32),
                                   in_inds=self.inds_123)
        mc5 = CompositeModel(self.ds_1, [m9, m10], new_name="CompositeTest4")
        mc5.fit(verbose=0)
        mc5.analyze_visually(plot_acf=False, verbose=False, n_steps=(2,))

    def test_full_composite(self):
        # Define model
        full_comp = get_full_composite_model()

        # Predict
        s = full_comp.data.seq_len - 1
        init_state = 0.5 * np.ones((1, s, 10), dtype=np.float32)
        out = full_comp.predict(init_state)

        # Check prediction
        const_weather = np.allclose(out[0, :2], 0.5)
        self.assertTrue(const_weather, "Weather part not constant")
        increase_room = np.allclose(out[0, 2:5], 1.0)
        self.assertTrue(increase_room, "Room part not correct!")


test_kwargs = {'hidden_sizes': (9,),
               'n_iter_max': 2,
               'input_noise_std': 0.001,
               'lr': 0.01}


class TestRNN(TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        n, n_feat = 200, 7
        self.ds = get_full_model_dataset(n)
        self.p_inds = np.array([0, 1, 3], dtype=np.int32)
        self.fix_kwargs = self.get_kwargs(self.ds)
        self.mod_test = RNNDynamicModel(name="TestRNN", **test_kwargs, **self.fix_kwargs)

    def get_kwargs(self, ds):
        return {'data': ds,
                'residual_learning': True,
                'weight_vec': None,
                'out_inds': self.p_inds,
                'constraint_list': None}

    def test_fit_2(self):
        ds2 = Dataset.copy(self.ds)
        ds2.name = ds2.name + "_2020-01-21"
        kws = self.get_kwargs(ds2)
        mod2 = RNNDynamicModel(name="TestRNN", **test_kwargs, **kws)
        mod2.fit()
        # print(self.mod_test.name)
        # print(mod2.name)
        assert mod2.name != self.mod_test.name, f"Names matching: {mod2.name}"

    def test_fit(self):
        self.mod_test.fit()

    def test_hyperopt(self):
        self.mod_test.optimize(1, verbose=0)

    pass


class TestPCRNN(TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ds = get_full_model_dataset(add_battery=False)
        p_inds = np.array([5], dtype=np.int32)
        self.fix_kwargs = {'data': self.ds,
                           'residual_learning': True,
                           'weight_vec': None,
                           'out_inds': p_inds,
                           'constraint_list': None}
        self.mod_test = PhysicallyConsistentRNN(name="TestPC_RNN", **test_kwargs, **self.fix_kwargs)

    def test_init(self):
        self.mod_test.fit()

    pass


def get_test_battery_model(n: int = 150,
                           noise: float = 1.0,
                           linear: bool = True):
    """Constructs a fitted battery model."""
    data = np.ones((n, 2)) * np.array([50, 0.0])
    ds2 = construct_test_ds(n, c_series=1, n_feats=2)
    ds2.data = data
    if noise > 0.0:
        ds2.data += np.random.normal(0.0, noise, (n, 2))
        ds2.standardize()
    ds2.split_data()
    m2 = BatteryModel(ds2)
    m2.name = "BatteryTest"
    par3 = 0 if linear else 1
    m2.params = np.array([0, 1, par3])
    return m2


class TestBattery(TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Define models
        n = 150
        self.m = get_test_battery_model(n, noise=0.0, linear=True)
        self.m2 = get_test_battery_model(n, noise=1.0, linear=False)

    def test_eval(self):
        val = 5.0
        self.assertAlmostEqual(self.m._eval_at(val), val, msg="_eval_at sucks!")
        self.assertAlmostEqual(self.m2._eval_at(val), 2 * val, msg="_eval_at sucks!")
        self.assertAlmostEqual(self.m2._eval_at(-val), -val, msg="_eval_at sucks!")

    def test_battery(self):
        n = 10
        in_data = np.ones((n, 1, 2))
        soc = np.arange(n) + 30
        in_data[:, 0, 0] = soc
        p = self.m.predict(in_data)
        self.assertTrue(np.array_equal(p.reshape(n), soc + 1))

    def test_nonlinear_battery(self):
        # This fucking fails sometimes, but not often...
        n = 5
        in_data = np.ones((n, 1, 2))
        soc = np.arange(n) + 30
        in_data[:, 0, 0] = soc
        p = self.m2.predict(in_data)
        self.assertTrue(np.array_equal(p.reshape(n), soc + 2))

    pass
