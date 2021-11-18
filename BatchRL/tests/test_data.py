from typing import Optional, Tuple, List
from unittest import TestCase

import numpy as np

from data_processing.dataset import ModelDataView, SeriesConstraint, Dataset
from data_processing.preprocess import standardize, fill_holes_linear_interpolate, remove_outliers, clean_data, \
    interpolate_time_series, gaussian_filter_ignoring_nans
from rest.client import test_rest_client, DataStruct
from util.numerics import nan_array_equal, num_nans
from util.util import EULER, skip_if_no_internet
from util.visualize import plot_dataset

SYNTH_DATA_NAME = "SyntheticModelData"


def construct_test_ds(n: int = 201, c_series: int = 3, n_feats: int = 4) -> Dataset:
    """Constructs a dataset for testing.

    Args:
        n: Number of rows.
        c_series: The index of the control series.
        n_feats: Number of columns in data.

    Returns:
        A dataset with 4 series, one of which is controllable.
    """
    # Check input
    assert n_feats >= 2, "At least two columns required!"
    assert c_series < n_feats, "Control index out of bounds!"

    # Define dataset
    dat = np.empty((n, n_feats), dtype=np.float32)
    dat[:, 0] = np.arange(n)
    dat[:, 1] = np.reciprocal(1.0 + np.arange(n))
    if n_feats > 2:
        dat[:, 2] = np.reciprocal(1.0 + np.arange(n))
    if n_feats > 3:
        dat[:, 3] = 1 + np.reciprocal(1.0 + np.arange(n))
    if n_feats > 4:
        for k in range(n_feats - 4):
            dat[:, 4 + k] = np.sin(np.arange(n))
    c_inds = np.array([c_series])
    ds = get_test_ds(dat, c_inds, dt=60 * 6, name=SYNTH_DATA_NAME)
    ds.seq_len = 8
    ds.val_percent = 0.33
    ds.split_data()
    return ds


def get_test_ds(dat: np.ndarray, c_inds: np.ndarray,
                name: str = SYNTH_DATA_NAME,
                dt: int = 60 * 12,
                t_init: str = '2019-01-01 12:00:00') -> Dataset:
    """Constructs a test dataset.

    Args:
        dat: The data array.
        c_inds: The control indices.
        name: The name of the dataset.
        dt: Timestep in minutes.
        t_init: Initial time.

    Returns:
        New dataset with dummy descriptions and unscaled data.
    """
    n_series = dat.shape[1]
    descs = np.array([f"Series {i} [unit{i}]" for i in range(n_series)])
    is_sc = np.array([False for _ in range(n_series)])
    sc = np.empty((n_series, 2), dtype=np.float32)
    ds = Dataset(dat, dt, t_init, sc, is_sc, descs, c_inds, name=name)
    return ds


def get_full_model_dataset(n: int = 150, dt: int = 12 * 60, add_battery: bool = True) -> Dataset:
    """Creates a test dataset for the full model.

    Includes the room and the battery.

    Args:
        n: Number of rows in data.
        dt: Number of minutes in a timestep.
        add_battery: Whether to include the battery.

    Returns:
        Dataset with normal data.
    """
    n_series = 10 if add_battery else 8
    shape = (n, n_series)
    data = np.random.normal(1.0, 1.0, shape)
    c_inds = np.array([4, 9]) if add_battery else np.array([4])
    ds = get_test_ds(data, c_inds, dt=dt)

    # Scale sin and cos time accordingly
    dat = np.copy(data[:, 6])
    data[:, 6] = np.sin(dat)
    data[:, 7] = np.cos(dat)

    if add_battery:
        # Assert SoC bounds
        data[:, 8] += 50.0

    # Overwrite default values
    ds.seq_len = 8
    ds.val_percent = 0.3

    return ds


class TestDataset(TestCase):
    """Tests the dataset class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Define dataset
        self.dat = np.array([0, 2, 3, 7, 8,
                             1, 3, 4, 8, 9,
                             1, 4, 5, 7, 8,
                             2, 5, 6, 7, 9], dtype=np.float32).reshape((4, -1))
        self.c_inds = np.array([1, 3])
        self.ds = get_test_ds(np.copy(self.dat), self.c_inds)
        self.ds.standardize()

        # Dataset containing nans
        self.dat_nan = np.array([0, 2, 3, 7, 8,
                                 1, 3, 4, 8, np.nan,
                                 2, 3, 4, 8, np.nan,
                                 3, 3, 4, 8, np.nan,
                                 4, 4, 5, 7, 8,
                                 5, 4, 5, 7, 8,
                                 6, 4, 5, 7, 8,
                                 7, 4, 5, 7, 8,
                                 8, 4, np.nan, 7, 8,
                                 9, 4, np.nan, 7, 8,
                                 10, 5, 6, 7, 9], dtype=np.float32).reshape((-1, 5))
        ds_nan = get_test_ds(self.dat_nan, self.c_inds, name="SyntheticTest")
        ds_nan.seq_len = 2
        self.ds_nan = ds_nan
        self.ds_nan.val_percent = 0.33
        self.ds_nan.split_data()

        # Create MDV
        self.mdv = ModelDataView(self.ds_nan, "Test", 2, 7)

    def scaling_helper(self, sc_state, uns_state, **kwargs):
        sc_state_exp = self.ds.scale(uns_state, remove_mean=True, **kwargs)
        uns_state_exp = self.ds.scale(sc_state, remove_mean=False, **kwargs)
        assert np.allclose(sc_state, sc_state_exp, atol=1.e-6), f"{sc_state} != {sc_state_exp}"
        assert np.allclose(uns_state, uns_state_exp, atol=1.e-6), f"{uns_state} != {uns_state_exp}"

    def test_scale_fun(self):
        # This fails sometimes...
        ind = np.random.randint(0, len(self.dat))
        uns_state = self.dat[ind]
        sc_state = self.ds.data[ind]
        self.scaling_helper(sc_state, uns_state)

    def test_scale_fun_2(self):
        ind = np.random.randint(0, len(self.dat))
        uns_state_c = self.dat[ind][self.c_inds]
        sc_state_c = self.ds.data[ind][self.c_inds]
        self.scaling_helper(sc_state_c, uns_state_c, control_only=True)

    def test_scale_fun_3(self):
        uns_state_c = self.dat[:, self.ds.non_c_inds]
        sc_state_c = self.ds.data[:, self.ds.non_c_inds]
        self.scaling_helper(sc_state_c, uns_state_c, state_only=True)

    def test_slice_time(self):
        ds_sliced = self.ds_nan.slice_time(4, 8)
        self.assertTrue(np.all(np.logical_not(np.isnan(ds_sliced.data))))

    def test_full_test_ds(self):
        n_rows = 10
        ds = get_full_model_dataset(n_rows)
        self.assertEqual(ds.data.shape, (n_rows, 10))
        ex_inds = np.array([1, 5, 8, 9])
        exp_prep_inds = np.array([1, 4, 7, 9])
        self.assertTrue(np.array_equal(exp_prep_inds, ds.to_prepared(ex_inds)),
                        "Indices conversion invalid!")

        for k in [6, 7]:
            self.assertTrue(np.all(ds.data[:, k] <= 1.0), "Too large values!")
            self.assertTrue(np.all(ds.data[:, k] >= -1.0), "Too small values!")

    def test_saving(self):
        self.ds.save()

    def test_plot(self):
        plot_dataset(self.ds, False, ["Test", "Fuck"])

    def test_model_data_view_1(self):
        mod_dat = self.mdv.get_rel_data()
        self.assertTrue(nan_array_equal(mod_dat, self.dat_nan[2:9]),
                        "Something's fucking wrong with model data view's get_rel_data!!")

    def test_model_data_view_2(self):
        self.mdv.extract_streak(3)
        str_dat, i = self.mdv.extract_streak(3)
        exp_dat = np.array([
            self.dat_nan[5:7],
            self.dat_nan[6:8],
        ])
        if not np.array_equal(str_dat, exp_dat) or not i == 3:
            raise AssertionError("Something in MDVs extract_streak is fucking wrong!!")

    def test_model_data_view_3(self):
        # Test disjoint streak extraction
        dis_dat, dis_inds = self.mdv.extract_disjoint_streaks(2, 1)
        exp_dis = np.array([[
            self.dat_nan[4:6],
            self.dat_nan[5:7],
        ]])
        if not np.array_equal(dis_dat, exp_dis) or not np.array_equal(dis_inds, np.array([2])):
            raise AssertionError("Something in extract_disjoint_streaks is fucking wrong!!")

    def test_model_data_view_4(self):
        # Test longest streak extraction
        dis_dat, i = self.mdv.extract_streak(3, True, True)
        exp_dat = np.array([
            self.dat_nan[4:6],
            self.dat_nan[5:7],
            self.dat_nan[6:8],
        ])
        self.assertTrue(np.array_equal(dis_dat, exp_dat), "Extracted data wrong!")
        self.assertEqual(i, 2, "Extracted indices wrong!")

    def test_split_data(self):
        # Test split_data
        test_dat = self.ds_nan.split_dict['test'].get_rel_data()
        val_dat = self.ds_nan.split_dict['val'].get_rel_data()
        if not nan_array_equal(test_dat, self.dat_nan[7:]):
            raise AssertionError("Something in split_data is fucking wrong!!")
        if not nan_array_equal(val_dat, self.dat_nan[3:7]):
            raise AssertionError("Something in split_data is fucking wrong!!")

    def test_get_day(self):
        # Test get_day
        day_dat, ns = self.ds_nan.get_days('val')
        exp_first_out_dat = np.array([
            [5, 5.0, 8.0],
            [6, 5.0, 8.0],
        ], dtype=np.float32)
        if ns[0] != 4 or not np.array_equal(day_dat[0][1], exp_first_out_dat):
            raise AssertionError("get_days not implemented correctly!!")

    def test_standardize(self):
        self.assertTrue(np.allclose(self.ds.scaling[0][0], 1.0), "Standardizing failed!")

    def test_scaling_prop(self):
        self.assertTrue(self.ds.fully_scaled, "Dataset should be fully scaled!")
        self.assertFalse(self.ds.partially_scaled, "Dataset should be fully scaled!")

    def test_get_scaling_mul(self):
        # Test get_scaling_mul
        scaling, is_sc = self.ds.get_scaling_mul(0, 3)
        is_sc_exp = np.array([True, True, True])
        sc_mean_exp = np.array([1.0, 1.0, 1.0])
        self.assertTrue(np.array_equal(is_sc_exp, is_sc), "get_scaling_mul failed!")
        self.assertTrue(np.allclose(sc_mean_exp, scaling[:, 0]), "get_scaling_mul failed!")

    def test_transform_c_list(self):
        c_list = [
            SeriesConstraint('interval', np.array([0.0, 1.0])),
            SeriesConstraint(),
            SeriesConstraint(),
            SeriesConstraint(),
            SeriesConstraint(),
        ]
        self.ds.transform_c_list(c_list)
        if not np.allclose(c_list[0].extra_dat[1], 0.0):
            raise AssertionError("Interval transformation failed!")

    def test_index_trafo(self):
        # Specify index tests
        test_list = [
            (np.array([2, 4], dtype=np.int32), np.array([1, 2], dtype=np.int32), self.ds.to_prepared),
            (np.array([2, 3], dtype=np.int32), np.array([1, 4], dtype=np.int32), self.ds.to_prepared),
            (np.array([0, 1, 2, 3, 4], dtype=np.int32), np.array([0, 3, 1, 4, 2], dtype=np.int32), self.ds.to_prepared),
            (np.array([0, 1, 2], dtype=np.int32), np.array([0, 2, 4], dtype=np.int32), self.ds.from_prepared),
            (np.array([2, 3, 4], dtype=np.int32), np.array([4, 1, 3], dtype=np.int32), self.ds.from_prepared),
        ]

        # Run index tests
        for t in test_list:
            inp, sol, fun = t
            out = fun(inp)
            if not np.array_equal(sol, out):
                print("Test failed :(")
                raise AssertionError(f"Function: {fun} with input: {inp} "
                                     f"not giving: {sol} but: {out}!!!")

    pass


class TestDataSynthetic(DataStruct):
    """Synthetic and short dataset to be used for debugging.

    Overrides `get_data` to avoid having to use the REST
    client to load the data, instead it is defined each time
    it is called.
    """

    def __init__(self):
        name = "SyntheticTest"
        super().__init__([0, 1, 2], name=name)

    def get_data(self, verbose: int = 0) -> Optional[Tuple[List, List]]:
        # First Time series
        dict1 = {'description': "Synthetic Data Series 1: Base Series", 'unit': "Test Unit 1"}
        val_1 = np.array([1.0, 2.3, 2.3, 1.2, 2.3, 0.8])
        dat_1 = np.array([
            np.datetime64('2005-02-25T03:31'),
            np.datetime64('2005-02-25T03:39'),
            np.datetime64('2005-02-25T03:48'),
            np.datetime64('2005-02-25T04:20'),
            np.datetime64('2005-02-25T04:25'),
            np.datetime64('2005-02-25T04:30'),
        ], dtype='datetime64')

        # Second Time series
        dict2 = {'description': "Synthetic Data Series 2: Delayed and longer", 'unit': "Test Unit 2"}
        val_2 = np.array([1.0, 1.4, 2.1, 1.5, 3.3, 1.8, 2.5])
        dat_2 = np.array([
            np.datetime64('2005-02-25T03:51'),
            np.datetime64('2005-02-25T03:59'),
            np.datetime64('2005-02-25T04:17'),
            np.datetime64('2005-02-25T04:21'),
            np.datetime64('2005-02-25T04:34'),
            np.datetime64('2005-02-25T04:55'),
            np.datetime64('2005-02-25T05:01'),
        ], dtype='datetime64')

        # Third Time series
        dict3 = {'description': "Synthetic Data Series 3: Repeating Values", 'unit': "Test Unit 3"}
        val_3 = np.array([0.0, 1.4, 1.4, 0.0, 3.3, 3.3, 3.3, 3.3, 0.0, 3.3, 2.5])
        dat_3 = np.array([
            np.datetime64('2005-02-25T03:45'),
            np.datetime64('2005-02-25T03:53'),
            np.datetime64('2005-02-25T03:59'),
            np.datetime64('2005-02-25T04:21'),
            np.datetime64('2005-02-25T04:23'),
            np.datetime64('2005-02-25T04:25'),
            np.datetime64('2005-02-25T04:34'),
            np.datetime64('2005-02-25T04:37'),
            np.datetime64('2005-02-25T04:45'),
            np.datetime64('2005-02-25T04:55'),
            np.datetime64('2005-02-25T05:01'),
        ], dtype='datetime64')
        return [(val_1, dat_1), (val_2, dat_2), (val_3, dat_3)], [dict1, dict2, dict3]


TestData2 = TestDataSynthetic()


class TestDataProcessing(TestCase):
    """Tests the dataset class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Define data
        seq1 = np.array([1, 2, np.nan, 1, 2, np.nan, 3, 2, 1, 3, 4, np.nan, np.nan, 3, 2, 3, 1, 3, np.nan, 1, 2])
        seq2 = np.array([3, 4, np.nan, np.nan, 2, np.nan, 3, 2, 1, 3, 4, 7, np.nan, 3, 2, 3, 1, np.nan, np.nan, 3, 4])
        n = seq1.shape[0]
        all_dat = np.empty((n, 2), dtype=np.float32)
        all_dat[:, 0] = seq1
        all_dat[:, 1] = seq2
        self.all_dat = all_dat

        self.dt_mins = 15
        self.dat, self.m = TestData2.get_data()

    def test_data_struct_copy(self):
        dat_cp = TestData2.copy()
        self.assertEqual(TestData2.start_date, dat_cp.start_date)
        self.assertEqual(TestData2.end_date, dat_cp.end_date)
        self.assertEqual(TestData2.data_ids, dat_cp.data_ids)
        self.assertNotEqual(id(TestData2.data_ids), id(dat_cp.data_ids))
        self.assertNotEqual(TestData2, dat_cp)

    def test_data_struct_slice(self):
        dat_cp = TestData2[0:1]
        assert len(dat_cp.data_ids) == 1
        assert dat_cp.data_ids == TestData2.data_ids[0:1], \
            f"Inds: {dat_cp.data_ids}  != {TestData2.data_ids[0:1]}!"

    def test_interpolate(self):
        interpolate_time_series(self.dat[0], self.dt_mins, verbose=False)

    def test_gaussian_filter(self):
        data2, dt_init2 = interpolate_time_series(self.dat[1], self.dt_mins, verbose=False)
        gaussian_filter_ignoring_nans(data2)

    @skip_if_no_internet
    def test_rest(self):
        if not EULER:
            test_rest_client()

    def test_standardize(self):
        # Test Standardizing
        m = [{}, {}]
        all_dat, m = standardize(self.all_dat, m)
        self.assertAlmostEqual(np.nanmean(all_dat[:, 0]).item(), 0.0, msg="Standardizing failed")
        mas = m[0].get('mean_and_std')
        self.assertIsNotNone(mas, msg="Mean and std not added to dict!")
        self.assertAlmostEqual(np.nanmean(self.all_dat[:, 0]).item(), mas[0],
                               msg="Standardizing mutating data!")

    def test_hole_fill(self):
        # Test hole filling by interpolation
        test_ts = np.array([np.nan, np.nan, 1, 2, 3.5, np.nan, 4.5, np.nan])
        test_ts2 = np.array([1, 2, 3.5, np.nan, np.nan, 5.0, 5.0, np.nan, 7.0])
        fill_holes_linear_interpolate(test_ts, 1)
        fill_holes_linear_interpolate(test_ts2, 2)
        self.assertEqual(num_nans(test_ts), 3, msg="fill_holes_linear_interpolate not working!")
        self.assertEqual(num_nans(test_ts2), 0, msg="fill_holes_linear_interpolate not working!")

    def test_outlier_removal(self):
        # Test Outlier Removal
        test_ts3 = np.array([1, 2, 3.5, np.nan, np.nan, 5.0,
                             5.0, 17.0, 5.0, 2.0, -1.0, np.nan,
                             7.0, 7.0, 17.0, np.nan, 20.0, 5.0, 6.0])
        min_val = 0.0
        max_val = 100.0
        remove_outliers(test_ts3, 5.0, [min_val, 100.0])
        self.assertTrue(max_val >= np.nanmin(test_ts3).item() >= min_val, "Outlier removing failed!")

    def test_clean_data(self):
        # Clean data
        dat = self.dat[2]
        len_dat = len(dat[0])
        values, dates = clean_data(dat, [0.0], 4, [3.3], verbose=False)
        self.assertEqual(len(values), len_dat - 3, "Removed too many values!")
        values2, dates2 = clean_data(dat, n_cons_least=4, verbose=False)
        self.assertEqual(len(values2), len_dat - 4,
                         "Removed too many or too few values!")
        self.assertEqual(len(values2), len(dates2), "clean_data failed horribly!")
        pass
