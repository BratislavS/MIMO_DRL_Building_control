from unittest import TestCase

import numpy as np
from keras import Input, Sequential, backend as K
from keras.engine import Layer
from keras.layers import Add

from data_processing.dataset import SeriesConstraint
from ml.keras_layers import SeqInput, ConstrainedNoise, FeatureSlice, \
    ExtractInput, IdDense, IdRecurrent, ClipByValue, ConstrainOutput
from util.util import rem_first
from util.numerics import check_in_range


def get_multi_input_layer_output(layer: Layer, inp_list, learning_phase: float = 1.0):
    """Tests layers with multiple input and / or output.

    Args:
        layer: The layer to test.
        inp_list: The list with input arrays.
        learning_phase: Whether to use learning or testing mode.

    Returns:
        The processed input.
    """
    if not isinstance(inp_list, list):
        inp_list = [inp_list]
    inputs = [Input(shape=rem_first(el.shape)) for el in inp_list if el is not None]
    if len(inputs) == 1:
        layer_out_tensor = layer(*inputs)
    else:
        layer_out_tensor = layer(inputs)
    k_fun = K.function([*inputs, K.learning_phase()], [layer_out_tensor])
    layer_out = k_fun([*inp_list, learning_phase])[0]
    return layer_out


def get_test_layer_output(layer: Layer, np_input, learning_phase: float = 1.0):
    """Test a keras layer.

    Builds a model with only the layer given and
    returns the output when given `np.input` as input.

    Args:
        layer: The keras layer.
        np_input: The input to the layer.
        learning_phase: Whether learning is active or not.

    Returns:
        The layer output.
    """
    # Construct sequential model with only one layer
    m = Sequential()
    m.add(layer)
    out, inp = m.output, m.input
    k_fun = K.function([inp, K.learning_phase()], [out])
    layer_out = k_fun([np_input, learning_phase])[0]
    return layer_out


class TestKeras(TestCase):
    """Test case class for keras tests.

    Run from `BatchRL` folder, otherwise the relative paths
    will be wrong and there will be folders generated in the wrong place.
    Use the Powershell script `run_tests.ps1` if possible.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Define shapes
        self.seq_shape = (2, 6, 4)
        self.seq_len_red = 2
        self.b_s, self.seq_len_test, self.n_feats = self.seq_shape
        self.seq_shape_red = (self.b_s, self.seq_len_red, self.n_feats)

        # Define the data
        self.seq_input = np.arange(self.b_s * self.seq_len_red * self.n_feats).reshape(self.seq_shape_red)
        self.seq_input_long = np.arange(self.b_s * self.seq_len_test * self.n_feats).reshape(self.seq_shape)
        self.feat_indices = np.array([0, 2], dtype=np.int32)
        self.n_feats_chosen = len(self.feat_indices)
        self.output = -1. * np.arange(self.b_s * self.n_feats_chosen).reshape((self.b_s, self.n_feats_chosen))
        self.id_1 = np.array([[1, 2, 3]])
        self.id_2 = np.array([[2, 2, 2]])

    def test_multiple_input(self):
        # Test multi input test
        add_out = get_multi_input_layer_output(Add(), [self.id_1, self.id_2])
        exp_out = self.id_1 + self.id_2
        self.assertTrue(np.array_equal(add_out, exp_out),
                        "Multi Input layer test not working!")

    def test_seq_input(self):
        # Test SeqInput
        inp_layer = SeqInput(input_shape=(self.seq_len_red, self.n_feats))
        layer_out = get_test_layer_output(inp_layer, self.seq_input, 1.0)
        self.assertTrue(np.allclose(layer_out, self.seq_input), "SeqInput layer not implemented correctly!!")

    def test_constraint_layer(self):
        # Test Constraint Layer
        consts = [
            SeriesConstraint('interval', [0.0, 1.0]),
            SeriesConstraint(),
            SeriesConstraint('exact'),
            SeriesConstraint('exact'),
        ]
        noise_level = 5.0
        const_layer = ConstrainedNoise(noise_level, consts, input_shape=(self.seq_len_red, self.n_feats))
        layer_out = get_test_layer_output(const_layer, self.seq_input, 1.0)
        layer_out_test = get_test_layer_output(const_layer, self.seq_input, 0.0)
        self.assertTrue(np.allclose(layer_out[:, :, 2:], self.seq_input[:, :, 2:]),
                        "Exact constraint in Constrained Noise layer not implemented correctly!!")
        self.assertTrue(check_in_range(layer_out[:, :, 0], 0.0, 1.00001),
                        "Interval constraint in Constrained Noise layer not implemented correctly!!")
        self.assertTrue(np.allclose(layer_out_test[:, :, 1:], self.seq_input[:, :, 1:]),
                        "Noise layer during testing still active!!")

    def test_feature_slice(self):
        # Test FeatureSlice layer
        lay = FeatureSlice(np.array(self.feat_indices), input_shape=(self.seq_len_red, self.n_feats))
        layer_out = get_test_layer_output(lay, self.seq_input)
        self.assertTrue(np.array_equal(layer_out, self.seq_input[:, -1, self.feat_indices]),
                        "FeatureSlice layer not working!!")

    def test_extract_input_layer(self):
        # Test ExtractInput layer
        lay = ExtractInput(np.array(self.feat_indices), seq_len=3, curr_ind=1)
        l_out = get_multi_input_layer_output(lay, [self.seq_input_long, self.output])
        l_out2 = get_multi_input_layer_output(lay, [self.seq_input_long, None])
        l_out3 = get_multi_input_layer_output(lay, self.seq_input_long)
        exp_out32 = np.copy(self.seq_input_long)[:, 1:4, :]
        exp_out = np.copy(exp_out32)
        exp_out[:, -1, self.feat_indices] = self.output
        self.assertTrue(np.array_equal(l_out, exp_out), "ExtractInput layer not working!")
        self.assertTrue(np.array_equal(l_out2, exp_out32), "ExtractInput layer not working!")
        self.assertTrue(np.array_equal(l_out3, exp_out32), "ExtractInput layer not working!")

    def test_id_recurrent_layer(self):
        # Test IdRecurrent layer
        lay = IdRecurrent(3, input_shape=rem_first(self.seq_input_long.shape))
        l_out = get_test_layer_output(lay, self.seq_input_long)
        self.assertTrue(np.array_equal(l_out, self.seq_input_long[:, :, :3]),
                        "IdRecurrent not working correctly!")
        lay = IdDense(1, input_shape=rem_first(self.output.shape))
        l_out = get_test_layer_output(lay, self.output)
        self.assertTrue(np.array_equal(l_out, self.output[:, :1]),
                        "IdDense not working correctly!")

    def test_clip_layer(self):
        # Test ClipByValue layer
        c_layer = ClipByValue(0.0, 1.0, input_shape=rem_first(self.seq_shape))
        l_out = get_test_layer_output(c_layer, self.seq_input_long)
        self.assertTrue(np.all(l_out >= 0.0) and np.all(l_out <= 1.0),
                        "ClipByValue not working correctly!")

    def test_constrain_output(self):
        # Test ConstrainOutput layer
        ints = [(0.0, 2.0), (-1.0, 5.0), (-1.0, 5.0), (-1.0, 5.0)]
        c_layer = ConstrainOutput(ints, input_shape=rem_first(self.seq_shape))
        l_out = get_test_layer_output(c_layer, self.seq_input_long)
        self.assertTrue(np.all(l_out[:, :, 0] <= 2.0) and np.all(l_out[:, :, 1:] <= 5.0),
                        "ClipByValue not working correctly!")
        self.assertTrue(np.all(l_out[:, :, 0] >= 0.0) and np.all(l_out[:, :, 1:] >= -1.0),
                        "ClipByValue not working correctly!")

        # Test 2d input
        c_layer_2d = ConstrainOutput(ints[:2], input_shape=rem_first(self.output.shape))
        l_out = get_test_layer_output(c_layer_2d, self.output)
        self.assertTrue(np.all(l_out[:, 0] <= 2.0) and np.all(l_out[:, 1:] <= 5.0),
                        "ClipByValue not working correctly!")
        self.assertTrue(np.all(l_out[:, 0] >= 0.0) and np.all(l_out[:, 1:] >= -1.0),
                        "ClipByValue not working correctly!")

    pass
