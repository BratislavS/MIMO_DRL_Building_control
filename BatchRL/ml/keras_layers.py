"""Custom keras layers.

Define your custom keras layers here.
There is also a function that tests the layers
for some example input.
"""
from keras import backend as K, activations
from keras.layers import Layer, GaussianNoise, Lambda

from data_processing.dataset import SeriesConstraint
from util.util import *


class SeqInput(Layer):
    """Dummy Layer, it lets you specify the input shape
    when used as a first layer in a Sequential model.
    """

    def __init__(self, **kwargs):
        """Initializes the layer.

        Args:
            **kwargs: kwargs for super.
        """
        super(SeqInput, self).__init__(**kwargs)

    def call(self, x, **kwargs):
        """Returns `x` unchanged.

        Args:
            x: Input tensor.

        Returns:
            `x` unchanged.
        """
        return x

    def compute_output_shape(self, input_shape):
        """The shape stays the same.

        Args:
            input_shape: The shape of the input.

        Returns:
            Same as input.
        """
        return input_shape


class IdRecurrent(Layer):
    """Dummy Layer, it passes the same values that are input into it.

    If `n` is specified, only a reduced number of features is
    returned, cannot be larger than the total number of features.
    If `return_sequences` is True, sequences are returned, else
    only the last element of the sequence.
    """

    def __init__(self, n: int = None, return_sequences: bool = True, **kwargs):
        """Initializes the layer.

        Args:
            n: Number of output features.
            return_sequences: Whether to return sequences.
            **kwargs: kwargs for super.
        """
        super().__init__(**kwargs)
        self.n = n
        self.r_s = return_sequences

    def call(self, x, **kwargs):
        """Returns `x` unchanged."""
        assert len(x.shape) == 3, "Only implemented for 3D tensor input."
        if self.n is None:
            ret_val = x
        else:
            ret_val = x[:, :, :self.n]
        if self.r_s:
            return ret_val
        return ret_val[:, -1, :]

    def compute_output_shape(self, input_shape):
        """The shape stays the same."""
        out_shape = [k for k in input_shape]
        if self.n is not None:
            out_shape[1] = self.n
        if not self.r_s:
            out_shape = [out_shape[0], out_shape[2]]
        return tuple(out_shape)


class IdDense(Layer):
    """Dummy Layer, it passes the same values that are input into it.

    If `n` is specified, only a reduced number of features is
    returned, cannot be larger than the total number of features.
    """

    def __init__(self, n: int = None, **kwargs):
        """Initializes the layer.

        Args:
            n: Number of output features.
            **kwargs: kwargs for super.
        """
        super().__init__(**kwargs)
        self.n = n

    def call(self, x, **kwargs):
        """Returns `x` unchanged."""
        assert len(x.shape) == 2, "Only implemented for 2D tensor input."
        if self.n is None:
            ret_val = x
        else:
            ret_val = x[:, :self.n]
        return ret_val

    def compute_output_shape(self, input_shape):
        """The shape stays the same."""
        out_shape = [k for k in input_shape]
        if self.n is not None:
            out_shape[1] = self.n
        return tuple(out_shape)


class ClipByValue(Layer):
    """Clipping layer.

    Clips all values in the input tensors into the
    range [`low`, `high`].
    """

    def __init__(self, low: float = 0.0, high: float = 1.0, **kwargs):
        """Initializes the layer.

        Args:
        """
        super().__init__(**kwargs)
        self.low = low
        self.high = high

    def call(self, x, **kwargs):
        """Returns clipped `x`."""
        return K.clip(x, self.low, self.high)

    def compute_output_shape(self, input_shape):
        """The shape stays the same."""
        return input_shape


class ConstrainedNoise(Layer):
    """
    Constrained noise layer.
    Note that the clipping will be active during testing.
    """

    consts: Sequence[SeriesConstraint]  #: Sequence of constraints
    input_noise_std: float  #: The std of the Gaussian noise to add
    is_input: bool

    def __init__(self, input_noise_std: float = 0,
                 consts: Sequence[SeriesConstraint] = None,
                 is_input: bool = True,
                 **kwargs):
        """
        Adds Gaussian noise with mean 0 and std as specified
        and then applies the constraints.

        Args:
            input_noise_std: The level of noise.
            consts: The list of constraints.
            is_input: Whether it is applied to an input tensor (3D)
                or an output tensor (2D).
            **kwargs: Layer kwargs.
        """
        super(ConstrainedNoise, self).__init__(**kwargs)
        self.input_noise_std = input_noise_std
        self.consts = consts
        self.is_input = is_input

    def call(self, x, **kwargs):
        """
        Builds the layer given the input x.

        Args:
            x: The input to the layer.

        Returns:
            The output of the layer satisfying the constraints.
        """
        x_modify = x

        # Add noise if std > 0
        if self.input_noise_std > 0:
            gn_layer = GaussianNoise(self.input_noise_std)
            x_modify = gn_layer(x_modify)

        # Enforce constraints
        if self.consts is not None:

            # Check shape
            n_feats = len(self.consts)
            # n_feats_actual = x_modify.shape[-1]
            # assert n_feats == n_feats_actual, f"Shape mismatch: {n_feats} constraints " \
            #                                   f"and {n_feats_actual} features!"

            noise_x = x_modify

            # Split features
            if self.is_input:
                feature_tensors = [noise_x[:, :, ct:(ct + 1)] for ct in range(n_feats)]
            else:
                feature_tensors = [noise_x[:, ct:(ct + 1)] for ct in range(n_feats)]
            for ct, c in enumerate(self.consts):
                if c[0] is None:
                    continue
                elif c[0] == 'interval':
                    iv = c[1]
                    feature_tensors[ct] = K.clip(feature_tensors[ct], iv[0], iv[1])
                elif c[0] == 'exact' and self.input_noise_std > 0:
                    feature_tensors[ct] = x[:, :, ct:(ct + 1)]
                else:
                    raise ValueError(f"Constraint type {c[0]} not supported!!")

            # Concatenate again
            x_modify = K.concatenate(feature_tensors, axis=-1)

        return x_modify

    def compute_output_shape(self, input_shape):
        """
        The shape stays the same.

        Args:
            input_shape: The shape of the input.

        Returns:
            Same as input.
        """
        return input_shape


class FeatureSlice(Layer):
    """Extracts specified features from tensor.

    TODO: Make it more efficient by considering not single slices but
        multiple consecutive ones.
    """

    slicing_indices: np.ndarray  #: The array with the indices.
    n_feats: int  #: The number of selected features.
    n_dims: int  #: The number of dimensions of the input tensor.
    return_last_seq: bool  #: Whether to only return the last slice of each sequence.

    def __init__(self, s_inds: np.ndarray,
                 n_dims: int = 3,
                 return_last_seq: bool = True,
                 **kwargs):
        """Initialize layer.

        Args:
            s_inds: The array with the indices.
            n_dims: The number of dimensions of the input tensor.
            **kwargs: The kwargs for super(), e.g. `name`.
        """
        super(FeatureSlice, self).__init__(**kwargs)
        self.slicing_indices = s_inds
        self.n_feats = len(s_inds)
        self.n_dims = n_dims
        self.return_last_seq = return_last_seq

        if n_dims == 2:
            raise NotImplementedError("Not implemented for 2D tensors!")

    def call(self, x, **kwargs):
        """
        Builds the layer given the input x. Selects the features
        specified in `slicing_indices` and concatenates them.

        Args:
            x: The input to the layer.

        Returns:
            The output of the layer containing the slices.
        """
        s = -1 if self.return_last_seq else slice(None)
        feature_tensors = [x[:, s, ct:(ct + 1)] for ct in self.slicing_indices]
        return K.concatenate(feature_tensors, axis=-1)

    def compute_output_shape(self, input_shape):
        """
        The shape only changes in the feature dimension.

        Args:
            input_shape: The shape of the input.

        Returns:
            Same as input with the last dimension changed.
        """
        s = input_shape
        if self.return_last_seq:
            return s[0], self.n_feats
        return s[0], s[1], self.n_feats


class ExtractInput(Layer):
    """Input extraction layer.

    Given a tensor with a large sequence length,
    this layer constructs the next input for the RNN
    with the previous output of the RNN.
    """

    slicing_indices: np.ndarray  #: The array with the indices.
    n_feats: int  #: The number of selected features that are predicted.
    seq_len: int  #: The sequence length.
    curr_ind: int  #: The current prediction index.

    def __init__(self, s_inds: np.ndarray,
                 seq_len: int,
                 curr_ind: int = 0,
                 **kwargs):
        """Initialize layer.

        Args:
            s_inds: The array with the indices.
            seq_len: The sequence length.
            curr_ind: The current offset.
            **kwargs: The kwargs for super(), e.g. `name`.
        """
        super(ExtractInput, self).__init__(**kwargs)
        self.slicing_indices = s_inds
        self.n_feats = len(s_inds)
        self.seq_len = seq_len
        self.curr_ind = curr_ind

    def call(self, x, **kwargs):
        """
        Builds the layer given the full data and the
        last prediction.

        Args:
            x: A list with the full data and the last prediction. If there is
                only one input, we will just return the input slice.

        Returns:
            The output of the layer that can be fed to the basic RNN.
        """
        end_ind = self.curr_ind + self.seq_len
        if not isinstance(x, list):
            # No prediction from last step given.
            return x[:, self.curr_ind: end_ind, :]
        x_in, x_out = x
        if len(x_out.shape) == 2:
            x_out = K.reshape(x_out, (-1, 1, self.n_feats))

        x_s = x_in.shape
        if x_s[-2] <= end_ind:
            raise ValueError("curr_ind or seq_len too big!")
        x_prev = x_in[:, self.curr_ind: (end_ind - 1), :]
        x_next = x_in[:, (end_ind - 1): end_ind, :]

        # Extract slices
        pred_ind = np.zeros((x_s[-1],), dtype=np.bool)
        pred_ind[self.slicing_indices] = True
        feat_tensor_list = []
        ct = 0
        for k in range(x_s[-1]):
            if not pred_ind[k]:
                feat_tensor_list += [x_next[:, :, k: (k + 1)]]
            else:
                feat_tensor_list += [x_out[:, :, ct: (ct + 1)]]
                ct += 1

        # Concatenate
        out_next = K.concatenate(feat_tensor_list, axis=-1)
        return K.concatenate([x_prev, out_next], axis=-2)

    def compute_output_shape(self, input_shape):
        """
        The sequence length changes to `seq_len`.

        Args:
            input_shape: The shape of the input.

        Returns:
            Same as input with the second dimension set to sequence length.

        Raises:
            ValueError: If the tensor shapes are incompatible with the layer.
        """
        s0 = input_shape
        if isinstance(s0, list):
            if len(s0) != 2:
                raise ValueError("Need exactly two tensors if there are multiple!")
            if s0[1][1] != self.n_feats:
                raise ValueError("Invalid indices or tensor shape!")
            s0 = s0[0]
        if len(s0) != 3:
            raise ValueError("Only implemented for 3D tensors!")
        return s0[0], self.seq_len, s0[2]


class ConstrainOutput(Layer):
    """Activation layer.

    Applies sigmoid to the tensor and scales the
    intermediate output to assert an output in a given range
    for each output feature.
    """

    def __init__(self, ranges: List[Tuple[Num, Num]], **kwargs):
        """Initializes the layer.

        Args:
            ranges: List of intervals.
        """
        super().__init__(**kwargs)
        self.low = np.array([i[0] for i in ranges], dtype=np.float32)
        self.dist = np.array([i[1] - i[0] for i in ranges], dtype=np.float32)

    def call(self, x, **kwargs):
        """Returns clipped `x`."""
        activated = activations.sigmoid(x)
        l_tensor = K.constant(self.low.reshape((1, -1)))
        d_tensor = K.constant(self.dist.reshape((1, -1)))
        return activated * d_tensor + l_tensor

    def compute_output_shape(self, input_shape):
        """The shape stays the same."""
        return input_shape


def get_constrain_layer(ranges: List[Tuple[Num, Num]]):
    """The same layer as the one above, but using a Lambda layer."""
    low = np.array([i[0] for i in ranges], dtype=np.float32)
    dist = np.array([i[1] - i[0] for i in ranges], dtype=np.float32)

    l_tensor = K.constant(low.reshape((1, -1)))
    d_tensor = K.constant(dist.reshape((1, -1)))

    def constrain_sigmoid(x):
        activated = activations.sigmoid(x)
        return activated * d_tensor + l_tensor

    ConstrainOutputLambda = Lambda(constrain_sigmoid)
    return ConstrainOutputLambda
