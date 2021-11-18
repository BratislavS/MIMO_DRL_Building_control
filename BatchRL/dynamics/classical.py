import numpy as np

from data_processing.dataset import Dataset
from dynamics.base_model import BaseDynamicsModel
from ml.sklearn_util import SKLoader, get_skl_model_name
from util.numerics import check_shape, trf_mean_and_std
from util.util import train_decorator


class SKLearnModel(BaseDynamicsModel):
    """The naive model that predicts the last input seen.

    """

    name: str = "Linear"  #: Base name of model.
    n_out: int  #: Number of series that are predicted.

    def __init__(self, data: Dataset, skl_model, residual: bool = True, clip_ind: int = None, **kwargs):
        """Initializes the constant model.

        All series specified by prep_inds are predicted by the last seen value.

        Args:
            dataset: Dataset containing the data.
            kwargs: Kwargs for base class, e.g. `in_inds`.
        """
        # Construct meaningful name
        name = get_skl_model_name(skl_model)
        if kwargs.get('name'):
            # I need the walrus!
            name = kwargs['name']
        kwargs['name'] = name

        # Init base class
        super().__init__(data, **kwargs)

        # Save model
        self.m = SKLoader(skl_model, self)

        # Save data
        self.n_out = len(self.out_inds)
        self.nc = data.n_c
        self.residual_learning = residual

        # Fitting model
        self.is_fitted = False
        self.skl_mod = skl_model

        self.clip_ind = clip_ind
        self.clip_val = None
        if clip_ind is not None:
            assert clip_ind < len(self.out_inds)
            self.clip_val = trf_mean_and_std(0, mean_and_std=self.data.scaling[clip_ind], remove=True)

    @train_decorator()
    def fit(self, verbose: int = 0, train_data: str = "train") -> None:
        """Fit linear model."""

        # Check if already fitted
        if self.is_fitted and train_data == self.fit_data:
            if verbose:
                print("Already fitted!")
            return
        self.fit_data = train_data

        # Prepare the data
        input_data, output_data = self.get_fit_data(train_data,
                                                    residual_output=self.residual_learning)
        in_sh = input_data.shape
        first_sh, last_sh = in_sh[0], in_sh[-1]
        input_data_2d = input_data.reshape((first_sh, -1))

        # Fit
        if verbose > 0:
            print("Fitting sklearn model...")
        self.skl_mod.fit(input_data_2d, output_data)
        self.is_fitted = True

    def predict(self, in_data: np.ndarray) -> np.ndarray:
        """Make predictions by applying the linear model.

        Args:
            in_data: Prepared data.

        Returns:
            The predictions.
        """
        check_shape(in_data, (-1, -1, -1))

        # Add previous state contribution
        prev_state = self._extract_output(in_data)

        # Flatten
        sh = in_data.shape
        in_data_res = in_data.reshape((sh[0], -1))
        prev_state_res = prev_state.reshape((sh[0], -1))

        # Predict
        p = prev_state_res + self.skl_mod.predict(in_data_res)
        if self.clip_ind is not None:
            p_to_be_clipped = np.copy(p[:, self.clip_ind])
            p_to_be_clipped = np.clip(p_to_be_clipped, self.clip_val, a_max=None)
            p[:, self.clip_ind] = p_to_be_clipped
        return p
