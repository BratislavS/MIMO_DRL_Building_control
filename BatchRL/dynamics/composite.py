from typing import List

import numpy as np

from data_processing.dataset import Dataset
from dynamics.base_model import BaseDynamicsModel
from util.numerics import has_duplicates
from util.util import ProgWrap, prog_verb


class CompositeModel(BaseDynamicsModel):
    """The composite model, combining multiple models.

    All models need to be based on the same dataset.
    """

    model_list: List[BaseDynamicsModel]

    def __init__(self, dataset: Dataset, model_list: List[BaseDynamicsModel], new_name: str = None):
        """Initialize the Composite model.

        All individual model need to be initialized with the same dataset!

        Args:
            dataset: The common `Dataset`.
            model_list: A list of dynamics models defined for the same dataset.
            new_name: The name to give to this model, default produces very long names.

        Raises:
            ValueError: If the model in list do not have access to `dataset` or if
                any series is predicted by multiple models.
        """
        # Compute name and check datasets
        name = dataset.name + "Composite"
        for m in model_list:
            name += f"_{m.name}"
            if m.data != dataset:
                raise ValueError(f"Model {m.name} needs to model the same dataset "
                                 "as the Composite model.")
        if new_name is not None:
            name = new_name

        # Collect indices and initialize base class.
        n_pred_full = dataset.d - dataset.n_c
        all_out_inds = np.concatenate([m.out_inds for m in model_list])
        if has_duplicates(all_out_inds):
            raise ValueError("Predicting one or more series multiple times.")
        out_inds = dataset.from_prepared(np.arange(n_pred_full))
        super().__init__(dataset, name, out_inds, None)

        # Reset the indices, since we do not want to permute twice!
        self.p_in_indices = np.arange(dataset.d)

        # We allow only full models, i.e. when combined, the models have to predict
        # all series except for the controlled ones.
        if self.n_pred != n_pred_full or len(all_out_inds) != n_pred_full:
            raise ValueError("You need to predict all non-control series!")

        # Save models
        self.model_list = model_list

    def init_1day(self, day_data: np.ndarray) -> None:
        """Calls the same function on all models in list.

        Args:
            day_data: The data for the initialization.
        """
        for m in self.model_list:
            m.init_1day(day_data)

    def fit(self, verbose: int = 0, train_data: str = "train") -> None:
        """Fits all the models."""
        self.fit_data = train_data
        with ProgWrap(f"Fitting sub-models on part: '{train_data}'...", verbose > 0):
            for ct, m in enumerate(self.model_list):
                print(f"Fitting model {ct}: {m.name}")
                m.fit(verbose=prog_verb(verbose), train_data=train_data)

    def predict(self, in_data: np.ndarray) -> np.ndarray:
        """Aggregated prediction by predicting with all models.

        Args:
            in_data: Prepared data.

        Returns:
            Aggregated predictions.
        """
        # Get shape of prediction
        in_sh = in_data.shape
        out_dat = np.empty((in_sh[0], self.n_pred), dtype=in_data.dtype)

        # Predict with all the models
        for m in self.model_list:
            in_inds = m.p_in_indices
            out_inds = m.p_out_inds
            pred_in_dat = in_data[:, :, in_inds]
            preds = m.predict(pred_in_dat)
            out_dat[:, out_inds] = preds

        return out_dat

    def disturb(self):
        """Returns a sample of noise.
        """
        out_dat = np.empty((self.n_pred,), dtype=np.float32)

        # Disturb with all the models
        curr_ind = 0
        for m in self.model_list:
            n_pred_m = m.n_pred
            out_inds = m.p_out_inds
            out_dat[out_inds] = m.disturb()
            curr_ind += n_pred_m

        return out_dat

    def model_disturbance(self, data_str: str = 'train'):
        """Models the disturbances for all sub-models."""
        for m in self.model_list:
            m.model_disturbance(data_str)
        self.modeled_disturbance = True
