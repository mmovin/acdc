from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

from carla.models.api import MLModel

from ...api import RecourseMethod
from ...processing import merge_default_parameters


class NN(RecourseMethod):
    """
    Implementation of Dice from Mothilal et.al. [1]_.

    Parameters
    ----------
    mlmodel : carla.model.MLModel
        Black-Box-Model
    hyperparams : dict
        Dictionary containing hyperparameters. See notes below for its contents.

    Methods
    -------
    get_counterfactuals:
        Generate counterfactual examples for given factuals.
    encode_normalize_order_factuals:
        Uses encoder and scaler from black-box-model to preprocess data as needed.

    Notes
    -----
    - Hyperparams
        Hyperparameter contains important information for the recourse method to initialize.
        Please make sure to pass all values as dict with the following keys.

        * "num": int, default: 1
            Number of counterfactuals per factual to generate
        * "desired_class": int, default: 1
            Given a binary class label, the desired class a counterfactual should have (e.g., 0 or 1)
        * "posthoc_sparsity_param": float, default: 0.1
            Fraction of post-hoc preprocessing steps.
    - Restrictions:
        *   Only the model agnostic approach (backend: sklearn) is used in our implementation.
        *   ML model needs to have a transformation pipeline for normalization, encoding and feature order.
            See pipelining at carla/models/catalog/catalog.py for an example ML model class implementation

    .. [1] R. K. Mothilal, Amit Sharma, and Chenhao Tan. 2020. Explaining machine learning classifiers
            through diverse counterfactual explanations
    """

    _DEFAULT_HYPERPARAMS = {"leaf_size": 2}

    def __init__(self, mlmodel: MLModel, hyperparams: Dict[str, Any]) -> None:
        super().__init__(mlmodel)
        self._continuous = mlmodel.data.continuous
        self._categoricals = mlmodel.data.categoricals
        self._target = mlmodel.data.target

        checked_hyperparams = merge_default_parameters(
            hyperparams, self._DEFAULT_HYPERPARAMS
        )
        self._mlmodel = mlmodel
        self.positive_examples = self.get_positive_examples()
        self.tree = KDTree(self.positive_examples, leaf_size=checked_hyperparams['leaf_size'], metric='l1')

    def get_positive_examples(self):
        data = self._mlmodel.data.raw
        data = data[data[self._mlmodel.data.target] == 1].drop(self._mlmodel.data.target, axis=1).reset_index(drop=True)
        return self._mlmodel.scaler.transform(data)


    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        # Prepare factuals
        querry_instances = self._mlmodel.perform_pipeline(factuals.copy()).to_numpy()

        # check if querry_instances are not empty
        if not querry_instances.shape[0] > 0:
            raise ValueError("Factuals should not be empty")

        _, ind = self.tree.query(querry_instances, k=1)
        df_cfs = pd.DataFrame(self.positive_examples[ind].reshape(querry_instances.shape), columns=self._mlmodel.feature_input_order).astype(np.float32)

        return df_cfs
