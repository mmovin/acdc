import numpy as np
import pandas as pd

from carla.evaluation import remove_nans
from sklearn.neighbors import LocalOutlierFactor

from carla.evaluation.api import Evaluation


class Lof(Evaluation):
    """
    Calculates the Local Outlier Factor Metric (Breuniq et al. 2000)
    Added by Movin et al. as suggested by Kanamori et al. 2020, and Laugel et al. 2019)
    """

    def __init__(self, mlmodel):
        super().__init__(mlmodel)
        self.columns = ["LOF"]
        self._mlmodel = mlmodel
        self._lof = self.fit_lof()

    def get_evaluation(self, factuals, counterfactuals):
        # only keep the rows for which counterfactuals could be found
        counterfactuals_without_nans, factuals_without_nans = remove_nans(
            counterfactuals, factuals
        )

        # return empty dataframe if no successful counterfactuals
        if counterfactuals_without_nans.empty:
            return pd.DataFrame(columns=self.columns)

        arr_cf = self.mlmodel.get_ordered_features(
            counterfactuals_without_nans
        ).to_numpy()

        lofs = self._lof.predict(arr_cf)
        lofs = np.array([1 if l == -1 else 0 for l in lofs]).reshape((-1, 1))

        return pd.DataFrame(lofs, columns=self.columns)

    def fit_lof(self):
        lof = LocalOutlierFactor(n_neighbors=5, novelty=True)
        positive_data = self._mlmodel.data.df_train[self._mlmodel.data.df_train[self._mlmodel.data.target] == 1] \
            .drop(self._mlmodel.data.target, axis=1)
        lof.fit(positive_data)
        return lof