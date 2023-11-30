import numpy as np
import pandas as pd
from alibi.explainers import CounterfactualProto

from carla import RecourseMethod
from carla.recourse_methods.processing import merge_default_parameters


class Proto(RecourseMethod):
    _DEFAULT_HYPERPARAMS = {"max_iterations": 1000,
                            "theta": 100,
                            "c_steps": 1
                            }

    def __init__(self,
                 sess = None,
                 classifier_model=None,
                 hyperparams=None
                 ):
        """Init method

        :param data_interface: an interface class to access data related params.
        :param model_interface: an interface class to access trained ML model.

        """

        checked_hyperparams = merge_default_parameters(
            hyperparams, self._DEFAULT_HYPERPARAMS
        )

        # initializing model related variables
        self._mlmodel = classifier_model
        self.num_features = len(self._mlmodel.feature_input_order)

        shape = (1,) + self._mlmodel.data.df_train[self._mlmodel.data.continuous].to_numpy().shape[1:]
        feature_range = (self._mlmodel.data.df_train[self._mlmodel.data.continuous].min().min(),
                         self._mlmodel.data.df_train[self._mlmodel.data.continuous].max().max())
        predict_fn = lambda x: self._mlmodel.predict_proba(x)

        self.proto = CounterfactualProto(predict_fn, shape, gamma=0, theta=checked_hyperparams['theta'],
                            max_iterations=checked_hyperparams['max_iterations'],
                            feature_range=feature_range, c_init=0, c_steps=checked_hyperparams['c_steps'], use_kdtree=True,
                            sess=sess)

        self.proto.fit(self._mlmodel.data.df_train[self._mlmodel.data.continuous].to_numpy())

    def get_counterfactuals(self, factuals: pd.DataFrame):
        cfs = []
        factuals = factuals.copy()[self._mlmodel.data.continuous]
        for index, factual in factuals.iterrows():
            query_instance = factual.to_numpy().reshape(1, self.num_features)
            cf = self.proto.explain(query_instance, k=5, k_type='mean')
            if cf.cf is not None:
                cfs.append(cf.cf['X'].flatten())
            else:
                cfs.append(query_instance.flatten())
        df_cfs = pd.DataFrame(cfs, columns=self._mlmodel.feature_input_order).astype(np.float32)
        return df_cfs

