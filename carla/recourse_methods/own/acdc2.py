import timeit

import dice_ml
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Input, Model, regularizers, constraints, initializers
from keras.layers import Dense
from sklearn.neighbors import KDTree

from carla import RecourseMethod
from carla.recourse_methods.own.rbflayer import RBFLayer, InitCentersKMeans
from carla.recourse_methods.own.save_load import get_full_path
from carla.recourse_methods.processing import merge_default_parameters

"""
Class to generate Consistent Counterfactuals via ACDC. Code inspired by https://github.com/interpretml/DiCE
"""


class ACDC2(RecourseMethod):
    _DEFAULT_HYPERPARAMS = {"proximity_weight": 1,
                            "outlier_weight": 1,
                            "yloss_weight": 0,
                            "use_logits": True,
                            "learning_rate": 0.1,
                            "use_mlmodel": False,
                            "rbf_params": {
                                "train": True,
                                "save": False,
                                "beta": 1,
                                "centers": 8,
                                "epochs": 50,
                                "batch_size": 32,
                                "train_data_rate": 1.0,
                            }
                            }

    def __init__(self,
                 classifier_model=None,
                 rbf=None,
                 hyperparams=None
                 ):
        """Init method

        :param data_interface: an interface class to access data related params.
        :param model_interface: an interface class to access trained ML model.

        """

        checked_hyperparams = merge_default_parameters(
            hyperparams, self._DEFAULT_HYPERPARAMS
        )

        self.use_mlmodel = checked_hyperparams["use_mlmodel"]
        # initializing model related variables
        self._mlmodel = classifier_model
        self.num_features = len(self._mlmodel.feature_input_order)

        ## initiating acdc parameters
        self.rbf = rbf
        self.rbf_params = checked_hyperparams["rbf_params"]
        if rbf is None:
            self.rbf = self.load_or_train_rbf()
        self.budget = None

        # Parameters for loss functions
        self.proximity_weight = checked_hyperparams["proximity_weight"]
        self.yloss_weight = checked_hyperparams["yloss_weight"]
        self.outlier_weight = checked_hyperparams["outlier_weight"]
        self.outlier_weight_init = checked_hyperparams["outlier_weight"]


        self.data_interface = dice_ml.Data(dataframe=self._mlmodel.data.df_train,
                                           continuous_features=self._mlmodel.data.continuous,
                                           outcome_name=self._mlmodel.data.target)

        # initiating data related parameters
        self.minx, self.maxx = self.get_minx_maxx()

        # variables required to generate CFs - see generate_counterfactuals() for more info
        self.cfs = []
        self.features_to_vary = []
        self.cf_init_weights = []  # total_CFs, algorithm, features_to_vary
        self.loss_weights = []  # yloss_type, diversity_loss_type, feature_weights
        self.feature_weights_input = ''
        self.feature_weights_list = tf.constant([1]*len(self._mlmodel.feature_input_order), dtype=tf.float32)
        self.hyperparameters = [1, 1, 1]  # proximity_weight, diversity_weight, categorical_penalty
        self.learning_rate = checked_hyperparams['learning_rate']


        features_to_vary_idx = []
        for i in range(len(self._mlmodel.feature_input_order)):
            if self._mlmodel.feature_input_order[i] not in self._mlmodel.data.immutables:
                features_to_vary_idx.append(i)

        self.do_cf_initializations(features_to_vary_idx)

        # Init KD Tree
        positive_samples = self._mlmodel.data.df_train[self._mlmodel.data.df_train[self._mlmodel.data.target] == 1]
        positive_samples = self._mlmodel.get_ordered_features(positive_samples.copy()).to_numpy()
        self.KDTree = KDTree(positive_samples, leaf_size=2, metric='l1')

    def do_cf_initializations(self, features_to_vary_idx):
        """Intializes CFs and other related variables."""

        # freeze those columns that need to be fixed
        if features_to_vary_idx != self.features_to_vary:
            self.features_to_vary = features_to_vary_idx
        self.freezer = tf.constant([1.0 if ix in self.features_to_vary else 0.0 for ix in range(len(self.minx[0]))])

    def do_loss_initializations(self, yloss_type, feature_weights):
        """Intializes variables related to main loss function"""

        self.loss_weights = [yloss_type, feature_weights]

        # define the loss parts
        self.yloss_type = yloss_type

        # define feature weights
        if feature_weights != self.feature_weights_input:
            self.feature_weights_input = feature_weights
            if feature_weights == "inverse_mad":
                normalized_mads = self.data_interface.get_valid_mads(normalized=True)
                feature_weights = {}
                for feature in normalized_mads:
                    feature_weights[feature] = round(1 / normalized_mads[feature], 2)

            feature_weights_list = []
            for feature in self.data_interface.ohe_encoded_feature_names:
                if feature in feature_weights:
                    feature_weights_list.append(feature_weights[feature])
                else:
                    feature_weights_list.append(1.0)
            self.feature_weights_list = tf.constant([feature_weights_list], dtype=tf.float32)


    def do_optimizer_initializations(self, optimizer, learning_rate):
        """Initializes gradient-based TensorFLow optimizers."""
        # optimizater initialization
        if optimizer == "adam":
            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        elif optimizer == "rmsprop":
            self.optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=learning_rate)

    def compute_dist(self, x_hat, x1):
        """Compute weighted distance between two vectors."""
        return tf.reduce_sum(tf.multiply((tf.abs(x_hat - x1)), self.feature_weights_list))

    def compute_proximity_loss(self):
        """Compute the second part (distance from x1) of the loss function."""
        proximity_loss = 0.0
        proximity_loss += self.compute_dist(self.cfs, self.x1)
        return proximity_loss / self.budget

    def compute_outlier_loss(self):
        score = self.rbf(tf.reshape(self.cfs, (1, self.num_features)))
        return -score

    def compute_yloss(self):
        bce = tf.keras.losses.BinaryCrossentropy()
        return bce([[0.0, 1.0]], self._mlmodel._model(tf.reshape(self.cfs, (1, len(self._mlmodel.feature_input_order)))))

    def compute_loss(self):
        """Computes the overall loss"""
        self.proximity_loss = self.compute_proximity_loss() if self.proximity_weight > 0 else 0.0
        self.outlier_loss = self.compute_outlier_loss() if self.outlier_weight > 0 else 0.0
        self.yloss = self.compute_yloss() if self.yloss_weight > 0 else 0.0

        self.loss = (self.proximity_weight * self.proximity_loss) + (self.outlier_loss * self.outlier_weight)\
                    + (self.yloss_weight * self.yloss)
        return self.loss


    def initialize_CFs(self, query_instance, init_near_query_instance=False):
        """Initialize counterfactuals 2."""
        if self.rbf is not None and not init_near_query_instance:
            ind = tf.argsort(tf.map_fn(
                fn=lambda w: tf.reduce_sum(tf.square(query_instance - tf.reshape(w, tf.shape(query_instance)))),
                elems=self.rbf.weights[1]), axis=0).numpy()[0]
            rbf_weight = self.rbf.weights[1][ind]
            one_init = []
            for i in range(len(rbf_weight)):
                if i in self.features_to_vary:
                    one_init.append(rbf_weight[i])
                else:
                    one_init.append(float(query_instance[i]))
            one_init = np.array(one_init, dtype=np.float32)
            self.cfs = tf.Variable(tf.reshape(one_init, tf.shape(query_instance)), dtype=tf.float32)
        else:
            self.cfs = tf.Variable(query_instance, dtype=tf.float32)


    def round_off_cfs(self, assign=False):
        """function for intermediate projection of CFs."""

        cf = np.reshape(self.cfs.numpy().copy(), (-1, 1)).T

        for i in range(len(self._mlmodel.feature_input_order)):
            if np.abs(cf[0][i] - self.query_instance[i]) < 0.01:
                cf[0][i] = self.query_instance[i]

        if assign:
            self.cfs.assign(cf.flatten())

        if assign:
            return None
        else:
            return cf

    def stop_loop(self, itr, loss_diff):
        """Determines the stopping condition for gradient descent."""

        # do GD for min iterations
        if itr < self.min_iter:
            return False

        # stop GD if max iter is reached
        if itr >= self.max_iter:
            return True

        # else stop when loss diff is small
        if loss_diff <= self.loss_diff_thres:
            self.loss_converge_iter += 1
            if self.loss_converge_iter > self.loss_converge_maxiter:
                return True
            else:
                return False
        else:
            self.loss_converge_iter = 0
            return False


    def find_counterfactuals(self, query_instance, desired_class=1.0,
                             optimizer="adam", min_iter=10, max_iter=100, loss_diff_thres=1e-4,
                             loss_converge_maxiter=1, verbose=False,
                             init_near_query_instance=False):
        """Finds counterfactuals by gradient-descent."""

        self.x1 = tf.constant(query_instance, dtype=tf.float32)

        # find the predicted value of query_instance
        self.target_cf_class = np.array([[desired_class]], dtype=np.float32)

        self.min_iter = min_iter
        self.max_iter = max_iter
        self.loss_diff_thres = loss_diff_thres
        # no. of iterations to wait to confirm that loss has converged
        self.loss_converge_maxiter = loss_converge_maxiter
        self.loss_converge_iter = 0
        self.converged = False

        # running optimization steps
        self.final_cfs = []
        self.initialize_CFs(query_instance, init_near_query_instance)

        # initialize optimizer
        self.do_optimizer_initializations(optimizer, self.learning_rate)

        iterations = 0
        loss_diff = 1.0
        prev_loss = 0.0

        while self.stop_loop(iterations, loss_diff) is False:

            # compute loss and tape the variables history
            with tf.GradientTape() as tape:
                loss_value = self.compute_loss()

            # get gradients
            grads = tape.gradient(loss_value, [self.cfs])

            # freeze features other than features_to_vary
            grads *= self.freezer

            # apply gradients and update the variables
            self.optimizer.apply_gradients(zip(grads, [self.cfs]))

            # projection step
            temp_cf = self.cfs.numpy()
            clip_cf = np.clip(temp_cf, self.minx, self.maxx).astype(np.float32)  # clipping
            # to remove -ve sign before 0.0 in some cases
            clip_cf = np.add(clip_cf,
                             np.zeros([self.minx.shape[1]])).reshape(temp_cf.shape).astype(np.float32)
            self.cfs.assign(clip_cf)

            if verbose:
                if (iterations) % 50 == 0:
                    print('step %d,  loss=%g' % (iterations + 1, loss_value))
                    print(self._mlmodel.predict(self.cfs.numpy().reshape(1,len(self._mlmodel.feature_input_order))))
                    print(self._mlmodel.predict(self.round_off_cfs(assign=False).reshape(1,len(self._mlmodel.feature_input_order))))

            loss_diff = abs(loss_value - prev_loss)
            prev_loss = loss_value
            iterations += 1

        # rounding off final cfs
        self.round_off_cfs(assign=True)

        # max iterations at which GD stopped
        self.max_iterations_run = iterations
        return self.cfs


    def get_counterfactuals(self, factuals: pd.DataFrame):
        factuals = self._mlmodel.get_ordered_features(factuals)

        cfs = []
        for _, factual in factuals.iterrows():
            df = pd.DataFrame([factual.to_list()], columns=factual.index)
            self.query_instance = df.to_numpy().flatten()
            self.budget = self.get_distance_to_nn(np.array(self.query_instance))
            cf = self.find_counterfactuals(query_instance=self.query_instance).numpy()
            for i in range(len(self._mlmodel.feature_input_order)):
                if i not in self.features_to_vary:
                    cf[i] = float(self.query_instance[i])
            cfs.append(cf)
        df_cfs = pd.DataFrame(cfs, columns=self._mlmodel.feature_input_order).astype(np.float32)
        return df_cfs


    def load_or_train_rbf(self):
        if self.rbf_params['train']:
            return self.train_rbf()
        else:
            return self.load_rbf()


    def train_rbf(self):
        positive_examples = self.get_positive_examples().sample(frac=self.rbf_params["train_data_rate"], replace=False, random_state=111)
        rbf = self.build_rbf(positive_examples, self.rbf_params)
        sample_size = min(2000, len(positive_examples)-1)
        input = positive_examples.to_numpy()[:sample_size, :]
        rbf.fit(input, np.ones(input.shape[0]), epochs=self.rbf_params['epochs'], batch_size=self.rbf_params['batch_size'])

        if self.rbf_params['save']:
            full_path = get_full_path(self._mlmodel.data.name, self.rbf_params)
            rbf.save(full_path)
        return rbf


    def load_rbf(self):
        full_path = get_full_path(self._mlmodel.data.name, self.rbf_params)
        rbf = keras.models.load_model(
            full_path, custom_objects={"RBFLayer": RBFLayer}
        )
        return rbf


    def build_rbf(self, positive_examples, params):
        inputs = Input(shape=(self.num_features,))
        x = RBFLayer(output_dim=params['centers'], betas=params['beta'],
                     initializer=InitCentersKMeans(positive_examples))(inputs)
        output = Dense(1, activation='tanh', use_bias=False, kernel_regularizer=regularizers.l2(0.00001),
                       kernel_constraint=constraints.nonneg(), kernel_initializer=initializers.ones())(x)
        rbf = Model(inputs, output)
        adam = tf.keras.optimizers.Adam(lr=0.0005, clipvalue=0.5)
        rbf.compile(optimizer=adam, loss='mse')
        return rbf


    def get_positive_examples(self):
        data = self._mlmodel.data.df_train
        if self.use_mlmodel:
            y_pred = np.array([round(x[0]) for x in self._mlmodel.predict(self._mlmodel.data.df_train.drop(self._mlmodel.data.target, axis=1))])
            positive_predicted_data = data[y_pred == 1].drop(self._mlmodel.data.target, axis=1).reset_index(drop=True)
            return positive_predicted_data
        else:
            return data[data[self._mlmodel.data.target] == 1].drop(self._mlmodel.data.target, axis=1).reset_index(drop=True)


    def get_distance_to_nn(self, query_instance):
        dist, _ = self.KDTree.query(query_instance.reshape(1, -1), k=2)
        for d in dist.flatten():
            if d != 0:
                return d
        return None


    def check_cfs(self):
        if np.round(self._mlmodel.predict(self.cfs.numpy().reshape(1,-1))[0]) == 1:
            return True
        else:
            return False


    def get_minx_maxx(self):
        """Gets the min/max value of features in normalized or de-normalized form."""
        minx = np.array([[0.0] * len(self._mlmodel.feature_input_order)])
        maxx = np.array([[1.0] * len(self._mlmodel.feature_input_order)])
        ranges = self.get_features_range()
        for idx, feature_name in enumerate(self._mlmodel.data.continuous):
            minx[0][idx] = ranges[feature_name][0]
            maxx[0][idx] = ranges[feature_name][1]
        return minx, maxx


    def get_features_range(self):
        ranges = {}
        # Getting default ranges based on the dataset
        for feature_name in self._mlmodel.feature_input_order:
            ranges[feature_name] = [
                self._mlmodel.data.df_train[feature_name].min(), self._mlmodel.data.df_train[feature_name].max()]
        return ranges