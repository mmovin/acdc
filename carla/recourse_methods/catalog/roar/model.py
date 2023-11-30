"""
Class to implement RObust Algorithmic Recourse (ROAR) from Upadhyay, S.; Joshi, S.; and Lakkaraju, H. 2021. Towards
robust and reliable algorithmic recourse. Advances in Neural
Information Processing Systems, 34: 16926–16937.

Code was implemented as a Tensorflow version of the code provided by Upadhyay et al. https://github.com/AI4LIFE-GROUP/ROAR
and inspired by (some functions copied from) Dice-Ml: https://github.com/interpretml/DiCE
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.optimize import linprog
from sklearn.linear_model import LogisticRegression

import lime
from carla import RecourseMethod
from carla.recourse_methods.processing import merge_default_parameters


class ROAR(RecourseMethod):
    _DEFAULT_HYPERPARAMS = {"delta_max": 0.1,
                            "lambda": 0.1,
                            "learning_rate": 0.1,
                            "model_loss_weight": 1
                            }

    def __init__(self,
                 mlmodel=None,
                 hyperparams=None,
                 model_type="ann"
                 ):
        """Init method

        :param data_interface: an interface class to access data related params.
        :param model_interface: an interface class to access trained ML model.

        """

        checked_hyperparams = merge_default_parameters(
            hyperparams, self._DEFAULT_HYPERPARAMS
        )
        # initializing model related variables
        self._mlmodel = mlmodel
        self.num_features = len(self._mlmodel.feature_input_order)

        # Parameters for loss functions
        self.delta_max = checked_hyperparams["delta_max"]
        self.lambda_value = checked_hyperparams["lambda"]
        self.model_loss_weight = checked_hyperparams["model_loss_weight"]
        self.W = None
        self.W0 = None

        # initiating data related parameters
        self.minx, self.maxx = self.get_minx_maxx()

        # variables required to generate CFs - see generate_counterfactuals() for more info
        self.cfs = []
        self.features_to_vary = []
        self.cf_init_weights = []  # total_CFs, algorithm, features_to_vary
        self.loss_weights = []
        self.feature_weights_input = ''
        self.feature_weights_list = tf.constant([1] * len(self._mlmodel.feature_input_order), dtype=tf.float32)
        self.learning_rate = checked_hyperparams['learning_rate']

        features_to_vary_idx = []
        for i in range(len(self._mlmodel.feature_input_order)):
            if self._mlmodel.feature_input_order[i] not in self._mlmodel.data.immutables:
                features_to_vary_idx.append(i)

        self.do_cf_initializations(features_to_vary_idx)
        self.model_type = model_type

        ## LIME libary with small changes to get the weights of the linear model in the same order as the features.
        ## Lime from https://github.com/AI4LIFE-GROUP/ROAR is used.
        self.lime = lime.lime_tabular.LimeTabularExplainer(self._mlmodel.data.df_train[self._mlmodel.data.continuous],
                                                           feature_names=self._mlmodel.data.continuous,
                                                           class_names=self._mlmodel.data.target,
                                                           discretize_continuous=False, feature_selection='none',
                                                           sample_around_instance=False)

    def do_cf_initializations(self, features_to_vary_idx):
        """Intializes CFs and other related variables."""

        # freeze those columns that need to be fixed
        if features_to_vary_idx != self.features_to_vary:
            self.features_to_vary = features_to_vary_idx
        self.freezer = tf.constant([1.0 if ix in self.features_to_vary else 0.0 for ix in range(len(self.minx[0]))])

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
        return proximity_loss

    def compute_robust_model_loss(self):
        BCE = tf.keras.losses.BinaryCrossentropy()

        coefs = self.original_coef + self.delta_weights
        intercept = self.original_intercept + self.delta_intercept
        pred = tf.nn.sigmoid(tf.multiply(coefs, self.cfs) + intercept)
        return BCE(1.0, pred)

    def compute_loss(self):
        """Computes the overall loss"""
        self.proximity_loss = self.compute_proximity_loss() if self.lambda_value > 0 else 0.0
        self.model_loss = self.compute_robust_model_loss() if self.model_loss_weight > 0 else 0.0
        self.loss = (self.lambda_value * self.proximity_loss) + self.model_loss * self.model_loss_weight
        return self.loss

    def initialize_CFs(self, query_instance):
        """Initialize counterfactuals 2."""
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
                             optimizer="adam", min_iter=5, max_iter=100, loss_diff_thres=1e-4,
                             loss_converge_maxiter=1, verbose=False):
        """Finds counterfactuals by gradient-descent."""

        self.x1 = tf.constant(query_instance, dtype=tf.float32)

        self.delta_intercept = tf.Variable(tf.zeros_like(self.original_intercept), dtype=tf.float32)
        self.delta_weights = tf.Variable(tf.zeros_like(self.original_coef, dtype=tf.float32))
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

        self.initialize_CFs(query_instance)

        # initialize optimizer
        self.do_optimizer_initializations(optimizer, self.learning_rate)

        iterations = 0
        loss_diff = 1.0
        prev_loss = 0.0

        while self.stop_loop(iterations, loss_diff) is False:
            self.delta_weights, self.delta_intercept = self.find_delta_linprog()

            loss = self.update_cfs(verbose=verbose)
            loss_diff = abs(loss - prev_loss)
            prev_loss = loss
            iterations += 1

        # max iterations at which GD stopped
        self.max_iterations_run = iterations
        return self.cfs

    def get_counterfactuals(self, factuals: pd.DataFrame):
        factuals = self._mlmodel.get_ordered_features(factuals)
        cfs = []
        for _, factual in factuals.iterrows():
            df = pd.DataFrame([factual.to_list()], columns=factual.index)
            self.query_instance = df.to_numpy().flatten()
            if self.model_type == 'linear':
                self._delta_model = self._mlmodel.raw_model
            else:
                self._delta_model = LogisticRegression(max_iter=1000)
            if self.model_type != "linear":
                exp = self.lime.explain_instance(data_row=self.query_instance,
                                                 predict_fn=self._mlmodel.predict_proba,
                                                 model_regressor=self._delta_model)
            self.original_coef = tf.constant(self._delta_model.coef_.copy(), dtype=tf.float32)
            self.original_intercept = tf.constant(self._delta_model.intercept_.copy(), dtype=tf.float32)
            cf = self.find_counterfactuals(query_instance=self.query_instance).numpy()
            for i in range(len(self._mlmodel.feature_input_order)):
                if i not in self.features_to_vary:
                    cf[i] = float(self.query_instance[i])
            cfs.append(cf)
        df_cfs = pd.DataFrame(cfs, columns=self._mlmodel.feature_input_order).astype(np.float32)
        return df_cfs

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

    def compute_worst_model_loss(self, cfs_copy):
        pred = tf.nn.sigmoid(tf.multiply(self.W, cfs_copy))
        BCE = tf.keras.losses.BinaryCrossentropy()
        bce = BCE(1.0, pred)
        return bce

    def find_delta_linprog(self):

        self.W = tf.Variable(tf.concat([self.original_coef, [self.original_intercept]], 1))
        cfs_copy = tf.constant(tf.concat([tf.constant(self.cfs.numpy().copy()), [1]], 0))

        with tf.GradientTape() as tape1:
            model_loss_value = self.compute_worst_model_loss(cfs_copy)

        gradient_w_loss = tape1.gradient(model_loss_value, [self.W])

        A_eq = np.empty((0, self.W.shape[1]), float)

        b_eq = np.array([])

        c = list(np.array(gradient_w_loss) * np.array([-1] * len(gradient_w_loss)))
        bound = (-self.delta_max, self.delta_max)
        bounds = [bound] * len(gradient_w_loss)

        res = linprog(c, bounds=bounds, A_eq=A_eq, b_eq=b_eq, method='simplex')
        delta_opt = res.x  # the delta value that maximizes the function
        delta_W, delta_W0 = np.array(delta_opt[:-1]), np.array([delta_opt[-1]])
        return delta_W, delta_W0

    def update_cfs(self, verbose=False):

        for i in range(100):
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

            if (i) % 50 == 0 and verbose:
                print('step %d,  loss=%g' % (i + 1, loss_value))
                print(grads)
                print(self._mlmodel.predict(self.cfs.numpy().reshape(1, len(self._mlmodel.feature_input_order))))

            return loss_value
