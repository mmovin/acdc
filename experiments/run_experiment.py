# flake8: noqa
import os
import re
import tensorflow
from tensorflow import Graph
from tensorflow.compat.v1 import Session

from carla.recourse_methods.catalog.nn.model import NN
from carla.recourse_methods.catalog.roar.model import ROAR
from carla.recourse_methods.own.acdc2 import ACDC2

from carla import log
from carla.recourse_methods.own.proto import Proto

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings

import pandas as pd

warnings.simplefilter(action="ignore", category=FutureWarning)

import argparse
from typing import Dict, Optional

import pickle
import numpy as np
import yaml

import carla.evaluation.catalog as evaluation_catalog
from carla.data.api import Data
from carla.data.catalog import OnlineCatalog
from carla.evaluation import Benchmark
from carla.models.api import MLModel
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances
from carla.recourse_methods import *
from carla.recourse_methods.api import RecourseMethod

tensorflow.compat.v1.enable_eager_execution()


## Added by Movin et al to evaluate consistency over other trained models.
def get_success_on_retrained(benchmark, backend, dataset, num_models=20, model_type=['ann']):
    successes = pd.DataFrame()
    cfs = benchmark._counterfactuals.dropna()
    if len(cfs) == 0:
        for m in model_type:
            successes[m] = [0] * num_models
        return successes
    for m in model_type:
        success_per_model = []
        if m == 'forest':
            eval_backend = 'sklearn'
        elif backend == 'sklearn' or backend == 'xgboost':
            eval_backend = 'tensorflow'
        else:
            eval_backend = backend
        for i in range(num_models):
            model_i = MLModelCatalog(dataset, model_type=m, backend=eval_backend, model_number=i)
            if model_i.data.target in cfs.columns:
                cfs = cfs.drop(model_i.data.target, axis=1)
            predictions = model_i.predict(cfs).flatten()
            success = [round(x) for x in predictions]
            success_per_model.append(success)
        successes[m] = success_per_model
    return successes


def save_cfs(result: dict, alt_path: Optional[str], data_name: Optional[str]) -> None:
    data_name = "" if data_name is None else data_name
    data_home = os.environ.get("CF_DATA", os.path.join("~", "carla", "results", data_name))

    data_home = os.path.expanduser(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    path = os.path.join(data_home, "cfs.pkl") if alt_path is None else alt_path
    file = open(path, "wb")
    pickle.dump(result, file)
    file.close()


def save_result(result: pd.DataFrame, alt_path: Optional[str], data_name: Optional[str]) -> None:
    data_name = "" if data_name is None else data_name
    data_home = os.environ.get("CF_DATA", os.path.join("~", "carla", "results", data_name))

    data_home = os.path.expanduser(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    path = os.path.join(data_home, "results.csv") if alt_path is None else alt_path

    result.to_csv(path, index=False)


def load_setup() -> Dict:
    with open("experimental_setup.yaml", "r") as f:
        setup_catalog = yaml.safe_load(f)

    return setup_catalog["recourse_methods"]


def initialize_recourse_method(
    method: str,
    mlmodel: MLModel,
    data: Data,
    data_name: str,
    model_type: str,
    setup: Dict,
    fraction: float = 1.0,
    sess: Session = None,
    predicted: bool = False
) -> RecourseMethod:

    if method == 'acdc':
        short_dataset_name = re.sub(r'_\d', '', data_name)
        key = method + '_' + short_dataset_name
        print("key: ", key)
    else:
        key = method

    if key not in setup.keys():
        raise KeyError("Method not in experimental setup")

    hyperparams = setup[key]["hyperparams"]

    latent_dim = round(len(mlmodel.feature_input_order) / 3)
    latent_dim = 1 if latent_dim == 0 else latent_dim

    if method == "ar":
        coeffs, intercepts = None, None
        if model_type == "linear":
            # get weights and bias of linear layer for negative class 0
            coeffs_neg = mlmodel.raw_model.layers[0].get_weights()[0][:, 0]
            intercepts_neg = np.array(mlmodel.raw_model.layers[0].get_weights()[1][0])

            # get weights and bias of linear layer for positive class 1
            coeffs_pos = mlmodel.raw_model.layers[0].get_weights()[0][:, 1]
            intercepts_pos = np.array(mlmodel.raw_model.layers[0].get_weights()[1][1])

            coeffs = -(coeffs_neg - coeffs_pos)
            intercepts = -(intercepts_neg - intercepts_pos)

        ar = ActionableRecourse(mlmodel, hyperparams, coeffs, intercepts)
        act_set = ar.action_set

        # some datasets need special configuration for possible actions
        if data_name == "give_me_some_credit":
            act_set["NumberOfTimes90DaysLate"].mutable = False
            act_set["NumberOfTimes90DaysLate"].actionable = False
            act_set["NumberOfTime60-89DaysPastDueNotWorse"].mutable = False
            act_set["NumberOfTime60-89DaysPastDueNotWorse"].actionable = False

        ar.action_set = act_set

        return ar
    elif "cem" in method:
        hyperparams["data_name"] = data_name
        return CEM(sess, mlmodel, hyperparams)
    elif "proto" in method:
        return Proto(sess, mlmodel, hyperparams)
    elif method == "clue":
        hyperparams["data_name"] = data_name
        hyperparams["latent_dim"] = latent_dim
        return Clue(data, mlmodel, hyperparams)

    elif method == "acdc":
        hyperparams["use_mlmodel"] = predicted
        hyperparams["rbf_params"]["train_data_rate"] = fraction
        return ACDC2(mlmodel, hyperparams=hyperparams)
    elif method == "dice":
        return Dice(mlmodel, hyperparams)
    elif "face" in method:
        return Face(mlmodel, hyperparams)
    elif method == "gs":
        return GrowingSpheres(mlmodel)
    elif method == "cchvae":
        hyperparams["data_name"] = data_name
        # variable input layer dimension is first time here available
        hyperparams["vae_params"]["layers"] = [
            len(mlmodel.feature_input_order)
        ] + hyperparams["vae_params"]["layers"] + [latent_dim]
        return CCHVAE(mlmodel, hyperparams)
    elif method == "revise":
        hyperparams["data_name"] = data_name
        # variable input layer dimension is first time here available
        hyperparams["vae_params"]["layers"] = [
            len(mlmodel.feature_input_order)
        ] + hyperparams["vae_params"]["layers"] + [latent_dim]
        return Revise(mlmodel, data, hyperparams)
    elif "wachter" in method:
        return Wachter(mlmodel, hyperparams)
    elif "nn" in method:
        return NN(mlmodel, hyperparams)
    elif "crud" in method:
        hyperparams["data_name"] = data_name
        if data_name == 'mnist':
            latent_dim = 32
        else:
            latent_dim = 8
        hyperparams["vae_params"]["layers"] = [len(mlmodel.feature_input_order)]\
                                              + hyperparams["vae_params"]["layers"]\
                                              + [latent_dim]
        return CRUD(mlmodel, hyperparams)
    elif "roar" in method:
        return ROAR(mlmodel, hyperparams)
    else:
        raise ValueError("Recourse method not known")


parser = argparse.ArgumentParser(description="Run experiments from paper")
parser.add_argument(
    "-d",
    "--dataset",
    nargs="*",
    default=["give_me_some_credit", "synthetic", "spotify_classic", "breast_cancer", "mnist"],
    choices=[
        "give_me_some_credit",
        "synthetic",
        "spotify_classic",
        "breast_cancer",
        "mnist"],
    help="Datasets for experiment",
)
parser.add_argument(
    "-t",
    "--type",
    nargs="*",
    default=["ann"],
    choices=["ann", "linear", "forest"],
    help="Model type for experiment",
)
parser.add_argument(
    "-r",
    "--recourse_method",
    nargs="*",
    default=[
        "dice",
        "cchvae",
        "crud",
        "proto",
        "roar",
        "wachter",
        "gs",
        "acdc"
    ],
    choices=[
        "dice",
        "ar",
        "cchvae",
        "cem",
        "cem-vae",
        "clue",
        "face_knn",
        "face_epsilon",
        "gs",
        "revise",
        "wachter",
        "nn",
        "acdc",
        "crud",
        "cchvae",
        "proto",
        "roar"
    ],
    help="Recourse methods for experiment",
)
parser.add_argument(
    "-n",
    "--number_of_samples",
    type=int,
    default=100,
    help="Number of instances per dataset",
)
parser.add_argument(
    "-p",
    "--path",
    type=str,
    default=None,
    help="Save path for the output csv. If None, the output is written to the cache.",
)
parser.add_argument(
    "--retrained",
    default=False,
    action='store_true'
)

parser.add_argument(
    "--predicted",
    default=False,
    action='store_true'
)

parser.add_argument(
    "--fraction_of_train",
    default=False,
    action='store_true'
)
parser.add_argument(
    "-m",
    "--number_of_models",
    type=int,
    default=20,
    help="Number of models to evaluate per dataset",
)

parser.add_argument(
    "-e",
    "--eval_type",
    nargs="*",
    default=["ann", "linear", "forest"],
    choices=["ann", "linear", "forest"],
    help="Model type for experiment",
)

args = parser.parse_args()
setup = load_setup()

path = args.path
retrained = args.retrained

datasets = args.dataset
torch_methods = ["clue", "wachter", "revise", "crud", "cchvae"]

fractions = [1.0]
session_models = ["cem", "cem-vae", "proto"]
torch_methods = ["cchvae", "clue", "crud", "wachter", "revise"]
for data_name in args.dataset:
    dataset = OnlineCatalog(data_name)
    cfs = {}
    results = pd.DataFrame()
    rms = ""
    types = ""
    for type in args.type:
        types = types + "_" + type
    types = str(types[1:])
    for rm in args.recourse_method:
        rms = rms + '_' + rm
        backend = "tensorflow"
        if rm in torch_methods:
            backend = "pytorch"
        for model_type in args.type:
            log.info("=====================================")
            log.info("Recourse method: {}".format(rm))
            log.info("Dataset: {}".format(data_name))
            log.info("Model type: {}".format(model_type))
            if model_type == 'forest':
                backend = 'sklearn'
            if rm in session_models:
                graph = Graph()
                with graph.as_default():
                    ann_sess = Session()
                    with ann_sess.as_default():
                        mlmodel_sess = MLModelCatalog(dataset, model_type, backend)

                        factuals_sess = predict_negative_instances(
                            mlmodel_sess, dataset.df_test
                        )
                        factuals_sess = factuals_sess.iloc[: args.number_of_samples]
                        factuals_sess = factuals_sess.reset_index(drop=True)

                        recourse_method_sess = initialize_recourse_method(
                            rm,
                            mlmodel_sess,
                            dataset,
                            data_name,
                            model_type,
                            setup,
                            sess=ann_sess,
                        )

                        benchmark = Benchmark(
                            mlmodel_sess, recourse_method_sess, factuals_sess
                        )
                        if retrained:
                            successes = get_success_on_retrained(benchmark, backend, dataset,
                                                                 num_models=args.number_of_models,
                                                                 model_type=args.eval_type)
                        evaluation_measures = [
                            evaluation_catalog.YNN(benchmark.mlmodel, {"y": 5, "cf_label": 1}),
                            evaluation_catalog.Distance(benchmark.mlmodel),
                            evaluation_catalog.SuccessRate(),
                            evaluation_catalog.Redundancy(benchmark.mlmodel, {"cf_label": 1}),
                            evaluation_catalog.ConstraintViolation(benchmark.mlmodel),
                            evaluation_catalog.AvgTime({"time": benchmark.timer}),
                            evaluation_catalog.Lof(benchmark.mlmodel)
                        ]
                        df_benchmark = benchmark.run_benchmark(evaluation_measures)
                        df_benchmark["Recourse_Method"] = rm
                        df_benchmark["Dataset"] = data_name
                        df_benchmark["ML_Model"] = model_type

                        df_benchmark = df_benchmark[
                            [
                                "Recourse_Method",
                                "Dataset",
                                "ML_Model",
                                "L0_distance",
                                "L1_distance",
                                "L2_distance",
                                "Linf_distance",
                                "Constraint_Violation",
                                "Redundancy",
                                "y-Nearest-Neighbours",
                                "LOF",
                                "Success_Rate",
                                "avg_time",
                            ]
                        ]
                        if retrained:
                            for m in args.eval_type:
                                for i in range(args.number_of_models):
                                    df_benchmark["success_model_{}_{}".format(m, i)] = successes[m].iloc[i]
                        cfs[rm] = benchmark._counterfactuals.to_numpy()
                        results = pd.concat([results, df_benchmark], axis=0)
                        print("=====================================")

            else:
                results_fractions = pd.DataFrame()
                if rm == 'acdc' and args.fraction_of_train:
                    fractions = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
                for fraction in fractions:
                    mlmodel = MLModelCatalog(dataset, model_type, backend)

                    factuals = predict_negative_instances(mlmodel, dataset.df_test)
                    factuals = factuals.iloc[: args.number_of_samples]
                    factuals = factuals.reset_index(drop=True)

                    recourse_method = initialize_recourse_method(
                        rm, mlmodel, dataset, data_name, model_type, setup, fraction=fraction, predicted=args.predicted
                    )

                    benchmark = Benchmark(
                        mlmodel, recourse_method, factuals
                    )
                    if retrained:
                        successes = get_success_on_retrained(benchmark, backend, dataset, num_models=args.number_of_models, model_type=args.eval_type)

                    evaluation_measures = [
                        evaluation_catalog.YNN(benchmark.mlmodel, {"y": 5, "cf_label": 1}),
                        evaluation_catalog.Distance(benchmark.mlmodel),
                        evaluation_catalog.SuccessRate(),
                        evaluation_catalog.Redundancy(benchmark.mlmodel, {"cf_label": 1}),
                        evaluation_catalog.ConstraintViolation(benchmark.mlmodel),
                        evaluation_catalog.AvgTime({"time": benchmark.timer}),
                        evaluation_catalog.Lof(benchmark.mlmodel)
                    ]
                    df_benchmark = benchmark.run_benchmark(evaluation_measures)
                    if retrained:
                        for m in args.eval_type:
                            for i in range(args.number_of_models):
                                df_benchmark["success_model_{}_{}".format(m, i)] = successes[m].iloc[i]
                    df_benchmark["Fraction"] = fraction
                    results_fractions = pd.concat([results_fractions, df_benchmark])
                df_benchmark = results_fractions

                df_benchmark["Recourse_Method"] = rm
                df_benchmark["Dataset"] = data_name
                df_benchmark["ML_Model"] = model_type

                ## Added by Movin et al to evaluate consistency
                models = []
                for m in args.eval_type:
                    for i in range(args.number_of_models):
                        models.append("success_model_{}_{}".format(m, i))

                df_benchmark = df_benchmark[
                    [
                        "Recourse_Method",
                        "Dataset",
                        "ML_Model",
                        "L0_distance",
                        "L1_distance",
                        "L2_distance",
                        "Linf_distance",
                        "Constraint_Violation",
                        "Redundancy",
                        "y-Nearest-Neighbours",
                        "LOF",
                        "Success_Rate",
                        "avg_time",
                        "Fraction",
                    ] + models
                ]

                cfs[rm] = benchmark._counterfactuals.to_numpy()
                results = pd.concat([results, df_benchmark], axis=0)
                print("=====================================")

    if retrained:
        rms = rms + '_' + 'retrained'
    if args.predicted and "acdc" in rms:
        rms = rms + '_' + 'predicted'
    if args.fraction_of_train and "acdc" in rms:
        rms = rms + '_' + 'fraction'

    rms = str(rms[1:])
    save_result(results, path, data_name=data_name + '/' + types + '/' + rms)
    save_cfs(cfs, path, data_name=data_name + '/' + types + '/' + rms)
