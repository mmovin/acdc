# Code to reproduce results from Consistent Counterfactual Examples via Anomlay Control and Data Coherence

## To note

Most of this code is directly taken from the open sourced Counterfactual and Recourse Library CARLA, for which you can
find extensive documentation [here](https://carla-counterfactual-and-recourse-library.readthedocs.io/en/latest/)

CARLA is also presented in this [paper](https://arxiv.org/pdf/2108.00783.pdf).

Pls also see `README_CARLA.md` for more information about the library.

We thank the authors of CARLA for providing this open sourced library, allowing for easy experimentation. 
## Added by Movin et al:

* recourse_mehtods -> catalog -> own 
  * Implementation of ACDC (our method) and [PROTO](https://github.com/SeldonIO/alibi/blob/master/alibi/explainers/cfproto.py)
* recourse_mehtods -> catalog -> roar
  * Implementation of [ROAR](https://github.com/AI4LIFE-GROUP/ROAR)
* train_models 
  * Code for training the predictive models
* evaluation -> catalog -> lof.py, violations.py
  * Metrics for plausibility violation and Local outlier factor
* Changes in run_experiment.py to include metrics on consistency
* recourses -> data 
  * Code to create datasets
  * Datasets used in this work

* Notebooks for generating results 

* LIME is added with small modifications from the library [lime](https://github.com/marcotcr/lime) to make ROAR to work, as described in https://github.com/AI4LIFE-GROUP/ROAR

Hyperparameters are set in `experimental_setup.yaml` and the benchmark is run by running `run_experiments.py`


