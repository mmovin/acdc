from carla import MLModelCatalog
from carla.data.catalog import OnlineCatalog

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

datasets = ['synthetic', 'breast_cancer', 'give_me_some_credit', 'mnist', 'spotify_classic']
num_models = 20
model_type = 'forest'
backends = ['sklearn', 'xgboost']

epochs = 40
lr = 0.01
batch_size = 32


for data_name in datasets:
    for backend in backends:
        for num in range(num_models):
            data = OnlineCatalog(data_name)
            model = MLModelCatalog(data, model_type, backend, load_online=False, model_number=num)
            model.train(learning_rate=lr, epochs=epochs, batch_size=batch_size, force_train=True)