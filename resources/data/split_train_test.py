import argparse
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_SEED = 111

parser = argparse.ArgumentParser(description="Run experiments from paper")
parser.add_argument(
    "-d",
    "--dataset",
    nargs="*",
    default=["give_me_some_credit", "spotify_classic", "breast_cancer"],
    choices=[
        "give_me_some_credit",
        "spotify_classic",
        "breast_cancer",
        "mnist"],
    help="Datasets for experiment",
)

args = parser.parse_args()

data_home = os.environ.get("CF_DATA", os.path.join("~", "cf-data"))

data_home = os.path.expanduser(data_home)
if not os.path.exists(data_home):
    os.makedirs(data_home)
for dataset in args.dataset:
    path = data_home + '/' + dataset + '.csv'
    df = pd.read_csv(path)

    df_train, df_test = train_test_split(df, random_state=RANDOM_SEED)

    df_train.reset_index(inplace=True, drop=True)
    df_test.reset_index(inplace=True, drop=True)

    ## Save the data:
    data_directory = str(datetime.date(datetime.now()))
    Path(data_directory).mkdir(parents=True, exist_ok=True)

    df_train.to_csv('{}/{}_train.csv'.format(data_directory, dataset), index=False)
    df_test.to_csv('{}/{}_test.csv'.format(data_directory, dataset), index=False)