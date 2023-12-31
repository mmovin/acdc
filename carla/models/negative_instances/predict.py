from typing import Any

import numpy as np
import pandas as pd
from carla import Data


def predict_negative_instances(model: Any, data: pd.DataFrame) -> pd.DataFrame:

    """Predicts the data target and retrieves the negative instances. (H^-)

    Assumption: Positive class label is at position 1

    Parameters
    ----------
    name : Tensorflow or PyTorch Model
        Model object retrieved by :func:`load_model`
    data : pd.DataFrame
        Dataset used for predictions
    Returns
    -------
    df :  data.api Data() class with negative predicted instances
    """
    df = data.copy()
    df["yy"] = predict_label(model, df)
    df = df[df["yy"] == 0]
    df = df.drop("yy", axis="columns")
    return df



def predict_label(model: Any, df: pd.DataFrame, as_prob: bool = False) -> np.ndarray:
    """Predicts the data target

    Assumption: Positive class label is at position 1

    Parameters
    ----------
    name : Tensorflow or PyTorch Model
        Model object retrieved by :func:`load_model`
    df : pd.DataFrame
        Dataset used for predictions
    Returns
    -------
    predictions :  2d numpy array with predictions
    """
    predictions = model.predict(df)

    if not as_prob:
        predictions = predictions.round()
    return predictions
