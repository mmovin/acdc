import os


def get_home(models_home=None):
    """Return a path to the cache directory for trained autoencoders.

    This directory is then used by :func:`save`.

    If the ``models_home`` argument is not specified, it tries to read from the
    ``CF_MODELS`` environment variable and defaults to ``~/cf-bechmark/models``.

    """

    if models_home is None:
        models_home = os.environ.get(
            "CF_MODELS", os.path.join("~", "carla", "models", "rbfs")
        )

    models_home = os.path.expanduser(models_home)
    if not os.path.exists(models_home):
        os.makedirs(models_home)

    return models_home

def get_full_path(name, params):

    path = get_home()
    dir = "{}_{}_{}_{}".format(name, params['centers'], params['beta'], params['epochs'])
    full_path = path + "/" + dir

    return full_path