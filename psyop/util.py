import numpy as np

def get_rng(seed: int | np.random.Generator | None) -> np.random.Generator:
    """ Get a numpy random number generator from a seed. """
    if isinstance(seed, np.random.Generator):
        rng = seed
    elif isinstance(seed, int):
        rng = np.random.default_rng(seed)
    elif seed is None:
        rng = np.random.default_rng()
    else:
        raise TypeError(
            f"seed must be None, int, or numpy.random.Generator, not {type(seed)}"
        )

    return rng