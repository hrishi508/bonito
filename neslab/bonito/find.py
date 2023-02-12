import math
import numpy as np
import pandas as pd

from scipy.optimize import minimize_scalar

from neslab.find import distributions
from neslab.find import Model

np.seterr(divide='ignore')

def objective(scale, t_chr):
    m = Model(scale, "Geometric", t_chr, n_slots=t_chr * 20000)
    return m.disco_latency()


def optimize_scale(t_chr):
    scale_range = distributions.Geometric.get_scale_range(t_chr)
    res = minimize_scalar(
        objective,
        bounds=scale_range,
        method="bounded",
        args=(t_chr),
    )

    return res.x

def process_csv(path: str):
    """Read the csv and return the optimized scales as an array

    Args:
        path (str): absolute path to the optimized scales csv file

    Returns:
        np.ndarray: optimized scales of geometric distro
    """
    df = pd.read_csv(path, index_col="t_chr")
    df.sort_index(inplace=True)
    df = df.reindex(np.arange(10, df.index[-1] - 30, 10))
    df.interpolate(inplace=True)

    table = df["x_opt"].values.astype(np.float32)

    return table

def lookup_scale(t_chr: int, table: np.ndarray):
    """Returns the optimized scale for the given charging time

    Args:
        t_chr (int): Latest charging time (in slots)
        table (np.ndarray): Table with optimized scale of geometric distro

    Returns:
        float: optimized scale
    """
    if t_chr < 10: return table[0]
    elif t_chr > 2560: return table[255]

    idx_low = int(t_chr/10 - 1)
    val_low = table[idx_low]
    val_high = table[int(t_chr/10)]
    frac = (t_chr % 10)/10

    return val_low + frac * (val_high - val_low)

def geometric_itf_sample(p: float):
    """Return the delay value sampled from geometric distro

    Args:
        p (float): optimized scale for geometric distro

    Returns:
        int: randomly sampled delay
    """
    y = np.random.uniform()
    res = int(math.log(1 - y)/math.log(1 - p) - 1)
    # res = int(math.log(y)/math.log(1 - p)) + 1
    # res = np.random.geometric(p)

    return res

def Find(path: str, t_chr: int):
    """Calculate the random waiting time given the latest current charging time

    Args:
        path (str): absolute path to the optimized scales csv file
        t_chr (int): Latest charging time (in slots)

    Returns:
        int: waiting time (in slots)
    """

    # Either use the lookup table or run the optimize the scale dynamically
    
    # Lookup table
    table = process_csv(path)
    wait_time = geometric_itf_sample(lookup_scale(t_chr, table))
    
    # Dynamic optimization
    # wait_time = geometric_itf_sample(optimize_scale(t_chr))


    return wait_time
