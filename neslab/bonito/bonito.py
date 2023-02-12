import time
import math

import h5py
import numpy as np
import click
import matplotlib.pyplot as plt

from find import Find

from neslab.bonito.distributions import inverse_joint_cdf
from neslab.bonito import NormalDistribution
from neslab.bonito import ExponentialDistribution
from neslab.bonito import GaussianMixtureModel

model_map = {"norm": NormalDistribution, "exp": ExponentialDistribution, "gmm": GaussianMixtureModel}

def Bonito(tcharges: tuple, dist_models: tuple, slot_length: float, target_probability: float = 0.99, initial_offset: float = 0, max_offset: float = 848e-6, path: str = "opt_scale.csv"):
    """Bonito connection protocol.

    With every new observation, each node updates a model of their charging time distribution.
    They exchange the model of their charging times and select a common connection interval according to the inverse joint cdf.
    If the connection interval is greater than the current charging time of both devices, the encounter is considered a success.

    Args:
        tcharges (tuple): tuple of 1-d arrays with charging time observations
        dist_models (tuple): tuple of classes of charging time distribution
        slot_length (float): Node activity time in seconds
        target_probability (float, optional): Defaults to 0.99.
        initial_offset (float, optional): Initial offset between the two nodes (in seconds). Defaults to 0.
        max_offset (float, optional): Defaults to 848e-6 (in seconds)
        path (str, optional): Defaults to "opt_scale.csv"
    Yields:
        resulting connection interval and boolean specifying successful or failed encounter
    """
    dist1 = dist_models[0]()
    dist2 = dist_models[1]()

    t1 = initial_offset
    t2 = 0

    prev_c1 = 0.7
    prev_c2 = 0.7

    for c1, c2 in zip(*tcharges):
        if abs(t1 - t2) <= max_offset:
            conn_int = inverse_joint_cdf((dist1, dist2), target_probability)
            
            if (c1 <= conn_int) and (c2 <= conn_int):
                t1 += conn_int
                t2 += conn_int
                yield conn_int, True

            else:
                # print(f"Bonito Failure: t1: {t1}, t2: {t2}, conn:{conn_int}, c1:{c1}, c2:{c2}")
                print(f"Bonito Failure: conn:{conn_int}, c1:{c1}, c2:{c2}")
                # time.sleep(0.1)

                t1 += max(conn_int, c1)
                t2 += max(conn_int, c2)
                yield conn_int, False

        else:
            print(f"Find Failure: t1: {t1}, t2: {t2}")
            # time.sleep(0.1)

            w1 = Find(path, math.ceil(prev_c1/slot_length))
            w2 = Find(path, math.ceil(prev_c2/slot_length))

            t1 += c1 + (w1 * slot_length)     
            t2 += c2 + (w2 * slot_length)
            yield conn_int, False

        dist1.sgd_update(c1)
        dist2.sgd_update(c2)

        prev_c1 = c1     
        prev_c2 = c2     

def run_protocol(grp_pair: h5py.Dataset, protocol: str, slot_length, bonito_target_probability=0.99, offset=0):
    tchrg_node0 = grp_pair["node0"][:]
    tchrg_node1 = grp_pair["node1"][:]

    if protocol == "bonito":
        dist_cls_node0 = model_map[grp_pair["node0"].attrs["model"]]
        dist_cls_node1 = model_map[grp_pair["node1"].attrs["model"]]
        pro_gen = Bonito((tchrg_node0, tchrg_node1), (dist_cls_node0, dist_cls_node1), slot_length, bonito_target_probability, offset)
    else:
        raise NotImplementedError

    connection_intervals = np.empty((len(grp_pair["time"]),))

    for i, (ci, success) in enumerate(pro_gen):
        if success:
            # print("Success")
            connection_intervals[i] = ci
        else:
            # print("Failure")
            connection_intervals[i] = np.nan

    return connection_intervals

@click.command()
@click.option("--input", "-i", "input_path", type=click.Path(exists=True, dir_okay=False), required=True, help="path to hdf file with charging time traces")
@click.option("--pair", "-p", type=(int, int), help="pair of nodes")
@click.option("--slot-length", "-s", type=float, help="Slot length (node activity time) (in seconds)", default=0.001)
@click.option("--target-probability", "-t", type=float, help="Bonito target probability", default=0.99)
@click.option("--offset", "-o", type=float, help="initial offset between the two nodes (in seconds)", default=0)
def cli(input_path, pair, slot_length, target_probability, offset):
    with h5py.File(input_path, "r") as hf:
        try:
            grp = hf[str(pair)]
        except KeyError:
            grp = hf[str(tuple(reversed(pair)))]

        f, ax = plt.subplots()
        ax.plot(grp["time"][:], grp["node0"][:], color="gray", linestyle="--", label="charging time node0")
        ax.plot(grp["time"][:], grp["node1"][:], color="gray", linestyle="-.", label="charging time node1")

        cis = run_protocol(grp, "bonito", slot_length, target_probability, offset)
        ax.plot(grp["time"][:], cis, label=f"connection interval bonito")
        success_rate = np.count_nonzero(~np.isnan(cis)) / len(grp["time"])
        delay = np.nanmedian(cis)
        print(f"bonito: success_rate={success_rate*100:.2f}% delay={delay:.3f}s")

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Time [s]")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    cli()