from glm.spk_train import Spike_train as SPK
from mle.inference import Maximum_likelihood_estimator as MLE
import pickle
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np


data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
fig_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figs")

def simulate_spk_train(N=100, Nt=1000000, save=True):
    spk_train = SPK(N=N, Nt=Nt, dt=0.05, p=0.3, weight_factor=0.9)
    spk_train.simulate_poisson()
    if save:
        with open(os.path.join(data_path, f"spk_train_{N}.pickle"), "wb") as f:
            pickle.dump(spk_train, f)
    return spk_train


def load_spk_train(N=100):
    with open(os.path.join(data_path, f"spk_train_{N}.pickle"), "rb") as f:
        spk_train = pickle.load(f)
    return spk_train


def infer_J_ij(spk_train, i, j, observed_neurons=range(1), with_basis=False):
    mle = MLE(filter_length=100, dt=0.1, observed=observed_neurons, tau=1)
    # mle.plot_inferred(spike_train=spk_train.spike_train, W_true=spk_train.weight_matrix,
    #                   ylim=0.3, basis_free_infer=True, savefig=False, figname='')
    if with_basis:
        return mle.infer_J_ij_basis(i, j, spike_train=spk_train)
    else:
        return mle.infer_J_ij(i, j, spike_train=spk_train)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Flags to control spike train simulation.')
    parser.add_argument('--rerun-simulation', dest='rerun', action='store_true', default=False,
                        help='rerun the simulation to generate spike train')

    args = parser.parse_args()

    # step 1: simulate spike train or load existing spk_train data
    if args.rerun:
        spk_train = simulate_spk_train(N=100)
    else:
        spk_train = load_spk_train(N=100)
        # np.savetxt(os.path.join(data_path, f"spk_train_weights_{100}.txt"), spk_train.weight_matrix)
        # spk_train.simulate_poisson(Nt=2000000)

    # step 2: infer neuron connection between pairs of observed neurons
    for n_observed in [2, 5, 10, 20, 50, 100]:
        J_01 = infer_J_ij(spk_train.spike_train, 0, 1, observed_neurons=range(n_observed), with_basis=True)
        # J_10 = infer_J_ij(spk_train.spike_train, 1, 0)
        np.savetxt(os.path.join(data_path, f"J01_no_basis_{n_observed}.txt"), J_01)
    # np.savetxt(os.path.join(data_path, "J10.txt"), J_10)

        # plt.scatter(range(len(J_00)), J_00)
        # plt.ylim(-0.3, 0.3)
        # plt.savefig(os.path.join(fig_path, f"test_J00_no_basis_{n_observed}.png"))
        # plt.show()

    # plt.scatter(range(len(J_10)), J_10)
    # plt.ylim(-0.3, 0.3)
    # plt.savefig(os.path.join(fig_path, "test_J10.png"))

    # print("J00", J_00)
    # print("J10", J_10)
    # step 3: make plots
