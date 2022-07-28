from glm.spk_train import Spike_train as SPK
from mle.inference import Maximum_likelihood_estimator as MLE
import pickle
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import date

data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
fig_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figs")

def simulate_spk_train(N=100, Nt=1000000, save=True):
    spk_train = SPK(N=N, Nt=Nt, dt=0.05, p=0.3, weight_factor=0.9)
    spk_train.weight_matrix[1, 0] = 0.3
    spk_train.weight_matrix[0, 1] = 0
    spk_train.simulate_poisson()
    if save:
        with open(os.path.join(data_path, f"spk_train_{N}_{Nt}.pickle"), "wb") as f:
            pickle.dump(spk_train, f)
    return spk_train


def load_spk_train(N, Nt):
    with open(os.path.join(data_path, f"spk_train_{N}_{Nt}.pickle"), "rb") as f:
        spk_train = pickle.load(f)
    return spk_train


def infer_J_ij(spk_train, i, j, basis_order=[0, 1, 2], observed_neurons=range(1), data_percent=1, tol=1e-4, exclude_self_copuling=False, with_basis=False, save=True):
    mle = MLE(filter_length=100, dt=0.1, basis_order=basis_order, observed=observed_neurons, tau=1)
    # mle.plot_inferred(spike_train=spk_train.spike_train, W_true=spk_train.weight_matrix,
    #                   ylim=0.3, basis_free_infer=True, savefig=False, figname='')
    Nt = int(spk_train.shape[0] * data_percent)
    print(f"inferring with {Nt} data and {len(observed_neurons)} observed neurons with basis order {basis_order}...")
    start_time = time.time()
    if with_basis:
        inferred, hess, intercept = mle.infer_J_ij_basis(i, j, spike_train=spk_train[:Nt,:], tol=tol, exclude_self_coupling=exclude_self_copuling)
    else:
        inferred =  mle.infer_J_ij(i, j, spike_train=spk_train)
    total_time = time.time()-start_time
    print(f"Time took for MLE {total_time:.2f} s")
    if save:
        file_path = os.path.join(data_path, f"{date.today()}")
        os.makedirs(file_path, exist_ok=True)
        np.savetxt(os.path.join(file_path, f"J_{i}{j}_{len(basis_order)}_basis_{len(observed_neurons)}_observed_{Nt}_data.txt"), inferred)
        np.savetxt(os.path.join(file_path, f"J_{i}{j}_{len(basis_order)}_basis_{len(observed_neurons)}_observed_{Nt}_data_hessian.txt"), hess)
    return inferred, intercept


def cov_estimate(spk_train, N_i, N_j, max_t_steps=100, data_percent=1, norm=True, save=True):
    Nt = int(spk_train.shape[0] * data_percent)
    print(f"correlation estimation with {Nt} data...")
    start_time = time.time()
    normalized_spk_train_1 = spk_train[:Nt, N_i]-np.mean(spk_train[:, N_i])
    normalized_spk_train_2 = spk_train[:Nt, N_j]-np.mean(spk_train[:, N_j])
    if norm:
        normalization = (np.std(normalized_spk_train_1)
                            * np.std(normalized_spk_train_2))
    else:
        normalization = 1
    cross_correlation = np.array([np.mean([normalized_spk_train_1[t]*normalized_spk_train_2[t+dt] for t in range(len(normalized_spk_train_1)-max_t_steps)])
                                    for dt in range(max_t_steps)])/normalization
    total_time = time.time()-start_time  
    print(f"Time took for MLE {total_time:.2f} s")
    if save:
        file_path = os.path.join(data_path, f"{date.today()}")
        os.makedirs(file_path, exist_ok=True)
        np.savetxt(os.path.join(file_path, f"correlation_{Nt}_data.txt"), cross_correlation)
    
    return cross_correlation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Flags to control spike train simulation.')
    parser.add_argument('--rerun-simulation', dest='rerun', action='store_true', default=False,
                        help='rerun the simulation to generate spike train')

    args = parser.parse_args()

    # step 1: simulate spike train or load existing spk_train data
    if args.rerun:
        spk_train = simulate_spk_train(N=128, Nt=400000)
    else:
        spk_train = load_spk_train(N=128, Nt=400000)
        # np.savetxt(os.path.join(data_path, f"spk_train_weights_{100}.txt"), spk_train.weight_matrix)
        # spk_train.simulate_poisson(Nt=2000000)
    # print(spk_train.spike_train.shape)

    # step 2: infer neuron connection between pairs of observed neurons
    # for data in [0.2, 0.4, 0.6, 0.8, 1]:
    # for data in [0.4, 0.6, 0.8, 1]:#[0.02, 0.04, 0.06, 0.08, 0.1, 0.2]:
    # for n_observed in [2**i for i in range(1, 8)]:
    #     J_00 = infer_J_ij(spk_train.spike_train, 0, 0, basis_order=[1], observed_neurons=range(n_observed), with_basis=True, data_percent=1, save=True)
    #     J_01 = infer_J_ij(spk_train.spike_train, 0, 1, basis_order=[1], observed_neurons=range(n_observed), with_basis=True, data_percent=1, save=True)
    #     J_10 = infer_J_ij(spk_train.spike_train, 1, 0, basis_order=[1], observed_neurons=range(n_observed), with_basis=True, data_percent=1, save=True)
        # print("hess", hess)
            # cov_00 = spk_train.plot_correlation(0, 0, 100)
        # cov_00 = cov_estimate(spk_train.spike_train, 0, 0, data_percent=data)
        # J_10 = infer_J_ij(spk_train.spike_train, 1, 0)
    n_observed = 128
    # for data in [0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4]:
    for data in [0.1, 0.2, 0.4, 0.6, 0.8, 1]:
        # J_00 = infer_J_ij(spk_train.spike_train, 0, 0, basis_order=[1], observed_neurons=range(n_observed), with_basis=True, tol=1e-5, exclude_self_copuling=True, data_percent=data, save=True)
        J_01, J_01_intercept = infer_J_ij(spk_train.spike_train, 0, 1, basis_order=[1], observed_neurons=range(n_observed), with_basis=True, tol=1e-5, data_percent=data, exclude_self_copuling=True, save=True)
        J_10, J_10_intercept = infer_J_ij(spk_train.spike_train, 1, 0, basis_order=[1], observed_neurons=range(n_observed), with_basis=True, tol=1e-5, data_percent=data, exclude_self_copuling=True, save=True)
    # # np.savetxt(os.path.join(data_path, "J10.txt"), J_10)
        print("J_01_intercept", J_01_intercept)
        print("J_10_intercept", J_10_intercept)
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
