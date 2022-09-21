from mle.inference import Maximum_likelihood_estimator as MLE
import pickle
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import date
import scipy.io

data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
fig_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figs")


def infer_J_ij(spk_train, i, j, basis_order=[0, 1, 2], observed_neurons=range(1), data_percent=1, tol=1e-4, exclude_self_copuling=False, with_basis=False, save=True):
    mle = MLE(filter_length=100, dt=0.1, basis_order=basis_order, observed=observed_neurons, tau=1)
    # mle.plot_inferred(spike_train=spk_train.spike_train, W_true=spk_train.weight_matrix,
    #                   ylim=0.3, basis_free_infer=True, savefig=False, figname='')
    Nt = int(spk_train.shape[0] * data_percent)

    start_time = time.time()
    if with_basis:
        print(f"inferring with {Nt} data and {len(observed_neurons)} observed neurons with basis order {basis_order}...")
        inferred, hess, intercept = mle.infer_J_ij_basis(i, j, spike_train=spk_train[:Nt,:], tol=tol, exclude_self_coupling=exclude_self_copuling)
        filename_coef = f"J_{i}{j}_{len(basis_order)}_basis_{len(observed_neurons)}_observed_{Nt}_data.txt"
        filename_intercept = f"J_{i}{j}_{len(basis_order)}_basis_{len(observed_neurons)}_observed_{Nt}_data_intercept.txt"
    else:
        print(f"inferring with {Nt} data and {len(observed_neurons)} observed neurons without basis...")
        inferred, intercept =  mle.infer_J_ij(i, j, spike_train=spk_train[:Nt,:])
        # print(inferred, intercept)
        if type(i) == list:
            filename_coef = f"J_{i}_{j}_{len(observed_neurons)}_observed_{Nt}_data.txt"
            filename_intercept = f"J_{i}_{j}_{len(observed_neurons)}_observed_{Nt}_data_intercept.txt"
        else:
            filename_coef = f"J_{i}{j}_{len(observed_neurons)}_observed_{Nt}_data.txt"
            filename_intercept = f"J_{i}{j}_{len(observed_neurons)}_observed_{Nt}_data_intercept.txt"
    total_time = time.time()-start_time
    print(f"Time took for MLE {total_time:.2f} s")
    if save:
        file_path = os.path.join(data_path, f"{date.today()}")
        os.makedirs(file_path, exist_ok=True)
        np.savetxt(os.path.join(file_path, filename_coef), inferred)
        np.savetxt(os.path.join(file_path, filename_intercept), [intercept])
        # np.savetxt(os.path.join(file_path, f"J_{i}{j}_{len(basis_order)}_basis_{len(observed_neurons)}_observed_{Nt}_data_hessian.txt"), hess)
    return inferred

def plot_filter(filter, N_i, N_j,observed_neurons):
    plt.title(f"inferred_filter_J{N_i}{N_j}")
    plt.scatter(np.arange(len(filter)), filter, label=f"{len(observed_neurons)} observed")
    file_path = os.path.join(fig_path, f"{date.today()}")
    os.makedirs(file_path, exist_ok=True)
    plt.savefig(os.path.join(file_path, f"inferred_filter_J{N_i}{N_j}_{len(observed_neurons)}_observed.png"))
    # plt.legend()
    plt.show()

def infer_plot(spk_train, N_i, N_j, observed_neurons):
    filter = infer_J_ij(spk_train, N_i, N_j, with_basis=False, save=True, observed_neurons=observed_neurons)
    plot_filter(filter, N_i, N_j, observed_neurons=observed_neurons)

def cov_filter_similarity(cov, filter):
    return np.corrcoef(cov,filter)[0,1]


def load_spk_train(N, Nt, filename=None):
    if filename is None:
        filename = f"spk_train_{N}_{Nt}.pickle"
    with open(os.path.join(data_path, filename), "rb") as f:
        spk_train = pickle.load(f)
    return spk_train

def plot_inferred(data_path):
    filter = np.loadtxt(data_path)
    plt.scatter(range(100), filter[:100])
    plt.savefig(os.path.join(os.path.dirname(data_path), "inferred_filter.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Flags to control spike train simulation.')
    parser.add_argument('--rerun-simulation', dest='rerun', action='store_true', default=False,
                        help='rerun the simulation to generate spike train')

    args = parser.parse_args()

    # step 1: simulate spike train or load existing spk_train data
    if args.rerun:
        pass
    else:
        N, Nt = 64, 1000000
        spk_train = load_spk_train(N=N, Nt=Nt, filename=f"spk_train_{N}_{Nt}_b-2")



    # print("firing rate: ", np.mean(spk_train.spike_train, 0) * 10)
    # plt.hist(np.mean(spk_train.spike_train, 0) * 10)
    # print("median:", np.median(np.mean(spk_train.spike_train, 0) * 10))
    # print("mean:", np.mean(np.mean(spk_train.spike_train, 0) * 10))
    # plt.xlabel("Firing rate (spike/second)")
    # plt.savefig("firing_rate.png")
    # exit()
    # inferred_filter = infer_J_ij(spk_train.spike_train, 0, 0)
    # plt.scatter(np.arange(100), inferred_filter)
    # plt.savefig(os.path.join(fig_path, "inferred_filter.png"))
    # for obs in [1, 5, 10, 30, 50, 70, 90, 100]:
    #     infer_plot(spk_train.spike_train, 0, 0, observed_neurons=range(obs))
    # N_j = 0
    # # obs = 64
    # for obs in [3, 12, 36, 64]:
    #     for N_j in [0, 1, 2]:
    #         infer_J_ij(spk_train.spike_train, [0,1,2], N_j, data_percent=0.1, with_basis=False, save=True, observed_neurons=range(obs))
    N_i, N_j =0,0
    dp = 1
    obs = 64
    inferred_no_basis = infer_J_ij(spk_train.spike_train, N_i, N_j, data_percent=dp, with_basis=False, save=True, observed_neurons=range(obs))
    inferred_basis = infer_J_ij(spk_train.spike_train, N_i, N_j, data_percent=dp, with_basis=True, save=True, observed_neurons=range(obs))
    # plot_inferred("/home/tong/hidden-neuron-simulation/data/2022-09-13/J_[0, 1, 2]_0_3_observed_600000_data.txt")
    plt.scatter([i*0.1 for i in range(100)], inferred_no_basis, label="Inferred filter no basis")
    plt.plot([i*0.1 for i in range(100)], inferred_basis, label="Inferred filter with basis")

    cov = np.loadtxt(f"/home/tong/hidden-neuron-simulation/data/2022-09-14/64_neuron_correlation_{N_i}_{N_j}_1000000_data.txt")
    # # # # print(filters[i])
    # # # # plt.scatter(cov[1:], filters[i][1:])
    # print("inferred no basis:", inferred_no_basis)
    plt.scatter([i*0.1 for i in range(99)], cov[1:]*30, label=f"Autocorrelogram for neuron {N_i}_{N_j}")
    plt.text(5, 0.075, f"Pearson correlation: {cov_filter_similarity(cov[1:], inferred_no_basis[:-1]):.3f}")
    plt.title(f"Correlogram (x30 scaled, 0 lag suppressed) vs. MLE inferred filter\n 64 Neuron network {obs} observed")
    plt.xlabel("Time (s)")
    plt.legend()
    figure_dir = os.path.join(fig_path, str(date.today()))
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    plt.savefig(os.path.join(figure_dir, f"64Neuron_1m_data_J_{N_i}{N_j}_{obs}_observed_cov.png"))    # inferred = infer_J_ij(spk_train.spike_train[:200000,:], 1, N_j, with_basis=False, save=False, observed_neurons=range(obs))
    
    plt.cla()
    # print(inferred)
    # plt.plot(inferred)
    # plt.savefig("test.png")

    # mle = MLE(filter_length=100, dt=0.1, basis_order=[0,1,2], observed=range(obs), tau=1)
    # nll_results = mle.fit_nll(spk_train.spike_train, N_j, fit_with_basis=False)
    # print(nll_results)


    # filters = np.loadtxt("/home/tong/hidden-neuron-simulation/data/2022-09-07/J_[0, 1, 2]_0_3_observed_2000000_data.txt")
    # print(filters.shape)
    # for N_i in [0, 1, 2]:
    #     plot_filter(filters[N_i], N_i, 0, range(3))

    # from concurrent.futures import ProcessPoolExecutor

    # # trials_in_parallel = 28
    # with ProcessPoolExecutor() as executor:
    #         test = [executor.submit(infer_J_ij, spk_train.spike_train, i, j) for i in [0, 1, 2] for j in [0, 1, 2]]
    # print(test[0].result())

    # observed 3
    # i, j = 0, 1
    # # filters = np.loadtxt(f"/home/tong/hidden-neuron-simulation/data/2022-09-07/J_[0, 1, 2]_{j}_3_observed_2000000_data.txt")
    # # cov = np.loadtxt(f"/home/tong/hidden-neuron-simulation/data/2022-09-06/correlation_{i}_{j}_2000000_data.txt")
    cov = np.loadtxt(f"/home/tong/hidden-neuron-simulation/data/2022-09-14/64_neuron_correlation_{N_i}_{N_j}_500000_data.txt")
    # # # # print(filters[i])
    # # # # plt.scatter(cov[1:], filters[i][1:])
    # print("inferred no basis:", inferred_no_basis)
    plt.scatter([i*0.1 for i in range(99)], cov[1:]*40, label=f"Estimated autocorrelogram for neuron {N_i}{N_j}")
    plt.scatter([i*0.1 for i in range(100)], inferred_no_basis, label=f"MLE inferred filter J_{N_i}{N_j}")
    plt.xlabel("Time lag (s)")
    # plt.ylabel("")
    plt.title("Correlogram (x40 scaled, 0 lag suppressed) vs. MLE inferred filter")
    plt.legend()
    plt.savefig(f"/home/tong/hidden-neuron-simulation/figs/2022-09-14/64Neuron_J{N_i}{N_j}_scatter_cov_filter_{obs}_observed_500k_data.png")
    print(f"J_{N_i}{N_j} and cov_{N_i}{N_j}", cov_filter_similarity(cov[1:], inferred_no_basis[:-1]))

    # filters = np.loadtxt("/home/tong/hidden-neuron-simulation/data/2022-09-07/J_[0, 1, 2]_0_3_observed_2000000_data.txt")
    # cov = np.loadtxt("/home/tong/hidden-neuron-simulation/data/2022-09-13/correlation_1_1_2000000_data.txt")
    # print("J_10 and cov_10", cov_filter_similarity(cov[1:], filters[1][:-1]))
    # plt.hist(np.mean(spk_train.spike_train, 0))
    # plt.savefig("firing_rate.png")

    # J00 vs cov00
    # 64 obs: 0.822
    # 32 obs: 0.971
    # 16 obs: 0.984
    # 8 obs: 0.998
    # 4 obs: 0.99997

    # 500k data:
    # 64 obs: 0.557
    # 16 obs: 0.9831

    # J11 vs cov11
    # 64 obs: 0.930
    # 32 obs: 
    # 16 obs: 0.988
    # 8 obs: 
    # 2 obs: 0.99994

    # 500k data:
    # 16 obs: 0.9724


    # J01 vs cov01
    # 64 obs: 0.99899
    # 32 obs: 
    # 16 obs: 0.9993
    # 8 obs: 
    # 2 obs: 0.99997

    # 500k data:
    # 16 obs: 0.9986

    # 09142022 to do:
    # check zeros
    # data volume
    # parallel computing the cov vs Jxx for all pairs
