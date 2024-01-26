from glm.spk_train import Spike_train as SPK
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

def simulate_spk_train(N=100, Nt=1000000, baseline=-2, nonlinear="exp", weight_matrix=None, weight_factor=2, filename=None, save=True):
    spk_train = SPK(N=N, Nt=Nt, dt=0.1, p=0.5, weight_factor=weight_factor, nonlinear=nonlinear)
    
    if weight_matrix is not None:
        spk_train.weight_matrix = weight_matrix
        if filename is None:
            filename = f"spk_train_{N}_{Nt}_m_new.pickle"
    else:
        # spk_train.weight_matrix[1, 0] = 0.3
        # spk_train.weight_matrix[0, 1] = 0
        if filename is None:
            filename = f"spk_train_{N}_{Nt}.pickle"
    spk_train.simulate_poisson(b=baseline)
    if save:
        with open(os.path.join(data_path, filename), "wb") as f:
            pickle.dump(spk_train, f)
    return spk_train


def load_spk_train(N, Nt, filename=None):
    if filename is None:
        filename = f"spk_train_{N}_{Nt}.pickle"
    with open(os.path.join(data_path, filename), "rb") as f:
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


def infer_weight_matrix(spk_train, basis_order=[1], observed_neurons=range(1), data_percent=1, tol=1e-4, filename=None):
    mle = MLE(filter_length=100, dt=0.1, basis_order=basis_order, observed=observed_neurons, tau=1)
    Nt = int(spk_train.shape[0] * data_percent)
    print(f"inferring with {Nt} data and {len(observed_neurons)} observed neurons with basis order {basis_order}...")
    start_time = time.time()
    file_path = os.path.join(data_path, f"{date.today()}")
    os.makedirs(file_path, exist_ok=True)
    weight_matrix = mle.infer_weight_matrix(spike_train=spk_train[:Nt,:] , tol=tol)
    if filename is None:
        filename = f"inferred_weight_matrix_{len(observed_neurons)}_observed.txt"
    np.savetxt(os.path.join(file_path, filename), np.array(weight_matrix))
    total_time = time.time() - start_time
    print(f"Time took for inferring weight matrix {total_time:.2f} s")
    return weight_matrix


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


def firing_rate(spk_train, dt=0.1):
    print("firing rate: ", np.mean(spk_train, 0)/dt)
    plt.hist(np.mean(spk_train, 0)/dt)
    print("median:", np.median(np.mean(spk_train, 0)/dt))
    print("mean:", np.mean(np.mean(spk_train, 0)/dt))
    plt.xlabel("Firing rate (spike/second)")
    plt.savefig("firing_rate.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Flags to control spike train simulation.')
    parser.add_argument('--rerun-simulation', dest='rerun', action='store_true', default=False,
                        help='rerun the simulation to generate spike train')

    args = parser.parse_args()

    # step 1: simulate spike train or load existing spk_train data
    if args.rerun:
        # weight = np.array([[0, 1], [-1, 0]])
        N, Nt = 64, 2000000
        # spk_train = load_spk_train(N=N, Nt=Nt, filename=f"spk_train_64_2000000_b_-2_-1_diag_weight_1.5")
        weight_matrix = np.loadtxt("/home/tong/hidden-neuron-simulation/src/E-I-network/e-i-weight_64_random.txt")
        J0 = 1
        weight_matrix = J0 * weight_matrix
        np.fill_diagonal(weight_matrix, -1)
        N, Nt = 64, 4000000
        for J_0 in [0.037*i for i in [1, 0.5, 0.25]]:
#         J_0 = 0.037
            weight_matrix = np.ones((N, N))*J_0
            # weight_matrix = 2*spk_train.weight_matrix
            # np.fill_diagonal(weight_matrix, -1)
            print("weight matrix", weight_matrix)
            spk_train = simulate_spk_train(N=N, Nt=Nt, weight_matrix=weight_matrix, baseline=-2, weight_factor=0, nonlinear="exp", filename=f"spk_train_{N}_{Nt}_b_-2_J_{J_0}_all2all_01172024")
        
#         spk_train = simulate_spk_train(N=N, Nt=Nt, weight_matrix=weight_matrix, baseline=-2, weight_factor=0, nonlinear="exp", filename=f"spk_train_{N}_{Nt}_b_-2_-1_diag_EI_network_sigma_1_weight_{J0}")
        # spk_train.plot_raster(t_window=[7000,8000], savefig=True, fig_path="./src/Figures/e-i-raster.png")
        # m_new = scipy.io.loadmat('/home/tong/hidden-neuron-simulation/src/m_new.mat')['Mnew']   
        # for i in range(9, 10):
        #     m_new = np.loadtxt(f"/home/tong/hidden-neuron-simulation/data/weight_matrix/weight_matrix_new_{i}")
            # spk_train = simulate_spk_train(N=256, weight_matrix=m_new, Nt=200000, filename=f"spk_train_256_m_new_p06_{i}")
    else:
        N, Nt = 64, 1000000
        # spk_train = load_spk_train(N=N, Nt=Nt, filename=f"spk_train_{N}_{Nt}_b-2_weight_2")
        spk_train = load_spk_train(N=N, Nt=Nt, filename=f"spk_train_64_2000000_b_-2_-1_diag_weight_1.5_sigmoid")


    firing_rate(spk_train.spike_train)
    # spk_train = spk_train[:20000,:]
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

    # n_observed = 128
    # # # for data in [0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4]:


    # for data in [0.1, 0.2, 0.4, 0.6, 0.8, 1]:
    #     # J_00 = infer_J_ij(spk_train.spike_train, 0, 0, basis_order=[1], observed_neurons=range(n_observed), with_basis=True, tol=1e-5, exclude_self_copuling=True, data_percent=data, save=True)
    #     J_01, J_01_intercept = infer_J_ij(spk_train.spike_train, 0, 1, basis_order=[1], observed_neurons=range(n_observed), with_basis=True, tol=1e-5, data_percent=data, exclude_self_copuling=False, save=True)
    #     J_10, J_10_intercept = infer_J_ij(spk_train.spike_train, 1, 0, basis_order=[1], observed_neurons=range(n_observed), with_basis=True, tol=1e-5, data_percent=data, exclude_self_copuling=False, save=True)
    
    # for i in range(10):
    #     spk_train = load_spk_train(N=256, Nt=200000, filename=f"spk_train_256_m_new_p06_{i}")
    #     infer_weight_matrix(spk_train.spike_train, basis_order=[1], observed_neurons=range(256), filename=f"inferred_weight_256_{i}")



    # # # np.savetxt(os.path.join(data_path, "J10.txt"), J_10)
    #     print("J_01_intercept", J_01_intercept)
    #     print("J_10_intercept", J_10_intercept)
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

    # start = time.time()
    # J_01, J_01_intercept = infer_J_ij(spk_train.spike_train[:20000,:], 1, 1, basis_order=[1], observed_neurons=range(128), with_basis=True, tol=1e-6, data_percent=1, exclude_self_copuling=False, save=False)
    # print("time for sklearn", time.time() - start)
    
    # start = time.time()
    # mle = MLE(filter_length=100, dt=0.1, basis_order=[1], observed=range(128), tau=1)
    # test_x = None #list(J_01)+[J_01_intercept] #0.11012991312678913
    # # print("design matrix shape", mle.design_matrix(spk_train.spike_train, to_neuron=1).shape)
    # theta_constraint = np.empty(129)
    # theta_constraint[:] = np.nan
    # theta_constraint[0] = 0
    # coef, intercept = mle.fit_nll(spk_train.spike_train[:20000,:], to_neuron=1, theta_constraint=theta_constraint, tol=1e-6, test_x=test_x)
    # print("time for nll", time.time() - start)
    # print("coef shape, coef:", coef.shape, coef[:5])
    # print('intercept', intercept)
    # print("J_01, intercept", J_01[:5], J_01_intercept)

    