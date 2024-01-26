# calculate the Pearson correlation between cov and mle inferred filters
from covariance import cov_estimate
import math
from concurrent.futures import ProcessPoolExecutor
from filter_inference import load_spk_train
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def cov_parallel(spk_train, dir_name, neuron_pairs=None, dp=1,n_processes = 16, normalization=True):
    n_time_step, n_neuron = spk_train.shape
    if neuron_pairs is None:
        neuron_pairs = [[i, j] for j in range(n_neuron) for i in range(n_neuron)]
    print(f"{len(neuron_pairs)} covariance will be estimated for {n_neuron} neurons")
    # print(neuron_pairs)
    # total = 0
    n_batch = math.ceil(len(neuron_pairs)/n_processes)
    print(n_batch)
    for batch in tqdm(range(n_batch)):
        neuron_pairs_batch = neuron_pairs[batch*n_processes:(batch+1)*n_processes]
        print(neuron_pairs_batch)
        # total += len(neuron_pairs_batch)
        with ProcessPoolExecutor() as executor:
            test = [executor.submit(cov_estimate, spk_train=spk_train[:int(n_time_step*dp),:], N_i=i, N_j=j, norm=normalization, dir_name=dir_name, filename=f"cov_{i}_{j}_{int(n_time_step*dp)}" ) for i,j in neuron_pairs_batch]
    # print(total)
if __name__ == "__main__":
    N, Nt = 64, 2000000
    # J_0 = 0.037*0.25
    # spk_train = load_spk_train(N=N, Nt=Nt, filename=f"spk_train_{N}_{Nt}_b_-2_J_{J_0}_all2all_01172024")
    spk_train = load_spk_train(N=N, Nt=Nt, filename=f"spk_train_64_2000000_b_-2_-1_diag_EI_network_weight_7")
    for dp in [1]:
        print("data volume", dp*Nt)
        cov_parallel(spk_train.spike_train, dp=dp, n_processes=16, normalization=False, dir_name=f"Spk64_2m_Data_volume_obs_-1_diag_covariance_EI_network_weight_7_unnormalized")
    # cov = np.loadtxt("/home/tong/hidden-neuron-simulation/data/2022-09-20/cov_4_4_1000000")
    # plt.plot(cov)
    # plt.savefig("cov.png")