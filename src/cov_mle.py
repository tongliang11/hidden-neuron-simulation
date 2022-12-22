# calculate the Pearson correlation between cov and mle inferred filters
from covariance import cov_estimate
import math
from concurrent.futures import ProcessPoolExecutor
from filter_inference import load_spk_train
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def cov_parallel(spk_train, neuron_pairs=None, dp=1,n_processes = 16):
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
            test = [executor.submit(cov_estimate, spk_train=spk_train[:int(n_time_step*dp),:], N_i=i, N_j=j, dir_name="Spk64_2m_Data_volume_obs_-1_diag_covariance_weight_0_5", filename=f"cov_{i}_{j}_{int(n_time_step*dp)}" ) for i,j in neuron_pairs_batch]
    # print(total)
if __name__ == "__main__":
    N, Nt = 64, 2000000
    spk_train = load_spk_train(N=N, Nt=Nt, filename=f"spk_train_{N}_{Nt}_b_-2_-1_diag_weight_0.5")
    for dp in np.linspace(2, 10, 5)/10:
        print("data volume", dp*Nt)
        cov_parallel(spk_train.spike_train, dp=dp, n_processes=16)
    # cov = np.loadtxt("/home/tong/hidden-neuron-simulation/data/2022-09-20/cov_4_4_1000000")
    # plt.plot(cov)
    # plt.savefig("cov.png")