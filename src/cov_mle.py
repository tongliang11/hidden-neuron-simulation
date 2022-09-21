# calculate the Pearson correlation between cov and mle inferred filters
from covariance import cov_estimate
import math
from concurrent.futures import ProcessPoolExecutor


def cov_parallel(spk_train):
    n_time_step, n_neuron = spk_train.shape
    n_processes = 16
    neuron_pairs = [[[i, j] for i in range(j+1)] for j in range(n_neuron)]
    print(f"{len(neuron_pairs)} covariance will be estimated for {n_neuron} neurons")
    for i in range(math.ceil(len(neuron_pairs)/n_processes)):
        neuron_pairs_batch = neuron_pairs[i:i+n_processes]
        with ProcessPoolExecutor() as executor:
            test = [executor.submit(cov_estimate, spk_train=spk_train, N_i=i, N_j=j, filename=f"cov_{i}_{j}" ) for i,j in range(neuron_pairs_batch)]

