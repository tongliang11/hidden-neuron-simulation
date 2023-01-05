import numpy as np
import matplotlib.pyplot as plt
import time, os, argparse, pickle
from datetime import date

data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
fig_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figs")

def cov_estimate(spk_train, N_i, N_j, max_t_steps=100, data_percent=1, norm=True, save=True, dir_name=None, filename=None):
    Nt = int(spk_train.shape[0] * data_percent)
    N = spk_train.shape[1]
    print(f"correlation {[N_i, N_j]} estimation with {Nt} data...")
    start_time = time.time()
    normalized_spk_train_1 = spk_train[:Nt, N_i]-np.mean(spk_train[:Nt, N_i])
    normalized_spk_train_2 = spk_train[:Nt, N_j]-np.mean(spk_train[:Nt, N_j])
    if norm:
        normalization = (np.std(normalized_spk_train_1)
                            * np.std(normalized_spk_train_2))
    else:
        normalization = 1
    cross_correlation = np.array([np.mean([normalized_spk_train_1[t]*normalized_spk_train_2[t+dt] for t in range(len(normalized_spk_train_1)-max_t_steps)])
                                    for dt in range(max_t_steps)])/normalization
    total_time = time.time()-start_time  
    print(f"Time took for covariance estimation {total_time:.2f} s")
    if save:
        if dir_name is None:
            dir_name = date.today()
        file_path = os.path.join(data_path, dir_name)
        os.makedirs(file_path, exist_ok=True)
        if filename is None:
            filename = f"{N}_neuron_correlation_{N_i}_{N_j}_{Nt}_data.txt"

        np.savetxt(os.path.join(file_path, filename), cross_correlation)
    
    return cross_correlation


def plot_cov(cross_correlation,N_i, N_j, start=0, max_t_steps=100, dt=0.1, ax=None, savefig=True, figname=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.vlines(np.arange(start, max_t_steps)*dt, 0, cross_correlation[start:],colors='black')
    ax.set_xlabel('Lag (s)', size=18)
    ax.set_ylabel('Correlation', size=18)
    ax.set_title(f'Correlogram of {N_i} and {N_j}', size=18)
    if savefig:
        if figname is None:
            figname = f"cov_{N_i}_{N_j}.png"
        fig_dir = os.path.join(fig_path, f"{date.today()}")
        os.makedirs(fig_dir, exist_ok=True)
        fig.savefig(os.path.join(fig_dir, figname))
        

def cov_estimate_plot(spk_train, N_i, N_j, dp, filename=None, figname=None):
    cov = cov_estimate(spk_train, N_i, N_j, data_percent=dp, filename=filename)
    plot_cov(cov,N_i,N_j,start=1, figname=figname)


def load_spk_train(N, Nt, filename=None):
    if filename is None:
        filename = f"spk_train_{N}_{Nt}.pickle"
    with open(os.path.join(data_path, filename), "rb") as f:
        spk_train = pickle.load(f)
    return spk_train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Flags to control spike train simulation.')
    parser.add_argument('--rerun-simulation', dest='rerun', action='store_true', default=False,
                        help='rerun the simulation to generate spike train')

    args = parser.parse_args()

    # step 1: simulate spike train or load existing spk_train data
    if args.rerun:
        pass
        
        # m_new = scipy.io.loadmat('/home/tong/hidden-neuron-simulation/src/m_new.mat')['Mnew']   
        # for i in range(9, 10):
        #     m_new = np.loadtxt(f"/home/tong/hidden-neuron-simulation/data/weight_matrix/weight_matrix_new_{i}")
            # spk_train = simulate_spk_train(N=256, weight_matrix=m_new, Nt=200000, filename=f"spk_train_256_m_new_p06_{i}")
    else:
        N, Nt = 64, 1000000
        spk_train = load_spk_train(N=N, Nt=Nt, filename=f"spk_train_{N}_{Nt}_b-2_weight_0")

    spk_train = spk_train.spike_train #[:500000,:]
    for dp in [0.2, 0.4, 0.6, 0.8, 1]:
        cov0 = cov_estimate_plot(spk_train, 0, 0, dp,filename=f"{N}_neuron_correlation_{0}_{0}_{int(dp*Nt)}_data_independent.txt", figname=f"cov_{0}_{0}_independent.png")
        cov = cov_estimate_plot(spk_train, 0, 1, dp,filename=f"{N}_neuron_correlation_{0}_{1}_{int(dp*Nt)}_data_independent.txt", figname=f"cov_{0}_{1}_independent.png")
        cov2 = cov_estimate_plot(spk_train, 1, 1, dp,filename=f"{N}_neuron_correlation_{1}_{1}_{int(dp*Nt)}_data_independent.txt", figname=f"cov_{1}_{1}_independent.png")
    # cov = np.loadtxt("/home/tong/hidden-neuron-simulation/data/2022-09-13/2_neuron_correlation_1_1_20000000_data.txt")
    # plot_cov(cov, 1, 1, start=1, figname="2_neuron_20m_Data_cov_1_1.png")
    # plot_cov(cov,0, 0,start=1)

    # from concurrent.futures import ProcessPoolExecutor

    # # trials_in_parallel = 28
    # with ProcessPoolExecutor() as executor:
    #         test = [executor.submit(cov_estimate_plot, spk_train.spike_train, i, j) for i in [0, 1, 2] for j in [0, 1, 2]]
    # print(test[0].result())