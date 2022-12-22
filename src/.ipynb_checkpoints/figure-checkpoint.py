import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import string
import matplotlib.style as style
style.use('seaborn-colorblind')
from matplotlib import cm
from filter_inference import calculate_corr_all
import seaborn as sns


def plot_schematic(inferred_with_basis, inferred_no_basis, savefig=False):
    fig = plt.figure(constrained_layout=False, figsize=(9, 7), dpi=100)

    ax_schematic = plt.subplot(gridspec.GridSpec(
        1, 1, left=0.1, right=1, top=1, bottom=0.55, wspace=0,hspace=0)[0, 0])
    ax_schematic.text(-0.17, 1.05, "A", transform=ax_schematic.transAxes,
                      size=20, weight='bold')
    plt.axis('off')

    gs = gridspec.GridSpec(1, 2, width_ratios=[
                           1, 0.05], left=0, right=0.44, top=0.5, bottom=0, wspace=0,hspace=0)
    ax = plt.subplot(gs[0, 0])
    ax.scatter(np.arange(0,10, 0.1), inferred_no_basis, color='r', s=8, label="MLE inferred w/o basis functions")
    ax.plot(np.arange(0,10, 0.1), inferred_with_basis,"-r", lw=5, label="MLE inferred w/ alpha basis functions")
    ax.plot(np.arange(0,10, 0.1), [0 for i in range(100)], "--", lw=5, label="Ground truth filter")
    ax.set_ylim(-0.4, 0.4)
    ax.set_ylabel("Filter strength (a.u.)", size=18)
    ax.set_xlabel("Time (s)", size=18)
    ax.text(-0.15, 1.05, "B", transform=ax.transAxes,
            size=20, weight='bold')
    ax.legend(frameon=False)
    if savefig:
        fig.savefig(f'./Figures/schematic.pdf', bbox_inches="tight")
        print("figure saved!")
        
        
def plot_mle_cov(cov_3_3, inferred_with_basis_3_3, inferred_no_basis_3_3, w_true, savefig=False):
    fig, axs = plt.subplots(3, 3, figsize=(9, 9), dpi=150)
    filter_length = 100
    dt = 0.1
    observed = [0, 1, 2]
    for j in range(3):
        for i in range(3):
            ax2 = axs[i, j].twinx()
            axs[i, j].vlines(np.arange(filter_length)[:-1] * dt, 0, cov_3_3[i][j][1:], color='black',alpha=0.3, label="Estimated Correlation", zorder=0)
            axs[i, j].set_ylim(-0.0055, 0.0055)
    #         axs[i, j].set_axisbelow(True)
            axs[i, j].yaxis.tick_right()
            axs[i, j].legend(bbox_to_anchor=(1, 0.79),frameon=False, loc="upper right", prop={'size': 7})
            
            ax2.scatter(np.arange(filter_length) * dt, inferred_no_basis_3_3[i][j], s=5, color='red', label="Inferred w/o basis")

            ax2.plot(np.arange(filter_length) * dt, inferred_with_basis_3_3[i][j], linewidth=5, label=r"Inferred $J^{{eff}}_{{{},{}}}$ w/ basis".format(observed[j], observed[i]), color='red', zorder=100)
            ax2.set_ylim(-0.3, 0.3)
            alpha_filter = [w_true[observed[j], observed[i]]* k*np.exp(-k) for k in np.arange(filter_length)*dt]
            ax2.plot(np.arange(filter_length) * dt,
                           alpha_filter, '--', linewidth=5, label=r"Ground-truth $J_{{{},{}}}$".format(observed[j], observed[i]))
            ax2.yaxis.tick_left()
    #         if legend:
            handles, labels = ax2.get_legend_handles_labels()
            order = [2, 0, 1]
            ax2.legend([handles[idx] for idx in order], [
                             labels[idx] for idx in order], frameon=False, loc="upper right", prop={'size': 7})
            if j < 2:
                axs[i, j].set_yticklabels([])
    #         if i < 2:
    #             ax2.set_xticklabels([])
            if j > 0:
                ax2.set_yticklabels([])
        # plt.text(0,1,'Time',size=15)
            # ax2.set_ylabel('Filter Strength',size=15)
    fig.suptitle('MLE inferred effective coupling filters', size=18)
    fig.text(0.5, -0.01, 'Time Preceding (s)', ha='center', size=18)
    fig.text(-0.01, 0.5, 'Filter Strength (a.u.)',
             va='center', rotation='vertical', size=18)
    fig.text(1.00, 0.5, 'Correlation Strength (a.u.)',
             va='center', rotation='vertical', size=18)
    fig.tight_layout()
    if savefig:
        fig.savefig(f'./Figures/MLE_COV.pdf', bbox_inches="tight")
        print("figure saved!")
        
        
def plot_mle_cov_vary_obs(w_true, n=3, obs=[4, 8, 16, 32, 48, 64],figsize=(9.5, 7), data_dir="Spk64_2m_Data_volume_obs_-1_diag_weight_1_5", filename="MLE_vary_obs.pdf", savefig=False):
    fig, axs = plt.subplots(n, n, figsize=figsize, dpi=150)
    filter_length = 100
    dt = 0.1
    observed = list(range(n))
    for j in range(n):
        for i in range(n):
            ax2 = axs[i, j].twinx()
            axs[i, j].yaxis.tick_right()
            for N_obs in obs:
                inferred_wo_basis = np.loadtxt(f"/home/tong/hidden-neuron-simulation/data/{data_dir}/J_{i}_{j}_{N_obs}_observed_2000000_data_no_basis.txt")

                ax2.plot(np.arange(filter_length) * dt, inferred_wo_basis, label="Inferred w/o basis {} observed".format(N_obs))
            ax2.set_ylim(-0.4, 0.4)
            alpha_filter = [w_true[observed[j], observed[i]]* k*np.exp(-k) for k in np.arange(filter_length)*dt]
            ax2.plot(np.arange(filter_length) * dt,
                           alpha_filter, '--', linewidth=5, label="Ground-truth")
            ax2.yaxis.tick_left()
            ax2.text(0.8, 0.05, r"$J_{{{}{}}}$".format(i, j), transform=ax2.transAxes,
            size=15, weight='bold')
            if i == 1 and j == 1:
                ax2.legend(frameon=False, bbox_to_anchor=(1.05, 0), loc='lower left', prop={'size': 8})
            axs[i, j].set_yticklabels([])
            if j > 0:
                ax2.set_yticklabels([])
    fig.text(0.4, 1, 'MLE inferred effective coupling filters', ha='center', size=18)
    fig.text(0.4, -0.01, 'Time Preceding (s)', ha='center', size=18)
    fig.text(-0.01, 0.5, 'Filter Strength (a.u.)',
             va='center', rotation='vertical', size=18)
    fig.tight_layout()
    if savefig:
        fig.savefig(f'./Figures/{filename}', bbox_inches="tight")
        print("figure saved!")


        
def plot_mle_cov_vary_dp(w_true, n=3, obs=64, dp=[0.2, 0.4, 0.8, 1],figsize=(9.5, 7), total_data=2000000, data_dir="Spk64_2m_Data_volume_obs_-1_diag_weight_1_5", filename="MLE_vary_dp.pdf", savefig=False):
    fig, axs = plt.subplots(n, n, figsize=figsize, dpi=150)
    filter_length = 100
    dt = 0.1
    observed = list(range(n))
    for j in range(n):
        for i in range(n):
            ax2 = axs[i, j].twinx()
            axs[i, j].yaxis.tick_right()
            for data_volume in [int(dp_*total_data) for dp_ in dp]:
                inferred_wo_basis = np.loadtxt(f"/home/tong/hidden-neuron-simulation/data/{data_dir}/J_{i}_{j}_{obs}_observed_{data_volume}_data_no_basis.txt")

                ax2.plot(np.arange(filter_length) * dt, inferred_wo_basis, label="Inferred w/o basis {} data".format(data_volume))
            ax2.set_ylim(-0.4, 0.4)
            alpha_filter = [w_true[observed[j], observed[i]]* k*np.exp(-k) for k in np.arange(filter_length)*dt]
            ax2.plot(np.arange(filter_length) * dt,
                           alpha_filter, '--', linewidth=5, label="Ground-truth")
            ax2.yaxis.tick_left()
            ax2.text(0.8, 0.05, r"$J_{{{}{}}}$".format(i, j), transform=ax2.transAxes,
            size=15, weight='bold')
            if i == 1 and j == 1:
                ax2.legend(frameon=False, bbox_to_anchor=(1.05, 0), loc='lower left', prop={'size': 8})
            axs[i, j].set_yticklabels([])
            if j > 0:
                ax2.set_yticklabels([])
    fig.text(0.4, 1, 'MLE inferred effective coupling filters', ha='center', size=18)
    fig.text(0.4, -0.01, 'Time Preceding (s)', ha='center', size=18)
    fig.text(-0.01, 0.5, 'Filter Strength (a.u.)',
             va='center', rotation='vertical', size=18)
    fig.tight_layout()
    if savefig:
        fig.savefig(f'./Figures/{filename}', bbox_inches="tight")
        print("figure saved!")

        
def plot_correlation(cov_path="/home/tong/hidden-neuron-simulation/data/2022-10-04", filter_path="/home/tong/hidden-neuron-simulation/data/2022-10-05-data-volume", filter_type="self-coupling", data_volume_percent=[0.2, 0.4, 0.6, 0.8, 1], total_data=1000000, ylim=[0.08, 1.15], fig_name="correlation.pdf", savefig=False):
        # self-coupling correlation
    fig, axs = plt.subplots(3, 2, figsize=(7, 10), dpi=150)

    corr_per_dp_median = []
    corr_per_dp_mean = []
    for j, dp in enumerate(data_volume_percent):
        corr_per_observed = {i: [] for i in [2, 4, 8, 16, 32, 48, 64]}
    #     print(corr_per_observed)
        for N in [i for i in range(64)]:
            N_j = N
            if filter_type == "cross-coupling-random":
                N_i = 0 if N == 0 else np.random.choice(N, 1)[0]
            elif filter_type == "cross-coupling-zero":
                w_true = np.loadtxt("/home/tong/hidden-neuron-simulation/src/figure_data/weight_matrix_ground_truth.txt")
                w_true_zero = [[idx for idx, i in enumerate(w_true[j,:]) if i == 0] for j in range(w_true.shape[0])]
                N_i = 0 if N == 0 else np.random.choice(w_true_zero[N_i], 1)[0]
            elif filter_type == "cross_coupling-partition-EI":
                pass
            else:
                N_i = N
    #         print(N_i, N_j)
#             if dp == 1:
#                 corr = calculate_corr_all(N_i=N_i, N_j=N_j, cov_path="/home/tong/hidden-neuron-simulation/data/2022-09-27", filter_path="/home/tong/hidden-neuron-simulation/data/2022-10-05-data-volume", dp=dp)
#             else:
            corr = calculate_corr_all(N_i=N_i, N_j=N_j, cov_path=cov_path, filter_path=filter_path, dp=dp, total_data=total_data)

            for k, v in corr.items():
                corr_per_observed[k].append(v)

        sns.violinplot(data=[np.array(corr_per_observed[i]) for i in [2, 4, 8, 16, 32, 48, 64]], scale='width', ax=axs[j//2, j%2], cut=0)
#         if filter_type == "cross-coupling":
#             axs[j//2, j%2].set_ylim([0.8, 1.05])
#         else:
        axs[j//2, j%2].set_ylim(ylim)
        axs[j//2, j%2].set_xticklabels([f"{i/64:.0%}" for i in [2, 4, 8, 16, 32, 48, 64]])
        axs[j//2, j%2].text(-0.2, 0.08+ylim[0], f"Spike train data volume: {int(dp*total_data)}", size=9)
        axs[j//2, j%2].text(-0.1, 1.0, string.ascii_uppercase[j], transform=axs[j//2, j%2].transAxes,
                             size=12, weight='bold')
    #     if j//2 == 2:
    #         axs[j//2, j%2].set_xlabel("Percentage of Neuron Observed")
        corr_per_dp_median.append([np.median(corr) for corr in corr_per_observed.values()])
        corr_per_dp_mean.append([np.mean(corr) for corr in corr_per_observed.values()])
    extent = [-0.5, 6.5, -0.5, 4.5]
    im = axs[2,1].imshow(np.array(corr_per_dp_median), extent=extent, origin='lower', aspect='auto',cmap=cm.get_cmap('spring'))
    # ax.colorbar()
    axs[2,1].text(-1.8,2, "Data Volume ($\\times$ $10^6$)", va='center', rotation='vertical')
    # axs[2,1].set_xlabel("Percentage of Neuron Observed")
    axs[2,1].set_xticks(np.arange(7))
    axs[2,1].set_xticklabels([f"{i/64:.0%}" for i in [2, 4, 8, 16, 32, 48, 64]])

    axs[2,1].set_yticks(np.arange(len(data_volume_percent)))
    axs[2,1].set_yticklabels([dp*total_data/1000000 for dp in data_volume_percent])
    axs[2,1].text(-0.1, 1.0, "F", transform=axs[2,1].transAxes,
                             size=12, weight='bold')
    jump_x = 7 / (1 * 7)
    jump_y = 1 / (1 * 5)

    x_positions = np.linspace(start=-1, stop=6, num=7, endpoint=False)
    y_positions = np.linspace(start=-0.2, stop=4.8, num=5, endpoint=False)
    for y_index, y in enumerate(y_positions):
        for x_index, x in enumerate(x_positions):
            label = f"{np.array(corr_per_dp_median)[y_index, x_index]:.3f}"
            text_x = x + jump_x
            text_y = y + jump_y
            axs[2,1].text(text_x, text_y, label, color='black', ha='center', va='center',size=8)

    # axs[2,1].set_title("Correlation of inferred filter and spike train correlation", size=10)
    # plt.colorbar(im, ticks=np.linspace(0.75, 1, 11), ax=axs[2,1])
    fig.text(0.05, 0.5, "Correlation between spike train autocorrelation and MLE inferred self-coupling filters", va='center', rotation='vertical', size=12)
    fig.text(0.5, 0.08, "Percentage of Neuron Observed", ha='center', size=12)
    if savefig:
        fig.savefig(f'./Figures/{fig_name}', bbox_inches="tight")
        print(f"figure saved at './Figures/{fig_name}'!")
        
        
def plot_corr_dp():
    corr_per_dp_median = []
    corr_per_dp_mean = []
    for dp in [0.2, 0.4, 0.6, 0.8, 1]:
        corr_per_observed = {i: [] for i in [2, 4, 8, 16, 32, 48, 64]}
    #     print(corr_per_observed)
        for N in [i for i in range(64)]:
            N_i, N_j = N, N
    #         print(N_i, N_j)
            if dp == 1:
                corr = calculate_corr_all(N_i=N_i, N_j=N_j, cov_path="/home/tong/hidden-neuron-simulation/data/2022-09-27", filter_path="/home/tong/hidden-neuron-simulation/data/2022-10-05-data-volume", dp=dp)
            else:
                corr = calculate_corr_all(N_i=N_i, N_j=N_j, cov_path="/home/tong/hidden-neuron-simulation/data/2022-10-04", filter_path="/home/tong/hidden-neuron-simulation/data/2022-10-05-data-volume", dp=dp)
    #         print(corr)

            for k, v in corr.items():
    #             print(k)
                corr_per_observed[k].append(v)

            obs = list(corr.keys())
            corr = list(corr.values())
    #         corr_per_observed.append(np.median(corr))
    #         print(corr)
            plt.scatter([i for i in obs], corr, label=f"J_{N_i}{N_j}")
    #         plt.ylim(0.3,1)
        plt.xlabel("Observed Neurons")
        plt.ylabel("Correlation between filter and spk train correlation")
        plt.title(f"Self-coupling filter vs Spike train correlation {dp} data")
    #     plt.legend(bbox_to_anchor=(1.25, 0), loc='lower right')
        plt.show()
        corr_per_dp_median.append([np.median(corr) for corr in corr_per_observed.values()])
        corr_per_dp_mean.append([np.mean(corr) for corr in corr_per_observed.values()])