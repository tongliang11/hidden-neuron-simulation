import numpy as np

# to-do: clean up simulate_poisson ode stuff; networknx plot self connections


class Spike_train:
    def __init__(self, N, Nt, dt, p=0.2, W=None, weight_factor=1):
        self.N = N
        self.Nt = Nt
        self.dt = dt
        self.p = p
        self.J0 = weight_factor
        if np.all(W == None):
            self.weight_matrix = self.generate_weight_matrix(p=self.p)
        else:
            self.weight_matrix = W
        self.spike_train = None
        self.spike_time = None

    def simulate_poisson(self, Nt=None, b=-1, tau=1.0):
        """
        Generate spike trains based on connectivity matrix and an alpha function as the spike-history/interaction filter
            Calculating the convolution of spike trains and the alpha filter 1/tau**2 * t * np.exp(-t/tau) requires keeping track of
            all the past spikes.

        Args:
            Nt (int): number of iterations 
            W (N*N 2darray): the weight matrix
            b (float): the baseline activation, a constant
            dt (float): time step
            N (int): number of neurons
            tau (float): time constant of the alpha function
            Return:
            spk (ndarray): the generated spike train
        """
        # Runge-Kutta 4th order: dt larger than 0.5 may introduce excessive error
        def f1(s, s_d, t):
            return s_d

        def f2(s, s_d, t):
            return -2*a*s_d - a**2*s

        def softplus(x): return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

        def alpha(t): return 1/tau**2*t*np.exp(-t/tau)

        if not Nt:
            Nt = self.Nt

        dt = self.dt
        W = self.weight_matrix
        spk = np.zeros((Nt, self.N))
        spk_time = [[] for _ in range(self.N)]
        s = np.zeros((self.N,), dtype=np.float64)
        s0 = np.zeros((self.N,), dtype=np.float64)
        s_dummy = np.zeros((self.N,), dtype=np.float64)
        a = 1 / tau
        t = 0
        input_s = np.zeros((Nt, self.N), dtype=np.float64)
        # s_s=np.zeros((Nt,N),dtype=np.float64)

        r_vec = np.zeros((Nt, self.N))
        for i in range(1, Nt):
            t += dt
            # s_dummy += dt*(-2*a*s_dummy -a**2*s)
            # s += dt*s_dummy
            # s_s[i-1]=s
            k11 = dt*f1(s, s_dummy, t)
            k21 = dt*f2(s, s_dummy, t)
            k12 = dt*f1(s+0.5*k11, s_dummy+0.5*k21, t+0.5*dt)
            k22 = dt*f2(s+0.5*k11, s_dummy+0.5*k21, t+0.5*dt)
            k13 = dt*f1(s+0.5*k12, s_dummy+0.5*k22, t+0.5*dt)
            k23 = dt*f2(s+0.5*k12, s_dummy+0.5*k22, t+0.5*dt)
            k14 = dt*f1(s+k13, s_dummy+k23, t+dt)
            k24 = dt*f2(s+k13, s_dummy+k23, t+dt)
            s += (k11+2*k12+2*k13+k14)/6
            s_dummy += (k21+2*k22+2*k23+k24)/6
            # if(i > 500):
            #     l = i-500
            # else:
            #     l = 0
            # s_exact=np.sum([[spk[j,k]*alpha((i-j)*dt) for j in range(l,i)] for k in range(N)],1)
            input_s[i-1] = W@s+b

            rate = np.exp(input_s[i-1])*dt
            # rate = softplus(input_s[i-1])*dt
            spk[i] = np.random.poisson(rate, size=(self.N,))
            r_vec[i] = rate
            s_dummy += spk[i]*a**2
        self.spike_train = spk

        for i in range(Nt):
            for j in range(self.N):
                if spk[i, j] > 0:
                    spk_time[j].append(i*self.dt)

        self.spike_time = spk_time

    def generate_weight_matrix(self,  p=0.2, diag=False):

        N = self.N
        W0 = np.random.rand(N, N)
        mask = W0 < p
        W0[np.logical_not(mask)] = 0
        W0[mask] = 1

        if diag:
            # autapses to allow for refractoriness or burstiness
            np.fill_diagonal(W0, 1)
        else:
            # disallow autapses; set to -1 for refractory-like effects
            np.fill_diagonal(W0, 0)

        # matrix of random signs
        signs = 2*np.random.randint(0, 2, size=(N, N))-1

        W0 = W0*signs  # imposes a random sign to each entry

        return (self.J0*np.random.randn(N, N)/np.sqrt(p*N))*W0

    def plot_network(self, savefig=False, name='5neurons_network.png'):
        import networkx as nx
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        W = self.weight_matrix
        G = nx.convert_matrix.from_numpy_matrix(W.T, create_using=nx.DiGraph)
        for e in G.edges():
            if G[e[0]][e[1]]['weight'] < 0:
                G[e[0]][e[1]]['color'] = 'green'
            else:
                G[e[0]][e[1]]['color'] = 'red'
        # np.fill_diagonal(w,0)
        options = {
            #     'node_color': 'orange',
            'node_size': 1000,
            'width': [2*G[u][v]['weight'] for u, v in G.edges()],
            # red for excitation; green for inhibation
            'edge_color': [G[u][v]['color'] for u, v in G.edges()]
        }

        h2 = nx.draw_circular(G, with_labels=True, font_weight='bold',
                              connectionstyle='arc3, rad = 0.1', arrowsize=20, label='network', **options)

        def make_proxy(clr, mappable, **kwargs):
            return Line2D([0, 1], [0, 1], color=clr, **kwargs)

        clrs = ["red", "green"]
        # generate proxies with the above function
        proxies = [make_proxy(clr, h2, lw=5) for clr in clrs]
        # and some text for the legend -- you should use something from df.
        labels = ["Excitatory", "Inhibitory"]
        plt.legend(proxies, labels, frameon=False)
        plt.show()
        if savefig == True:
            plt.savefig(name)

    def plot_histogram(self):
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator

        ax = plt.subplot()
        ax.hist(self.spike_train)
        ax.set_xlabel('Spike count', size=18)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.show()

    def plot_raster(self, t_window=None, ll=0.5, savefig=False, fig_path=None):
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
        def binary_search(spk_time, target):
                cutoff = []
                for times in spk_time:
                    l, r = 0, len(times)-1
                    while l < r:
                        mid = int((r-l)/2+l)
                        if times[mid] >= target:
                            r = mid - 1
                        else:
                            l = mid + 1
                    cutoff.append(l)
                return cutoff
        if isinstance(t_window, list) and len(t_window) == 2:
            print(t_window)
            cutoff_start = binary_search(self.spike_time, t_window[0])
            cutoff_end = binary_search(self.spike_time, t_window[1])
            print(cutoff_start)
            print(cutoff_end)
            spk_time = [self.spike_time[i][cutoff_start[i]:cutoff_end[i]] for i in range(self.N)]
        elif isinstance(t_window, int):
            cutoff_end = binary_search(self.spike_time, t_window)
            spk_time = [self.spike_time[i][:cutoff_end[i]] for i in range(self.N)]
        else:
            spk_time = self.spike_time
        ax = plt.subplot()
        ax.eventplot(spk_time, linelengths=ll)
        ax.set_ylabel('Neuron', size=18)
        ax.set_xlabel('Time (s)', size=18)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        if savefig:
            if fig_path is None:
                fig_path = './Figures/raster.pdf'
            plt.savefig(fig_path, bbox_inches="tight")
        plt.show()

    def plot_correlation(self, N_i, N_j, max_t_steps, norm=True, plot=False, start=None, ax=None, savefig=False, figname='Corr'):
        """Plot auto/cross-correlation for neuron pairs

        Args:
            N_i ([int]): 1st neuron index
            N_j ([int]): 2nd neuron index
        """
        import matplotlib.pyplot as plt

        normalized_spk_train_1 = self.spike_train[:,
                                                  N_i]-np.mean(self.spike_train[:, N_i])
        normalized_spk_train_2 = self.spike_train[:,
                                                  N_j]-np.mean(self.spike_train[:, N_j])
        if norm:
            normalization = (np.std(normalized_spk_train_1)
                             * np.std(normalized_spk_train_2))
        else:
            normalization = 1
        cross_correlation = np.array([np.mean([normalized_spk_train_1[t]*normalized_spk_train_2[t+dt] for t in range(len(normalized_spk_train_1)-max_t_steps)])
                                      for dt in range(max_t_steps)])/normalization
        if plot:
            if ax is None:
                fig, ax = plt.subplots()
            if start is None:
                start = 1 if N_i == N_j else 0

            ax.vlines(np.arange(start, max_t_steps)*self.dt, 0,
                      cross_correlation[start:], colors='black')
            ax.set_xlabel('Lag (s)', size=18)
            ax.set_ylabel('Correlation', size=18)
            ax.set_title(
                'Correlogram of Neuron {} and {}'.format(N_i, N_j), size=18)
            if savefig:
                fig.savefig(f'./Figures/{figname}.pdf', bbox_inches="tight")
        return cross_correlation

    def plot_cross_correlation(self, N_i, N_j, max_t_steps, savefig=False, figname='Corr'):
        import matplotlib.pyplot as plt
        corr = []
        fig, axes = plt.subplots(nrows=len(N_i), ncols=len(
            N_j), figsize=(3*len(N_j), 3*len(N_i)))
        for i in range(len(N_i)):
            for j in range(len(N_j)):

                #                 if i<j:
                #                     axes[i, j].axis('off')
                #                 else:
                normalized_spk_train_1 = self.spike_train[:,
                                                          N_i[i]]-np.mean(self.spike_train[:, N_i[i]])
                normalized_spk_train_2 = self.spike_train[:,
                                                          N_j[j]]-np.mean(self.spike_train[:, N_j[j]])
                cross_correlation = [np.mean([normalized_spk_train_1[t]*normalized_spk_train_2[t+dt] for t in range(len(normalized_spk_train_1)-max_t_steps)])
                                     for dt in range(max_t_steps)]/(np.std(normalized_spk_train_1)*np.std(normalized_spk_train_2))
                corr.append(cross_correlation)

                start = 1 if i == j else 0
                # suppress 0 lag for autocorrelation
                axes[i, j].vlines(np.arange(
                    start, max_t_steps)*self.dt, 0, cross_correlation[start:], colors='black')
                axes[i, j].set_title(f'{N_i[i]} to {N_j[j]}')

        fig.suptitle('Cross-correlation', size=18)
        fig.text(0.5, -0.01, 'Time Lag (s)', ha='center', size=18)
        fig.text(-0.01, 0.5, 'Correlation', va='center',
                 rotation='vertical', size=18)
        fig.tight_layout()
        plt.show()
        return corr
