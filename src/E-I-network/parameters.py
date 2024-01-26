import numpy as np

class network_params:
    def __init__(self, ):
        self.N = 64  # Number of neurons
        self.tauE = 15  # Time constant
        self.tauI = 10  # Time constant for inhibitory population
        self.gain = 40.0  # Gain of threshold-linear input-output function
        self.trans = 0.  # Transient period
        self.tstop = 1000.  # Total simulation time
        self.dt = .1  # Simulation time-step
        self.tStim = 1000. / self.dt
        self.tStimStop = 750. / self.dt
        self.maxSpikes = 5000 * self.N * self.tstop / 1000  # 500 Hz neuron
        self.Nt = int(self.tstop / self.dt)  # Number of simulation time-points
        self.Etr = 5  # Refractory period for excitatory neurons
        self.Itr = 5  # Refractory period for inhibitory neurons
        self.n_runs = 5  # Number of simulations
        self.nE = 0.80  # Fraction of excitatory neurons
        self.nI = 0.20  # Fraction of inhibotory neurons
        self.NE = round(self.nE * self.N)  # Number of excitatory neurons
        self.clusterSize = int(self.N / 10)  # Size of each Exh + Inh Cluster
        self.numClusters = 1 # np.maximum(int(self.N / self.clusterSize), 1)  # Number of clusters
        self.NI = round(self.nI * self.N)  # Number of inhibitory neurons
        self.EClusterSize = int(self.NE / self.numClusters)  # Size of excitatory cluster
        self.IClusterSize = int(self.NI / self.numClusters)  # Size of inhibitory clusters
        self.t_EE = 3  # time constant from excitatory to excitatory/inhibitory synapses
        self.t_II = 2  # time constant from inhibitory to inhibitory/excitatory synapses
        self.pEE = 0.2  # Probability of connections from excitatory to excitatory neurons
        self.pXX = 0.5  # Probability of connections for other neurons
        self.IeE = 1.10  # 1.10 # Mean value of external current for the excitatory population
        self.IeI = 1.00  # 1.05 # Mean value of external current for the inhibitory population
        self.IeE_var = 0.1  # Variance of external current for the excitatory population
        self.IeI_var = 0.05  # Variance of external current for the inhibitory population
        self.Vthres = 1.0
        self.Vreset = 0
        self.R = 1.0
        self.WRatioE = 6.5  # Ratio of Win / Wout(synaptic weight of within group to neurons outside the group)
        self.pRatioE = 1.0
        self.jSelf = self.Vthres - self.Vreset
        self.wEE = 0.70  # 0.71
        self.wEI = 1.33  # 1.32
        self.wIE = 0.33  # 0.30
        self.wII = 1.33  # 1.30
        self.m0 = 0.5  # 1.0
        self.sigma = 1
