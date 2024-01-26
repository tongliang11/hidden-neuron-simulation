from parameters import network_params
from weight_matrix import get_weight_matrix_ExcInh_Cluster_1
import numpy as np
import matplotlib.pyplot as plt


def get_weight_matrix_ExcInh_Cluster_N(parameters):

    WRatioE = parameters.WRatioE  # Ratio of Win / Wout(synaptic weight of within group to neurons outside the group)
    pRatioE = parameters.pRatioE
    WRatioI = 1 + parameters.R * (parameters.WRatioE - 1)
    pRatioI = 1 + parameters.R * (parameters.pRatioE - 1)

    wEE = parameters.wEE / parameters.N  # 1.52, 1.165
    wEI = parameters.wEI / parameters.N  # 2.80, 1.2 * 2.991
    wIE = parameters.wIE / parameters.N  # 1.20, 0.737
    wII = parameters.wII / parameters.N  # 4.90, 4.731
    pEE = parameters.pEE
    pEI = parameters.pXX
    pIE = parameters.pXX
    pII = parameters.pXX
    sigma = parameters.sigma / parameters.N

    wEEsub = wEE / ((1 / parameters.numClusters) + (1 - (1 / parameters.numClusters)) / WRatioE)  # Average weight for sub - clusters
    pEEsub = pEE / ((1 / parameters.numClusters) + (1 - (1 / parameters.numClusters)) / pRatioE)
    wEE = wEEsub / WRatioE
    pEE = pEEsub / pRatioE

    if parameters.R != 0:
        wEIsub = wEI / ((1 / parameters.numClusters) + (1 - (1 / parameters.numClusters)) / WRatioI)  # Average weight for sub - clusters
        pEIsub = pEI / ((1 / parameters.numClusters) + (1 - (1 / parameters.numClusters)) / pRatioI)
        wEI = wEIsub / WRatioI
        pEI = pEIsub / pRatioI

        # wIEsub = wIE / ((1 / parameters.numClusters) + (1 - (1 / parameters.numClusters)) / WRatioE)  # Average weight for sub - clusters
        # pIEsub = pIE / ((1 / parameters.numClusters) + (1 - (1 / parameters.numClusters)) / pRatioE)
        # wIE = wIEsub / WRatioE
        # pIE = pIEsub / pRatioE

        # wIIsub = wII / ((1 / parameters.numClusters) + (1 - (1 / parameters.numClusters)) * WRatioI)  # Average weight for sub - clusters
        # pIIsub = pII / ((1 / parameters.numClusters) + (1 - (1 / parameters.numClusters)) * pRatioI)
        # wII = wIIsub * WRatioI
        # pII = pIIsub * pRatioI

    weightsEI = np.random.binomial(1, pEI, (parameters.NE, parameters.NI))         # Weight matrix of inhibitory to single compartment excitatory LIF units
    weightsEI = np.random.normal(wEI, sigma, (parameters.NE, parameters.NI)) * weightsEI

    weightsIE = np.random.binomial(1, pIE, (parameters.NI, parameters.NE))         # Weight matrix of excitatory to inhibitory cells
    weightsIE = np.random.normal(wIE, sigma, (parameters.NI, parameters.NE)) * weightsIE

    weightsII = np.random.binomial(1, pII, (parameters.NI, parameters.NI))         # Weight matrix of inhibitory to inhibitory cells
    weightsII = np.random.normal(wII, sigma, (parameters.NI, parameters.NI)) * weightsII

    weightsEE = np.random.binomial(1, pEE, (parameters.NE, parameters.NE))         # Weight matrix of excitatory to excitatory cells
    weightsEE = np.random.normal(wEE, sigma, (parameters.NE, parameters.NE)) * weightsEE

    # Create the group weight matrices and update the total weight matrix
    for i in range(parameters.numClusters):
        weightsEEsub = np.random.binomial(1, pEEsub, (parameters.EClusterSize, parameters.EClusterSize))
        weightsEEsub = np.random.normal(wEEsub, sigma, (parameters.EClusterSize, parameters.EClusterSize)) * weightsEEsub
        weightsEE[i * parameters.EClusterSize:(i + 1) * parameters.EClusterSize, i * parameters.EClusterSize:(i + 1) * parameters.EClusterSize] = weightsEEsub

    if parameters.R != 0:
        # Create the group weight matrices for Exc to Inh and update the total weight matrix
        # for i in range(parameters.numClusters):
        #     weightsIEsub = np.random.binomial(1, pIEsub, (parameters.IClusterSize, parameters.EClusterSize))
        #     weightsIEsub = wIEsub * weightsIEsub
        #     weightsIE[i * parameters.IClusterSize:(i+1) * parameters.IClusterSize, i * parameters.EClusterSize:(i+1) * parameters.EClusterSize] = weightsIEsub
        #
        # Create the group weight matrices for Inh to Exc and update the total weight matrix
        for i in range(parameters.numClusters):
            weightsEIsub = np.random.binomial(1, pEIsub, (parameters.EClusterSize, parameters.IClusterSize))
            weightsEIsub = np.random.normal(wEIsub, sigma, (parameters.EClusterSize, parameters.IClusterSize)) * weightsEIsub
            weightsEI[i * parameters.EClusterSize:(i+1) * parameters.EClusterSize, i * parameters.IClusterSize:(i+1) * parameters.IClusterSize] = weightsEIsub

        # Create the group weight matrices and update the total weight matrix
        # for i in range(parameters.numClusters):
        #     weightsIIsub = np.random.binomial(1, pIIsub, (parameters.IClusterSize, parameters.IClusterSize))
        #     weightsIIsub = wIIsub * weightsIIsub
        #     weightsII[i * parameters.IClusterSize:(i + 1) * parameters.IClusterSize, i * parameters.IClusterSize:(i + 1) * parameters.IClusterSize] = weightsIIsub

    np.fill_diagonal(weightsII, 0.)
    np.fill_diagonal(weightsEE, -0.)

    W = np.zeros((parameters.N, parameters.N))
    W[:parameters.NE, :parameters.NE] = weightsEE
    W[parameters.NE:, parameters.NE:] = -weightsII
    W[parameters.NE:, :parameters.NE] = weightsIE
    W[:parameters.NE, parameters.NE:] = -weightsEI

    return W

if __name__ == "__main__":
    params = network_params()
    weight = get_weight_matrix_ExcInh_Cluster_1(params)
    print(weight)
    print(weight.shape)
    np.savetxt("/home/tong/hidden-neuron-simulation/src/E-I-network/e-i-weight_64_random.txt", weight)
    plt.imshow(weight)
    plt.colorbar()
    plt.savefig("/home/tong/hidden-neuron-simulation/src/E-I-network/e-i-weight_64_random.png")