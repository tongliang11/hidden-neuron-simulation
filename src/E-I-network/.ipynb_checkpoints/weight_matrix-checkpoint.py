import numpy as np

def get_weight_matrix_ExcInh_Cluster(parameters):

    WRatioE = parameters.WRatioE  # Ratio of Win / Wout(synaptic weight of within group to neurons outside the group)
    pRatioE = parameters.pRatioE
    WRatioI = 1 + parameters.R * (parameters.WRatioE - 1)
    pRatioI = 1 + parameters.R * (parameters.pRatioE - 1)

    wEE = parameters.wEE / np.sqrt(parameters.N)  # 1.52, 1.165
    wEI = parameters.wEI / np.sqrt(parameters.N)  # 2.80, 1.2 * 2.991
    wIE = parameters.wIE / np.sqrt(parameters.N)  # 1.20, 0.737
    wII = parameters.wII / np.sqrt(parameters.N)  # 4.90, 4.731
    pEE = parameters.pEE
    pEI = parameters.pXX
    pIE = parameters.pXX
    pII = parameters.pXX
    # wSE = 0.011 * np.sqrt(parameters.N)
    # wSI = 4 * 0.011 * np.sqrt(parameters.N)

    if parameters.numClusters != 1:
        wEEsub = WRatioE * wEE
        pEEsub = pRatioE * parameters.pEE
        wEE = wEE * (parameters.numClusters - WRatioE) / (parameters.numClusters - 1)  # Average weight for sub - clusters
        pEE = parameters.pEE * (parameters.numClusters - pRatioE) / (parameters.numClusters - 1)

        if parameters.R != 0:
            wIEsub = WRatioE * wIE
            pIEsub = pRatioE * parameters.pXX
            wIE = wIE * (parameters.numClusters - WRatioE) / (parameters.numClusters - 1)  # Average weight for sub - clusters
            pIE = parameters.pXX * (parameters.numClusters - pRatioE) / (parameters.numClusters - 1)

            wEIsub = wEI / WRatioI
            pEIsub = parameters.pXX / pRatioI
            wEI = wEI / ((parameters.numClusters - WRatioI) / (parameters.numClusters - 1))  # Average weight for sub - clusters
            pEI = parameters.pXX / ((parameters.numClusters - pRatioI) / (parameters.numClusters - 1))

            wIIsub = wII / WRatioI
            pIIsub = parameters.pXX / pRatioI
            wII = wII / ((parameters.numClusters - WRatioI) / (parameters.numClusters - 1))  # Average weight for sub - clusters
            pII = parameters.pXX / ((parameters.numClusters - pRatioI) / (parameters.numClusters - 1))
    else:
        wEEsub = wEE
        wIEsub = wIE
        wEIsub = wEI
        wIIsub = wII

    weightsEI = np.random.binomial(1, pEI, (parameters.NE, parameters.NI))         # Weight matrix of inhibitory to single compartment excitatory LIF units
    weightsEI = wEI * weightsEI

    weightsIE = np.random.binomial(1, pIE, (parameters.NI, parameters.NE))         # Weight matrix of excitatory to inhibitory cells
    weightsIE = wIE * weightsIE

    weightsII = np.random.binomial(1, pII, (parameters.NI, parameters.NI))         # Weight matrix of inhibitory to inhibitory cells
    weightsII = wII * weightsII

    weightsEE = np.random.binomial(1, pEE, (parameters.NE, parameters.NE))         # Weight matrix of excitatory to excitatory cells
    weightsEE = wEE * weightsEE

    # Create the group weight matrices and update the total weight matrix
    for i in range(parameters.numClusters):
        weightsEEsub = np.random.binomial(1, pEEsub, (parameters.EClusterSize, parameters.EClusterSize))
        weightsEEsub = wEEsub * weightsEEsub
        weightsEE[i * parameters.EClusterSize:(i + 1) * parameters.EClusterSize, i * parameters.EClusterSize:(i + 1) * parameters.EClusterSize] = weightsEEsub

    if parameters.R != 0:
        # Create the group weight matrices for Exc to Inh and update the total weight matrix
        for i in range(parameters.numClusters):
            weightsIEsub = np.random.binomial(1, pIEsub, (parameters.IClusterSize, parameters.EClusterSize))
            weightsIEsub = wIEsub * weightsIEsub
            weightsIE[i * parameters.IClusterSize:(i+1) * parameters.IClusterSize, i * parameters.EClusterSize:(i+1) * parameters.EClusterSize] = weightsIEsub

        # Create the group weight matrices for Inh to Exc and update the total weight matrix
        for i in range(parameters.numClusters):
            weightsEIsub = np.random.binomial(1, pEIsub, (parameters.EClusterSize, parameters.IClusterSize))
            weightsEIsub = wEIsub * weightsEIsub
            weightsEI[i * parameters.EClusterSize:(i+1) * parameters.EClusterSize, i * parameters.IClusterSize:(i+1) * parameters.IClusterSize] = weightsEIsub

        # Create the group weight matrices and update the total weight matrix
        for i in range(parameters.numClusters):
            weightsIIsub = np.random.binomial(1, pIIsub, (parameters.IClusterSize, parameters.IClusterSize))
            weightsIIsub = wIIsub * weightsIIsub
            weightsII[i * parameters.IClusterSize:(i + 1) * parameters.IClusterSize, i * parameters.IClusterSize:(i + 1) * parameters.IClusterSize] = weightsIIsub

    # Ensure the diagonals are zero
    # np.fill_diagonal(weightsII, parameters.nI * wSI)
    # np.fill_diagonal(weightsEE, parameters.nE * -wSE)

    np.fill_diagonal(weightsII, 1.0)
    np.fill_diagonal(weightsEE, -1.0)

    W = np.zeros((parameters.N, parameters.N))
    W[:parameters.NE, :parameters.NE] = weightsEE
    W[parameters.NE:, parameters.NE:] = -weightsII
    W[parameters.NE:, :parameters.NE] = weightsIE
    W[:parameters.NE, parameters.NE:] = -weightsEI

    return W

def get_weight_matrix_ExcInh_Cluster_1(parameters):

    WRatioE = parameters.WRatioE  # Ratio of Win / Wout(synaptic weight of within group to neurons outside the group)
    pRatioE = parameters.pRatioE
    WRatioI = 1 + parameters.R * (parameters.WRatioE - 1)
    pRatioI = 1 + parameters.R * (parameters.pRatioE - 1)

    wEE = parameters.wEE / np.sqrt(parameters.N)  # 1.52, 1.165
    wEI = parameters.wEI / np.sqrt(parameters.N)  # 2.80, 1.2 * 2.991
    wIE = parameters.wIE / np.sqrt(parameters.N)  # 1.20, 0.737
    wII = parameters.wII / np.sqrt(parameters.N)  # 4.90, 4.731
    pEE = parameters.pEE
    pEI = parameters.pXX
    pIE = parameters.pXX
    pII = parameters.pXX

    wEEsub = wEE / ((1 / parameters.numClusters) + (1 - (1 / parameters.numClusters)) / WRatioE)  # Average weight for sub - clusters
    pEEsub = pEE / ((1 / parameters.numClusters) + (1 - (1 / parameters.numClusters)) / pRatioE)
    wEE = wEEsub / WRatioE
    pEE = pEEsub / pRatioE

    if parameters.R != 0:
        # wEIsub = wEI / ((1 / parameters.numClusters) + (1 - (1 / parameters.numClusters)) * WRatioI)  # Average weight for sub - clusters
        # pEIsub = pEI / ((1 / parameters.numClusters) + (1 - (1 / parameters.numClusters)) * pRatioI)
        # wEI = wEIsub * WRatioI
        # pEI = pEIsub * pRatioI

        # wIEsub = wIE / ((1 / parameters.numClusters) + (1 - (1 / parameters.numClusters)) / WRatioE)  # Average weight for sub - clusters
        # pIEsub = pIE / ((1 / parameters.numClusters) + (1 - (1 / parameters.numClusters)) / pRatioE)
        # wIE = wIEsub / WRatioE
        # pIE = pIEsub / pRatioE

        wIIsub = wII / ((1 / parameters.numClusters) + (1 - (1 / parameters.numClusters)) / WRatioI)  # Average weight for sub - clusters
        pIIsub = pII / ((1 / parameters.numClusters) + (1 - (1 / parameters.numClusters)) / pRatioI)
        wII = wIIsub / WRatioI
        pII = pIIsub / pRatioI

    weightsEI = np.random.binomial(1, pEI, (parameters.NE, parameters.NI))         # Weight matrix of inhibitory to single compartment excitatory LIF units
    weightsEI = wEI * weightsEI

    weightsIE = np.random.binomial(1, pIE, (parameters.NI, parameters.NE))         # Weight matrix of excitatory to inhibitory cells
    weightsIE = wIE * weightsIE

    weightsII = np.random.binomial(1, pII, (parameters.NI, parameters.NI))         # Weight matrix of inhibitory to inhibitory cells
    weightsII = wII * weightsII

    weightsEE = np.random.binomial(1, pEE, (parameters.NE, parameters.NE))         # Weight matrix of excitatory to excitatory cells
    weightsEE = wEE * weightsEE

    # Create the group weight matrices and update the total weight matrix
    for i in range(parameters.numClusters):
        weightsEEsub = np.random.binomial(1, pEEsub, (parameters.EClusterSize, parameters.EClusterSize))
        weightsEEsub = wEEsub * weightsEEsub
        weightsEE[i * parameters.EClusterSize:(i + 1) * parameters.EClusterSize, i * parameters.EClusterSize:(i + 1) * parameters.EClusterSize] = weightsEEsub

    if parameters.R != 0:
        # Create the group weight matrices for Exc to Inh and update the total weight matrix
        # for i in range(parameters.numClusters):
        #     weightsIEsub = np.random.binomial(1, pIEsub, (parameters.IClusterSize, parameters.EClusterSize))
        #     weightsIEsub = wIEsub * weightsIEsub
        #     weightsIE[i * parameters.IClusterSize:(i+1) * parameters.IClusterSize, i * parameters.EClusterSize:(i+1) * parameters.EClusterSize] = weightsIEsub
        #
        # # Create the group weight matrices for Inh to Exc and update the total weight matrix
        # for i in range(parameters.numClusters):
        #     weightsEIsub = np.random.binomial(1, pEIsub, (parameters.EClusterSize, parameters.IClusterSize))
        #     weightsEIsub = wEIsub * weightsEIsub
        #     weightsEI[i * parameters.EClusterSize:(i+1) * parameters.EClusterSize, i * parameters.IClusterSize:(i+1) * parameters.IClusterSize] = weightsEIsub

        # Create the group weight matrices and update the total weight matrix
        for i in range(parameters.numClusters):
            weightsIIsub = np.random.binomial(1, pIIsub, (parameters.IClusterSize, parameters.IClusterSize))
            weightsIIsub = wIIsub * weightsIIsub
            weightsII[i * parameters.IClusterSize:(i + 1) * parameters.IClusterSize, i * parameters.IClusterSize:(i + 1) * parameters.IClusterSize] = weightsIIsub

    np.fill_diagonal(weightsII, 1.0)
    np.fill_diagonal(weightsEE, -1.0)

    W = np.zeros((parameters.N, parameters.N))
    W[:parameters.NE, :parameters.NE] = weightsEE
    W[parameters.NE:, parameters.NE:] = -weightsII
    W[parameters.NE:, :parameters.NE] = weightsIE
    W[:parameters.NE, parameters.NE:] = -weightsEI

    return W


def get_weight_matrix_Exc(parameters):

    wEE = 0.01 / np.sqrt(parameters.N)

    W = np.random.normal(wEE, 0.1 * wEE, (parameters.N, parameters.N))  # Weight matrix of excitatory to excitatory cells
    # W = W * np.random.binomial(1, parameters.pEE, (parameters.N, parameters.N))

    np.fill_diagonal(W, -24.0*wEE)

    return W

def get_weight_matrix_ExcInh_Cluster_MF_1(parameters):
    WRatioE = parameters.WRatioE  # Ratio of Win / Wout(synaptic weight of within group to neurons outside the group)
    pRatioE = parameters.pRatioE
    WRatioI = 1 + parameters.R * (parameters.WRatioE - 1)
    pRatioI = 1 + parameters.R * (parameters.pRatioE - 1)

    wEE = parameters.wEE / np.sqrt(parameters.N)  # 1.52, 1.165
    wEI = parameters.wEI / np.sqrt(parameters.N)  # 2.80, 1.2 * 2.991
    wIE = parameters.wIE / np.sqrt(parameters.N)  # 1.20, 0.737
    wII = parameters.wII / np.sqrt(parameters.N)  # 4.90, 4.731
    pEE = parameters.pEE
    pEI = parameters.pXX
    pIE = parameters.pXX
    pII = parameters.pXX
    # wSE = 0.011 * np.sqrt(parameters.N)
    # wSI = 4 * 0.011 * np.sqrt(parameters.N)

    wEEsub = wEE / ((1 / parameters.numClusters) + (1 - (1 / parameters.numClusters)) / WRatioE)  # Average weight for sub - clusters
    pEEsub = pEE / ((1 / parameters.numClusters) + (1 - (1 / parameters.numClusters)) / pRatioE)
    wEE = wEEsub / WRatioE
    pEE = pEEsub / pRatioE

    if parameters.R != 0:
        # wEIsub = wEI / ((1 / parameters.numClusters) + (1 - (1 / parameters.numClusters)) * WRatioI)  # Average weight for sub - clusters
        # pEIsub = pEI / ((1 / parameters.numClusters) + (1 - (1 / parameters.numClusters)) * pRatioI)
        # wEI = wEIsub * WRatioI
        # pEI = pEIsub * pRatioI
        #
        # wIEsub = wIE / ((1 / parameters.numClusters) + (1 - (1 / parameters.numClusters)) / WRatioE)  # Average weight for sub - clusters
        # pIEsub = pIE / ((1 / parameters.numClusters) + (1 - (1 / parameters.numClusters)) / pRatioE)
        # wIE = wIEsub / WRatioE
        # pIE = pIEsub / pRatioE

        wIIsub = wII / ((1 / parameters.numClusters) + (1 - (1 / parameters.numClusters)) * WRatioI)  # Average weight for sub - clusters
        pIIsub = pRatioE / ((1 / parameters.numClusters) + (1 - (1 / parameters.numClusters)) * pRatioI)
        wII = wIIsub * WRatioI
        pII = pIIsub * pRatioI
    else:
        wEIsub = wEI
        wIEsub = wIE
        wIIsub = wII
        pEIsub = pEI
        pIEsub = pIE
        pIIsub = pII

    W = np.zeros((2 * parameters.numClusters,  2 * parameters.numClusters))

    for i in range(parameters.numClusters):
        for j in range(parameters.numClusters):
            if i == j:
                W[i, j] = wEEsub * pEEsub * parameters.EClusterSize - 1
                W[i + parameters.numClusters, j] = wIEsub * pIEsub * parameters.EClusterSize
                W[i, j + parameters.numClusters] = -wEIsub * pEIsub * parameters.IClusterSize
                W[i + parameters.numClusters, j + parameters.numClusters] = -wIIsub * pIIsub * parameters.IClusterSize - 1
            else:
                W[i, j] = wEE * pEE * parameters.EClusterSize
                W[i + parameters.numClusters, j] = wIE * pIE * parameters.EClusterSize
                W[i, j + parameters.numClusters] = -wEI * pEI * parameters.IClusterSize
                W[i + parameters.numClusters, j + parameters.numClusters] = -wII * pII * parameters.IClusterSize

    return W

def get_weight_matrix_ExcInh_Cluster_MF(parameters):
    WRatioE = parameters.WRatioE  # Ratio of Win / Wout(synaptic weight of within group to neurons outside the group)
    pRatioE = parameters.pRatioE
    WRatioI = 1 + parameters.R * (parameters.WRatioE - 1)
    pRatioI = 1 + parameters.R * (parameters.pRatioE - 1)

    wEE = parameters.wEE / np.sqrt(parameters.N)  # 1.52, 1.165
    wEI = parameters.wEI / np.sqrt(parameters.N)  # 2.80, 1.2 * 2.991
    wIE = parameters.wIE / np.sqrt(parameters.N)  # 1.20, 0.737
    wII = parameters.wII / np.sqrt(parameters.N)  # 4.90, 4.731
    pEE = parameters.pEE
    pEI = parameters.pXX
    pIE = parameters.pXX
    pII = parameters.pXX
    # wSE = 0.011 * np.sqrt(parameters.N)
    # wSI = 4 * 0.011 * np.sqrt(parameters.N)

    if parameters.numClusters != 1:
        wEEsub = WRatioE * wEE
        pEEsub = pRatioE * parameters.pEE
        wEE = wEE * (parameters.numClusters - WRatioE) / (
                    parameters.numClusters - 1)  # Average weight for sub - clusters
        pEE = parameters.pEE * (parameters.numClusters - pRatioE) / (parameters.numClusters - 1)

        if parameters.R != 0:
            wIEsub = WRatioE * wIE
            pIEsub = pRatioE * parameters.pXX
            wIE = wIE * (parameters.numClusters - WRatioE) / (
                        parameters.numClusters - 1)  # Average weight for sub - clusters
            pIE = parameters.pXX * (parameters.numClusters - pRatioE) / (parameters.numClusters - 1)

            wEIsub = WRatioI * wEI
            pEIsub = pRatioI * parameters.pXX
            wEI = wEI * (parameters.numClusters - WRatioI) / (
                        parameters.numClusters - 1)  # Average weight for sub - clusters
            pEI = parameters.pXX * (parameters.numClusters - pRatioI) / (parameters.numClusters - 1)

            wIIsub = WRatioI * wII
            pIIsub = pRatioI * parameters.pXX
            wII = wII * (parameters.numClusters - WRatioI) / (
                        parameters.numClusters - 1)  # Average weight for sub - clusters
            pII = parameters.pXX * (parameters.numClusters - pRatioI) / (parameters.numClusters - 1)
    else:
        wEEsub = wEE
        wIEsub = wIE
        wEIsub = wEI
        wIIsub = wII

    W = np.zeros((2 * parameters.numClusters, 2 * parameters.numClusters))

    for i in range(parameters.numClusters):
        for j in range(parameters.numClusters):
            if i == j:
                W[i, j] = wEE * parameters.pEE * parameters.EClusterSize - 1.0
                W[i + parameters.numClusters, j] = wIEsub * parameters.pXX * parameters.EClusterSize
                W[i, j + parameters.numClusters] = -wEIsub * parameters.pXX * parameters.IClusterSize
                W[i + parameters.numClusters, j + parameters.numClusters] = -wII * parameters.pXX * parameters.IClusterSize - 1.0
            else:
                W[i, j] = wEE * parameters.pEE * parameters.EClusterSize
                W[i + parameters.numClusters, j] = wIE * parameters.pXX * parameters.EClusterSize
                W[i, j + parameters.numClusters] = -wEI * parameters.pXX * parameters.IClusterSize
                W[i + parameters.numClusters, j + parameters.numClusters] = -wII * parameters.pXX * parameters.IClusterSize

    return W

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
    sigma = 0.1 / parameters.N

    wEEsub = wEE / ((1 / parameters.numClusters) + (1 - (1 / parameters.numClusters)) / WRatioE)  # Average weight for sub - clusters
    pEEsub = pEE / ((1 / parameters.numClusters) + (1 - (1 / parameters.numClusters)) / pRatioE)
    wEE = wEEsub / WRatioE
    pEE = pEEsub / pRatioE

    if parameters.R != 0:
        # wEIsub = wEI / ((1 / parameters.numClusters) + (1 - (1 / parameters.numClusters)) * WRatioI)  # Average weight for sub - clusters
        # pEIsub = pEI / ((1 / parameters.numClusters) + (1 - (1 / parameters.numClusters)) * pRatioI)
        # wEI = wEIsub * WRatioI
        # pEI = pEIsub * pRatioI

        # wIEsub = wIE / ((1 / parameters.numClusters) + (1 - (1 / parameters.numClusters)) / WRatioE)  # Average weight for sub - clusters
        # pIEsub = pIE / ((1 / parameters.numClusters) + (1 - (1 / parameters.numClusters)) / pRatioE)
        # wIE = wIEsub / WRatioE
        # pIE = pIEsub / pRatioE

        wIIsub = wII / ((1 / parameters.numClusters) + (1 - (1 / parameters.numClusters)) * WRatioI)  # Average weight for sub - clusters
        pIIsub = pII / ((1 / parameters.numClusters) + (1 - (1 / parameters.numClusters)) * pRatioI)
        wII = wIIsub * WRatioI
        pII = pIIsub * pRatioI

    weightsEI = np.random.binomial(1, pEI, (parameters.NE, parameters.NI))  # Weight matrix of inhibitory to single compartment excitatory LIF units
    weightsEI = wEI * weightsEI

    weightsIE = np.random.binomial(1, pIE, (parameters.NI, parameters.NE))  # Weight matrix of excitatory to inhibitory cells
    weightsIE = wIE * weightsIE

    weightsII = np.random.binomial(1, pII, (parameters.NI, parameters.NI))  # Weight matrix of inhibitory to inhibitory cells
    weightsII = wII * weightsII

    weightsEE = np.random.binomial(1, pEE, (parameters.NE, parameters.NE))  # Weight matrix of excitatory to excitatory cells
    weightsEE = wEE * weightsEE

    # Create the group weight matrices and update the total weight matrix
    for i in range(parameters.numClusters):
        weightsEEsub = np.random.binomial(1, pEEsub, (parameters.EClusterSize, parameters.EClusterSize))
        weightsEEsub = wEEsub * weightsEEsub
        weightsEE[i * parameters.EClusterSize:(i + 1) * parameters.EClusterSize, i * parameters.EClusterSize:(i + 1) * parameters.EClusterSize] = weightsEEsub

    if parameters.R != 0:
        # Create the group weight matrices for Exc to Inh and update the total weight matrix
        # for i in range(parameters.numClusters):
        #     weightsIEsub = np.random.binomial(1, pIEsub, (parameters.IClusterSize, parameters.EClusterSize))
        #     weightsIEsub = wIEsub * weightsIEsub
        #     weightsIE[i * parameters.IClusterSize:(i+1) * parameters.IClusterSize, i * parameters.EClusterSize:(i+1) * parameters.EClusterSize] = weightsIEsub
        #
        # # Create the group weight matrices for Inh to Exc and update the total weight matrix
        # for i in range(parameters.numClusters):
        #     weightsEIsub = np.random.binomial(1, pEIsub, (parameters.EClusterSize, parameters.IClusterSize))
        #     weightsEIsub = wEIsub * weightsEIsub
        #     weightsEI[i * parameters.EClusterSize:(i+1) * parameters.EClusterSize, i * parameters.IClusterSize:(i+1) * parameters.IClusterSize] = weightsEIsub

        # Create the group weight matrices and update the total weight matrix
        for i in range(parameters.numClusters):
            weightsIIsub = np.random.binomial(1, pIIsub, (parameters.IClusterSize, parameters.IClusterSize))
            weightsIIsub = wIIsub * weightsIIsub
            weightsII[i * parameters.IClusterSize:(i + 1) * parameters.IClusterSize, i * parameters.IClusterSize:(i + 1) * parameters.IClusterSize] = weightsIIsub

    np.fill_diagonal(weightsII, 1.0)
    np.fill_diagonal(weightsEE, -1.0)

    W = np.zeros((parameters.N, parameters.N))
    W[:parameters.NE, :parameters.NE] = weightsEE
    W[parameters.NE:, parameters.NE:] = -weightsII
    W[parameters.NE:, :parameters.NE] = weightsIE
    W[:parameters.NE, parameters.NE:] = -weightsEI

    return W
