from parameters import network_params
from weight_matrix import get_weight_matrix_ExcInh_Cluster_1
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    params = network_params()
    weight = get_weight_matrix_ExcInh_Cluster_1(params)
    print(weight)
    print(weight.shape)
    np.savetxt("e-i-weight_64.txt", weight)
    plt.imshow(weight)
    plt.colorbar()
    plt.savefig("e-i-weight.png")