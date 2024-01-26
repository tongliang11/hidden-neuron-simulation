# Simulations of GLM spiking neural network and MLE connection inference
This github repository contains notebooks and scripts for the results presented in 
"[Statistically inferred neuronal connections in subsampled neural networks strongly correlate with spike train covariance](https://www.biorxiv.org/content/10.1101/2023.02.01.526673v1)," 
Tong Liang and Braden A. W. Brinkman.

The reorganized version of the repository will be available upon acceptance of the manuscript.

## Simulating point process generalized linear model for spiking neurons

`./src/main.py` can be used to simulate the spike train process either with given weight matrix or generate a random network weight matrix. It imports
code from `./src/glm/spk_train` for simulating the point process generalized linear model.

`./src/cov_mle.py` is used to calculate the spike train covariances with Python's `multiprocessing` module.

`./src/filter_inference.py` is used to infer neuronal connections based on the recorded spike trains and calculate the Pearson correlation between the infered
neuronal connections and the corresponding spike train covariances. It imports code from `./src/mle/inference` forthe maximum
likelihood estimation. 
