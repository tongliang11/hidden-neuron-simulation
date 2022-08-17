# class Poisson_GLM:
#     def __init__(self, *, alpha=1.0,
#                  fit_intercept=True, family='normal', link='auto',
#                  solver='lbfgs', max_iter=100, tol=1e-4, warm_start=False,
#                  verbose=0):
#         self.alpha = alpha
#         self.fit_intercept = fit_intercept
#         self.family = family
#         self.link = link
#         self.solver = solver
#         self.max_iter = max_iter
#         self.tol = tol
#         self.warm_start = warm_start
#         self.verbose = verbose
         
#     def fit(self):

import scipy
import numpy as np


def y_pred_likelihood_derivative(coef, X, y, weights):
    y_pred = np.exp(X @ coef)

    return y_pred, deri



def neg_loglikelihood(X, y, theta, mu):
    """Calculate the negative log likelihood function, given design matrix X and target y for parameters theta
    Args:
        X (2d array):
            X is the design matrix in shape (n_observations, filter_length * n_observed_neurons)
        
        y (1d array):
            y is the target with shape (n_observations, )
        
        theta (1d array):
            theta is the filter parameters with shape (filter_length, )
        
        mu (float):
            baseline firing rate

    Return:
        negative log likelihood, gradient wrt each component of theta
    
    """
    n_samples, n_features = X.shape
    z = X @ theta
    y_hat = np.exp(z + mu)
    eps = np.spacing(1)
    nlogL = - 1. / n_samples * np.sum(y * np.log(y_hat + eps) - y_hat)

    grad = [np.sum((y_hat - y) * z[i]) for i in range(n_features)]

    return nlogL, grad

args = (X, y, weights, self.alpha, family, link)

opt_res = scipy.optimize.minimize(
    func, coef, method="L-BFGS-B", jac=True,
    options={
        "maxiter": self.max_iter,
        "iprint": (self.verbose > 0) - 1,
        "gtol": self.tol,
        "ftol": 1e3*np.finfo(float).eps,
    },
    args=args)
self.n_iter_ = _check_optimze_result("lbfgs", opt_res)
coef = opt_res.xi

self.hess = opt_res.hess_inv.todense()
if self.fit_intercept:
    self.intercept_ = coef[0]
    self.coef_ = coef[1:]
else:
    # set intercept to zero as the other linear models do
    self.intercept_ = 0.
    self.coef_ = coef