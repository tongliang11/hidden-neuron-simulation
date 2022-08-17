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



def neg_loglikelihood(coef, X, y, weights):
    y_pred, deri = y_pred_likelihood_derivative(
        coef, X, y, weights)
    dev = family.deviance(y, y_pred, weights)
    nlll = np.sum(y_pred - y * np.log(y_pred))
    # offset if coef[0] is intercept
    offset = 1 if self.fit_intercept else 0
    coef_scaled = alpha * coef[offset:]
    obj = 0.5 * dev + 0.5 * (coef[offset:] @ coef_scaled)
    objp = 0.5 * devp
    objp[offset:] += coef_scaled
    return obj, objp

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