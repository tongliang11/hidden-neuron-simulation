import numpy as np
import scipy

def neg_loglikelihood(theta, X, y, mu):
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
        
        # print("y_hat shape", y_hat.shape)
        grad = 1. / n_samples * np.array([np.sum((y_hat - y) * X[:, i]) for i in range(n_features)])

        return nlogL, grad


def minimize(func, initial_theta, tol=1e-4, args=None):
    opt_res = scipy.optimize.minimize(
                func,
                initial_theta,
                method="L-BFGS-B",
                jac=True,
                options={
                    "maxiter": 1000,
                    # "maxls": 50,  # default is 20
                    "iprint": 0,
                    "gtol": tol,
                    # The constant 64 was found empirically to pass the test suite.
                    # The point is that ftol is very small, but a bit larger than
                    # machine precision for float64, which is the dtype used by lbfgs.
                    "ftol": 1e3 * np.finfo(float).eps,
                },
                args=args
            )
    
    print(opt_res)
    # n_iterations = self._check_optimize_result('lbfgs', opt_res)
    # nll = self.neg_loglikelihood(X=design_matrix, y=spike_train[:, to_neuron], theta=opt_res.x, mu=0)
    # print("nll niter", n_iterations)
    # print("theta_initiala", theta_inital)
    # print("nll after minimization", nll)
    # if test_x is not None:
    #     print("nll with J_01 fit", self.neg_loglikelihood(X=design_matrix, y=spike_train[:, to_neuron], theta=test_x, mu=0))

    # if fit_intercept:
    #     return opt_res.x[:-1], opt_res.x[-1]
    # else:
    #     return opt_res.x, 0

if __name__ == "__main__":
    initial_theta = np.random.normal(size=(100,4))
    minimize(neg_loglikelihood,initial_theta=initial_theta, args={X, y, mu})