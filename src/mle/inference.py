import numpy as np
from sklearn.linear_model import TweedieRegressor
from scipy.linalg import block_diag, hankel
import matplotlib.pyplot as plt
import logging

# to-do: fix dt=0.001 and filter_length problem
# to-do: implement mle with np.minimize by writing down the loglikelihood function explicitly


class Maximum_likelihood_estimator:
    def __init__(self, filter_length, alpha=0, dt=0.1, tau=1.0, basis_order=[0, 1, 2], observed=[0]) -> None:
        self.filter_length = filter_length
        self.dt = dt
        self.tau = tau
        self.basis_order = basis_order
        self.observed = observed
        self.alpha = alpha  # regularization alpha = 0 is for no regularization

    def alpha_basis(self):
        alpha_filters = np.array([[1/(np.math.factorial(i)*self.tau**(i+1))*j**i*np.exp(-j/self.tau)
                                   for j in np.arange(self.filter_length)*self.dt] for i in self.basis_order]).T

        return alpha_filters[::-1]

    def laguerre_basis(self):
        # l0 = lambda x: 1 * np.exp(-x/2)
        # l1 = lambda x: (-x + 1) * np.exp(-x/2)
        # l2 = lambda x: 1/2 * (x**2 - 4*x + 2) * np.exp(-x/2)
        # l3 = lambda x: 1/6 * (-x**3 + 9*x**2 - 18*x + 6) * np.exp(-x/2)
        def l0(x): return 1 * np.exp(-x)

        def l1(x): return (-2*x + 1) * np.exp(-x)

        def l2(x): return 1/2 * (4*x**2 - 8*x + 2) * np.exp(-x)

        def l3(x): return 1/6 * (-8*x**3 + 36*x**2 - 36*x + 6) * np.exp(-x)
        # def l0(t):
        #     return 1 * np.exp(-t/2)
        #
        basis = np.array([[l0(t) for t in np.arange(self.filter_length)*self.dt],
                          [l1(t)
                              for t in np.arange(self.filter_length)*self.dt],
                          [l2(t)
                              for t in np.arange(self.filter_length)*self.dt],
                          [l3(t) for t in np.arange(self.filter_length)*self.dt]]).T
        # basis = np.array([[eval(f"l{i}(t)") for t in  np.arange(100)*0.1] for i in [0,1,2,3]]).T

        return basis[::-1]

    def raised_cos_basis():
        a = 2  # create raised cosine basis
        c = 0
        phi = np.linspace(0, 5, 10)
        rcos = np.array([(np.cos(np.minimum(np.maximum(
            a*np.log(i+c)-phi, -np.pi), np.pi))+1)/2 for i in np.linspace(0, 50, 50)])
        return rcos

    def design_matrix(self, spike_train, exclude_self=False, to_neuron=None):
        if not exclude_self:
            spk_train_padding = np.vstack(
                (np.zeros((self.filter_length, len(self.observed))), spike_train[:, self.observed]))
            full_design = np.hstack(np.array([hankel(
                spk_train_padding[:, i][:-self.filter_length], spk_train_padding[:, i][-self.filter_length-1:-1]) for i in range(len(self.observed))]))
        else:
            spk_train_padding = np.vstack(
                (np.zeros((self.filter_length, len(self.observed)-1)), spike_train[:, [obs for obs in self.observed if obs != to_neuron]]))
            full_design = np.hstack(np.array([hankel(
                spk_train_padding[:, i][:-self.filter_length], spk_train_padding[:, i][-self.filter_length-1:-1]) for i in range(len(self.observed)-1)]))

        return full_design

    def fit_basis(self, spike_train, to_neuron, basis, tol=1e-4):
        diag_basis = block_diag(*[basis for _ in range(len(self.observed))])
        design_with_basis = self.design_matrix(spike_train) @ diag_basis
        sklearnm = TweedieRegressor(
            power=1, alpha=self.alpha, max_iter=200000, link='log', tol=tol, fit_intercept=True)
        sklearnm.fit(X=design_with_basis, y=spike_train[:, to_neuron])
        return (sklearnm, basis)

    def fit_basis_no_self_coupling(self, spike_train, to_neuron, basis, tol=1e-4):
        diag_basis = block_diag(*[basis for _ in range(len(self.observed)-1)])
        design_with_basis = self.design_matrix(spike_train, exclude_self=True, to_neuron=to_neuron) @ diag_basis
        sklearnm = TweedieRegressor(
            power=1, alpha=self.alpha, max_iter=200000, link='log', tol=tol, fit_intercept=True)
        sklearnm.fit(X=design_with_basis, y=spike_train[:, to_neuron])
        return (sklearnm, basis)

    def fit_basis_free(self, spike_train, to_neuron, tol=1e-4):
        sklearnm = TweedieRegressor(
            power=1, alpha=self.alpha, max_iter=200000, link='log', tol=tol, fit_intercept=True)
        sklearnm.fit(X=self.design_matrix(spike_train),
                     y=spike_train[:, to_neuron])
        return sklearnm

    def infer_J_ij(self, i, j, spike_train):
        return self.fit_basis_free(spike_train, j).coef_[
                        i*self.filter_length:i*self.filter_length+self.filter_length][::-1]

    def infer_J_ij_basis(self, i, j, spike_train, tol=1e-4, exclude_self_coupling=False):
        # return self.fit_basis_free(spike_train, j).coef_[
        #                 i*self.filter_length:i*self.filter_length+self.filter_length][::-1]
        if not exclude_self_coupling:
            fitted_with_basis, basis = self.fit_basis(
                        spike_train, to_neuron=self.observed[j], basis=self.alpha_basis(), tol=tol)
            return (basis@fitted_with_basis.coef_[i*len(self.basis_order):i*len(self.basis_order)+len(
                        self.basis_order)])[::-1], fitted_with_basis.hess, fitted_with_basis.intercept_
            # return fitted_with_basis
        else:
            fitted_with_basis, basis = self.fit_basis_no_self_coupling(
                        spike_train, to_neuron=self.observed[j], basis=self.alpha_basis(), tol=tol)
            if i == j:
                return [0]*len(basis), None, None
            elif i > j:
                i -= 1
            # return fitted_with_basis
            return (basis@fitted_with_basis.coef_[i*len(self.basis_order):i*len(self.basis_order)+len(
                        self.basis_order)])[::-1], fitted_with_basis.hess, fitted_with_basis.intercept_


    def infer_weight_matrix(self, spike_train, tol=1e-4):
        weight_matrix = []
        for j in range(len(self.observed)):
            fitted_with_basis, basis = self.fit_basis(
                        spike_train, to_neuron=self.observed[j], basis=self.alpha_basis(), tol=tol)
            weight_matrix.append(fitted_with_basis.coef_)
            print(f"{j}-th neuron weights inferred:", fitted_with_basis.coef_)
        return weight_matrix


    def plot_inferred(self, spike_train, W_true, ylim=0.3, basis_free_infer=True, basis_type='alpha', legend=True, savefig=False, figname='Fig'):
        fig, axs = plt.subplots(len(self.observed), len(
            self.observed), figsize=(9, 9), dpi=150)
        if len(self.observed) == 1:
            axs = np.array([[axs]])
        for j in range(len(self.observed)):
            if basis_free_infer:
                fitted_basis_free = self.fit_basis_free(
                    spike_train, to_neuron=self.observed[j])
            if basis_type == 'alpha':
                fitted_with_basis, basis = self.fit_basis(
                    spike_train, to_neuron=self.observed[j], basis=self.alpha_basis())
                print(
                    f"coeff after fitting for neuron {j} are {fitted_with_basis.coef_}")
            elif basis_type == 'laguerre':
                fitted_with_basis, basis = self.fit_basis(
                    spike_train, to_neuron=self.observed[j], basis=self.laguerre_basis())
            for i in range(len(self.observed)):
                if basis_free_infer:
                    axs[i, j].scatter(np.arange(self.filter_length) * self.dt, fitted_basis_free.coef_[
                        i*self.filter_length:i*self.filter_length+self.filter_length][::-1], s=5, color='red', label="Inferred w/o basis")

                axs[i, j].plot(np.arange(self.filter_length) * self.dt, (basis@fitted_with_basis.coef_[i*len(self.basis_order):i*len(self.basis_order)+len(
                    self.basis_order)])[::-1], linewidth=5, label=r"Inferred $J^{{eff}}_{{{},{}}}$ w/ basis".format(self.observed[j], self.observed[i]), color='red')
                axs[i, j].set_ylim(-ylim, ylim)
                alpha_filter = [W_true[self.observed[j], self.observed[i]]*1 /
                                self.tau**2*k*np.exp(-k/self.tau) for k in np.arange(self.filter_length)*self.dt]
                axs[i, j].plot(np.arange(self.filter_length) * self.dt,
                               alpha_filter, '--', linewidth=5, label=r"Ground-truth $J_{{{},{}}}$".format(self.observed[j], self.observed[i]))

                if legend:
                    handles, labels = axs[i, j].get_legend_handles_labels()
                    order = [0, 2, 1]
                    axs[i, j].legend([handles[idx] for idx in order], [
                                     labels[idx] for idx in order], frameon=False, loc="upper right", prop={'size': 10})
                if j > 0:
                    axs[i, j].set_yticklabels([])
                if i < (len(self.observed)-1):
                    axs[i, j].set_xticklabels([])
            # plt.text(0,1,'Time',size=15)
                # axs[i, j].set_ylabel('Filter Strength',size=15)
        fig.suptitle('MLE inferred effective coupling filters', size=18)
        fig.text(0.5, -0.01, 'Time Preceding (s)', ha='center', size=18)
        fig.text(-0.01, 0.5, 'Filter Strength (a.u.)',
                 va='center', rotation='vertical', size=18)
        fig.tight_layout()
        if savefig:
            fig.savefig(f'./Figures/{figname}.pdf', bbox_inches="tight")
