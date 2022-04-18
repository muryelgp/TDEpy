import numpy as np
import scipy.optimize as op
from astropy.constants import h, c, k_B, sigma_sb
import astropy.units as u
import emcee
from TDEpy import light_curves
from multiprocessing import Pool
from TDEpy import plots as plots

class BB_FT_FS:

    def __init__(self):
        self.model_name = 'BB_FT_FS'
        pass

    def gen_theta_init(self, observable, good_epoch):
        size = int(np.sum(good_epoch[good_epoch==True]))
        theta_init = np.array([[4.2, 44] for i in range(size)])
        return theta_init

    def gen_good_epoch(self, observable):
        sn_epochs = len(observable.epochs)
        good_epochs = np.ones(sn_epochs, dtype='bool')
        uv_index = (observable.bands == 'sw_w1') | (observable.bands == 'sw_m2') | (observable.bands == 'sw_w2')
        for i in range(sn_epochs):
            if np.isnan(observable.sed[i, uv_index]).all():
                good_epochs[i] = False
            else:
                if np.sum(np.isfinite(observable.sed[i, :])) >= 2:
                    good_epochs[i] = True
                else:
                    good_epochs[i] = False
        return good_epochs

    def minimize(self, observable, good_epoch, theta_init):
        theta = theta_init

        for i in range(len(theta)):
            nll = lambda *args: -lnlike(*args)
            bounds = [(4, 5), (41, 46)]
            #good_epoch = good_epoch == True
            epoch = observable.epochs[good_epoch][i]
            epoch_index = np.where(observable.epochs == epoch)[0][0]
            sed = observable.sed[epoch_index, :]
            sed_err = observable.sed_err[epoch_index, :]
            wl = observable.band_wls
            result = op.minimize(nll, theta_init[i], args=(wl, sed, sed_err), bounds=bounds, method='Powell')
            theta[i] = result["x"]
        return theta

    def run_mcmc(self, observable, good_epoch, theta_init, mcmc_args):
        log_T, log_T_err, log_BB, log_BB_err, log_R, log_R_err = [], [], [], [], [], []
        sed_chain = []
        wl_list = np.linspace(1e3, 1e4, 1000)

        n_walkers = mcmc_args['n_walkers']
        n_inter = mcmc_args['n_inter']
        n_burn = mcmc_args['n_burn']
        n_cores = mcmc_args['n_cores']
        ndim, nwalkers = 2, n_walkers
        for i in range(len(theta_init)):

            pos = [[np.random.normal(theta_init[i][0], theta_init[i][0]*0.1),
                np.random.normal(theta_init[i][1], theta_init[i][1]*0.1)] for j in range(nwalkers)]


            #good_epoch = good_epoch == True
            epoch = observable.epochs[good_epoch][i]
            epoch_index = np.where(observable.epochs == epoch)[0][0]
            sed = observable.sed[epoch_index, :]
            sed_err = observable.sed_err[epoch_index, :]
            wl = observable.band_wls
            with Pool(int(n_cores)) as pool:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(wl, sed, sed_err), pool=pool)
                sampler.run_mcmc(pos, int(n_inter / 2), progress=True, skip_initial_state_check=True)
            samples = sampler.chain[:, int(n_burn / 2):, :].reshape((-1, ndim))

            T, L = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                                             zip(*np.percentile(samples, [16, 50, 84], axis=0)))
            L_sample = np.random.normal(L[0],  np.mean(L[1:]), 100)
            T_sample = np.random.normal(T[0], np.mean(T[1:]), 100)
            R_sample = np.log10(np.sqrt((10 ** L_sample) / (4. * 3.14 * 5.6704e-5 * (10 ** T_sample) ** 4)))
            R = list(map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                         [np.percentile(R_sample, [16, 50, 84])]))[0]

            log_T_i, log_T_err_i = T[0], np.mean(T[1:])
            log_L_i, log_L_err_i = L[0], np.mean(L[1:])
            log_R_i, log_R_err_i = R[0], np.mean(R[1:])
            log_T.append(log_T_i)
            log_T_err.append(log_T_err_i)
            log_BB.append(log_L_i)
            log_BB_err.append(log_L_err_i)
            log_R.append(log_R_i)
            log_R_err.append(log_R_err_i)

            sed_chain_i = [model_expression(wl_list, np.random.normal(log_T_i, log_T_err_i), np.random.normal(log_L_i, log_L_err_i)) for h in range(100)]
            sed_chain.append(sed_chain_i)

        BB_evol = log_T, log_T_err, log_BB, log_BB_err, log_R, log_R_err
        plots.plot_SEDs(self.model_name, observable, sed_chain, log_T, log_T_err, good_epoch)
        light_curves.save_blackbody_evol(self.model_name, observable.epochs[good_epoch], BB_evol, good_epoch)


def model_expression(wl, log_T, log_L):
    wl = wl * u.Angstrom
    T = (10 ** log_T) * u.K
    nu = c.cgs / wl.cgs
    flux_wl = ((2 * nu.cgs ** 3 * h.cgs) / c.cgs ** 2) / (np.exp((h.cgs * nu) / (k_B.cgs * T.cgs)) - 1).cgs*nu
    lum_sed = (10 ** log_L) * ((np.pi * flux_wl) / (sigma_sb.cgs * ((T.value * u.K) ** 4)).cgs).cgs.value
    return lum_sed

def lnlike(theta, wl, sed, sed_err):
    T, L = theta
    model = model_expression(wl, T, L)
    inv_sigma2 = 1.0 / (sed_err ** 2.)
    return -0.5 * (np.nansum(inv_sigma2 * (sed - model) ** 2.))

def lnprior(theta):
    T, L = theta
    T_prior = 4 <= T <= 5
    L_prior = 41 <= L <= 48
    if T_prior and L_prior:
        return 0.0
    else:
        return -np.inf


def lnprob(theta, wl, sed, sed_err):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    else:
        prob = lp + lnlike(theta, wl, sed, sed_err)
        return prob
