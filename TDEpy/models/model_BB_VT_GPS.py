import numpy as np
import scipy.optimize as op
from astropy.constants import h, c, k_B, sigma_sb
import astropy.units as u
import emcee
from TDEpy import light_curves
from multiprocessing import Pool
from TDEpy import plots as plots


class BB_VT_GPS:

    def __init__(self, T_step):
        self.model_name = 'BB_VT_GPS'
        self.T_step = T_step
        pass

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

    def gen_theta_init(self, observable, good_epoch):

        n_T = len(np.arange(-60, 301, self.T_step))

        t_peak_init = observable.epochs[np.where(observable.sed == np.nanmax(observable.sed[:, 0]))[0]][0]
        theta_init = [44, t_peak_init, 10, 10, 5/3] #log_L_BB_peak, t_peak, sigma, t0, p, *T_grid
        for i in range(n_T):
            theta_init.append(4.3)
        return theta_init

    def minimize(self, observable, good_epoch, theta_init):


        n_T = len(np.arange(-60, 301, self.T_step))
        nll = lambda *args: -self.lnlike(*args)
        t_peak_init = observable.epochs[np.where(observable.sed == np.nanmax(observable.sed[:, 0]))[0]][0]
        bounds = [(41, 48), (observable.epochs[0], t_peak_init+10), (0, 10**1.5), (0, 15), (0.0, 2)]
        for i in range(n_T):
            bounds.append((4.1, 4.7))

        sed = observable.sed[:, :]
        sed_err = observable.sed_err[:, :]
        wl = observable.band_wls
        epochs = observable.epochs

        result = op.minimize(nll, theta_init, args=(wl, epochs, sed, sed_err, good_epoch), bounds=bounds, method='Powell')
        theta = result["x"]
        return theta

    def run_mcmc(self, observable, good_epoch, theta_init, mcmc_args):

        wl_list = np.linspace(1e3, 1e4, 1000)
        n_T = len(np.arange(-60, 301, self.T_step))
        n_walkers = mcmc_args['n_walkers']
        n_inter = mcmc_args['n_inter']
        n_burn = mcmc_args['n_burn']
        n_cores = mcmc_args['n_cores']

        log_L_BB_opt, t_peak_opt, sigma_opt, t0_opt, p_opt, *Ts_opt = theta_init  # will be used to initialise the walkers

        # Posterior emcee sampling
        ndim, nwalkers = int(5 + n_T), n_walkers
        pos = [np.concatenate(([np.random.normal(log_L_BB_opt, 0.1),
                                np.random.normal(t_peak_opt, 2),
                                np.random.normal(sigma_opt, 2),
                                np.random.normal(t0_opt, 2),
                                np.random.normal(p_opt, 0.1)],
                               [Ts_opt[j] + np.random.normal(0, 0.05) for j in range(n_T)])) for i in range(n_walkers)]

        sed = observable.sed[:, :]
        sed_err = observable.sed_err[:, :]
        wl = observable.band_wls
        epochs = observable.epochs

        with Pool(int(n_cores)) as pool:
            sampler = emcee.EnsembleSampler(n_walkers, ndim, self.lnprob, args=(wl, epochs, sed, sed_err, good_epoch), pool=pool)
            sampler.run_mcmc(pos, n_inter, progress=True, skip_initial_state_check=True)

        samples = sampler.chain[:, n_burn:, :].reshape((-1, ndim))


        L_BB_peak, t_peak, sigma, t0, p, *T_grid = map(lambda v: [v[1], np.nanmean([v[2] - v[1], v[1] - v[0]])],
                                                          zip(*np.nanpercentile(samples, [16, 50, 84], axis=0)))


        T_grid = [T_grid[i][0] for i in range(len(T_grid))], [T_grid[i][1] for i in range(len(T_grid))]
        t_grid = t_peak[0] + np.arange(-60, 301, self.T_step)
        flag_T_grid = self.gen_flag_T_grid(epochs, good_epoch, t_grid, T_grid[0], self.T_step)
        log_T = np.interp(epochs, t_grid[flag_T_grid], np.array(T_grid[0])[flag_T_grid])
        log_T = np.interp(epochs, epochs[good_epoch], log_T[good_epoch])

        log_T_err = np.interp(epochs, t_grid[flag_T_grid], np.array(T_grid[1])[flag_T_grid])
        log_T_err = np.interp(epochs, epochs[good_epoch], log_T_err[good_epoch])

        log_L = np.zeros(np.shape(epochs))
        delt_t = epochs - t_peak[0]

        before_peak = delt_t <= 0
        log_L[before_peak] = np.log10((10 ** L_BB_peak[0]) * np.exp(-1 * (delt_t[before_peak]) ** 2 / (2 * sigma[0] ** 2)))
        after_peak = delt_t > 0
        log_L[after_peak] = np.log10((10 ** L_BB_peak[0]) * ((delt_t[after_peak] + t0[0]) / t0[0]) ** (-1 * p[0]))


        log_L_i = np.zeros((np.shape(epochs)[0], 100))

        for j in range(100):
            L_BB_peak_j = np.random.normal(L_BB_peak[0], L_BB_peak[1])
            sigma_j = np.random.normal(sigma[0], sigma[1])
            p_j = np.random.normal(p[0], p[1])
            t0_j = np.random.normal(t0[0], t0[1])


            log_L_i[before_peak, j] = np.log10((10 ** L_BB_peak_j) * np.exp(-1 * (delt_t[before_peak]) ** 2 / (2 * sigma_j ** 2)))
            log_L_i[after_peak, j] = np.log10((10 ** L_BB_peak_j) * ((delt_t[after_peak] + t0_j) / t0_j) ** (-1 * p_j))

        log_L_err = [np.nanmean([np.nanpercentile(log_L_i[i], 84) - np.nanpercentile(log_L_i[i], 50),
                                    np.nanpercentile(log_L_i[i], 50) - np.nanpercentile(log_L_i[i], 16)]) for i in range(len(epochs))]

        log_R, log_R_err = [], []
        for i in range(len(epochs)):
            L_sample = np.random.normal(log_L[i], log_L_err[i], 100)
            T_sample = np.random.normal(log_T[i], log_T_err[i], 100)
            R_sample = np.log10(np.sqrt((10 ** L_sample) / (4. * 3.14 * 5.6704e-5 * (10 ** T_sample) ** 4)))
            log_R.append(np.median(R_sample))
            log_R_err.append(np.nanmean([np.percentile(R_sample, 84) - np.percentile(R_sample, 50),
                                    np.percentile(R_sample, 50) - np.percentile(R_sample, 16)]))


        BB_evol = log_T, log_T_err, log_L, log_L_err, log_R, log_R_err
        sed_chain = []
        for i in range(len(epochs)):
            sed_chain_i = [self.BB_SED(wl_list, np.random.normal(log_T[i], log_T_err[i]), np.random.normal(log_L[i], log_L_err[i])) for h in range(100)]
            sed_chain.append(sed_chain_i)

        plots.plot_SEDs(self.model_name, observable, sed_chain, log_T, log_T_err, good_epoch)
        log_T[~good_epoch] = np.nan
        log_T_err[~good_epoch] = np.nan
        light_curves.save_blackbody_evol(self.model_name, observable.epochs, BB_evol, good_epoch)

        L_BB_peak, t_peak, sigma, t0, p, *T_grid = map(lambda v: (v[1], v[1] - v[0], v[2] - v[1]), zip(*np.nanpercentile(samples, [16, 50, 84], axis=0)))

        g = open('best_fitting_pars.txt', 'w')
        g.write('Parameter' + '\t' + 'median' + '\t' + 'p16' + 'p84' + '\n')
        g.write('L_BB_peak' + '\t' + '{:.2f}'.format(L_BB_peak[0]) + '\t' + '{:.2f}'.format(L_BB_peak[1]) + '\t' + '{:.2f}'.format(L_BB_peak[2]) + '\n')
        g.write('t_peak' + '\t' + '{:.2f}'.format(t_peak[0]) + '\t' + '{:.2f}'.format(t_peak[1]) + '\t' + '{:.2f}'.format(t_peak[2]) + '\n')
        g.write('sigma' + '\t' + '{:.2f}'.format(sigma[0]) + '\t' + '{:.2f}'.format(sigma[1]) + '\t' +'{:.2f}'.format(sigma[2]) + '\n')
        g.write('t0' + '\t' + '{:.2f}'.format(t0[0]) + '\t' + '{:.2f}'.format(t0[1]) + '\t' + '{:.2f}'.format(t0[2]) + '\n')
        g.write('p' + '\t' + '{:.2f}'.format(p[0]) + '\t' + '{:.2f}'.format(p[1]) + '\t' + '{:.2f}'.format(p[2]) + '\n')
        g.close()

    def BB_SED(self, wl, log_T, log_L):
        wl = wl * u.Angstrom
        T = (10 ** log_T) * u.K
        nu = c.cgs / wl.cgs
        flux_wl = ((2 * nu.cgs ** 3 * h.cgs) / c.cgs ** 2) / (np.exp((h.cgs * nu) / (k_B.cgs * T.cgs)) - 1).cgs*nu
        lum_sed = (10 ** log_L) * ((np.pi * flux_wl) / (sigma_sb.cgs * ((T.value * u.K) ** 4)).cgs).cgs.value
        return lum_sed

    def model_expression(self, theta, wl, epochs, good_epochs):
        log_L_BB_peak, t_peak, sigma, t0, p, *T_grid = theta

        t_array = np.tile(epochs, (len(wl), 1)).transpose()
        light_curve_shape = np.zeros(np.shape(t_array))
        delt_t = t_array - t_peak

        before_peak = delt_t <= 0
        light_curve_shape[before_peak] = np.exp(-1 * (delt_t[before_peak]) ** 2 / (2 * sigma ** 2))

        after_peak = delt_t > 0
        light_curve_shape[after_peak] = ((delt_t[after_peak] + t0) / t0) ** (-1 * p)

        t_grid = t_peak + np.arange(-60, 301, self.T_step)
        flag_T_grid = self.gen_flag_T_grid(epochs, good_epochs, t_grid, T_grid, self.T_step)
        T_t = np.interp(epochs, t_grid[flag_T_grid], np.array(T_grid)[flag_T_grid])
        T_t = np.interp(epochs, epochs[~good_epochs], T_t[~good_epochs])

        wl_array = np.tile(wl, (len(epochs), 1))
        T_t_array = np.tile(T_t, (len(wl), 1)).transpose()

        BB_light_curve = 10**log_L_BB_peak * light_curve_shape

        model = self.BB_SED(wl_array, T_t_array, np.log10(BB_light_curve))

        return model

    def gen_flag_T_grid(self, t, good_epochs, t_grid, T_grid, T_interval):
        flag_left_T_grid = np.concatenate(([np.sum(
            (t[good_epochs] >= t_grid[i]) & (t[good_epochs] < t_grid[i] + T_interval)) > 0 for i in
                                            range(len(T_grid) - 1)], [False]))
        flag_right_T_grid = np.concatenate(([False], [np.sum(
            (t[good_epochs][t[good_epochs] < t_grid[-1]][-1] > t_grid[i]) & (
                        t[good_epochs][t[good_epochs] < t_grid[-1]][-1] <= t_grid[i] + T_interval)) > 0 for i in
                                                      range(len(T_grid) - 1)]))
        flag_finite = np.isfinite(T_grid)
        flag_T_grid = (flag_right_T_grid | flag_left_T_grid) & flag_finite
        return flag_T_grid

    def lnlike(self, theta, wl, epochs, sed, sed_err, good_epoch):

        model = self.model_expression(theta, wl, epochs, good_epoch)
        inv_sigma2 = 1.0 / (sed_err ** 2.)
        return -0.5 * (np.nansum(inv_sigma2 * (sed - model) ** 2.))

    def lnprior(self, theta, epochs, good_epochs):
        log_L_peak, t_peak, sigma, t0, p, *T_grid = theta

        sigma_prior = 0 <= sigma <= 50
        t0_prior = 1 <= t0 <= 20
        p_prior = 0 <= p <= 5


        t_grid = t_peak + np.arange(-60, 301, self.T_step)
        flag_T_grid = self.gen_flag_T_grid(epochs, good_epochs, t_grid, T_grid, self.T_step)
        T_t = np.interp(epochs, t_grid[flag_T_grid], np.array(T_grid)[flag_T_grid])
        T_t = np.interp(epochs, epochs[~good_epochs], T_t[~good_epochs])
        var_T_prior = (np.abs(np.diff(10 ** T_t) / np.diff(epochs)) < 500).all()
        T_prior_4 = (4.0 <= np.array(T_grid)[flag_T_grid]).all()
        T_prior_5 = (np.array(T_grid)[flag_T_grid] <= 5).all()
        #print(T_prior_4, T_prior_5, var_T_prior)
        if sigma_prior and t0_prior and p_prior and T_prior_4 and T_prior_5 and var_T_prior:
            return 0.0
        else:
            return -np.inf

    def lnprob(self, theta, wl, epochs, sed, sed_err, good_epoch):
        lp = self.lnprior(theta, epochs, good_epoch)
        print(lp)

        if not np.isfinite(lp):
            return -np.inf
        else:
            prob = lp + self.lnlike(theta, wl, epochs, sed, sed_err, good_epoch)
            return prob