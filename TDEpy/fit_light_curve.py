import os
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import scipy.optimize as op
import emcee
from multiprocessing import Pool

import TDEpy
from . import models as models
from . import tools as tools
from. import plots as plots


def read_sw_light_curve(band, phot_dir):
    file = os.path.join(phot_dir, band + '.txt')
    _, mjd, _, mag_err, flux_dens, flux_dens_err, snr = np.loadtxt(file, unpack=True, skiprows=2)
    flag_finite = (mjd < 0) | (flux_dens_err < 0) | (flux_dens == 0) | (snr < 0.2) | (mag_err > 0.5)
    mjd[flag_finite] = np.nan
    flux_dens[flag_finite] = np.nan
    flux_dens_err[flag_finite] = np.nan
    return mjd, flux_dens, flux_dens_err


def read_ztf_light_curve(band, phot_dir):
    file = os.path.join(phot_dir, band + '.txt')
    mjd, _, _, flux_dens, flux_dens_err = np.loadtxt(file, unpack=True, skiprows=2)
    order = np.argsort(mjd)
    mjd = mjd[order]
    flux_dens = flux_dens[order]
    flux_dens_err = flux_dens_err[order]
    return mjd, flux_dens, flux_dens_err


def gen_observables(tde_dir, z, bands, mode):
    # defining variables
    phot_dir = os.path.join(tde_dir, 'photometry', 'host_sub')
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    dist = cosmo.luminosity_distance(float(z)).to('cm')

    # filter wavelengths in the order: w2, m2, w1, U, B, g, r
    all_bands = ['sw_w2', 'sw_m2', 'sw_w1', 'sw_uu', 'sw_bb', 'ztf_g', 'ztf_r']
    band_wls = np.array([2085.73, 2245.78, 2684.14, 3465, 4392.25, 4722.74, 6339.61])

    # loading data:
    # Swift:
    mjd_w2, flux_dens_w2, flux_dens_err_w2 = read_sw_light_curve('sw_w2', phot_dir)
    mjd_m2, flux_dens_m2, flux_dens_err_m2 = read_sw_light_curve('sw_m2', phot_dir)
    mjd_w1, flux_dens_w1, flux_dens_err_w1 = read_sw_light_curve('sw_w1', phot_dir)
    mjd_U, flux_dens_U, flux_dens_err_U = read_sw_light_curve('sw_uu', phot_dir)
    mjd_B, flux_dens_B, flux_dens_err_B = read_sw_light_curve('sw_bb', phot_dir)
    # ZTF:
    try:
        mjd_g, flux_dens_g, flux_dens_err_g = read_ztf_light_curve('ztf_g', phot_dir)
    except:
        mjd_g, flux_dens_g, flux_dens_err_g = np.array([np.nan]), np.array([np.nan]), np.array([np.nan])
    try:
        mjd_r, flux_dens_r, flux_dens_err_r = read_ztf_light_curve('ztf_r', phot_dir)
    except:
        mjd_r, flux_dens_r, flux_dens_err_r = np.array([np.nan]), np.array([np.nan]), np.array([np.nan])

    # Working the epochs for later interpolation
    real_band_list = []
    for band in [mjd_w2, mjd_m2, mjd_w1, mjd_U, mjd_B]:
        if np.isfinite(band).any():
            real_band_list.append(band)

    mjd_sw = np.nanmean(real_band_list, axis=0)
    sw_max_mjd = (np.nanmax(mjd_sw))
    sw_min_mjd = (np.nanmin(mjd_sw))

    # Adding ZTF epochs
    mjd_ztf = np.array([])
    if mjd_r is not None:
        mjd_ztf = np.concatenate((mjd_ztf, mjd_g))
    if mjd_g is not None:
        mjd_ztf = np.concatenate((mjd_ztf, mjd_r))
    if len(mjd_ztf) > 0:
        mjd_ztf = mjd_ztf[np.argsort(mjd_ztf)]
        ztf_max_mjd = (np.max(mjd_ztf))
        ztf_min_mjd = (np.min(mjd_ztf))
        mjd_ztf = [mjd_ztf_i for mjd_ztf_i in mjd_ztf]

    # creating array with ZTF + SW epochs
    epochs = np.array([])
    if len(mjd_ztf) > 0:
        for mjd in mjd_ztf:
            if mjd <= sw_min_mjd and np.isfinite(mjd):
                epochs = np.append(epochs, mjd)

    ordering = np.argsort(mjd_sw)
    epochs = np.concatenate((epochs, mjd_sw[ordering]))
    if len(mjd_ztf) > 0:
        for mjd in mjd_ztf:
            if mjd >= sw_max_mjd and np.isfinite(mjd):
                epochs = np.append(epochs, mjd)

    epochs = epochs[np.isfinite(epochs)]
    n_bands = 7

    # Creating an array to represent the SED(t) with the same time bin as the epoch array
    sed_x_t = np.zeros((len(epochs), n_bands))
    sed_x_t[:] = np.nan
    # And another one for the uncertainties
    sed_err_x_t = np.zeros((len(epochs), n_bands))
    sed_err_x_t[:] = np.nan

    mjd_list = np.array([mjd_w2, mjd_m2, mjd_w1, mjd_U, mjd_B, mjd_g, mjd_r], dtype=object)
    lum_list = 4 * np.pi * (dist.value ** 2) * np.array(
        [flux_dens_w2 * band_wls[0], flux_dens_m2 * band_wls[1], flux_dens_w1 * band_wls[2],
         flux_dens_U * band_wls[3], flux_dens_B * band_wls[4], flux_dens_g * band_wls[5],
         flux_dens_r * band_wls[6]], dtype=object)

    lum_err_list = 4 * np.pi * (dist.value ** 2) * np.array(
        [flux_dens_err_w2 * band_wls[0], flux_dens_err_m2 * band_wls[1], flux_dens_err_w1 * band_wls[2],
         flux_dens_err_U * band_wls[3], flux_dens_err_B * band_wls[4], flux_dens_err_g * band_wls[5],
         flux_dens_err_r * band_wls[6]], dtype=object)

    for i in range(len(lum_list[:5])):
        flag = (epochs >= sw_min_mjd) & (epochs <= sw_max_mjd)
        flag_finite = np.isfinite(mjd_sw[ordering])
        sed_x_t[flag, i] = lum_list[i][ordering][flag_finite]
        sed_err_x_t[flag, i] = lum_err_list[i][ordering][flag_finite]

    for i in range(2):
        if mjd_list[5 + i] is not None:
            if i == 0:
                g_min_mjd, g_max_mjd = np.min(mjd_list[5 + i]), np.max(mjd_list[5 + i])
                flag = (epochs >= g_min_mjd) & (epochs <= g_max_mjd)
            elif i == 1:
                r_min_mjd, r_max_mjd = np.min(mjd_list[5 + i]), np.max(mjd_list[5 + i])
                flag = (epochs >= r_min_mjd) & (epochs <= r_max_mjd)

            sed_x_t[flag, 5 + i] = np.interp(epochs[flag], mjd_list[5 + i], lum_list[5 + i])
            sed_err_x_t[flag, 5 + i] = np.interp(epochs[flag], mjd_list[5 + i], lum_err_list[5 + i])

            #Eliminating B and U bands were there is ztf
            #sed_x_t[flag, 3] = np.NaN
            #sed_err_x_t[flag, 3] = np.NaN
            #sed_x_t[flag, 4] = np.NaN
            #sed_err_x_t[flag, 4] = np.NaN

    for band in all_bands:
        if band not in bands:
            i = all_bands.index(band)
            sed_x_t[:, i] = np.NaN
            sed_err_x_t[:, i] = np.NaN

    band_wls = band_wls / (1 + z)
    return epochs, band_wls, sed_x_t, sed_err_x_t


def read_model1(model_dir):
    path_to_model_file = os.path.join(model_dir, 'light_curve_model.txt')
    theta_median, p16, p84 = np.loadtxt(path_to_model_file, skiprows=2, max_rows=5, unpack=True, usecols=(1, 2, 3))
    return theta_median, p16, p84


def read_model2(model_dir):
    path_to_model_file = os.path.join(model_dir, 'light_curve_model.txt')
    theta_median, p16, p84 = np.loadtxt(path_to_model_file, skiprows=10, unpack=True, usecols=(1, 2, 3))
    return theta_median, p16, p84


def read_BB_evolution(model_dir):
    path_to_model_file = os.path.join(model_dir, 'blackbody_evolution.txt')

    try:
        t, log_BB, log_BB_err, log_R, log_R_err, log_T, log_T_err, single_band = np.loadtxt(path_to_model_file, skiprows=2, unpack=True)
    except:
        t, log_BB, log_BB_err, log_R, log_R_err, log_T, log_T_err = np.loadtxt(path_to_model_file,
                                                                                            skiprows=2, unpack=True)
    return t, log_BB, log_BB_err, log_R, log_R_err, log_T, log_T_err, single_band


def run_fit(tde, pre_peak=True, bands='All', T_interval=30, n_cores=None, n_walkers=100, n_inter=2000, n_burn=1500, show=True):
    tde_name, tde_dir, z = tde.name, tde.tde_dir, float(tde.z)

    if n_cores is None:
        n_cores = os.cpu_count() / 2
    if n_burn > n_inter - 50:
        n_burn = int((2./3.) * n_inter)

    # Creating directory to save model results
    modelling_dir = os.path.join(tde_dir, 'modelling')
    try:
        os.chdir(modelling_dir)
    except:
        os.mkdir(modelling_dir)
        os.chdir(modelling_dir)

    all_bands = ['sw_w2', 'sw_m2', 'sw_w1', 'sw_uu', 'sw_bb', 'ztf_g', 'ztf_r']
    # loading observables
    if bands == 'All':
        bands = all_bands
    else:
        for band in bands:
            if band in all_bands:
                pass
            else:
                raise Exception("your 'bands' list should contain bands between these ones: 'sw_w2', 'sw_m2', 'sw_w1', 'sw_uu', 'sw_bb', 'ztf_g', 'ztf_r'")

    t, band_wls, sed_x_t, sed_err_x_t = gen_observables(tde_dir, z, bands, mode='fit')

    # Fitting Model 1 -> Constant temperature Blackbody with Gaussian rise and exponential decay
    if pre_peak:
        model_name = 'const_T_gauss_rise_exp_decay'
        observables = [t, band_wls, sed_x_t, sed_err_x_t]

        L_W2_peak_init = np.nanmax(sed_x_t[:, 0])
        t_peak_init = t[np.where(sed_x_t == np.nanmax(sed_x_t[:, 0]))[0]][0]
        log_L_peak_init, t_peak_init, sigma_init, tau_init, T0_init = np.log10(
            L_W2_peak_init), t_peak_init, 20, 30, np.log10(25000)

        theta_init = [log_L_peak_init, t_peak_init, sigma_init, tau_init, T0_init]
        nll = lambda *args: -models.lnlike(*args)
        bounds = [(log_L_peak_init - 2, log_L_peak_init + 2), (t_peak_init - 100, t_peak_init + 100), (1, 100), (1, 200), (4, 5)]
        result = op.minimize(nll, theta_init, args=(model_name, observables), bounds=bounds, method='Powell')
        log_L_peak_opt, t_peak_opt, sigma_opt, tau_opt, T0_opt = result["x"]  # will be used to initialise the walkers

        ndim, nwalkers = 5, n_walkers
        pos = [[np.random.normal(log_L_peak_opt, 1),
                np.random.normal(t_peak_opt, 30),
                np.random.normal(sigma_opt, 10),
                np.random.normal(tau_opt, 30),
                np.random.normal(T0_opt, 0.2)]
               for i in range(nwalkers)]

        with Pool(int(n_cores)) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, models.lnprob, args=(model_name, observables), pool=pool)
            sampler.run_mcmc(pos, int(n_inter/2), progress=True, skip_initial_state_check=True)

        samples = sampler.chain[:, int(n_burn/2):, :].reshape((-1, ndim))
        L_peak, t_peak, sigma, tau, T0 = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                                             zip(*np.percentile(samples, [16, 50, 84], axis=0)))
        theta_median = [L_peak[0], t_peak[0], sigma[0], tau[0], T0[0]]

    if not pre_peak:
        model_name = 'const_T_exp_decay'
        observables = [t, band_wls, sed_x_t, sed_err_x_t]

        L_W2_peak_init = np.nanmax(sed_x_t[:, 0])
        log_L_peak_init, tau_init, T0_init = np.log10(L_W2_peak_init), 30, np.log10(25000)

        theta_init = [log_L_peak_init, tau_init, T0_init]
        nll = lambda *args: -models.lnlike(*args)
        bounds = [(log_L_peak_init - 2, log_L_peak_init + 2), (1, 200), (4, 5)]
        result = op.minimize(nll, theta_init, args=(model_name, observables), bounds=bounds, method='Powell')
        log_L_peak_opt, tau_opt, T0_opt = result["x"]  # will be used to initialise the walkers
        ndim, nwalkers = 3, n_walkers
        pos = [[np.random.normal(log_L_peak_opt, 1),
                np.random.normal(tau_opt, 30),
                np.random.normal(T0_opt, 0.2)]
               for i in range(nwalkers)]

        with Pool(int(n_cores)) as pool:
            sampler = emcee.EnsembleSampler(n_walkers, ndim, models.lnprob, args=(model_name, observables), pool=pool)
            sampler.run_mcmc(pos, int(n_inter / 2), progress=True)

        samples = sampler.chain[:, int(n_burn / 2):, :].reshape((-1, ndim))
        L_peak, tau, T0 = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                                             zip(*np.percentile(samples, [16, 50, 84], axis=0)))
        sigma, t_peak = [np.NaN, np.NaN, np.NaN], [t[0], np.NaN, np.NaN]
        theta_median = [L_peak[0], tau[0], T0[0]]


    # Saving Model 1 results
    try:
        os.mkdir(os.path.join(tde_dir, 'modelling', 'plots'))
    except:
        pass
    model_file = open(os.path.join(modelling_dir, 'light_curve_model.txt'), 'w')
    model_file.write('# Model 1: Constant Temperature Blackbody with Gaussian rise and exponential decay\n')
    model_file.write('# Parameter' + '\t' + 'median' + '\t' + 'err_p16' + '\t' + 'err_p84' + '\n')

    L_peak = tools.round_small(L_peak, 0.01)
    model_file.write('log_L_W2' + '\t' + '{:.2f}'.format(L_peak[0]) + '\t' + '{:.2f}'.format(L_peak[2]) + '\t' +
                     '{:.2f}'.format(L_peak[1]) + '\n')

    if pre_peak:
        t_peak = tools.round_small(t_peak, 0.1)
        model_file.write('t_peak   ' + '\t' + '{:.1f}'.format(t_peak[0]) + '\t' + '{:.1f}'.format(t_peak[2]) + '\t' +
                         '{:.1f}'.format(t_peak[1]) + '\n')
    if not pre_peak:
        model_file.write('t_peak   ' + '\t' + '{:.1f}'.format(t[0]) + '\t' + '{:.1f}'.format(np.NaN) + '\t' +
                         '{:.1f}'.format(np.NaN) + '\n')

    if pre_peak:
        sigma = tools.round_small(sigma, 0.1)
        model_file.write('sigma    ' + '\t' + '{:.1f}'.format(sigma[0]) + '\t' + '{:.1f}'.format(sigma[2]) + '\t' +
                        '{:.1f}'.format(sigma[1]) + '\n')
    if not pre_peak:
        model_file.write('sigma    ' + '\t' + '{:.1f}'.format(np.NaN) + '\t' + '{:.1f}'.format(np.NaN) + '\t' +
                         '{:.1f}'.format(np.NaN) + '\n')

    tau = tools.round_small(tau, 0.1)
    model_file.write('tau     ' + '\t' + '{:.1f}'.format(tau[0]) + '\t' + '{:.1f}'.format(tau[2]) + '\t' +
                     '{:.1f}'.format(tau[1]) + '\n')

    T0 = tools.round_small(T0, 0.01)
    model_file.write('log_T0   ' + '\t' + '{:.2f}'.format(T0[0]) + '\t' + '{:.2f}'.format(T0[2]) + '\t' +
                     '{:.2f}'.format(T0[1]) + '\n')
    model_file.close()


    plot_dir = os.path.join(modelling_dir, 'plots')
    try:
        os.mkdir(plot_dir)
    except:
        pass
    fig_name = 'corner_plot_model1.pdf'
    if pre_peak:
        labels = [r"$L_{W2\,peak}$", r'$t_{peak}$', r'$\sigma$', r'$\tau$', r'log T$_0$']
        plots.plot_lc_corner(tde_dir, fig_name, theta_median, samples, labels, show=show)
    if not pre_peak:
        labels = [r"$L_{W2\,peak}$", r'$\tau$', r'log T$_0$']
        plots.plot_lc_corner(tde_dir, fig_name, theta_median, samples, labels, show=show)
        theta_median = [L_peak[0], t[0], np.NaN, tau[0], T0[0]]

    # Fitting Model 2 -> Blackbody variable temperature with Gaussian rise and power-law decay

    if pre_peak:
        model_name = 'Blackbody_var_T_gauss_rise_powerlaw_decay'

        n_T = len(np.arange(-60, 301, T_interval))
        observables = [t, band_wls, T_interval, theta_median, sed_x_t, sed_err_x_t]

        L_BB_init = (10 ** L_peak[0]) * models.bolometric_correction(T0[0], band_wls[0])
        log_L_BB_init, t_peak_init, sigma_init, t0_init, p_init = np.log10(L_BB_init), theta_median[1], theta_median[2], 50, 5. / 3.
        theta_init = np.concatenate(([log_L_BB_init, t_peak_init, sigma_init, t0_init, p_init], [T0[0] for j in range(n_T)]))

        nll = lambda *args: -models.lnlike(*args)
        bounds = np.concatenate(([(log_L_BB_init - 0.5, log_L_BB_init + 0.5), (theta_init[1] - 15, theta_init[1] + 15), (sigma_init - 5, sigma_init + 5), (1, 1000), (0, 3)],
        [(T0[0] - 0.2, T0[0] + 0.2) for i in range(n_T)]))
        result = op.minimize(nll, theta_init, args=(model_name, observables), bounds=bounds, method='Powell')  # Some rough initial guesses
        log_L_BB_opt, t_peak_opt, sigma_opt, t0_opt, p_opt, *Ts_opt = result["x"]  # will be used to initialise the walkers

        # Posterior emcee sampling
        ndim, nwalkers = int(5+n_T), n_walkers
        pos = [np.concatenate(([np.random.normal(log_L_BB_opt, 0.2),
                                np.random.normal(t_peak_opt, 2),
                                np.random.normal(sigma_opt, 2),
                                np.random.normal(t0_opt, 5),
                                np.random.normal(p_opt, 0.2)],
                               [Ts_opt[j] + np.random.normal(0, 0.2) for j in range(n_T)])) for i in range(n_walkers)]

        with Pool(int(n_cores)) as pool:
            sampler = emcee.EnsembleSampler(n_walkers, ndim, models.lnprob, args=(model_name, observables), pool=pool)
            sampler.run_mcmc(pos, n_inter, progress=True, skip_initial_state_check=True)

        samples = sampler.chain[:, n_burn:, :].reshape((-1, ndim))

        samples_redim = np.zeros((np.shape(samples)[0], 6))
        samples_redim[:, 0:5] = samples[:, 0:5]
        samples_redim[:, 5] = samples[:, int(60/T_interval) + 5]


        L_BB_peak, t_peak, sigma, t0, p, log_T_peak = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                                                                  zip(*np.percentile(samples_redim, [16, 50, 84], axis=0)))

        _, _, _, _, _, *T_t = np.nanpercentile(samples, 50, axis=0)
        _, _, _, _, _, *T_t_p16 = map(lambda v: v[1] - v[0],
                                      zip(*np.nanpercentile(samples, [16, 50], axis=0)))
        _, _, _, _, _, *T_t_p84 = map(lambda v: v[1] - v[0],
                                      zip(*np.nanpercentile(samples, [50, 84], axis=0)))

        model_file = open(os.path.join(modelling_dir, 'light_curve_model.txt'), 'a')
        model_file.write('\n')
        model_file.write('# Model 2: Evolving Blackbody with Gaussian rise and power-law decay\n')
        model_file.write('# Parameter' + '\t' + 'median' + '\t' + 'err_p16' + '\t' + 'err_p84' + '\n')

        L_BB_peak = tools.round_small(L_BB_peak, 0.01)
        model_file.write('log_L_BB' + '\t' + '{:.2f}'.format(L_BB_peak[0]) + '\t' + '{:.2f}'.format(L_BB_peak[2]) + '\t' +
                         '{:.2f}'.format(L_BB_peak[1]) + '\n')

        t_peak = tools.round_small(t_peak, 0.1)
        model_file.write('t_peak  ' + '\t' + '{:.1f}'.format(t_peak[0]) + '\t' + '{:.1f}'.format(t_peak[2]) + '\t' +
                         '{:.1f}'.format(t_peak[1]) + '\n')

        sigma = tools.round_small(sigma, 0.1)
        model_file.write('sigma   ' + '\t' + '{:.1f}'.format(sigma[0]) + '\t' + '{:.1f}'.format(sigma[2]) + '\t' +
                         '{:.1f}'.format(sigma[1]) + '\n')

        t0 = tools.round_small(t0, 0.1)
        model_file.write('t0      ' + '\t' + '{:.1f}'.format(t0[0]) + '\t' + '{:.1f}'.format(t0[2]) + '\t' +
                         '{:.1f}'.format(t0[1]) + '\n')

        p = tools.round_small(p, 0.1)
        model_file.write('p       ' + '\t' + '{:.1f}'.format(p[0]) + '\t' + '{:.1f}'.format(p[2]) + '\t' +
                         '{:.1f}'.format(p[1]) + '\n')

        T_t, T_t_p16, T_t_p84 = np.array(T_t), np.array(T_t_p16), np.array(T_t_p84)

        T_t_p16 = tools.round_small(T_t_p16, 0.01)
        T_t_p84 = tools.round_small(T_t_p84, 0.01)
        t_grid = t_peak[0] + np.arange(-60, 301, T_interval)
        single_color = np.array([np.sum(np.isfinite(sed_x_t[i, :])) == 1 for i in range(len(t))])
        flag_T_grid = models.gen_flag_T_grid(t, single_color, t_grid, T_t, T_interval)
        T_t[~flag_T_grid] = np.nan
        T_t_p16[~flag_T_grid] = np.nan
        T_t_p84[~flag_T_grid] = np.nan
        delt_t = [str(np.arange(-60, 301, T_interval)[i]) for i in range(n_T)]
        delt_t[int(60/T_interval)] = 'peak'

        for i in range(n_T):
            T_i = np.array([T_t[i], T_t_p16[i], T_t_p84[i]])
            T_i = tools.round_small(T_i, 0.01)
            model_file.write(
                'T(' + str(delt_t[i]) + ')' + '\t' + '{:.2f}'.format(T_i[0]) + '\t' + '{:.2f}'.format(T_i[2]) + '\t' +
                '{:.2f}'.format(T_i[1]) + '\n')
        model_file.close()

        theta_median = np.concatenate(([L_BB_peak[0], t_peak[0], sigma[0], t0[0], p[0]], T_t))
        theta_err_p16 = np.concatenate(([L_BB_peak[2], t_peak[2], sigma[2], t0[2], p[2]], T_t_p16))
        theta_err_p84 = np.concatenate(([L_BB_peak[1], t_peak[1], sigma[1], t0[1], p[1]], T_t_p84))
        theta_err = np.min([theta_err_p16, theta_err_p84], axis=0)

        log_T, log_T_err, log_BB, log_BB_err, log_R, log_R_err = models.Blackbody_evolution(t, single_color, T_interval, theta_median, theta_err)

        model_file = open(os.path.join(modelling_dir, 'blackbody_evolution.txt'), 'w')
        model_file.write('# Blackbody Evolution:\n')
        model_file.write('MJD' +
                         '\t' + 'log_L_BB' + '\t' + 'log_L_BB_err' +
                         '\t' + 'log_R' + '\t' + 'log_R_err' +
                         '\t' + 'log_T' + '\t' + 'log_T_err' + '\t' + 'single_band_flag' +  '\n')
        flag_300_days = (t - t_peak[0]) < 300
        for yy in range(len(t[flag_300_days])):
            model_file.write('{:.2f}'.format(t[yy]) +
                             '\t' + '{:.2f}'.format(log_BB[yy]) + '\t' + '{:.2f}'.format(log_BB_err[yy]) +
                             '\t' + '{:.2f}'.format(log_R[yy]) + '\t' + '{:.2f}'.format(log_R_err[yy]) +
                             '\t' + '{:.2f}'.format(log_T[yy]) + '\t' + '{:.2f}'.format(log_T_err[yy]) + '\t' +
                             str(int(single_color[yy])) + '\n')
        model_file.close()

        theta_median_redim = [L_BB_peak[0], t_peak[0], sigma[0], t0[0], p[0], log_T_peak[0]]
        labels = [r"log $L_{BB\,peak}$", r'$t_{peak}$', r'$\sigma$', r'$t_0$', r'$p$', r"log $T_{peak}$"]

        fig_name = 'corner_plot_model2'
        plots.plot_lc_corner(tde_dir, fig_name, theta_median_redim, samples_redim, labels, show=show)
        plots.plot_models(tde_name, tde_dir, z, bands, T_interval, show=show)
        plots.plot_BB_evolution(tde_name, tde_dir, show=show)
        plots.plot_SED(tde_name, tde_dir, z, bands, sampler, nwalkers, n_burn, n_inter, show=show)

    if not pre_peak:
        model_name = 'Blackbody_var_T_powerlaw_decay'
        theta_median = [L_peak[0], t[0], tau[0], T0[0]]
        n_T = len(np.arange(-60, 301, T_interval))
        observables = [t, band_wls, T_interval, theta_median, sed_x_t, sed_err_x_t]

        L_BB_init = (10 ** L_peak[0]) * models.bolometric_correction(T0[0], band_wls[0])
        log_L_BB_init, t0_init, p_init = np.log10(L_BB_init), theta_median[2], 5. / 3.
        theta_init = np.concatenate(([log_L_BB_init, t0_init, p_init], [T0[0] for j in range(n_T)]))

        nll = lambda *args: -models.lnlike(*args)
        bounds = np.concatenate(([(log_L_BB_init - 0.5, log_L_BB_init + 0.5), (1, 1000), (0, 3)], [(T0[0] - 0.2, T0[0] + 0.2) for i in range(n_T)]))

        result = op.minimize(nll, theta_init, args=(model_name, observables), bounds=bounds,
                             method='Powell')  # Some rough initial guesses
        log_L_BB_opt, t0_opt, p_opt, *Ts_opt = result["x"]  # will be used to initialise the walkers

        # Posterior emcee sampling
        ndim, nwalkers = int(3 + n_T), n_walkers
        pos = [np.concatenate(([np.random.normal(log_L_BB_opt, 0.2),
                                np.random.normal(t0_opt, 5),
                                np.random.normal(p_opt, 0.2)],
                               [Ts_opt[j] + np.random.normal(0, 0.2) for j in range(n_T)])) for i in range(nwalkers)]

        with Pool(int(n_cores)) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, models.lnprob, args=(model_name, observables), pool=pool)
            sampler.run_mcmc(pos, n_inter, progress=True, skip_initial_state_check=True)

        samples = sampler.chain[:, n_burn:, :].reshape((-1, ndim))

        samples_redim = np.zeros((np.shape(samples)[0], 4))
        samples_redim[:, 0:3] = samples[:, 0:3]
        samples_redim[:, 3] = samples[:, int(60 / T_interval) + 3]

        L_BB_peak, t0, p, log_T_peak = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                                                          zip(*np.percentile(samples_redim, [16, 50, 84], axis=0)))

        _, _, _, *T_t = np.nanpercentile(samples, 50, axis=0)
        _, _, _, *T_t_p16 = map(lambda v: v[1] - v[0],
                                      zip(*np.nanpercentile(samples, [16, 50], axis=0)))
        _, _, _, *T_t_p84 = map(lambda v: v[1] - v[0],
                                      zip(*np.nanpercentile(samples, [50, 84], axis=0)))

        model_file = open(os.path.join(modelling_dir, 'light_curve_model.txt'), 'a')
        model_file.write('\n')
        model_file.write('# Model 2: Evolving Blackbody with Gaussian rise and power-law decay\n')
        model_file.write('# Parameter' + '\t' + 'median' + '\t' + 'err_p16' + '\t' + 'err_p84' + '\n')

        L_BB_peak = tools.round_small(L_BB_peak, 0.01)
        model_file.write(
            'log_L_BB' + '\t' + '{:.2f}'.format(L_BB_peak[0]) + '\t' + '{:.2f}'.format(L_BB_peak[2]) + '\t' +
            '{:.2f}'.format(L_BB_peak[1]) + '\n')

        model_file.write('t_peak  ' + '\t' + '{:.1f}'.format(t[0]) + '\t' + '{:.1f}'.format(np.nan) + '\t' +
                         '{:.1f}'.format(np.nan) + '\n')

        model_file.write('sigma   ' + '\t' + '{:.1f}'.format(np.nan) + '\t' + '{:.1f}'.format(np.nan) + '\t' +
                         '{:.1f}'.format(np.nan) + '\n')

        t0 = tools.round_small(t0, 0.1)
        model_file.write('t0      ' + '\t' + '{:.1f}'.format(t0[0]) + '\t' + '{:.1f}'.format(t0[2]) + '\t' +
                         '{:.1f}'.format(t0[1]) + '\n')

        p = tools.round_small(p, 0.1)
        model_file.write('p       ' + '\t' + '{:.1f}'.format(p[0]) + '\t' + '{:.1f}'.format(p[2]) + '\t' +
                         '{:.1f}'.format(p[1]) + '\n')

        T_t, T_t_p16, T_t_p84 = np.array(T_t), np.array(T_t_p16), np.array(T_t_p84)

        T_t_p16 = tools.round_small(T_t_p16, 0.01)
        T_t_p84 = tools.round_small(T_t_p84, 0.01)
        t_grid = t[0] + np.arange(-60, 301, T_interval)
        single_color = np.array([np.sum(np.isfinite(sed_x_t[i, :])) == 1 for i in range(len(t))])
        flag_T_grid = models.gen_flag_T_grid(t, single_color, t_grid, T_t, T_interval)
        T_t[~flag_T_grid] = np.nan
        T_t_p16[~flag_T_grid] = np.nan
        T_t_p84[~flag_T_grid] = np.nan
        delt_t = [str(np.arange(-60, 301, T_interval)[i]) for i in range(n_T)]
        delt_t[int(60 / T_interval)] = 'peak'

        for i in range(n_T):
            T_i = np.array([T_t[i], T_t_p16[i], T_t_p84[i]])
            T_i = tools.round_small(T_i, 0.01)
            model_file.write(
                'T(' + str(delt_t[i]) + ')' + '\t' + '{:.2f}'.format(T_i[0]) + '\t' + '{:.2f}'.format(T_i[2]) + '\t' +
                '{:.2f}'.format(T_i[1]) + '\n')
        model_file.close()

        theta_median = np.concatenate(([L_BB_peak[0], t[0], np.nan, t0[0], p[0]], T_t))
        theta_err_p16 = np.concatenate(([L_BB_peak[2], np.nan, np.nan, t0[2], p[2]], T_t_p16))
        theta_err_p84 = np.concatenate(([L_BB_peak[1], np.nan, np.nan, t0[1], p[1]], T_t_p84))
        theta_err = np.min([theta_err_p16, theta_err_p84], axis=0)

        log_T, log_T_err, log_BB, log_BB_err, log_R, log_R_err = models.Blackbody_evolution(t, single_color, T_interval,
                                                                                            theta_median, theta_err)

        model_file = open(os.path.join(modelling_dir, 'blackbody_evolution.txt'), 'w')
        model_file.write('# Blackbody Evolution:\n')
        model_file.write('MJD' +
                         '\t' + 'log_L_BB' + '\t' + 'log_L_BB_err' +
                         '\t' + 'log_R' + '\t' + 'log_R_err' +
                         '\t' + 'log_T' + '\t' + 'log_T_err' + '\t' + 'single_band_flag' + '\n')
        flag_300_days = (t - t[0]) < 300
        for yy in range(len(t[flag_300_days])):
            model_file.write('{:.2f}'.format(t[yy]) +
                             '\t' + '{:.2f}'.format(log_BB[yy]) + '\t' + '{:.2f}'.format(log_BB_err[yy]) +
                             '\t' + '{:.2f}'.format(log_R[yy]) + '\t' + '{:.2f}'.format(log_R_err[yy]) +
                             '\t' + '{:.2f}'.format(log_T[yy]) + '\t' + '{:.2f}'.format(log_T_err[yy]) + '\t' +
                             str(int(single_color[yy])) + '\n')
        model_file.close()

        theta_median_redim = [L_BB_peak[0], t0[0], p[0], log_T_peak[0]]
        labels = [r"log $L_{BB\,peak}$", r'$t_0$', r'$p$', r"log $T_{peak}$"]

        fig_name = 'corner_plot_model2.pdf'
        plots.plot_lc_corner(tde_dir, fig_name, theta_median_redim, samples_redim, labels, show=show)
        plots.plot_models(tde_name, tde_dir, z, bands, T_interval, show=show)
        plots.plot_BB_evolution(tde_name, tde_dir, show=show)
        plots.plot_SED(tde_name, tde_dir, z, bands, sampler, n_walkers, n_burn, n_inter, show=show)