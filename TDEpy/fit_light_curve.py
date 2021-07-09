import os
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import matplotlib.pyplot as plt
import scipy.optimize as op
import emcee
from . import models as models
from . import tools as tools
import itertools
import corner
import time
from multiprocessing import Pool


def read_sw_light_curve(band, phot_dir):
    file = os.path.join(phot_dir, band + '.txt')
    _, mjd, _, _, flux_dens, flux_dens_err, _ = np.loadtxt(file, unpack=True, skiprows=2)
    flag_finite = (mjd < 0) | (flux_dens_err < 0) | (flux_dens == 0)
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


def gen_observables(tde_dir, z):
    # defining variables
    phot_dir = os.path.join(tde_dir, 'photometry', 'host_sub')
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    dist = cosmo.luminosity_distance(float(z)).to('cm')

    # filter wavelengths in the order: w2, m2, w1, U, B, g, r
    band_wls = np.array([2085.73, 2245.78, 2684.14, 3520.95, 4346.25, 4722.74, 6339.61]) / (1 + z)

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
        mjd_g, flux_dens_g, flux_dens_err_g = None, None, None
    try:
        mjd_r, flux_dens_r, flux_dens_err_r = read_ztf_light_curve('ztf_r', phot_dir)
    except:
        mjd_r, flux_dens_r, flux_dens_err_r = None, None, None

    # Working the epochs for later interpolation
    mjd_sw = np.nanmean([mjd_w2, mjd_m2, mjd_w1, mjd_U, mjd_B], axis=0)
    sw_max_mjd = (np.nanmax(mjd_sw))
    sw_min_mjd = (np.nanmin(mjd_sw))

    # Adding ZTF epochs
    mjd_ztf = np.array([])
    if mjd_r is not None:
        mjd_ztf = np.concatenate((mjd_ztf, mjd_g))
    if mjd_g is not None:
        mjd_ztf = np.concatenate((mjd_ztf, mjd_r))
    if len(mjd_ztf) > 0:
        ztf_max_mjd = (np.max(mjd_ztf))
        ztf_min_mjd = (np.min(mjd_ztf))
        mjd_ztf = [mjd_ztf_i for mjd_ztf_i in mjd_ztf]

    # creating array with ZTF + SW epochs
    epochs = np.array([])
    if len(mjd_ztf) > 0:
        for mjd in mjd_ztf:
            if mjd <= sw_min_mjd:
                epochs = np.append(epochs, mjd)

    ordering = np.argsort(mjd_sw)
    epochs = np.concatenate((epochs, mjd_sw[ordering]))
    if len(mjd_ztf) > 0:
        for mjd in mjd_ztf:
            if mjd >= sw_max_mjd:
                epochs = np.append(epochs, mjd)

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
        sed_x_t[flag, i] = lum_list[i][ordering]
        sed_err_x_t[flag, i] = lum_err_list[i][ordering]

    for i in range(len(lum_list[5:])):
        if mjd_list[5 + i] is not None:
            flag = (epochs >= ztf_min_mjd) & (epochs <= ztf_max_mjd)
            sed_x_t[flag, 5 + i] = np.interp(epochs[flag], mjd_list[5 + i], lum_list[5 + i])
            sed_err_x_t[flag, 5 + i] = np.interp(epochs[flag], mjd_list[5 + i], lum_err_list[5 + i])

    return epochs, band_wls, sed_x_t, sed_err_x_t

def read_model1(model_dir):
    path_to_model_file = os.path.join(model_dir, 'light_curve_model.txt')
    theta_median, p16, p84 = np.loadtxt(path_to_model_file, skiprows=2, max_rows=5, unpack=True, usecols=(1, 2, 3))
    return theta_median, p16, p84


def run_fit(tde_dir, z):
    # Creating directory to save model results
    modelling_dir = os.path.join(tde_dir, 'modelling')
    try:
        os.chdir(modelling_dir)
    except:
        os.mkdir(modelling_dir)
        os.chdir(modelling_dir)

    # loading observables
    t, band_wls, sed_x_t, sed_err_x_t = gen_observables(tde_dir, z)
    color = ['magenta', 'darkviolet', 'navy', 'blue', 'cyan', 'green', 'red']
    label = [r'$UV~W2$', r'$UV~M2$', r'$UV~W1$', 'U', 'B', 'g', 'r']

    # Fitting Model 1 -> Constant temperature Blackbody with Gaussian rise and exponential decay

    model_name = 'const_T_gauss_rise_exp_decay'
    observables = [t, band_wls, sed_x_t, sed_err_x_t]
    L_W2_peak_init = np.nanmax(sed_x_t[:, 0])
    t_peak_init = t[np.where(sed_x_t == np.nanmax(sed_x_t[:, 0]))[0]][0]
    log_L_peak_init, t_peak_init, sigma_init, tau_init, T0_init = np.log10(
        L_W2_peak_init), t_peak_init, 20, 30, np.log10(25000)
    ndim, nwalkers = 5, 100

    pos = [[np.random.normal(log_L_peak_init, 1),
            np.random.normal(t_peak_init, 15),
            np.random.normal(sigma_init, 10),
            np.random.normal(tau_init, 10),
            np.random.normal(T0_init, 0.1)]
           for i in range(nwalkers)]

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, models.lnprob, args=(model_name, observables), pool=pool)
        sampler.run_mcmc(pos, 100, progress=True, skip_initial_state_check=True)

    samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

    L_peak, t_peak, sigma, tau, T0 = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                                                          zip(*np.percentile(samples, [16, 50, 84], axis=0)))
    theta_median = [L_peak[0], t_peak[0], sigma[0], tau[0], T0[0]]
    _ = [print(i) for i in theta_median]
    t_model = np.arange(-50, 360, 1) + t_peak[0]
    model = models.const_T_gauss_rise_exp_decay(t_model, band_wls, theta_median)


    # Saving Model 1 results
    model_file = open(os.path.join(modelling_dir, 'light_curve_model.txt'), 'w')
    model_file.write('# Model 1: Constant Temperature Blackbody with Gaussian rise and exponential decay\n')
    model_file.write('# Parameter' + '\t' + 'median' + '\t' + 'err_p16' + '\t' + 'err_p84' + '\n')

    L_peak = tools.round_small(L_peak, 0.01)
    model_file.write('log_L_W2' + '\t' + '{:.2f}'.format(L_peak[0]) + '\t' + '{:.2f}'.format(L_peak[2]) + '\t' +
                     '{:.2f}'.format(L_peak[1]) + '\n')

    t_peak = tools.round_small(t_peak, 0.1)
    model_file.write('t_peak   ' + '\t' + '{:.1f}'.format(t_peak[0]) + '\t' + '{:.1f}'.format(t_peak[2]) + '\t' +
                     '{:.1f}'.format(t_peak[1]) + '\n')

    sigma = tools.round_small(sigma, 0.1)
    model_file.write('sigma    ' + '\t' + '{:.1f}'.format(sigma[0]) + '\t' + '{:.1f}'.format(sigma[2]) + '\t' +
                     '{:.1f}'.format(sigma[1]) + '\n')

    tau = tools.round_small(tau, 0.1)
    model_file.write('tau     ' + '\t' + '{:.1f}'.format(tau[0]) + '\t' + '{:.1f}'.format(tau[2]) + '\t' +
                     '{:.1f}'.format(tau[1]) + '\n')

    T0 = tools.round_small(T0, 0.01)
    model_file.write('log_T0   ' + '\t' + '{:.2f}'.format(T0[0]) + '\t' + '{:.2f}'.format(T0[2]) + '\t' +
                     '{:.2f}'.format(T0[1]) + '\n')
    model_file.close()


    # Fitting Model 2 -> Blackbody variable temperature with Gaussian rise and power-law decay
    model_name = 'Blackbody_var_T_gauss_rise_powerlaw_decay'

    observables = [t, band_wls, t_peak[0], sigma[0], T0[0], sed_x_t, sed_err_x_t]

    L_BB_init = L_peak[0]*models.bolometric_correction(T0[0], band_wls[0])
    log_L_BB_init, t0_init, p_init = np.log10(L_BB_init), 50, -5. / 3.
    T = np.zeros(15)
    T[:] = T0[0]

    # Posterior emcee sampling
    ndim, nwalkers = 16, 100
    pos = [np.concatenate(([np.random.normal(log_L_BB_init, 0.5),
                            np.random.normal(t0_init, 30),
                            np.random.normal(p_init, 0.1)],
                           np.append([T0[0], T[0]], [np.log10(10**T[0] + np.random.uniform(-50, 200)*dt) for dt in np.arange(0, 301, 30)]))) for i in range(nwalkers)]


    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, models.lnprob, args=(model_name, observables), pool=pool)
        sampler.run_mcmc(pos, 100, progress=True, skip_initial_state_check=True)

    samples = sampler.chain[:, 50:, :].reshape((-1, ndim))


    samples_redim = np.zeros((np.shape(samples)[0], 5))
    samples_redim[:, 0:3] = samples[:, 0:3]
    samples_redim[:, 3] = samples[:, 5]
    t_grid = np.array([t_peak[0] + np.arange(-60, 301, 30) for i in range(np.shape(samples)[0])])
    T_grid = np.array(samples[:, 3:])
    flag_T_grid = models.gen_flag_T_grid(t, t_grid[0, :], T_grid[0, :])
    samples_redim[:, 4] = np.mean(abs(np.diff(10**T_grid[:, flag_T_grid], axis=1)/np.diff(t_grid[:, flag_T_grid])), axis=1)

    L_BB_peak, t0, p, log_T_peak, dT_dt = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                                       zip(*np.percentile(samples_redim, [16, 50, 84], axis=0)))

    _, _, _, *T_t = np.percentile(samples, 50, axis=0)
    _, _, _, *T_t_p16 = map(lambda v: v[1] - v[0],
                                       zip(*np.percentile(samples, [16, 50], axis=0)))
    _, _, _, *T_t_p84 = map(lambda v: v[1] - v[0],
                                  zip(*np.percentile(samples, [50, 84], axis=0)))



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
    T_t[~flag_T_grid] = np.nan
    T_t_p16[~flag_T_grid] = np.nan
    T_t_p84[~flag_T_grid] = np.nan
    delt_t = [str(np.arange(-60, 301, 30)[i]) for i in range(13)]
    delt_t[2] = 'peak'

    for i in range(13):
        T_i = np.array([T_t[i], T_t_p16[i], T_t_p84[i]])
        T_i = tools.round_small(T_i, 0.01)
        model_file.write('T(' + str(delt_t[i]) +')' + '\t' + '{:.2f}'.format(T_i[0]) + '\t' + '{:.2f}'.format(T_i[2]) + '\t' +
                         '{:.2f}'.format(T_i[1]) + '\n')

    model_file.write(
        '<dT/dt>' + '\t' + '{:.1f}'.format(dT_dt[0]) + '\t' + '{:.1f}'.format(dT_dt[2]) + '\t' +
        '{:.1f}'.format(dT_dt[1]) + '\n')
    model_file.close()

    theta_median = np.concatenate(([L_BB_peak[0], t_peak[0], sigma[0], t0[0], p[0]], T_t))



















    '''
    fig, ax1 = plt.subplots(figsize=(8,8))
    for i in range(np.shape(model)[1]):
        y = sed_x_t[:, i]
        y_err = sed_err_x_t[:, i]
        flag = np.isfinite(y)
        x = t - t_peak[0]
        ax1.errorbar(x[flag], y[flag], yerr=y_err[flag], fmt='o', color=color[i], label=label[i])
        model_i = model[:, i]
        ax1.plot(t_model - t_peak[0], model_i, color=color[i])

    plt.show()
    
    data = np.zeros(np.shape(samples_redim))
    for i, x in enumerate(samples_redim):
        a, b, c, d, e, f, g = x
        data[i, :] = a, b, c, d, e, f, g
    bounds = []
    for i in range(np.shape(samples_redim)[1]):
        sig1 = theta_max_redim[i] - np.percentile((data[:, i]), 16)
        sig2 = np.percentile((data[:, i]), 84) - theta_max_redim[i]
        mean_dist = np.mean([sig1, sig2])
        if i == 4 and ((theta_max_redim[i] - 4 * mean_dist) < -5):
            bounds.append((-5, theta_max_redim[i] + 4 * mean_dist))
        else:
            bounds.append((theta_max_redim[i] - 4 * mean_dist, theta_max_redim[i] + 4 * mean_dist))

    figure = corner.corner(samples_redim,
                           labels=[r"$L_{peak}$", r"$t_{peak}$", r"$\sigma$", 't0', 'p', r"log $\rm{T_{peak}}$",
                                   r'$\left \langle \frac{dT}{dt} \right \rangle$'],  #
                           quantiles=[0.16, 0.5, 0.84],
                           show_titles=True, title_kwargs={"fontsize": 12}, range=bounds)

    plt.show()
    '''