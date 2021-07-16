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
from astropy.constants import h, c, k_B, sigma_sb


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

    band_wls = band_wls / (1 + z)
    return epochs, band_wls, sed_x_t, sed_err_x_t


def read_model1(model_dir):
    path_to_model_file = os.path.join(model_dir, 'light_curve_model.txt')
    theta_median, p16, p84 = np.loadtxt(path_to_model_file, skiprows=2, max_rows=5, unpack=True, usecols=(1, 2, 3))
    return theta_median, p16, p84


def read_model2(model_dir):
    path_to_model_file = os.path.join(model_dir, 'light_curve_model.txt')
    theta_median, p16, p84 = np.loadtxt(path_to_model_file, skiprows=10, max_rows=18, unpack=True, usecols=(1, 2, 3))
    return theta_median, p16, p84


def read_BB_evolution(model_dir):
    path_to_model_file = os.path.join(model_dir, 'light_curve_model.txt')
    t, log_BB, log_BB_err, log_R, log_R_err, log_T, log_T_err = np.loadtxt(path_to_model_file, skiprows=32, unpack=True)
    return t, log_BB, log_BB_err, log_R, log_R_err, log_T, log_T_err


def plot_models(tde_name, tde_dir, z, print_name=True, show=True):
    t, band_wls, sed_x_t, sed_err_x_t = gen_observables(tde_dir, z)
    modelling_dir = os.path.join(tde_dir, 'modelling')
    color = ['magenta', 'darkviolet', 'navy', 'blue', 'cyan', 'green', 'red']
    label = [r'$UV~W2$', r'$UV~M2$', r'$UV~W1$', 'U', 'B', 'g', 'r']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    theta_median, p16, p84 = read_model1(modelling_dir)
    t_model = theta_median[1] + np.arange(-50, 301, 1)
    model = models.const_T_gauss_rise_exp_decay(t_model, band_wls, theta_median)
    for i in range(np.shape(model)[1]):
        y = sed_x_t[:, i]
        y_err = sed_err_x_t[:, i]
        flag = np.isfinite(y)
        x = t - theta_median[1]

        ax1.errorbar(x[flag], y[flag], yerr=y_err[flag], marker="o", linestyle='', color=color[i], linewidth=1,
                     markeredgewidth=0.5, markeredgecolor='black', alpha=0.9, markersize=6, elinewidth=0.7, capsize=0,
                     label=label[i])
        model_i = model[:, i]
        ax1.plot(t_model - theta_median[1], model_i, color=color[i])

    theta_median, p16, p84 = read_model2(modelling_dir)
    model = models.Blackbody_var_T_gauss_rise_powerlaw_decay(t_model, band_wls, theta_median)
    for i in range(np.shape(model)[1]):
        t_peak = theta_median[1]
        y = sed_x_t[:, i]
        y_err = sed_err_x_t[:, i]
        flag = np.isfinite(y)
        x = t - t_peak
        ax2.errorbar(x[flag], y[flag], yerr=y_err[flag], marker="o", linestyle='', color=color[i], linewidth=1,
                     markeredgewidth=0.5, markeredgecolor='black', alpha=0.9, markersize=6, elinewidth=0.7, capsize=0,
                     label=label[i])
        model_i = model[:, i]
        ax2.plot(t_model - t_peak, model_i, color=color[i])

    ax1.set_yscale('log')
    ax1.set_xlim(-60, 301)
    ax2.set_yscale('log')
    ax2.set_xlim(-60, 301)

    ax2.set_xlabel('Days since peak')
    ax1.set_ylabel(r'$\rm{\nu\,L_{\nu} \ [erg \ s^{-1}]}$')
    ax2.set_ylabel(r'$\rm{\nu\,L_{\nu} \ [erg \ s^{-1}]}$')
    plt.tight_layout()
    if print_name:
        ax1.text((0.2, 0.05), tde_name, horizontalalignment='left', verticalalignment='center', fontsize=12)
    plt.savefig(os.path.join(tde_dir, 'plots', 'modelling', 'model_light_curves.png'), bbox_inches='tight')
    if show:
        plt.show()


def plot_BB_evolution(tde_name, tde_dir, print_name=True, show=True):
    modelling_dir = os.path.join(tde_dir, 'modelling')
    t, log_BB, log_BB_err, log_R, log_R_err, log_T, log_T_err = read_BB_evolution(modelling_dir)
    theta_median, p16, p84 = read_model2(modelling_dir)
    t_peak = theta_median[1]
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))
    ax1.errorbar(t - t_peak, log_BB, yerr=log_BB_err, marker="o", linestyle='-', color='b', linewidth=1,
                 markeredgewidth=1, markerfacecolor='blue', markeredgecolor='black', markersize=4, elinewidth=0.7,
                 capsize=2)
    ax1.set_ylabel(r'log $\rm{L_{BB} \ [erg \ s^{-1}]}$')
    ax2.errorbar(t - t_peak, log_R, yerr=log_R_err, marker="o", linestyle='-', color='b', linewidth=1,
                 markeredgewidth=1, markerfacecolor='blue', markeredgecolor='black', markersize=4, elinewidth=0.7,
                 capsize=2)
    ax2.set_ylabel('log R [cm]')
    ax3.errorbar(t - t_peak, log_T, yerr=log_T_err, marker="o", linestyle='-', color='b', linewidth=1,
                 markeredgewidth=1, markerfacecolor='blue', markeredgecolor='black', markersize=4, elinewidth=0.7,
                 capsize=2)
    ax3.set_ylabel('log T [K]')
    ax3.set_xlabel('Days since peak')
    plt.tight_layout()
    if print_name:
        ax1.text((0.1, 0.05), tde_name, horizontalalignment='left', verticalalignment='center', fontsize=12)
    plt.savefig(os.path.join(tde_dir, 'plots', 'modelling', 'Blackbody_evolution.png'), bbox_inches='tight')
    if show:
        plt.show()


def plot_SED(tde_name, tde_dir, z, sampler, nwalkers, nburn, ninter, print_name=True, show=True):
    modelling_dir = os.path.join(tde_dir, 'modelling')
    t, band_wls, sed_x_t, sed_err_x_t = gen_observables(tde_dir, z)
    t_BB, log_BB, log_BB_err, log_R, log_R_err, log_T, log_T_err = read_BB_evolution(modelling_dir)
    theta_median, p16, p84 = read_model2(modelling_dir)
    t_model = theta_median[1] + np.arange(-40, 301, 1)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
    label = [r'$UV~W2$', r'$UV~M2$', r'$UV~W1$', 'U', 'B', 'g', 'r']
    marker = ["o", "s", "p", "d", "*", "x", "+"]
    t_peak = theta_median[1]
    first_300days = (t - t_peak) <= 300
    for i in range(np.shape(sed_x_t)[1]):

        y = sed_x_t[first_300days, i] * models.bolometric_correction(log_T, band_wls[i])
        y_err = sed_err_x_t[first_300days, i] / sed_x_t[first_300days, i] * y
        flag = np.isfinite(y)
        x = t[first_300days] - t_peak
        ax1.errorbar(x[flag], y[flag], yerr=y_err[flag], marker=marker[i], ecolor='black', linestyle='', mfc='None',
                     mec='black', linewidth=1,
                     markeredgewidth=0.5, markersize=7, elinewidth=0.7, capsize=0)

    string = r'$\sigma={:.1f}  \ p={:.1f}  \ t_0={:.1f}$'.format(theta_median[2], theta_median[4], theta_median[3])
    L_BB = models.L_bol(t_model, theta_median)
    ax1.plot(t_model - t_peak, L_BB, c='blue', alpha=1, label=string)
    randint = np.random.randint
    for i in range(100):
        theta = sampler.chain[randint(nwalkers), nburn + randint(ninter - nburn), :]
        L_BB = models.L_bol(t_model, theta)
        ax1.plot(t_model - t_peak, L_BB, c='blue', alpha=0.05)
    ax1.legend(fontsize='x-small', loc=1)
    ax1.set_yscale('log')
    ax1.set_xlim(-60, 311)
    ax1.set_ylabel('Bolometric Luminosity [erg s$^{-1}$]', fontsize=12)
    ax1.set_xlabel('Days since peak', fontsize=12)
    ax1.set_xticks(np.arange(-50, 301, 50))
    ax1.set_xticklabels(np.arange(-50, 301, 50), fontsize=12)
    ax1.tick_params(axis='y', labelsize=12)



    delt_t = t - t_peak
    near_peak = np.where(abs(delt_t) == np.nanmin(abs(delt_t)))
    t_near_peak = t[near_peak]
    flag_peak = abs(t - t_near_peak) <= 2
    flag_peak_BB = abs(t_BB - t_near_peak) <= 2
    T_near_peak = np.mean(log_T[flag_peak_BB])
    L_BB_near_peak = 10**theta_median[0]
    for i in range(np.shape(sed_x_t)[1]):
        y = sed_x_t[flag_peak, i]
        y_err = sed_err_x_t[flag_peak, i]
        flag = np.isfinite(y)
        wl = band_wls[i] * u.Angstrom
        nu = np.zeros(np.shape(y[flag]))
        nu[:] = c.cgs / wl.cgs

        ax2.errorbar(nu, y[flag], yerr=y_err[flag], marker=marker[i], ecolor='black', linestyle='', mfc='None',
                     mec='black', linewidth=1,
                     markeredgewidth=0.5, markersize=7, elinewidth=0.7, capsize=0)

    nu_list = (c.cgs / (np.arange(1300, 10000, 10) * u.Angstrom)).cgs
    A = L_BB_near_peak / ((sigma_sb.cgs * ((10 ** T_near_peak * u.K) ** 4)).cgs / np.pi).cgs.value
    bb_sed = (A * models.blackbody(10**T_near_peak, (c.cgs/nu_list).to('AA').value))
    ax2.plot(nu_list.value, bb_sed, ls='--', c='blue')
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_ylabel(r'$\rm{\nu\,L_{\nu} \ [erg \ s^{-1}]}$', fontsize=14)
    ax2.set_xlabel('Rest-frame frequency (Hz)', fontsize=12)
    ax2.set_xticks([4e14, 6e14, 1e15, 2e15])
    ax2.set_xticklabels([r'$4\times10^{14}$', r'$6\times10^{14}$', r'$1\times10^{15}$', r'$2\times10^{15}$'], fontsize=11)
    ax2.tick_params(axis='y', labelsize=12)
    ax2.set_xlim(nu_list[-1].value, 2.1e15)
    up_lim, lo_lim = np.max(bb_sed.value), np.min(bb_sed.value)
    ax2.set_ylim(10**(np.log10(lo_lim) - 0.3), 10**(np.log10(up_lim) + 0.3))
    title = r'$t={:.0f} \pm 2$ days pos max; $T={:.0f} \  K$'.format(int(t_near_peak-t_peak), 10**T_near_peak)
    ax2.set_title(title, fontsize=12)
    ax2.legend(fontsize='x-small', loc=4)


    near_200 = np.where(abs(delt_t-200) == np.nanmin(abs(delt_t-200)))
    t_near_200 = t[near_200]
    flag_200 = abs(t - t_near_200) <= 2
    flag_200_BB = abs(t_BB - t_near_200) <= 2
    T_near_200 = np.mean(log_T[flag_200_BB])
    L_BB_near_200 = 10**np.mean(log_BB[flag_200_BB])
    for i in range(np.shape(sed_x_t)[1]):
        y = sed_x_t[flag_200, i]
        y_err = sed_err_x_t[flag_200, i]
        flag = np.isfinite(y)
        wl = band_wls[i] * u.Angstrom
        nu = np.zeros(np.shape(y[flag]))
        nu[:] = c.cgs / wl.cgs

        ax3.errorbar(nu, y[flag], yerr=y_err[flag], marker=marker[i], ecolor='black', linestyle='', mfc='None',
                     mec='black', linewidth=1,
                     markeredgewidth=0.5, markersize=7, elinewidth=0.7, capsize=0)

    nu_list = (c.cgs / (np.arange(1300, 10000, 10) * u.Angstrom)).cgs
    A = L_BB_near_200 / ((sigma_sb.cgs * ((10 ** T_near_200 * u.K) ** 4)).cgs / np.pi).cgs.value
    bb_sed = (A * models.blackbody(10**T_near_200, (c.cgs/nu_list).to('AA').value))
    ax3.plot(nu_list.value, bb_sed, ls='--', c='blue')
    ax3.set_yscale('log')
    ax3.set_xscale('log')
    ax3.set_ylabel(r'$\rm{\nu\,L_{\nu} \ [erg \ s^{-1}]}$', fontsize=14)
    ax3.set_xlabel('Rest-frame frequency (Hz)', fontsize=12)
    ax3.set_xticks([4e14, 6e14, 1e15, 2e15])
    ax3.set_xticklabels([r'$4\times10^{14}$', r'$6\times10^{14}$', r'$1\times10^{15}$', r'$2\times10^{15}$'], fontsize=11)
    ax3.tick_params(axis='y', labelsize=12)
    ax3.set_xlim(nu_list[-1].value, 2.1e15)
    up_lim, lo_lim = np.max(bb_sed.value), np.min(bb_sed.value)
    ax3.set_ylim(10**(np.log10(lo_lim) - 0.3), 10**(np.log10(up_lim) + 0.3))
    title = r'$t={:.0f} \pm 2$ days pos max; $T={:.0f} \ K$'.format(int(t_near_200 - t_peak), 10**T_near_200)
    ax3.set_title(title, fontsize=12)
    up_lim = np.max([ax2.get_ylim(), ax3.get_ylim()])
    lo_lim = np.min([ax2.get_ylim(), ax3.get_ylim()])
    ax2.set_ylim(lo_lim, up_lim)
    ax3.set_ylim(lo_lim, up_lim)
    plt.tight_layout()
    if print_name:
        ax1.text((0.2, 0.05), tde_name, horizontalalignment='left', verticalalignment='center', fontsize=12)
    plt.savefig(os.path.join(tde_dir, 'plots', 'modelling', 'SED_evolution.png'), bbox_inches='tight')
    if show:
        plt.show()


def plot_corner(plot_dir, fig_name, theta_median, sample, labels, show=True):
    data = np.zeros(np.shape(sample))
    for i, x in enumerate(sample):
        data[i, :] = x
    bounds = []
    for i in range(np.shape(sample)[1]):
        sig1 = np.percentile((data[:, i]), 50) - np.percentile((data[:, i]), 16)
        sig2 = np.percentile((data[:, i]), 84) - np.percentile((data[:, i]), 50)
        mean_dist = np.mean([sig1, sig2])
        bounds.append((theta_median[i] - 4 * mean_dist, theta_median[i] + 4 * mean_dist))

    figure = corner.corner(sample,
                           labels=labels,  #
                           quantiles=[0.16, 0.5, 0.84],
                           show_titles=True, title_kwargs={"fontsize": 12}, range=bounds)
    plt.savefig(os.path.join(plot_dir, fig_name), bbox_inches='tight')
    if show:
        plt.show()


def run_fit(tde_name, tde_dir, z, n_cores, nwalkers=100, ninter=1000, nburn=500):
    # Creating directory to save model results
    modelling_dir = os.path.join(tde_dir, 'modelling')
    try:
        os.chdir(modelling_dir)
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
    ndim, nwalkers = 5, nwalkers

    pos = [[np.random.normal(log_L_peak_init, 1),
            np.random.normal(t_peak_init, 15),
            np.random.normal(sigma_init, 10),
            np.random.normal(tau_init, 10),
            np.random.normal(T0_init, 0.1)]
           for i in range(nwalkers)]

    with Pool(int(n_cores)) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, models.lnprob, args=(model_name, observables), pool=pool)
        sampler.run_mcmc(pos, ninter, progress=True, skip_initial_state_check=True)

    samples = sampler.chain[:, nburn:, :].reshape((-1, ndim))

    L_peak, t_peak, sigma, tau, T0 = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                                         zip(*np.percentile(samples, [16, 50, 84], axis=0)))
    theta_median = [L_peak[0], t_peak[0], sigma[0], tau[0], T0[0]]
    t_model = np.arange(-50, 360, 1) + t_peak[0]

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
    labels = [r"$L_{W2\,peak}$", r'$t_{peak}$', r'$\sigma$', r'$\tau$', r'T$_0$']

    plot_dir = os.path.join(tde_dir, 'plots', 'modelling')
    try:
        os.mkdir(plot_dir)
    except:
        pass
    fig_name = 'corner_plot_model1'
    plot_corner(plot_dir, fig_name, theta_median, samples, labels, show=True)

    # Fitting Model 2 -> Blackbody variable temperature with Gaussian rise and power-law decay
    model_name = 'Blackbody_var_T_gauss_rise_powerlaw_decay'

    observables = [t, band_wls, theta_median, sed_x_t, sed_err_x_t]

    L_BB_init = (10 ** L_peak[0]) * models.bolometric_correction(T0[0], band_wls[0])
    log_L_BB_init, t_peak_init, sigma_init, t0_init, p_init = np.log10(L_BB_init), t_peak[0], sigma[0], 10, 5. / 3.

    # Posterior emcee sampling
    ndim, nwalkers = 18, nwalkers
    pos = [np.concatenate(([np.random.normal(log_L_BB_init, 0.5),
                            np.random.normal(t_peak_init, 1),
                            np.random.normal(sigma_init, 1),
                            np.random.normal(t0_init, 5),
                            np.random.normal(p_init, 0.2)],
                           np.append([T0[0], T0[0]], [np.log10(10 ** T0[0] + np.random.uniform(-50, 400) * dt) + np.random.normal(0, 0.05) for dt in
                                                      np.arange(0, 301, 30)]))) for i in range(nwalkers)]

    with Pool(int(n_cores)) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, models.lnprob, args=(model_name, observables), pool=pool)
        sampler.run_mcmc(pos, ninter, progress=True, skip_initial_state_check=True)

    samples = sampler.chain[:, nburn:, :].reshape((-1, ndim))

    samples_redim = np.zeros((np.shape(samples)[0], 7))
    samples_redim[:, 0:5] = samples[:, 0:5]
    samples_redim[:, 5] = samples[:, 7]
    samples_redim[:, 6] = np.log10(10 ** (np.nanmax(samples[:, 5:], axis=1)) - 10 ** (samples[:, 7]))

    L_BB_peak, t_peak, sigma, t0, p, log_T_peak, delt_T = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
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
    t_grid = t_peak[0] + np.arange(-60, 301, 30)
    flag_T_grid = models.gen_flag_T_grid(t, t_grid, T_t)
    T_t[~flag_T_grid] = np.nan
    T_t_p16[~flag_T_grid] = np.nan
    T_t_p84[~flag_T_grid] = np.nan
    delt_t = [str(np.arange(-60, 301, 30)[i]) for i in range(13)]
    delt_t[2] = 'peak'

    for i in range(13):
        T_i = np.array([T_t[i], T_t_p16[i], T_t_p84[i]])
        T_i = tools.round_small(T_i, 0.01)
        model_file.write(
            'T(' + str(delt_t[i]) + ')' + '\t' + '{:.2f}'.format(T_i[0]) + '\t' + '{:.2f}'.format(T_i[2]) + '\t' +
            '{:.2f}'.format(T_i[1]) + '\n')

    model_file.write(
        'ΔT' + '\t' + '{:.1f}'.format(delt_T[0]) + '\t' + '{:.1f}'.format(delt_T[2]) + '\t' +
        '{:.1f}'.format(delt_T[1]) + '\n')
    model_file.close()

    theta_median = np.concatenate(([L_BB_peak[0], t_peak[0], sigma[0], t0[0], p[0]], T_t))
    theta_err_p16 = np.concatenate(([L_BB_peak[2], t_peak[2], sigma[2], t0[2], p[2]], T_t_p16))
    theta_err_p84 = np.concatenate(([L_BB_peak[1], t_peak[1], sigma[1], t0[1], p[1]], T_t_p84))
    theta_err = np.nanmin([theta_err_p16, theta_err_p84], axis=0)

    log_T, log_T_err, log_BB, log_BB_err, log_R, log_R_err = models.Blackbody_evolution(t, theta_median, theta_err)

    model_file = open(os.path.join(modelling_dir, 'light_curve_model.txt'), 'a')
    model_file.write('\n')
    model_file.write('# Blackbody Evolution:\n')
    model_file.write('MJD' +
                     '\t' + 'log_L_BB' + '\t' + 'log_L_BB_err' +
                     '\t' + 'log_R' + '\t' + 'log_R_err' +
                     '\t' + 'log_T' + '\t' + 'log_T_err' + '\n')
    flag_300_days = (t - t_peak[0]) < 300
    for yy in range(len(t[flag_300_days])):
        model_file.write('{:.2f}'.format(t[yy]) +
                         '\t' + '{:.2f}'.format(log_BB[yy]) + '\t' + '{:.2f}'.format(log_BB_err[yy]) +
                         '\t' + '{:.2f}'.format(log_R[yy]) + '\t' + '{:.2f}'.format(log_R_err[yy]) +
                         '\t' + '{:.2f}'.format(log_T[yy]) + '\t' + '{:.2f}'.format(log_T_err[yy]) + '\n')

    model_file.close()

    theta_median_redim = [L_BB_peak[0], t_peak[0], sigma[0], t0[0], p[0], log_T_peak[0], delt_T[0]]
    labels = [r"log $L_{BB\,peak}$", r'$t_{peak}$', r'$\sigma$', r'$t_0$', r'$p$', r"log $T_{peak}$",
              r'log $\Delta\,T$']

    plot_dir = os.path.join(tde_dir, 'plots', 'modelling')
    fig_name = 'corner_plot_model2'
    plot_corner(plot_dir, fig_name, theta_median_redim, samples_redim, labels, show=True)
    plot_models(tde_name, tde_dir, z, show=True)
    plot_BB_evolution(tde_name, tde_dir, show=True)
    plot_SED(tde_name, tde_dir, z, sampler, nwalkers, nburn, ninter, show=True)
