from . import models as models
from . import tools as tools
from . import plots as plots
import os
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import scipy.optimize as op
import emcee
from multiprocessing import Pool


def read_BB_evolution(model_dir):
    path_to_model_file = os.path.join(model_dir, 'blackbody_evolution.txt')
    try:
        t, log_BB, log_BB_err, log_R, log_R_err, log_T, log_T_err, single_band = np.loadtxt(path_to_model_file, skiprows=2, unpack=True)
    except:
        t, log_BB, log_BB_err, log_R, log_R_err, log_T, log_T_err = np.loadtxt(path_to_model_file,
                                                                                            skiprows=2, unpack=True)
        single_band = np.zeros(len(t), dtype=bool)
    return t, log_BB, log_BB_err, log_R, log_R_err, log_T, log_T_err, single_band

def save_blackbody_evol(model, epochs, BB_evol, good_epochs):
    log_T, log_T_err, log_BB, log_BB_err, log_R, log_R_err = BB_evol
    if model == 'BB_FT_FS':
        model_file = open(os.path.join('blackbody_evolution.txt'), 'w')
        model_file.write('# Blackbody Evolution:\n')
        model_file.write('MJD' +
                         '\t' + 'log_L_BB' + '\t' + 'log_L_BB_err' +
                         '\t' + 'log_R' + '\t' + 'log_R_err' +
                         '\t' + 'log_T' + '\t' + 'log_T_err' + '\n')
        for yy in range(len(epochs)):
            model_file.write('{:.2f}'.format(epochs[yy]) +
                             '\t' + '{:.2f}'.format(log_BB[yy]) + '\t' + '{:.2f}'.format(log_BB_err[yy]) +
                             '\t' + '{:.2f}'.format(log_R[yy]) + '\t' + '{:.2f}'.format(log_R_err[yy]) +
                             '\t' + '{:.2f}'.format(log_T[yy]) + '\t' + '{:.2f}'.format(log_T_err[yy]) + '\n')
        model_file.close()
    if model == 'BB_VT_GPS':
        model_file = open(os.path.join('blackbody_evolution.txt'), 'w')
        model_file.write('# Blackbody Evolution:\n')
        model_file.write('MJD' +
                         '\t' + 'log_L_BB' + '\t' + 'log_L_BB_err' +
                         '\t' + 'log_R' + '\t' + 'log_R_err' +
                         '\t' + 'log_T' + '\t' + 'log_T_err' + '\t' +  'no_UV_flag' + '\n')
        for yy in range(len(epochs)):
            model_file.write('{:.2f}'.format(epochs[yy]) +
                             '\t' + '{:.2f}'.format(log_BB[yy]) + '\t' + '{:.2f}'.format(log_BB_err[yy]) +
                             '\t' + '{:.2f}'.format(log_R[yy]) + '\t' + '{:.2f}'.format(log_R_err[yy]) +
                             '\t' + '{:.2f}'.format(log_T[yy]) + '\t' + '{:.2f}'.format(log_T_err[yy]) + '\t' +  '{:d}'.format(~good_epochs[yy])+ '\n')
        model_file.close()


def read_sw_light_curve(band, phot_dir):
    file = os.path.join(phot_dir, band + '.txt')
    _, mjd, _, mag_err, flux_dens, flux_dens_err, snr = np.loadtxt(file, unpack=True, skiprows=2)
    flag_finite = (mjd < 0) | (flux_dens_err < 0) | (flux_dens == 0) | (snr < 0.1) | (mag_err > 1.0)
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


def gen_observables(tde):
    # defining variables
    phot_dir = os.path.join(tde.tde_dir, 'photometry', 'host_sub')
    cosmo = FlatLambdaCDM(H0=67, Om0=0.3)
    dist = cosmo.luminosity_distance(float(tde.z)).to('cm')

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

            #Eliminating B if there is ztf g and r
            sed_x_t[flag, 4] = np.NaN
            sed_err_x_t[flag, 4] = np.NaN

    band_wls = band_wls / (1 + tde.z)
    return epochs, np.array(all_bands), band_wls, sed_x_t, sed_err_x_t,

class observables:
    def __init__(self, tde):
        self.epochs,  self.bands, self.band_wls, self.sed, self.sed_err, = gen_observables(tde)
