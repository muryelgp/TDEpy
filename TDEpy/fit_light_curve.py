import os
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import matplotlib.pyplot as plt


def read_sw_light_curve(band, phot_dir):
    file = os.path.join(phot_dir, band + '.txt')
    _, mjd, _, _, flux_dens, flux_dens_err, _ = np.loadtxt(file, unpack=True, skiprows=2)
    flag_finite = (mjd < 0) | (flux_dens_err < 0)
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


def run_fit(tde_dir, z):
    # defining variables
    phot_dir = os.path.join(tde_dir, 'photometry', 'host_sub')
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    dist = cosmo.luminosity_distance(float(z)).to('cm')

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


    # filter wavelengths in the order: w2, m2, w1, U, B, g, r
    band_wls = [2085.73, 2245.78, 2684.14, 3520.95, 4346.25, 4722.74, 6339.61]

    # Creating an array to represent the SED(t) with the same time bin as the epoch array
    sed_x_t = np.zeros((len(epochs), n_bands))
    sed_x_t[:] = np.nan
    # And another one for the uncertainties
    sed_err_x_t = np.zeros((len(epochs), n_bands))
    sed_err_x_t[:] = np.nan

    mjd_list = np.array([mjd_w2, mjd_m2, mjd_w1, mjd_U, mjd_B, mjd_g, mjd_r])
    lum_list = 4 * np.pi * (dist.value ** 2) * np.array(
        [flux_dens_w2 * band_wls[0], flux_dens_m2 * band_wls[1], flux_dens_w1 * band_wls[2],
         flux_dens_U * band_wls[3], flux_dens_B * band_wls[4], flux_dens_g * band_wls[5],
         flux_dens_r * band_wls[6]])
    lum_err_list = 4 * np.pi * (dist.value ** 2) * np.array(
        [flux_dens_err_w2 * band_wls[0], flux_dens_err_m2 * band_wls[1], flux_dens_err_w1 * band_wls[2],
         flux_dens_err_U * band_wls[3], flux_dens_err_B * band_wls[4], flux_dens_err_g * band_wls[5],
         flux_dens_err_r * band_wls[6]])

    for i in range(len(lum_list[:5])):
        flag = (epochs >= sw_min_mjd) & (epochs <= sw_max_mjd)
        sed_x_t[flag, i] = lum_list[i][ordering]
        sed_err_x_t[flag, i] = lum_err_list[i][ordering]

    for i in range(len(lum_list[5:])):
        if mjd_list[5+i] is not None:
            flag = (epochs >= ztf_min_mjd) & (epochs <= ztf_max_mjd)
            sed_x_t[flag, 5+i] = np.interp(epochs[flag], mjd_list[5+i], lum_list[5+i])
            sed_err_x_t[flag, 5+i] = np.interp(epochs[flag], mjd_list[5+i], lum_err_list[5+i])

    color = ['magenta', 'darkviolet', 'navy', 'blue', 'cyan', 'green', 'red']
    for i in range(7):
        y = sed_x_t[:, i]
        y_err = sed_err_x_t[:, i]
        flag = np.isfinite(y)
        x = epochs
        plt.errorbar(x[flag], y[flag], yerr=y_err[flag], fmt='o', color=color[i])

    plt.show()