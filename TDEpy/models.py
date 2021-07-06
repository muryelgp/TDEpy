import itertools

import numpy as np
import scipy
from astropy.constants import h, c, k_B, sigma_sb
import astropy.units as u

def blackbody(T, wl):
    """ Blackbody flux as a function of wavelength (A) and temperature (K).
        returns units of erg/s/cm2
        """
    #from scipy.constants import h, k, c

    wl = wl * u.Angstrom # convert to metres
    T = T * u.K
    nu = c.cgs/wl.cgs
    flux_wl = ((2 * nu.cgs**3 * h.cgs) / c.cgs ** 2) / (np.exp((h.cgs * nu) / (k_B.cgs * T.cgs)) - 1).cgs  #

    return flux_wl*nu


def bolometric_correction(T, wl):
    K_wl = ((sigma_sb.cgs * ((10 ** T * u.K) ** 4)).cgs / (np.pi * blackbody(10 ** T, wl))).cgs.value
    return K_wl


def Blackbody_var_T_gauss_rise_powerlaw_decay(t, wl, theta):

    log_L_peak, t_peak, sigma, t0, p, *T_grid = theta

    t_array = np.tile(t, (len(wl), 1)).transpose()
    light_curve_shape = np.zeros(np.shape(t_array))
    delt_t = t_array - t_peak

    before_peak = delt_t <= 0
    light_curve_shape[before_peak] = np.exp(-1 * (delt_t[before_peak]) ** 2 / (2 * sigma ** 2))

    after_peak = delt_t > 0
    light_curve_shape[after_peak] = ((delt_t[after_peak] + t0)/t0)**(p)

    t_grid = t_peak + np.arange(-60, 361, 30)
    flag_T_grid = gen_flag_T_grid(t, t_grid, T_grid)
    T_t = np.interp(t, t_grid[flag_T_grid], np.array(T_grid)[flag_T_grid])

    wl_array = np.tile(wl, (len(t), 1))
    T_t_array = np.tile(T_t, (len(wl), 1)).transpose()

    blackbody_t_wl = ((np.pi * blackbody(10**T_t_array, wl_array)) / (sigma_sb.cgs * ((10**T_t_array * u.K) ** 4)).cgs).cgs.value

    flag_one_year = delt_t > 360
    blackbody_t_wl[flag_one_year] = np.nan

    model = 10**log_L_peak * blackbody_t_wl * light_curve_shape

    return model


def gen_flag_T_grid(t, t_grid, T_grid):
    flag_right_T_grid = np.append(
        [np.sum((t > t_grid[i]) & (t < t_grid[i] + 30)) > 0 for i in range(len(T_grid) - 1)],
        False)
    flag_left_T_grid = np.concatenate(
        ([False], [np.sum((t < t_grid[i]) & (t > t_grid[i] - 30)) > 0 for i in range(1, len(T_grid))]))
    flag_T_grid = flag_right_T_grid | flag_left_T_grid
    return flag_T_grid


def const_T_gauss_rise_exp_decay(t, wl, theta):
    log_L_W2_peak, t_peak, sigma, tau, log_T0 = theta

    t_array = np.tile(t, (len(wl), 1)).transpose()
    light_curve_shape = np.zeros(np.shape(t_array))
    delt_t = t_array - t_peak

    before_peak = delt_t <= 0
    light_curve_shape[before_peak] = np.exp(-1 * (delt_t[before_peak]) ** 2 / (2 * sigma ** 2))

    after_peak = delt_t > 0
    light_curve_shape[after_peak] = np.exp(-1*delt_t[after_peak]/tau)

    wl_array = np.tile(wl, (len(t), 1))
    T_t_array = np.zeros(np.shape(wl_array))
    T_t_array[:] = log_T0
    wl_ref = np.zeros(np.shape(wl_array))
    wl_ref[:] = wl[0]

    blackbody_t_wl = blackbody(10 ** T_t_array, wl_array) / blackbody(10 ** T_t_array, wl_ref)
    flag_100_days = delt_t > 100
    blackbody_t_wl[flag_100_days] = np.nan

    model = 10 ** log_L_W2_peak * blackbody_t_wl * light_curve_shape
    return model


def lnprior(theta, model_name, observables):
    if model_name == 'Blackbody_var_T_gauss_rise_powerlaw_decay':

        log_L_peak, t_peak, sigma, t0, p, *T_grid = theta  # , p
        t, wl, T0, sed, sed_err = observables

        # setting flat priors
        t_max_L = t[np.where(sed == np.nanmax(sed))[0]][0]
        t_peak_prior = t_max_L - 30 <= t_peak <= t_max_L + 30
        sigma_prior = 1 <= sigma <= 10 ** 1.5
        t0_prior = 1 <= t0 <= 500
        p_prior = -5 <= p <= 0

        t_grid = t_peak + np.arange(-60, 361, 30)
        flag_T_grid = gen_flag_T_grid(t, t_grid, T_grid)
        T_t = np.interp(t, t_grid[flag_T_grid], np.array(T_grid)[flag_T_grid])
        T_grid_prior = (abs(np.diff(10**np.array(T_t))/(np.diff(t)+0.1)) < 200).all()

        if sigma_prior and t0_prior and t_peak_prior and p_prior and T_grid_prior:
            return np.nansum(-0.5*(np.array(T_grid)[flag_T_grid]-T0)**2/0.1**2)
        else:
            return -np.inf

    if model_name == 'const_T_gauss_rise_exp_decay':

        log_L_W2_peak, t_peak, sigma, tau, log_T0 = theta
        t, wl, sed, sed_err = observables

        # setting flat priors
        log_L_W2_peak_prior = np.nanmax(sed[:, 0])/2 <= 10**log_L_W2_peak <= np.nanmax(sed[:, 0]) * 2
        t_max_L = t[np.where(sed[:, 0] == np.nanmax(sed[:, 0]))[0]][0]
        t_peak_prior = t_max_L - 30 <= t_peak <= t_max_L + 30
        sigma_prior = 1 <= sigma <= 10 ** 1.5
        tau_prior = 1 <= tau <= 10**3
        T0_grid_prior = 4 <= log_T0 <= 5
        if sigma_prior  and log_L_W2_peak_prior and t_peak_prior and tau_prior and T0_grid_prior:
            return 0.0
        else:
            return -np.inf


def lnlike(theta, model_name, observables):
    if model_name == 'Blackbody_var_T_gauss_rise_powerlaw_decay':
        t, wl, T0, sed, sed_err = observables

        model = Blackbody_var_T_gauss_rise_powerlaw_decay(t, wl, theta)
        err = sed_err
        obs = sed
        # MLE is y - model squared over sigma squared
        like = -0.5 * np.nansum(((obs - model) ** 2.) / err ** 2)
        return like

    if model_name == 'const_T_gauss_rise_exp_decay':
        t, wl, sed, sed_err = observables

        model = const_T_gauss_rise_exp_decay(t, wl, theta)
        err = sed_err
        obs = sed
        # MLE is y - model squared over sigma squared
        like = -0.5 * np.nansum(((obs - model) ** 2.) / err ** 2)
        return like


def lnprob(theta, model_name, observables):
    lp = lnprior(theta, model_name, observables)
    if not np.isfinite(lp):
        return -np.inf
    else:
        prob = lp + lnlike(theta, model_name, observables)
        return prob
