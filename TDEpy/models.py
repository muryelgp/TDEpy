import numpy as np
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

def Blackbody_evolution(t, single_color, T_interval, theta, theta_err):
    log_L_BB, t_peak, sigma, t0, p, *T_grid = theta
    log_L_BB_err, t_peak_err, sigma_err, t0_err, p_err, *T_grid_err = theta_err


    # Temperature evolution
    t_grid = t_peak + np.arange(-60, 301, T_interval)
    flag_T_grid = gen_flag_T_grid(t, t_grid, T_grid, T_interval)
    log_T_t = np.interp(t, t_grid[flag_T_grid], np.array(T_grid)[flag_T_grid])
    log_T_t = np.interp(t, t[~single_color], log_T_t[~single_color])
    log_T_t_err = np.interp(t, t_grid[flag_T_grid], np.array(T_grid_err)[flag_T_grid])
    log_T_t_err = np.interp(t, t[~single_color], log_T_t_err[~single_color])

    # BB Luminosity Evolution
    log_L_BB_sample = np.tile(np.random.normal(log_L_BB, log_L_BB_err, 100), (len(t), 1)).transpose()

    sigma_sample = np.tile(np.random.normal(sigma, sigma_err, 100), (len(t), 1)).transpose()
    t0_sample = np.tile(np.random.normal(t0, t0_err, 100), (len(t), 1)).transpose()
    p_sample = np.tile(np.random.normal(p, p_err, 100), (len(t), 1)).transpose()

    t_array = np.tile(t, (100, 1))
    light_curve_shape = np.zeros(np.shape(t_array))
    delt_t = t_array - t_peak

    before_peak = t <= t_peak
    light_curve_shape[:, before_peak] = np.exp((-1 * (delt_t[:, before_peak]) ** 2)/ (2. * sigma_sample[:, before_peak] ** 2.))

    after_peak = t > t_peak

    light_curve_shape[:, after_peak] = ((delt_t[:, after_peak] + t0_sample[:, after_peak])/t0_sample[:, after_peak])**(-1.*p_sample[:, after_peak])

    L_BB_t_sample = 10**log_L_BB_sample * light_curve_shape
    L_BB_t_median= np.nanmedian(L_BB_t_sample, axis=0)
    L_BB_t_err_p16 = L_BB_t_median - np.nanpercentile(L_BB_t_sample, 16, axis=0)
    L_BB_t_err_p84 = np.nanpercentile(L_BB_t_sample, 84, axis=0) - L_BB_t_median
    L_BB_t_err = np.mean([L_BB_t_err_p16, L_BB_t_err_p84], axis=0)

    T_t_sample = np.array([np.random.normal(log_T_t[i], log_T_t_err[i], 100) for i in range(len(log_T_t))]).transpose()

    R_sample = np.sqrt(L_BB_t_sample / (4. * 3.14 * 5.6704e-5 * (10**T_t_sample)**4))
    R_median = np.nanmedian(R_sample, axis=0)
    R_err_p16 = R_median - np.nanpercentile(R_sample, 16, axis=0)
    R_t_err_p84 = np.nanpercentile(R_sample, 84, axis=0) - R_median
    R_err = np.mean([R_err_p16, R_t_err_p84], axis=0)


    return log_T_t, log_T_t_err, np.log10(L_BB_t_median), 0.432*(L_BB_t_err/L_BB_t_median), np.log10(R_median), 0.432*(R_err/R_median)


def bolometric_correction(T, wl):
    K_wl = ((sigma_sb.cgs * ((10 ** T * u.K) ** 4)).cgs / (np.pi * blackbody(10 ** T, wl))).cgs.value
    return K_wl


def L_bol(t, theta):
    log_L_BB_peak, t_peak, sigma, t0, p, *T_grid = theta
    delt_t = t - t_peak
    light_curve_shape = np.zeros(np.shape(delt_t))
    before_peak = delt_t <= 0
    light_curve_shape[before_peak] = np.exp(-1 * (delt_t[before_peak]) ** 2 / (2 * sigma ** 2))

    after_peak = delt_t > 0
    light_curve_shape[after_peak] = ((delt_t[after_peak] + t0) / t0) ** (-1 * p)
    model = 10 ** log_L_BB_peak * light_curve_shape
    return model


def Blackbody_var_T_gauss_rise_powerlaw_decay(t, single_color, wl, T_interval, theta):

    log_L_BB_peak, t_peak, sigma, t0, p, *T_grid = theta

    t_array = np.tile(t, (len(wl), 1)).transpose()
    light_curve_shape = np.zeros(np.shape(t_array))
    delt_t = t_array - t_peak

    before_peak = delt_t <= 0
    light_curve_shape[before_peak] = np.exp(-1 * (delt_t[before_peak]) ** 2 / (2 * sigma ** 2))

    after_peak = delt_t > 0
    light_curve_shape[after_peak] = ((delt_t[after_peak] + t0)/t0)**(-1*p)

    t_grid = t_peak + np.arange(-60, 301, T_interval)
    flag_T_grid = gen_flag_T_grid(t, t_grid, T_grid, T_interval)
    T_t = np.interp(t, t_grid[flag_T_grid], np.array(T_grid)[flag_T_grid])
    T_t = np.interp(t, t[np.invert(single_color)], T_t[np.invert(single_color)])


    wl_array = np.tile(wl, (len(t), 1))
    T_t_array = np.tile(T_t, (len(wl), 1)).transpose()

    blackbody_t_wl = ((np.pi * blackbody(10**T_t_array, wl_array)) / (sigma_sb.cgs * ((10**T_t_array * u.K) ** 4)).cgs).cgs.value

    model = 10**log_L_BB_peak * blackbody_t_wl * light_curve_shape

    return model


def gen_flag_T_grid(t, t_grid, T_grid, T_interval):
    flag_right_T_grid = np.append(
        [np.sum((t > t_grid[i]) & (t < t_grid[i] + T_interval)) > 0 for i in range(len(T_grid) - 1)],
        False)
    flag_left_T_grid = np.concatenate(
        ([False], [np.sum((t < t_grid[i]) & (t > t_grid[i] - T_interval)) > 0 for i in range(1, len(T_grid))]))
    flag_finite = np.isfinite(T_grid)
    flag_T_grid = (flag_right_T_grid | flag_left_T_grid) & flag_finite
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
    model = 10 ** log_L_W2_peak * blackbody_t_wl.value * light_curve_shape
    return model


def lnprior(theta, model_name, observables):
    if model_name == 'Blackbody_var_T_gauss_rise_powerlaw_decay':

        log_L_peak, t_peak, sigma, t0, p, *T_grid = theta  # , p
        t, wl, T_interval, theta_median, sed, sed_err = observables
        _, t_peak_model1, sigma_model1, _, T0 = theta_median

        # setting flat priors
        t_peak_prior = t_peak_model1 - 5 <= t_peak <= t_peak_model1 + 5
        sigma_prior = sigma_model1 - 0.3*sigma_model1 <= sigma <= sigma_model1 + 0.3*sigma_model1
        t0_prior = 1 <= t0 <= 300
        p_prior = 0 <= p <= 5

        t_grid = t_peak + np.arange(-60, 301, T_interval)
        flag_T_grid = gen_flag_T_grid(t, t_grid, T_grid, T_interval)
        T_grid_prior = (abs(np.diff(10 ** np.array(T_grid)[flag_T_grid]) / (np.diff(t_grid[flag_T_grid]))) < 300)[1:].all()
        T_grid_prior = T_grid_prior & ((np.array(T_grid) < 5) & (np.array(T_grid) > 4)).all()

        if sigma_prior and t0_prior and t_peak_prior and p_prior and T_grid_prior:
            return 0.0 #np.nansum(-0.5*(np.array(T_grid)[flag_T_grid]-T0)**2/0.1**2)
        else:
            return -np.inf

    if model_name == 'const_T_gauss_rise_exp_decay':

        log_L_W2_peak, t_peak, sigma, tau, log_T0 = theta
        t, wl, sed, sed_err = observables

        # setting flat priors
        log_L_W2_peak_prior = np.nanmax(sed[:, 0])/100 <= 10**log_L_W2_peak <= np.nanmax(sed[:, 0]) * 100
        t_max_L = t[np.where(sed[:, 0] == np.nanmax(sed[:, 0]))[0]][0]
        t_peak_prior = t_max_L - 100 <= t_peak <= t_max_L + 100
        sigma_prior = 1 <= sigma <= 10 ** 2
        tau_prior = 1 <= tau <= 300
        T0_grid_prior = 4 <= log_T0 <= 5
        if sigma_prior and log_L_W2_peak_prior and t_peak_prior and tau_prior and T0_grid_prior:
            return 0.0
        else:
            return -np.inf


def lnlike(theta, model_name, observables):
    if model_name == 'Blackbody_var_T_gauss_rise_powerlaw_decay':
        t, wl, T_interval, theta_median, sed, sed_err = observables
        single_color = np.array([np.sum(np.isfinite(sed[:, :])) == 1 for i in range(len(t))])

        t_peak = theta[1]
        model = Blackbody_var_T_gauss_rise_powerlaw_decay(t, single_color, wl, T_interval, theta)
        err = sed_err
        obs = sed
        flag_300_days = (t - t_peak) <= 300
        # MLE is y - model squared over sigma squared
        like = -0.5 * np.nansum(((obs[flag_300_days] - model[flag_300_days]) ** 2.) / err[flag_300_days] ** 2)
        return like

    if model_name == 'const_T_gauss_rise_exp_decay':
        t, wl, sed, sed_err = observables

        model = const_T_gauss_rise_exp_decay(t, wl, theta)
        err = sed_err
        obs = sed
        flag_100_days = (t - theta[1]) <= 100
        # MLE is y - model squared over sigma squared
        like = -0.5 * np.nansum(((obs[flag_100_days] - model[flag_100_days]) ** 2.) / err[flag_100_days] ** 2)
        return like


def lnprob(theta, model_name, observables):
    lp = lnprior(theta, model_name, observables)
    if not np.isfinite(lp):
        return -np.inf
    else:
        prob = lp + lnlike(theta, model_name, observables)
        return prob
