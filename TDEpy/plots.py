import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
from astropy.utils.exceptions import AstropyWarning
from astropy.cosmology import FlatLambdaCDM

warnings.simplefilter('ignore', category=AstropyWarning)
import corner
import pkg_resources
from astropy.constants import h, c, k_B, sigma_sb
import astropy.units as u
from . import fit_light_curve as fit_light_curve
from . import models as models


def plot_light_curve(tde, host_sub, units, show, title=True):
    # Creating color and legend dictionaries for each band
    color_dic = dict(sw_uu='dodgerblue', sw_bb='cyan', sw_vv='gold', sw_w1='navy', sw_m2='darkviolet',
                     sw_w2='magenta', ztf_g='green', ztf_r='red', ogle_I='darkred')
    legend_dic = dict(sw_uu='$U$', sw_bb='$B$', sw_vv='$V$', sw_w1=r'$UV~W1$', sw_m2=r'$UV~M2$',
                      sw_w2=r'$UV~W2$', ztf_g='g', ztf_r='r', ogle_I='I')

    band_dic = dict(sw_uu='U', sw_bb='B', sw_vv='V', sw_w1='UVW1', sw_m2='UVM2',
                    sw_w2='UVW2', ztf_g='g', ztf_r='r', ogle_I='I')

    wl_dic = dict(sw_uu=3465, sw_bb=4392.25, sw_vv=5411, sw_w1=2684.14, sw_m2=2245.78,
                  sw_w2=2085.73, ztf_g=4722.74, ztf_r=6339.61, ogle_I=8060)

    fig, ax = plt.subplots(figsize=(12, 7))
    if not host_sub:
        bands = ['sw_uu', 'sw_bb', 'sw_vv', 'sw_w1', 'sw_m2', 'sw_w2']

        mjd_max, mjd_min, bands_plotted = 0, 1e10, []
        for band in bands:
            try:
                data_path = os.path.join(tde.tde_dir, 'photometry', 'obs', str(band) + '.txt')
                obsid, mjd, abmag, abmage, flu, flue = np.loadtxt(data_path, skiprows=1, unpack=True)
            except:
                continue

            flag = (abmag > 0) & (abmage < 1)
            if np.sum(flag) > 0:
                ax.errorbar(mjd[flag], abmag[flag], yerr=abmage[flag], marker="o", linestyle='',
                            color=color_dic[band],
                            linewidth=1, markeredgewidth=0.5, markeredgecolor='black', markersize=8, elinewidth=0.7,
                            capsize=0,
                            label=legend_dic[band])
                bands_plotted.append(band)
                if np.max(mjd[flag]) > mjd_max:
                    mjd_max = np.max(mjd[flag])
                if np.min(mjd[flag]) < mjd_min:
                    mjd_min = np.min(mjd[flag])
        try:
            host_bands, model_wl_c, model_ab_mag, model_ab_mag_err, model_flux, model_flux_err, catalogs = \
                np.loadtxt(os.path.join(tde.host_dir, 'host_phot_model.txt'),
                           dtype={'names': (
                               'band', 'wl_0', 'ab_mag', 'ab_mag_err',
                               'flux_dens', 'flux_dens_err', 'catalog'),
                               'formats': (
                                   'U5', np.float, np.float, np.float,
                                   np.float, np.float, 'U10')},
                           unpack=True, skiprows=1)
            for band in bands_plotted:
                delt_mjd = (mjd_max - mjd_min) * 0.1
                if band[0] == 's':
                    band_host_flag = host_bands == band_dic[band]
                    ax.errorbar(mjd_max + delt_mjd, model_ab_mag[band_host_flag][0],
                                yerr=model_ab_mag_err[band_host_flag][0],
                                marker="*", linestyle='', color=color_dic[band], linewidth=1, markeredgewidth=0.5,
                                markeredgecolor='black', markersize=15, elinewidth=0.7, capsize=0)


        except:
            pass
        ax.set_ylabel('AB mag', fontsize=20)
        ax.invert_yaxis()

    elif host_sub:
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        dist = cosmo.luminosity_distance(float(tde.z)).to('cm')
        bands = ['sw_uu', 'sw_bb', 'sw_vv', 'sw_w1', 'sw_m2', 'sw_w2', 'ztf_r', 'ztf_g', 'ogle_I']

        for band in bands:
            # Loading and plotting Swift data
            if band[0] == 's':
                try:
                    data_path = os.path.join(tde.tde_dir, 'photometry', 'host_sub', str(band) + '.txt')
                    obsid, mjd, abmag, abmage, flux, fluxe, signal_host = np.loadtxt(data_path, skiprows=2,
                                                                                     unpack=True)
                except:
                    continue
                flag = (abmag > 0) & (abmage < 1)
                if np.sum(flag) > 0:
                    if units == 'lum':
                        ax.errorbar(mjd[flag], 4 * np.pi * (dist.value ** 2) * flux[flag] * wl_dic[band],
                                    yerr=4 * np.pi * (dist.value ** 2) * fluxe[flag] * wl_dic[band], marker="o", linestyle='',
                                    color=color_dic[band], linewidth=1, markeredgewidth=0.5, markeredgecolor='black',
                                    markersize=8, elinewidth=0.7, capsize=0, label=legend_dic[band])
                    if units == 'mag':
                        ax.errorbar(mjd[flag], abmag[flag], yerr=abmage[flag], marker="o",
                                    linestyle='', color=color_dic[band], linewidth=1, markeredgewidth=0.5, markeredgecolor='black',
                                    markersize=8, elinewidth=0.7, capsize=0, label=legend_dic[band])
            elif band[0] == 'z':
                if os.path.exists(os.path.join(tde.tde_dir, 'photometry', 'host_sub', str(band) + '.txt')):
                    mjd, abmag, abmage, flux, fluxe = np.loadtxt(
                        os.path.join(tde.tde_dir, 'photometry', 'host_sub', str(band) + '.txt'), skiprows=2,
                        unpack=True)
                    if units == 'lum':
                        ax.errorbar(mjd, 4 * np.pi * (dist.value ** 2) * flux * wl_dic[band],
                                    yerr=4 * np.pi * (dist.value ** 2) * fluxe * wl_dic[band],
                                    marker="o", linestyle='', color=color_dic[band], linewidth=1, markeredgewidth=0.5,
                                    markeredgecolor='black', markersize=8, elinewidth=0.7, capsize=0,
                                    label=legend_dic[band])
                    if units == 'mag':
                        ax.errorbar(mjd, abmag, yerr=abmage, marker="o",
                                    linestyle='', color=color_dic[band], linewidth=1, markeredgewidth=0.5,
                                    markeredgecolor='black',
                                    markersize=8, elinewidth=0.7, capsize=0, label=legend_dic[band])

            elif band[0] == 'o':
                if os.path.exists(os.path.join(tde.tde_dir, 'photometry', 'host_sub', str(band) + '.txt')):
                    mjd, abmag, abmage, flux, fluxe = np.loadtxt(
                        os.path.join(tde.tde_dir, 'photometry', 'host_sub', str(band) + '.txt'), skiprows=2,
                        unpack=True)
                    ax.errorbar(mjd, 4 * np.pi * (dist.value ** 2) * flux * wl_dic[band],
                                yerr=4 * np.pi * (dist.value ** 2) * fluxe * wl_dic[band],
                                marker="o", linestyle='', color=color_dic[band], linewidth=1, markeredgewidth=0.5,
                                markeredgecolor='black', markersize=8, elinewidth=0.7, capsize=0,
                                label=legend_dic[band])
            if units == 'lum':
                ax.set_ylabel(r'$\rm{\nu\,L_{\nu} \ [erg \ s^{-1}]}$', fontsize=20)
                ax.set_yscale('log')
            if units == 'mag':
                ax.set_ylabel('AB mag', fontsize=20)
                ax.invert_yaxis()

    ax.set_xlabel('MJD', fontsize=20)

    if host_sub:
        fig_name = 'host_sub_light_curve.'
    else:
        fig_name = 'light_curve.'
    if title:
        ax.set_title(tde.name)

    plt.tight_layout()
    plt.legend(ncol=2)

    try:
        os.mkdir(os.path.join(tde.plot_dir))
    except:
        pass
    try:
        os.mkdir(os.path.join(tde.plot_dir, 'photometry'))
    except:
        pass
    plt.savefig(os.path.join(tde.plot_dir, 'photometry', fig_name + '.pdf'), bbox_inches='tight')
    if show:
        plt.show()


def plot_host_sed(tde, show):
    try:
        band, wl_c, ab_mag, ab_mag_err, catalogs, apertures = np.loadtxt(
            os.path.join(tde.host_dir, 'host_phot_obs.txt'),
            dtype={'names': (
                'band', 'wl_0', 'ab_mag', 'ab_mag_err',
                'catalog', 'aperture'),
                'formats': (
                    'U5', np.float, np.float, np.float, 'U10', 'U10')},
            unpack=True, skiprows=2)
    except:
        raise Exception('We should run download_host_data() before trying to plot it.')

    color_dic = {"WISE": "maroon", "UKIDSS": "coral", "2MASS": 'red', 'PAN-STARRS': 'green', 'DES': 'lime',
                 'SkyMapper': 'greenyellow', 'SDSS': 'blue', 'GALEX': 'darkviolet', 'Swift/UVOT': 'darkviolet'}

    finite = (np.isfinite(ab_mag)) & (np.isfinite(ab_mag_err))

    fig, ax = plt.subplots(figsize=(10, 10))
    for catalog in np.unique(catalogs[finite]):
        flag = (catalogs == catalog) & (np.isfinite(ab_mag)) & (np.isfinite(ab_mag_err))
        ax.errorbar(wl_c[flag], ab_mag[flag], yerr=ab_mag_err[flag], marker='D', linestyle=' ',
                    color=color_dic[catalog],
                    linewidth=3, markeredgecolor='black', markersize=8, elinewidth=3, capsize=5, capthick=3,
                    markeredgewidth=1, label=catalog)
    ax.invert_yaxis()
    for catalog in np.unique(catalogs[~finite]):
        flag = (catalogs == catalog) & (~np.isfinite(ab_mag_err))
        ax.errorbar(wl_c[flag], ab_mag[flag], yerr=0.5, lolims=np.ones(np.shape(ab_mag[flag]), dtype=bool),
                    marker='D', linetyle=' ', color=color_dic[catalog],
                    markeredgecolor='black', markersize=8, elinewidth=2, capsize=6, capthick=3,
                    markeredgewidth=1, label=catalog)
    plt.xscale('log')
    ax.set_xlim(700, 100000)
    ax.set_xticks([1e3, 1e4, 1e5])
    ymin, ymax = np.min(ab_mag) * 0.85, np.max(ab_mag) * 1.1
    ax.set_xticklabels(['0.1', '1', '10'])
    ax.tick_params(axis='both', labelsize=16)
    ax.set_ylim(ymax, ymin)
    ax.set_ylabel('AB mag', fontsize=20)
    ax.set_xlabel(r'Wavelength $\rm{[\mu m]}$', fontsize=20)
    plt.legend(loc=4)
    plt.tight_layout()



    try:
        os.mkdir(os.path.join(tde.plot_dir))
    except:
        pass
    try:
        os.mkdir(os.path.join(tde.plot_dir, 'host'))
    except:
        pass

    plt.savefig(os.path.join(tde.plot_dir, 'host', 'host_sed_obs.pdf'), bbox_inches='tight', dpi=300)
    if show:
        plt.show()


def plot_host_sed_fit(tde, title=True):
    fig, ax = plt.subplots(figsize=(10, 10))

    band, obs_wl_c, obs_ab_mag, obs_ab_mag_err, catalogs, apertures = \
        np.loadtxt(os.path.join(tde.host_dir, 'host_phot_obs.txt'),
                   dtype={'names': (
                       'band', 'wl_0', 'ab_mag', 'ab_mag_err',
                       'catalog', 'aperture'),
                       'formats': (
                           'U5', np.float, np.float, np.float,
                           'U10', 'U10')},
                   unpack=True, skiprows=2)

    band, model_wl_c, model_ab_mag, model_ab_mag_err, model_flux, model_flux_err, catalogs = \
        np.loadtxt(os.path.join(tde.host_dir, 'host_phot_model.txt'),
                   dtype={'names': (
                       'band', 'wl_0', 'ab_mag', 'ab_mag_err',
                       'flux_dens', 'flux_dens_err', 'catalog'),
                       'formats': (
                           'U5', np.float, np.float, np.float,
                           np.float, np.float, 'U10')},
                   unpack=True, skiprows=1)

    n_bands = int(np.where(band == 'V')[0])
    band_flag = [i < n_bands for i in range(len(model_wl_c))]

    spec_wl_0, spec_ab_mag, spec_ab_mag_p16, spec_ab_mag_p84, spec_flux, spec_flux_p16, spec_flux_p84 = np.loadtxt(
        os.path.join(tde.host_dir, 'host_spec_model.txt'),
        dtype={'names': (
            'wl_0', 'ab_mag', 'spec_ab_mag_p16', 'spec_ab_mag_p16'
                                                 'flux_dens', 'spec_flux_p16', 'spec_ab_mag_p84', 'tde/host'),
            'formats': (
                np.float, np.float, np.float,
                np.float, np.float, np.float, np.float, np.float)},
        unpack=True, skiprows=1)

    ax.plot(spec_wl_0, spec_ab_mag, label='Model spectrum (MAP)',
            lw=0.7, color='grey', alpha=0.8)
    ax.fill_between(spec_wl_0, spec_ab_mag_p16, spec_ab_mag_p84, alpha=.3, color='grey', label='Posterior')
    ax.errorbar(model_wl_c[band_flag], model_ab_mag[band_flag], yerr=model_ab_mag_err[band_flag],
                label='Model photometry (MAP)',
                marker='s', markersize=8, alpha=0.85, ls='', lw=3, ecolor='black', capsize=5,
                markerfacecolor='none', markeredgecolor='black',
                markeredgewidth=3)

    is_up_lim = ~np.isfinite(obs_ab_mag_err)
    ax.errorbar(obs_wl_c[~is_up_lim], obs_ab_mag[~is_up_lim], yerr=obs_ab_mag_err[~is_up_lim],
                label='Observed photometry', ecolor='red',
                marker='o', markersize=8, ls='', lw=3, alpha=0.85, capsize=5,
                markerfacecolor='none', markeredgecolor='red',
                markeredgewidth=3)
    ax.invert_yaxis()
    ax.errorbar(obs_wl_c[is_up_lim], obs_ab_mag[is_up_lim], yerr=0.5,
                lolims=np.ones(np.shape(obs_ab_mag_err[is_up_lim]), dtype=bool),
                marker='o', fmt='o', ecolor='red', alpha=0.85, lw=3, markeredgecolor='red',
                markerfacecolor='none', markersize=8, elinewidth=2, capsize=6, capthick=3,
                markeredgewidth=3)

    temp = np.interp(np.linspace(700, 100000, 10000), spec_wl_0, spec_ab_mag)
    ymin, ymax = temp.min() * 0.75, temp.max() * 1.1
    plt.xscale('log')
    ax.set_xlim(700, 100000)
    ax.set_xticks([1e3, 1e4, 1e5])
    ax.set_xticklabels(['0.1', '1', '10'])
    ax.tick_params(axis='both', labelsize=16)
    ax.set_ylim(ymax, ymin)
    ax.set_ylabel('AB mag', fontsize=20)
    ax.set_xlabel(r'Wavelength $\rm{[\mu m]}$', fontsize=20)

    if title:
        ax.set_title('Host Galaxy SED Fit (' + tde.name + ')')
    plt.legend(loc=2)

    _, map, median, p16, p84 = \
        np.loadtxt(os.path.join(tde.host_dir, 'host_properties.txt'),
                   dtype={'names': ('Parameter', 'MAP', 'median', 'p16', 'p84'),
                          'formats': ('U10', np.float, np.float, np.float, np.float)},
                   unpack=True, skiprows=1)
    labels = [r'log $M_{*}$', r'log $Z/Z_{\odot}$', r'$\rm{E(B-V)}$', r'$t_{\rm{age}}$',
              r'$\tau_{\rm{sfh}}$']
    units = [r'$M_{\odot}$', '', '', 'Gyr', 'Gyr']
    flag_min = p16 == 0.00
    p16[flag_min] = 0.01
    flag_min = p84 == 0.00
    p84[flag_min] = 0.01
    for i in range(5):
        plt.text(0.98, 0.30-0.06*i, labels[i] + ' = ' + r'$%.2f_{%.2f}^{%.2f}$' % (median[i], p16[i], p84[i]) + ' ' + units[i], verticalalignment='center', horizontalalignment='right', transform=ax.transAxes, fontsize=20)
    plt.tight_layout()

    return fig


def host_corner_plot(result, obs, model, sps, ebv, z):
    imax = np.argmax(result['lnprobability'])

    i, j = np.unravel_index(imax, result['lnprobability'].shape)
    theta_max = result['chain'][i, j, :].copy()

    try:
        parnames = np.array(result['theta_labels'], dtype='U20')
    except KeyError:
        parnames = np.array(result['model'].theta_labels())
    ind_show = slice(None)
    thin = 5
    chains = slice(None)
    start = 0
    # Get the arrays we need (trace, wghts)
    trace = result['chain'][..., ind_show]
    if trace.ndim == 2:
        trace = trace[None, :]
    trace = trace[chains, start::thin, :]
    wghts = result.get('weights', None)
    if wghts is not None:
        wghts = wghts[start::thin]
    samples = trace.reshape(trace.shape[0] * trace.shape[1], trace.shape[2])
    logify = ["mass"]
    # logify some parameters
    xx = samples.copy()
    for p in logify:
        if p in parnames:
            idx = parnames.tolist().index(p)
            xx[:, idx] = np.log10(xx[:, idx])
            parnames[idx] = "log({})".format(parnames[idx])
    bounds = []
    data = np.zeros(np.shape(xx))



    for i, x in enumerate(xx):
        a, b, c, d, e = x
        _,  _ , mfrac = model.predict(x, obs=obs, sps=sps)
        data[i, :] = np.log10((10**a)*mfrac), b, c/3.1, d, e
        #print(i)

    from astropy.cosmology import FlatLambdaCDM
    import astropy.units as u
    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)
    _, _, mfrac = model.predict(theta_max, obs=obs, sps=sps)
    theta_max = [np.log10(theta_max[0]*mfrac), theta_max[1], (theta_max[2])/3.1 , theta_max[3], theta_max[4]]
    lim = [[6, 12],
           [-2, 0.3],
           [ebv, 2/3.1],
           [0.01, cosmo.age(z).value],
           [1e-1, 1e2]]

    for i in range(np.shape(xx)[1]):
        sig1 = abs(theta_max[i] - np.percentile((data[:, i]), 16))
        sig2 = abs(np.percentile((data[:, i]), 84) - theta_max[i])
        if (theta_max[i] - 3 * sig1 < lim[i][0]):
            bounds.append((lim[i][0], theta_max[i] + 3 * sig2))
        elif ((theta_max[i] + 3 * sig2 > lim[i][1])):
            bounds.append((theta_max[i] - 3 * sig1, lim[i][1]))
        else:
            big_sig = max([sig1, sig2])
            bounds.append((theta_max[i] - 3 * big_sig, theta_max[i] + 3 * big_sig))

    labels = [r'log $M_{*}/M_{\odot}$', r'log $Z/Z_{\odot}$', r'$\rm{E(B-V)}$', r'$t_{\rm{age}}$',
              r'$\tau_{\rm{sfh}}$']
    cornerfig = corner.corner(data,
                              labels=labels,
                              quantiles=[0.16, 0.5, 0.84],
                              show_titles=True, title_kwargs={"fontsize": 12}, range=bounds)



    return cornerfig, data, theta_max


def color_mass(mass_list, color_list):
    from scipy.ndimage import gaussian_filter
    import matplotlib

    _, sdss_mass, _, _, sdss_color = np.loadtxt(pkg_resources.resource_filename("TDEpy", 'data/sdss_cor_M.txt'),
                                                unpack=True, skiprows=1)
    fig, ax = plt.subplots(figsize=(7, 5))
    h, xx, yy = np.histogram2d(sdss_mass, sdss_color, range=[[7.6, 11.3], [0.70, 3.05]], bins=15, density=True)
    xx, yy = np.meshgrid(xx, yy)

    xx = xx[:-1, :-1]  # + np.mean(np.diff(xx))
    yy = yy[:-1, :-1]  # + np.mean(np.diff(yy))
    # h = gaussian_filter(h, sigma=0.1)
    cvals = np.array([0, 1 - 0.95, 1 - 0.86, 1 - 0.5, 1 - 0.38, 1])
    colors = ['whitesmoke', 'silver', "darkgrey", "gray", "grey", "dimgrey"]
    norm = plt.Normalize(min(cvals), max(cvals))
    tuples = list(zip(map(norm, cvals), colors))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

    cm = ax.contourf(xx, yy, h.T, cmap=cmap, levels=[0, 1 - 0.95, 1 - 0.86, 1 - 0.65, 1 - 0.38, 1], norm=norm)
    ax.contour(xx, yy, h.T, colors='dimgrey', levels=[1 - 0.95, 1 - 0.86, 1 - 0.65, 1 - 0.38, 1], norm=norm)

    mass_gv = np.arange(7.6, 11.3, 0.1)
    gv_up = -0.4 + 0.25 * mass_gv
    gv_lo = gv_up - 0.2

    ax.plot(mass_gv, gv_up, ls='--', c='green')
    ax.plot(mass_gv, gv_lo, ls='--', c='green')
    ax.errorbar(mass_list[0], color_list[0], yerr=color_list[1],
                xerr=np.array([mass_list[1], mass_list[2]]).reshape(2, 1), c='red', marker='o')
    # a.plot(ma)
    ax.set_xlim(7.6, 11.05)
    ax.set_ylim(0.70, 2.8)
    ax.set_ylabel(r'$^{0.0}u-r$ color', fontsize=18)
    ax.set_xlabel(r'log($M_{\star}/M_{\odot}$)', fontsize=18)
    ax.tick_params(axis='both', labelsize=14)

    return fig


def plot_models(tde_name, tde_dir, z, bands, print_name=True, show=True):
    t, band_wls, sed_x_t, sed_err_x_t = fit_light_curve.gen_observables(tde_dir, z, bands, mode='fit')
    modelling_dir = os.path.join(tde_dir, 'modelling')
    color = ['magenta', 'darkviolet', 'navy', 'dodgerblue', 'cyan', 'green', 'red']
    label = [r'$UV~W2$', r'$UV~M2$', r'$UV~W1$', 'U', 'B', 'g', 'r']

    fig1, ax1 = plt.subplots(figsize=(12, 7))
    theta_median, p16, p84 = fit_light_curve.read_model1(modelling_dir)

    t_peak = theta_median[1]
    first_300days = (t - t_peak) <= 300
    if np.max(t[first_300days] - t_peak) < 300:
        t_model = theta_median[1] + np.arange(np.min(t - t_peak) - 10, np.max(t[first_300days] - t_peak), 1)
    else:
        t_model = theta_median[1] + np.arange(np.min(t - t_peak) - 10, 300, 1)

    model = models.const_T_gauss_rise_exp_decay(t_model, band_wls, theta_median)
    for i in range(np.shape(model)[1]):
        y = sed_x_t[:, i]
        if np.isfinite(y).any():
            y_err = sed_err_x_t[:, i]
            flag = np.isfinite(y)
            x = t - theta_median[1]

            ax1.errorbar(x[flag], y[flag], yerr=y_err[flag], marker="o", linestyle='', color=color[i], linewidth=1,
                         markeredgewidth=0.5, markeredgecolor='black', alpha=0.9, markersize=8, elinewidth=0.7,
                         capsize=0,
                         label=label[i])
    ax1.set_yscale('log')
    ax1.set_ylim(10**(np.log10(ax1.get_ylim()[0]) - 0.5), ax1.get_ylim()[1])
    for i in range(np.shape(model)[1]):
        y = sed_x_t[:, i]
        if np.isfinite(y).any():
            model_i = model[:, i]
            flag_before100 = t_model - theta_median[1] <= 100
            ax1.plot(t_model[flag_before100] - theta_median[1], model_i[flag_before100], color=color[i])
            flag_after100 = t_model - theta_median[1] > 100
            ax1.plot(t_model[flag_after100] - theta_median[1], model_i[flag_after100], color=color[i], ls='--')
    ax1.tick_params(axis='both', labelsize=18)
    ax1.set_yscale('log')
    ax1.set_xlabel('Days since peak', fontsize=20)
    ax1.set_ylabel(r'$\rm{\nu\,L_{\nu} \ [erg \ s^{-1}]}$', fontsize=20)
    plt.tight_layout()

    if np.max(t[first_300days] - t_peak) < 300:
        ax1.set_xlim(np.min(t - t_peak) - 5, np.max(t[first_300days] - t_peak) + 5)
    else:
        ax1.set_xlim(np.min(t - t_peak) - 5, 305)
    if print_name:
        ax1.text(0.2, 0.05, tde_name, horizontalalignment='left', verticalalignment='center', fontsize=16,
                 transform=ax1.transAxes)
    plt.legend(ncol=2)
    try:
        os.mkdir(os.path.join(tde_dir, 'plots', 'modelling'))
    except:
        pass
    plt.savefig(os.path.join(tde_dir, 'plots', 'modelling', 'model1_light_curves.pdf'), bbox_inches='tight')
    if show:
        plt.show()

    fig2, ax2 = plt.subplots(figsize=(12, 7))
    theta_median, p16, p84 = fit_light_curve.read_model2(modelling_dir)
    T_interval = int(np.diff(np.linspace(-60, 300, (len(theta_median) - 5)))[0])
    single_color = np.zeros(np.shape(t_model), dtype=bool)
    model = models.Blackbody_var_T_gauss_rise_powerlaw_decay(t_model, single_color, band_wls, T_interval, theta_median)
    for i in range(np.shape(model)[1]):
        y = sed_x_t[:, i]
        if np.isfinite(y).any():
            y_err = sed_err_x_t[:, i]
            flag = np.isfinite(y)
            x = t - t_peak
            ax2.errorbar(x[flag], y[flag], yerr=y_err[flag], marker="o", linestyle='', color=color[i], linewidth=1,
                         markeredgewidth=0.5, markeredgecolor='black', alpha=0.9, markersize=8, elinewidth=0.7,
                         capsize=0,
                         label=label[i])
            model_i = model[:, i]
            ax2.plot(t_model - t_peak, model_i, color=color[i])
    ax2.tick_params(axis='both', labelsize=18)
    ax2.set_yscale('log')
    ax2.set_xlabel('Days since peak', fontsize=20)
    ax2.set_ylabel(r'$\rm{\nu\,L_{\nu} \ [erg \ s^{-1}]}$', fontsize=20)
    plt.tight_layout()


    if np.max(t[first_300days] - t_peak) < 300:
        ax2.set_xlim(np.min(t - t_peak) - 5, np.max(t[first_300days] - t_peak) + 5)
    else:
        ax2.set_xlim(np.min(t - t_peak) - 5, 305)
    if print_name:
        ax2.text(0.2, 0.05, tde_name, horizontalalignment='left', verticalalignment='center', fontsize=16,
                 transform=ax2.transAxes)
    plt.legend(ncol=2)
    plt.savefig(os.path.join(tde_dir, 'plots', 'modelling', 'model2_light_curves.pdf'), bbox_inches='tight')
    if show:
        plt.show()


def plot_BB_evolution(tde_name, tde_dir, print_name=True, show=True):
    modelling_dir = os.path.join(tde_dir, 'modelling')
    t, log_BB, log_BB_err, log_R, log_R_err, log_T, log_T_err, single_band = fit_light_curve.read_BB_evolution(
        modelling_dir)
    single_band = np.array(single_band) == 1
    theta_median, p16, p84 = fit_light_curve.read_model2(modelling_dir)
    t_peak = theta_median[1]
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12), sharex=True, gridspec_kw={'hspace': 0.0})

    ax1.errorbar((t - t_peak)[~single_band], log_BB[~single_band], yerr=log_BB_err[~single_band], marker="o",
                 linestyle='-', color='b', linewidth=1,
                 markeredgewidth=1, markerfacecolor='blue', markeredgecolor='black', markersize=5, elinewidth=0.7,
                 capsize=2, fillstyle='full')
    ax1.errorbar((t - t_peak)[single_band], log_BB[single_band], yerr=log_BB_err[single_band], marker="o",
                 linestyle='', color='b', linewidth=1,
                 markeredgewidth=1, markerfacecolor='blue', markeredgecolor='black', markersize=5, elinewidth=0.7,
                 capsize=2, fillstyle='none')
    ax1.set_ylabel(r'log $\rm{L_{BB} \ [erg \ s^{-1}]}$', fontsize=18)
    ax1.tick_params(axis="x", direction="in", length=8, top=True)
    ax2.errorbar((t - t_peak)[~single_band], log_R[~single_band], yerr=log_R_err[~single_band], marker="o",
                 linestyle='-', color='b', linewidth=1,
                 markeredgewidth=1, markerfacecolor='blue', markeredgecolor='black', markersize=5, elinewidth=0.7,
                 capsize=2, fillstyle='full')
    ax2.errorbar((t - t_peak)[single_band], log_R[single_band], yerr=log_R_err[single_band], marker="o",
                 linestyle='', color='b', linewidth=1,
                 markeredgewidth=1, markerfacecolor='blue', markeredgecolor='black', markersize=5, elinewidth=0.7,
                 capsize=2, fillstyle='none')
    ax2.set_ylabel('log R [cm]', fontsize=18)
    ax2.tick_params(axis="x", direction="inout", length=8, top=True)

    ax3.errorbar((t - t_peak)[~single_band], log_T[~single_band], yerr=log_T_err[~single_band], marker="o",
                 linestyle='-', color='b', linewidth=1,
                 markeredgewidth=1, markerfacecolor='blue', markeredgecolor='black', markersize=5, elinewidth=0.7,
                 capsize=2)
    ax3.set_ylabel('log T [K]', fontsize=18)
    ax3.set_xlabel('Days since peak', fontsize=18)
    ax3.tick_params(axis="x", direction="inout", length=8, top=True)

    plt.tight_layout()
    if print_name:
        ax1.text(0.1, 0.1, tde_name, horizontalalignment='left', verticalalignment='center', fontsize=14,
                 transform=ax1.transAxes)
    plt.savefig(os.path.join(tde_dir, 'plots', 'modelling', 'Blackbody_evolution.pdf'), bbox_inches='tight')
    if show:
        plt.show()


def plot_SED(tde_name, tde_dir, z, bands, sampler, nwalkers, nburn, ninter, print_name=True, show=True):
    modelling_dir = os.path.join(tde_dir, 'modelling')
    t, band_wls, sed_x_t, sed_err_x_t = fit_light_curve.gen_observables(tde_dir, z, bands, mode='plot')
    t_BB, log_BB, log_BB_err, log_R, log_R_err, log_T, log_T_err, single_band = fit_light_curve.read_BB_evolution(
        modelling_dir)
    theta_median, p16, p84 = fit_light_curve.read_model2(modelling_dir)
    single_band = np.array(single_band) == 1

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
    label = [r'$UV~W2$', r'$UV~M2$', r'$UV~W1$', 'U', 'B', 'g', 'r']
    marker = ["o", "s", "p", "d", "*", "x", "+"]
    t_peak = theta_median[1]
    first_300days = (t - t_peak) <= 300
    if np.max(t[first_300days] - t_peak) < 300:
        if ~np.isfinite(theta_median[2]):
            t_model = theta_median[1] + np.arange(0, np.max(t[first_300days] - t_peak) + 5, 1)
        else:
            t_model = theta_median[1] + np.arange(np.min(t - t_peak) - 10, np.max(t[first_300days] - t_peak) + 5, 1)
    else:
        if ~np.isfinite(theta_median[2]):
            t_model = theta_median[1] + np.arange(0, 300, 1)
        else:
            t_model = theta_median[1] + np.arange(np.min(t - t_peak) - 10, 300, 1)

    for i in range(np.shape(sed_x_t)[1]):
        y = sed_x_t[first_300days, i][~single_band] * models.bolometric_correction(log_T[~single_band], band_wls[i])
        y_err = sed_err_x_t[first_300days, i][~single_band] / sed_x_t[first_300days, i][~single_band] * y
        flag = np.isfinite(y)
        x = (t[first_300days] - t_peak)[~single_band]
        ax1.errorbar(x[flag], y[flag], yerr=y_err[flag], marker=marker[i], ecolor='black', linestyle='', mfc='None',
                     mec='black', linewidth=1,
                     markeredgewidth=0.5, markersize=7, elinewidth=0.7, capsize=0)
    if ~np.isfinite(theta_median[2]):
        string = r'$p={:.1f}  \ t_0={:.1f}$'.format(theta_median[4], theta_median[3])
    else:
        string = r'$\sigma={:.1f}  \ p={:.1f}  \ t_0={:.1f}$'.format(theta_median[2], theta_median[4], theta_median[3])
    L_BB = models.L_bol(t_model, theta_median)
    ax1.plot(t_model - t_peak, L_BB, c='blue', alpha=1, label=string)
    randint = np.random.randint
    for i in range(100):
        theta = sampler.chain[randint(nwalkers), nburn + randint(ninter - nburn), :]
        if ~np.isfinite(theta_median[2]):
            theta = np.concatenate((np.array([theta[0], t[0], np.nan]), np.array(theta[1:])))
        L_BB = models.L_bol(t_model, theta)
        ax1.plot(t_model - t_peak, L_BB, c='blue', alpha=0.05)
    ax1.legend(fontsize='x-small', loc=1)
    ax1.set_yscale('log')
    ax1.set_ylabel('Blackbody Luminosity [erg s$^{-1}$]', fontsize=12)
    ax1.set_xlabel('Days since peak', fontsize=12)
    ax1.set_xticks(np.arange(-50, 301, 50))
    ax1.set_xticklabels(np.arange(-50, 301, 50), fontsize=12)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.set_ylim(10 ** (np.log10(L_BB[-1]) - 1), 10 ** (theta_median[0] + 0.5))

    #redimensioning sed_x_t
    sed_x_t = sed_x_t[first_300days, :]
    sed_err_x_t = sed_err_x_t[first_300days, :]


    # Picking good epoch of good SED near peak
    first_50days = (t_BB - t_peak) <= 50
    n_obs_peak_max = np.max(np.sum(np.isfinite(sed_x_t[first_50days, :]), axis=1))
    where_has_max = np.sum(np.isfinite(sed_x_t[first_50days, :]), axis=1) == n_obs_peak_max
    distance_to_peak = abs(t_BB[first_50days][where_has_max] - t_peak)
    flag_peak = distance_to_peak <= np.min(distance_to_peak) + 2

    T_near_peak = np.mean(log_T[first_50days][where_has_max][flag_peak])
    T_err_near_peak = np.mean(log_T_err[first_50days][where_has_max][flag_peak])
    L_BB_near_peak = 10 ** np.mean(log_BB[first_50days][where_has_max][flag_peak])
    L_BB_err_near_peak = 10 ** np.mean(0.432 * (log_BB_err / log_BB)[first_50days][where_has_max][flag_peak])
    t_near_peak = np.mean(t_BB[first_50days][where_has_max][flag_peak])

    for i in range(np.shape(sed_x_t)[1]):
        y = sed_x_t[first_50days, i][where_has_max][flag_peak]

        if np.isfinite(y).any():
            y_err = sed_err_x_t[first_50days, i][where_has_max][flag_peak]
            flag = np.isfinite(y)
            wl = band_wls[i] * u.Angstrom
            nu = np.zeros(np.shape(y[flag]))
            nu[:] = c.cgs / wl.cgs

            ax2.errorbar(nu, y[flag], yerr=y_err[flag], marker=marker[i], ecolor='black', linestyle='', mfc='None',
                         mec='black', linewidth=1,
                         markeredgewidth=0.7, markersize=8, elinewidth=0.7, capsize=0, label=label[i])
    ax2.legend(fontsize='xx-small')
    nu_list = (c.cgs / (np.arange(1300, 10000, 10) * u.Angstrom)).cgs
    A = L_BB_near_peak / ((sigma_sb.cgs * ((10 ** T_near_peak * u.K) ** 4)).cgs / np.pi).cgs.value
    bb_sed_mean = (A * models.blackbody(10 ** T_near_peak, (c.cgs / nu_list).to('AA').value))
    ax2.plot(nu_list.value, bb_sed_mean, c='blue')

    for i in range(100):
        A = np.random.normal(L_BB_near_peak, L_BB_err_near_peak) / ((sigma_sb.cgs * (
                (10 ** np.random.normal(T_near_peak, T_err_near_peak) * u.K) ** 4)).cgs / np.pi).cgs.value
        bb_sed = (A * models.blackbody(10 ** T_near_peak, (c.cgs / nu_list).to('AA').value))
        ax2.plot(nu_list.value, bb_sed, c='blue', alpha=0.05)

    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_ylabel(r'$\rm{\nu\,L_{\nu} \ [erg \ s^{-1}]}$', fontsize=14)
    ax2.set_xlabel('Rest-frame frequency (Hz)', fontsize=12)
    ax2.set_xticks([4e14, 6e14, 1e15, 2e15])
    ax2.set_xticklabels([r'$4\times10^{14}$', r'$6\times10^{14}$', r'$1\times10^{15}$', r'$2\times10^{15}$'],
                        fontsize=11)
    ax2.tick_params(axis='y', labelsize=12)
    ax2.set_xlim(nu_list[-1].value, 2.1e15)
    up_lim, lo_lim = np.max(bb_sed_mean.value), np.min(bb_sed_mean.value)
    ax2.set_ylim(10 ** (np.log10(lo_lim) - 0.5), 10 ** (np.log10(up_lim) + 0.7))
    title = r'$t={:.0f} \pm 2$ days pos max; log $T={:.2f} \pm {:.2f} \ K$'.format(int(t_near_peak - t_peak), T_near_peak, T_err_near_peak)
    ax2.set_title(title, fontsize=12)
    ax2.legend(fontsize='xx-small', loc=4)


    # Picking good epoch of good SED near 200 days
    last_days = (t_BB - t_peak) > 50
    if np.sum(last_days) == 0:
        last_days = (t_BB - t_peak) <= 50
    n_obs_200_max = np.max(np.sum(np.isfinite(sed_x_t[last_days, :]), axis=1))
    where_has_max = np.where(np.sum(np.isfinite(sed_x_t[last_days, :]), axis=1) == n_obs_200_max)
    distance_to_200 = abs(t_BB[last_days][where_has_max] - (t_peak + 150))
    flag_200 = distance_to_200 <= np.min(distance_to_200) + 2

    T_near_200 = np.mean(log_T[last_days][where_has_max][flag_200])
    T_err_near_200 = np.mean(log_T_err[last_days][where_has_max][flag_200])
    L_BB_near_200 = 10 ** np.mean(log_BB[last_days][where_has_max][flag_200])
    L_BB_err_near_200 = 10 ** np.mean(0.432 * (log_BB_err / log_BB)[last_days][where_has_max][flag_200])
    t_near_200 = np.mean(t_BB[last_days][where_has_max][flag_200])

    for i in range(np.shape(sed_x_t)[1]):
        y_200 = sed_x_t[last_days, i][where_has_max][flag_200]
        if np.isfinite(y_200).any():
            y = y_200
            y_err = sed_err_x_t[last_days, i][where_has_max][flag_200]
            flag = np.isfinite(y)
            wl = band_wls[i] * u.Angstrom
            nu = np.zeros(np.shape(y[flag]))
            nu[:] = c.cgs / wl.cgs

            ax3.errorbar(nu, y[flag], yerr=y_err[flag], marker=marker[i], ecolor='black', linestyle='', mfc='None',
                         mec='black', linewidth=1,
                         markeredgewidth=0.7, markersize=8, elinewidth=0.7, capsize=0, label=label[i])

    ax3.legend(fontsize='xx-small')
    nu_list = (c.cgs / (np.arange(1300, 10000, 10) * u.Angstrom)).cgs
    A = L_BB_near_200 / ((sigma_sb.cgs * ((10 ** T_near_200 * u.K) ** 4)).cgs / np.pi).cgs.value
    bb_sed_mean = (A * models.blackbody(10 ** T_near_200, (c.cgs / nu_list).to('AA').value))
    ax3.plot(nu_list.value, bb_sed_mean, c='blue')
    for i in range(100):
        A = np.random.normal(L_BB_near_200, L_BB_err_near_200) / ((sigma_sb.cgs * (
                (10 ** np.random.normal(T_near_200, T_err_near_200) * u.K) ** 4)).cgs / np.pi).cgs.value
        bb_sed = (A * models.blackbody(10 ** T_near_200, (c.cgs / nu_list).to('AA').value))
        ax3.plot(nu_list.value, bb_sed, c='blue', alpha=0.05)
    ax3.set_yscale('log')
    ax3.set_xscale('log')
    ax3.set_ylabel(r'$\rm{\nu\,L_{\nu} \ [erg \ s^{-1}]}$', fontsize=14)
    ax3.set_xlabel('Rest-frame frequency (Hz)', fontsize=12)
    ax3.set_xticks([4e14, 6e14, 1e15, 2e15])
    ax3.set_xticklabels([r'$4\times10^{14}$', r'$6\times10^{14}$', r'$1\times10^{15}$', r'$2\times10^{15}$'],
                        fontsize=11)
    ax3.tick_params(axis='y', labelsize=12)
    ax3.set_xlim(nu_list[-1].value, 2.1e15)
    up_lim, lo_lim = np.max(bb_sed_mean.value), np.min(bb_sed_mean.value)
    ax3.set_ylim(10 ** (np.log10(lo_lim) - 0.5), 10 ** (np.log10(up_lim) + 0.7))
    title = r'$t={:.0f} \pm 2$ days pos max; log $T={:.2f} \pm {:.2f} \ K$'.format(int(t_near_200 - t_peak), T_near_200,
                                                                                   T_err_near_200)
    ax3.set_title(title, fontsize=12)
    up_lim = np.max([ax2.get_ylim(), ax3.get_ylim()])
    lo_lim = np.min([ax2.get_ylim(), ax3.get_ylim()])
    ax2.set_ylim(lo_lim, up_lim)
    ax3.set_ylim(lo_lim, up_lim)
    if np.max(t[first_300days] - t_peak) < 300:
        ax1.set_xlim(np.min(t - t_peak) - 10, np.max(t[first_300days] - t_peak) + 10)
    else:
        ax1.set_xlim(np.min(t - t_peak) - 10, 305)

    plt.tight_layout()
    if print_name:
        ax1.text(0.2, 0.05, tde_name, horizontalalignment='left', verticalalignment='center', fontsize=14,
                 transform=ax1.transAxes)
    plt.savefig(os.path.join(tde_dir, 'plots', 'modelling', 'SED_evolution.pdf'), bbox_inches='tight')
    if show:
        plt.show()

def plot_lc_corner(tde_dir, fig_name, theta_median, sample, labels, show=True):
    data = np.zeros(np.shape(sample))
    for i, x in enumerate(sample):
        data[i, :] = x
    bounds = []
    for i in range(np.shape(sample)[1]):
        sig1 = np.nanpercentile((data[:, i]), 50) - np.nanpercentile((data[:, i]), 16)
        sig2 = np.nanpercentile((data[:, i]), 84) - np.nanpercentile((data[:, i]), 50)
        mean_dist = np.nanmean([sig1, sig2])
        bounds.append((theta_median[i] - 4 * mean_dist, theta_median[i] + 4 * mean_dist))

    corner.corner(sample,
                  labels=labels,  #
                  quantiles=[0.16, 0.5, 0.84],
                  show_titles=True, title_kwargs={"fontsize": 12}, range=bounds)
    try:
        os.mkdir(os.path.join(tde_dir, 'plots', 'modelling'))
    except:
        pass
    plt.savefig(os.path.join(tde_dir, 'plots', 'modelling', fig_name), bbox_inches='tight')
    if show:
        plt.show()
