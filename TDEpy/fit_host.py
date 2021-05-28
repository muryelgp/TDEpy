import os
import sedpy
from prospect.io import write_results as writer
import matplotlib.pyplot as plt
import numpy as np
from prospect.fitting import fit_model
from prospect.fitting import lnprobfn
import prospect.io.read_results as reader
from prospect.sources import CSPSpecBasis
from prospect.models.templates import TemplateLibrary, describe
import pkg_resources
from . import tools as tools


# re-defining plotting defaults

plt.rcParams.update({'font.size': 16})




def build_obs(path, tde):
    """Build a dictionary of observational data.

    :returns obs:
        A dictionary of observational data to use in the fit.
    """
    from prospect.utils.obsutils import fix_obs
    import sedpy

    # The obs dictionary, empty for now
    obs = {}

    tde_dir = os.path.join(path, tde)
    try:
        band, wl_c, ab_mag, ab_mag_err, catalogs, apertures = np.loadtxt(os.path.join(tde_dir, 'host', 'host_phot_obs.txt'),
                                                              dtype={'names': (
                                                                  'band', 'wl_0', 'ab_mag', 'ab_mag_err', 'catalog', 'apertures'),
                                                                  'formats': (
                                                                      'U5', np.float, np.float, np.float, 'U10', 'U10')},
                                                              unpack=True, skiprows=2)
    except:
        raise Exception('We should run download_host_data() before trying to fit it.')

    filter_dic = {'WISE_W4': 'wise_w4', 'WISE_W3': 'wise_w3', 'WISE_W2': 'wise_w2', 'WISE_W1': 'wise_w1',
                  'UKIDSS_Y': 'UKIRT_Y', 'UKIDSS_J': 'UKIRT_J', 'UKIDSS_H': 'UKIRT_H', 'UKIDSS_K': 'UKIRT_K',
                  '2MASS_J': 'twomass_J', '2MASS_H': 'twomass_H', '2MASS_Ks': 'twomass_Ks',
                  'PAN-STARRS_y': 'PAN-STARRS_y', 'PAN-STARRS_z': 'PAN-STARRS_z', 'PAN-STARRS_i': 'PAN-STARRS_i',
                  'PAN-STARRS_r': 'PAN-STARRS_r', 'PAN-STARRS_g': 'PAN-STARRS_g',
                  'DES_Y': 'decam_Y', 'DES_z': 'decam_z', 'DES_i': 'decam_i', 'DES_r': 'decam_r', 'DES_g': 'decam_g',
                  'SkyMapper_u': 'SkyMapper_u', 'SkyMapper_z': 'SkyMapper_z', 'SkyMapper_i': 'SkyMapper_i', 'SkyMapper_r': 'SkyMapper_r',
                  'SkyMapper_g': 'SkyMapper_g', 'SkyMapper_v': 'SkyMapper_v',
                  'SDSS_u': 'sdss_u0', 'SDSS_z': 'sdss_z0', 'SDSS_g': 'sdss_g0', 'SDSS_r': 'sdss_r0', 'SDSS_i': 'sdss_i0',
                  'GALEX_NUV': 'galex_NUV', 'GALEX_FUV': 'galex_FUV',
                  'Swift/UVOT_UVW1': 'uvot_w1', 'Swift/UVOT_UVW2': 'uvot_w2', 'Swift/UVOT_UVM2': 'uvot_m2',
                  'Swift/UVOT_V': 'uvot_V', 'Swift/UVOT_B': 'uvot_B', 'Swift/UVOT_U': 'uvot_U'}

    flag = np.isfinite(ab_mag * ab_mag_err)

    catalog_bands = [catalogs[i] + '_' + band[i] for i in range(len(catalogs))]
    filternames = [filter_dic[i] for i in catalog_bands]
    # And here we instantiate the `Filter()` objects using methods in `sedpy`,
    # and put the resultinf list of Filter objects in the "filters" key of the `obs` dictionary
    obs["filters"] = np.ndarray((0))
    for i in range(len(filternames)):
        # print(filternames[i])
        obs["filters"] = np.append(obs["filters"], sedpy.observate.Filter(filternames[i], directory=pkg_resources.resource_filename("TDEpy", 'filters')))



    obs["phot_wave"] = np.zeros(flag.shape)
    obs["phot_mask"] = np.zeros(flag.shape)
    obs["maggies"] = np.zeros(flag.shape)
    obs["maggies_unc"] = np.zeros(flag.shape)
    obs["phot_mask"] = np.ones(np.shape(flag), dtype=bool)


    # Measurments
    for i in range(len(flag)):
        if flag[i]:
            mags = np.array(ab_mag[i])
            mags_err = np.array(ab_mag_err[i])
            signal = tools.mag_to_flux(mags, wl_c[i])
            noise = tools.dmag_to_df(mags_err, signal)
            snr = signal/noise
            obs["maggies"][i] = 10 ** (-0.4 * mags)
            obs["maggies_unc"][i] = obs["maggies"][i] * (1 / snr)
            obs["phot_wave"][i] = np.array(wl_c[i])
        else:
            obs["maggies"][i] = 0
            obs["maggies_unc"][i] = 10 ** (-0.4 * ab_mag[i])
            obs["phot_wave"][i] = np.array(wl_c[i])

    obs["wavelength"] = None
    obs["spectrum"] = None
    obs['unc'] = None
    obs['mask'] = None

    # This function ensures all required keys are present in the obs dictionary,
    # adding default values if necessary
    obs = fix_obs(obs)

    return obs


def build_model(object_redshift=None, init_theta=None, add_duste=True):
    """Build a prospect.models.SedModel object

    :param object_redshift: (optional, default: None)
        If given, produce spectra and observed frame photometry appropriate
        for this redshift. Otherwise, the redshift will be zero.

    :param  init_theta: (optional, default: [1e10, 0, 0.05, 1, 1])
        The initial guess on the parameters for mcmc.

    :returns model:
        An instance of prospect.models.SedModel
    """
    from prospect.models.sedmodel import SedModel
    from prospect.models.templates import TemplateLibrary
    from prospect.models import priors

    # Get (a copy of) one of the prepackaged model set dictionaries.
    model_params = TemplateLibrary["parametric_sfh"]

    model_params["sfh"]["init"] = 4
    #print(init_theta)
    # Changing the initial values appropriate for our objects and data
    model_params["mass"]["init"] = init_theta[0]
    model_params["logzsol"]["init"] = init_theta[1]
    model_params["dust2"]["init"] = init_theta[2]
    model_params["tage"]["init"] = init_theta[3]
    model_params["tau"]["init"] = init_theta[4]




    # Setting the priors forms and limits
    model_params["mass"]["prior"] = priors.LogUniform(mini=1e6, maxi=1e12)
    model_params["logzsol"]["prior"] = priors.Uniform(mini=-1.8, maxi=-0.1)
    model_params["dust2"]["prior"] = priors.ClippedNormal(mean=0.05, sigma=0.1, mini=0.0, maxi=0.5)
    #model_params["dust2"]["prior"] = priors.Uniform(mini=0.0, maxi=0.5)
    model_params["tage"]["prior"] = priors.Uniform(mini=2, maxi=14)
    model_params["tau"]["prior"] = priors.Uniform(mini=0.001, maxi=30)

    # Setting the spread of the walkers for the mcmc sampling
    model_params["mass"]["disp_floor"] = 1e8
    model_params["logzsol"]['disp_floor'] = 0.5
    model_params["dust2"]["disp_floor"] = 0.1
    model_params["tage"]["disp_floor"] = 5.0
    model_params["tau"]["disp_floor"] = 3.0

    model_params["mass"]["init_disp"] = 1e8
    model_params["logzsol"]['init_disp'] = 0.5
    model_params["dust2"]["init_disp"] = 0.1
    model_params["tage"]["init_disp"] = 5.0
    model_params["tau"]["init_disp"] = 3.0


    # Fixing and defining the object redshift
    model_params["zred"]['isfree'] = False
    model_params["zred"]['init'] = object_redshift

    #ldist = cosmo.luminosity_distance(object_redshift).value
    #model_params["lumdist"] = {"N": 1, "isfree": False, "init": ldist, "units": "Mpc"}

    # Now instantiate the model object using this dictionary of parameter specifications
    model = SedModel(model_params)

    return model


def build_sps(zcontinuous=1):
    sps = CSPSpecBasis(zcontinuous=zcontinuous)
    return sps


def plot_resulting_fit(tde_name, path):
    tde_dir = os.path.join(path, tde_name)
    host_dir = os.path.join(tde_dir, 'host')

    fig, ax = plt.subplots(figsize=(16, 8))

    band, obs_wl_c, obs_ab_mag, obs_ab_mag_err, catalogs, apertures = \
        np.loadtxt(os.path.join(host_dir, 'host_phot_obs.txt'),
                   dtype={'names': (
                       'band', 'wl_0', 'ab_mag', 'ab_mag_err',
                       'catalog', 'aperture'),
                       'formats': (
                           'U5', np.float, np.float, np.float,
                           'U10', 'U10')},
                   unpack=True, skiprows=2)


    band, model_wl_c, model_ab_mag, model_ab_mag_err, model_flux, model_flux_err, catalogs = \
        np.loadtxt(os.path.join(host_dir, 'host_phot_model.txt'),
                   dtype={'names': (
                       'band', 'wl_0', 'ab_mag', 'ab_mag_err',
                       'flux_dens', 'flux_dens_err', 'catalog'),
                       'formats': (
                           'U5', np.float, np.float, np.float,
                           np.float, np.float, 'U10')},
                   unpack=True, skiprows=1)

    n_bands = int(np.where(band == 'V')[0])
    band_flag = [i < n_bands for i in range(len(model_wl_c))]

    spec_wl_0, spec_ab_mag, spec_ab_mag_p16, spec_ab_mag_p84,  spec_flux, spec_flux_p16, spec_flux_p84 = np.loadtxt(
        os.path.join(host_dir, 'host_spec_model.txt'),
        dtype={'names': (
            'wl_0', 'ab_mag', 'spec_ab_mag_p16', 'spec_ab_mag_p16'
            'flux_dens', 'spec_flux_p16', 'spec_ab_mag_p84', 'tde/host'),
            'formats': (
                np.float, np.float, np.float,
                np.float, np.float, np.float, np.float, np.float)},
        unpack=True, skiprows=1)


    ax.plot(spec_wl_0, spec_ab_mag, label='Model spectrum (MAP)',
            lw=0.7, color='green', alpha=0.8)
    ax.fill_between(spec_wl_0, spec_ab_mag_p16, spec_ab_mag_p84, alpha=.3, color='green', label='Posterior')
    ax.errorbar(model_wl_c[band_flag], model_ab_mag[band_flag], yerr=model_ab_mag_err[band_flag],
                label='Model photometry (MAP)',
                marker='s', markersize=8, alpha=0.85, ls='', lw=3, ecolor='green', capsize=5,
                markerfacecolor='none', markeredgecolor='green',
                markeredgewidth=3)

    is_up_lim = ~np.isfinite(obs_ab_mag_err)
    ax.errorbar(obs_wl_c[~is_up_lim], obs_ab_mag[~is_up_lim], yerr=obs_ab_mag_err[~is_up_lim],
                label='Observed photometry', ecolor='red',
                marker='o', markersize=8, ls='', lw=3, alpha=0.85, capsize=5,
                markerfacecolor='none', markeredgecolor='red',
                markeredgewidth=3)
    ax.invert_yaxis()
    ax.errorbar(obs_wl_c[is_up_lim], obs_ab_mag[is_up_lim], yerr=0.5, lolims=np.ones(np.shape(obs_ab_mag_err[is_up_lim]), dtype=bool),
               marker='o', fmt='o',  ecolor='red', alpha=0.85, lw=3, markeredgecolor='red',
               markerfacecolor='none', markersize=8, elinewidth=2, capsize=6, capthick=3,
               markeredgewidth=3)


    temp = np.interp(np.linspace(700, 300000, 10000), spec_wl_0, spec_ab_mag)
    ymin, ymax = temp.min() * 0.85, temp.max() * 1.1

    plt.xscale('log')
    ax.set_xlim(700, 300000)
    ax.set_xticks([1e3, 1e4, 1e5])
    ax.set_xticklabels(['0.1', '1', '10'])
    ax.set_ylim(ymax, ymin)
    ax.set_ylabel('AB mag', fontsize=14)
    ax.set_xlabel(r'Wavelength $[\mu m]$', fontsize=14)
    ax.set_title('Host Galaxy SED Fit (' + tde_name + ')')
    plt.legend(loc=4)
    return fig


def corner_plot(result):
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
    logify = ["mass", "tau"]
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
        data[i, :] = a, b, c, d, e

    for i in range(np.shape(xx)[1]):
        sig1 = theta_max[i] - np.percentile((data[:, i]), 16)
        sig2 = np.percentile((data[:, i]), 84) - theta_max[i]
        mean_dist = np.mean([sig1, sig2])
        if i == 0 or i == 4:
            bounds.append((np.log10(theta_max[i]) - 4 * mean_dist, np.log10(theta_max[i]) + 4 * mean_dist))
        else:
            bounds.append((theta_max[i] - 4 * mean_dist, theta_max[i] + 4 * mean_dist))


    cornerfig = reader.subcorner(result, thin=5,
                                 fig=plt.subplots(5, 5, figsize=(27, 27))[0], logify=["mass", "tau"], range=bounds)

    return cornerfig


def gen_post_filters(result, model, obs, sps, nwalkers, n_burn, niter, i):
    randint = np.random.randint
    # selecting a randon walker at a any step
    theta = result['chain'][randint(nwalkers), n_burn[-1] + randint(niter)]
    # getting phot and spec ate this position
    mspec, mphot, mextra = model.mean_model(theta, obs, sps=sps)
    return mphot, mspec


def save_results(result, model, obs, sps, theta_max, tde_name, path, n_walkers, n_inter, n_burn, n_cores):
    tde_dir = os.path.join(path, tde_name)
    band, _, _, _, catalogs, _ = np.loadtxt(os.path.join(tde_dir, 'host', 'host_phot_obs.txt'),
                                         dtype={'names': (
                                             'band', 'wl_0', 'ab_mag', 'ab_mag_err', 'catalog', 'apertures'),
                                             'formats': (
                                                 'U5', np.float, np.float, np.float, 'U10', 'U10')},
                                         unpack=True, skiprows=2)

    # Adding new bands (Swift, SDSS, HST)
    wphot = obs["phot_wave"]

    obs["filters"] = np.append(obs["filters"], sedpy.observate.Filter('uvot_V', directory=pkg_resources.resource_filename("TDEpy", 'filters')))
    wphot = np.append(wphot, 5468)
    band = np.append(band, 'V')
    catalogs = np.append(catalogs, 'Swift/UVOT')

    obs["filters"] = np.append(obs["filters"], sedpy.observate.Filter('uvot_B', directory=pkg_resources.resource_filename("TDEpy", 'filters')))
    wphot = np.append(wphot, 4392)
    band = np.append(band, 'B')
    catalogs = np.append(catalogs, 'Swift/UVOT')

    obs["filters"] = np.append(obs["filters"], sedpy.observate.Filter('uvot_U', directory=pkg_resources.resource_filename("TDEpy", 'filters')))
    wphot = np.append(wphot, 3465)
    band = np.append(band, 'U')
    catalogs = np.append(catalogs, 'Swift/UVOT')

    obs["filters"] = np.append(obs["filters"], sedpy.observate.Filter('uvot_w1'))
    wphot = np.append(wphot, 2684)
    band = np.append(band, 'UVW1')
    catalogs = np.append(catalogs, 'Swift/UVOT')

    obs["filters"] = np.append(obs["filters"], sedpy.observate.Filter('uvot_m2'))
    wphot = np.append(wphot, 2245)
    band = np.append(band, 'UVM2')
    catalogs = np.append(catalogs, 'Swift/UVOT')

    obs["filters"] = np.append(obs["filters"], sedpy.observate.Filter('uvot_w2'))
    wphot = np.append(wphot, 2085)
    band = np.append(band, 'UVW2')
    catalogs = np.append(catalogs, 'Swift/UVOT')

    obs["filters"] = np.append(obs["filters"], sedpy.observate.Filter('sdss_u0'))
    wphot = np.append(wphot, 3551)
    band = np.append(band, 'u')
    catalogs = np.append(catalogs, 'SDSS')

    obs["filters"] = np.append(obs["filters"], sedpy.observate.Filter('sdss_g0'))
    wphot = np.append(wphot, 4686)
    band = np.append(band, 'g')
    catalogs = np.append(catalogs, 'SDSS')

    obs["filters"] = np.append(obs["filters"], sedpy.observate.Filter('sdss_r0'))
    wphot = np.append(wphot, 6166)
    band = np.append(band, 'r')
    catalogs = np.append(catalogs, 'SDSS')

    obs["filters"] = np.append(obs["filters"], sedpy.observate.Filter('sdss_i0'))
    wphot = np.append(wphot, 7480)
    band = np.append(band, 'i')
    catalogs = np.append(catalogs, 'SDSS')

    obs["filters"] = np.append(obs["filters"], sedpy.observate.Filter('sdss_z0'))
    wphot = np.append(wphot, 8932)
    band = np.append(band, 'z')
    catalogs = np.append(catalogs, 'SDSS')

    obs["filters"] = np.append(obs["filters"], sedpy.observate.Filter('wfc3_uvis_f275w'))
    wphot = np.append(wphot, 2750)
    band = np.append(band, 'F275W')
    catalogs = np.append(catalogs, 'HST/WFC3')

    obs["filters"] = np.append(obs["filters"], sedpy.observate.Filter('wfc3_uvis_f336w'))
    wphot = np.append(wphot, 3375)
    band = np.append(band, 'F336W')
    catalogs = np.append(catalogs, 'HST/WFC3')

    obs["filters"] = np.append(obs["filters"], sedpy.observate.Filter('wfc3_uvis_f475w'))
    wphot = np.append(wphot, 4550)
    band = np.append(band, 'F475W')
    catalogs = np.append(catalogs, 'HST/WFC3')

    obs["filters"] = np.append(obs["filters"], sedpy.observate.Filter('wfc3_uvis_f555w'))
    wphot = np.append(wphot, 5410)
    band = np.append(band, 'F555W')
    catalogs = np.append(catalogs, 'HST/WFC3')

    obs["filters"] = np.append(obs["filters"], sedpy.observate.Filter('wfc3_uvis_f606w'))
    wphot = np.append(wphot, 5956)
    band = np.append(band, 'F606W')
    catalogs = np.append(catalogs, 'HST/WFC3')

    obs["filters"] = np.append(obs["filters"], sedpy.observate.Filter('wfc3_uvis_f814w'))
    wphot = np.append(wphot, 8353)
    band = np.append(band, 'F814W')
    catalogs = np.append(catalogs, 'HST/WFC3')

    # Creating modelled photometry and spectra
    mspec_map, mphot_map, _ = model.mean_model(theta_max, obs, sps=sps)

    # Correcting for redshift
    a = 1.0 + model.params.get('zred', 0.0)  # cosmological redshifting
    wspec = sps.wavelengths
    wspec *= a

    # Measuring errors on the modelled spectra and photometry
    randint = np.random.randint
    nwalkers, niter = n_walkers, n_inter - n_burn[-1]
    n_visits = int(0.2 * (n_inter - n_burn[-1])*n_walkers)
    err_phot = np.zeros((n_visits, len(mphot_map)))
    err_spec = np.zeros((n_visits, len(mspec_map)))


    for i in range(n_visits):
        # selecting a randon walker at a any step
        theta = result['chain'][randint(nwalkers), n_burn[-1] + randint(niter)]
        # getting phot and spec ate this position
        mspec, mphot, mextra = model.mean_model(theta, obs, sps=sps)
        err_phot[i, :] = mphot
        err_spec[i, :] = mspec

    '''
    # Tried to implement multiprocessing, it is taking more time than the for loop (?), get back to it later!
    from multiprocessing import Pool
    pool = Pool(int(n_cores))
    func = partial(gen_post_filters, result, model, obs, sps, nwalkers, n_burn, niter)
    starttime = time.time()
    chunksize, extra = divmod(int(1e4), n_cores * 16)
    if extra:
        chunksize += 1
    result = pool.imap(func, range(0, int(1e4)), chunksize=chunksize)
    for i, res in enumerate(result):
        #print(i)
        err_phot[i, :], err_spec[i, :] = res
    endtime = time.time()
    print(f"Time taken {endtime - starttime} seconds")
    '''

    # Saving modelled photometry
    err_phot_mag = np.log10(err_phot) / -0.4
    err_phot_flux = tools.mag_to_flux(err_phot_mag, wphot)

    mphot_map_mag = np.log10(mphot_map) / -0.4
    mphot_map_flux = tools.mag_to_flux(mphot_map_mag, wphot)

    err_phot_flux_std = np.percentile(abs(err_phot_flux - mphot_map_flux), 68, axis=0)
    err_phot_mag_std = tools.df_to_dmag(mphot_map_flux, err_phot_flux_std, wphot)

    small_err = np.round(err_phot_mag_std, 2) < 0.01
    err_phot_mag_std[small_err] = 0.01

    host_dir = os.path.join(tde_dir, 'host')
    host_file = open(os.path.join(host_dir, 'host_phot_model.txt'), 'w')
    host_file.write(
        'band' + '\t' + 'wl_0' + '\t' + 'ab_mag' + '\t' + 'ab_mag_err' + '\t' + 'flux_dens' + '\t' + 'flux_dens_err' + '\t' + 'catalog/instrument' + '\n')
    for yy in range(len(wphot)):
        host_file.write(
            str(band[yy]) + '\t' + str(int(wphot[yy])) + '\t' + '{:.2f}'.format(mphot_map_mag[yy]) + '\t' + '{:.2f}'.format(
                err_phot_mag_std[yy]) + '\t' + '{:.2e}'.format(mphot_map_flux[yy]) + '\t' + '{:.2e}'.format(
                err_phot_flux_std[yy]) + '\t' + str(catalogs[yy]) + '\n')
    host_file.close()

    # saving memory
    err_phot_mag = err_phot_flux = mphot_map_mag = mphot_map_flux = err_phot_flux_std = err_phot_mag_std = None

    # Saving modelled spectrum
    spec_mag = -2.5 * np.log10(mspec_map)
    spec_flux = tools.mag_to_flux(spec_mag, wspec)



    # Dealing with posterior percentis
    flux_p16 = tools.mag_to_flux(-2.5 * np.log10(np.percentile(err_spec, 16, axis=0)), wspec)
    flux_p84 = tools.mag_to_flux(-2.5 * np.log10(np.percentile(err_spec, 84, axis=0)), wspec)
    mag_p16 = tools.flux_to_mag(flux_p16, wspec)
    mag_p84 = tools.flux_to_mag(flux_p84, wspec)


    host_file = open(os.path.join(host_dir, 'host_spec_model.txt'), 'w')
    host_file.write('wl_0' + '\t' + 'ab_mag' + '\t' + 'ab_mag_p16' + '\t' + 'ab_mag_p84' + '\t' + 'flux_dens' + '\t' + 'flux_dens_p16' + '\t' + 'flux_dens_p84' + '\n')
    for yy in range(len(wspec)):
        host_file.write('{:.2f}'.format(wspec[yy]) + '\t' + '{:.2f}'.format(spec_mag[yy]) + '\t' +
                        '{:.2f}'.format(mag_p16[yy]) + '\t' + '{:.2f}'.format(mag_p84[yy]) + '\t' + '{:.2e}'.format(spec_flux[yy]) +
                        '\t' + '{:.2e}'.format(flux_p16[yy]) + '\t' + '{:.2e}'.format(flux_p84[yy]) + '\n')
    host_file.close()


def host_sub_lc(tde_name, path):
    tde_dir = os.path.join(path, tde_name)
    host_dir = os.path.join(tde_dir, 'host')
    band_dic = dict(sw_uu='U', sw_bb='B', sw_vv='V', sw_w1='UVW1', sw_m2='UVM2',
                    sw_w2='UVW2')
    band_wl_dic = dict(sw_uu=3465, sw_bb=4392, sw_vv=5468, sw_w1=2600, sw_m2=2246,
                    sw_w2=1928)

    bands = ['sw_uu', 'sw_bb', 'sw_vv', 'sw_w1', 'sw_m2', 'sw_w2']


    host_bands, model_wl_c, model_ab_mag, model_ab_mag_err, model_flux, model_flux_err, catalogs = \
        np.loadtxt(os.path.join(host_dir, 'host_phot_model.txt'),
                   dtype={'names': (
                       'band', 'wl_0', 'ab_mag', 'ab_mag_err',
                       'flux_dens', 'flux_dens_err', 'catalog'),
                       'formats': (
                           'U5', np.float, np.float, np.float,
                           np.float, np.float, 'U10')},
                   unpack=True, skiprows=1)

    for band in bands:
        # Loading and plotting Swift data
        data_path = os.path.join(tde_dir, 'photometry', 'obs', str(band) + '.txt')
        if os.path.exists(data_path):
            obsid, mjd, abmag, abmage, flu, flue = np.loadtxt(data_path, skiprows=2, unpack=True)

            # selecting model fluxes at the respective band
            host_band = host_bands == band_dic[band]
            band_wl = model_wl_c[host_band][0]
            host_flux = model_flux[host_band][0]
            host_flux_err = model_flux_err[host_band][0]

            # Subtracting host contribution from the light curve
            host_sub_flu = flu - host_flux
            host_sub_flue = np.sqrt(flue**2 + host_flux_err**2)

            #dealing with negative fluxes
            host_sub_abmag = np.zeros(np.shape(host_sub_flu))
            host_sub_abmage = np.zeros(np.shape(host_sub_flu))
            sig_host = np.zeros(np.shape(host_sub_flu))

            is_pos_flux = host_sub_flu > 0
            host_sub_abmag[is_pos_flux] = tools.flux_to_mag(host_sub_flu[is_pos_flux], band_wl)
            host_sub_abmag[~is_pos_flux] = -99
            host_sub_abmage[is_pos_flux] = tools.df_to_dmag(host_sub_flu[is_pos_flux], host_sub_flue[is_pos_flux], band_wl)
            host_sub_abmage[~is_pos_flux] = -99
            host_sub_flu[~is_pos_flux] = 0


            sig_host[is_pos_flux] = host_sub_flu[is_pos_flux] / host_flux
            sig_host[~is_pos_flux] = 0.00

            write_path = os.path.join(tde_dir, 'photometry', 'host_sub', str(band) + '.txt')
            try:
                os.mkdir(os.path.join(tde_dir, 'photometry', 'host_sub'))
            except:
                pass


            g = open(write_path, 'w')
            g.write('#Values corrected for Galactic extinction and free from host contribution\n')
            g.write('obsid' + '\t' + 'mjd' + '\t' + 'ab_mag' + '\t' + 'ab_mag_err' + '\t' + 'flux_dens' + '\t' + 'flux_dens_err' + '\t' + 'TDE/host' + '\n')
            for yy in range(len(mjd)):
                if mjd[yy] > 0:
                    obsid_yy = str('000' + str(int(obsid[yy])))
                    g.write(obsid_yy + '\t' + '{:.2f}'.format(mjd[yy]) + '\t' + '{:.2f}'.format(host_sub_abmag[yy]) + '\t' +
                            '{:.2f}'.format(host_sub_abmage[yy]) + '\t' + '{:.2e}'.format(host_sub_flu[yy]) + '\t' +
                            '{:.2e}'.format(host_sub_flue[yy]) + '\t' + '{:.2f}'.format(sig_host[yy]) + '\n')
            g.close()


def configure(tde_name, path, z, init_theta, n_walkers, n_inter, n_burn):
    # Setting same paraters for the mcmc sampling
    run_params = {"object_redshift": z, "fixed_metallicity": False, "add_duste": True, "verbose": True,
                  "optimize": True, "emcee": True, "dynesty": False, "nwalkers": n_walkers, "niter": n_inter, "nburn": n_burn,
                  "min_method": "lm", "nmin": 2}

    # Instantiating observation object and sps
    obs = build_obs(path, tde_name)
    sps = build_sps()

    # Instantiating model object
    model = build_model(object_redshift=z, init_theta=init_theta)

    return obs, sps, model, run_params


def run_prospector(tde_name, path, z, withmpi, n_cores, show_figs=True, init_theta=None, n_walkers=None, n_inter=None, n_burn=None, read_only=False):
    os.chdir(os.path.join(path, tde_name, 'host'))

    if init_theta is None:
        init_theta = [1e10, -1.5, 0.01, 13, 1]

    if n_walkers is None:
        n_walkers = 100

    if n_inter is None:
        n_inter = 1000

    if n_burn is None:
        n_burn = [500]

    obs, sps, model, run_params = configure(tde_name, path, z, init_theta, n_walkers, n_inter, n_burn)

    model_params = TemplateLibrary["parametric_sfh"]
    if not withmpi:
        n_cores = 1

    if not read_only:
        print("Initial guess: {}".format(model.initial_theta))
        print('Sampling the the SPS grid..')
        if withmpi & ('logzsol' in model.free_params):
            dummy_obs = dict(filters=None, wavelength=None)

            logzsol_prior = model.config_dict["logzsol"]['prior']
            lo, hi = logzsol_prior.range
            logzsol_grid = np.around(np.arange(lo, hi, step=0.1), decimals=2)
            sps.update(**model.params)  # make sure we are caching the correct IMF / SFH / etc
            for logzsol in logzsol_grid:
                model.params["logzsol"] = np.array([logzsol])
                _ = model.predict(model.theta, obs=dummy_obs, sps=sps)
        print('Done')
        from functools import partial
        lnprobfn_fixed = partial(lnprobfn, sps=sps)
        print('Starting posterior emcee sampling..')
        if withmpi:
            from multiprocessing import Pool
            from multiprocessing import cpu_count

            with Pool(int(n_cores)) as pool:
                nprocs = n_cores
                output = fit_model(obs, model, sps, pool=pool, queue_size=nprocs, lnprobfn=lnprobfn_fixed,
                                   **run_params)
        else:
            output = fit_model(obs, model, sps, lnprobfn=lnprobfn_fixed, **run_params)

        # output = fit_model(obs, model, sps, lnprobfn=lnprobfn, **run_params)
        print('done emcee in {0}s'.format(output["sampling"][1]))

        if os.path.exists("prospector_result.h5"):
            os.system('rm prospector_result.h5')

        hfile = "prospector_result.h5"
        writer.write_hdf5(hfile, run_params, model, obs,
                          output["sampling"][0], output["optimization"][0],
                          tsample=output["sampling"][1],
                          toptimize=output["optimization"][1])

        print('Finished')

    # Loading results file
    result, _, _ = reader.results_from("prospector_result.h5", dangerous=False)

    # Finding the Maximum A Posteriori (MAP) model
    imax = np.argmax(result['lnprobability'])
    i, j = np.unravel_index(imax, result['lnprobability'].shape)
    theta_max = result['chain'][i, j, :].copy()

    # saving results
    save_results(result, model, obs, sps, theta_max, tde_name, path, n_walkers, n_inter, n_burn, n_cores)

    # Writing host subtracted light curves
    host_sub_lc(tde_name, path)

    print('MAP value: {}'.format(theta_max))
    fit_plot = plot_resulting_fit(tde_name, path)

    try:
        os.mkdir(os.path.join(path, tde_name, 'plots'))
    except:
        pass

    fit_plot.savefig(os.path.join(path, tde_name, 'plots', tde_name + '_host_fit.png'), bbox_inches='tight', dpi=300)
    if show_figs:
        plt.show()
    corner_fig = corner_plot(result)
    corner_fig.savefig(os.path.join(path, tde_name, 'plots', tde_name + '_cornerplot.png'), bbox_inches='tight',
                       dpi=300)
    if show_figs:
        plt.show()
    os.chdir(path)
