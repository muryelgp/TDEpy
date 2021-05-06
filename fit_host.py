import os

import sedpy
from prospect.io import write_results as writer
import matplotlib.pyplot as plt
import numpy as np
from prospect.fitting import fit_model
from prospect.fitting import lnprobfn
from prospect.likelihood import lnlike_spec, lnlike_phot
from prospect.models.templates import TemplateLibrary
import prospect.io.read_results as reader
from prospect.sources import CSPSpecBasis

# re-defining plotting defaults

plt.rcParams.update({'font.size': 16})


def mag_to_flux(ab_mag, wl):
    fnu = (10. ** (-0.4 * (48.6 + ab_mag)))
    flam_g = (2.99792458e+18 * fnu) / (wl ** 2.)
    return flam_g


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
        band, wl_c, ab_mag, ab_mag_err, catalogs = np.loadtxt(os.path.join(tde_dir, 'host', 'host_phot.txt'),
                                                              dtype={'names': (
                                                                  'band', 'wl_0', 'ab_mag', 'ab_mag_err', 'catalog'),
                                                                  'formats': (
                                                                      'U3', np.float, np.float, np.float, 'U10')},
                                                              unpack=True, skiprows=1)
    except:
        raise Exception('We should run download_host_data() before trying to fit it.')

    filter_dic = {'WISE_W4': 'wise_w4', 'WISE_W3': 'wise_w3', 'WISE_W2': 'wise_w2', 'WISE_W1': 'wise_w1',
                  'UKIDSS_Y': 'decam_Y', 'UKIDSS_J': 'twomass_J', 'UKIDSS_H': 'twomass_H', 'UKIDSS_K': 'twomass_Ks',
                  '2MASS_J': 'twomass_J', '2MASS_H': 'twomass_H', '2MASS_Ks': 'twomass_Ks',
                  'PAN-STARRS_y': 'hsc_y', 'PAN-STARRS_z': 'hsc_z', 'PAN-STARRS_i': 'hsc_i', 'PAN-STARRS_r': 'hsc_r',
                  'PAN-STARRS_g': 'hsc_g',
                  'DES_Y': 'decam_Y', 'DES_z': 'decam_z', 'DES_i': 'decam_i', 'DES_r': 'decam_r', 'DES_g': 'decam_g',
                  'SkyMapper_u': 'sdss_u0', 'SkyMapper_z': 'hsc_z', 'SkyMapper_i': 'hsc_i', 'SkyMapper_r': 'hsc_r',
                  'SkyMapper_g': 'hsc_g', 'SkyMapper_v': 'stromgren_v',
                  'SDSS_u': 'sdss_u0', 'GALEX_NUV': 'galex_NUV', 'GALEX_FUV': 'galex_FUV'}

    flag = np.isfinite(ab_mag * ab_mag_err)

    catalog_bands = [catalogs[flag][i] + '_' + band[flag][i] for i in range(len(catalogs[flag]))]
    filternames = [filter_dic[i] for i in catalog_bands]
    # And here we instantiate the `Filter()` objects using methods in `sedpy`,
    # and put the resultinf list of Filter objects in the "filters" key of the `obs` dictionary
    obs["filters"] = sedpy.observate.load_filters(filternames)

    # Now we store the measured fluxes for a single object, **in the same order as "filters"**
    # The units of the fluxes need to be maggies (Jy/3631) so we will do the conversion here too.
    mags = np.array(ab_mag[flag])
    obs["maggies"] = 10 ** (-0.4 * mags)

    # And now we store the uncertainties (again in units of maggies)
    # In this example we are going to fudge the uncertainties based on the supplied `snr` meta-parameter.
    obs["maggies_unc"] = np.array(ab_mag_err[flag]) * obs["maggies"]

    # Now we need a mask, which says which flux values to consider in the likelihood.
    # IMPORTANT: the mask is *True* for values that you *want* to fit,
    # and *False* for values you want to ignore.  Here we ignore the spitzer bands.
    obs["phot_mask"] = np.array(np.isfinite(ab_mag[flag]) * np.isfinite(ab_mag_err[flag]))

    # This is an array of effective wavelengths for each of the filters.
    # It is not necessary, but it can be useful for plotting so we store it here as a convenience
    obs["phot_wave"] = np.array(wl_c[flag])

    obs["wavelength"] = None
    obs["spectrum"] = None
    obs['unc'] = None
    obs['mask'] = None

    # This function ensures all required keys are present in the obs dictionary,
    # adding default values if necessary
    obs = fix_obs(obs)

    return obs


def build_model(object_redshift=None, init_theta=None):
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

    model_params["sfh"]["init"] = 1

    # Changing the initial values appropriate for our objects and data
    model_params["dust2"]["init"] = init_theta[2]
    model_params["tau"]["init"] = init_theta[4]
    model_params["mass"]["init"] = init_theta[0]
    model_params["logzsol"]["init"] = init_theta[1]
    model_params["tage"]["init"] = init_theta[3]

    # Setting the priors forms and limits
    model_params["dust2"]["prior"] = priors.TopHat(mini=0.0, maxi=2.85)
    model_params["tau"]["prior"] = priors.TopHat(mini=0.1, maxi=30)
    model_params["mass"]["prior"] = priors.LogUniform(mini=1e6, maxi=1e12)
    model_params["logzsol"]["prior"] = priors.TopHat(mini=-1.8, maxi=0.2)
    model_params["tage"]["prior"] = priors.TopHat(mini=8, maxi=13.3)

    # Setting the spread of the walkers for the mcmc sampling
    model_params["dust2"]["disp_floor"] = 0.1
    model_params["tau"]["disp_floor"] = 1
    model_params["mass"]["disp_floor"] = 1e7
    model_params["logzsol"]['disp_floor'] = 0.1
    model_params["tage"]["disp_floor"] = 1

    # Fixing and defining the object redshift
    model_params["zred"]['isfree'] = False
    model_params["zred"]['init'] = object_redshift

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


    band, obs_wl_c, obs_ab_mag, obs_ab_mag_err, catalogs = np.loadtxt(os.path.join(host_dir, 'host_phot.txt'),
                                                          dtype={'names': (
                                                              'band', 'wl_0', 'ab_mag', 'ab_mag_err',
                                                              'catalog'),
                                                              'formats': (
                                                                  'U3', np.float, np.float, np.float, 'U10')},
                                                          unpack=True, skiprows=1)
    n_bands = len(band)

    band, model_wl_c, model_ab_mag, model_ab_mag_err, catalogs = np.loadtxt(os.path.join(host_dir, 'host_model_phot.txt'),
                                                                      dtype={'names': (
                                                                          'band', 'wl_0', 'ab_mag', 'ab_mag_err',
                                                                          'catalog'),
                                                                          'formats': (
                                                                              'U3', np.float, np.float, np.float,
                                                                              'U10')},
                                                                      unpack=True, skiprows=1)
    band_flag = [i < n_bands for i in range(len(model_wl_c))]

    spec_wl_0, spec_ab_mag, spec_ab_mag_err, spec_flux, spec_flux_err = np.loadtxt(os.path.join(host_dir, 'host_model_spec.txt'),
                                                                      dtype={'names': (
                                                                          'wl_0', 'ab_mag', 'ab_mag_err',
                                                                          'flux', 'flux_err'),
                                                                          'formats': (
                                                                              np.float, np.float, np.float,
                                                                              np.float, np.float)},
                                                                      unpack=True, skiprows=1)



    ax.plot(spec_wl_0, spec_ab_mag, label='Model spectrum (MAP)',
               lw=0.7, color='green', alpha=0.8)
    ax.fill_between(spec_wl_0, spec_ab_mag+spec_ab_mag_err, spec_ab_mag-spec_ab_mag_err, color='green', alpha=0.2, label='Posterior')
    ax.errorbar(model_wl_c[band_flag], model_ab_mag[band_flag], yerr=model_ab_mag_err[band_flag], label='Model photometry (MAP)',
                 marker='s', markersize=8, alpha=0.85, ls='', lw=3, ecolor='green', capsize=5,
                 markerfacecolor='none', markeredgecolor='green',
                 markeredgewidth=3)
    ax.errorbar(obs_wl_c, obs_ab_mag, yerr=obs_ab_mag_err,
                 label='Observed photometry', ecolor='red',
                 marker='o', markersize=8, ls='', lw=3, alpha=0.85, capsize=5,
                 markerfacecolor='none', markeredgecolor='red',
                 markeredgewidth=3)

    temp = np.interp(np.linspace(700, 300000, 10000), spec_wl_0, spec_ab_mag)
    ymin, ymax = temp.min() * 0.85, temp.max() * 1.1
    ax.invert_yaxis()
    plt.xscale('log')
    ax.set_xlim(700, 300000)
    ax.set_xticks([1e3, 1e4, 1e5])
    ax.set_xticklabels(['0.1', '1', '10'])
    ax.set_ylim(ymax, ymin)
    ax.set_ylabel('AB mag', fontsize=14)
    ax.set_xlabel(r'Wavelength $[\mu m]$', fontsize=14)
    ax.set_title('Host Galaxy SED Fit (' + tde_name + ')')
    plt.legend(loc=4)
    plt.show()
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
        if i == 0:
            bounds.append((np.log10(theta_max[i]) - 4 * mean_dist, np.log10(theta_max[i]) + 4 * mean_dist))
        else:
            bounds.append((theta_max[i] - 4 * mean_dist, theta_max[i] + 4 * mean_dist))

    cornerfig = reader.subcorner(result, thin=5,
                                 fig=plt.subplots(5, 5, figsize=(27, 27))[0], logify=["mass", "tau"], range=bounds)

    return cornerfig


def configure(tde_name, path, z, init_theta):
    # Setting same paraters for the mcmc sampling
    run_params = {"object_redshift": z, "fixed_metallicity": False, "add_duste": False, "verbose": True,
                  "optimize": False, "emcee": True, "dynesty": False, "nwalkers": 100, "niter": 1000, "nburn": [500]}

    # Instantiating observation object and sps
    obs = build_obs(path, tde_name)
    sps = build_sps()

    # Instantiating model object
    model = build_model(object_redshift=z, init_theta=init_theta)

    return obs, sps, model, run_params


def save_results(result, model, obs, sps, theta_max, tde_name, path):
    tde_dir = os.path.join(path, tde_name)
    band, _, _, _, catalogs = np.loadtxt(os.path.join(tde_dir, 'host', 'host_phot.txt'),
                                         dtype={'names': (
                                             'band', 'wl_0', 'ab_mag', 'ab_mag_err', 'catalog'),
                                             'formats': (
                                                 'U3', np.float, np.float, np.float, 'U10')},
                                         unpack=True, skiprows=1)

    # Adding new bands (Swift, SDSS, HST)
    wphot = obs["phot_wave"]

    obs['filters'].append(sedpy.observate.Filter('bessell_V'))
    wphot = np.append(wphot, 5468)
    band = np.append(band, 'V')
    catalogs = np.append(catalogs, 'Swift/UVOT')

    obs['filters'].append(sedpy.observate.Filter('bessell_B'))
    wphot = np.append(wphot, 4392)
    band = np.append(band, 'B')
    catalogs = np.append(catalogs, 'Swift/UVOT')

    obs['filters'].append(sedpy.observate.Filter('bessell_U'))
    wphot = np.append(wphot, 3465)
    band = np.append(band, 'U')
    catalogs = np.append(catalogs, 'Swift/UVOT')

    obs['filters'].append(sedpy.observate.Filter('uvot_w1'))
    wphot = np.append(wphot, 2600)
    band = np.append(band, 'UV-W1')
    catalogs = np.append(catalogs, 'Swift/UVOT')

    obs['filters'].append(sedpy.observate.Filter('uvot_m2'))
    wphot = np.append(wphot, 2246)
    band = np.append(band, 'UV-M2')
    catalogs = np.append(catalogs, 'Swift/UVOT')

    obs['filters'].append(sedpy.observate.Filter('uvot_w2'))
    wphot = np.append(wphot, 1928)
    band = np.append(band, 'UV-W2')
    catalogs = np.append(catalogs, 'Swift/UVOT')

    obs['filters'].append(sedpy.observate.Filter('sdss_u0'))
    wphot = np.append(wphot, 3551)
    band = np.append(band, 'u')
    catalogs = np.append(catalogs, 'SDSS')

    obs['filters'].append(sedpy.observate.Filter('sdss_g0'))
    wphot = np.append(wphot, 4686)
    band = np.append(band, 'g')
    catalogs = np.append(catalogs, 'SDSS')

    obs['filters'].append(sedpy.observate.Filter('sdss_r0'))
    wphot = np.append(wphot, 6166)
    band = np.append(band, 'r')
    catalogs = np.append(catalogs, 'SDSS')

    obs['filters'].append(sedpy.observate.Filter('sdss_i0'))
    wphot = np.append(wphot, 7480)
    band = np.append(band, 'i')
    catalogs = np.append(catalogs, 'SDSS')

    obs['filters'].append(sedpy.observate.Filter('sdss_z0'))
    wphot = np.append(wphot, 8932)
    band = np.append(band, 'z')
    catalogs = np.append(catalogs, 'SDSS')

    obs['filters'].append(sedpy.observate.Filter('wfc3_uvis_f275w'))
    wphot = np.append(wphot, 2750)
    band = np.append(band, 'F275W')
    catalogs = np.append(catalogs, 'HST/WFC3')

    obs['filters'].append(sedpy.observate.Filter('wfc3_uvis_f336w'))
    wphot = np.append(wphot, 3375)
    band = np.append(band, 'F336W')
    catalogs = np.append(catalogs, 'HST/WFC3')

    obs['filters'].append(sedpy.observate.Filter('wfc3_uvis_f475w'))
    wphot = np.append(wphot, 4550)
    band = np.append(band, 'F475W')
    catalogs = np.append(catalogs, 'HST/WFC3')

    obs['filters'].append(sedpy.observate.Filter('wfc3_uvis_f555w'))
    wphot = np.append(wphot, 5410)
    band = np.append(band, 'F555W')
    catalogs = np.append(catalogs, 'HST/WFC3')

    obs['filters'].append(sedpy.observate.Filter('wfc3_uvis_f606w'))
    wphot = np.append(wphot, 5956)
    band = np.append(band, 'F606W')
    catalogs = np.append(catalogs, 'HST/WFC3')

    obs['filters'].append(sedpy.observate.Filter('wfc3_uvis_f814w'))
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
    nwalkers, niter = 100, 500
    err_phot = []
    err_spec = []
    for i in range(int(1e4)):
        theta = result['chain'][randint(nwalkers), 500 + randint(niter)]
        mspec, mphot, mextra = model.mean_model(theta, obs, sps=sps)
        err_phot.append(mphot)
        err_spec.append(mspec)

    err_phot = np.std(err_phot, axis=0)
    err_phot_mag = abs(np.log10(abs(mphot_map - err_phot)) / -0.4 - np.log10(mphot_map) / -0.4)
    small_err = np.round(err_phot_mag, 2) < 0.01
    err_phot_mag[small_err] = 0.01

    # Saving modelled photometry
    host_dir = os.path.join(tde_dir, 'host')
    host_file = open(os.path.join(host_dir, 'host_model_phot.txt'), 'w')
    host_file.write('band' + '\t' + 'wl_0' + '\t' + 'ab_mag' + '\t' + 'ab_mag_err' + '\t' + 'catalog' + '\n')
    phot_mag = -2.5 * np.log10(mphot_map)
    for yy in range(len(wphot)):
        host_file.write(
            str(band[yy]) + '\t' + str(wphot[yy]) + '\t' + '{:.2f}'.format(phot_mag[yy]) + '\t' + '{:.2f}'.format(
                err_phot_mag[yy]) + '\t' + str(catalogs[yy]) + '\n')
    host_file.close()

    # Saving modelled spectrum
    spec_mag = -2.5 * np.log10(mspec_map)
    err_spec = (np.std(err_spec, axis=0))
    err_spec_mag = abs(np.log10(abs(mspec_map - err_spec)) / -0.4 - np.log10(mspec_map) / -0.4)
    spec_flux = mag_to_flux(spec_mag, wspec)
    err_spec_flux = spec_flux - mag_to_flux(spec_mag + err_spec_mag, wspec)

    host_file = open(os.path.join(host_dir, 'host_model_spec.txt'), 'w')
    host_file.write('wl_0' + '\t' + 'ab_mag' + '\t' + 'ab_mag_err' + '\t' + 'flux' + '\t' + 'flux_err' + '\n')
    for yy in range(len(wspec)):
        host_file.write('{:.2f}'.format(wspec[yy]) + '\t' + '{:.2f}'.format(spec_mag[yy]) + '\t' +
                        '{:.2f}'.format(err_spec_mag[yy]) + '\t' + str(spec_flux[yy]) +
                        '\t' + str(err_spec_flux[yy]) + '\n')
    host_file.close()


def run_prospector(tde_name, path, z, withmpi, n_cores, init_theta=None):
    os.chdir(os.path.join(path, tde_name, 'host'))

    if init_theta is None:
        init_theta = [1e10, 0, 0.05, 1, 1]

    obs, sps, model, run_params = configure(tde_name, path, z, init_theta)
    print(model)
    print("\nInitial free parameter vector theta:\n  {}\n".format(model.theta))

    if withmpi & ('logzsol' in model.free_params):
        dummy_obs = dict(filters=None, wavelength=None)

        logzsol_prior = model.config_dict["logzsol"]['prior']
        lo, hi = logzsol_prior.range
        logzsol_grid = np.around(np.arange(lo, hi, step=0.1), decimals=2)
        sps.update(**model.params)  # make sure we are caching the correct IMF / SFH / etc
        for logzsol in logzsol_grid:
            print(logzsol)
            model.params["logzsol"] = np.array([logzsol])
            _ = model.predict(model.theta, obs=dummy_obs, sps=sps)

    from functools import partial
    lnprobfn_fixed = partial(lnprobfn, sps=sps)

    if withmpi:
        from multiprocessing import Pool
        from multiprocessing import cpu_count

        with Pool() as pool:
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
    result, obs, _ = reader.results_from("prospector_result.h5", dangerous=False)

    # Finding the Maximum A Posteriori (MAP) model
    imax = np.argmax(result['lnprobability'])
    i, j = np.unravel_index(imax, result['lnprobability'].shape)
    theta_max = result['chain'][i, j, :].copy()

    # saving results
    save_results(result, model, obs, sps, theta_max, tde_name, path)


    print('MAP value: {}'.format(theta_max))
    fit_plot = plot_resulting_fit(tde_name, path)

    try:
        os.mkdir(os.path.join(path, tde_name, 'plots'))
    except:
        pass

    fit_plot.savefig(os.path.join(path, tde_name, 'plots', tde_name + '_host_fit.png'), bbox_inches='tight', dpi=300)
    plt.show()
    corner_fig = corner_plot(result)
    corner_fig.savefig(os.path.join(path, tde_name, 'plots', tde_name + '_cornerplot.png'), bbox_inches='tight', dpi=300)
    plt.show()
    os.chdir(path)
