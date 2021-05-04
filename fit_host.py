import os
from prospect.io import write_results as writer
import matplotlib.pyplot as plt
import numpy as np
from prospect.fitting import fit_model
from prospect.fitting import lnprobfn
from prospect.likelihood import lnlike_spec, lnlike_phot
from prospect.models.templates import TemplateLibrary
import prospect.io.read_results as reader

# re-defining plotting defaults

plt.rcParams.update({'font.size': 16})


def build_obs(path, tde, **extras):
    """Build a dictionary of observational data.

    :param snr:
        The S/N to assign to the photometry, since none are reported
        in Johnson et al. 2013

    :param ldist:
        The luminosity distance to assume for translating absolute magnitudes
        into apparent magnitudes.

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

    catalog_bands = [catalogs[i] + '_' + band[i] for i in range(len(catalogs))]
    filternames = [filter_dic[i] for i in catalog_bands]
    # And here we instantiate the `Filter()` objects using methods in `sedpy`,
    # and put the resultinf list of Filter objects in the "filters" key of the `obs` dictionary
    obs["filters"] = sedpy.observate.load_filters(filternames)

    # Now we store the measured fluxes for a single object, **in the same order as "filters"**
    # The units of the fluxes need to be maggies (Jy/3631) so we will do the conversion here too.
    mags = np.array(ab_mag)
    obs["maggies"] = 10 ** (-0.4 * mags)

    # And now we store the uncertainties (again in units of maggies)
    # In this example we are going to fudge the uncertainties based on the supplied `snr` meta-parameter.
    obs["maggies_unc"] = np.array(ab_mag_err) * obs["maggies"]

    # Now we need a mask, which says which flux values to consider in the likelihood.
    # IMPORTANT: the mask is *True* for values that you *want* to fit,
    # and *False* for values you want to ignore.  Here we ignore the spitzer bands.
    obs["phot_mask"] = np.array(np.isfinite(ab_mag) * np.isfinite(ab_mag_err))

    # This is an array of effective wavelengths for each of the filters.
    # It is not necessary, but it can be useful for plotting so we store it here as a convenience
    obs["phot_wave"] = np.array(wl_c)

    obs["wavelength"] = None
    obs["spectrum"] = None
    obs['unc'] = None
    obs['mask'] = None

    # This function ensures all required keys are present in the obs dictionary,
    # adding default values if necessary
    obs = fix_obs(obs)

    return obs


def build_model(object_redshift=None, init_theta=None, **extras):
    """Build a prospect.models.SedModel object

    :param object_redshift: (optional, default: None)
        If given, produce spectra and observed frame photometry appropriate
        for this redshift. Otherwise, the redshift will be zero.

    :param ldist: (optional, default: 10)
        The luminosity distance (in Mpc) for the model.  Spectra and observed
        frame (apparent) photometry will be appropriate for this luminosity distance.

    :param fixed_metallicity: (optional, default: None)
        If given, fix the model metallicity (:math:`log(Z/Z_sun)`) to the given value.

    :param add_duste: (optional, default: False)
        If `True`, add dust emission and associated (fixed) parameters to the model.

    :returns model:
        An instance of prospect.models.SedModel
    """
    from prospect.models.sedmodel import SedModel
    from prospect.models.templates import TemplateLibrary
    from prospect.models import priors

    # Get (a copy of) one of the prepackaged model set dictionaries.
    # This is, somewhat confusingly, a dictionary of dictionaries, keyed by parameter name
    model_params = TemplateLibrary["parametric_sfh"]

    # Now add the lumdist parameter by hand as another entry in the dictionary.
    # This will control the distance since we are setting the redshift to zero.
    # In `build_obs` above we used a distance of 10Mpc to convert from absolute to apparent magnitudes,
    # so we use that here too, since the `maggies` are appropriate for that distance.
    model_params["sfh"]["init"] = 1
    # ldist = cosmo.luminosity_distance(object_redshift).value

    # model_params["lumdist"] = {"N": 1, "isfree": False, "init": ldist, "units": "Mpc"}

    # Let's make some changes to initial values appropriate for our objects and data
    model_params["dust2"]["init"] = init_theta[2]
    model_params["tau"]["init"] = init_theta[4]
    model_params["mass"]["init"] = init_theta[0]
    model_params["logzsol"]["init"] = init_theta[1]
    model_params["tage"]["init"] = init_theta[3]

    # These are dwarf galaxies, so lets also adjust the metallicity prior,
    # the tau parameter upward, and the mass prior downward
    model_params["dust2"]["prior"] = priors.TopHat(mini=0.0, maxi=2.85)
    model_params["tau"]["prior"] = priors.TopHat(mini=0.1, maxi=30)
    model_params["mass"]["prior"] = priors.LogUniform(mini=1e6, maxi=1e12)
    model_params["logzsol"]["prior"] = priors.TopHat(mini=-1.8, maxi=0.2)
    model_params["tage"]["prior"] = priors.TopHat(mini=8, maxi=13.3)

    # If we are going to be using emcee, it is useful to provide a
    # minimum scale for the cloud of walkers (the default is 0.1)
    model_params["dust2"]["disp_floor"] = 0.1
    model_params["tau"]["disp_floor"] = 1
    model_params["mass"]["disp_floor"] = 1e7
    model_params["logzsol"]['disp_floor'] = 0.1
    model_params["tage"]["disp_floor"] = 1

    model_params["zred"]['isfree'] = False
    model_params["zred"]['init'] = object_redshift

    # Now instantiate the model object using this dictionary of parameter specifications
    model = SedModel(model_params)

    return model


def build_sps(zcontinuous=1, **extras):
    """
    :param zcontinuous:
        A vlue of 1 insures that we use interpolation between SSPs to
        have a continuous metallicity parameter (`logzsol`)
        See python-FSPS documentation for details
    """
    from prospect.sources import CSPSpecBasis
    sps = CSPSpecBasis(zcontinuous=zcontinuous)
    return sps


def plot_resulting_fit(model, obs, sps, theta_max, tde_name, path):
    mspec_map, mphot_map, _ = model.mean_model(theta_max, obs, sps=sps)
    wphot = obs["phot_wave"]
    # spectroscopic wavelengths

    a = 1.0 + model.params.get('zred', 0.0)  # cosmological redshifting
    if obs["wavelength"] is None:
        # *restframe* spectral wavelengths, since obs["wavelength"] is None
        wspec = sps.wavelengths
        wspec *= a  # redshift them
    else:
        wspec = obs["wavelength"]

    xmin, xmax = np.min(wphot) * 0.8, np.max(wphot) / 0.8
    temp = np.interp(np.linspace(xmin, xmax, 10000), wspec, mspec_map)
    ymin, ymax = temp.min() * 0.5, temp.max() / 0.3
    fig = plt.figure(figsize=(16, 8))

    mask = obs["phot_mask"]

    # loglog(wspec, mspec, label='Model spectrum (random draw)',
    # lw=0.7, color='navy', alpha=0.7)
    plt.loglog(wspec, mspec_map, label='Model spectrum (MAP)',
               lw=0.7, color='green', alpha=0.7)
    # loglog(wspec, initial_spec, label='Model spectrum (init)',
    #       lw=0.7, color='black', alpha=0.7)

    # errorbar(wphot[mask], mphot[mask], label='Model photometry (random draw)',
    #     marker='s', markersize=10, alpha=0.8, ls='', lw=3,
    #     markerfacecolor='none', markeredgecolor='blue',
    #     markeredgewidth=3)
    plt.errorbar(wphot[mask], mphot_map[mask], label='Model photometry (MAP)',
                 marker='s', markersize=10, alpha=0.8, ls='', lw=3,
                 markerfacecolor='none', markeredgecolor='green',
                 markeredgewidth=3)
    plt.errorbar(wphot[mask], obs['maggies'][mask], yerr=obs['maggies_unc'][mask],
                 label='Observed photometry', ecolor='red',
                 marker='o', markersize=10, ls='', lw=3, alpha=0.8,
                 markerfacecolor='none', markeredgecolor='red',
                 markeredgewidth=3)
    # errorbar(wphot[mask], initial_phot[mask], label='Model photometry (init)',
    #         marker='s', markersize=10, alpha=0.8, ls='', lw=3,
    #         markerfacecolor='none', markeredgecolor='grey',
    #         markeredgewidth=3)

    plt.xlabel('Wavelength [A]')
    plt.ylabel('Flux Density [maggies]')
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.legend(loc='best', fontsize=20)
    plt.tight_layout()

    return fig


def corner_plot(result, tde_name, path):
    imax = np.argmax(result['lnprobability'])

    i, j = np.unravel_index(imax, result['lnprobability'].shape)
    theta_max = result['chain'][i, j, :].copy()

    try:
        parnames = np.array(result['theta_labels'], dtype='U20')
    except(KeyError):
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
                                 fig=plt.subplots(5, 5, figsize=(27, 27))[0], logify=["mass"], range=bounds)

    return cornerfig


def configure(tde_name, path, z, init_theta):
    run_params = {}
    run_params["object_redshift"] = z
    run_params["fixed_metallicity"] = False
    run_params["add_duste"] = False
    run_params["verbose"] = True

    # Instantiating observation object and sps
    obs = build_obs(path, tde_name, **run_params)
    sps = build_sps(**run_params)

    # Instantiating model object
    model = build_model(object_redshift=z, init_theta=init_theta)

    # Generate the model SED at the initial value of theta
    run_params["optimize"] = False
    run_params["emcee"] = True
    run_params["dynesty"] = False
    run_params["nwalkers"] = 100
    run_params["niter"] = 1000
    run_params["nburn"] = [500]

    return obs, sps, model, run_params


def save_results(model, obs, sps, theta_max, tde_name, path):
    mspec_map, mphot_map, _ = model.mean_model(theta_max, obs, sps=sps)
    wphot = obs["phot_wave"]

    pass


def run_prospector(tde_name, path, z, withmpi, n_cores, init_theta=None):
    os.chdir(os.path.join(path, tde_name, 'host'))

    if init_theta is None:
        init_theta = [1e10, 0, 0.05, 1, 5]

    obs, sps, model, run_params = configure(tde_name, path, z, init_theta)
    print(model)
    print("\nInitial free parameter vector theta:\n  {}\n".format(model.theta))
    '''
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

   

    if os.path.exists("demo_emcee_mcmc.h5"):
        os.system('rm demo_emcee_mcmc.h5')

    hfile = "prospector_result.h5"
    writer.write_hdf5(hfile, run_params, model, obs,
                      output["sampling"][0], output["optimization"][0],
                      tsample=output["sampling"][1],
                      toptimize=output["optimization"][1])

    print('Finished')
    '''
    result, obs, _ = reader.results_from("prospector_result.h5", dangerous=False)
    imax = np.argmax(result['lnprobability'])

    i, j = np.unravel_index(imax, result['lnprobability'].shape)
    theta_max = result['chain'][i, j, :].copy()


    # saving results
    save_results(model, obs, sps, theta_max, tde_name, path)
    



    print('MAP value: {}'.format(theta_max))
    fit_plot = plot_resulting_fit(model, obs, sps, theta_max, tde_name, path)
    try:
        os.mkdir(os.path.join(path, tde_name, 'plots'))
    except:
        pass
    fit_plot.savefig(os.path.join(path, tde_name, 'plots', tde_name + '_host_fit.png'), dpi=300)
    plt.show()
    cornerfig = corner_plot(result, tde_name, path)
    cornerfig.savefig(os.path.join(path, tde_name, 'plots', tde_name + '_cornerplot.png'), dpi=300)
    plt.show()
    os.chdir(path)
