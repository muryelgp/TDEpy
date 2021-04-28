import os

from matplotlib.pyplot import *
from prospect.fitting import fit_model
from prospect.fitting import lnprobfn
from prospect.likelihood import lnlike_spec, lnlike_phot
from prospect.models.templates import TemplateLibrary

# re-defining plotting defaults

rcParams.update({'font.size': 16})


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
        band, wl_c, ab_mag, ab_mag_err, catalogs = np.loadtxt(os.path.join(tde_dir, 'host', 'host_photometry.txt'),
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

def build_model(object_redshift=None, fixed_metallicity=None, add_duste=False, **extras):
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
    #ldist = cosmo.luminosity_distance(object_redshift).value

    #model_params["lumdist"] = {"N": 1, "isfree": False, "init": ldist, "units": "Mpc"}

    # Let's make some changes to initial values appropriate for our objects and data
    model_params["dust2"]["init"] = 0.05
    model_params["tau"]["init"] = 5
    model_params["mass"]["init"] = 1e10
    model_params["logzsol"]["init"] = 0
    model_params["tage"]["init"] = 10


    # These are dwarf galaxies, so lets also adjust the metallicity prior,
    # the tau parameter upward, and the mass prior downward
    model_params["dust2"]["prior"] = priors.TopHat(mini=0.0, maxi=2.85)
    model_params["tau"]["prior"] = priors.TopHat(mini=0.1, maxi=30)
    model_params["mass"]["prior"] = priors.LogUniform(mini=1e6, maxi=1e12)
    model_params["logzsol"]["prior"] = priors.TopHat(mini=-1.8, maxi=0.2)
    model_params["tage"]["prior"] = priors.TopHat(mini=8, maxi=10.5)

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

def plot_init_model(model, obs, sps, theta):
    initial_spec, initial_phot, initial_mfrac = model.sed(theta, obs=obs, sps=sps)
    title_text = ','.join(["{}={}".format(p, model.params[p][0])
                           for p in model.free_params])

    a = 1.0 + model.params.get('zred', 0.0)  # cosmological redshifting
    # photometric effective wavelengths
    wphot = obs["phot_wave"]
    # spectroscopic wavelengths
    if obs["wavelength"] is None:
        # *restframe* spectral wavelengths, since obs["wavelength"] is None
        wspec = sps.wavelengths
        wspec *= a  # redshift them
    else:
        wspec = obs["wavelength"]

    # establish bounds
    xmin, xmax = np.min(wphot) * 0.8, np.max(wphot) / 0.8
    temp = np.interp(np.linspace(xmin, xmax, 10000), wspec, initial_spec)
    ymin, ymax = temp.min() * 0.8, temp.max() / 0.4
    figure(figsize=(16, 8))

    # plot model + data
    loglog(wspec, initial_spec, label='Model spectrum',
           lw=0.7, color='navy', alpha=0.7)
    errorbar(wphot, initial_phot, label='Model photometry',
             marker='s', markersize=10, alpha=0.8, ls='', lw=3,
             markerfacecolor='none', markeredgecolor='blue',
             markeredgewidth=3)
    errorbar(wphot, obs['maggies'], yerr=obs['maggies_unc'],
             label='Observed photometry',
             marker='o', markersize=10, alpha=0.8, ls='', lw=3,
             ecolor='red', markerfacecolor='none', markeredgecolor='red',
             markeredgewidth=3)
    title(title_text)

    # plot Filters
    for f in obs['filters']:
        w, t = f.wavelength.copy(), f.transmission.copy()
        t = t / t.max()
        t = 10 ** (0.2 * (np.log10(ymax / ymin))) * t * ymin
        loglog(w, t, lw=3, color='gray', alpha=0.7)

    # prettify
    xlabel('Wavelength [A]')
    ylabel('Flux Density [maggies]')
    xlim([xmin, xmax])
    ylim([ymin, ymax])
    legend(loc='best', fontsize=20)
    tight_layout()
    show()

def plot_resulting_fit(model, obs, sps, theta, theta_max, tde_name, path):
    initial_spec, initial_phot, initial_mfrac = model.sed(theta, obs=obs, sps=sps)
    mspec, mphot, mextra = model.mean_model(theta, obs, sps=sps)
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
    ymin, ymax = temp.min() * 0.8, temp.max() / 0.4
    figure(figsize=(16, 8))

    mask = obs["phot_mask"]

    loglog(wspec, mspec, label='Model spectrum (random draw)',
          lw=0.7, color='navy', alpha=0.7)
    loglog(wspec, mspec_map, label='Model spectrum (MAP)',
           lw=0.7, color='green', alpha=0.7)
    # loglog(wspec, initial_spec, label='Model spectrum (init)',
    #       lw=0.7, color='black', alpha=0.7)

    errorbar(wphot[mask], mphot[mask], label='Model photometry (random draw)',
         marker='s', markersize=10, alpha=0.8, ls='', lw=3,
         markerfacecolor='none', markeredgecolor='blue',
         markeredgewidth=3)
    errorbar(wphot[mask], mphot_map[mask], label='Model photometry (MAP)',
             marker='s', markersize=10, alpha=0.8, ls='', lw=3,
             markerfacecolor='none', markeredgecolor='green',
             markeredgewidth=3)
    errorbar(wphot[mask], obs['maggies'][mask], yerr=obs['maggies_unc'][mask],
             label='Observed photometry', ecolor='red',
             marker='o', markersize=10, ls='', lw=3, alpha=0.8,
             markerfacecolor='none', markeredgecolor='red',
             markeredgewidth=3)
    # errorbar(wphot[mask], initial_phot[mask], label='Model photometry (init)',
    #         marker='s', markersize=10, alpha=0.8, ls='', lw=3,
    #         markerfacecolor='none', markeredgecolor='grey',
    #         markeredgewidth=3)

    xlabel('Wavelength [A]')
    ylabel('Flux Density [maggies]')
    xlim([xmin, xmax])
    ylim([ymin, ymax])
    legend(loc='best', fontsize=20)
    tight_layout()
    try:
        os.mkdir(os.path.join(path, tde_name, 'plots'))
    except:
        pass
    savefig(os.path.join(path, tde_name, 'plots', tde_name + '_fit_host_sed.png'), dpi=300)
    show()



def run_prospector(tde_name, path, z, init_theta=None, add_duste=False):
    # TemplateLibrary.show_contents()
    #TemplateLibrary.describe("parametric_sfh")

    run_params = {}
    run_params["object_redshift"] = z
    run_params["fixed_metallicity"] = None
    run_params["add_duste"] = add_duste
    run_params["verbose"] = True

    # Instantiating observation object and sps
    obs = build_obs(path, tde_name, **run_params)
    sps = build_sps(**run_params)
    # print(sps.ssp.libraries)

    # Instantiating model object
    model = build_model(**run_params)
    print(model)
    print("\nInitial free parameter vector theta:\n  {}\n".format(model.theta))

    # Generate the model SED at the initial value of theta
    theta = model.theta.copy()

    # Plotting initial model
    # plot_init_model(model, obs, sps, theta)

    # Optmizing first and then sampling with mcmc
    run_params["optimize"] = True
    run_params["emcee"] = True
    run_params["dynesty"] = False
    # Number of emcee walkers
    run_params["nwalkers"] = 128
    # Number of iterations of the MCMC sampling
    run_params["niter"] = 512
    # Number of iterations in each round of burn-in
    # After each round, the walkers are reinitialized based on the
    # locations of the highest probablity half of the walkers.
    run_params["nburn"] = [16, 32, 64]
    run_params["min_method"] = 'lm'
    run_params["nmin"] = 2

    output = fit_model(obs, model, sps, lnprobfn=lnprobfn, **run_params)
    print('done emcee in {0}s'.format(output["sampling"][1]))

    from prospect.io import write_results as writer
    os.chdir(os.path.join(path, tde_name, 'host'))

    if os.path.exists("demo_emcee_mcmc.h5"):
        os.system('rm demo_emcee_mcmc.h5')

    hfile = "demo_emcee_mcmc.h5"
    writer.write_hdf5(hfile, run_params, model, obs,
                      output["sampling"][0], output["optimization"][0],
                      tsample=output["sampling"][1],
                      toptimize=output["optimization"][1])

    print('Finished')

    import prospect.io.read_results as reader
    results_type = "emcee" # | "dynesty"
    # grab results (dictionary), the obs dictionary, and our corresponding models
    # When using parameter files set `dangerous=True`
    result, obs, _ = reader.results_from("demo_{}_mcmc.h5".format(results_type), dangerous=False)
    (results, topt) = output["optimization"]
    # Find which of the minimizations gave the best result,
    # and use the parameter vector for that minimization
    ind_best = np.argmin([r.cost for r in results])

    theta_best = results[ind_best].x.copy()
    #The following commented lines reconstruct the model and sps object,
    # if a parameter file continaing the `build_*` methods was saved along with the results
    #model = reader.get_model(result)
    #sps = reader.get_sps(result)

    # let's look at what's stored in the `result` dictionary
    print(result.keys())

    # maximum a posteriori (of the locations visited by the MCMC sampler)
    imax = np.argmax(result['lnprobability'])
    if results_type == "emcee":
        i, j = np.unravel_index(imax, result['lnprobability'].shape)
        theta_max = result['chain'][i, j, :].copy()
        thin = 5
    else:
        theta_max = result["chain"][imax, :]
        thin = 1

    # randomly chosen parameters from chain
    randint = np.random.randint
    if results_type == "emcee":
        nwalkers, niter = run_params['nwalkers'], run_params['niter']
        theta = result['chain'][randint(nwalkers), randint(niter)]
    else:
        theta = result["chain"][randint(len(result["chain"]))]

    plot_resulting_fit(model, obs, sps, theta, theta_max, tde_name, path)

    print('Optimization value: {}'.format(theta_best))
    print('MAP value: {}'.format(theta_max))
    cornerfig = reader.subcorner(result, start=0, thin=thin, truths=theta_best,
                                 fig=subplots(5, 5, figsize=(27, 27))[0])

    cornerfig.savefig(os.path.join(path, tde_name, 'plots', tde_name + '_fit_host_cornerplot.png'), dpi=300)
    show()
    os.chdir(path)

