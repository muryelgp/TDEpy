import os
import warnings
import matplotlib.pyplot as plt
import numpy as np

from astropy.utils.exceptions import AstropyWarning

warnings.simplefilter('ignore', category=AstropyWarning)

from astropy.coordinates import FK5, SkyCoord
from astropy.utils.exceptions import AstropyWarning
from astroquery.irsa_dust import IrsaDust
from astropy.io import fits

from astroquery.vizier import Vizier
from astroquery.simbad import Simbad
import astropy.units as units
import gPhoton.gAperture

import fit_host as fit_host
import reduction as reduction
import tools as tools

warnings.simplefilter('ignore', category=AstropyWarning)


class TDE:

    def __init__(self, tde_name: str, path: str):

        # Checking for names incompatibilities

        if (str(tde_name)[0:2] == 'AT') or (str(tde_name)[0:2] == 'at'):
            at_name = str(tde_name)[2:]
            tde_name = str('AT' + at_name)
            self.is_tns = True
        else:
            self.is_tns = False
            if tde_name.find(' ') != -1:
                f = tde_name.find(' ')
                tde_name = tde_name[0:f] + '_' + tde_name[(f + 1):]
        self.__file__ = os.getcwd()
        self.work_dir = path
        self.name = tde_name
        self.tde_dir = os.path.join(path, self.name)
        self.sw_dir = os.path.join(self.tde_dir, 'swift')
        self.plot_dir = os.path.join(self.tde_dir, 'plots')
        self.host_dir = os.path.join(self.tde_dir, 'host')
        self.other_name, self.ra, self.dec, self.target_id, self.n_sw_obs, self.ebv, self.z, self.host_name, self.discovery_date = \
            None, None, None, None, None, None, None, None, None

        # Checking if object/folder was already created
        try:
            os.mkdir(self.tde_dir)
            os.chdir(self.tde_dir)
        except:
            os.chdir(self.tde_dir)

        # Checking if info file has already been created
        if os.path.exists(os.path.join(self.tde_dir, self.name + '_info.fits')):
            self.other_name, self.ra, self.dec, self.target_id, self.n_sw_obs, self.ebv, self.z, self.host_name, self.discovery_date = \
                self._load_info()

    def download_data(self, target_id=None, n_obs=None):
        """
        This function download all Swift/UVOT observations on the TDE, as well as ZTF data if available:

        -Finds the position of the event (Ra, Dec) based on IAU's Transient Name Server (TNS)
        -Finds and download all Swift observation to 'path'
        -Search for ZTF observations and download it to 'path'
        -finds Galactic extinction in the LoS, E(B-V)
        -Save all infos into tde_name_info.fits file in the 'path'

        obs: This function only works for TNS objects, i.e. those with names beginning with 'AT', e.g. AT2018dyb.
        """

        if (not self.is_tns) & (target_id is None):
            raise Exception(
                '\nFor now this functions only works for AT* named sources (e.g, AT2020zso), for non IAU names you\n'
                'need to insert both the Swift Target ID (target_id) of the source as well as the number of observations(n_obs)\n'
                'to be downloaded. You can search for this at: https://www.swift.ac.uk/swift_portal/')

        print('Searching for ' + str(self.name) + ' information...')
        try:
            # Getting RA and DEC and other infos from TNS
            self.ra, self.dec, self.z, self.host_name, self.other_name, self.discovery_date = reduction.search_tns(str(self.name)[2:])
        except:
            raise Exception('Not able to retrieve data on ' + self.name)

        # Getting Swift Target ID and number of observations
        print('Searching for Swift observations.....')


        if (target_id is None) & (n_obs is None):
            self.target_id, self.n_sw_obs = reduction.get_target_id(self.name, self.ra, self.dec)
        else:
            self.target_id, self.n_sw_obs = target_id, n_obs
        # Creating/entering Swift dir
        try:
            os.mkdir(self.sw_dir)
            os.chdir(self.sw_dir)
        except:
            os.chdir(self.sw_dir)

        # Checking if some or any data on the source was already downloaded
        if os.path.exists(os.path.join(self.sw_dir, str(self.target_id) + '001')):
            if os.path.exists(os.path.join(self.sw_dir, str(self.target_id) + str(int(self.n_sw_obs)).rjust(3, '0'))):
                pass
            else:
                for i in range(2, int(self.n_sw_obs)):
                    # Looking if the folder content is updated in relation to Swift archive
                    if os.path.exists(os.path.join(self.sw_dir, str(self.target_id) + str(int(i)).rjust(3, '0'))):
                        pass
                    else:
                        print('Updating observations..')
                        # Downloading new observations not present in the event dir
                        reduction.download_swift(self.target_id, self.n_sw_obs, init=int(i - 1))
                        break
        else:
            # Downloading all available Swift data
            print('Downloading Swift data..... it will take some time!')
            reduction.download_swift(self.target_id, self.n_sw_obs, init=1)
        print('All Swift data for ' + str(self.name) + ' downloaded!')

        # Getting foreground Milky Way extinction along the line of sight, from dust map astroquery.irsa_dust
        self.ebv = self.get_ebv()  # E(B-V) from Schlafly & Finkbeiner 2011

        # Saving info into a fits file
        self.save_info()

        os.chdir(self.tde_dir)

        # Looking to ZTF data
        print('Looking for ZTF data')
        if self.other_name[0:3] == 'ZTF':
            ztf_name = self.other_name
            try:
                os.mkdir('ztf')
            except:
                pass
            try:
                reduction.download_ztfdata_lasair(ztf_name)
            except:
                pass
            try:
                os.mkdir('photometry')
            except:
                pass
            # saving ZTF data
            reduction.load_ztfdata(ztf_name, self.ebv)
        else:
            print('There is not ZTF for this source')
            pass

        os.chdir(self.work_dir)

    def sw_photometry(self, radius=None, coords=None, write_to_file=True, show_light_curve=True, aper_cor=False):
        """
        This function does the photometry on the Swift observations:

        -Creates .reg files and does photometry in the data using 'uvotsource' from Nasa's Hearsec Software
        -Corrects for reddening effect
        -Saves resulting photometry files for each band into a 'photometry' dir
        -Plots the light curve(s)

        Parameters
        ----------------

        radius : list
            The radius of the source and background regions in arcseconds, the default is radius=[5,50], therefore 5''
            and 50'', respectively for source and background region.

        coords : list
            Only has effect if the data was not downloaded with 'download_data()', the format is: coords=[ra, dec, z].
             ra and dec should be in degrees, and z can be None if it is not known. The ra and dec values will be one
             that will be used to create the region files, so they should be precisely centered on the source.

        write_to_file : Boolean
            Whether to create files with the resulting photometry data. Default is True.

        show_light_curve : Boolean
            Whether to show the light curve plot or not. Default is True.
        """

        if radius is None:
            radius = [5, 50]
        elif type(radius) != list:
            raise Exception('radius should be a list with science and background region radius in arcsec, e.g. [5, 50]')

        if show_light_curve:
            plt.ion()

        os.chdir(self.work_dir)

        if not self.is_tns:
            self.ra, self.dec, self.z = coords
            self.other_name, self.target_id, self.n_sw_obs, self.host_name = 'None', 'None', 'None', 'None'
            self.ebv = self.get_ebv()
            self.save_info()
        if self.is_tns:

            self.other_name, self.ra, self.dec, self.target_id, self.n_sw_obs, self.ebv, self.z, self.host_name, self.discovery_date = \
                self._load_info()

            os.chdir(self.tde_dir)
            try:
                # trying to find swift data folder
                os.chdir(self.sw_dir)
            except:
                os.mkdir(self.sw_dir)

            print('Starting photometry for ' + self.name)

            # Checking whether the .reg files already exist
            if os.path.exists(os.path.join(self.sw_dir, 'source.reg')) and os.path.exists(
                    os.path.join(self.sw_dir, 'bkg.reg')):
                print('Swift .reg files present.')
                pass
            else:
                print('Creating .reg files...')
                # Creating and plotting the .reg (science and background) files using the Ra and Dec
                reduction.create_reg(self.ra, self.dec, radius, self.sw_dir)

            # Doing photometry in the Swift data

            reduction.do_sw_photo(self.sw_dir, aper_cor)

            # saving to files
            if write_to_file:
                reduction.write_files(self.tde_dir, self.sw_dir, self.ebv)
            else:
                pass

            # Leaving Swift dir
            os.chdir(self.tde_dir)

            self.plot_light_curve(write_plot=True, show_plot=show_light_curve)
        # returning to the working path
        os.chdir(self.work_dir)

    def plot_light_curve(self, host_sub=False, bands=None, write_plot=True, show_plot=True, figure_ext='png',
                         plot_host=None):
        """
        This function plots the TDEs light curves.

        Parameters
        ----------------
        host_sub : Boolean
            Whether the host galaxy contribution should be discounted. Default is False.

        bands : list
            Which bands should be plotted. The default is all the Swift bands. The available bands are:
            'sw_w2', 'sw_m2', 'sw_w1', 'sw_uu', 'sw_bb', 'sw_vv', 'ztf_g' and 'ztf_r'.

        write_plot : Boolean
            Whether to create the write the figure a file. Default is True.

        show_plot : Boolean
            Whether to show the light curve plot or not. Default is True.

        figure_ext: string
            The format of the figure file to be saved. Default is 'png' but it can also be 'pdf'.

        plot_host:
            In construction!
        """

        # Creating color and legend dictionaries for each band
        color_dic = dict(sw_uu='blue', sw_bb='cyan', sw_vv='gold', sw_w1='navy', sw_m2='darkviolet',
                         sw_w2='magenta', ztf_g='green', ztf_r='red')
        legend_dic = dict(sw_uu='$U$', sw_bb='$B$', sw_vv='$V$', sw_w1=r'$UV~W1$', sw_m2=r'$UV~M2$',
                          sw_w2=r'$UV~W2$', ztf_g='g', ztf_r='r')

        band_dic = dict(sw_uu='U', sw_bb='B', sw_vv='V', sw_w1='UVW1', sw_m2='UVM2',
                        sw_w2='UVW2', ztf_g='g', ztf_r='r')
        bands_plotted = []
        if host_sub:
            host_bands, model_wl_c, model_ab_mag, model_ab_mag_err, model_flux, model_flux_err, catalogs = \
                np.loadtxt(os.path.join(self.host_dir, 'host_model_phot.txt'),
                           dtype={'names': (
                               'band', 'wl_0', 'ab_mag', 'ab_mag_err',
                               'flux', 'flux_err', 'catalog'),
                               'formats': (
                                   'U5', np.float, np.float, np.float,
                                   np.float, np.float, 'U10')},
                           unpack=True, skiprows=1)

        if not host_sub:
            if bands is None:
                bands = ['sw_uu', 'sw_bb', 'sw_vv', 'sw_w1', 'sw_m2', 'sw_w2']
        elif host_sub:
            if bands is None:
                bands = ['sw_uu', 'sw_bb', 'sw_vv', 'sw_w1', 'sw_m2', 'sw_w2', 'ztf_r', 'ztf_g']

        mjd_max = 0
        fig, ax = plt.subplots(figsize=(12, 8))
        for band in bands:
            # Loading and plotting Swift data
            print(band)
            if band[0] == 's':
                if host_sub:
                    try:
                        data_path = os.path.join(self.tde_dir, 'photometry', 'host_sub', str(band) + '.txt')
                        obsid, mjd, abmag, abmage, flu, flue, signal_host = np.loadtxt(data_path, skiprows=2,
                                                                                       unpack=True)

                    except:
                        continue
                else:
                    try:
                        data_path = os.path.join(self.tde_dir, 'photometry', 'obs', str(band) + '.txt')
                        obsid, mjd, abmag, abmage, flu, flue = np.loadtxt(data_path, skiprows=2,
                                                                          unpack=True)
                    except:
                        continue

                flag = abmag > 0
                if np.sum(flag) > 0:
                    ax.errorbar(mjd[flag], abmag[flag], yerr=abmage[flag], marker="o", linestyle='',
                                color=color_dic[band],
                                linewidth=1, markeredgewidth=0.5, markeredgecolor='black', markersize=8, elinewidth=0.7,
                                capsize=0,
                                label=legend_dic[band])
                    bands_plotted.append(band)
                    if np.max(mjd[flag]) > mjd_max:
                        mjd_max = np.max(mjd[flag])

            # Loading and plotting ZTF data, only if it is present and host_sub=True
            elif band[0] == 'z':
                if os.path.exists(os.path.join(self.tde_dir, 'photometry', 'host_sub', str(band) + '.txt')):
                    mjd, abmag, abmage, flux, fluxe = np.loadtxt(
                        os.path.join(self.tde_dir, 'photometry', 'host_sub', str(band) + '.txt'), skiprows=2,
                        unpack=True)
                    ax.errorbar(mjd, abmag, yerr=abmage, marker="o", linestyle='',
                                color=color_dic[band],
                                linewidth=1, markeredgewidth=0.5, markeredgecolor='black', markersize=8, elinewidth=0.7,
                                capsize=0,
                                label=legend_dic[band])
                    bands_plotted.append(band)
                    if np.max(mjd) > mjd_max:
                        mjd_max = np.max(mjd)
                else:
                    pass
            else:
                break

        if plot_host is not None:
            # Plotting host mag
            for band in bands_plotted:
                # Loading and plotting Swift data

                if host_sub:
                    if band[0] == 's':
                        band_host_flag = host_bands == band_dic[band]
                        ax.errorbar(mjd_max + plot_host, model_ab_mag[band_host_flag][0],
                                    yerr=model_ab_mag_err[band_host_flag][0],
                                    marker="*", linestyle='', color=color_dic[band], linewidth=1, markeredgewidth=0.5,
                                    markeredgecolor='black', markersize=15, elinewidth=0.7, capsize=0)
                    elif band[0] == 'z':
                        band_host_flag = host_bands == band_dic[band]
                        ax.errorbar(mjd_max + plot_host, model_ab_mag[band_host_flag][0],
                                    yerr=model_ab_mag_err[band_host_flag][0],
                                    marker="*", linestyle='', color=color_dic[band], linewidth=1, markeredgewidth=0.5,
                                    markeredgecolor='black', markersize=15, elinewidth=0.7, capsize=0)

        ax.set_xlabel('MJD', fontsize=14)
        ax.set_ylabel('AB mag', fontsize=14)
        ax.invert_yaxis()
        if host_sub:
            title = self.name + ' host subtracted light curve'
            fig_name = self.name + '_host_sub_light_curve.'
        else:
            title = self.name + ' light curve'
            fig_name = self.name + '_light_curve.'
        ax.set_title(title)
        if len(bands_plotted) > 3:
            plt.legend(ncol=2)
        else:
            plt.legend(ncol=1)
        if write_plot:
            try:
                os.chdir(self.plot_dir)
            except:
                os.mkdir(self.plot_dir)
                os.chdir(self.plot_dir)
            plt.savefig(os.path.join(self.plot_dir, fig_name + figure_ext), bbox_inches='tight')
        if show_plot:
            plt.show()

    def download_host_data(self, cor_2mass=False):
        """
        This function downloads and saves the host galaxy SED, from MID-IR to UV. It uses the following photometric
        catalogs:

        Mid-IR: AllWISE (W1, W2, W3, W4);
        Near-IR: UKIDS (Y, J, H, K) if available, otherwise uses 2MASS (J, H, Ks);
        Optical: For Dec > -30 uses PAN-STARRS (y, z, i, r, g), for Dec < -30 uses, if avalible, DES (Y, z, i, r, g),
        otherwise uses Southern SkyMapper (u, z, i, r, g, v). Also uses SDSS u band if available.
        UV: Uses gPhoton package to measure GALEX (NUV, FUV) photometry, if there is no detection in the coordinates,
        it does aperture photometry in the image.

        Results are written in 'host' directory inside the TDE directory.
        """
        ra_host = dec_host = None
        print('Searching for ' + self.name + ' host galaxy data:')
        if self.host_name != 'None':
            print('The host galaxy name is ' + str(self.host_name))
            result = Simbad.query_object(self.host_name)
            if result is not None:
                ra_host = result['RA'][0]
                dec_host = result['DEC'][0]
                coords_host = SkyCoord(ra=ra_host, dec=dec_host, unit=(units.hourangle, units.deg), frame=FK5)
                ra_host = coords_host.ra.deg
                dec_host = coords_host.dec.deg
            else:
                self.host_name = 'None'

        if self.host_name == 'None':
            result = Simbad.query_region(SkyCoord(ra=self.ra, dec=self.dec, unit=(units.deg, units.deg)),
                                         radius=0.0014 * units.deg)
            if result is not None:
                ra_host = result['RA'][0]
                dec_host = result['DEC'][0]
                self.host_name = (result['MAIN_ID'][0]).decode('utf-8')
                coords_host = SkyCoord(ra=ra_host, dec=dec_host, unit=(units.hourangle, units.deg), frame=FK5)
                ra_host = coords_host.ra.deg
                dec_host = coords_host.dec.deg
                self.save_info()
            else:
                ra_host = self.ra
                dec_host = self.dec

        coords_host = SkyCoord(ra=ra_host, dec=dec_host, unit=(units.deg, units.deg))
        W4 = W3 = W2 = W1 = Y = J = H = K = Ks = y = z = i = r = g = u = v = fuv = nuv = None
        e_W4 = e_W3 = e_W2 = e_W1 = e_Y = e_J = e_H = e_K = e_Ks = e_y = e_z = e_i = e_r = e_g = e_u = e_v = e_fuv = e_nuv = None

        try:
            os.mkdir(self.host_dir)
            os.chdir(self.host_dir)
        except:
            os.chdir(self.host_dir)
        host_file = open(os.path.join(self.host_dir, 'host_phot.txt'), 'w')
        host_file.write("# if ab_mag_err = nan it means the measurement is a upper limit\n")
        host_file.write('band' + '\t' + 'wl_0' + '\t' + 'ab_mag' + '\t' + 'ab_mag_err' + '\t' + 'catalog' + '\n')

        # Searching for Wise data
        print('Searching WISE data...')

        v = Vizier(
            columns=['RAJ2000', 'DEJ2000', 'W1mag', 'e_W1mag', 'W2mag', 'e_W2mag', 'W3mag', 'e_W3mag', 'W4mag',
                     'e_W4mag', "+_r"])
        result = v.query_region(coords_host, radius=0.0014 * units.deg, catalog=['II/328'])
        try:
            obj = result[0]
            W1, W2, W3, W4 = tools.vega_to_ab(obj['W1mag'][0], 'W1'), tools.vega_to_ab(obj['W2mag'][0], 'W2'), \
                             tools.vega_to_ab(obj['W3mag'][0], 'W3'), tools.vega_to_ab(obj['W4mag'][0], 'W4')
            e_W1, e_W2, e_W3, e_W4 = obj['e_W1mag'][0], obj['e_W2mag'][0], obj['e_W3mag'][0], obj['e_W4mag'][0]

            if np.isfinite(W4 * e_W4):
                host_file.write(tools.format_host_photo('W4', 221940, W4, e_W4, 'WISE'))
            if np.isfinite(W3 * e_W3):
                host_file.write(tools.format_host_photo('W3', 120820, W3, e_W3, 'WISE'))
            if np.isfinite(W2 * e_W2):
                host_file.write(tools.format_host_photo('W2', 46180, W2, e_W2, 'WISE'))
            if np.isfinite(W1 * e_W1):
                host_file.write(tools.format_host_photo('W1', 33500, W1, e_W1, 'WISE'))
        except:
            print('No WISE data found')
            pass

        # searching for UKIDSS
        print('Searching UKIDSS data...')
        v = Vizier(
            columns=['RAJ2000', 'DEJ2000', 'pYmag', 'pJmag', 'pHmag', 'pKmag', 'e_pYmag', 'e_pJmag', 'e_pHmag',
                     'e_pKmag', "+_r"])
        result = v.query_region(coords_host, radius=0.0014 * units.deg, catalog=['II/314/las8'])

        try:

            obj = result[0]

            Y, J, H, K = tools.vega_to_ab(obj['pYmag'][0], 'Y'), tools.vega_to_ab(obj['pJmag'][0], 'J'), \
                         tools.vega_to_ab(obj['pHmag'][0], 'H'), tools.vega_to_ab(obj['pKmag'][0], 'K')
            e_Y, e_J, e_H, e_K = obj['e_pYmag'][0], obj['e_pJmag'][0], obj['e_pHmag'][0], obj['e_pKmag'][0]

            if np.isfinite(K):
                host_file.write(tools.format_host_photo('K', 22010, K, e_K, 'UKIDSS'))
            if np.isfinite(H):
                host_file.write(tools.format_host_photo('H', 16313, H, e_H, 'UKIDSS'))
            if np.isfinite(J):
                host_file.write(tools.format_host_photo('J', 12483, J, e_J, 'UKIDSS'))
            if np.isfinite(Y):
                host_file.write(tools.format_host_photo('Y', 10305, Y, e_Y, 'UKIDSS'))

        except:
            print('No UKIDSS data found')
            pass

        # Searching for 2MASS data if UKIDSS data were not found
        if len(result) == 0:
            if cor_2mass == False:
                print('Searching 2MASS data...')
                v = Vizier(
                    columns=['RAJ2000', 'DEJ2000', 'Jmag', 'Hmag', 'Kmag', 'e_Jmag', 'e_Hmag', 'e_Kmag', "+_r"])
                result = v.query_region(coords_host, radius=0.0014 * units.deg, catalog=['II/246'])
                try:
                    obj = result[0]
                    J, H, Ks = tools.vega_to_ab(obj['Jmag'][0], 'J'), tools.vega_to_ab(obj['Hmag'][0], 'H'), \
                               tools.vega_to_ab(obj['Kmag'][0], 'Ks')
                    e_J, e_H, e_Ks = obj['e_Jmag'][0], obj['e_Hmag'][0], obj['e_Kmag'][0]

                    if np.isfinite(Ks):
                        host_file.write(tools.format_host_photo('Ks', 21590, Ks, e_Ks, '2MASS'))
                    if np.isfinite(H):
                        host_file.write(tools.format_host_photo('H', 16313, H, e_H, '2MASS'))
                    if np.isfinite(J):
                        host_file.write(tools.format_host_photo('J', 12483, J, e_J, '2MASS'))
                except:
                    print('No 2MASS Data found')
                    pass
            if cor_2mass:
                print('Searching 2MASS data...')
                v = Vizier(
                    columns=['RAJ2000', 'DEJ2000', 'Jstdap', 'Hstdap', 'Kstdap', 'e_Jstdap', 'e_Hstdap', 'e_Kstdap',
                             "+_r"])
                result = v.query_region(coords_host, radius=0.0014 * units.deg, catalog=['II/246'])
                try:
                    obj = result[0]
                    J, H, Ks = tools.vega_to_ab(obj['Jstdap'][0], 'J'), tools.vega_to_ab(obj['Hstdap'][0], 'H'), \
                               tools.vega_to_ab(obj['Kstdap'][0], 'Ks')
                    e_J, e_H, e_Ks = obj['e_Jstdap'][0], obj['e_Hstdap'][0], obj['e_Kstdap'][0]

                    if np.isfinite(Ks):
                        host_file.write(tools.format_host_photo('Ks', 21590, Ks, e_Ks, '2MASS'))
                    if np.isfinite(H):
                        host_file.write(tools.format_host_photo('H', 16313, H, e_H, '2MASS'))
                    if np.isfinite(J):
                        host_file.write(tools.format_host_photo('J', 12483, J, e_J, '2MASS'))
                except:
                    print('No 2MASS Data found')
                    pass
            if cor_2mass is None:
                pass

        if dec_host > -30:
            print('Searching PAN-STARRS data...')
            v = Vizier(
                columns=['RAJ2000', 'DEJ2000', 'objID', 'yKmag', 'zKmag', 'iKmag', 'rKmag', 'gKmag', 'e_yKmag',
                         'e_zKmag',
                         'e_iKmag', 'e_rKmag', 'e_gKmag', "+_r"])
            result = v.query_region(coords_host, radius=0.0014 * units.deg, catalog=['II/349/ps1'])
            try:
                obj = result[0]

                y, z, i, r, g = obj['yKmag'][0], obj['zKmag'][0], obj['iKmag'][0], obj['rKmag'][0], obj['gKmag'][0]
                e_y, e_z, e_i, e_r, e_g = obj['e_yKmag'][0], obj['e_zKmag'][0], obj['e_iKmag'][0], obj['e_rKmag'][0], \
                                          obj['e_gKmag'][0]

                if np.isfinite(y):
                    host_file.write(tools.format_host_photo('y', 9633, y, e_y, 'PAN-STARRS'))
                if np.isfinite(z):
                    host_file.write(tools.format_host_photo('z', 8679, z, e_z, 'PAN-STARRS'))
                if np.isfinite(i):
                    host_file.write(tools.format_host_photo('i', 7545, i, e_i, 'PAN-STARRS'))
                if np.isfinite(r):
                    host_file.write(tools.format_host_photo('r', 6215, r, e_r, 'PAN-STARRS'))
                if np.isfinite(g):
                    host_file.write(tools.format_host_photo('g', 4866, g, e_g, 'PAN-STARRS'))

            except:
                print('No PAN-STARRS Data found')
                pass

        if dec_host <= -30:
            print('Searching DES data...')
            v = Vizier(
                columns=['RAJ2000', 'DEJ2000', 'Ymag', 'zmag', 'imag', 'rmag', 'gmag', 'e_Ymag', 'e_zmag', 'e_imag',
                         'e_rmag', 'e_gmag', "+_r"])
            result = v.query_region(coords_host,
                                    radius=0.0014 * units.deg,
                                    catalog=['II/357/des_dr1'])
            try:
                obj = result[0]
                Y, z, i, r, g = obj['Ymag'][0], obj['zmag'][0], obj['imag'][0], obj['rmag'][0], obj['gmag'][0]
                e_Y, e_z, e_i, e_r, e_g = obj['e_Ymag'][0], obj['e_zmag'][0], obj['e_imag'][0], obj['e_rmag'][0], \
                                          obj['e_gmag'][0]
                if np.isfinite(Y):
                    host_file.write(tools.format_host_photo('Y', 10305, Y, e_Y, 'DES'))
                if np.isfinite(z):
                    host_file.write(tools.format_host_photo('z', 8660, z, e_z, 'DES'))
                if np.isfinite(i):
                    host_file.write(tools.format_host_photo('i', 7520, i, e_i, 'DES'))
                if np.isfinite(r):
                    host_file.write(tools.format_host_photo('r', 6170, r, e_r, 'DES'))
                if np.isfinite(g):
                    host_file.write(tools.format_host_photo('g', 4810, g, e_g, 'DES'))
            except:
                print('No DES Data found')
                pass

            if len(result) == 0:
                print('Searching SkyMapper data...')
                v = Vizier(
                    columns=['RAJ2000', 'DEJ2000', 'uPetro', 'e_uPetro', 'vPetro', 'gPetro', 'rPetro', 'iPetro',
                             'zPetro',
                             'e_vPetro',
                             'e_gPetro', 'e_rPetro', 'e_iPetro', 'e_zPetro', "+_r"])
                result = v.query_region(coords_host, radius=0.0014 * units.deg, catalog=['II/358/smss'])
                try:
                    obj = result[0]
                    u, z, i, r, g, v = obj['uPetro'][0], obj['zPetro'][0], obj['iPetro'][0], obj['rPetro'][0], \
                                       obj['gPetro'][0], obj['vPetro'][0]
                    e_u, e_z, e_i, e_r, e_g, e_v = obj['e_uPetro'][0], obj['e_zPetro'][0], obj['e_iPetro'][0], \
                                                   obj['e_rPetro'][0], obj['e_gPetro'][0], obj['e_vPetro'][0]

                    if np.isfinite(z):
                        host_file.write(tools.format_host_photo('z', 8660, z, e_z, 'SkyMapper'))
                    if np.isfinite(i):
                        host_file.write(tools.format_host_photo('i', 7520, i, e_i, 'SkyMapper'))
                    if np.isfinite(r):
                        host_file.write(tools.format_host_photo('r', 6170, r, e_r, 'SkyMapper'))
                    if np.isfinite(g):
                        host_file.write(tools.format_host_photo('g', 4810, g, e_g, 'SkyMapper'))
                    if np.isfinite(v):
                        host_file.write(tools.format_host_photo('v', 4110, v, e_v, 'SkyMapper'))
                    print(u)
                except:
                    print('No SkyMapper Data found')
                    pass

        # Searching SDSS u band photometry
        print('Searching SDSS data...')
        v = Vizier(
            columns=['RAJ2000', 'DEJ2000', 'umag', 'e_umag', "+_r"])
        result = v.query_region(coords_host, radius=0.0014 * units.deg, catalog=['V/147/sdss12'])
        try:
            obj = result[0]
            u, e_u = obj['umag'][0] - 0.04, obj['e_umag'][0]
            if np.isfinite(u):
                host_file.write(tools.format_host_photo('u', 3500, u, e_u, 'SDSS'))
            else:
                try:
                    if np.isfinite(u):
                        host_file.write(tools.format_host_photo('u', 3551, u, e_u, 'SkyMapper'))
                except:
                    print('No SDSS Data found')
                    pass
        except:
            try:
                if np.isfinite(u):
                    host_file.write(tools.format_host_photo('u', 3551, u, e_u, 'SkyMapper'))
            except:
                pass

        # Getting GALEX data
        print('Measuring UV photometry from GALEX data...')
        try:
            nuv_data = gPhoton.gAperture("NUV", [ra_host, dec_host], radius=0.0014, annulus=[0.0015, 0.0050],
                                         coadd=True, overwrite=True)
            try:
                nuv = nuv_data['mag'][0]
            except:
                nuv = np.nan
            try:
                e_nuv = nuv_data['mag_err_1'][0]
            except:
                e_nuv = np.nan
        except:
            nuv = np.nan
            e_nuv = np.nan

        try:
            fuv_data = gPhoton.gAperture('FUV', [ra_host, dec_host], radius=0.0014, annulus=[0.0015, 0.0050],
                                         coadd=True, overwrite=True)
            try:
                fuv = fuv_data['mag'][0]
            except:
                fuv = np.nan
            try:
                e_fuv = fuv_data['mag_err_1'][0]
            except:
                e_fuv = np.nan
        except:
            fuv = np.nan
            e_fuv = np.nan

        if np.isfinite(nuv):
            host_file.write(tools.format_host_photo('NUV', 2271, nuv, e_nuv, 'GALEX'))
        if np.isfinite(fuv):
            host_file.write(tools.format_host_photo('FUV', 1528, fuv, e_fuv, 'GALEX'))
        host_file.close()
        self.plot_host_sed()
        os.chdir(self.work_dir)

    def plot_host_sed(self, show_plot=True):
        """
        This function plots the host galaxy SED.
        You should run download_host_data() first.
        """
        try:
            band, wl_c, ab_mag, ab_mag_err, catalogs = np.loadtxt(os.path.join(self.host_dir, 'host_phot.txt'),
                                                                  dtype={'names': (
                                                                      'band', 'wl_0', 'ab_mag', 'ab_mag_err',
                                                                      'catalog'),
                                                                      'formats': (
                                                                          'U5', np.float, np.float, np.float, 'U10')},
                                                                  unpack=True, skiprows=2)
        except:
            raise Exception('We should run download_host_data() before trying to plot it.')

        color_dic = {"WISE": "maroon", "UKIDSS": "coral", "2MASS": 'red', 'PAN-STARRS': 'green', 'DES': 'lime',
                     'SkyMapper': 'greenyellow', 'SDSS': 'blue', 'GALEX': 'darkviolet', 'Swift/UVOT': 'darkviolet'}

        finite = (np.isfinite(ab_mag)) & (np.isfinite(ab_mag_err))

        fig, ax = plt.subplots(figsize=(16, 8))
        for catalog in np.unique(catalogs[finite]):
            flag = (catalogs == catalog) & (np.isfinite(ab_mag)) & (np.isfinite(ab_mag_err))
            ax.errorbar(wl_c[flag], ab_mag[flag], yerr=ab_mag_err[flag], marker='D', fmt='o',
                        color=color_dic[catalog],
                        linewidth=3, markeredgecolor='black', markersize=8, elinewidth=3, capsize=5, capthick=3,
                        markeredgewidth=1, label=catalog)
        ax.invert_yaxis()
        for catalog in np.unique(catalogs[~finite]):
            flag = (catalogs == catalog) & (~np.isfinite(ab_mag_err))
            ax.errorbar(wl_c[flag], ab_mag[flag], yerr=0.5, lolims=np.ones(np.shape(ab_mag[flag]), dtype=bool),
                        marker='D', fmt='o', color=color_dic[catalog],
                        markeredgecolor='black', markersize=8, elinewidth=2, capsize=6, capthick=3,
                        markeredgewidth=1, label=catalog)

        plt.xscale('log')
        ax.set_xlim(700, 300000)
        ax.set_xticks([1e3, 1e4, 1e5])
        ax.set_xticklabels(['0.1', '1', '10'])
        ymin, ymax = np.min(ab_mag) * 0.85, np.max(ab_mag) * 1.1
        ax.set_ylim(ymax, ymin)
        ax.set_ylabel('AB mag', fontsize=14)
        ax.set_xlabel(r'Wavelength $[\mu m]$', fontsize=14)
        ax.set_title('Host Galaxy SED (' + self.name + ')')
        plt.legend(loc=4)
        try:
            os.chdir(self.plot_dir)
        except:
            os.mkdir(self.plot_dir)
            os.chdir(self.plot_dir)
        plt.savefig(os.path.join(self.plot_dir, self.name + '_host_sed.png'), bbox_inches='tight', dpi=300)
        if show_plot:
            plt.show()
        os.chdir(self.work_dir)

    def fit_host_sed(self, n_cores, multi_processing=True, init_theta=None, n_walkers=None, n_inter=None, n_burn=None):
        if self.z is None:
            self.z = np.nan
        if np.isfinite(float(self.z)):
            print('Starting fitting ' + self.name + ' host galaxy SED')
            print(
                'THIS PROCESS WILL TAKE A LOT OF TIME!! try to increase the numbers of processing cores (n_cores), if possible..')
            self.save_info()
            fit_host.run_prospector(self.name, self.work_dir, self.z, withmpi=multi_processing, n_cores=n_cores,
                                    init_theta=init_theta, n_walkers=n_walkers, n_inter=n_inter, n_burn=n_burn)
        else:
            raise Exception('You need to define a redshift (z) for the source before fitting the host SED')

    def plot_host_sed_fit(self, corner_plot=False):
        os.chdir(self.host_dir)

        result, obs, _ = fit_host.reader.results_from("prospector_result.h5", dangerous=False)
        imax = np.argmax(result['lnprobability'])

        i, j = np.unravel_index(imax, result['lnprobability'].shape)
        theta_max = result['chain'][i, j, :].copy()
        print('MAP value: {}'.format(theta_max))
        fit_plot = fit_host.plot_resulting_fit(self.name, self.work_dir)
        fit_plot.savefig(os.path.join(path, tde_name, 'plots', tde_name + '_host_fit.png'), bbox_inches='tight',
                         dpi=300)
        plt.show()

        if corner_plot:
            c_plt = fit_host.corner_plot(result)
            c_plt.savefig(os.path.join(path, tde_name, 'plots', tde_name + '_cornerplot.png'), bbox_inches='tight',
                          dpi=300)
            plt.show()
        os.chdir(self.work_dir)

    def save_info(self):
        from astropy.table import Table
        t = Table({'TDE_name': np.array([str(self.name)]),
                   'other_name': np.array([str(self.other_name)]),
                   'ra': np.array([float(self.ra)]),
                   'dec': np.array([float(self.dec)]),
                   'sw_target_id': np.array([self.target_id]),
                   'n_sw_obs': np.array([self.n_sw_obs]),
                   'E(B-V)': np.array([str(self.ebv)]),
                   'z': np.array([str(self.z)]),
                   'host_name': np.array([str(self.host_name)]),
                  'discovery_date(MJD)': np.array([float(self.discovery_date)])})
        t.write(self.tde_dir + '/' + str(self.name) + '_info.fits', format='fits', overwrite=True)


    def _load_info(self):
        info = fits.open(os.path.join(self.work_dir, self.name, str(self.name) + '_info.fits'))
        other_name = info[1].data['other_name'][0]
        ra = info[1].data['ra'][0]
        dec = info[1].data['dec'][0]
        target_id = info[1].data['sw_target_id'][0]
        n_sw_obs = info[1].data['n_sw_obs'][0]
        ebv = float(info[1].data['E(B-V)'][0])
        z, host_name = (info[1].data['z'][0]), info[1].data['host_name'][0]
        discovery_date = info[1].data['discovery_date(MJD)'][0]
        if z == 'None':
            z = None
        if host_name == 'None':
            host_name = None
        info.close()
        return other_name, ra, dec, target_id, n_sw_obs, ebv, z, host_name, discovery_date

    def get_ebv(self):
        """
        This function return the E(B-V) Galactic extinction in the line of sight at position of the source.


        Returns
        ----------------
        ebv : float
            E(B-V) Galactic extinction in the line of sight at ra and dec given.
        """
        coo = SkyCoord(ra=float(self.ra), dec=float(self.dec), unit=units.deg, frame=FK5)
        table = IrsaDust.get_query_table(coo, section='ebv')
        ebv = table['ext SandF mean'][0]
        return ebv


if __name__ == "__main__":
    tde_name = 'AT2020ocn'
    path = '/home/muryel/Dropbox/data/TDEs/'

    tde = TDE(tde_name, path)
    tde.download_data()