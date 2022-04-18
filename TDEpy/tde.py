import warnings
import matplotlib.pyplot as plt
import numpy as np
from astropy.utils.exceptions import AstropyWarning
import warnings

import matplotlib.pyplot as plt
import numpy as np
from astropy.utils.exceptions import AstropyWarning
#import pyvo

warnings.simplefilter('ignore', category=AstropyWarning)
from astropy.coordinates import FK5, SkyCoord
from astropy.utils.exceptions import AstropyWarning
from astropy.time import Time
from astroquery.irsa_dust import IrsaDust
from astropy.io import fits
import pandas as pd
from astroquery.simbad import Simbad
import astropy.units as units
import tarfile
import os
import requests
import time
import sys
import io
import pyvo

from . import fit_host as fit_host
from . import reduction as reduction
from . import download_host as download_host
from . import tools as tools
from . import plots as plots
from . import xrt_reduction as xtr_red

from .light_curves import observables
from .models.model_BB_FT_FS import BB_FT_FS
from .models.model_BB_VT_GPS import BB_VT_GPS

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
        self.host_dir = os.path.join(self.tde_dir, 'host')
        self.phot_dir = os.path.join(self.tde_dir, 'photometry')
        self.plot_dir = os.path.join(self.tde_dir, 'plots')
        self.other_name, self.ra, self.dec, self.target_id, self.n_sw_obs, self.ebv, self.z, self.host_name, self.discovery_date = \
            None, None, None, None, None, None, None, None, None
        self.M_bh = [None, None, None]
        self.spec_class = None
        self.host_radius = None
        self.host_mass, self.host_color = None, None

        # Checking if object/folder was already created
        if self.is_tns:
            try:
                os.chdir(self.tde_dir)
                if os.path.exists('modelling'):
                    os.rename('modelling', 'modeling')
            except:
                print('Searching for ' + str(self.name) + ' information...')
                if self.is_tns:
                    try:
                        # Getting RA and DEC and other infos from TNS
                        self.ra, self.dec, self.z, self.host_name, self.other_name, self.discovery_date = reduction.search_tns(
                            str(self.name)[2:])
                    except:
                        raise Exception('Not able to retrieve data on ' + self.name)

                # Getting foreground Milky Way extinction along the line of sight, from dust map astroquery.irsa_dust
                try:
                    self.ebv = self.get_ebv()  # E(B-V) from Schlafly & Finkbeiner 2011
                except:
                    pass
                os.mkdir(self.tde_dir)
                self.save_info()

        if not self.is_tns:
            try:
                os.chdir(self.tde_dir)
            except:
                os.mkdir(self.tde_dir)

        # Checking if info file has already been created
        if os.path.exists(os.path.join(self.tde_dir, self.name + '_info.fits')):
            self._load_info()

    def download_swift(self, target_id=None, n_obs=None):
        """
        This function download all Swift/UVOT observations on the TDE, as well as ZTF data if available:

        -Finds the position of the event (Ra, Dec) based on IAU's Transient Name Server (TNS)
        -Finds and download all Swift observation to 'path'
        -Search for ZTF observations and download it to 'path'
        -finds Galactic extinction in the LoS, E(B-V)
        -Save all infos into tde_name_info.fits file in the 'path'

        obs: This function only works automatically for TNS objects, i.e. those with names beginning with 'AT',
        e.g. AT2018dyb, for other sources the Swift target ID and the number of observations needs to be insert
        manually using the 'target_id' and 'n_obs' parameters.
        """

        if (not self.is_tns) & ((self.ra is None) | (self.ra is None)):
            raise Exception(
                '\nFor now this functions only downloads data automatically for AT* named sources (e.g, AT2020zso),\n'
                'for non IAU names you need to insert both RA and DEC (self.ra and self.dec) manually before downloading swift data \n')

        os.chdir(self.tde_dir)
        # Looking to ZTF data
        print('Starting downloading data on ' + str(self.name))


        # Getting Swift Target ID and number of observations
        print('Searching for Swift observations.....')
        try:
            os.mkdir(self.sw_dir)
            os.chdir(self.sw_dir)
        except:
            os.chdir(self.sw_dir)

        if (target_id is None) & (n_obs is None):
            # Creating/entering Swift dir

            name_list, target_id_list, n_obs_list = reduction.get_target_id(self.ra, self.dec)
            for i in range(len(target_id_list)):
                try:
                    dfs = pd.read_html('https://www.swift.ac.uk/archive/selectseq.php?tid=' + target_id_list[
                        i] + '&source=obs&reproc=1&referer=portal')[3]
                except:
                    continue
                start_time = Time(dfs['Start time (UT)'][1], format='isot', scale='utc').mjd

                if self.discovery_date is None:
                    print(
                        'Downloading Swift data for Target ID ' + str(target_id_list[i]) + ', it will take some time!')
                    reduction.download_swift(target_id_list[i], int(n_obs_list[i]), init=1)
                elif start_time < self.discovery_date - 180.0:
                    try:
                        os.mkdir(self.sw_dir)
                        os.chdir(self.sw_dir)
                    except:
                        os.chdir(self.sw_dir)
                    try:
                        os.mkdir('swift_host')
                        os.chdir('swift_host')
                    except:
                        os.chdir('swift_host')
                    print(
                        'Downloading Swift data for Target ID ' + str(target_id_list[i]) + ', it will take some time!')
                    reduction.download_swift(target_id_list[i], int(n_obs_list[i]), init=1)
                    os.chdir(self.sw_dir)
                elif start_time >= self.discovery_date:
                    print(
                        'Downloading Swift data for Target ID ' + str(target_id_list[i]) + ', it will take some time!')
                    reduction.download_swift(target_id_list[i], int(n_obs_list[i]), init=1)


        else:
            print('Downloading Swift data for Target ID ' + str(target_id) + ', it will take some time!')
            reduction.download_swift(target_id, n_obs, init=1)
            print('All Swift data for Target ID ' + str(target_id) + ' downloaded!')

        os.chdir(self.work_dir)


    def download_ztf(self, where='Lasair'):
        """
           This function looks for ZTF data in the source:

           Parameters
           ----------------

           where : string
               The data archive where to look for the data. Default is to look at 'Lasair',
               the other option is on the ZTF 'forced' photometry server.

        """
        print('Looking for ZTF data at Lasair')
        if (str(self.other_name)[0:3] == 'ZTF' and self.is_tns) or (str(self.name)[0:3] == 'ZTF'):

            ztf_name = self.other_name
            try:
                print('a')
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

        if where == 'forced' or where == 'Forced':

            try:
                os.mkdir('photometry')
            except:
                pass

            try:
                os.mkdir(os.path.join(self.tde_dir, 'photometry', 'host_sub'))
            except:
                pass
            try:
                os.mkdir(os.path.join(self.tde_dir, 'photometry', 'obs'))
            except:
                pass

            #data_path = os.path.join(self.tde_dir, 'photometry', 'host_sub', 'ztf_r.txt')
            #mjd, *_ = np.loadtxt(data_path, skiprows=2, unpack=True)
            url_to_pass = 'https://ztfweb.ipac.caltech.edu/cgi-bin/requestForcedPhotometry.cgi?ra=' + str(self.ra) + '&dec=' + str(self.dec) + '&jdstart=' + str((self.discovery_date - 60 + 2400000.5)) + '&jdend=' + str((self.discovery_date + 500 + 2400000.5)) + '&email=mguolop1@jhu.edu&userpass=gmop545'

            # Making a get request
            import requests
            from requests.auth import HTTPBasicAuth
            response = requests.get(url_to_pass,
                                    auth=HTTPBasicAuth('ztffps', 'dontgocrazy!'))
            print(response)

    def download_atlas(self, days_before=60):

        if not os.path.exists(os.path.join(self.tde_dir, 'atlas', 'atlas_dif_' + self.name + '.csv')):

            BASEURL = "https://fallingstar-data.com/forcedphot"
            resp = requests.post(url=f"{BASEURL}/api-token-auth/",
                                 data={'username': "mguolop1", 'password': "Waytec97"})

            if resp.status_code == 200:
                token = resp.json()['token']
                print(f'Your token is {token}')
                headers = {'Authorization': f'Token {token}', 'Accept': 'application/json'}
            else:
                print(f'ERROR {resp.status_code}')
                print(resp.json())

            task_url = None
            while not task_url:
                with requests.Session() as s:
                    resp = s.post(f"{BASEURL}/queue/", headers=headers, data={
                        'ra': self.ra, 'dec': self.dec, 'mjd_min': float(self.discovery_date) - days_before, 'mjd_max':  float(self.discovery_date) + 360 ,'send_email': False})

                    if resp.status_code == 201:  # successfully queued
                        task_url = resp.json()['url']
                        print(f'The task URL is {task_url}')
                    elif resp.status_code == 429:  # throttled
                        message = resp.json()["detail"]
                        print(f'{resp.status_code} {message}')
                        t_sec = resp.findall(r'available in (\d+) seconds', message)
                        t_min = resp.findall(r'available in (\d+) minutes', message)
                        if t_sec:
                            waittime = int(t_sec[0])
                        elif t_min:
                            waittime = int(t_min[0]) * 60
                        else:
                            waittime = 10
                        print(f'Waiting {waittime} seconds')
                        time.sleep(waittime)
                    else:
                        print(f'ERROR {resp.status_code}')
                        print(resp.json())
                        sys.exit()



            result_url = None
            while not result_url:
                with requests.Session() as s:
                    resp = s.get(task_url, headers=headers)

                    if resp.status_code == 200:  # HTTP OK
                        if resp.json()['finishtimestamp']:
                            result_url = resp.json()['result_url']
                            print(f"Task is complete with results available at {result_url}")
                            break
                        elif resp.json()['starttimestamp']:
                            print(f"Task is running (started at {resp.json()['starttimestamp']})")
                        else:
                            print("Waiting for job to start. Checking again in 10 seconds...")
                        time.sleep(10)
                    else:
                        print(f'ERROR {resp.status_code}')
                        print(resp.json())
                        sys.exit()

            try:
                os.mkdir('atlas')
            except:
                pass
            pwd = os.getcwd()
            os.chdir('atlas')

            with requests.Session() as s:
                textdata = s.get(result_url, headers=headers).text

            dfresult = pd.read_csv(io.StringIO(textdata.replace("###", "")), delim_whitespace=True)
            dfresult.to_csv('atlas_dif_' + self.name + '.csv')


        os.chdir(self.tde_dir)

        try:
            os.mkdir('photometry')
        except:
            pass

        try:
            os.mkdir(os.path.join(self.tde_dir, 'photometry', 'host_sub'))
        except:
            pass
        try:
            os.mkdir(os.path.join(self.tde_dir, 'photometry', 'obs'))
        except:
            pass


        if os.path.exists(os.path.join(self.tde_dir, 'atlas', 'atlas_dif_' + self.name + '.csv')):
            print('Doing forced photometry on ATLAS...')

            id, mjd, mag, mag_err, flux, flux_err, filter, _, _, _, _, _, _, _, _, _, _, mag_5_sigma, _, _ = \
                np.genfromtxt(os.path.join(self.tde_dir, 'atlas', 'atlas_dif_' + self.name + '.csv'), delimiter=",",
                              comments='#', skip_header=1,
                              dtype=str, unpack=True)
            flux = np.array(flux, dtype=float)
            flux_err = np.array(flux_err, dtype=float)
            snr = flux / flux_err
            flag = (filter == 'o') & (snr > 0)
            flux = flux[flag]
            flux_err = flux_err[flag]
            mjd = np.array(mjd[flag], dtype=float)

            mjd_u = np.unique(np.array([int(mjd_i) for mjd_i in mjd]))
            mjd_u_list = np.zeros(np.shape(mjd_u))
            flux_u = np.zeros(np.shape(mjd_u))
            flux_err_u = np.zeros(np.shape(mjd_u))
            mag_u = np.zeros(np.shape(mjd_u))
            mag_err_u = np.zeros(np.shape(mjd_u))

            for i in range(len(mjd_u)):
                flag_mjd_i = np.array([int(mjd_i) for mjd_i in mjd]) == mjd_u[i]
                flux_u[i] = np.mean(flux[flag_mjd_i])
                mjd_u_list[i] = np.mean(mjd[flag_mjd_i])
                flux_err_u[i] = np.sqrt(np.sum((flux_err[flag_mjd_i] / len(flux[flag_mjd_i])) ** 2))
                #flux_err_u[i] = np.nanstd(flux[flag_mjd_i])
                mag_u[i] = (-2.5 * np.log10(flux_u[i]) + 23.9) - (self.ebv * 2.418)
                mag_err_u[i] = abs(2.5 * flux_err_u[i] / (flux_u[i] * np.log(10)))

                if flux_u[i] < 3*flux_err_u[i]:
                    flux_u[i] = 3*flux_err_u[i]
                    mag_u[i] = (-2.5 * np.log10(3*flux_err_u[i]) + 23.9) - (self.ebv * 2.418)
                    mag_err_u[i] = np.nan
                    flux_err_u[i] = np.nan


            atlas_o = open(os.path.join(self.tde_dir, 'photometry', 'host_sub', 'atlas_o.txt'), 'w')
            atlas_o.write('#Values are correct for Galactic extinction\n')
            atlas_o.write('#if ab_mag_err = nan, the measurement is an upper limit\n')
            atlas_o.write(
                'mjd' + '\t' + 'ab_mag' + '\t' + 'ab_mag_err' + '\t' + 'flux_dens' + '\t' + 'flux_dens_err' + '\n')
            for yy in range(len(mjd_u_list)):
                atlas_o.write(
                    '{:.2f}'.format(mjd_u_list[yy]) + '\t' + '{:.2f}'.format(mag_u[yy]) + '\t' + '{:.2f}'.format(
                        mag_err_u[yy]) + '\t' + '{:.2e}'.format(
                        tools.mag_to_flux(mag_u[yy], 6790)) + '\t' + '{:.2e}'.format(
                        tools.dmag_to_df(mag_err_u[yy], tools.mag_to_flux(mag_u[yy], 6790))) + '\n')
            atlas_o.close()


        if not os.path.exists(os.path.join(self.tde_dir, 'atlas', 'atlas_obs_' + self.name + '.csv')):

            BASEURL = "https://fallingstar-data.com/forcedphot"
            resp = requests.post(url=f"{BASEURL}/api-token-auth/",
                                 data={'username': "mguolop1", 'password': "Waytec97"})

            if resp.status_code == 200:
                token = resp.json()['token']
                print(f'Your token is {token}')
                headers = {'Authorization': f'Token {token}', 'Accept': 'application/json'}
            else:
                print(f'ERROR {resp.status_code}')
                print(resp.json())

            task_url = None
            while not task_url:
                with requests.Session() as s:
                    resp = s.post(f"{BASEURL}/queue/", headers=headers, data={
                        'ra': self.ra, 'dec': self.dec, 'mjd_min': float(self.discovery_date) - days_before, 'mjd_max':  float(self.discovery_date) + 360 ,'send_email': False, 'use_reduced': True})

                    if resp.status_code == 201:  # successfully queued
                        task_url = resp.json()['url']
                        print(f'The task URL is {task_url}')
                    elif resp.status_code == 429:  # throttled
                        message = resp.json()["detail"]
                        print(f'{resp.status_code} {message}')
                        t_sec = resp.findall(r'available in (\d+) seconds', message)
                        t_min = resp.findall(r'available in (\d+) minutes', message)
                        if t_sec:
                            waittime = int(t_sec[0])
                        elif t_min:
                            waittime = int(t_min[0]) * 60
                        else:
                            waittime = 10
                        print(f'Waiting {waittime} seconds')
                        time.sleep(waittime)
                    else:
                        print(f'ERROR {resp.status_code}')
                        print(resp.json())
                        sys.exit()



            result_url = None
            while not result_url:
                with requests.Session() as s:
                    resp = s.get(task_url, headers=headers)

                    if resp.status_code == 200:  # HTTP OK
                        if resp.json()['finishtimestamp']:
                            result_url = resp.json()['result_url']
                            print(f"Task is complete with results available at {result_url}")
                            break
                        elif resp.json()['starttimestamp']:
                            print(f"Task is running (started at {resp.json()['starttimestamp']})")
                        else:
                            print("Waiting for job to start. Checking again in 10 seconds...")
                        time.sleep(10)
                    else:
                        print(f'ERROR {resp.status_code}')
                        print(resp.json())
                        sys.exit()

            try:
                os.mkdir('atlas')
            except:
                pass
            pwd = os.getcwd()
            os.chdir('atlas')

            with requests.Session() as s:
                textdata = s.get(result_url, headers=headers).text

            dfresult = pd.read_csv(io.StringIO(textdata.replace("###", "")), delim_whitespace=True)
            dfresult.to_csv('atlas_obs_' + self.name + '.csv')


        os.chdir(self.tde_dir)

        try:
            os.mkdir('photometry')
        except:
            pass

        try:
            os.mkdir(os.path.join(self.tde_dir, 'photometry', 'host_sub'))
        except:
            pass
        try:
            os.mkdir(os.path.join(self.tde_dir, 'photometry', 'obs'))
        except:
            pass


        if os.path.exists(os.path.join(self.tde_dir, 'atlas', 'atlas_obs_' + self.name + '.csv')):
            print('Doing forced photometry on ATLAS...')

            id, mjd, mag, mag_err, flux, flux_err, filter, _, _, _, _, _, _, _, _, _, _, mag_5_sigma, _, _ = \
                np.genfromtxt(os.path.join(self.tde_dir, 'atlas', 'atlas_obs_' + self.name + '.csv'), delimiter=",",
                              comments='#', skip_header=1,
                              dtype=str, unpack=True)
            flux = np.array(flux, dtype=float)
            flux_err = np.array(flux_err, dtype=float)
            snr = flux / flux_err
            flag = (filter == 'o') & (snr > 0)
            flux = flux[flag]
            flux_err = flux_err[flag]
            mjd = np.array(mjd[flag], dtype=float)

            mjd_u = np.unique(np.array([int(mjd_i) for mjd_i in mjd]))
            mjd_u_list = np.zeros(np.shape(mjd_u))
            flux_u = np.zeros(np.shape(mjd_u))
            flux_err_u = np.zeros(np.shape(mjd_u))
            mag_u = np.zeros(np.shape(mjd_u))
            mag_err_u = np.zeros(np.shape(mjd_u))

            for i in range(len(mjd_u)):
                flag_mjd_i = np.array([int(mjd_i) for mjd_i in mjd]) == mjd_u[i]
                flux_u[i] = np.mean(flux[flag_mjd_i])
                mjd_u_list[i] = np.mean(mjd[flag_mjd_i])
                flux_err_u[i] = np.sqrt(np.sum((flux_err[flag_mjd_i] / len(flux[flag_mjd_i])) ** 2))
                # flux_err_u[i] = np.nanstd(flux[flag_mjd_i])
                mag_u[i] = -2.5 * np.log10(flux_u[i]) + 23.9
                mag_err_u[i] = abs(2.5 * flux_err_u[i] / (flux_u[i] * np.log(10)))
                if flux_u[i] < 3 * flux_err_u[i]:
                    flux_u[i] = 3 * flux_err_u[i]
                    mag_u[i] = -2.5 * np.log10(3*flux_err_u[i]) + 23.9
                    mag_err_u[i] = np.nan
                    flux_err_u[i] = np.nan

            atlas_o = open(os.path.join(self.tde_dir, 'photometry', 'obs', 'atlas_o.txt'), 'w')
            atlas_o.write('#if ab_mag_err = nan, the measurement is an upper limit\n')
            atlas_o.write(
                'mjd' + '\t' + 'ab_mag' + '\t' + 'ab_mag_err' + '\t' + 'flux_dens' + '\t' + 'flux_dens_err' + '\n')
            for yy in range(len(mjd_u_list)):
                atlas_o.write(
                    '{:.2f}'.format(mjd_u_list[yy]) + '\t' + '{:.2f}'.format(mag_u[yy]) + '\t' + '{:.2f}'.format(
                        mag_err_u[yy]) + '\t' + '{:.2e}'.format(
                        tools.mag_to_flux(mag_u[yy], 6790)) + '\t' + '{:.2e}'.format(
                        tools.dmag_to_df(mag_err_u[yy], tools.mag_to_flux(mag_u[yy], 6790))) + '\n')
            atlas_o.close()

    def ztt_forced_photo(self, lc_link=None):
        reduction.ztf_forced_photo(self, lc_link)

    def download_wise(self):
        url = "https://irsa.ipac.caltech.edu/SCS?table=neowiser_p1bs_psd"
        objects = pyvo.conesearch(url, pos=(self.ra, self.dec), radius=0.0014)
        table = objects.to_table()

        mjd = table['mjd']
        w1 = table['w1mpro']
        w2 = table['w2mpro']
        w1_err = table['w1sigmpro']
        w2_err = table['w2sigmpro']

        mjd_unique = np.unique(np.array([round(mjd_i, -2) for mjd_i in mjd]))
        w1_u = np.zeros(np.shape(mjd_unique))
        w1_err_u = np.zeros(np.shape(mjd_unique))
        w2_u = np.zeros(np.shape(mjd_unique))
        w2_err_u = np.zeros(np.shape(mjd_unique))
        mjd_u = np.zeros(np.shape(mjd_unique))

        for i in range(len(mjd_unique)):
            # flags
            flag_mjd_u = np.array([round(mjd_i, -2) for mjd_i in mjd]) == mjd_unique[i]
            flag_isw1finite = np.isfinite(w1[flag_mjd_u])
            flag_isw2finite = np.isfinite(w2[flag_mjd_u])

            # medians and std
            mjd_u[i] = np.median(mjd[flag_mjd_u])
            w1_u[i] = tools.vega_to_ab(np.nanmedian(w1[flag_mjd_u][flag_isw1finite]), 'W1')
            w2_u[i] = tools.vega_to_ab(np.nanmedian(w2[flag_mjd_u][flag_isw2finite]), 'W2')
            w1_err_u[i] = np.nanstd(w1[flag_mjd_u][flag_isw1finite])
            w2_err_u[i] = np.nanstd(w2[flag_mjd_u][flag_isw2finite])

            # Dealing with few points
            if w1_err_u[i] == 0.0:
                w1_err_u[i] = np.nanmean(w1_err[flag_mjd_u][flag_isw1finite])
            if w2_err_u[i] == 0.0:
                w2_err_u[i] = np.nanmean(w2_err[flag_mjd_u][flag_isw2finite])

        wise_w1 = open(os.path.join(self.tde_dir, 'photometry', 'obs', 'wise_w1.txt'), 'w')
        #wise_w1.write('#if ab_mag_err = nan, the measurement is an upper limit\n')
        wise_w1.write(
            'mjd' + '\t' + 'ab_mag' + '\t' + 'ab_mag_err' + '\t' + 'flux_dens' + '\t' + 'flux_dens_err' + '\n')
        for yy in range(len(mjd_u)):
            wise_w1.write(
                '{:.2f}'.format(mjd_u[yy]) + '\t' + '{:.2f}'.format(w1_u[yy]) + '\t' + '{:.2f}'.format(
                    w1_err_u[yy]) + '\t' + '{:.2e}'.format(
                    tools.mag_to_flux(w1_u[yy], 33526)) + '\t' + '{:.2e}'.format(
                    tools.dmag_to_df(w1_err_u[yy], tools.mag_to_flux(w1_u[yy], 33526))) + '\n')
        wise_w1.close()

        wise_w2 = open(os.path.join(self.tde_dir, 'photometry', 'obs', 'wise_w2.txt'), 'w')
        # wise_w1.write('#if ab_mag_err = nan, the measurement is an upper limit\n')
        wise_w2.write(
            'mjd' + '\t' + 'ab_mag' + '\t' + 'ab_mag_err' + '\t' + 'flux_dens' + '\t' + 'flux_dens_err' + '\n')
        for yy in range(len(mjd_u)):
            wise_w2.write(
                '{:.2f}'.format(mjd_u[yy]) + '\t' + '{:.2f}'.format(w2_u[yy]) + '\t' + '{:.2f}'.format(
                    w2_err_u[yy]) + '\t' + '{:.2e}'.format(
                    tools.mag_to_flux(w2_u[yy], 46028.00)) + '\t' + '{:.2e}'.format(
                    tools.dmag_to_df(w2_err_u[yy], tools.mag_to_flux(w2_u[yy], 46028.00))) + '\n')
        wise_w2.close()

    def uvot_photometry(self, radius=None, coords=None, aper_cor=False, sigma=3, show=True):
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

        aper_cor: Boolean
            Whether to apply aperture correction or not. See 'uvotsource' documentation for details. Default is False.

        sigma: int
            minimum significance level to be passed to 'uvotsource' task.

        show: Boolean
            Whether to show or not the .reg file figures. Deafault is True.
        """

        if radius is None:
            if self.host_radius is not None:
                if float(self.host_radius) > 5:
                    src_radius = int(round(self.host_radius) + 1)
                    radius = [src_radius, 50]
                else:
                    radius = [5, 50]
            else:
                radius = [5, 50]

        elif type(radius) != list:
            raise Exception('radius should be a list with science and background region radius in arcsec, e.g. [5, 50]')

        os.chdir(self.work_dir)

        if not self.is_tns:
            if coords is None:
                raise Exception('For sources not presented in Transient Name Server (AT* names) you need to input the '
                                'coordinates (ra, dec, z), using the coords = [ra, dec, z] parameter!')
            self.ra, self.dec, self.z = coords
            self.other_name, self.target_id, self.n_sw_obs, self.host_name = 'None', 'None', 'None', 'None'
            self.ebv = self.get_ebv()
            self.save_info()
        if self.is_tns:
            self._load_info()

        os.chdir(self.tde_dir)
        try:
            # trying to find swift data folder
            os.chdir(self.sw_dir)
        except:
            os.mkdir(self.sw_dir)

        print('Starting photometry for ' + self.name)

        if (not os.path.exists('source.reg')) & (not os.path.exists('bkg.reg')):
            # Creating and plotting the .reg (science and background) files using the Ra and Dec
            reduction.create_reg(self.ra, self.dec, radius, self.sw_dir, show)
        elif (os.path.exists('source.reg')) & (os.path.exists('bkg.reg')):
            src = open('source.reg', 'r').readline()
            r = int(np.float(src[int(src.find(')') - 4):int(src.find(')') - 1)]))
            if r != radius[0]:
                reduction.create_reg(self.ra, self.dec, radius, self.sw_dir, show)
            else:
                pass
        else:
            reduction.create_reg(self.ra, self.dec, radius, self.sw_dir, show)

        # Doing photometry in the Swift data
        reduction.do_sw_photo(self.sw_dir, aper_cor, sigma)

        # saving to files
        reduction.write_files(self.tde_dir, self.sw_dir, self.ebv)

        # Leaving Swift dir
        os.chdir(self.tde_dir)

        self.plot_light_curve(show=show)
        # returning to the working path
        os.chdir(self.work_dir)

    def plot_light_curve(self, host_sub=False, host_level=False, units='mag', show=True, title=True):
        """
        This function plots the TDEs light curves.

        Parameters
        ----------------
        host_sub : Boolean
            Whether the host galaxy contribution should be discounted. Default is False.

        units : string
            The units to be plotted, either magnitudes 'mag' or luminosity 'lum'

        show : Boolean
            Whether to show the light curve plot or not. Default is True.

        title : Boolean
            Whether to show the tde.name in the title of the plot.
        """

        if host_sub:
            if units == 'lum':
                if self.z is None:
                    raise Exception('A redshift (z) needs to be inserted for this source')
        plots.plot_light_curve(self, host_sub, host_level, units, show, title)

    def download_host_data(self, mir='Model', nir='default/Petro', opt='Kron/Petro', uv='5', title=False):
        """
        This function downloads and saves the host galaxy SED, from MID-IR to UV.  Results are written in 'host'
        directory inside the TDE directory. For each band distinct aperture are available:

        mir: 'Model' or None
            Mid infrared data are from WISE, the only option is the 'Model' photometry. None excludes the MIR photometry.

        nir: 'default/Petro', 'standard/PSF' or None
            Near Infrared is from 2MASS or UKIDISS, for 2MASS the options are 'default' or 'standard', for UKIDSS
            the options are 'Petro' (Petrosian) or 'PSF' (Point Spread Function), see the catalogs documentation for
            descriptions. None excludes the NIR photometry.

        opt: 'Kron/Petro', 'PSF' or None
            Optical data are from SDSS, PAN-STARSS, DES or SkyMapper. The options are 'Kron/Petro' or 'PSF'.
            None excludes the optical photometry.

        uv: string or None
            UV are taken from GALEX data, the uv variable defines the radius of the aperture in arcseconds. default is '5'.
        """
        ra_host = dec_host = None

        print('Searching for ' + self.name + ' host galaxy data:')
        if self.host_name is not None:
            print('The host galaxy name is ' + str(self.host_name))
            result = Simbad.query_object(self.host_name)
            if result is not None:
                ra_host = result['RA'][0]
                dec_host = result['DEC'][0]
                coords_host = SkyCoord(ra=ra_host, dec=dec_host, unit=(units.hourangle, units.deg), frame=FK5)
                ra_host = coords_host.ra.deg
                dec_host = coords_host.dec.deg
            else:
                self.host_name = None

        if (self.host_name is None) | (str(self.host_name)[0:4] == 'NAME'):

            result = Simbad.query_region(SkyCoord(ra=self.ra, dec=self.dec, unit=(units.deg, units.deg)),
                                         radius=0.0014 * units.deg)

            if result is not None:
                if result is not None:
                    if len(result['MAIN_ID']) > 1:
                        name_flag = [(result['MAIN_ID'][i][0:4] != 'NAME') &
                                     (result['MAIN_ID'][i][0:2] != str(self.name[0:2])) for i in
                                     range(len(result['MAIN_ID']))]
                        ra_host = result['RA'][name_flag][0]
                        dec_host = result['DEC'][name_flag][0]
                        self.host_name = (result['MAIN_ID'][name_flag][0])
                    else:
                        ra_host = result['RA'][0]
                        dec_host = result['DEC'][0]
                        self.host_name = result['MAIN_ID'][0]
                coords_host = SkyCoord(ra=ra_host, dec=dec_host, unit=(units.hourangle, units.deg), frame=FK5)
                ra_host = coords_host.ra.deg
                dec_host = coords_host.dec.deg
                self.save_info()
                print('The host galaxy name is ' + str(self.host_name))
            else:
                ra_host = self.ra
                dec_host = self.dec

        coords_host = SkyCoord(ra=ra_host, dec=dec_host, unit=(units.deg, units.deg))

        self.host_radius = download_host.get_host_radius(coords_host)
        self.save_info()

        try:
            os.mkdir(self.host_dir)
            os.chdir(self.host_dir)
        except:
            os.chdir(self.host_dir)

        host_file_path = os.path.join(self.host_dir, 'host_phot_obs.txt')
        host_file = open(host_file_path, 'w')
        host_file.write("# if ab_mag_err = nan, the measurement is an upper limit\n")
        host_file.write(
            'band' + '\t' + 'wl_0' + '\t' + 'ab_mag' + '\t' + 'ab_mag_err' + '\t' + 'catalog' + '\t' + 'aperture' + '\n')
        host_file.close()

        # Downloading MIR (WISE) data
        if mir == 'Model':
            download_host.download_mir(coords_host, host_file_path)
        elif mir is None:
            pass
        else:
            raise Exception(
                "You need to choose which MIR magnitude (mir) to use, the options are: 'Model', or None")

        # Downloading NIR (UKIDSS or 2MASS) data
        if nir == 'default/Petro':
            download_host.download_nir(nir, coords_host, host_file_path)
        elif nir == 'standard/PSF':
            download_host.download_nir(nir, coords_host, host_file_path)
        elif nir is None:
            pass
        else:
            raise Exception(
                "You need to choose which NIR magnitude (nir) to use, the options are: 'default/Petro', 'standard/PSF' or None")

        # Downloading Optical (Pan-Starrs or DES and/or SkyMapper and/or SDSS) data
        if opt == 'Kron/Petro':
            download_host.download_opt(opt, coords_host, host_file_path)
        elif opt == 'PSF':
            download_host.download_opt(opt, coords_host, host_file_path)
        elif opt is None:
            pass
        else:
            raise Exception(
                "You need to choose which optical (opt) magnitude to use, the options are: 'Kron/Petro', 'PSF' or 'None'")

        # Downloading UV (GALEX or UVOT) data
        if (self.host_radius is not None) and (uv is not None):
            download_host.download_uv(self.host_radius, coords_host, self.host_dir, self.sw_dir)
        elif (self.host_radius is None) and (uv is not None):
            if (np.float(uv) > 1):
                download_host.download_uv(uv, coords_host, self.host_dir, self.sw_dir)
            else:
                Exception(
                    "You need to choose the size of the aperture (uv) for UV data in arcsecs, it should be greater than 1")
        elif uv is None:
            pass

        self.plot_host_sed(title=title)
        os.chdir(self.work_dir)

    def plot_host_sed(self, show=True, title=False):
        """
        This function plots the host galaxy SED.
        You should run download_host_data() first.

         Parameters
        ----------------

        show : Boolean
            Whether to show the light curve plot or not. Default is True.
        """
        plots.plot_host_sed(self, show=show, title=title)
        os.chdir(self.work_dir)

    def fit_host_sed(self, n_cores=None, n_walkers=100, n_inter=2000, n_burn=1500, init_theta=None, show=True,
                     read_only=False, add_agn=False):
        """
        This function fit the host SED using Prospector software (https://github.com/bd-j/prospector/),
         saves modelled host data, and the host subtracted light curves:

        n_cores: integer
            The number of CPU cores to run in parallel.

        n_walkers: integer
            Number os walkers for emcee posterior sampling. Default is 100.

        n_inter: integer
            Number of interactions for emcee sampling. Default is 2000.

        n_burn: integer
            In which interactions the emcee burns will be done. Default is 1500.

        show: Boolean
            Whether to show or not the plots while running. Default is True.

        init_theta: list
            The initial values for the fit. The format is: [mass, log_z, age, tau]. The defualt is [1e10, -1.0, 6, 0.5].

        """
        if self.z is None:
            self.z = np.nan
        if np.isfinite(float(self.z)):

            if not read_only:
                print('Starting fitting ' + self.name + ' host galaxy SED')
                print(
                    'THIS PROCESS WILL TAKE A LOT OF TIME!! try to increase the numbers of processing cores ('
                    'n_cores), if possible..')
                self.save_info()
                if n_cores is None:
                    n_cores = os.cpu_count() / 2

                if n_inter < 1000:
                    n_burn = int(n_inter / 2)
                fit_host.run_prospector(self, n_cores=n_cores, n_walkers=n_walkers, n_inter=n_inter, n_burn=[n_burn],
                                        init_theta=init_theta, show=show, read_only=read_only, add_agn=add_agn)
                print('Creating posterior corner plot!...')
                self.plot_host_sed_fit(corner_plot=True, color_mass=True)
            print('Creating host subtracted light curves!...')
            self.subtract_host(show=show)
            os.chdir(self.host_dir)

            result, obs, _ = fit_host.reader.results_from("prospector_result.h5", dangerous=False)
            self.host_mass, self.host_color = fit_host.get_host_properties(result, self.host_dir, self.ebv)
            self.save_info()

        else:
            raise Exception('You need to define a redshift (z) for the source before fitting the host SED')

    def plot_host_sed_fit(self, corner_plot=False, color_mass=False, show=True, title=True, info=True):
        """
        This function plots the host galaxy SED model.
        You should run fit_host_sed() before calling it.

        show: Boolean
            Whether to show the plots.

        title : Boolean
            Whether to show the tde.name in the title of the plot.

        corner_plot: Boolean
            Whether to plot posterior sampling corner plot too. Default is False.

        color_mass: Boolean
            Whether to plot the color-massdiagram. Default is False.
        """
        os.chdir(self.host_dir)



        result, _, _ = fit_host.reader.results_from("prospector_result.h5", dangerous=False)
        sps = fit_host.build_sps(zcontinuous=1)
        init_theta = [1e10, -1.0, 6, 0.5]
        model = fit_host.build_model(self.ebv, object_redshift=self.z, init_theta=init_theta)
        obs = fit_host.build_obs(self.work_dir, self.name)

        if corner_plot:
            c_plt, chains, map = plots.host_corner_plot(result, obs, model, sps, self.z, self.ebv)
            c_plt.savefig(os.path.join(self.plot_dir, 'host', 'host_model_cornerplot.pdf'), bbox_inches='tight',
                          dpi=300)

            fit_host.save_host_properties(self.host_dir, chains, map)
            if show:
                plt.show()


        fit_plot = plots.plot_host_sed_fit(self, title=title, info=info)
        fit_plot.savefig(os.path.join(self.plot_dir, 'host', 'host_sed_model.pdf'), bbox_inches='tight', dpi=300)
        if show:
            plt.show()

        if color_mass:
            if (self.host_mass is None) or (self.host_color is None):
                self.host_mass, self.host_color = fit_host.get_host_properties(result, self.host_dir, self.ebv)
                self.save_info()
            color_mass_plt = plots.color_mass(self.host_mass, self.host_color)
            color_mass_plt.savefig(os.path.join(self.plot_dir, 'host', 'color_mass_diagram.pdf'), bbox_inches='tight',
                                   dpi=300)
            plt.show()
        os.chdir(self.work_dir)

    def subtract_host(self, show=True):
        """
        This function creates and plot the host subtracted light curves.
        You should run fit_host_sed() before calling it.

        show: Boolean
            Whether to show the host subtract light curves plot.
        """

        if os.path.exists(os.path.join(self.host_dir, 'host_phot_model.txt')):
            fit_host.host_sub_lc(self.name, self.work_dir, self.ebv)
            self.plot_light_curve(host_sub=True, show=show)
            self.plot_light_curve(host_sub=False, show=False)
        else:
            raise FileExistsError(
                "'host_phot_model.txt' not found in host folder, please run the prospector host SED fitting first, i.e. fit_host_sed()")

    def save_info(self):
        '''
        This function updates (save) the info.fits file of the TDE. It should be called after some properties of the
        source are modified.
        '''
        from astropy.table import Table
        t = Table({'name': np.array([str(self.name)]),
                   'other_name': np.array([str(self.other_name)]),
                   'ra': np.array([str(self.ra)]),
                   'dec': np.array([str(self.dec)]),
                   'z': np.array([str(self.z)]),
                   'e(b-v)': np.array([str(self.ebv)]),
                   'discovery_date(mjd)': np.array([str(self.discovery_date)]),
                   'spec_class': np.array([str(self.spec_class)]),
                   'M_bh': np.array([str(self.M_bh[0])]),
                   'M_bh_p16': np.array([str(self.M_bh[1])]),
                   'M_bh_p84': np.array([str(self.M_bh[2])]),
                   'host_name': np.array([str(self.host_name)])})

        from astropy.table import Column

        if self.host_radius is not None:
            c = Column(f'{float(self.host_radius):.2f}', name='host_radius')
            t.add_column(c, index=-1)

        if self.host_mass is not None:
            self.host_mass = tools.round_small(self.host_mass, 0.01)
            c = Column(f'{float(self.host_mass[0]):.2f}', name='host_mass')
            t.add_column(c, index=-1)
            c = Column(f'{float(self.host_mass[1]):.2f}', name='host_mass_p16')
            t.add_column(c, index=-1)
            c = Column(f'{float(self.host_mass[2]):.2f}', name='host_mass_p84')
            t.add_column(c, index=-1)

        if self.host_color is not None:
            self.host_color = tools.round_small(self.host_color, 0.01)
            c = Column(f'{float(self.host_color[0]):.2f}', name='host_color(u-r)')
            t.add_column(c, index=-1)
            c = Column(f'{float(self.host_color[1]):.2f}', name='host_color_err')
            t.add_column(c, index=-1)
        t.write(self.tde_dir + '/' + str(self.name) + '_info.fits', format='fits', overwrite=True)

    def _load_info(self):
        info = fits.open(os.path.join(self.work_dir, self.name, str(self.name) + '_info.fits'))

        try:
            self.ra = float(info[1].data['ra'][0])
        except:
            self.ra = None

        try:
            self.dec = float(info[1].data['dec'][0])
        except:
            self.dec = None

        try:
            self.ebv = float(info[1].data['E(B-V)'][0])
        except:
            self.ebv = None

        try:
            self.other_name = info[1].data['other_name'][0]
        except:
            self.other_name = None

        try:
            self.z = float(info[1].data['z'][0])
            if self.z == 'None':
                self.z = None
        except:
            self.z = None

        try:
            self.host_name = info[1].data['host_name'][0]
            if self.host_name == 'None':
                self.host_name = None
        except:
            self.host_name = None

        try:
            self.discovery_date = float(info[1].data['discovery_date(MJD)'][0])
            if self.discovery_date == 'None':
                self.discovery_date = None
        except:
            self.discovery_date = None

        try:
            mass = float(info[1].data['host_mass'])
            mass_p16 = float(info[1].data['host_mass_p16'])
            mass_p84 = float(info[1].data['host_mass_p84'])
            self.host_mass = [mass, mass_p16, mass_p84]
        except:
            self.host_mass = None

        try:
            color = float(info[1].data['host_color(u-r)'])
            color_err = float(info[1].data['host_color_err'])
            self.host_color = [color, color_err]
        except:
            self.host_color = None

        try:
            self.host_radius = float(info[1].data['host_radius'])
        except:
            self.host_radius = None

        info.close()

    def _get_target_id(self):
        name_list, t_id_list, n_obs_list = reduction.get_target_id(self.name, self.ra, self.dec)
        return name_list, t_id_list, n_obs_list

    def get_host_radius(self):
        coords_host = SkyCoord(ra=self.ra, dec=self.dec, unit=(units.deg, units.deg))
        self.host_radius = download_host.get_host_radius(coords_host)
        self.save_info()

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

    def gen_tar_result(self):
        '''
        This functions saves the resulting files of the TDE in a tar.gz file inside the source folder.
        '''
        pwd = os.getcwd()
        os.chdir(self.work_dir)

        with tarfile.open(os.path.join(self.tde_dir, self.name + '.tar.gz'), "w:gz") as tar_handle:
            if os.path.exists(self.name + '/plots'):
                tar_handle.add(self.name + '/plots')

            if os.path.exists(self.name + '/photometry'):
                tar_handle.add(self.name + '/photometry')

            if os.path.exists(self.name + '/host'):
                tar_handle.add(self.name + '/host')

            if os.path.exists(self.name + '/modeling'):
                tar_handle.add(self.name + '/modeling')

            if os.path.exists(self.name + '/' + self.name + '_info.fits'):
                tar_handle.add(self.name + '/' + self.name + '_info.fits')

        tar_handle.close()
        os.chdir(pwd)

    def fit_light_curve(self, model='BB_FT_FS', pre_peak=True, T_step=30, n_cores=None, n_walkers=100, n_inter=2000,
                        n_burn=1500, verbose=True, title=False):
        if n_cores is None:
            n_cores = 1
        mcmc_args = {'n_walkers': n_walkers,
                     'n_inter': n_inter,
                    'n_burn': n_burn,
                     'n_cores': n_cores}
        modeling_dir = os.path.join(self.tde_dir, 'modeling')
        try:
            os.chdir(modeling_dir)
        except:
            os.mkdir(modeling_dir)
            os.chdir(modeling_dir)

        obs = observables(self)
        if model == 'BB_FT_FS':
            Model = BB_FT_FS()
            from_peak = False
        if model == 'BB_VT_GPS':
            from_peak = pre_peak
            Model = BB_VT_GPS(T_step=T_step)


        good_epochs = Model.gen_good_epoch(obs)
        theta_init = Model.gen_theta_init(obs, good_epochs)
        theta = Model.minimize(obs, good_epochs, theta_init)
        Model.run_mcmc(obs, good_epochs, theta, mcmc_args)
        plots.plot_BB_evolution(self, Model.model_name, from_peak=from_peak, show=verbose, title=title)


    def plot_light_curve_models(self, model, bands='All', show=True, title=False):
        if model == 'BB_FT_FS':
            from_peak = False
        all_bands = ['sw_w2', 'sw_m2', 'sw_w1', 'sw_uu', 'sw_bb', 'ztf_g', 'ztf_r']
        # loading observables
        if bands == 'All':
            bands = all_bands
        else:
            for band in bands:

                if band in all_bands:
                    pass
                else:
                    raise Exception(
                        "your 'bands' list should contain bands among these ones: 'sw_w2', 'sw_m2', 'sw_w1', 'sw_uu', 'sw_bb', 'ztf_g', 'ztf_r'")
        #plots.plot_models(self.name, self.tde_dir, float(self.z), bands)
        plots.plot_BB_evolution(self, model, from_peak=from_peak, title=title, show=show)

    def xrt_reduction(self, sci_radius=None, bkg_radius=None):
        if sci_radius is None:
            sci_radius = 30
        if bkg_radius is None:
            bkg_radius = 100

        os.chdir(self.sw_dir)
        os.system('ls -d 0* >> datadirs.txt')
        dirs = [line.rstrip('\n').rstrip('/') for line in open('datadirs.txt')]
        os.system('rm datadirs.txt')
        xtr_red.xrt_descompression(self, dirs)
        xtr_red.gen_xrt_spec(self, sci_radius, bkg_radius, dirs)

