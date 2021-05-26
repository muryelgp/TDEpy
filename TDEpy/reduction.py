
import json
import os
import random
import urllib.request as r
import warnings
from collections import OrderedDict

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import WCS
from photutils import centroid_sources, centroid_com

warnings.simplefilter('ignore', category=AstropyWarning)
from astropy.coordinates import FK5, SkyCoord
from astropy.io import fits
from astropy.time import Time
import tools as tools


def do_sw_photo(sw_dir, aper_cor):
    bands = ['uu', 'bb', 'vv', 'w1', 'm2', 'w2']
    os.system('ls -d 0* >> datadirs.txt')
    dirs = [line.rstrip('\n').rstrip('/') for line in open('datadirs.txt')]
    os.system('rm datadirs.txt')
    for band in bands:

        for i in range(len(dirs)):
            try:
                os.system('cp *.reg ' + sw_dir + '/' + str(dirs[i]) + '/uvot/image/')
                os.chdir(sw_dir + '/' + str(dirs[i]) + '/uvot/image/')
            except OSError:
                pass
            exists = os.path.isfile('sw' + str(dirs[i]) + 'u' + band + '_sk.img.gz')
            if exists:
                os.system('rm ' + band + '.fits')
                if aper_cor:
                    os.system('uvotsource image=sw' + str(dirs[i]) + 'u' + band + '_sk.img.gz srcreg=source.reg '
                                                                                  'bkgreg=bkg.reg sigma=3 '
                                                                                  'syserr=no centroid=yes '
                                                                                  'apercorr=CURVEOFGROWTH '
                                                                                  'clobber=yes outfile=' + band +
                              '.fits')
                else:
                    os.system('uvotsource image=sw' + str(dirs[i]) + 'u' + band + '_sk.img.gz srcreg=source.reg '
                                                                                  'bkgreg=bkg.reg sigma=3 '
                                                                                  'syserr=no centroid=yes '
                                                                                  'clobber=yes outfile=' + band +
                              '.fits')
            else:
                pass

        os.chdir(sw_dir)


def write_files(tde_dir, sw_dir, ebv):
    extcorr = np.array([5.00, 4.16, 3.16, 6.74, 8.53, 8.14]) * ebv
    bands = ['uu', 'bb', 'vv', 'w1', 'm2', 'w2']

    for band_i, band in enumerate(bands):
        obsid, mjd, ab_mag, ab_mag_err, flux, flux_err, = [], [], [], [], [], []
        os.chdir(sw_dir)
        os.system('ls -d 0* >> datadirs.txt')
        dirs = [line.rstrip('\n').rstrip('/') for line in open('datadirs.txt')]
        os.system('rm datadirs.txt')
        for i in range(len(dirs)):
            try:
                os.chdir(sw_dir + '/' + str(dirs[i]) + '/uvot/image/')
            except OSError:
                continue
            exists = os.path.isfile(band + '.fits')
            obsid.append(str(dirs[i]))
            if exists:
                try:
                    f = fits.open('sw' + str(dirs[i]) + 'u' + band + '_sk.img.gz')
                except:
                    continue
                # get timestamp
                mjdref = float(f[0].header['MJDREFI']) + float(f[0].header['MJDREFF'])
                t = float(f[0].header['TSTART']) / (24. * 60. * 60.)  # convert from seconds to days
                mjd.append(mjdref + t)
                f.close()
                f = fits.open(band + '.fits')
                if f[1].data['AB_MAG'][0] <= f[1].data['AB_MAG_LIM'][0]:
                    ab_mag.append(f[1].data['AB_MAG'][0] - extcorr[band_i])
                    ab_mag_err.append(f[1].data['AB_MAG_ERR'][0])
                    flux.append(f[1].data['FLUX_AA'][0] / (10. ** (-0.4 * extcorr[band_i])))
                    flux_err.append(f[1].data['FLUX_AA_ERR'][0])
                else:
                    ab_mag.append(-99)
                    ab_mag_err.append(-99)
                    flux.append(-99)
                    flux_err.append(-99)
                f.close()
                os.chdir(sw_dir)
            else:
                # append null values
                mjd.append(-1)
                ab_mag.append(-99)
                ab_mag_err.append(-99)
                flux.append(-99)
                flux_err.append(-99)
                os.chdir(sw_dir)

        os.chdir(tde_dir)
        phot_dir = os.path.join(tde_dir, 'photometry')

        try:
            os.chdir(phot_dir)
        except:
            os.mkdir(phot_dir)
            os.chdir(phot_dir)

        obs_dir = os.path.join(phot_dir, 'obs')

        try:
            os.chdir(obs_dir)
        except:
            os.mkdir(obs_dir)
            os.chdir(obs_dir)

        g = open(os.path.join(obs_dir, 'sw_' + band + '.txt'), 'w')
        g.write('#Values already corrected for Galactic extinction for an E(B-V) = ' + str(ebv) + '\n')
        g.write('obsid' + '\t' + 'mjd' + '\t' + 'ab_mag' + '\t' +
                'ab_mag_err' + '\t' + 'flux_dens' + '\t' + 'flux__denserr' + '\n')
        for yy in range(len(mjd)):
            g.write(str(obsid[yy]) + '\t' + '{:.2f}'.format(mjd[yy]) + '\t' + '{:.2f}'.format(ab_mag[yy]) + '\t' + '{:.2f}'.format(
                ab_mag_err[yy]) + '\t' + '{:.2e}'.format(flux[yy]) + '\t' + '{:.2e}'.format(flux_err[yy]) + '\n')
        g.close()


def create_reg(ra, dec, radius, dir):
    os.system('ls -d 0* >> datadirs.txt')
    dirs = [line.rstrip('\n').rstrip('/') for line in open('datadirs.txt')]
    os.system('rm datadirs.txt')
    image = None
    for i in range(len(dirs)):

        try_file = '%s/%s/uvot/image/sw%sum2_sk.img.gz' % (str(dir), str(dirs[i]), str(dirs[i]))
        if os.path.isfile(try_file):
            image = fits.open(try_file)
            break

        try_file = '%s/%s/uvot/image/sw%suw2_sk.img.gz' % (str(dir), str(dirs[i]), str(dirs[i]))
        if os.path.isfile(try_file):
            image = fits.open(try_file)
            break

        try_file = '%s/%s/uvot/image/sw%suw1_sk.img.gz' % (str(dir), str(dirs[i]), str(dirs[i]))
        if os.path.isfile(try_file):
            image = fits.open(try_file)
            break

        try_file = '%s/%s/uvot/image/sw%suuu_sk.img.gz' % (str(dir), str(dirs[i]), str(dirs[i]))
        if os.path.isfile(try_file):
            image = fits.open(try_file)
            break

        try_file = '%s/%s/uvot/image/sw%suvv_sk.img.gz' % (str(dir), str(dirs[i]), str(dirs[i]))
        if os.path.isfile(try_file):
            image = fits.open(try_file)
            break

        try_file = '%s/%s/uvot/image/sw%subb_sk.img.gz' % (str(dir), str(dirs[i]), str(dirs[i]))
        if os.path.isfile(try_file):
            image = fits.open(try_file)
            break

        else:
            pass

    w = WCS(image[1].header)
    img = image[1].data
    y, x = np.indices(np.shape(img))

    # Creating source region

    x_source, y_source = w.world_to_pixel(SkyCoord(ra=float(ra), dec=float(dec), unit=u.deg, frame=FK5))
    new_x_source, new_y_source = centroid_sources(img, x_source, y_source, box_size=21,
                                                  centroid_func=centroid_com)

    with open(dir + '/' + 'source.reg', "w") as text_file:
        text_file.write('fk5;circle(%.6f, %.6f, %.1f") # color=green' % (ra, dec, np.round(radius[0], 1)))

    # Creating background region
    rho = random.randrange(0, 300, 1)
    phi = random.randrange(0, 360, 1)
    x_0 = int(rho * np.cos(phi / 180 * np.pi)) + img.shape[1] / 2
    y_0 = int(rho * np.sin(phi / 180 * np.pi)) + img.shape[0] / 2

    r_map = np.sqrt((x - x_0) ** 2 + (y - y_0) ** 2)
    bkg_reg = r_map < radius[1]
    max_bkg = np.max(img[bkg_reg])
    std_bkg = np.std(img[bkg_reg][img[bkg_reg] != 0])
    median_bkg = np.median(img[bkg_reg][img[bkg_reg] != 0])

    while (max_bkg > 5 * (median_bkg + std_bkg)) or (
            ((x_0 - x_source) ** 2 + (y_0 - y_source) ** 2) < radius[1] ** 2):
        rho = random.randrange(0, 300, 1)
        phi = random.randrange(0, 360, 1)
        x_0 = int(rho * np.cos(phi / 180 * np.pi)) + img.shape[1] / 2
        y_0 = int(rho * np.sin(phi / 180 * np.pi)) + img.shape[0] / 2
        r_map = np.sqrt((x - x_0) ** 2 + (y - y_0) ** 2)
        bkg_reg = r_map < radius[1]
        max_bkg = np.max(img[bkg_reg])
        std_bkg = np.std(img[bkg_reg][img[bkg_reg] != 0])
        median_bkg = np.median(img[bkg_reg][img[bkg_reg] != 0])

    bkg_coords = w.pixel_to_world(x_0, y_0)
    image.close()
    with open(dir + '/' + 'bkg.reg', "w") as text_file:
        text_file.write('fk5;circle(%.6f, %.6f, %.1f") # color=green' % (
            bkg_coords.ra.deg, bkg_coords.dec.deg, np.round(radius[1], 1)))
    fig, ax = plt.figure(), plt.subplot(projection=w)
    plt.ioff()
    plot = ax.imshow(img, vmin=0, vmax=8 * (median_bkg + std_bkg), origin='lower')
    plt.colorbar(plot, ax=ax)
    c_b = plt.Circle((x_0, y_0), radius[1], color='red', fill=False)
    c_s = plt.Circle((new_x_source, new_y_source), radius[0], color='red', fill=False)
    ax.add_patch(c_s)
    ax.add_patch(c_b)
    plt.savefig(dir + '/regions.png', bbox_inches='tight')
    plt.show()


def download_ztfdata_lasair(ztf_name):
    # This is the url that contains the data
    url = "https://lasair.roe.ac.uk/object/" + ztf_name + "/json/"
    # Access the webpage and read contents
    response = r.urlopen(url)
    data = json.loads(response.read())
    cwd_tde = os.getcwd()
    # Dump data to ascii for easierand/offline access
    with open(os.path.join(cwd_tde, 'ztf', ztf_name + '.json'), 'w') as outfile:
        json.dump(data, outfile)


def load_ztfdata(ztf_name, ebv):
    cwd_tde = os.getcwd()
    # Read offline copy of json file
    with open(os.path.join(cwd_tde, 'ztf', ztf_name + '.json')) as json_file:
        data = json.load(json_file)

    mjd_r, mag_r, mage_r, flux_r, fluxe_r = [], [], [], [], []
    mjd_g, mag_g, mage_g, flux_g, fluxe_g = [], [], [], [], []

    try:
        print('There is ' + str(len(data['candidates'])) + ' ZTF observations for this source')
    except:
        return
    # extinction corrector factor [g, r]
    ext_corr = np.array([3.60, 2.70]) * ebv
    wl_o = np.array([4866, 6215])
    for i in range(len(data['candidates'])):
        try:
            if data['candidates'][i]['dc_mag_r02'] == -1:
                mjd_g.append(data['candidates'][i]['mjd'])
                mag_g.append(data['candidates'][i]['magpsf'] - ext_corr[0])
                mage_g.append(data['candidates'][i]['sigmapsf'])
                flux_g.append(tools.mag_to_flux(data['candidates'][i]['magpsf'] - ext_corr[0], wl_o[0]))
                fluxe_g.append(
                    tools.mag_to_flux(data['candidates'][i]['magpsf'] - ext_corr[0], wl_o[0]) - tools.mag_to_flux(
                        data['candidates'][i]['magpsf'] - ext_corr[0] + data['candidates'][i]['sigmapsf'], wl_o[0]))
            elif data['candidates'][i]['dc_mag_g02'] == -1:
                mjd_r.append(data['candidates'][i]['mjd'])
                mag_r.append(data['candidates'][i]['magpsf'])
                mage_r.append(data['candidates'][i]['sigmapsf'] - ext_corr[1])
                flux_r.append(tools.mag_to_flux(data['candidates'][i]['magpsf'] - ext_corr[0], wl_o[1]))
                fluxe_r.append(
                    tools.mag_to_flux(data['candidates'][i]['magpsf'] - ext_corr[0], wl_o[1]) - tools.mag_to_flux(
                        data['candidates'][i]['magpsf'] - ext_corr[1] + data['candidates'][i]['sigmapsf'], wl_o[1]))
        except:
            pass

    try:
        os.path.join(cwd_tde, 'photometry')
    except:
        pass
    try:
        os.mkdir(os.path.join(cwd_tde, 'photometry', 'host_sub'))
    except:
        pass

    ztf_g = open(os.path.join(cwd_tde, 'photometry', 'host_sub', 'ztf_g.txt'), 'w')
    ztf_g.write('#Values corrected for Galactic extinction and free from host contribution\n')
    ztf_g.write('mjd' + '\t' + 'ab_mag' + '\t' + 'ab_mag_err' + '\t' + 'flux_dens' + '\t' + 'flux_flux_denserr' + '\n')
    for yy in range(len(mjd_g)):
        ztf_g.write('{:.2f}'.format(mjd_g[yy]) + '\t' + '{:.2f}'.format(mag_g[yy]) + '\t' + '{:.2f}'.format(
            mage_g[yy]) + '\t' + '{:.2e}'.format(flux_g[yy]) + '\t' + '{:.2e}'.format(fluxe_g[yy]) + '\n')
    ztf_g.close()

    ztf_r = open(os.path.join(cwd_tde, 'photometry', 'host_sub', 'ztf_r.txt'), 'w')
    ztf_r.write('#Values corrected for Galactic extinction and free from host contribution\n')
    ztf_r.write('mjd ab_mag ab_mag_err \n')
    for yy in range(len(mjd_r)):
        ztf_r.write('{:.2f}'.format(mjd_r[yy]) + '\t' + '{:.2f}'.format(mag_r[yy]) + '\t' + '{:.2f}'.format(
            mage_r[yy]) + '\t' + str(flux_r[yy]) + '\t' + str(fluxe_r[yy]) + '\n')
    ztf_r.close()


def search_tns(at_name):
    tns = "www.wis-tns.org"
    url_tns_api = "https://" + tns + "/api/get"
    api_key = "ef41397b0519fe63719ae40b6b8c97a6e65a5e17"
    YOUR_BOT_ID = 2409
    YOUR_BOT_NAME = "Muryel Guolo Pereira"

    json_list = [("objname", str(at_name)), ("objid", ""), ("photometry", "0"), ("spectra", "0")]
    # url for get obj
    get_url = url_tns_api + '/object'
    # change json_list to json format
    json_file = OrderedDict(json_list)
    # construct a dictionary of api key data and get obj data
    get_data = {'api_key': api_key, 'data': json.dumps(json_file)}
    headers = {'User-Agent': 'tns_marker{"tns_id":' + str(YOUR_BOT_ID) + ', "type":"user",' \
                                                                         ' "name":"' + YOUR_BOT_NAME + '"}'}
    # get obj using request module
    response = requests.post(get_url, headers=headers, data=get_data)
    # return response
    if None not in response:
        # Here we just display the full json data as the response
        parsed = json.loads(response.text, object_pairs_hook=OrderedDict)
        json_data = json.dumps(parsed, indent=4)
        json_string = json.loads(json_data)
        try:
            reply = json_string['data']['reply']
            ra = reply['radeg']
            dec = reply['decdeg']
            print(str(reply['name_prefix']) + str(reply['objname']) + ' was found at Ra: ' + '{:.6f}'.format(ra) +
                  ' and Dec: ' + '{:.6f}'.format(dec))
            discovery_time = reply['discoverydate']
            discovery_time = Time(discovery_time, format='iso', scale='utc').mjd
            z = reply['redshift']
            host_name = reply['hostname']
            other_names = reply['internal_names']
            i = other_names.find('ZTF')
            f = other_names[i:].find(', ')
            if (f == -1) and (i != -1):
                ztf_name = other_names[i:]
            elif (f == -1) and (i == -1):
                ztf_name = ''
            else:
                ztf_name = other_names[i:(i + f)]

            if ztf_name == '':
                other_name = reply['internal_names']
            else:
                other_name = ztf_name

            if type(z) == 'NoneType':
                z = None
            if type(host_name) == 'NoneType':
                host_name = None

            return ra, dec, z, host_name, other_name, discovery_time

        except Exception:
            print(str(json_string['data']['reply']['name']['110']['message'])[:-1] + ' for ' + str(object))
            return None
    else:
        print(response[1])
        return None


def get_target_id(name, ra, dec):
    try:
        dfs = pd.read_html('https://www.swift.ac.uk/swift_portal/getobject.php?name=' +
                           str(ra) +
                           '%2C+' +
                           str(dec) +
                           '&submit=Search+Names',
                           header=0)[0]
    except:
        raise Exception('No Swift target found at Ra:' + str(ra) + ' and Dec:' + str(dec))


    t_id_list = [str(dfs['Target ID'][i]) for i in range(len(dfs['Target ID']))]
    n_obs_list = [str(dfs['Number of observations'][i]) for i in range(len(dfs['Number of observations']))]
    name_list = [str(dfs['Name'][i]) for i in range(len(dfs['Name']))]
    if len(t_id_list) > 1:
        t_id_list = t_id_list[0:-1]
        n_obs_list = (np.array(n_obs_list[0:-1]))


    print('Target ID, Number of observations:')
    for i in range(len(t_id_list)):
        print(t_id_list[i], ',', n_obs_list[i])



    return name_list, t_id_list, n_obs_list


def download_swift(target_id, n_obs, init, end=None):
    if end is None:
        end = n_obs
    for i in range(init, (end + 1)):
        print('[' + str(i) + '/' + str(n_obs) + ']')
        os.system('wget -q -nv -nc -w 2 -nH --cut-dirs=2 -r --no-parent --reject "index.html*" '
                  'http://www.swift.ac.uk/archive/reproc/' + str(target_id) + str(int(i)).rjust(3,
                                                                                                '0') + '/uvot'
                                                                                                       '/image/')
