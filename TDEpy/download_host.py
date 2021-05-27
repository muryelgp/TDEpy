import tools as tools
from astroquery.vizier import Vizier
from astroquery.simbad import Simbad
import astropy.units as units
from astropy.io import fits
import numpy as np
import gPhoton.gAperture
import os
import reduction as reduction


def download_mir(coords_host, host_file_path):
    host_file = open(host_file_path, 'a')

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
            host_file.write(tools.format_host_photo('W4', 221940, W4, e_W4, 'WISE', 'Model'))
        if np.isfinite(W3 * e_W3):
            host_file.write(tools.format_host_photo('W3', 120820, W3, e_W3, 'WISE', 'Model'))
        if np.isfinite(W2 * e_W2):
            host_file.write(tools.format_host_photo('W2', 46180, W2, e_W2, 'WISE', 'Model'))
        if np.isfinite(W1 * e_W1):
            host_file.write(tools.format_host_photo('W1', 33500, W1, e_W1, 'WISE', 'Model'))
    except:
        print('No WISE data found')
        pass

    host_file.close()


def download_nir(aperture, coords_host, host_file_path):
    host_file = open(host_file_path, 'a')
    # Searching for Wise data
    if aperture == 'default/Petro':
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
                host_file.write(tools.format_host_photo('K', 21874, K, e_K, 'UKIDSS', 'Petrosian'))
            if np.isfinite(H):
                host_file.write(tools.format_host_photo('H', 16206, H, e_H, 'UKIDSS', 'Petrosian'))
            if np.isfinite(J):
                host_file.write(tools.format_host_photo('J', 12418, J, e_J, 'UKIDSS', 'Petrosian'))
            if np.isfinite(Y):
                host_file.write(tools.format_host_photo('Y', 10170, Y, e_Y, 'UKIDSS', 'Petrosian'))

        except:
            print('No UKIDSS data found')
            pass

        # Searching for 2MASS data if UKIDSS data were not found
        if len(result) == 0:
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
                    host_file.write(tools.format_host_photo('Ks', 21590, Ks, e_Ks, '2MASS', 'default'))
                if np.isfinite(H):
                    host_file.write(tools.format_host_photo('H', 16620, H, e_H, '2MASS', 'default'))
                if np.isfinite(J):
                    host_file.write(tools.format_host_photo('J', 12350, J, e_J, '2MASS', 'default'))
            except:
                print('No 2MASS Data found')
                pass
    elif aperture == 'standard/PSF':
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
                host_file.write(tools.format_host_photo('K', 21874, K, e_K, 'UKIDSS', 'PSF'))
            if np.isfinite(H):
                host_file.write(tools.format_host_photo('H', 16206, H, e_H, 'UKIDSS', 'PSF'))
            if np.isfinite(J):
                host_file.write(tools.format_host_photo('J', 12418, J, e_J, 'UKIDSS', 'PSF'))
            if np.isfinite(Y):
                host_file.write(tools.format_host_photo('Y', 10170, Y, e_Y, 'UKIDSS', 'PSF'))

        except:
            print('No UKIDSS data found')
            pass

        # Searching for 2MASS data if UKIDSS data were not found
        if len(result) == 0:
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
                    host_file.write(tools.format_host_photo('Ks', 21590, Ks, e_Ks, '2MASS', 'standard'))
                if np.isfinite(H):
                    host_file.write(tools.format_host_photo('H', 16620, H, e_H, '2MASS', 'standard'))
                if np.isfinite(J):
                    host_file.write(tools.format_host_photo('J', 12350, J, e_J, '2MASS', 'standard'))
            except:
                print('No 2MASS Data found')
                pass
    host_file.close()


def download_opt(aperture, coords_host, host_file_path):
    host_file = open(host_file_path, 'a')
    dec_host = coords_host.dec.deg

    if aperture == 'Kron/Petro':
        print('Searching SDSS data...')
        v = Vizier(
            columns=['RAJ2000', 'DEJ2000', 'petroMag_g', 'petroMagErr_g', 'petroMag_r', 'petroMagErr_r', 'petroMag_i', 'petroMagErr_i', 'petroMag_z', 'petroMagErr_z', "+_r"])
        result_sdss = v.query_region(coords_host, radius=0.0014 * units.deg, catalog=['V/147/sdss12'])
        try:
            obj = result_sdss[0]

            z, i, r, g = obj['zPmag'][0], obj['iPmag'][0], obj['rPmag'][0], obj['gPmag'][0]
            e_z, e_i, e_r, e_g = obj['e_zPmag'][0], obj['e_iPmag'][0], obj['e_rPmag'][0], \
                                      obj['e_gPmag'][0]

            if np.isfinite(z):
                host_file.write(tools.format_host_photo('z', 8932, z, e_z, 'SDSS', 'Petrosian'))
            if np.isfinite(i):
                host_file.write(tools.format_host_photo('i', 7480, i, e_i, 'SDSS', 'Petrosian'))
            if np.isfinite(r):
                host_file.write(tools.format_host_photo('r', 6166, r, e_r, 'SDSS', 'Petrosian'))
            if np.isfinite(g):
                host_file.write(tools.format_host_photo('g', 4686, g, e_g, 'SDSS', 'Petrosian'))

        except:
            print('No SDSS Data found')
            pass

        if len(result_sdss) == 0:
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
                        host_file.write(tools.format_host_photo('y', 9633, y, e_y, 'PAN-STARRS', 'Kron'))
                    if np.isfinite(z):
                        host_file.write(tools.format_host_photo('z', 8679, z, e_z, 'PAN-STARRS', 'Kron'))
                    if np.isfinite(i):
                        host_file.write(tools.format_host_photo('i', 7545, i, e_i, 'PAN-STARRS', 'Kron'))
                    if np.isfinite(r):
                        host_file.write(tools.format_host_photo('r', 6215, r, e_r, 'PAN-STARRS', 'Kron'))
                    if np.isfinite(g):
                        host_file.write(tools.format_host_photo('g', 4866, g, e_g, 'PAN-STARRS', 'Kron'))

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
                        host_file.write(tools.format_host_photo('Y', 10305, Y, e_Y, 'DES', 'Kron'))
                    if np.isfinite(z):
                        host_file.write(tools.format_host_photo('z', 8660, z, e_z, 'DES', 'Kron'))
                    if np.isfinite(i):
                        host_file.write(tools.format_host_photo('i', 7520, i, e_i, 'DES', 'Kron'))
                    if np.isfinite(r):
                        host_file.write(tools.format_host_photo('r', 6170, r, e_r, 'DES', 'Kron'))
                    if np.isfinite(g):
                        host_file.write(tools.format_host_photo('g', 4810, g, e_g, 'DES', 'Kron'))
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
                            host_file.write(tools.format_host_photo('z', 9091, z, e_z, 'SkyMapper', 'Petrosian'))
                        if np.isfinite(i):
                            host_file.write(tools.format_host_photo('i', 7712, i, e_i, 'SkyMapper', 'Petrosian'))
                        if np.isfinite(r):
                            host_file.write(tools.format_host_photo('r', 6040, r, e_r, 'SkyMapper', 'Petrosian'))
                        if np.isfinite(g):
                            host_file.write(tools.format_host_photo('g', 4968, g, e_g, 'SkyMapper', 'Petrosian'))
                        if np.isfinite(v):
                            host_file.write(tools.format_host_photo('v', 3870, v, e_v, 'SkyMapper', 'Petrosian'))
                    except:
                        print('No SkyMapper Data found')
                        pass

        # Searching SDSS u band photometry
        v = Vizier(
            columns=['RAJ2000', 'DEJ2000', 'petroMag_u', 'petroMagErr_u', "+_r"])
        result = v.query_region(coords_host, radius=0.0014 * units.deg, catalog=['V/147/sdss12'])
        try:
            obj = result[0]
            u, e_u = obj['uPmag'][0], obj['e_uPmag'][0]
            if np.isfinite(u):
                host_file.write(tools.format_host_photo('u', 3551, u, e_u, 'SDSS', 'Petrosian'))
            else:
                print('No SDSS Data found')
                try:
                    if np.isfinite(u):
                        host_file.write(tools.format_host_photo('u', 3551, u, e_u, 'SkyMapper', 'Petrosian'))
                except:

                    pass
        except:
           pass


    elif aperture == 'PSF':
        print('Searching SDSS data...')
        v = Vizier(
            columns=['RAJ2000', 'DEJ2000', 'gpmag', 'e_gpmag', 'rpmag', 'e_rpmag', 'ipmag', 'e_ipmag', 'zpmag', 'e_zpmag',
                     "+_r"])
        result_sdss = v.query_region(coords_host, radius=0.0014 * units.deg, catalog=['V/147/sdss12'])

        try:
            obj = result_sdss[0]

            z, i, r, g = obj['zpmag'][0], obj['ipmag'][0], obj['rpmag'][0], obj['gpmag'][0]
            e_z, e_i, e_r, e_g = obj['e_zpmag'][0], obj['e_ipmag'][0], obj['e_rpmag'][0], \
                                 obj['e_gpmag'][0]

            if np.isfinite(z):
                host_file.write(tools.format_host_photo('z', 8932, z, e_z, 'SDSS', 'Model'))
            if np.isfinite(i):
                host_file.write(tools.format_host_photo('i', 7480, i, e_i, 'SDSS', 'Model'))
            if np.isfinite(r):
                host_file.write(tools.format_host_photo('r', 6166, r, e_r, 'SDSS', 'Model'))
            if np.isfinite(g):
                host_file.write(tools.format_host_photo('g', 4686, g, e_g, 'SDSS', 'Model'))

        except:
            print('No SDSS Data found')
            pass

        if len(result_sdss) == 0:
            if dec_host > -30:
                    print('Searching PAN-STARRS data...')
                    v = Vizier(
                        columns=['RAJ2000', 'DEJ2000', 'objID', 'ymag', 'zmag', 'imag', 'rmag', 'gmag', 'e_ymag',
                                 'e_zmag',
                                 'e_imag', 'e_rmag', 'e_gmag', "+_r"])
                    result = v.query_region(coords_host, radius=0.0014 * units.deg, catalog=['II/349/ps1'])
                    try:
                        obj = result[0]

                        y, z, i, r, g = obj['ymag'][0], obj['zmag'][0], obj['imag'][0], obj['rmag'][0], obj['gmag'][0]
                        e_y, e_z, e_i, e_r, e_g = obj['e_ymag'][0], obj['e_zmag'][0], obj['e_imag'][0], obj['e_rmag'][0], \
                                                  obj['e_gmag'][0]

                        if np.isfinite(y):
                            host_file.write(tools.format_host_photo('y', 9633, y, e_y, 'PAN-STARRS', 'PSF'))
                        if np.isfinite(z):
                            host_file.write(tools.format_host_photo('z', 8679, z, e_z, 'PAN-STARRS', 'PSF'))
                        if np.isfinite(i):
                            host_file.write(tools.format_host_photo('i', 7545, i, e_i, 'PAN-STARRS', 'PSF'))
                        if np.isfinite(r):
                            host_file.write(tools.format_host_photo('r', 6215, r, e_r, 'PAN-STARRS', 'PSF'))
                        if np.isfinite(g):
                            host_file.write(tools.format_host_photo('g', 4866, g, e_g, 'PAN-STARRS', 'PSF'))

                    except:
                        print('No PAN-STARRS Data found')
                        pass

            if dec_host <= -30:
                print('Searching DES data...')
                v = Vizier(
                    columns=['RAJ2000', 'DEJ2000', 'YmagPSF', 'zmagPSF', 'imagPSF', 'rmagPSF', 'gmagPSF', 'e_YmagPSF',
                             'e_zmagPSF', 'e_imagPSF', 'e_rmagPSF', 'e_gmagPSF', "+_r"])
                result = v.query_region(coords_host,
                                        radius=0.0014 * units.deg,
                                        catalog=['II/357/des_dr1'])
                try:
                    obj = result[0]
                    Y, z, i, r, g = obj['YmagPSF'][0], obj['zmagPSF'][0], obj['imagPSF'][0], obj['rmagPSF'][0], \
                                    obj['gmagPSF'][0]
                    e_Y, e_z, e_i, e_r, e_g = obj['e_YmagPSF'][0], obj['e_zmagPSF'][0], obj['e_imagPSF'][0], \
                                              obj['e_rmagPSF'][0], \
                                              obj['e_gmagPSF'][0]
                    if np.isfinite(Y):
                        host_file.write(tools.format_host_photo('Y', 10305, Y, e_Y, 'DES', 'PSF'))
                    if np.isfinite(z):
                        host_file.write(tools.format_host_photo('z', 8660, z, e_z, 'DES', 'PSF'))
                    if np.isfinite(i):
                        host_file.write(tools.format_host_photo('i', 7520, i, e_i, 'DES', 'PSF'))
                    if np.isfinite(r):
                        host_file.write(tools.format_host_photo('r', 6170, r, e_r, 'DES', 'PSF'))
                    if np.isfinite(g):
                        host_file.write(tools.format_host_photo('g', 4810, g, e_g, 'DES', 'PSF'))
                except:
                    print('No DES Data found')
                    pass

                if len(result) == 0:
                    print('Searching SkyMapper data...')
                    v = Vizier(
                        columns=['RAJ2000', 'DEJ2000', 'uPSF', 'e_uPSF', 'vPSF', 'gPSF', 'rPSF', 'iPSF',
                                 'zPSF', 'e_vPSF', 'e_gPSF', 'e_rPSF', 'e_iPSF', 'e_zPSF', "+_r"])
                    result = v.query_region(coords_host, radius=0.0014 * units.deg, catalog=['II/358/smss'])
                    try:
                        obj = result[0]
                        u, z, i, r, g, v = obj['uPSF'][0], obj['zPSF'][0], obj['iPSF'][0], obj['rPSF'][0], \
                                           obj['gPSF'][0], obj['vPSF'][0]
                        e_u, e_z, e_i, e_r, e_g, e_v = obj['e_uPSF'][0], obj['e_zPSF'][0], obj['e_iPSF'][0], \
                                                       obj['e_rPSF'][0], obj['e_gPSF'][0], obj['e_vPSF'][0]

                        if np.isfinite(z):
                            host_file.write(tools.format_host_photo('z', 9091, z, e_z, 'SkyMapper', 'PSF'))
                        if np.isfinite(i):
                            host_file.write(tools.format_host_photo('i', 7712, i, e_i, 'SkyMapper', 'PSF'))
                        if np.isfinite(r):
                            host_file.write(tools.format_host_photo('r', 6040, r, e_r, 'SkyMapper', 'PSF'))
                        if np.isfinite(g):
                            host_file.write(tools.format_host_photo('g', 4968, g, e_g, 'SkyMapper', 'PSF'))
                        if np.isfinite(v):
                            host_file.write(tools.format_host_photo('v', 3870, v, e_v, 'SkyMapper', 'PSF'))
                    except:
                        print('No SkyMapper Data found')
                        pass

        # Searching SDSS u band photometry
        v = Vizier(
            columns=['RAJ2000', 'DEJ2000', 'umag', 'e_umag', "+_r"])
        result = v.query_region(coords_host, radius=0.0014 * units.deg, catalog=['V/147/sdss12'])
        try:
            obj = result[0]
            u, e_u = obj['upmag'][0], obj['e_upmag'][0]
            if np.isfinite(u):
                host_file.write(tools.format_host_photo('u', 3551, u, e_u, 'SDSS', 'PSF'))
            else:
                print('No SDSS Data found')
                try:
                    if np.isfinite(u):
                        host_file.write(tools.format_host_photo('u', 3498, u, e_u, 'SkyMapper', 'PSF'))
                except:
                    pass
        except:
            pass
    host_file.close()


def download_uv(aperture, coords_host, host_dir):
    host_file_path = os.path.join(host_dir, 'host_phot_obs.txt')
    host_file = open(host_file_path, 'a')
    dec_host = coords_host.dec.deg
    ra_host = coords_host.ra.deg

    aper_radius = np.float(aperture)

    # Getting GALEX data
    print('Measuring UV photometry from GALEX data...')
    try:
        r = (aper_radius / 5) * 0.0014
        nuv_data = gPhoton.gAperture("NUV", [ra_host, dec_host], radius=r,
                                     annulus=[r + 0.0001, 0.0050],
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
        r = (aper_radius / 5) * 0.0014
        fuv_data = gPhoton.gAperture('FUV', [ra_host, dec_host], radius=r, annulus=[r + 0.0001, 0.0050],
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
        host_file.write(tools.format_host_photo('NUV', 2304, nuv, e_nuv, 'GALEX', str(int(aper_radius)) + "''"))
    if np.isfinite(fuv):
        host_file.write(tools.format_host_photo('FUV', 1549, fuv, e_fuv, 'GALEX', str(int(aper_radius)) + "''"))

    if (~np.isfinite(nuv)) & (~np.isfinite(fuv)):
        sw_host_dit = os.path.join(host_dir, 'swift_host')
        if os.path.exists(sw_host_dit):
            os.chdir(sw_host_dit)
            reduction.create_reg(ra_host, dec_host, [aper_radius, 50], sw_host_dit)

            os.system('ls -d 0* >> datadirs.txt')
            dirs = [line.rstrip('\n').rstrip('/') for line in open('datadirs.txt')]
            os.system('rm datadirs.txt')

            band = 'w1'
            for i in range(len(dirs)):
                try:
                    os.system('cp *.reg ' + sw_host_dit + '/' + str(dirs[i]) + '/uvot/image/')
                    os.chdir(sw_host_dit + '/' + str(dirs[i]) + '/uvot/image/')
                except OSError:
                    continue
                exists = os.path.isfile('sw' + str(dirs[i]) + 'u' + band + '_sk.img.gz')
                if exists:
                    os.system('uvotsource image=sw' + str(dirs[i]) + 'u' + band + '_sk.img.gz srcreg=source.reg '
                                                                                  'bkgreg=bkg.reg sigma=3 '
                                                                                  'syserr=no centroid=yes '
                                                                                  'clobber=yes outfile=' + band +
                              '.fits')

                    f = fits.open(band + '.fits')
                    if f[1].data['AB_MAG'][0] == 99:
                        w1 = -2.5*np.log10(10**(-0.4*f[1].data['AB_MAG_LIM'][0])/3)
                        e_w1 = np.nan
                        host_file.write(tools.format_host_photo('UVW1', 2684, w1, e_w1, 'Swift/UVOT',  str(int(aper_radius)) + "''"))
                    else:
                        w1 = f[1].data['AB_MAG'][0]
                        e_w1 = f[1].data['AB_MAG_ERR'][0]
                        host_file.write(tools.format_host_photo('UVW1', 2684, w1, e_w1, 'Swift/UVOT',  str(int(aper_radius)) + "''"))
                    f.close()
                    break
                else:
                    continue

            band = 'm2'
            for i in range(len(dirs)):
                exists = os.path.isfile('sw' + str(dirs[i]) + 'u' + band + '_sk.img.gz')
                if exists:
                    os.system('uvotsource image=sw' + str(dirs[i]) + 'u' + band + '_sk.img.gz srcreg=source.reg '
                                                                                  'bkgreg=bkg.reg sigma=3 '
                                                                                  'syserr=no centroid=yes '
                                                                                  'clobber=yes outfile=' + band +
                              '.fits')

                    f = fits.open(band + '.fits')
                    if f[1].data['AB_MAG'][0] == 99:
                        m2 = -2.5*np.log10(10**(-0.4*f[1].data['AB_MAG_LIM'][0])/3)
                        e_m2 = np.nan
                        host_file.write(tools.format_host_photo('UVM2', 2245, m2, e_m2, 'Swift/UVOT',  str(int(aper_radius)) + "''"))
                    else:
                        m2 = f[1].data['AB_MAG'][0]
                        e_m2 = f[1].data['AB_MAG_ERR'][0]
                        host_file.write(tools.format_host_photo('UVM2', 2245, m2, e_m2, 'Swift/UVOT',  str(int(aper_radius)) + "''"))
                    f.close()
                    break
                else:
                    continue

            band = 'w2'
            for i in range(len(dirs)):
                exists = os.path.isfile('sw' + str(dirs[i]) + 'u' + band + '_sk.img.gz')
                if exists:
                    os.system('uvotsource image=sw' + str(dirs[i]) + 'u' + band + '_sk.img.gz srcreg=source.reg '
                                                                                  'bkgreg=bkg.reg sigma=3 '
                                                                                  'syserr=no centroid=yes '
                                                                                  'clobber=yes outfile=' + band +
                              '.fits')

                    f = fits.open(band + '.fits')
                    if f[1].data['AB_MAG'][0] == 99:
                        w2 = -2.5*np.log10(10**(-0.4*f[1].data['AB_MAG_LIM'][0])/3)
                        e_w2 = np.nan
                        host_file.write(tools.format_host_photo('UVW2', 2085, w2, e_w2, 'Swift/UVOT',  str(int(aper_radius)) + "''"))
                    else:
                        w2 = f[1].data['AB_MAG'][0]
                        e_w2 = f[1].data['AB_MAG_ERR'][0]
                        host_file.write(tools.format_host_photo('UVW2', 2085, w2, e_w2, 'Swift/UVOT',  str(int(aper_radius)) + "''"))
                    f.close()
                    break
                else:
                    continue
    host_file.close()
