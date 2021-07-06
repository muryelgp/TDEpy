import numpy as np


def mag_to_flux(ab_mag, wl):
    fnu = (10. ** (-0.4 * (48.6 + ab_mag)))
    flam_g = (2.99792458e+18 * fnu) / (wl ** 2.)
    return flam_g


def flux_to_mag(flux, wl):
    mag = -2.5 * np.log10((flux / 2.99792458e+18) * wl ** 2) - 48.6
    return mag


def df_to_dmag(f, df, wl):
    A = 2.5
    B = wl ** 2 / 2.99792458e+18
    dmag = abs(A * df / (f * np.log(10)))
    return dmag


def dmag_to_df(dmag, f):
    df = f * 0.4 * np.log(10) * dmag
    return df


def vega_to_ab(mag, band):
    if band == "W1":
        return mag + 2.699
    elif band == 'W2':
        return mag + 3.339
    if band == "W3":
        return mag + 5.174
    elif band == 'W4':
        return mag + 6.620
    elif band == 'Y':
        return mag + 0.634
    elif band == 'J':
        return mag + 0.940
    elif band == 'H':
        return mag + 1.38
    elif band == 'Ks':
        return mag + 1.86
    elif band == 'K':
        return mag + 1.90
    elif band == 'g':
        return mag - 0.103
    elif band == 'r':
        return mag + 0.146
    elif band == 'i':
        return mag + 0.366
    elif band == 'z':
        return mag + 0.533
    elif band == 'u':
        return mag + 0.927
    else:
        return None


def format_host_photo(band, lambda_0, ab_mag, ab_mag_err, catalog, aperture):
    if not np.isfinite(ab_mag):
        formatted_string = str(band) + '\t' + str(lambda_0) + '\t' + 'nan' + '\t' + 'nan' + '\t' + str(
            catalog) + '\t' + str(aperture) + '\n'
        return formatted_string

    if not np.isfinite(ab_mag_err):
        formatted_string = str(band) + '\t' + str(lambda_0) + '\t' + '{:.3f}'.format(
            ab_mag) + '\t' + 'nan' + '\t' + str(catalog) + '\t' + str(aperture) + '\n'
        return formatted_string
    else:
        formatted_string = str(band) + '\t' + str(lambda_0) + '\t' + '{:.3f}'.format(
            ab_mag) + '\t' + '{:.3f}'.format(ab_mag_err) + '\t' + str(catalog) + '\t' + str(aperture) + '\n'
        return formatted_string


def round_small(array, limit):
    array = np.array(array)
    flag = array < limit
    array[flag] = limit
    return array
