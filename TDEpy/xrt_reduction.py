import os
import subprocess
from astropy.wcs import WCS
import random
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import QTable
from xspec import *
from astropy.coordinates import FK5, SkyCoord
from photutils import centroid_sources, centroid_com
import matplotlib.pyplot as plt
from xspec import *
from photutils import centroids


def create_regions(detected, image, res_dir_obs, ra, dec, sci_radius, bkg_radius):
    w = WCS(image[0].header)
    img = image[0].data
    plt.imshow(img)


    if detected:
        x_source, y_source = w.world_to_pixel(SkyCoord(ra=float(ra), dec=float(dec), unit="deg", frame=FK5))
        # Creating source region
        new_x_sci, new_y_sci = centroids.centroid_sources(img, x_source, y_source, box_size=30)


        c = plt.Circle((new_x_sci, new_y_sci), radius=10, color='red', fill=False)

        # add circle to plot (gca means "get current axis")
        plt.gca().add_artist(c)


        src_coords = w.pixel_to_world(new_x_sci, new_y_sci)
        #print(src_coords)
        with open(res_dir_obs + '/' + 'sci.reg', "w") as text_file:
            text_file.write('fk5;circle(%.6f, %.6f, %.1f") # color=green' % ((src_coords.ra.degree),
                                                                                     float(src_coords.dec.degree),
                                                                                     np.round(sci_radius, 1)))
        text_file.close()
        x_source, y_source = new_x_sci, new_y_sci
    if not detected:
        with open(res_dir_obs + '/' + 'sci.reg', "w") as text_file:
            text_file.write('fk5;circle(%.6f, %.6f, %.1f") # color=green' % (ra,
                                                                                     dec,
                                                                                     np.round(sci_radius, 1)*2))
        text_file.close()
        x_source, y_source = w.world_to_pixel(SkyCoord(ra=float(ra), dec=float(dec), unit="deg", frame=FK5))
        new_x_sci, new_y_sci = x_source, y_source
    # Creating background region
    y, x = np.indices(np.shape(img))
    rho = random.randrange(0, 300, 1)
    phi = random.randrange(0, 360, 1)
    x_0 = int(rho * np.cos(phi / 180 * np.pi)) + img.shape[1] / 2
    y_0 = int(rho * np.sin(phi / 180 * np.pi)) + img.shape[0] / 2

    r_map = np.sqrt((x - x_0) ** 2 + (y - y_0) ** 2)
    bkg_reg = r_map < 150
    median_bkg = np.median(img[bkg_reg])
    img_median = np.median(img)
    img_std = np.std(img)

    while (median_bkg > img_median + img_std) or (
            ((x_0 - x_source) ** 2 + (y_0 - y_source) ** 2) < 100 ** 2):
        rho = random.randrange(0, 300, 1)
        phi = random.randrange(0, 360, 1)
        x_0 = int(rho * np.cos(phi / 180 * np.pi)) + img.shape[1] / 2
        y_0 = int(rho * np.sin(phi / 180 * np.pi)) + img.shape[0] / 2
        r_map = np.sqrt((x - x_0) ** 2 + (y - y_0) ** 2)
        bkg_reg = r_map < 150
        median_bkg = np.median(img[bkg_reg])

    bkg_coords = w.pixel_to_world(x_0, y_0)
    image.close()
    with open(res_dir_obs + '/' + 'bkg.reg', "w") as text_file:
        text_file.write('fk5;circle(%.6f, %.6f, %.1f") # color=green' % (
            bkg_coords.ra.deg, bkg_coords.dec.deg, 100))
    text_file.close()
    if not np.isfinite(new_x_sci):
        detected = False
    if detected:
        plt.show()
    return detected



def gen_xrt_spec(tde, sci_radius, bkg_radius, dirs):
    sw_dir = tde.sw_dir
    product_dir = os.path.join(tde.sw_dir, 'xrt_product')

    obs_id_list = []
    rate_list = []
    rate_err_list = []
    mjd_list = []

    for obs in dirs:
        obs_product = os.path.join(product_dir, obs)
        arf_path = os.path.join(obs_product, 'arf_pc_' + obs + '.fits')
        sci_path = os.path.join(obs_product, 'sci_spec_pc_' + obs + '.fits')
        bkg_path = os.path.join(obs_product, 'bkg_spec_pc_' + obs + '.fits')
        sci_grp_path = os.path.join(obs_product, 'sci_spec_pc_' + obs + '.fits_grp')
        os.system('rm ' + arf_path)
        os.system('rm ' + sci_path)
        os.system('rm ' + bkg_path)
        os.system('rm ' + sci_grp_path)
        if os.path.exists(arf_path):
            pass
        else:
            os.chdir(obs_product)
            process = subprocess.Popen(["ximage"],
                                       stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       universal_newlines=True,
                                       bufsize=0)
            try:
                img_file = ("sw{o}xpcw3po_sk.img").format(o=str(obs))
                image = fits.open(img_file)
            except:
                os.system('rm ' + obs_product + '/*')
                continue

            process.stdin.write("read " + img_file + "\n")
            process.stdin.write("detect/snr=2.5\n")
            process.stdin.close()
            lines = []
            for line in process.stdout:
                lines.append(line)
            result = (lines[-2])




            if result.strip()[0] == 'N' or result.strip()[0] == 'R':
                print(obs, 'Non-Detection')
                try_file = ("sw{o}xpcw3po_sk.img").format(o=str(obs))
                image = fits.open(try_file)
                detected = False
                create_regions(detected, image, obs_product, tde.ra, tde.dec, sci_radius, bkg_radius)


            else:
                print(len(result))

                # chunks = result.strip().split(' ')
                # x_pixel, y_pixel = float(chunks[4].strip()), float(chunks[7].strip())
                try_file = ("sw{o}xpcw3po_sk.img").format(o=str(obs))
                image = fits.open(try_file)
                detected = True
                check_detected = create_regions(detected, image, obs_product, tde.ra, tde.dec, sci_radius, bkg_radius)
                if check_detected:
                    print(obs, '3 sigma Detection')
                    print(result.strip().split(' '))
                else:
                    print(obs, 'Non-Detection')
                    detected = False

            # EXTRACTION
            print('Extracting ' + obs[-3:] + ' observation.')
            process = subprocess.Popen(["xselect"],
                                       stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       universal_newlines=True,
                                       bufsize=0)

            process.stdin.write("\n")
            process.stdin.write("\n")
            process.stdin.write("set mission SWIFT\n")
            process.stdin.write("read ev sw" + obs + "xpcw3po_cl.evt\n")
            process.stdin.write(".\n")
            process.stdin.write("\n")
            process.stdin.write('filter region sci.reg\n')
            process.stdin.write('extra spec\n')
            process.stdin.write('save spec sci_spec_pc_' + obs + '.fits\n')
            process.stdin.write('clear re\n')
            process.stdin.write('filter region bkg.reg\n')
            process.stdin.write('extra spec\n')
            process.stdin.write('save spec bkg_spec_pc_' + obs + '.fits\n')
            process.stdin.write('exit\n')
            process.stdin.write('no\n')
            print(process.communicate()[0])
            process.stdin.close()

            process = subprocess.Popen(['xrtmkarf'],
                                       stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       universal_newlines=True,
                                       bufsize=0)
            process.stdin.write('sw' + obs + 'xpcw3po_ex.img\n')
            process.stdin.write('sci_spec_pc_' + obs + '.fits\n')
            process.stdin.write("yes\n")
            process.stdin.write('arf_pc_' + obs + '.fits\n')
            process.stdin.write("-1\n")
            process.stdin.write("-1\n")
            print(process.communicate()[0])
            process.stdin.close()

            res_dir_obs = os.path.join(product_dir, obs)


            os.chdir(res_dir_obs)
            print('Binning ' + obs[-3:] + ' spectrum.')
            process = subprocess.Popen(
                ["grppha"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=0)

            process.stdin.write("sci_spec_pc_" + obs + ".fits\n")
            process.stdin.write("sci_spec_pc_" + obs + ".fits_grp\n")
            process.stdin.write(
                "chkey respfile /Users/mguolo/CALDB/data/swift/xrt/cpf/rmf/swxpc0to12s6_20130101v014.rmf\n")
            process.stdin.write("chkey ancrfile arf_pc_" + obs + ".fits\n")
            process.stdin.write("chkey backfile bkg_spec_pc_" + obs + ".fits\n")
            if detected:
                process.stdin.write("group min 1\n")
            process.stdin.write("exit\n")
            print(process.communicate()[0])
            process.stdin.close()

            product_dir = os.path.join(sw_dir, 'xrt_product')



            AllData.clear()
            os.chdir(res_dir_obs)
            s = Spectrum("sci_spec_pc_" + obs + ".fits_grp")
            s.ignore("0.0-0.2")
            mjd = fits.open("sci_spec_pc_" + obs + ".fits_grp")[0].header['MJD-OBS']
            mjd_list.append(mjd)
            obs_id_list.append(obs)

            if detected:
                if (s.rate[0] > 0) and (s.rate[0] > s.rate[1]):
                    rate_list.append(s.rate[0])
                    rate_err_list.append(s.rate[1])
                else:
                    rate_list.append(3 * s.rate[1])
                    rate_err_list.append(np.nan)
            else:
                rate_list.append(3 * s.rate[1])
                rate_err_list.append(np.nan)

    os.chdir(tde.tde_dir)
    try:
        os.mkdir('xrt')
    except:
        pass

    xrt = open(os.path.join(tde.tde_dir, 'xrt', '02_10_keV.txt'), 'w')
    xrt.write('obs_ID' + '\t' + 'mjd' + '\t' + 'rate' + '\t' + 'rate_err' + '\n')
    for yy in range(len(mjd_list)):
        xrt.write(str(obs_id_list[yy]) + '\t' + '{:d}'.format(int(mjd_list[yy])) + '\t' + '{:.2e}'.format(
            rate_list[yy]) + '\t' + '{:.2e}'.format(rate_err_list[yy]) + '\n')
    xrt.close()


def xrt_descompression(tde, dirs):
    sw_dir = tde.sw_dir
    product_dir = os.path.join(tde.sw_dir, 'xrt_product')


    try:
        os.mkdir(product_dir)
    except:
        pass

    for obs in dirs:
        raw_dir = os.path.join(tde.sw_dir, str(obs))
        obs_product = os.path.join(product_dir, obs)

        if os.path.exists(obs_product):
            descompress = False
            pass
        else:
            os.mkdir(obs_product)
            descompress = True



        # Descompressing the data

        if descompress:
            os.system(
                'xrtpipeline indir=' + raw_dir + ' outdir=' + obs_product + ' steminputs=' + obs + ' srcra=object srcdec=object createexpomap=yes')
        else:
            pass

