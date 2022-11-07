# Libraries

import warnings
warnings.simplefilter("ignore")

import numpy as np
import numpy
import glob
import logging
import subprocess
from astropy.table import Table
import requests
import os.path
from astropy import constants as const
from astropy.cosmology import Planck15 as cosmo
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits
from scipy.stats import sigmaclip
from scipy.ndimage.morphology import binary_dilation, binary_erosion
import wget

from utils import make_wcs
from mosaic_image import set_zeropoint, do_mosaic

# ================================= #
# ========= unWISE TILES ========== #
# ================================= #
def get_galex_images(ra, dec, size):
    # extend the box by 0.6 degree on each side
    npix = int((size + 0.55 * 2) * 3600)
    w = make_wcs(ra, dec, npix)
    galex_image_table = '/Users/tflowers/CODE/wallaby/galex_images.csv.gz'
    tab = Table.read(galex_image_table, format='ascii')
    coords = SkyCoord(tab['ra_cent'].astype(float),
            tab['dec_cent'].astype(float), unit='deg')
    index = w.footprint_contains(coords)
    return tab[index]


def download_galex_image(tab, outdir):
    """ Download the galex images based on matched table """
    root = 'http://galex.stsci.edu/data'
    for el in tab:
        for tag in ['nuv_filenpath', 'fuv_filenpath']:
            if el[tag] != 'null':
                url = '{}/{}'.format(root, el[tag])
                fname = os.path.basename(el[tag])
                logging.debug('downloading {}'.format(url))
                r = requests.get(url, allow_redirects=True)
                if r.status_code == 200:
                    name = '{}/{}'.format(outdir, fname)
                    with open(name, 'wb') as f:
                        f.write(r.content)
                    logging.debug('file downloaded, now unzip: {}'.format(name))
                    res = subprocess.run(['gzip', '-f', '-d', name])
                    if res.returncode != 0:
                        logging.warning('unzip file failed: {}'.format(name))
                    logging.debug('download success: {}'.format(name))
                else:
                    logging.debug('download failed: {}'.format(name))


def make_galex_weight(flist, suffix='.sub.weight'):
    for fimg in flist:
        logging.debug('make weight map for {}'.format(fimg))
        with fits.open(fimg) as f:
            img = f[0].data
            # make general mask
            index = binary_dilation(img > 0, iterations=25)
            index = binary_erosion(index, iterations=30)
            # reject outer region
            yy, xx = np.indices(img.shape)
            dist = np.sqrt((xx - xx.mean()) ** 2 + (yy - yy.mean()) ** 2)
            index[dist > 0.55 * 3600 / 1.5] = False
            # write file
            hdu = fits.PrimaryHDU(index.astype(int), header=f[0].header)
            hdu.writeto(fimg.replace('.fits', '{}.fits'.format(suffix)),
                    overwrite=True)


def set_galex_background(flist, suffix='.sub', weight_suffix='.sub.weight'):
    out = []
    for f in flist:
        img = fits.getdata(f)
        header = fits.getheader(f)
        wgt = fits.getdata(f.replace(
            '.fits', '{}.fits'.format(weight_suffix)))
        thresh = sigmaclip(img[wgt > 0], 3, 3)[2]
        index = img > thresh
        index = binary_dilation(index, iterations=3)
        index = (~index) * (wgt > 0)
        bb = img[index].mean()
        logging.debug('{}: background = {}'.format(f, bb))

        img = img - bb
        img[wgt <= 0] = 0
        hdu = fits.PrimaryHDU(img, header=header)
        hdu.header['BKG'] = bb
        oname = f.replace('.fits', '{}.fits'.format(suffix))
        hdu.writeto(oname, overwrite=True)
        logging.debug('write to file {}'.format(oname))
        out.append(oname)
    return out


def get_galex_image(ra, dec, size, outname, tmpdir=None, clean=True):
    '''
    ra:       Cutout centre position RA
    
    dec:      Cutout centre position Dec
    
    size:     Cutout image size [degrees]
    
    outname:  Cutout file name, to which nuv or fuv will be appended
    
    tmpdir:   Directory to hold temporary full GALEX tiles used to create cutout
    
    clean:    Whether to delete files from tmpdir
    
    Produces GALEX image cutout
    '''
    # make temporary directory
    if tmpdir is None:
        tmpdir = 'tmp_{}'.format(os.getpid())
    if os.path.exists(tmpdir):
        raise Exception('Temporary directory already exists.')
    os.mkdir(tmpdir)

    # run program
    tab = get_galex_images(ra, dec, size)
    logging.info('matched with {} images'.format(len(tab)))
    download_galex_image(tab, tmpdir)
    npix = int(size * 3600 / 1.5)  # GALEX: 1.5 arcsec/pixel
    for band in ['nuv', 'fuv']:
        oname = '{}-{}.fits'.format(outname, band)
        flist = glob.glob('{}/*-{}d-int.fits'.format(tmpdir, band[0]))
        if len(flist) > 0:
            logging.info('make mosaic from {} image(s)'.format(len(flist)))
            make_galex_weight(flist)
            imglist = set_galex_background(flist)
            if band == 'nuv':
                set_zeropoint(imglist, 20.08, force=True)
            else:
                set_zeropoint(imglist, 18.82, force=True)
            do_mosaic(ra, dec, npix, imglist, oname,
                    backvalue=0, weight=True)
            logging.info('mosaic finished: {}'.format(oname))
        else:
            logging.info('no image in {} band'.format(band))

    # clean
    if clean:
        logging.info('finish. clean the temporary directory')
        shutil.rmtree(tmpdir)
    else:
        logging.info('finish.')


# ================================= #
# =========== CONTSTANTS ========== #
# ================================= #
C_LIGHT  = const.c.to('km/s').value
H0       = cosmo.H(0).value
HI_REST  = 1420.406



# ================================= #
# =========== Switches ============ #
# ================================= #
open_catalogue           = True       # Opens SoFiA source catalogue
do_download_galex        = True       # Runs script to download GALEX images

# ================================= #
# ==== Specify Phase + Release ==== #
# ================================= #
tr_i                     = 6

survey_phase_list        = ['PHASE1', 'PHASE1', 'PHASE1', 
                            'PHASE2', 'PHASE2', 'PHASE2', 'PHASE2']
team_release_list        = ['Hydra_DR1', 'Hydra_DR2', 'NGC4636_DR1', 
                            'NGC4808_DR1', 'NGC5044_DR1', 'NGC5044_DR2', 'NGC5044_DR3']

survey_phase             = survey_phase_list[tr_i]
team_release             = team_release_list[tr_i]

# ================================= #
# ========= File Strings ========== #
# ================================= #
basedir                  = '/Users/tflowers/WALLABY/%s/%s/' % (survey_phase, team_release)
sofia_dir                = basedir + 'SOFIA/'
dataprod_dir             = basedir + 'SOFIA/%s_source_products/' % team_release
panstarrs_dir            = basedir + 'MULTIWAVELENGTH/PANSTARRS/'
wise_dir                 = basedir + 'MULTIWAVELENGTH/WISE/'
galex_dir                = basedir + 'MULTIWAVELENGTH/GALEX/'
parameter_dir            = basedir + 'PARAMETERS/'
hi_products_dir          = basedir + 'HI_DERIVED_PRODUCTS/'              
plots_dir                = basedir + 'PLOTS/'


if open_catalogue:
  print('============ %s ============' % team_release)
  fits_sofia     = sofia_dir + '%s_catalogue.fits' % team_release
  
  # ======== WALLABY PARAMETERS ======== #
  hdu_sofia      = fits.open(fits_sofia)
  data_sofia     = hdu_sofia[1].data
  
  gal_name = []
  for i in range(len(data_sofia['name'])):
    split_name = data_sofia['name'][i][8:]
    gal_name.append(split_name)
    
  galaxies       = np.array(gal_name)         # Galaxy name (e.g. J104059-270456)
  sofia_ra       = data_sofia['ra']           # HI detection RA
  sofia_dec      = data_sofia['dec']          # HI detection Dec
  

print(galaxies)


# ================================= #
# ======== Download GALEX ========= #
# ================================= #
if do_download_galex:
  for i in range(len(galaxies)):
    print(i, galaxies[i])
    galex_gal_dir  = galex_dir + galaxies[i] + '/'
    tmp_dir        = galex_gal_dir + 'tmpdir'
    if not os.path.exists(galex_gal_dir):
      os.system('mkdir %s' % galex_gal_dir)
      base_name      = galex_gal_dir + galaxies[i]
      get_galex_image(sofia_ra[i], sofia_dec[i], 0.15, base_name, clean=False, tmpdir=tmp_dir)
      os.system('rm -rf %s' % tmp_dir)
        








