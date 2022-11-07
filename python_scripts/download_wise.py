# Libraries

import warnings
warnings.simplefilter("ignore")

import numpy as np
import numpy
import glob
import logging
from astropy.table import Table
import requests
import os.path
from astropy import constants as const
from astropy.cosmology import Planck15 as cosmo
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits
import wget

from utils import make_wcs
from mosaic_image import set_zeropoint, do_mosaic

# ================================= #
# ========= unWISE TILES ========== #
# ================================= #
def get_unwise_tiles(ra, dec, size):
    wise_tile_table = '/Users/tflowers/CODE/wallaby/unwise_tiles.fits.gz' 
    tab = Table.read(wise_tile_table)
    npix = int(size * 3600)
    w = make_wcs(ra, dec, npix)
    a, d = w.wcs_pix2world([1, 1, npix, npix], [1, npix, 1, npix], 1)
    corners = SkyCoord(a, d, unit='deg')

    # cutout region corners contained by tile
    tab['flag1'] = np.zeros(len(tab), dtype=bool)
    for i in range(len(tab)):
        ww = make_wcs(tab[i]['ra'], tab[i]['dec'], 2048, pixsize=2.75)
        index = ww.footprint_contains(corners)
        tab[i]['flag1'] = index.sum() > 0

    # tile corners contained by cutout region
    coords = SkyCoord(tab['c1_ra'], tab['c1_dec'], unit='deg')
    index = w.footprint_contains(coords)
    coords = SkyCoord(tab['c2_ra'], tab['c2_dec'], unit='deg')
    index = np.logical_or(index, w.footprint_contains(coords))
    coords = SkyCoord(tab['c3_ra'], tab['c3_dec'], unit='deg')
    index = np.logical_or(index, w.footprint_contains(coords))
    coords = SkyCoord(tab['c4_ra'], tab['c4_dec'], unit='deg')
    index = np.logical_or(index, w.footprint_contains(coords))
    tab['flag2'] = index

    index = np.logical_or(tab['flag1'], tab['flag2'])
    return tab[index]

# ================================= #
# ========= unWISE IMAGE ========== #
# ================================= #
def download_unwise_image(tab, outdir):
    """ Download the unwise images based on matched tile table """

    for el in tab:
        # W1/W2: neowise
        root = 'http://unwise.me/data/neo6/unwise-coadds/fulldepth'
        for band in ['w1', 'w2']:
            fname = 'unwise-{}-{}-img-m.fits'.format(el['coadd_id'], band)
            url = '{}/{}/{}/{}'.format(
                root, el['coadd_id'][0:3], el['coadd_id'], fname)
            logging.debug('downloading {}'.format(url))
            r = requests.get(url, allow_redirects=True)
            if r.status_code == 200:
                name = '{}/{}'.format(outdir, fname)
                with open(name, 'wb') as f:
                    f.write(r.content)
                logging.debug('download success: {}'.format(name))
            else:
                logging.debug('download failed: {}'.format(name))

        # W3/W4: allwise
        root = 'http://unwise.me/data/allwise/unwise-coadds/fulldepth'
        for band in ['w3', 'w4']:
            fname = 'unwise-{}-{}-img-m.fits'.format(el['coadd_id'], band)
            url = '{}/{}/{}/{}'.format(
                root, el['coadd_id'][0:3], el['coadd_id'], fname)
            logging.debug('downloading {}'.format(url))
            r = requests.get(url, allow_redirects=True)
            if r.status_code == 200:
                name = '{}/{}'.format(outdir, fname)
                with open(name, 'wb') as f:
                    f.write(r.content)
                logging.debug('download success: {}'.format(name))
            else:
                logging.debug('download failed: {}'.format(name))

# ================================= #
# ========== GET/MOSAIC =========== #
# ================================= #
def get_unwise_image(ra, dec, size, outname, tmpdir=None, clean=True):
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
    tab = get_unwise_tiles(ra, dec, size)
    logging.info('matched with {} tiles'.format(len(tab)))
    npix = int(size * 3600 / 2.75)  # unWISE: 2.75 arcsec/pixel
    for band in ['w1', 'w2', 'w3', 'w4']:
        logging.info('processing {}'.format(band))
        download_unwise_image(tab, tmpdir)
        oname = '{}-{}.fits'.format(outname, band)
        flist = glob.glob('{}/unwise-*-{}-img-m.fits'.format(tmpdir, band))
        if len(flist) > 0:
            logging.info('make mosaic from {} image(s)'.format(len(flist)))
            set_zeropoint(flist, 22.5, key='MAGZP')  # zero point = 22.5 mag
            do_mosaic(ra, dec, npix, flist, oname, backsize=256)
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
open_catalogue            = True       # Opens SoFiA source catalogue
do_download_wise          = True       # Runs script to download WISE images

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
# ========= Download WISE ========= #
# ================================= #
if do_download_wise:
  for i in range(len(galaxies)):
    print(i, galaxies[i])
    wise_gal_dir   = wise_dir + galaxies[i] + '/'
    tmp_dir        = wise_gal_dir + 'tmpdir'
    if not os.path.exists(wise_gal_dir):
      os.system('mkdir %s' % wise_gal_dir)
      base_name      = wise_gal_dir + galaxies[i]
      get_unwise_image(sofia_ra[i], sofia_dec[i], 0.25, base_name, clean=False, tmpdir=tmp_dir)
      os.system('rm -rf %s' % tmp_dir)
        







