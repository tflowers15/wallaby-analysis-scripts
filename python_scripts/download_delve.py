# Libraries

import warnings
warnings.simplefilter("ignore")

import math
import sys
import numpy as np
from datetime import datetime
import numpy
from astropy.table import Table
import requests
from PIL import Image
from io import BytesIO
import urllib
import pylab
import matplotlib
import matplotlib as mpl
import astropy.io.fits as pyfits
import os.path
import fileinput
import astropy.stats
from os import system
from matplotlib import rc
from astropy import constants as const
from astropy import units
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import z_at_value, WMAP7
from astropy.coordinates import EarthLocation, SkyCoord, ICRS, Angle, match_coordinates_3d, Distance
from astropy.wcs import WCS, find_all_wcs
from astropy.io import fits
from astropy import units as u
import wget


from getpass import getpass
import warnings  
from astropy.utils.exceptions import AstropyWarning
#warnings.simplefilter('ignore', category=AstropyWarning) # to quiet Astropy warnings

# 3rd party
#import numpy as np
from numpy.core.defchararray import startswith
#import pylab as plt
#import matplotlib

from pyvo.dal import sia
from astropy.utils.data import download_file
from astropy.io import fits
#from astropy.wcs import WCS
from astropy.visualization import make_lupton_rgb

# Data Lab
#from dl import queryClient as qc, storeClient as sc, authClient as ac

# ================================= #
# ======== PanSTARRS Images ======= #
# ================================= #
def download_deepest_image(ra,dec,svc=sia.SIAService('https://datalab.noirlab.edu/sia'),fov=0.1,band='g'):
    imgTable = svc.search((ra,dec), (fov/np.cos(dec*np.pi/180), fov), verbosity=2).to_table()
    
    sel0 = startswith(imgTable['obs_bandpass'].astype(str),band)
    print("The full image list contains", len(imgTable[sel0]), "entries with bandpass="+band)

    sel = sel0 & ((imgTable['proctype'] == 'Stack') & (imgTable['prodtype'] == 'image')) # basic selection
    Table = imgTable[sel] # select
    if (len(Table)>0):
        row = Table[np.argmax(Table['exptime'].data.data.astype('float'))] # pick image with longest exposure time
        url = row['access_url'] # get the download URL
        print ('downloading deepest ' + band + ' image...')
        image = fits.getdata(download_file(url,cache=True,show_progress=False,timeout=120))

    else:
        print ('No image available.')
        image=None
        
    return image


def get_url_delve(ra,dec,svc=sia.SIAService('https://datalab.noirlab.edu/sia'),fov=0.1,band='g'):
    imgTable = svc.search((ra,dec), (fov/np.cos(dec*np.pi/180), fov), verbosity=2).to_table()
    
    sel0 = startswith(imgTable['obs_bandpass'].astype(str),band)
    print("The full image list contains", len(imgTable[sel0]), "entries with bandpass="+band)
    
    #print(imgTable[((imgTable['proctype'] == b'Stack') & (imgTable['prodtype'] == b'image'))])
    
    #print(np.array(imgTable['proctype']))
    
    #print(imgTable['prodtype'])
    
    sel = sel0 & ((imgTable['proctype'] == b'Stack') & (imgTable['prodtype'] == b'image')) # basic selection
    Table = imgTable[sel] # select
    #print(len(Table))
    #print(Table['obs_collection'])
    
    url = []
    
    if (len(Table)>0):
      max_exptime = np.argwhere(Table['exptime'].data.data.astype('float') == np.max(Table['exptime'].data.data.astype('float')))
      max_exptime = max_exptime.flatten()
      for i in range(len(max_exptime)):
        #row = Table[np.argmax(Table['exptime'].data.data.astype('float'))] # pick image with longest exposure time
        row = Table[max_exptime[i]] # pick image with longest exposure time
        url.append(row['access_url'].decode()) # get the download URL
      #print(url)
      #print ('downloading deepest ' + band + ' image...')
      #image = fits.getdata(download_file(url,cache=True,show_progress=False,timeout=120))

    else:
      print ('No image available.')
      url=None
        
    return url


# ================================= #
# =========== CONTSTANTS ========== #
# ================================= #
C_LIGHT  = const.c.to('km/s').value
H0       = cosmo.H(0).value
HI_REST  = 1420.406


# ================================= #
# =========== Switches ============ #
# ================================= #
open_catalogue            = True
do_download_delve         = True

# ================================= #
# ==== Specify Phase + Release ==== #
# ================================= #
tr_i                     = 1

survey_phase_list        = ['PHASE1', 'PHASE1', 'PHASE1', 'PHASE2', 'PHASE2', 'PHASE2']
team_release_list        = ['Hydra_DR1', 'Hydra_DR2', 'NGC4636_DR1', 'NGC4808_DR1', 'NGC5044_DR1', 'NGC5044_DR2']

survey_phase             = survey_phase_list[tr_i]
team_release             = team_release_list[tr_i]

# ================================= #
# ========= File Strings ========== #
# ================================= #
basedir                  = '/Users/tflowers/WALLABY/%s/%s/' % (survey_phase, team_release)
sofia_dir                = basedir + 'SOFIA/'
dataprod_dir             = basedir + 'SOFIA/%s_source_products/' % team_release
panstarrs_dir            = basedir + 'MULTIWAVELENGTH/PANSTARRS/'
delve_dir                = basedir + 'MULTIWAVELENGTH/DELVE_V2/'
wise_dir                 = basedir + 'MULTIWAVELENGTH/unWISE/'
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
    
  galaxies       = np.array(gal_name)
  sofia_ra       = data_sofia['ra']
  sofia_dec      = data_sofia['dec']
  

print(galaxies)


# ================================= #
# ====== Download PanSTARRS ======= #
# ================================= #
if do_download_delve:
  #DEF_ACCESS_URL = "https://datalab.noirlab.edu/sia/nsa"
  DEF_ACCESS_URL = "https://datalab.noirlab.edu/sia/coadd_all"
  svc = sia.SIAService(DEF_ACCESS_URL)
  
  do_bespoke_size = True
  #bands = ['g', 'r', 'i', 'z', 'y']
  bands = ['g', 'r']
  for i in range(len(galaxies)):
    if galaxies[i] == 'J101035-254920':
    #if i > -1:
      print('%i\t%s\t%.2f' % (i, galaxies[i], (100.*(i + 1.)/len(galaxies))))
      delve_gal_dir = delve_dir + galaxies[i] + '/'
      if not os.path.exists(delve_gal_dir):
        os.system('mkdir %s' % delve_gal_dir)
      if do_bespoke_size:
        wallaby_gal_dir = 'WALLABY_' + gal_name[i] + '/'
        fits_file       = dataprod_dir + wallaby_gal_dir + 'WALLABY_' + gal_name[i] + '_mom0.fits.gz'
        f1              = pyfits.open(fits_file)
        hdr             = f1[0].header
        size            = np.max(np.array([hdr['NAXIS1'], hdr['NAXIS2']])) * np.abs(hdr['CDELT1'])
        if size < 0.02:
          size = 0.02
        f1.close()
      else:
        size = 0.1
      for j in range(len(bands)):
        fits_tan  = galaxies[i] + '_' + bands[j] +'.fits'
        fdownload = delve_gal_dir + fits_tan
        if ~os.path.isfile(fdownload):
          #Download PanSTARRS FITS Image
          fitsurl   = get_url_delve(sofia_ra[i], sofia_dec[i], svc = svc, fov = size, band = bands[j])
          print(fitsurl)
          for k in range(len(fitsurl)):
            wget.download(fitsurl[k], fdownload + '_%i' % k)
          ##Convert from TAN to SIN Projection
          #f1              = pyfits.open(fdownload, mode='update')
          #hdr             = f1[0].header
          #hdr['CDELT1']   = hdr['CD1_1']
          #f1.flush()
          


'''
# ================================= #
# ============ Meerkat ============ #
# ================================= #
if do_meerkat:
  print('============ MEERKAT =============')
  basedir            = '/Users/tflowers/MEERKAT/'
  panstarrs_dir      = basedir + 'PANSTARRS/'
  
  fits_cat     = basedir + 'meerkat_marcin_catalogue_original.fits'
  
  hdu_cat      = fits.open(fits_cat)
  data_cat     = hdu_cat[1].data
  
  sofia_id     = data_cat['id']
  sofia_ra     = data_cat['RA2000']
  sofia_dec    = data_cat['DEC2000']
  
  ra           = data_cat['ra']
  dec          = data_cat['dec']
  gal_name     = []
  for i in range(len(data_cat['ra'])):
    split_ra  = data_cat['ra'][i][:2] + data_cat['ra'][i][3:5] + data_cat['ra'][i][6:8]
    split_dec = data_cat['dec'][i][:3] + data_cat['dec'][i][4:6] + data_cat['dec'][i][7:9]
    gal_name.append('J' + split_ra + split_dec)
    
  galaxies   = np.array(gal_name)
  
  table_str  = basedir + 'meerkat_marcin_catalogue.fits'
  tdata      = [sofia_id, galaxies, sofia_ra, sofia_dec]
  tcols      = ('ID', 'NAME', 'RA', 'DEC')
  t          = Table(tdata, names=tcols)
  t.write(table_str, format = 'fits')

# ================================= #
# ====== Download PanSTARRS ======= #
# ================================= #
if do_download_panstarrs_mk:
  bands = ['g', 'r']
  for i in range(len(galaxies)):
    #if galaxies[i] == 'J104059-270456':
    print(galaxies[i])
    panstarrs_gal_dir = panstarrs_dir + galaxies[i] + '/'
    if not os.path.exists(panstarrs_gal_dir):
      os.system('mkdir %s' % panstarrs_gal_dir)
    size = 512
    for j in range(len(bands)):
      fits_tan  = galaxies[i] + '_' + bands[j] +'.fits'
      mir_tan   = galaxies[i] + '_' + bands[j] +'.mir'
      mir_sin   = galaxies[i] + '_' + bands[j] +'.sin.mir'
      fits_sin  = galaxies[i] + '_' + bands[j] +'.sin.fits'
      if os.path.isfile(panstarrs_gal_dir + fits_tan):
        dummy = 0
      else:
        fitsurl   = geturl(sofia_ra[i], sofia_dec[i], size=size, filters=bands[j], format="fits")
        fdownload = panstarrs_gal_dir + fits_tan
        wget.download(fitsurl[0], fdownload)
        os.chdir(panstarrs_gal_dir)
        os.system('pwd')
        os.system('fits in=%s op=xyin out=%s' % (fits_tan, mir_tan))
        os.system('regrid in=%s out=%s project=SIN' % (mir_tan, mir_sin))
        os.system('fits in=%s op=xyout out=%s' % (mir_sin, fits_sin))
        os.system('rm -rf %s' % mir_tan)
        os.system('rm -rf %s' % mir_sin)
        #os.system('rm -rf %s' % fits_tan)
        os.chdir('/Users/tflowers')
        os.system('pwd')
'''


