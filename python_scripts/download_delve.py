# Libraries

import warnings
warnings.simplefilter("ignore")

import numpy as np
import numpy
import astropy.io.fits as pyfits
import os.path
from astropy import constants as const
from astropy.cosmology import Planck15 as cosmo
from astropy.wcs import WCS
from astropy.io import fits
import wget
import warnings  

from numpy.core.defchararray import startswith

from pyvo.dal import sia
from astropy.utils.data import download_file

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
delve_dir                = basedir + 'MULTIWAVELENGTH/DELVE/'
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
  # ===== Replace 'nsa' or 'coadd_all' with the name ===== #
  # ======== of the DELVE data release repository ======== #
  
  #DEF_ACCESS_URL = "https://datalab.noirlab.edu/sia/nsa"
  DEF_ACCESS_URL = "https://datalab.noirlab.edu/sia/coadd_all"
  svc = sia.SIAService(DEF_ACCESS_URL)
  
  do_bespoke_size = True
  #bands = ['g', 'r', 'i', 'z', 'y']
  bands = ['g', 'r']
  for i in range(len(galaxies)):
    #if galaxies[i] == 'J101035-254920':
    if i > -1:
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
          # === Download DECAM FITS Image === #
          fitsurl   = get_url_delve(sofia_ra[i], sofia_dec[i], svc = svc, fov = size, band = bands[j])
          print(fitsurl)
          for k in range(len(fitsurl)):
            wget.download(fitsurl[k], fdownload + '_%i' % k)
          
          
