# Libraries

import warnings
warnings.simplefilter("ignore")

import numpy as np
import numpy
from astropy.table import Table
import astropy.io.fits as pyfits
import os.path
from astropy import constants as const
from astropy.cosmology import Planck15 as cosmo
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits
import wget


# ================================= #
# ======== PanSTARRS Images ======= #
# ================================= #
def getimages(ra,dec,size=240,filters="grizy"):
    """Query ps1filenames.py service to get a list of images
    
    ra, dec = position in degrees
    
    size = image size in pixels (0.25 arcsec/pixel)
    
    filters = string with filters to include
    
    Returns a table with the results
    """
    service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    url = ("{service}?ra={ra}&dec={dec}&size={size}&format=fits"
           "&filters={filters}").format(**locals())
    table = Table.read(url, format='ascii')
    return table

# ================================= #
# ========= PanSTARRS URL ========= #
# ================================= #
def geturl(ra, dec, size=240, output_size=None, filters="grizy", format="jpg", color=False):
    """Get URL for images in the table
    
    ra, dec = position in degrees
    
    size = extracted image size in pixels (0.25 arcsec/pixel)
    
    output_size = output (display) image size in pixels (default = size).
                  output_size has no effect for fits format images.
    
    filters = string with filters to include
    
    format = data format (options are "jpg", "png" or "fits")
    
    color = if True, creates a color image (only for jpg or png format).
            Default is return a list of URLs for single-filter grayscale images.
    
    Returns a string with the URL
    """
    if color and format == "fits":
        raise ValueError("color images are available only for jpg or png formats")
    if format not in ("jpg","png","fits"):
        raise ValueError("format must be one of jpg, png, fits")
    table = getimages(ra,dec,size=size,filters=filters)
    url = ("https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
           "ra={ra}&dec={dec}&size={size}&format={format}").format(**locals())
    if output_size:
        url = url + "&output_size={}".format(output_size)
    # sort filters from red to blue
    flist = ["yzirg".find(x) for x in table['filter']]
    table = table[numpy.argsort(flist)]
    if color:
        if len(table) > 3:
            # pick 3 filters
            table = table[[0,len(table)//2,len(table)-1]]
        for i, param in enumerate(["red","green","blue"]):
            url = url + "&{}={}".format(param,table['filename'][i])
    else:
        urlbase = url + "&red="
        url = []
        for filename in table['filename']:
            url.append(urlbase+filename)
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
open_catalogue            = True       # Opens SoFiA source catalogue
do_download_panstarrs     = True       # Runs script to download PanSTARRS images

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


# ================================= #
# == Open SoFiA Source Catalogue == #
# ================================= #
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
# ====== Download PanSTARRS ======= #
# ================================= #
if do_download_panstarrs:
  do_bespoke_size = True
  #bands = ['g', 'r', 'i', 'z', 'y']
  bands = ['g', 'r']
  for i in range(len(galaxies)):
    print(i, galaxies[i])
    panstarrs_gal_dir = panstarrs_dir + galaxies[i] + '/'
    if ~os.path.exists(panstarrs_gal_dir):
      os.system('mkdir %s' % panstarrs_gal_dir)
    # Define PanSTARRS image size by SoFiA moment 0 map
    if do_bespoke_size:
      wallaby_gal_dir = 'WALLABY_' + gal_name[i] + '/'
      fits_file       = dataprod_dir + wallaby_gal_dir + 'WALLABY_' + gal_name[i] + '_mom0.fits.gz'
      f1              = pyfits.open(fits_file)
      hdr             = f1[0].header
      # Use largest moment 0 map dimension (pixels) and convert to PanSTARRS number of pixels
      size            = int(np.max(np.array([hdr['NAXIS1'], hdr['NAXIS2']])) * np.abs(hdr['CDELT1']) * 3600. / 0.25)
      if size < 240:    # If size <240 pixels set this as the minimum image size
        size = 240
      f1.close()
    else:
      size = 512    # Can set a default image size to use for all sources 
                    # (will be too large/small for resolved/unresolved sources)
    for j in range(len(bands)):
      fits_tan  = galaxies[i] + '_' + bands[j] +'.fits'
      fdownload = panstarrs_gal_dir + fits_tan
      if not os.path.isfile(fdownload):
        #Download PanSTARRS FITS Image
        fitsurl   = geturl(sofia_ra[i], sofia_dec[i], size=size, filters=bands[j], format="fits")
        wget.download(fitsurl[0], fdownload)
        
        # Correction to correctly interpret pixel increment in the RA direction
        f1              = pyfits.open(fdownload, mode='update')
        hdr             = f1[0].header
        hdr['CDELT1']   = hdr['PC001001'] * hdr['CDELT1']
        del hdr['PC001001']
        del hdr['PC001002']
        del hdr['PC002001']
        del hdr['PC002002']
        f1.flush()
        f1.close()
        



