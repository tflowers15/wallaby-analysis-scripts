# Libraries

import warnings
warnings.simplefilter("ignore")

import math
import sys
import numpy as np
from datetime import datetime
import matplotlib
import matplotlib as mpl
#mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid
import matplotlib.patches as mpatches
import astropy.io.fits as pyfits
import shutil
import emcee
import os.path
import fileinput
import astropy.stats
from os import system
from matplotlib import rc
from astropy import constants as const
from astropy import units
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import z_at_value, WMAP7
from astropy.coordinates import EarthLocation, SkyCoord, ICRS, Angle, match_coordinates_3d, Distance
#from astropy.coordinates import EarthLocation, SkyCoord, FK5, ICRS, match_coordinates_3d, Distance
from astropy.wcs import WCS, find_all_wcs
from astropy.io import fits
from astropy import units as u
from astropy.modeling.models import Ellipse2D, Disk2D, Gaussian2D
from astropy.stats import median_absolute_deviation as madev
from astropy.modeling.rotations import Rotation2D
from astropy.io.votable import parse_single_table
from astropy.convolution import convolve, Gaussian1DKernel
from astropy.table import Table, join
from random import shuffle as sfl
#from photutils.centroids import centroid_com
import scipy
from scipy import stats
from scipy.special import erf
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.ndimage.interpolation import rotate as scirotate
from scipy.stats import binned_statistic

import sympy as sp
from sympy.solvers import solve
from sympy import Symbol

import sqlite3
from sqlite3 import OperationalError


#from functions_plot import field_mom_plot, field_pv_plot, pv_plot, rotation_curve_plot, pv_plot_single, rotation_curve_plot_single, askap_mom_plot, askap_contour_plot, spec_panel_plot

#from envelope_tracing import open_fits_file, env_trace, env_trace_field

#from functions_asymmetry import *

from functions_plotting import *
from functions_calculations import *

#cosmo = FlatLambdaCDM(H0=100, Om0=0.3, Tcmb0=2.725)

#def fxn():
    #warnings.warn("deprecated", DeprecationWarning)

#with warnings.catch_warnings():
    #warnings.simplefilter("ignore")
    #fxn()

# ================================= #
# ======== Smoothing Kernel ======= #
# ================================= #
def hann_smooth(width):
    knl        = np.hanning(width)
    knl_sum    = np.sum(knl)
    smooth_knl = knl/knl_sum
    return smooth_knl

# ================================= #
# === Stellar Mass from Colour ==== #
# ================================= #
def calc_lgMstar_6df(mag,Bj,Rf,z,h=cosmo.h):
    distmod = cosmo.distmod(z).value
    lgMstar = 0.48 - 0.57 * (Bj - Rf) + (3.7 - (mag - distmod)) / 2.5
    return(lgMstar)

# ================================= #
# =========== CONTSTANTS ========== #
# ================================= #
C_LIGHT  = const.c.to('km/s').value
H0       = cosmo.H(0).value
#RHO_CRIT = cosmo.critical_density(0).value*100**3/1000
#OMEGA_M  = cosmo.Om(0)
#OMEGA_DE = cosmo.Ode(0)
HI_REST  = 1420.406


# ================================= #
# ==== Remove Fits Header Cards === #
# ================================= #
def remove_hdr_cards(fits_file):
    f1     = pyfits.open(fits_file, mode='update')
    data, hdr  = f1[0].data, f1[0].header
    c_hdr_cards = ['CTYPE4', 'CRVAL4', 'CDELT4', 'CRPIX4', 'CUNIT4', 'PC01_03', 'PC01_04', 'PC02_03', 'PC02_04', 'PC03_01', 'PC03_02', 'PC03_03', 'PC03_04', 'PC04_01', 'PC04_02', 'PC04_03', 'PC04_04', 'PC1_3', 'PC1_4', 'PC2_3', 'PC2_4', 'PC3_1', 'PC3_2', 'PC3_3', 'PC3_4', 'PC4_1', 'PC4_2', 'PC4_3', 'PC4_4']
    for i in range(len(c_hdr_cards)):
      if c_hdr_cards[i] in hdr:
        del hdr[c_hdr_cards[i]]
    f1.flush()
    f1.close()


# ================================= #
# ========== Axes Labels ========== #
# ================================= #
lbl_czhi   = r'c$z_{\mathrm{HI}}$ [km\,s$^{-1}$]'
lbl_czopt  = r'c$z_{\mathrm{opt}}$ [km\,s$^{-1}$]'
lbl_sint   = r'$\log(S_{\mathrm{int}}/\mathrm{Jy})$'
lbl_mstar  = r'$\log(M_*/\mathrm{M}_{\odot})$'
lbl_mhi    = r'$\log(M_{\mathrm{HI}}/\mathrm{M}_{\odot})$'
lbl_hidef  = r'$\mathrm{DEF}_{\mathrm{HI}}$'
lbl_dhi    = r'$d_{\mathrm{HI}}/\mathrm{kpc}$'
lbl_d25    = r'$d_{\mathrm{25}}/\mathrm{kpc}$'
lbl_hifrac = r'$\log(M_{\mathrm{HI}}/M_*)$'
lbl_dlum   = r'$D_{\mathrm{L}}$ [Mpc]'
lbl_aflux  = r'$A_{\mathrm{flux}}$' 
lbl_aspec  = r'$A_{\mathrm{spec}}$'
lbl_dvsys  = r'$\Delta V_{\mathrm{sys}}$ [km\,s$^{-1}$]'
lbl_w20    = r'$w_{20}$ [km\,s$^{-1}$]'
lbl_sep    = r'$R_{\mathrm{proj}}/R_{\mathrm{vir}}$'
lbl_sfr    = r'$\log$(SFR/[M$_{\odot}$\,yr$^{-1}$])'
lbl_nuvr   = r'$\mathrm{NUV}-r$ [Mag]'
lbl_ssfr   = r'$\log$(sSFR/[yr$^{-1}$])'
lbl_sratio = r'$d_{\mathrm{HI}}/d_{25}$'


# ================================= #
# ========= File Strings ========== #
# ================================= #

basedir   = '/Users/tflowers/WALLABY/ERIDANUS/HI_CLOUDS/'
outputdir = basedir + 'OUTPUT/'


# ================================= #
# =========== Switches ============ #
# ================================= #
do_imfit             = False
do_ellint            = False
do_hi_radius         = True
do_plot_prop         = False


files   = ['c1', 'c2']

ra  = ['03h39m12s', '03h37m23s']
dec = ['-22d23m22s', '-23d57m54s']

if do_imfit:
  for i in range(len(files)):
    print(files[i])
    fits_dir      = basedir
    mom0_fits     = files[i] + '_mom0.fits'
    mom0_gz       = mom0_fits + '.gz'
    mom0_mir      = files[i] + '_mom0.mir'
    model_mir     = files[i] + '_mom0_model.mir'
    model_fits    = files[i] + '_mom0_model.fits'
    model_file    = files[i] + '_model.txt'
    os.chdir(fits_dir)
    #os.system('rm -rf %s' % model_mir)
    #os.system('rm -rf %s' % model_fits)
    if os.path.isfile(model_fits):
      dummy = 'skip'
    else:
      #os.system('gunzip < %s > %s' % (mom0_gz, mom0_fits))
      os.system('fits in=%s op=xyin out=%s' % (mom0_fits, mom0_mir))
      os.system('imfit in=%s clip=212 object=gaussian out=%s > GAUSSIAN_MODELS/%s' % (mom0_mir, model_mir, model_file))
      os.system('fits in=%s op=xyout out=%s' % (model_mir, model_fits))
      #os.system('rm -rf %s' % mom0_fits)
      os.system('rm -rf %s' % mom0_mir)
      os.system('rm -rf %s' % model_mir)
    os.chdir('/Users/tflowers')


if do_ellint:
  for i in range(len(files)):
      print(files[i])
      fits_dir      = basedir
      mom0_fits     = files[i] + '_mom0.fits'
      mom0_gz       = mom0_fits + '.gz'
      mom0_mir      = files[i] + '_mom0.mir'
      res_mir       = files[i] + '_mom0_residual.mir'
      profile_file  = files[i] + '_ellint.txt'
      model_file    = basedir + 'GAUSSIAN_MODELS/' + files[i] + '_model.txt'
      os.chdir(fits_dir)
      f1            = pyfits.open(mom0_fits)
      data, hdr     = f1[0].data, f1[0].header
      wcs           = WCS(hdr)
      asec_p_pix    = np.abs(hdr['CDELT2']) * 3600.
      crval1        = hdr['CRVAL1']
      cdelt1        = hdr['CDELT1']
      crpix1        = hdr['CRPIX1']
      crval2        = hdr['CRVAL2']
      cdelt2        = hdr['CDELT2']
      crpix2        = hdr['CRPIX2']
      position      = SkyCoord(ra[i], dec[i], frame='icrs')
      ra_pix, dec_pix = wcs.all_world2pix(position.ra.deg, position.dec.deg, 0)
      ra_diff       = (ra_pix - crpix1) * (hdr['CDELT1'] * 3600.)
      dec_diff      = (dec_pix - crpix2) * (hdr['CDELT2'] * 3600.)
      #incl          = np.arcsin(np.sqrt(1. - k_ba[i]**2)) * 180. / math.pi
      gaus_params = np.genfromtxt(model_file, skip_header=22, max_rows=3, usecols=(3), unpack=True)
      #if gaus_params[0] > 35 and gaus_params[1] > 35:
        #gaus_params[0] = np.sqrt(gaus_params[0]**2 - 30.**2)
        #gaus_params[1] = np.sqrt(gaus_params[1]**2 - 30.**2)
      if gaus_params[0] < gaus_params[1]:
        hi_ba = gaus_params[0]/gaus_params[1]
      else:
        hi_ba = gaus_params[1]/gaus_params[0]
      if gaus_params[2] < 180:
        pa            = gaus_params[2] + 180.
      incl          = np.arcsin(np.sqrt(1. - hi_ba**2)) * 180. / math.pi
      scale_factor  = 1 #2.12 / (30 * 30 * math.pi / 4 / np.log(2)) #/ BEAM #chan_width / chan_width_hz
      #print(chan_width, chan_width_hz, scale_factor)
      os.system('rm -rf %s' % mom0_mir)
      if os.path.isdir(fits_dir + mom0_mir):
        dummy = 'skip'
      else:
        #os.system('gunzip < %s > %s' % (mom0_gz, mom0_fits))
        os.system('fits in=%s op=xyin out=%s' % (mom0_fits, mom0_mir))
        #os.system('rm -rf %s' % mom0_fits)
      #os.system('ellint in=%s center=%.0f,%.0f pa=%.2f incline=%.2f radius=0,500,30 scale=%.4e log=../PROFILES/%s' % (mom0_mir, ra_diff, dec_diff, pa, incl, scale_factor, profile_file))
      os.system('ellint in=%s center=%.0f,%.0f pa=%.2f incline=%.2f scale=%.4e log=PROFILES/%s' % (mom0_mir, ra_diff, dec_diff, pa, incl, scale_factor, profile_file))
      os.chdir('/Users/tflowers')
      #os.system('pwd')



if do_hi_radius:
  hi_sd_major    = np.zeros(len(files))
  hi_ba          = np.zeros(len(files))
  pa             = np.zeros(len(files))
  hi_sd_major[:] = np.nan
  for i in range(len(files)):
    sd_file     = basedir + 'PROFILES/' + files[i] + '_ellint.txt'
    model_file  = basedir + 'GAUSSIAN_MODELS/' + files[i] + '_model.txt'
    if os.path.isfile(sd_file):
      if os.path.isfile(model_file):
        gaus_params = np.genfromtxt(model_file, skip_header=22, max_rows=3, usecols=(3), unpack=True)
        if gaus_params[0] < gaus_params[1]:
          hi_ba[i] = gaus_params[0]/gaus_params[1]
        else:
          hi_ba[i] = gaus_params[1]/gaus_params[0]
        if gaus_params[2] < 180:
          pa[i]            = gaus_params[2] + 90.#180.
      else:
        if sofia_smaj[i] < sofia_smin[i]:
          hi_ba[i] = sofia_smaj[i]/sofia_smin[i]
        else:
          hi_ba[i] = sofia_smin[i]/sofia_smaj[i]
      rad_asec, mass_sd = np.genfromtxt(sd_file, usecols=(0, 2), skip_header=10, unpack=True)
      #mass_sd = mass_sd * 8.01 * 10**-21 * 2.33 * 10**20 / 900. * np.sqrt(1-hi_ba[i]**2) #* 28.3 / 36.
      mass_sd = mass_sd * 8.01 * 10**-21 * 2.33 * 10**20 / 900. * np.cos(np.arcsin(np.sqrt(1-hi_ba[i]**2)))
      func_interp       = interp1d(mass_sd, rad_asec, fill_value='extrapolate')
      hi_sd_major[i]    = func_interp(1.0)
    
  #hi_sd_major[hi_sd_major>40] = np.sqrt((hi_sd_major[hi_sd_major>40])**2 - 30.**2)
  #hi_sd_major[hi_sd_major<30] = np.nan
  hi_sd_major = 2. * hi_sd_major
  #hi_sd_major[hi_sd_major>30] = np.sqrt((hi_sd_major[hi_sd_major>30])**2 - 30.**2)
  hi_sd_major = np.sqrt((hi_sd_major)**2 - 30.**2)
  hi_sd_major = hi_sd_major / 2.
  hi_sd_minor = hi_sd_major * hi_ba
  
  print(2.*hi_sd_major, 2.*hi_sd_minor)





# ================================= #
# ======== Plot Properties ======== #
# ================================= #
if do_plot_prop:
  do_fig1        = True
  
  #cz         = np.array([1878.7, 1468.9])
  #lum_dist   = dist_lum(cz/C_LIGHT)
  lum_dist   = np.array([22.8, 22.8])
  dhi        = np.log10(lum_dist * np.tan((hi_sd_major*2.)/(60.*60.)*math.pi/180.) * 1000.)
  mhi        = np.log10(np.array([2.33*10**8, 1.38*10**8]))
  
  #print(lum_dist)
  print(np.round(dhi,2), np.round(10**dhi,1))
  
  # ================================= #
  # ======= dhi_mhi_scatter ========= #
  # ================================= #
  if do_fig1:
    do_wang2016 = True
    if do_wang2016:
      wang2016_file = '/Users/tflowers/WALLABY/Hydra_DR1/CATALOGUES/wang2016_table.txt'
      wang2016_dhi, wang2016_mhi, wang2016_d25 = np.genfromtxt(wang2016_file, skip_header=17, usecols=(1,2,7), unpack=True)
      wang2016_dhi = np.log10(wang2016_dhi)
    
    fig1 = plt.figure(26, figsize=(3, 3))
    
    gal_col       = 'red'
    
    xlbl          = [lbl_mhi]
    ylbl          = [r'$\log(d_{\mathrm{HI}}/\mathrm{kpc})$']
    
    scat_mean_plot(fig1, 1, 1, 1, wang2016_mhi, wang2016_dhi, False, False, 
                   'Wang+2016', xlbl[0], ylbl[0], 'lightgrey', '.', False)
    
    xpar    = [mhi]
    ypar    = [dhi]
    
    for i in range(len(xpar)):
      if xlbl[i] != False:
        scat_mean_plot(fig1, 1, 1, 1, xpar[i], ypar[i], False, False, 
                   'HI Clouds', xlbl[i], ylbl[i], gal_col, 'o', False)
    
    plot_name = basedir + 'PLOTS/dhi_mhi_scatter.pdf' #_h100
    #plot_name = basedir + dir_field + 'PLOTS/hydra_virgo_sfr.pdf'
    plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    #plt.show()
    plt.clf()


  


  













