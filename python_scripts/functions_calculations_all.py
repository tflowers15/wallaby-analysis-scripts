# Libraries

import math
import cmath
import sys
import numpy as np
import datetime
import matplotlib
import matplotlib as mpl
#mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid
import astropy.io.fits as pyfits
import shutil
import emcee
import os.path
from matplotlib import rc
from astropy import constants as const
from astropy import units
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import z_at_value
from astropy.coordinates import EarthLocation, SkyCoord, ICRS
from astropy.wcs import WCS, find_all_wcs
from astropy.io import fits
from astropy import units as u
from astropy.cosmology import WMAP7
from astropy.stats import median_absolute_deviation as madev
from random import shuffle as sfl
from astropy.convolution import convolve, Gaussian1DKernel
#from photutils.centroids import centroid_com
import scipy.ndimage as ndi
from scipy import stats
from scipy import signal
from scipy.special import erf
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.ndimage.interpolation import rotate as scirotate
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gaussian_kde
from scipy.stats import kde
from specutils import Spectrum1D
from specutils.manipulation import FluxConservingResampler

import sympy as sp
from sympy.solvers import  solve
from sympy import Symbol

import sqlite3
from sqlite3 import OperationalError

#from functions_plot import field_mom_plot, field_pv_plot

#from envelope_tracing import open_fits_file, env_trace

import warnings

#cosmo = FlatLambdaCDM(H0=100, Om0=0.3, Tcmb0=2.725)

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

#import warnings
warnings.filterwarnings("ignore")

rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{newtxmath}']


# ================================= #
# =========== CONTSTANTS ========== #
# ================================= #
C_LIGHT  = const.c.to('km/s').value
H0       = cosmo.H(0).value
#RHO_CRIT = cosmo.critical_density(0).value*100**3/1000
#OMEGA_M  = cosmo.Om(0)
#OMEGA_DE = cosmo.Ode(0)
HI_REST  = 1420.406



# ====================================== Funtions =======================================

# ================================ #
# =========== Calc VOPT ========== #
# ================================ #
def vel_opt(freq):
    return (1420.406/(freq/1e6) - 1) * 299792.458

# ================================ #
# === Compute Local Group Vel ==== #
# ================================ #
def compute_lgsr(cz, ra, dec):
    c_icrs = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    #print c_icrs.galactic.l.degree
    lon, lat = c_icrs.galactic.l.degree*math.pi/180.0, c_icrs.galactic.b.degree*math.pi/180.0
    v_lsr  = cz + 9 * np.cos(lon) * np.cos(lat) + 12 * np.sin(lon) * np.cos(lat) + 7 * np.sin(lat)
    v_gsr  = v_lsr + 220 * np.sin(lon) * np.cos(lat)
    v_lgsr = v_gsr - 62 * np.cos(lon) * np.cos(lat) + 40 * np.sin(lon) * np.cos(lat) - 35 * np.sin(lat)
    return v_lgsr

# ================================ #
# ========= Beam Factor ========== #
# ================================ #
def beam_factor(axis_maj, axis_min, pix):
    return 1.133*axis_maj*axis_min/(pix*pix)

# ================================ #
# =========== Calc DLUM ========== #
# ================================ #
def dist_lum(z):
    return cosmo.luminosity_distance(z).value

# ================================ #
# == WALLABY PS Flux Correction == #
# ================================ #
def wallaby_flux_scaling(flux):
    log_flux      = np.log10(flux)
    log_flux_corr = log_flux - 0.0285 * log_flux**3 + 0.439 * log_flux**2 - 2.294 * log_flux + 4.097
    return(10**log_flux_corr)

# ================================ #
# =========== Mag Abs to App ===== #
# ================================ #
def mag_Mtom(Mag, dist):
    dist = dist * 10**6
    return 5. * np.log10(dist / 10.) + Mag
  
# ================================ #
# =========== Mag App to Abs ===== #
# ================================ #
def mag_mtoM(mag, dist):
    dist = dist * 10**6
    return mag - 5. * np.log10(dist / 10.)


# ================================ #
# ==== Calc WISE Luminosity ====== #
# ================================ #
def wise_luminosity(mag, dist, band):
    if band == 'W1':
      i = 0
    if band == 'W2':
      i = 1
    if band == 'W3':
      i = 2
    if band == 'W4':
      i = 3
    f0         = np.array([309.540, 171.787, 31.674, 8.363])             # WISE constants
    wavelength = np.array([3.4, 4.6, 12., 22.])                          # micrometres
    freq       = C_LIGHT * 10**3 / (wavelength * 10**-6)                 # Hz
    area       = 4. * math.pi * dist**2. * (3.086 * 10**24.)**2.         # cm**2
    flux       = f0[i] * 10**(-1. * mag / 2.5)                           # Jy
    luminosity = flux * 10**(-23) * area / (3.828 * 10**33) * freq[i]   # Sol_lum
    return luminosity
  
# ================================ #
# ==== Calc WISE Flux       ====== #
# ================================ #
def wise_flux(mag, dist, band):
    if band == 'W1':
      i = 0
    if band == 'W2':
      i = 1
    if band == 'W3':
      i = 2
    if band == 'W4':
      i = 3
    f0         = np.array([309.540, 171.787, 31.674, 8.363])             # WISE constants
    flux       = f0[i] * 10**(-1. * mag / 2.5)                           # Jy
    return flux
  
# ================================ #
# ==== Calc WISE Flux       ====== #
# ================================ #
def wise_flux_to_lum(flux, dist, band):
    if band == 'W1':
      i = 0
    if band == 'W2':
      i = 1
    if band == 'W3':
      i = 2
    if band == 'W4':
      i = 3
    wavelength = np.array([3.4, 4.6, 12., 22.])                          # micrometres
    freq       = C_LIGHT * 10**3 / (wavelength * 10**-6)                 # Hz
    area       = 4. * math.pi * dist**2. * (3.086 * 10**24.)**2.         # cm**2
    luminosity = flux * 10**(-23) * area / (3.828 * 10**33) * freq[i]   # Sol_lum
    return luminosity
  
# ================================ #
# ==== Calc WISE Luminosity ====== #
# ================================ #
def wise_luminosity_watts(mag, dist, band):
    if band == 'W1':
      i = 0
    if band == 'W2':
      i = 1
    if band == 'W3':
      i = 2
    if band == 'W4':
      i = 3
    f0         = np.array([309.540, 171.787, 31.674, 8.363])             # WISE constants
    wavelength = np.array([3.4, 4.6, 12., 22.])                          # micrometres
    freq       = C_LIGHT * 10**3 / (wavelength * 10**-6)                 # Hz
    area       = 4. * math.pi * dist**2. * (3.086 * 10**22.)**2.         # cm**2
    flux       = f0[i] * 10**(-1. * mag / 2.5)                           # Jy
    luminosity = flux * 10**(-26) * area #* freq[i]    # Sol_lum
    #luminosity = luminosity * 1 * 10**-7 * 10**-5
    #luminosity = luminosity * 3.828 * 10**26                             # Watts
    luminosity = np.array(luminosity)
    return luminosity.astype(np.float64)
  
# ================================ #
# ==== Calc WISE Luminosity ====== #
# ================================ #
def wise_flux_luminosity(flux, dist, band):
    if band == 'W1':
      i = 0
    if band == 'W2':
      i = 1
    if band == 'W3':
      i = 2
    if band == 'W4':
      i = 3
    wavelength = np.array([3.4, 4.6, 12., 22.])                          # micrometres
    freq       = C_LIGHT * 10**3 / (wavelength * 10**-6)                 # Hz
    area       = 4. * math.pi * dist**2. * (3.086 * 10**24.)**2.         # cm**2
    luminosity = flux * 10**(-23) * area / (3.828 * 10**33) * freq[i]    # Sol_lum
    return luminosity
  
# ================================ #
# ==== Calc GALEX Luminosity ===== #
# ================================ #
def galex_luminosity(mag, dist, band):
    if band == 'NUV':
      i = 0
    if band == 'FUV':
      i = 1
    m0         = np.array([20.08, 18.82])
    f0         = np.array([2.06 * 10**(-16), 1.40 * 10**(-15)])
    #wavelength = np.array([1516., 2267.])                          # Angstrom
    wavelength = np.array([2315.7, 1538.6])                          # Angstrom
    frequency  = (C_LIGHT * 1000 / (wavelength[i] * 10**-10))      # Hz
    area       = 4. * math.pi * dist**2. * (3.086 * 10**24.)**2.   # cm**2
    flux       = f0[i] * 10**(-1. * (mag - m0[i]) / 2.5)           # erg/s/cm**2/Angstrom
    #print('flux')
    #print(flux)
    flux       = flux * wavelength[i] / frequency                  # erg/s/cm**2/Hz
    luminosity = flux * area                                       # erg/s/Hz
    #luminosity = flux * 10**(-23) * area * wavelength[i]           # Sol_lum
    return luminosity
  
# ================================ #
# ==== Calc GALEX Luminosity ===== #
# ================================ #
def galex_flux(mag, dist, band):
    if band == 'NUV':
      i = 0
    if band == 'FUV':
      i = 1
    m0         = np.array([20.08, 18.82])
    f0         = np.array([2.06 * 10**(-16), 1.40 * 10**(-15)])
    #wavelength = np.array([1516., 2267.])                          # Angstrom
    wavelength = np.array([2315.7, 1538.6])                          # Angstrom
    frequency  = (C_LIGHT * 1000 / (wavelength[i] * 10**-10))      # Hz
    flux       = f0[i] * 10**(-1. * (mag - m0[i]) / 2.5)           # erg/s/cm**2/Angstrom
    flux       = flux * wavelength[i] / frequency                  # erg/s/cm**2/Hz
    return flux


# ================================ #
# ==== Calc GALEX Luminosity ===== #
# ================================ #
def galex_luminosity_watts(mag, dist, band):
    if band == 'NUV':
      i = 0
    if band == 'FUV':
      i = 1
    m0         = np.array([20.08, 18.82])
    f0         = np.array([2.06 * 10**(-16), 1.40 * 10**(-15)])
    #wavelength = np.array([1516., 2267.])                          # Angstrom
    wavelength = np.array([2315.7, 1538.6])                          # Angstrom
    frequency  = (C_LIGHT * 1000 / (wavelength[i] * 10**-10))      # Hz
    area       = 4. * math.pi * dist**2. * (3.086 * 10**24.)**2.   # cm**2
    flux       = f0[i] * 10**(-1. * (mag - m0[i]) / 2.5)           # erg/s/cm**2/Angstrom
    flux       = flux * wavelength[i] #/ frequency                  # erg/s/cm**2  #/Hz
    luminosity = flux * area                                       # erg/s  #/Hz
    luminosity = luminosity * 1. * 10**-7                          # Watts
    #luminosity = flux * 10**(-23) * area * wavelength[i]           # Sol_lum
    return luminosity

# ================================ #
# ==== Calc GALEX Luminosity ===== #
# ================================ #
def calc_eso_mstar(bmag, rmag, dist):
    m0             = 9.075
    frequency      = 6.81 * 10**14
    mpc_to_cm      = 3.086 * 10**24.
    erg_to_sol     = 3.839 * 10**33
    area           = 4. * math.pi * dist**2. * (mpc_to_cm)**2.         # cm**2
    flux           = 10**(-1. * (bmag - m0) / 2.5)                      # Jy
    luminosity     = flux * 10**(-23) * area * frequency / erg_to_sol  # Sol_lum
    mass_to_light  = 10**(1.111 * (bmag - rmag) - 0.976)
    mstar          = np.log10(mass_to_light * luminosity)
    return mstar

# ================================ #
# ==== Calc 2MASS Luminosity ===== #
# ================================ #
def twomass_luminosity(mag, dist, band):
    i          = 0
    f0         = np.array([4.283e-14])
    wavelength = np.array([2.159])                                 # micron
    area       = 4. * math.pi * dist**2. * (3.086 * 10**24.)**2.   # cm**2
    flux       = f0[i] * 10**(-1. * (mag) / 2.5)                   # W cm**-2 micron**-1
    luminosity = flux * area * wavelength[i] / (3.83 * 10**26)     # Sol_lum
    return luminosity

# ================================ #
# ========== HI Mass Obs ========= #
# ================================ #
def observed_hi(flux,factor,z):
    # $\left(\frac{M_{\mathrm{HI}}}{\mathrm{M}_{\odot}}\right)=\frac{2.35\times10^5}{(1+z)^2}\left(\frac{D_{\mathrm{L}}}{\mathrm{Mpc}}\right)^2\left(\frac{S}{\mathrm{Jy\,km\,s}^{-1}}\right)$
    dist = dist_lum(z)
    #z = z_at_value(cosmo.distmod, dist)
    #flux = flux/factor
    return np.log10((2.356*10**5)*flux*dist**2/(1+z)**2)

def hi_mass_jyhz(flux,z):
    dist = dist_lum(z)
    return np.log10(49.7*flux*dist**2)

def observed_hi_dist(flux,z):
    # $\left(\frac{M_{\mathrm{HI}}}{\mathrm{M}_{\odot}}\right)=\frac{2.35\times10^5}{(1+z)^2}\left(\frac{D_{\mathrm{L}}}{\mathrm{Mpc}}\right)^2\left(\frac{S}{\mathrm{Jy\,km\,s}^{-1}}\right)$
    #dist = dist_lum(z)
    dist = z
    #z = z_at_value(cosmo.luminosity_distance, dist, 0.00001, 0.01)
    #flux = flux/factor
    return np.log10((2.356*10**5)*flux*dist**2)#/(1+z)**2)

# ================================ #
# ========== HI Mass Mag ========= #
# ================================ #
def himass_from_mag(a, b, mag):
    return a + b * mag

# ================================ #
# ======== Dynamical Mass ======== #
# ================================ #
def dyn_mass(vel, rad):
  rad = rad * units.kiloparsec
  rad = rad.to(units.meter)
  mass = vel**2*rad/const.G.value/const.M_sun.value*pow(1000,2)
  return mass.value

# ================================= #
# === Stellar Mass from Colour ==== #
# ================================= #
def calc_lgMstar(a,b,mag,col,MSol,z,h=cosmo.h):
    """
	Calculates the stellar mass estimate using a colour-based empirical relation

	<param: a,b (float)> - Fitted parameters of the stellar mass formula.
	<param: mag (float)> - The apparent magnitude of the model [mags].
	<param: col (float)> - The colour of the model [mags].
	<param: MSol (float)> - The absolute magnitude of the Sun in the given luminance band [mags].
	<param: z (float)> - The redshift of the galaxy.
	<param: h (float)> - The reduced Hubble's constant.
	
	<return: Mstar (float)> - The stellar mass [M_sol] as determined by the empirical relation in Taylor et al. 2011
	"""

    distmod = cosmo.distmod(z).value
    #distmod = distmod
    lgMstar = (a + b*col - 0.4*mag
                + 0.4*distmod + np.log10(1.0/np.power(10,-0.4*MSol))
                - np.log10(1.0+z) - 2.0*np.log10(cosmo.h/0.7))
    
    return(lgMstar)

# ================================= #
# ========== SFR from IR ========== #
# ================================= #
def calc_sfr_ir(f60, f100, distance):
    #exponent = pow(10,-44) * 10**7 * 10**44 
    return 7.9 *  (4. * math.pi * pow((distance * 3.086), 2) * (1.26 * (2.58 * f60 + f100))) * pow(10,-7)

# ================================ #
# ========== Log Errors ========== #
# ================================ #
def log_errors(value, error):
  return value - np.log10(10**value - 10**error)

# ================================ #
# ============== RMS ============= #
# ================================ #
def rms_calc(array):
    return np.sqrt(np.mean(np.square(array)))

# ================================ #
# ========= Interpolation ======== #
# ================================ #
def interpolate_value(x, y, a, i):
    b = (x[i+1] - x[i])/(y[i+1] - y[i])*(a - y[i]) + x[i]
    return b

# ================================ #
# ========= Busy Function ======== #
# ================================ #
def busy_function(x, A, B1, B2, W, XE, XP, C, N):
    x = np.array(x)
    return A/4.0 * (erf(B1*(W + x - XE)) + 1.0) * (erf(B2*(W - x + XE)) + 1.0) * (C*np.abs(x - XP)**N + 1.0)


# ================================ #
# ====== Half Gaussian Poly ====== #
# ================================ #
def half_gaussian_poly(x, A, sigma, x0):
    x = np.array(x)
    w = (x - x0) / sigma
    return np.abs(A * np.exp(-w**2 / 2))


# ================================ #
# == Cluster Centric Dist r/rvir = #
# ================================ #
def cluster_centric_distance(rvir, cluster_centre, galaxy_ra, galaxy_dec):
    cluster_centre       = np.array(cluster_centre)
    cluster_centre_coord = SkyCoord(cluster_centre[0]*u.deg, cluster_centre[1]*u.deg, frame='icrs')
    galaxy_coord         = SkyCoord(galaxy_ra*u.deg, galaxy_dec*u.deg, frame='icrs')
    cc_dist              = cluster_centre_coord.separation(galaxy_coord)
    cc_dist              = cc_dist.deg / rvir
    return(cc_dist)


# ================================ #
# ====== Gauss Hermite Poly ====== #
# ================================ #
def gauss_hermite_poly(x, A, sigma, x0, h3):
    x         = np.array(x)
    w         = (x - x0) / sigma
    alpha     = 1 / np.sqrt(2 * math.pi) * np.exp(-w**2 / 2)
    Hermite3  = (1 / np.sqrt(6)) * (2 * np.sqrt(2) * w**3 - 3 * np.sqrt(2) * w)
    return A / sigma * alpha * (1 + h3 * Hermite3)

def gauss_hermite_poly_der(x, A, sigma, x0, h3):
    x         = np.array(x)
    w         = (x - x0) / sigma
    return -A / (sp.sqrt(2 * sp.pi) * sigma**4) * (x - x0) * sp.exp(-w**2 / 2) * (h3 * (sp.sqrt(2) / sp.sqrt(6)) * (6 * w**2 - 3))


# ================================ #
# ========= MCMC Fit BF ========== #
# ================================ #
def bf_like(theta, vel, flux, flux_err):
    A, B1, B2, W, XE, XP, C, N = theta
    models = busy_function(vel, A, B1, B2, W, XE, XP, C, N)
    inv_sigma2 = 1.0/(flux_err**2)
    return -0.5*(np.sum((flux-models)**2*inv_sigma2 - np.log(inv_sigma2)))

def bf_prior(theta):
    A, B1, B2, W, XE, XP, C, N = theta
    if 0.0 < A < np.inf and 0.01 < B1 < 10.0 and 0.01 < B2 < 10.0 and 0.0 < W < 700.0 and -1000.0 < XE < 10000.0 and -1000.0 < XP < 10000.0 and 0.00000001 < C < 0.1 and 0.0 < N < 10.0:
    #if 0.0 < dT < dT_max and -10.0 < T0 < 10.0 and 0.0 < R_step < 20.0: # the dT max prior needs to be passed to this function!
        return 0.0
    return -np.inf

def bf_prob(theta, vel, flux, flux_err):
    lp = bf_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + bf_like(theta, vel, flux, flux_err)

def BF_MCMC(velocity, flux, noise, vsys, width):
    #chi2 = lambda *args: -2 * lnlike(*args)
    #result = op.minimize(chi2, [dT_true, R_step_true, T0_true], args=(R_data, T_data_list[i], sigma_T_list[i]))
    ndim, nwalkers = 8, 100
    #param_guess  = [20, 0.1, 0.1, 0.1, vsys, vsys, 0.0001, 1.5]
    param_guess, sigma, vel_fit, flux_fit = BF_fitting(velocity, flux, vsys, width/2.)
    #print param_guess
    pos = [param_guess + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    #print len(pos), len(pos[0])
    sampler = emcee.EnsembleSampler(nwalkers, ndim, bf_prob, args=(velocity, flux, noise))
    #print("Spectrum %d" % j)
    #print("----------------------------")
    #print("Running MCMC...")
    sampler.run_mcmc(pos, 600, rstate0=np.random.get_state())
    #print("Done.")
    #print
    #sampler_list.append(sampler)
    #sampler = sampler_list[j]
    burnin  = 150
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    A_mcmc, B1_mcmc, B2_mcmc, W_mcmc, XE_mcmc, XP_mcmc, C_mcmc, N_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))
    #print '%.2f\t%.3f\t%.3f\t%.3f\t%.2f\t%.2f\t%.4e\t%.2f' % (A_mcmc[0], B1_mcmc[0], B2_mcmc[0], W_mcmc[0], XE_mcmc[0], XP_mcmc[0], C_mcmc[0], N_mcmc[0])
    vel_bf = []
    for i in range(1000):
      vel_bf.append(velocity[0] + i*(velocity[len(velocity)-1]-velocity[0])/1000)
    busyfit_flux = busy_function(vel_bf, A_mcmc[0], B1_mcmc[0], B2_mcmc[0], W_mcmc[0], XE_mcmc[0], XP_mcmc[0], C_mcmc[0], N_mcmc[0])
    return np.array(vel_bf), np.array(busyfit_flux)


# ================================ #
# ============ Fit BF ============ #
# ================================ #
def BF_fitting(velocity, flux, vsys, halfwidth):
    #if np.max(flux) < 1000:
    #  param_guess  = [20, 0.1, 0.1, 0.1, vsys, vsys, 0.0001, 2.0]
    #else:
    param_guess  = [np.nanmax(flux), 0.1, 0.1, halfwidth, vsys, vsys, 0.0001, 4.0]
    lower_bounds = [0.0, 0.01, 0.01, halfwidth-halfwidth/2., vsys-25., vsys-25., 0.00000001, 1.0]
    upper_bounds = [np.inf, 10.0, 10.0, halfwidth+halfwidth/2., vsys+25., vsys+25., 0.1, 10.0]
    try:
      popt, pcov = curve_fit(busy_function, velocity, flux, param_guess, bounds=(lower_bounds, upper_bounds), maxfev=100000)
      busy_params = []
      for i in range(len(popt)):
        busy_params.append(popt[i])
      vel_bf = []
      for i in range(1000):
        vel_bf.append(velocity[0] + i*(velocity[len(velocity)-1]-velocity[0])/1000)
      A, B1, B2, W = popt[0], popt[1], popt[2], popt[3] 
      XE, XP, C, N = popt[4], popt[5], popt[6], popt[7]
      busyfit_flux = busy_function(vel_bf, A, B1, B2, W, XE, XP, C, N)
      sigma = np.sqrt(np.diag(pcov))
      sigma = np.round(sigma, 2)
      return popt, sigma, np.array(vel_bf), np.array(busyfit_flux)
    except RuntimeError:
      return [0], [0], 0, 0
      print("Error - curve_fit failed")



# ================================ #
# ========= MCMC Fit GH ========== #
# ================================ #
def gh_like(theta, vel, flux, flux_err):
    A, sigma, x0, h3 = theta
    models = gauss_hermite_poly(vel, A, sigma, x0, h3)
    inv_sigma2 = 1.0/(flux_err**2)
    return -0.5*(np.sum((flux-models)**2*inv_sigma2 - np.log(inv_sigma2)))

def gh_prior(theta):
    A, sigma, x0, h3 = theta
    if 0.0 < A < np.inf and 0.0 < sigma < np.inf and -500.0 < x0 < 1000.0 and -10.0 < h3 < 10.0:
    #if 0.0 < dT < dT_max and -10.0 < T0 < 10.0 and 0.0 < R_step < 20.0: # the dT max prior needs to be passed to this function!
        return 0.0
    return -np.inf

def gh_prob(theta, vel, flux, flux_err):
    lp = gh_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + gh_like(theta, vel, flux, flux_err)

def GH_MCMC(velocity, flux, noise, vsys, w50):
    #chi2 = lambda *args: -2 * lnlike(*args)
    #result = op.minimize(chi2, [dT_true, R_step_true, T0_true], args=(R_data, T_data_list[i], sigma_T_list[i]))
    ndim, nwalkers = 4, 100
    #param_guess  = [20, 0.1, 0.1, 0.1, vsys, vsys, 0.0001, 1.5]
    #param_guess, sigma, vel_fit, flux_fit = BF_fitting(velocity, flux, vsys)
    param_guess, stdev, vel_gh, flux_gh, peak_vel = GH_fitting(velocity, flux, vsys, w50)
    pos = [param_guess + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, gh_prob, args=(velocity, flux, noise))
    #print("Spectrum %d" % j)
    #print("----------------------------")
    #print("Running MCMC...")
    sampler.run_mcmc(pos, 600, rstate0=np.random.get_state())
    #print("Done.")
    #print
    #sampler_list.append(sampler)
    #sampler = sampler_list[j]
    burnin  = 150
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    A_mcmc, SIG_mcmc, x0_mcmc, h3_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                zip(*np.percentile(samples, [16, 50, 84], axis=0)))
    #print '%.2f\t%.3f\t%.3f\t%.3f\t%.2f\t%.2f\t%.6f\t%.2f' % (A_mcmc[0], B1_mcmc[0], B2_mcmc[0], W_mcmc[0], XE_mcmc[0], XP_mcmc[0], C_mcmc[0], N_mcmc[0])
    vel_gh = []
    for i in range(500):
      vel_gh.append(velocity[0] + i*(velocity[len(velocity)-1]-velocity[0])/500)
    #busyfit_flux = busy_function(vel_bf, A_mcmc[0], B1_mcmc[0], B2_mcmc[0], W_mcmc[0], XE_mcmc[0], XP_mcmc[0], C_mcmc[0], N_mcmc[0])
    flux_gh = gauss_hermite_poly(vel_gh, A_mcmc[0], SIG_mcmc[0], x0_mcmc[0], h3_mcmc[0])
    #print flux_gh
    if np.max(flux_gh) > 0.0000000001:
      peak_vel = vel_gh[np.argmax(flux_gh)]
      #print x0, peak_vel
    else:
      peak_vel = 0
    mean  = [A_mcmc[0], SIG_mcmc[0], x0_mcmc[0], h3_mcmc[0]]
    stdev = [A_mcmc[1], SIG_mcmc[1], x0_mcmc[1], h3_mcmc[1]]
    return peak_vel, mean, stdev


# ================================ #
# ========= Fit GH Poly ========== #
# ================================ #
def GH_fitting(velocity, flux, vsys, w50):
    param_guess  = [np.max(flux), w50/(2*np.sqrt(2*np.log(2))), vsys, 0.0]
    lower_bounds = [0, 0, vsys-400.0, -10]
    upper_bounds = [np.inf, np.inf, vsys+400.0, 10]
    try:
      popt, pcov = curve_fit(gauss_hermite_poly, velocity, flux, param_guess, bounds=(lower_bounds, upper_bounds), maxfev=100000)
      vel_gh = []
      for i in range(500):
        vel_gh.append(velocity[0] + i*(velocity[len(velocity)-1]-velocity[0])/500)
      vel_gh = np.array(vel_gh)
      A, SIG, x0, h3 = popt[0], popt[1], popt[2], popt[3]
      #print x0
      flux_gh = gauss_hermite_poly(vel_gh, A, SIG, x0, h3)
      sigma = np.sqrt(np.diag(pcov))
      sigma = np.round(sigma, 2)
      if np.max(flux_gh) > 0.0000000001:
        peak_vel = vel_gh[np.argmax(flux_gh)]
        #print x0, peak_vel
      else:
        peak_vel = 0
      return popt, sigma, vel_gh, flux_gh, peak_vel
    except RuntimeError:
      return [0], [0], 0, 0, 0
      print("Error - curve_fit failed")
    except ValueError:
      return [0], [0], 0, 0, 0
      print("Error - curve_fit failed")
      

# ================================ #
# ====== Fit GH Uncertainty ====== #
# ================================ #
def GH_uncertainty(velocity, popt, stdev, samples):
    random_A     = np.random.normal(popt[0], stdev[0], samples)
    random_sigma = np.random.normal(popt[1], stdev[1], samples)
    random_x0    = np.random.normal(popt[2], stdev[2], samples)
    random_h3    = np.random.normal(popt[3], stdev[3], samples)
    vel_gh = []
    for i in range(500):
        vel_gh.append(velocity[0] + i*(velocity[len(velocity)-1]-velocity[0])/500)
    vel_gh = np.array(vel_gh)
    random_vel, random_disp = [], []
    for i in range(samples):
      flux_gh = gauss_hermite_poly(vel_gh, random_A[i], random_sigma[i], random_x0[i], random_h3[i])
      if np.max(flux_gh) > 0.0000000001 and len(flux_gh) != 0:
        random_vel.append(vel_gh[np.argmax(flux_gh)])
        random_disp.append(calc_bf_width(vel_gh, flux_gh, vel_gh[np.argmax(flux_gh)], 50)) #np.max(flux_gh), 
    return random_vel, random_disp


# ================================ #
# ======= Profile Width ========== #
# ================================ #
def profile_width(nu_obs, delta_nu):
    return C_LIGHT*delta_nu/nu_obs

def profile_width(nu_obs, delta_nu):
    return C_LIGHT*delta_nu/nu_obs

# ================================ #
# ===== Flip Spec Residual ======= #
# ================================ #
#def spec_flip_residual(velocity_orig, flux_orig, vsys):
    #dvel = np.abs(velocity_orig[1] - velocity_orig[0])
    #velocity, flux = velocity_orig, flux_orig
    ##velocity = np.arange(velocity_orig[0], velocity_orig[len(velocity_orig)-1], dvel/100.)
    ##flux = []
    ##for i in range(len(flux_orig)):
      ##for j in range(100):
        ##flux.append(flux_orig[i])
    ##flux = np.array(flux)
    #lower_flux, upper_flux = [], []
    #lower_vel, upper_vel   = [], []
    #res = 0
    #for i in range(len(velocity)):
      #if velocity[i] < vsys and velocity[i+1] > vsys:
        #print(dvel, np.abs(velocity[i] - vsys), np.abs(velocity[i+1] - vsys))
      #if velocity[i] < vsys:# and np.abs(velocity[i] - vsys) > dvel:#/100.:
        #lower_flux.append(flux[i])
        #lower_vel.append(velocity[i])
      #if velocity[i] > vsys:# and np.abs(velocity[i] - vsys) > dvel:#/100.:
        #upper_flux.append(flux[i])
        #upper_vel.append(velocity[i])
    #lower_flux = np.flip(lower_flux, 0)
    #lower_vel  = np.flip(lower_vel, 0)
    ##chans = np.min(len(lower_flux), len(upper_flux)
    ##if len(lower_flux) > len(upper_flux):
    #for i in range(np.min([len(lower_flux),len(upper_flux)])):
      #if i < 10:
        #print(vsys, lower_vel[i], upper_vel[i])
      #res += np.abs(lower_flux[i] - upper_flux[i])
    ##np.sum(i > 0 for i in np.abs(flux - np.flip(flux, 0)))
    #return res/np.sum(flux)
    ##return round(np.sum(np.abs(flux - np.flip(flux, 0)))/np.sum(flux), 2)
    
def spec_flip_residual(velocity_orig, flux_orig, vsys):
    dvel = np.abs(velocity_orig[1] - velocity_orig[0])
    velocity, flux = np.array(velocity_orig), np.array(flux_orig)
    #velocity = np.concatenate((np.array([velocity[0] - 2.*dvel, velocity[0] - dvel]), velocity))
    #flux     = np.concatenate((np.array([0, 0]), flux))
    #vel_imax = len(velocity) - 1
    #velocity = np.concatenate((velocity, np.array([velocity[vel_imax] + dvel, velocity[vel_imax] + 2.*dvel])))
    #flux     = np.concatenate((flux, np.array([0, 0])))
    #print(velocity)
    #print(flux)
    #velocity = np.arange(velocity_orig[0], velocity_orig[len(velocity_orig)-1], dvel/100.)
    #flux = []
    #for i in range(len(flux_orig)):
      #for j in range(100):
        #flux.append(flux_orig[i])
    #flux = np.array(flux)
    lower_flux, upper_flux = [], []
    lower_vel, upper_vel   = [], []
    res = 0
    flux_new2 = []
    velocity_new2 = []
    fraction = 0
    vsys_index = len(velocity)/2.
    for i in range(len(velocity)):
      if vsys > velocity[i] and np.abs(velocity[i] - vsys) < dvel:
        #if vsys > velocity[i]:
        vsys_index = i
        fraction   = np.abs(velocity[i] - vsys)/(dvel) #  - dvel/2.
        #print(velocity[i], vsys, dvel/2., np.abs(velocity[i] - vsys), fraction, flux[i])
        #if vsys < velocity[i]:
          #vsys_index = i
          #fraction   = np.abs(velocity[i] - vsys)/(dvel)
        if velocity[i] > vsys:
          vel_shift  = velocity[i] - vsys
          #freq_shift = C_LIGHT/(vsys*1000.) - C_LIGHT/(velocity[i]*1000.)
        else:
          vel_shift  = vsys - velocity[i]
          #freq_shift = C_LIGHT/(velocity[i]*1000.) - C_LIGHT/(vsys*1000.)
    #resample_n = int(len(velocity)/(vel_shift/10.))
    ##flux_new, velocity_new = signal.resample(flux, resample_n, velocity)
    #flux[~np.isfinite(flux)] = 0
    if velocity[0] > velocity[len(velocity)-1]:
      velocity = np.flip(velocity, 0)
      flux     = np.flip(flux, 0)
    input_spec = Spectrum1D(spectral_axis=velocity*u.kilometer/u.second, flux=flux*u.jansky, velocity_convention='doppler_optical')
    velocity_new = np.arange(vsys - dvel * (vsys_index), vsys - dvel * (vsys_index - (len(velocity) - 1)), dvel) * u.kilometer/u.second
    fluxcon      = FluxConservingResampler()
    flux_new     = fluxcon(input_spec, velocity_new)
    velocity_new = flux_new.spectral_axis.value
    flux_new     = flux_new.flux.value
    velocity_new = velocity_new[np.isfinite(flux_new)]
    flux_new     = flux_new[np.isfinite(flux_new)]
    #lower_flux   = flux_new[velocity_new < vsys]
    #upper_flux   = flux_new[velocity_new > vsys-0.01]
    ##shift_fft = np.fft.rfft(freq_shift*1000.)
    ##freq_shift = C_LIGHT/(vel_shift)
    #vel_fft  = np.fft.rfft(velocity)
    #flux_fft = np.fft.rfft(flux)
    #vel_fft_new  = vel_fft * shift_fft #cmath.exp(freq_shift) #cmath.rect(1., freq_shift)
    #flux_fft_new = flux_fft * shift_fft #cmath.exp(freq_shift)  #cmath.rect(1., freq_shift)
    #vel_new  = np.fft.irfft(vel_fft_new)
    #flux_new = np.fft.irfft(flux_fft_new)
    for i in range(len(velocity) - 1): # - 1
      #flux_new.append(flux[i]/2. - flux[i] * fraction + flux[i+1]/2. + flux[i+1] * fraction)
      #flux_new2.append(flux[i] * (1 - fraction) + flux[i+1] * (fraction))
      flux_new2.append(flux[i] * (1 - fraction) + flux[i+1] * (fraction))
      velocity_new2.append(vsys - dvel * (vsys_index - i))
      #print(dvel, np.abs(velocity[i] - vsys), np.abs(velocity[i+1] - vsys))
      if velocity_new2[i] < vsys:# and np.abs(velocity[i] - vsys) > dvel:#/100.:
        lower_flux.append(flux_new2[i])
        lower_vel.append(velocity_new2[i])
      #if velocity_new2[i] > vsys:# and np.abs(velocity[i] - vsys) > dvel:#/100.:
      else:
        upper_flux.append(flux_new2[i])
        upper_vel.append(velocity_new2[i])
    #lower_flux, upper_flux = np.array(lower_flux), np.array(upper_flux)
    #for i in range(len(velocity)):
      #if velocity[i] < vsys and velocity[i+1] > vsys:
        #print(velocity[i], vsys, velocity[i+1])
    #fig10 = plt.figure(10, figsize=(5, 5), facecolor = '#007D7D')
    #plt.step(velocity, flux, color='darkblue')
    #plt.step(velocity_new, flux_new,color='peru')
    #plt.step(velocity_new2, flux_new2,color='green')
    #plt.axvline(vsys, linewidth=0.75, linestyle = '--', color = 'darkgrey')
    #plt.axvline(velocity[vsys_index], linewidth=0.75, linestyle = '-', color = 'black')
    #plt.show()
    lower_flux = np.flip(lower_flux, 0)
    #lower_vel  = np.flip(lower_vel, 0)
    #chans = np.min(len(lower_flux), len(upper_flux)
    #if len(lower_flux) > len(upper_flux):
    low_length  = len(lower_flux)
    high_length = len(upper_flux)
    if len(lower_flux) > len(upper_flux):
      for j in range(np.abs(low_length - high_length)):
        upper_flux = np.concatenate((upper_flux, np.array([0])))
    if len(lower_flux) < len(upper_flux):
      for j in range(np.abs(low_length - high_length)):
        lower_flux = np.concatenate((lower_flux, np.array([0])))
    for i in range(np.min([len(lower_flux),len(upper_flux)])):
      res += np.abs(lower_flux[i] - upper_flux[i])
    #np.sum(i > 0 for i in np.abs(flux - np.flip(flux, 0)))
    #print(np.sum(flux_new2), np.sum(lower_flux)+np.sum(upper_flux))
    return res/np.sum(flux_new2)#/resample_n
    #return round(np.sum(np.abs(flux - np.flip(flux, 0)))/np.sum(flux), 2)
    
    
# ================================ #
# ====== Flip Map Residual ======= #
# ================================ #
def map_flip_residual(infile, infile_rotate, resdir):
    f1        = pyfits.open(infile)
    data, hdr = f1[0].data, f1[0].header
    f1_rotate               = pyfits.open(infile_rotate)
    data_rotate, hdr_rotate = f1_rotate[0].data, f1_rotate[0].header
    #index_max = np.unravel_index(np.argmax(data), (hdr['NAXIS2'], hdr['NAXIS1']))
    #index_com = centroid_com(data)
    index_com = ndi.center_of_mass(data)
    #data_rotate  = scirotate(data, 180, axes=(0,1))
    #index_rotate = np.unravel_index(np.argmax(data_rotate), (hdr['NAXIS2'], hdr['NAXIS1']))
    #index_com_rotate = centroid_com(data_rotate)
    index_com_rotate = ndi.center_of_mass(data_rotate)
    #print index_com, index_com_rotate
    x_diff, y_diff = int(round(index_com[0] - index_com_rotate[0])), int(round(index_com[1] - index_com_rotate[1]))
    flux_total        = 0 #sum(data)
    asymmetry         = 0
    asymmetry_inverse = 0
    counter           = 0
    imdiff            = 0
    flux_abs          = 0
    for i in range(hdr['NAXIS1'] - np.abs(x_diff)):
      for j in range(hdr['NAXIS2'] - np.abs(y_diff)):
        if np.isfinite(data[j][i]):
          flux_total += data[j][i]
          flux_abs += np.abs(data[j][i])
        if np.isfinite(data[j][i]) and np.isfinite(data_rotate[j-y_diff][i-x_diff]):
          imdiff += np.abs(data[j][i] - data_rotate[j-y_diff][i-x_diff])
          if resdir == 'minor':
            if i < index_com[0]:
              if (data[j][i] - data_rotate[j-y_diff][i-x_diff]) != 0.0:
                counter += 1
              asymmetry += (data[j][i] - data_rotate[j-y_diff][i-x_diff])
          elif resdir == 'major':
            if j < index_com[1]:
              if (data[j][i] - data_rotate[j-y_diff][i-x_diff]) != 0.0:
                counter += 1
              asymmetry += (data[j][i] - data_rotate[j-y_diff][i-x_diff])
        #if i > index_com[0]:
        #  asymmetry_inverse += (data[j][i] - data_rotate[j-y_diff][i-x_diff])
    #print asymmetry, counter, flux_total
    asymmetry = np.abs(round(asymmetry/(counter * flux_total)*1000000, 2))
    meas_asym = imdiff/(2*flux_abs)
    #asymmetry_inverse = round(asymmetry_inverse, 2)
    #print asymmetry, asymmetry_inverse
    return asymmetry

def map_asymmetry_bias(infile, cubefile, maskfile):
    f1        = pyfits.open(infile)
    data, hdr = f1[0].data, f1[0].header
    #f1_rotate               = pyfits.open(infile_rotate)
    #data_rotate, hdr_rotate = f1_rotate[0].data, f1_rotate[0].header
    data_rotate      = scirotate(data, 180, axes=(0,1))
    #index_com        = centroid_com(data)
    #index_com_rotate = centroid_com(data_rotate)
    index_com        = ndi.center_of_mass(data)
    index_com_rotate = ndi.center_of_mass(data_rotate)
    x_diff, y_diff    = int(round(index_com[0] - index_com_rotate[0])), int(round(index_com[1] - index_com_rotate[1]))
    flux_total        = 0
    counter           = 0
    imdiff            = 0
    flux_abs          = 0
    for i in range(hdr['NAXIS1'] - np.abs(x_diff)):
      for j in range(hdr['NAXIS2'] - np.abs(y_diff)):
        if np.isfinite(data[j][i]):
          flux_total += data[j][i]
          flux_abs += np.abs(data[j][i])
        if np.isfinite(data[j][i]) and np.isfinite(data_rotate[j-y_diff][i-x_diff]):
          imdiff += np.abs(data[j][i] - data_rotate[j-y_diff][i-x_diff])
    meas_asym = imdiff/(2.*flux_abs)
    bias = compute_bias(cubefile, maskfile)
    corr_asym = meas_asym - bias/(2.*flux_abs)
    #print meas_asym, bias/(2.*flux_abs), corr_asym
    f1.close()
    return corr_asym

def compute_bias(cubefile, maskfile):
    f1_cube             = pyfits.open(cubefile)
    f1_mask             = pyfits.open(maskfile)
    data_cube, hdr_cube = f1_cube[0].data, f1_cube[0].header
    data_mask, hdr_mask = f1_mask[0].data, f1_mask[0].header
    #print data_mask.shape()
    mask_rotate         = scirotate(data_mask, 180, axes=(1,2))
    bias = 0
    #print len(data_cube), len(data_cube[0]), len(data_cube[0][0]), len(data_cube[0][0][0])
    #print len(data_mask), len(data_mask[0]), len(data_mask[0][0])
    array_dim = len(data_cube.shape)
    if array_dim > 3:
    #if len(data_cube[0][0][0]) > 1:
      full_size = len(data_cube[0][0][0])*len(data_cube[0][0])
    else:
      full_size = len(data_cube[0][0])*len(data_cube[0])
    if len(data_mask[0])*len(data_mask[0][0]) < full_size/2:
      for i in range(len(data_mask[0])):
        for j in range(len(data_mask[0][0])):
          pixsum     = 0
          pixsum_rot = 0
          for k in range(len(data_mask)):
            if array_dim > 3:
            #if len(data_cube[0][0][0]) > 1:
              if len(data_cube[0]) == len(data_mask):
                l = k
              else:
                l = k + int(np.abs(len(data_cube[0]) - len(data_mask))/2)
              if len(data_cube[0][0]) == len(data_mask[0]):
                m = k
              else:
                m = k + 10
              if len(data_cube[0][0][0]) == len(data_mask[0][0]):
                n = k
              else:
                n = k + 10
              pixsum     += data_cube[0][l][m][n] * data_mask[k][i][j]
              pixsum_rot += data_cube[0][l][m][n] * mask_rotate[k][i][j]
            else:
              if len(data_cube) == len(data_mask):
                l = k
              else:
                l = k + int(np.abs(len(data_cube) - len(data_mask))/2)
              if len(data_cube[0]) == len(data_mask[0]):
                m = k
              else:
                m = k + 10
              if len(data_cube[0][0]) == len(data_mask[0][0]):
                n = k
              else:
                n = k + 10
              pixsum     += data_cube[l][m][n] * data_mask[k][i][j]
              pixsum_rot += data_cube[l][m][n] * mask_rotate[k][i][j]
            #pixsum_rot += data_cube[k+5][i+10][j+10]*mask_rotate[len(data_mask)-k][len(data_mask[0])-i][len(data_mask[0][0])-j]
          bias += np.abs(pixsum - pixsum_rot)
    else:
      bias = 0
    f1_cube.close()
    f1_mask.close()
    return bias


def map_asym_noise(infile, cubefile, maskfile):
    f1        = pyfits.open(infile)
    data, hdr = f1[0].data, f1[0].header
    data = np.pad(data, (10,10), mode='constant', constant_values=0)
    #f1_rotate               = pyfits.open(infile_rotate)
    #data_rotate, hdr_rotate = f1_rotate[0].data, f1_rotate[0].header
    data_rotate      = scirotate(data, 180, axes=(0,1))
    #index_com        = centroid_com(data)
    #index_com_rotate = centroid_com(data_rotate)
    index_com         = ndi.center_of_mass(data)
    index_com_rotate  = ndi.center_of_mass(data_rotate)
    x_diff            = int(round(index_com[0] - index_com_rotate[0])) #- 2 
    y_diff            = int(round(index_com[1] - index_com_rotate[1])) #+ 2
    flux_total        = 0
    counter           = 0
    imdiff            = 0
    flux_abs          = 0
    for i in range(hdr['NAXIS1'] + 20 - np.abs(x_diff)):
      for j in range(hdr['NAXIS2'] + 20 - np.abs(y_diff)):
        if np.isfinite(data[j][i]):
          flux_total += data[j][i]
          flux_abs += np.abs(data[j][i])
        if np.isfinite(data[j][i]) and np.isfinite(data_rotate[j-y_diff][i-x_diff]):
          imdiff += np.abs(data[j][i] - data_rotate[j-y_diff][i-x_diff])
    #print(imdiff)
    meas_asym = imdiff/(2.*flux_abs)
    #bias = compute_bias_noise(cubefile, maskfile) * 18500.
    #print(meas_asym, bias, bias/(2.*flux_abs), flux_abs)
    corr_asym = meas_asym #- bias/(2.*flux_abs)
    #print(corr_asym)
    #print meas_asym, bias/(2.*flux_abs), corr_asym
    f1.close()
    return corr_asym


def compute_bias_noise(cubefile, maskfile):
    f1_cube             = pyfits.open(cubefile)
    f1_mask             = pyfits.open(maskfile)
    data_cube, hdr_cube = f1_cube[0].data, f1_cube[0].header
    data_mask, hdr_mask = f1_mask[0].data, f1_mask[0].header
    #data_mask[data_mask>0] = 1.
    #print data_mask.shape()
    mask_rotate         = scirotate(data_mask, 180)#, axes=(1,2))
    bias = 0
    #print len(data_cube), len(data_cube[0]), len(data_cube[0][0]), len(data_cube[0][0][0])
    #print len(data_mask), len(data_mask[0]), len(data_mask[0][0])
    noise_map        = data_cube[1] * data_mask
    noise_map_rotate = data_cube[1] * mask_rotate
    bias = np.sum(np.abs(noise_map - noise_map_rotate))
    #array_dim = len(data_cube.shape)
    #if array_dim > 3:
    ##if len(data_cube[0][0][0]) > 1:
      #full_size = len(data_cube[0][0][0])*len(data_cube[0][0])
    #else:
      #full_size = len(data_cube[0][0])*len(data_cube[0])
    ##if len(data_mask[0])*len(data_mask[0][0]) < full_size/2:
    #for i in range(len(data_mask[0])):
      #for j in range(len(data_mask[0][0])):
        #pixsum     = 0
        #pixsum_rot = 0
        #for k in range(len(data_mask)):
          #if array_dim > 3:
          ##if len(data_cube[0][0][0]) > 1:
            #if len(data_cube[0]) == len(data_mask):
              #l = k
            #else:
              #l = k + int(np.abs(len(data_cube[0]) - len(data_mask))/2)
            #if len(data_cube[0][0]) == len(data_mask[0]):
              #m = k
            #else:
              #m = k + 10
            #if len(data_cube[0][0][0]) == len(data_mask[0][0]):
              #n = k
            #else:
              #n = k + 10
            #pixsum     += data_cube[0][l][m][n] * data_mask[k][i][j]
            #pixsum_rot += data_cube[0][l][m][n] * mask_rotate[k][i][j]
          #else:
            #if len(data_cube) == len(data_mask):
              #l = k
            #else:
              #l = k + int(np.abs(len(data_cube) - len(data_mask))/2)
            #if len(data_cube[0]) == len(data_mask[0]):
              #m = k
            #else:
              #m = k + 10
            #if len(data_cube[0][0]) == len(data_mask[0][0]):
              #n = k
            #else:
              #n = k + 10
            #pixsum     += data_cube[l][m][n] * data_mask[k][i][j]
            #pixsum_rot += data_cube[l][m][n] * mask_rotate[k][i][j]
          ##pixsum_rot += data_cube[k+5][i+10][j+10]*mask_rotate[len(data_mask)-k][len(data_mask[0])-i][len(data_mask[0][0])-j]
        #bias += np.abs(pixsum - pixsum_rot)
    #else:
    #  bias = 0
    f1_cube.close()
    f1_mask.close()
    return bias

#def compute_bias_noise(cubefile, maskfile):
    #f1_cube             = pyfits.open(cubefile)
    #f1_mask             = pyfits.open(maskfile)
    #data_cube, hdr_cube = f1_cube[0].data, f1_cube[0].header
    #data_mask, hdr_mask = f1_mask[0].data, f1_mask[0].header
    ##print data_mask.shape()
    #mask_rotate         = scirotate(data_mask, 180, axes=(1,2))
    #bias = 0
    ##print len(data_cube), len(data_cube[0]), len(data_cube[0][0]), len(data_cube[0][0][0])
    ##print len(data_mask), len(data_mask[0]), len(data_mask[0][0])
    #array_dim = len(data_cube.shape)
    #if array_dim > 3:
    ##if len(data_cube[0][0][0]) > 1:
      #full_size = len(data_cube[0][0][0])*len(data_cube[0][0])
    #else:
      #full_size = len(data_cube[0][0])*len(data_cube[0])
    ##if len(data_mask[0])*len(data_mask[0][0]) < full_size/2:
    #for i in range(len(data_mask[0])):
      #for j in range(len(data_mask[0][0])):
        #pixsum     = 0
        #pixsum_rot = 0
        #for k in range(len(data_mask)):
          #if array_dim > 3:
          ##if len(data_cube[0][0][0]) > 1:
            #if len(data_cube[0]) == len(data_mask):
              #l = k
            #else:
              #l = k + int(np.abs(len(data_cube[0]) - len(data_mask))/2)
            #if len(data_cube[0][0]) == len(data_mask[0]):
              #m = k
            #else:
              #m = k + 10
            #if len(data_cube[0][0][0]) == len(data_mask[0][0]):
              #n = k
            #else:
              #n = k + 10
            #pixsum     += data_cube[0][l][m][n] * data_mask[k][i][j]
            #pixsum_rot += data_cube[0][l][m][n] * mask_rotate[k][i][j]
          #else:
            #if len(data_cube) == len(data_mask):
              #l = k
            #else:
              #l = k + int(np.abs(len(data_cube) - len(data_mask))/2)
            #if len(data_cube[0]) == len(data_mask[0]):
              #m = k
            #else:
              #m = k + 10
            #if len(data_cube[0][0]) == len(data_mask[0][0]):
              #n = k
            #else:
              #n = k + 10
            #pixsum     += data_cube[l][m][n] * data_mask[k][i][j]
            #pixsum_rot += data_cube[l][m][n] * mask_rotate[k][i][j]
          ##pixsum_rot += data_cube[k+5][i+10][j+10]*mask_rotate[len(data_mask)-k][len(data_mask[0])-i][len(data_mask[0][0])-j]
        #bias += np.abs(pixsum - pixsum_rot)
    ##else:
    ##  bias = 0
    #f1_cube.close()
    #f1_mask.close()
    #return bias

def map_asym_chan(infile, cubefile, chanfile):
    f1        = pyfits.open(infile)
    data, hdr = f1[0].data, f1[0].header
    wcs  = WCS(hdr)
    f1_chan        = pyfits.open(chanfile)
    data_chan, hdr_chan = f1_chan[0].data, f1_chan[0].header
    #data = data * data_chan
    #for i in range(45, 60):
      #for j in range(len(data[i])):
        #data[i][j] = 0
    #print(data[49])
    #print(data[50])
    #print(data[51])
    data = np.pad(data, 10, mode='constant', constant_values=0)
    #f1_rotate               = pyfits.open(infile_rotate)
    #data_rotate, hdr_rotate = f1_rotate[0].data, f1_rotate[0].header
    data_rotate      = scirotate(data, 180, axes=(0,1))
    #index_com        = centroid_com(data)
    #index_com_rotate = centroid_com(data_rotate)
    #print(len(data), len(data[0]), len(data_rotate), len(data_rotate[0]), hdr['NAXIS1'], hdr['NAXIS2'])
    #print(data[40][40], data_rotate[54][50])
    #print(data[40][40], data_rotate[int(len(data_rotate)-40-1)][int(len(data_rotate[0])-40-1)])
    index_com         = ndi.center_of_mass(data)
    index_com_rotate  = [int(len(data_rotate)-46-1), int(len(data_rotate[0])-45-1)]  #ndi.center_of_mass(data_rotate)
    x_diff1            = int(round(index_com[0]) - round(index_com_rotate[0])) #- 2 
    y_diff1            = int(round(index_com[1]) - round(index_com_rotate[1])) #+ 2
    #print(np.round(index_com), np.round(index_com_rotate))
    #print(data[46][45], data_rotate[int(len(data_rotate)-46-1)][int(len(data_rotate[0])-45-1)])
    #print(int(len(data_rotate)-46-1), int(len(data_rotate[0])-45-1))
    #print(data[int(round(index_com[1]))][int(round(index_com[0]))], data_rotate[int(round(index_com[1])) - y_diff1][int(round(index_com[0])) - x_diff1])
    #print(x_diff1, y_diff1)
    min_asymmetry = 1
    min_xdiff     = 100
    min_ydiff     = 100
    for k in range(11):
      for l in range(11):
        x_diff = x_diff1 - 5 + k
        y_diff = y_diff1 - 5 + l
        #print(x_diff, y_diff)
        #if x_diff == -3 and y_diff == 1:
          #print(x_diff, y_diff)
        #x_diff, y_diff = 0, 0
        #print(x_diff, y_diff)
        #fig1 = plt.figure(1, figsize=(6, 3))
        #ax1 = fig1.add_subplot(1, 2, 1, facecolor = 'w', projection=wcs)
        #ax1.imshow(data, origin='lower', aspect='auto', cmap='Greys')#, clim=(4000,13000))
        #ax1.scatter(index_com[1], index_com[0], color='green', marker='o')
        #print(index_com, index_com_rotate)
        #ax2 = fig1.add_subplot(1, 2, 2, facecolor = 'w', projection=wcs)
        #ax2.imshow(data_rotate, origin='lower', aspect='auto', cmap='Greys')#, clim=(4000,13000))
        #ax2.scatter(index_com_rotate[1], index_com_rotate[0], color='green', marker='o')
        #ax2.scatter(index_com[1], index_com[0], color='red', marker='o')
        #plt.show()
        flux_total        = 0
        counter           = 0
        imdiff            = 0
        flux_abs          = 0
        for i in range(hdr['NAXIS1'] + 20 - np.abs(x_diff)):
          for j in range(hdr['NAXIS2'] + 20 - np.abs(y_diff)):
            if np.isfinite(data[j][i]):
              flux_total += data[j][i]
              flux_abs += np.abs(data[j][i])
            if np.isfinite(data[j][i]) and np.isfinite(data_rotate[j-y_diff][i-x_diff]):
              imdiff += np.abs(data[j][i] - data_rotate[j-y_diff][i-x_diff])
        #print(imdiff)
        #print(flux_abs)
        meas_asym = imdiff/(2.*flux_abs)
        if meas_asym < min_asymmetry:
          min_asymmetry = meas_asym
          min_xdiff     = x_diff
          min_ydiff     = y_diff
    meas_asym = min_asymmetry
    print(np.round(meas_asym,2), min_xdiff, min_ydiff)
    #bias = compute_bias_chan_noise(cubefile, chanfile) * 18500.
    #print(meas_asym, bias, bias/(2.*flux_abs), flux_abs)
    corr_asym = meas_asym #- bias/(2.*flux_abs)
    #print(corr_asym)
    #print meas_asym, bias/(2.*flux_abs), corr_asym
    f1.close()
    return corr_asym


def compute_bias_chan_noise(cubefile, maskfile):
    f1                        = pyfits.open(cubefile)
    data1, hdr                 = f1[0].data, f1[0].header
    #data1                      = np.pad(data1, (0,10,10), mode='constant', constant_values=0)
    f1_mask                   = pyfits.open(maskfile)
    data_mask, hdr_mask       = f1_mask[0].data, f1_mask[0].header
    #data_mask                 = np.pad(data_mask, (10,10), mode='constant', constant_values=0)
    wcs  = WCS(hdr_mask)
    data_mask[data_mask != 0] = 1
    #for i in range(len(data1[0][0])):
      #print(data1[0][0][i])
    #print(data1[0][35][35])
    #print(data_mask[35])
    #print('cube')
    #print(data1[0][0][1])
    #print(len(data1), len(data1[0]), len(data1[0][0]))
    #print(data1)
    #print(len(data1), len(data1[0]), len(data[0]), len(data[::0]))
    #print(data1[1,35,:])
    data                      = data1[0,:,:]
    #print(data[35])
    data                      = data * data_mask
    #print(len(data), len(data[0]), len(data_mask), len(data_mask[0]))
    #print(data[35])
    #f1_rotate               = pyfits.open(infile_rotate)
    #data_rotate, hdr_rotate = f1_rotate[0].data, f1_rotate[0].header
    data_rotate      = scirotate(data, 180, axes=(0,1))
    #index_com        = centroid_com(data)
    #index_com_rotate = centroid_com(data_rotate)
    index_com         = ndi.center_of_mass(data)
    index_com_rotate  = ndi.center_of_mass(data_rotate)
    #print(index_com, index_com_rotate)
    x_diff, y_diff    = int(round(index_com[0] - index_com_rotate[0])), int(round(index_com[1] - index_com_rotate[1]))
    #fig1 = plt.figure(1, figsize=(6, 3))
    #ax1 = fig1.add_subplot(1, 2, 1, facecolor = 'w', projection=wcs)
    #ax1.imshow(data, origin='lower', aspect='auto', cmap='Greys')#, clim=(4000,13000))
    ##ax1.scatter(index_com[0], index_com[1], color='green', marker='o')
    ##print(index_com, index_com_rotate)
    #ax2 = fig1.add_subplot(1, 2, 2, facecolor = 'w', projection=wcs)
    #ax2.imshow(data_rotate, origin='lower', aspect='auto', cmap='Greys')#, clim=(4000,13000))
    ##ax2.scatter(index_com_rotate[0], index_com_rotate[1], color='green', marker='o')
    #plt.show()
    x_diff, y_diff = 0, 0
    flux_total        = 0
    counter           = 0
    imdiff            = 0
    flux_abs          = 0
    for i in range(hdr['NAXIS1']):# + 20 - np.abs(x_diff)):
      for j in range(hdr['NAXIS2']):# + 20 - np.abs(y_diff)):
        if np.isfinite(data[j][i]):
          flux_total += data[j][i]
          flux_abs += np.abs(data[j][i])
        if np.isfinite(data[j][i]) and np.isfinite(data_rotate[j-y_diff][i-x_diff]):
          imdiff += np.abs(data[j][i] - data_rotate[j-y_diff][i-x_diff])
    #print(imdiff)
    bias = imdiff#/(2.*flux_abs)
    #print(bias)
    f1.close()
    f1_mask.close()
    return bias


# ================================ #
# ====== Flip Moment 1 Map ======= #
# ================================ #
def weighted_median(data, weights):
    """
    Args:
      data (list or numpy.array): data
      weights (list or numpy.array): weights
    """
    data, weights = np.array(data).squeeze(), np.array(weights).squeeze()
    #print(data, weights)
    s_data, s_weights = map(np.array, zip(*sorted(zip(data, weights))))
    midpoint = 0.5 * sum(s_weights)
    if any(weights > midpoint):
        w_median = (data[weights == np.nanmax(weights)])[0]
    else:
        cs_weights = np.nancumsum(s_weights)
        idx = np.where(cs_weights <= midpoint)[0][-1]
        if cs_weights[idx] == midpoint:
            w_median = np.nanmean(s_data[idx:idx+2])
        else:
            w_median = s_data[idx+1]
    return w_median

def velfield_asymmetry(infile, mom0_file, pos_file):
    f1          = pyfits.open(infile)
    data, hdr   = f1[0].data, f1[0].header
    data = np.pad(data, (10,10), mode='constant', constant_values=np.nan)
    if np.nanmean(data) > 1e8:
      data = (1420.406/(data/1e6) - 1) * const.c.to('km/s').value
    data_rotate = np.rot90(data, 2)
    if mom0_file != False:
      f1_mom0             = pyfits.open(mom0_file)
      data_mom0, hdr_mom0 = f1_mom0[0].data, f1_mom0[0].header
      data_mom0 = np.pad(data_mom0, (10,10), mode='constant', constant_values=0)
      data_rotate_mom0 = np.rot90(data_mom0, 2)
      #index_com        = centroid_com(data_mom0)
      ##print (index_com)
      #index_com_rotate = centroid_com(data_rotate_mom0)
      index_com        = ndi.center_of_mass(data_mom0)
      #print (index_com)
      index_com_rotate = ndi.center_of_mass(data_rotate_mom0)
      xpos, ypos = index_com[0], index_com[1]
      x_diff, y_diff   = int(round(index_com[0] - index_com_rotate[0])), int(round(index_com[1] - index_com_rotate[1]))
      print(index_com, index_com_rotate)
      print(x_diff, y_diff)
    if pos_file != False:
      xpos_array, ypos_array = np.genfromtxt(pos_file, usecols=(7,8), unpack=True)
      xpos, ypos     = int(round(np.nanmean(xpos_array))), int(round(np.nanmean(ypos_array)))
      #print(xpos, ypos)
      xrot, yrot     = hdr['NAXIS1'] - xpos, hdr['NAXIS2'] - ypos
      x_diff, y_diff = xpos - xrot, ypos - yrot
    out_map = data*np.nan
    diff_map = data*np.nan
    for i in range(hdr['NAXIS1'] - int(np.abs(x_diff)/3)):
      for j in range(hdr['NAXIS2'] - int(np.abs(y_diff)/3)):
        if np.isfinite(data[j][i]) and np.isfinite(data_rotate[j-y_diff][i-x_diff]):
          #print(data[j][i])
          out_map[j][i] = data[j][i] + data_rotate[j-y_diff][i-x_diff]
          diff_map[j][i] = data[j][i] - data_rotate[j-y_diff][i-x_diff]
          #if np.abs(out_map[j][i]) > 10*np.abs(2*np.nanmean(data[j][i])):
          #  out_map[j][i] = np.nan
    med_abs_dev = madev(out_map, ignore_nan = True)
    out_map_flat = out_map.flatten()
    data_mom0_flat = data_mom0.flatten()
    #print(data_mom0_flat)
    for i in range(len(data_mom0_flat)):
      if data_mom0_flat[i] > 0:
        data_mom0_flat[i] = np.sqrt(data_mom0_flat[i])
    out_map_nonan   = out_map_flat[(np.isfinite(out_map_flat)) & (np.isfinite(data_mom0_flat))]
    data_mom0_nonan = data_mom0_flat[(np.isfinite(out_map_flat)) & (np.isfinite(data_mom0_flat))]
    map_wmedian = weighted_median(out_map_nonan, weights=data_mom0_nonan)
    #print(map_wmedian)
    wmed_abs_dev = weighted_median((np.abs(out_map_nonan-map_wmedian)), weights=data_mom0_nonan)
    #print('%.0f\t%.0f\t%.0f\t%.0f' % (med_abs_dev, wmed_abs_dev, np.nanmax(diff_map), np.percentile(diff_map[~np.isnan(diff_map)], 95)))
    f1.close()
    if mom0_file != False:
      f1_mom0.close()
    #return med_abs_dev, out_map, hdr, xpos, ypos, wmed_abs_dev
    #return med_abs_dev/np.nanmax(diff_map), out_map, hdr, xpos, ypos, wmed_abs_dev/np.nanmax(diff_map)
    #fig1 = plt.figure(111, figsize=(6, 3), facecolor = '#007D7D')
    #matplotlib.rcParams.update({'font.size': 13})
    #ax1 = fig1.add_axes([0.07,0.1,0.25,0.9], facecolor = 'w')
    #ax1.imshow(data, origin='lower', aspect='auto', cmap='viridis')
    #ax1.text(0.1, 0.9, 'Model', transform=ax1.transAxes)
    #ax1.set_xlabel(r'x [pixels]')
    #ax1.set_ylabel(r'y [pixels]')
    #ax2 = fig1.add_axes([0.32,0.1,0.25,0.9], facecolor = 'w')
    #image2 = ax2.imshow(data_rotate, origin='lower', aspect='auto', cmap='viridis')
    #ax2.text(0.1, 0.9, 'Rotated', transform=ax2.transAxes)
    #ax2.set_xlabel(r'x [pixels]')
    #ax2.set_yticklabels([])
    #cbar2 = plt.colorbar(image2, fraction=0.1, pad=-0.05)
    #cbar2.set_label(r'Velocity [km\,s$^{-1}$]')
    #ax3 = fig1.add_axes([0.7,0.1,0.25,0.9], facecolor = 'w')
    #image3 = ax3.imshow(out_map, origin='lower', aspect='auto', cmap='viridis')#, vmin=-200, vmax=200)
    #ax3.set_xlabel(r'x [pixels]')
    #ax3.text(0.1, 0.9, 'Model + Rotated', transform=ax3.transAxes)
    #ax3.set_yticklabels([])
    #cbar3 = plt.colorbar(image3, fraction=0.1, pad=-0.05)
    #cbar3.set_label(r'Velocity [km\,s$^{-1}$]')
    ##plot_name = basedir + 'PLOTS/MAPS/example_velmadev.pdf'
    ##plt.savefig(plot_name, bbox_inches = 'tight', dpi = 400)
    #plt.show()
    #plt.clf()
    return med_abs_dev/np.percentile(diff_map[~np.isnan(diff_map)], 95), out_map, hdr, xpos, ypos, wmed_abs_dev/np.percentile(diff_map[~np.isnan(diff_map)], 95)
  

def velfield_asymmetry_mask(infile, mom0_file, pos_file):
    f1          = pyfits.open(infile)
    data, hdr   = f1[0].data, f1[0].header
    f1_mom0             = pyfits.open(mom0_file)
    data_mom0, hdr_mom0 = f1_mom0[0].data, f1_mom0[0].header
    for i in range(hdr['NAXIS1']):
      for j in range(hdr['NAXIS2']):
        mask_limit = 0.15*np.nanmax(data_mom0)
        if data_mom0[j][i] < mask_limit:
          data[j][i] = np.nan
    f1_mom0.close()
    if np.nanmean(data) > 1e8:
      data = (1420.406/(data/1e6) - 1) * const.c.to('km/s').value
    data_rotate = np.rot90(data, 2)
    if mom0_file != False:
      f1_mom0             = pyfits.open(mom0_file)
      data_mom0, hdr_mom0 = f1_mom0[0].data, f1_mom0[0].header
      data_rotate_mom0 = np.rot90(data_mom0, 2)
      #index_com        = centroid_com(data_mom0)
      #index_com_rotate = centroid_com(data_rotate_mom0)
      index_com        = ndi.center_of_mass(data_mom0)
      index_com_rotate = ndi.center_of_mass(data_rotate_mom0)
      xpos, ypos = index_com[0], index_com[1]
      x_diff, y_diff   = int(round(index_com[0] - index_com_rotate[0])), int(round(index_com[1] - index_com_rotate[1]))
    if pos_file != False:
      xpos_array, ypos_array = np.genfromtxt(pos_file, usecols=(7,8), unpack=True)
      xpos, ypos     = int(round(np.nanmean(xpos_array))), int(round(np.nanmean(ypos_array)))
      #print(xpos, ypos)
      xrot, yrot     = hdr['NAXIS1'] - xpos, hdr['NAXIS2'] - ypos
      x_diff, y_diff = xpos - xrot, ypos - yrot
    out_map = data*np.nan
    diff_map = data*np.nan
    for i in range(hdr['NAXIS1'] - int(np.abs(x_diff)/3)):
      for j in range(hdr['NAXIS2'] - int(np.abs(y_diff)/3)):
        if np.isfinite(data[j][i]) and np.isfinite(data_rotate[j-y_diff][i-x_diff]):
          out_map[j][i] = data[j][i] + data_rotate[j-y_diff][i-x_diff]
          diff_map[j][i] = data[j][i] - data_rotate[j-y_diff][i-x_diff]
          #if np.abs(out_map[j][i]) > 10*np.abs(2*np.nanmean(data[j][i])):
          #  out_map[j][i] = np.nan
    med_abs_dev    = madev(out_map, ignore_nan = True)
    out_map_flat   = out_map.flatten()
    data_mom0_flat = data_mom0.flatten()
    for i in range(len(data_mom0_flat)):
      if data_mom0_flat[i] > 0:
        data_mom0_flat[i] = np.sqrt(data_mom0_flat[i])
    out_map_nonan   = out_map_flat[(np.isfinite(out_map_flat)) & (np.isfinite(data_mom0_flat))]
    data_mom0_nonan = data_mom0_flat[(np.isfinite(out_map_flat)) & (np.isfinite(data_mom0_flat))]
    map_wmedian    = weighted_median(out_map_nonan, weights=data_mom0_nonan)
    #print(map_wmedian)
    wmed_abs_dev   = weighted_median((np.abs(out_map_nonan-map_wmedian)), weights=data_mom0_nonan)
    print('%.0f\t%.0f\t%.0f\t%.0f' % (med_abs_dev, wmed_abs_dev, np.nanmax(diff_map), np.percentile(diff_map[~np.isnan(diff_map)], 95)))
    f1.close()
    if mom0_file != False:
      f1_mom0.close()
    #return med_abs_dev, out_map, hdr, xpos, ypos, wmed_abs_dev
    #return med_abs_dev/np.nanmax(diff_map), out_map, hdr, xpos, ypos, wmed_abs_dev/np.nanmax(diff_map)
    return med_abs_dev/np.percentile(diff_map[~np.isnan(diff_map)], 95), out_map, hdr, xpos, ypos, wmed_abs_dev/np.percentile(diff_map[~np.isnan(diff_map)], 95)

# ================================ #
# ======== Spec Flux Asym ======== #
# ================================ #
def spec_asym_flux(velocity, flux, rms, vsys, w20):
    sum_low  = 0
    sum_high = 0
    chan_low  = 0
    chan_high = 0
    #print(vsys, w20, w20/2.*12.5, vsys-w20/2.*12.5)
    delta_v = velocity[1] - velocity[0]
    #print(vsys, vsys-w20/2., vsys+w20/2., delta_v)
    #print(velocity)
    for i in range(len(flux)):
      #print('%.2f %.2f' % (velocity[i], flux[i]))
      #print(velocity[i], vsys-w20/2., np.abs(velocity[i] - vsys-w20/2.), delta_v)
      if velocity[i] < vsys-w20/2. and np.abs(velocity[i] - (vsys-w20/2.)) < delta_v/2.:
          sum_low += flux[i] * (1 - np.abs(velocity[i] - (vsys-w20/2.))/delta_v)
          chan_low += 1
          #print(sum_low)
          #sum_high += flux[i] * (1 - np.abs(velocity[i] - vsys)/delta_v)
      if velocity[i] > vsys-w20/2. and velocity[i] < vsys: #*12.5
        if np.abs(velocity[i] - (vsys-w20/2.)) < delta_v/2.:
          sum_low += flux[i] * (np.abs(velocity[i] - (vsys-w20/2.))/delta_v)
          chan_low += 1
          #print(sum_low)
        elif np.abs(velocity[i] - vsys) < delta_v/2.:
          sum_low += flux[i]/2. + flux[i] * np.abs(velocity[i] - vsys)/delta_v
          sum_high += flux[i]/2. - flux[i] * (np.abs(velocity[i] - vsys)/delta_v)
          chan_low += 1
          #print(flux[i], flux[i]/2. + flux[i] * np.abs(velocity[i] - vsys)/delta_v, flux[i]/2. - flux[i] * (np.abs(velocity[i] - vsys)/delta_v))
          #print(sum_low, sum_high)
        else:
          sum_low += flux[i]
          chan_low += 1
          #print(sum_low)
      if velocity[i] > vsys+w20/2. and np.abs(velocity[i] - (vsys+w20/2.)) < delta_v/2.:
          sum_high += flux[i] * (1 - np.abs(velocity[i] - (vsys+w20/2.))/delta_v)
          chan_high += 1
          #print(sum_high)
          #sum_high += flux[i] * (1 - np.abs(velocity[i] - vsys)/delta_v)
      if velocity[i] > vsys and velocity[i] < vsys+w20/2.: #*12.5
        if np.abs(velocity[i] - (vsys+w20/2.)) < delta_v/2.:
          sum_high += flux[i] * (np.abs(velocity[i] - (vsys+w20/2.))/delta_v)
          chan_high += 1
          #print(sum_high)
        elif np.abs(velocity[i] - vsys) < delta_v/2.:
          sum_low += flux[i]/2. - flux[i] * (np.abs(velocity[i] - vsys)/delta_v)
          sum_high += flux[i]/2. + flux[i] * (np.abs(velocity[i] - vsys)/delta_v)
          chan_high += 1
          #print(sum_low, sum_high)
        else:
          sum_high += flux[i]
          chan_high += 1
          #print(sum_high)
      #print(sum_low, sum_high)
    #print(chan_low, chan_high)
    #print('%.1f %.1f %i' % (sum_low, sum_high, len(velocity)))
    error_low  = np.sqrt(chan_low)*np.mean(rms)*1000.*delta_v/sum_high
    #print(rms)
    #print(error_low, np.sqrt(chan_low), np.mean(rms), delta_v ,sum_high)
    error_high = sum_low*np.sqrt(chan_high)*np.mean(rms)*1000.*delta_v/sum_high**2
    error = np.sqrt((error_low)**2 + (error_high)**2)
    if sum_high != 0 and sum_low != 0:
      if sum_low/sum_high > 1:
        asym = sum_low/sum_high
      else:
        asym = sum_high/sum_low
    else:
      asym = np.nan
    return asym, error


# ================================ #
# ======== Spec Peak Asym ======== #
# ================================ #
def spec_asym_peak(velocity, flux, vsys):
    #print velocity
    #print vsys
    #print len(velocity)
    for i in range(len(velocity)):
      #print velocity[i], vsys
      if velocity[0] < velocity[len(velocity)-1]:
        if velocity[i] < vsys and velocity[i+1] > vsys:
          index_sys = i
          break
      else:
        if velocity[i] > vsys and velocity[i+1] < vsys:
          index_sys = i
          break
    deriv2 = np.diff(np.sign(np.diff(flux)))
    #print deriv2
    local_max = []
    local_min = []
    for i in range(len(deriv2)):
      if deriv2[i] > 0:
        local_min.append(i)
      if deriv2[i] < 0:
        local_max.append(i)
    #print len(flux), len(deriv2)
    #print local_max, local_min
    #print index_sys
    #peak_left  = np.max(flux[:len(flux)/2])
    #peak_right = np.max(flux[len(flux)/2:])
    #if peak_right != 0 and peak_left != 0:
    peaks_low, peaks_high = [], []
    if len(local_max) > 1:
      for i in range(len(local_max)):
        if local_max[i] < index_sys: 
          #lm_low.append(local_max[i])
          peaks_low.append(flux[local_max[i]+1])
        if local_max[i] > index_sys:
          #lm_high.append(local_max[i])
          peaks_high.append(flux[local_max[i]+1])
      #print peaks_high, peaks_low
      if len(peaks_low)>0 and len(peaks_high)>0:
        if len(peaks_low) == 1:
          peak_left = peaks_low[0]
        else:
          peak_left = np.max(peaks_low)
        if len(peaks_high) == 1:
          peak_right = peaks_high[0]
        else:
          peak_right = np.max(peaks_high)
        #peak_right = flux[local_max[1]+1]
        #print peak_left, peak_right
        if peak_left/peak_right > 1:
          asym = peak_left/peak_right
        else:
          asym = peak_right/peak_left
      else:
        asym = 0
    else:
      asym = 0
    #print (asym)
    return asym

# ================================ #
# ====== Flux Weighted VSYS ====== #
# ================================ #
def flux_weighted_mean_vel(velocity, flux):
    vsys = np.sum(velocity*flux)/np.sum(flux)
    #for i in range(len(flux)):
    sint = np.sum(flux)
    w50  = 0
    return w50, vsys

'''
def flux_weighted_vsys(velocity, flux):
    sint     = np.sum(flux)
    flux_sum = 0
    vsys     = 0
    for i in range(len(velocity)):
      if flux[i] != 0 and flux_sum < sint/2.0:
        flux_sum += flux[i]
        vsys = velocity[i]
      else:
        break
    return vsys
'''

# ================================ #
# ===== Width/VSYS Flux Sum ====== #
# ================================ #
def flux_tophat(velocity, flux):
    sint       = np.sum(flux)
    flux_sum   = 0
    w50        = 0
    vsys       = 0
    index_low  = 0
    index_high = 0
    del_v      = np.abs(velocity[0]-velocity[1])
    for i in range(1000):
      test_flux = i*sint/1000.0
      if flux_sum < sint/2.0:
        for j in range(len(flux)-1):
          if flux[j] > test_flux and flux[j+1] < test_flux:
            vel_low = interpolate_value(velocity, flux, test_flux, j)
            index_low = j
          if flux[j] < test_flux and flux[j+1] > test_flux:
            vel_high = interpolate_value(velocity, flux, test_flux, j)
            index_high = j
          if index_low == 0:
            vel_low = velocity[0]
            index_low = 0
          if index_high == 0:
            vel_high   = velocity[len(flux) - 1]
            index_high = len(flux) - 1
        flux_sum = np.abs(test_flux*(vel_high - vel_low)/del_v)
        w50      = np.abs(vel_high - vel_low)
        vsys     = vel_low + w50/2.0
      else:
        break
    return w50, vsys

# ================================ #
# ===== Width/VSYS from Peak ===== #
# ================================ #
def calc_bf_width(velocity, flux, vsys_kms, percentage):
    check_vlow, check_vhigh = velocity[0], velocity[len(velocity)-1]
    velocity = HI_REST/(velocity/C_LIGHT + 1.)
    vsys     = HI_REST/(vsys_kms/C_LIGHT + 1.)
    #print vsys
    #print velocity
    if percentage == 50:
      factor = 2.
    if percentage == 20:
      factor = 5.
    half_peak  = np.max(flux)/factor
    #print half_peak
    lower_vel, upper_vel, lower_flux, upper_flux  = [], [], [], []
    for i in range(len(velocity)):
      if velocity[i] < vsys:
        lower_vel.append(velocity[i])
        lower_flux.append(flux[i])
      if velocity[i] > vsys:
        upper_vel.append(velocity[i])
        upper_flux.append(flux[i])
    lower_vel     = np.array(lower_vel)
    upper_vel     = np.array(upper_vel)
    lower_flux    = np.array(lower_flux)
    upper_flux    = np.array(upper_flux)
    #print lower_vel
    #print lower_flux
    #print upper_vel
    #print upper_flux
    if len(lower_flux) != 0:
      max_low       = np.argmax(lower_flux)
    else:
      max_low       = 0
    if len(upper_flux) != 0:
      max_high      = np.argmax(upper_flux)
    else:
      max_high      = 0
    if check_vlow > check_vhigh:
      vel_inv_low   = lower_vel[:max_low]
      flux_inv_low  = lower_flux[:max_low]
      vel_inv_high  = np.flip(upper_vel[max_high:-1], 0)
      flux_inv_high = np.flip(upper_flux[max_high:-1], 0)
    else:
      vel_inv_low   = np.flip(lower_vel[max_low:-1], 0)
      flux_inv_low  = np.flip(lower_flux[max_low:-1], 0)
      vel_inv_high  = upper_vel[:max_high]
      flux_inv_high = upper_flux[:max_high]
    vel_low, vel_high = 0, 0
    #print vel_inv_low
    #print flux_inv_low
    #print vel_inv_high
    #print flux_inv_high
    #print max_low, len(lower_flux), max_high, len(upper_flux)
    if max_low != len(lower_flux)-1 and max_low != len(lower_flux) and max_low != 0 and max_high != len(upper_flux)-1 and max_high != len(upper_flux) and max_high != 0:
      for i in range(len(flux_inv_low)-1):
        if (flux_inv_low[i] < half_peak and flux_inv_low[i+1] > half_peak) or (flux_inv_low[i] > half_peak and flux_inv_low[i+1] < half_peak):
          vel_low = interpolate_value(vel_inv_low, flux_inv_low, half_peak, i)
          break
      for i in range(len(flux_inv_high)-1):
        if (flux_inv_high[i] < half_peak and flux_inv_high[i+1] > half_peak) or (flux_inv_high[i] > half_peak and flux_inv_high[i+1] < half_peak):
          vel_high = interpolate_value(vel_inv_high, flux_inv_high, half_peak, i)
          break
    else:
      upper_vel = np.flip(upper_vel, 0)
      upper_flux = np.flip(upper_flux, 0)
      for i in range(len(lower_vel)-1):
        if (lower_flux[i] < half_peak and lower_flux[i+1] > half_peak) or (lower_flux[i] > half_peak and lower_flux[i+1] < half_peak):
          vel_low = interpolate_value(lower_vel, lower_flux, half_peak, i)
          break
      for i in range(len(upper_vel)-1):
        if (upper_flux[i] < half_peak and upper_flux[i+1] > half_peak) or (upper_flux[i] > half_peak and upper_flux[i+1] < half_peak):
          vel_high = interpolate_value(upper_vel, upper_flux, half_peak, i)
          break
    #print (HI_REST/vel_low - 1)*C_LIGHT, (HI_REST/vel_high - 1)*C_LIGHT
    if vel_high != 0 and vel_low != 0:
      W50  = (vel_high - vel_low)*C_LIGHT*(1. + (HI_REST)/vsys - 1.)/(HI_REST)
      VSYS = (HI_REST/vel_high - 1.)*C_LIGHT + W50/2.
    else:
      W50  = 0
      VSYS = 0
    #print(vel_low, vel_high, W50, VSYS)
    return W50, VSYS

# ================================ #
# ==== Width/VSYS from 2 Peak ==== #
# ================================ #
def calc_width_2peaks(velocity, flux, vsys, percentage):
    check_vlow, check_vhigh = velocity[0], velocity[len(velocity)-1]
    velocity = HI_REST/(velocity/C_LIGHT + 1)
    vsys     = HI_REST/(vsys/C_LIGHT + 1)
    if percentage == 50:
      factor = 2.
    if percentage == 20:
      factor = 5.
    half_peak  = np.max(flux)/factor
    lower_vel, upper_vel, lower_flux, upper_flux  = [], [], [], []
    for i in range(len(velocity)):
      if velocity[i] < vsys:
        lower_vel.append(velocity[i])
        lower_flux.append(flux[i])
      if velocity[i] > vsys:
        upper_vel.append(velocity[i])
        upper_flux.append(flux[i])
    lower_vel     = np.array(lower_vel)
    upper_vel     = np.array(upper_vel)
    lower_flux    = np.array(lower_flux)
    upper_flux    = np.array(upper_flux)
    if len(lower_flux) != 0:
      max_low       = np.argmax(lower_flux)
      half_peak_low = np.max(lower_flux)/factor
    else:
      max_low       = 0
      half_peak_low = 0
    if len(upper_flux) != 0:
      max_high       = np.argmax(upper_flux)
      half_peak_high = np.max(upper_flux)/factor
    else:
      max_high       = 0
      half_peak_high = 0
    if check_vlow > check_vhigh:
      vel_inv_low   = lower_vel[:max_low]
      flux_inv_low  = lower_flux[:max_low]
      vel_inv_high  = np.flip(upper_vel[max_high:-1], 0)
      flux_inv_high = np.flip(upper_flux[max_high:-1], 0)
    else:
      vel_inv_low   = np.flip(lower_vel[max_low:-1], 0)
      flux_inv_low  = np.flip(lower_flux[max_low:-1], 0)
      vel_inv_high  = upper_vel[:max_high]
      flux_inv_high = upper_flux[:max_high]
    vel_low, vel_high = 0, 0
    if max_low != len(lower_flux)-1 and max_low != len(lower_flux) and max_low != 0 and max_high != len(upper_flux)-1 and max_high != len(upper_flux) and max_high != 0:
      for i in range(len(flux_inv_low)-1):
        if (flux_inv_low[i] < half_peak_low and flux_inv_low[i+1] > half_peak_low) or (flux_inv_low[i] > half_peak_low and flux_inv_low[i+1] < half_peak_low):
          vel_low = interpolate_value(vel_inv_low, flux_inv_low, half_peak_low, i)
          break
      for i in range(len(flux_inv_high)-1):
        if (flux_inv_high[i] < half_peak_high and flux_inv_high[i+1] > half_peak_high) or (flux_inv_high[i] > half_peak_high and flux_inv_high[i+1] < half_peak_high):
          vel_high = interpolate_value(vel_inv_high, flux_inv_high, half_peak_high, i)
          break
    else:
      upper_vel = np.flip(upper_vel, 0)
      upper_flux = np.flip(upper_flux, 0)
      for i in range(len(lower_vel)-1):
        if (lower_flux[i] < half_peak and lower_flux[i+1] > half_peak) or (lower_flux[i] > half_peak and lower_flux[i+1] < half_peak):
          vel_low = interpolate_value(lower_vel, lower_flux, half_peak, i)
          break
      for i in range(len(upper_vel)-1):
        if (upper_flux[i] < half_peak and upper_flux[i+1] > half_peak) or (upper_flux[i] > half_peak and upper_flux[i+1] < half_peak):
          vel_high = interpolate_value(upper_vel, upper_flux, half_peak, i)
          break
    if vel_high != 0 and vel_low != 0:
      W50  = (vel_high - vel_low)*C_LIGHT*(1 + (HI_REST)/vsys - 1)/(HI_REST)
      VSYS = (HI_REST/vel_high - 1)*C_LIGHT + W50/2
    else:
      W50  = 0
      VSYS = 0
    return W50, VSYS


# ================================ #
# ==== Asymmetry Param Stats ===== #
# ================================ #
def param_stats(param_fit, clip_val):
    param_fit = [x for x in param_fit if ~np.isnan(x)]
    param_fit = [x for x in param_fit if x != 0]
    param_fit = np.clip(param_fit, np.median(param_fit) - clip_val, np.median(param_fit) + clip_val)
    return np.mean(param_fit), np.std(param_fit)

# ================================ #
# ====== Averaged Kinemetry ====== #
# ================================ #
def kinemetry_averaged(radius, vasym, vasym_err, low, high):
    vasym_sum     = 0
    vasym_err_sum = 0
    counter       = 0
    for j in range(len(radius)):
      if radius[j] > low and radius[j] < high:
        counter += 1
        vasym_sum += vasym[j]
        vasym_err_sum += vasym_err[j]
    #print counter
    if counter != 0:
      vasym_avg = vasym_sum/counter
      vasym_avg_err = (vasym_err_sum/counter)/np.sqrt(counter)
    else:
      vasym_avg = 0
      vasym_avg_err = 0
    return (high+low)/2, vasym_avg, vasym_avg_err


# ================================ #
# ===== Compute NN Distances ===== #
# ================================ #
def nearest_neightbour_distances(lvhis_c, catalogue_6df, sigma):
    density  = 0
    sum_dist = 0
    sep3d_list = []
    obj_id, sep2d, sep3d = match_coordinates_3d(lvhis_c, catalogue_6df, nthneighbor=1)
    if sep2d.deg[0] < 0.05:
      obj_id_sigma, sep2d_sigma, sep3d_sigma    = match_coordinates_3d(lvhis_c, catalogue_6df, nthneighbor=sigma+1)
      for k in range(2, sigma+2):
        obj_id, sep2d, sep3d    = match_coordinates_3d(lvhis_c, catalogue_6df, nthneighbor=k)
        sum_dist += sep3d.value**3
    else:
      obj_id_sigma, sep2d_sigma, sep3d_sigma    = match_coordinates_3d(lvhis_c, catalogue_6df, nthneighbor=sigma)
      for k in range(1, sigma+1):
        obj_id, sep2d, sep3d    = match_coordinates_3d(lvhis_c, catalogue_6df, nthneighbor=k)
        sum_dist += sep3d.value**3
    density_bayes = 11.48 / sum_dist
    sigma_bayes   = (sigma / (4. / 3. * math.pi * density_bayes))**(1./3.)
    return sigma_bayes, density_bayes


# ================================= #
# ====== PSD Escape Velocity ====== #
# ================================= #
#def psd_escape_velocity(mvir, rvir, c, radius):
    #vesc       = np.zeros(len(radius))
    #grav_const = const.G.value
    #mvir       = mvir * const.M_sun.value
    #rvir       = rvir * const.kpc.value * 1000.
    #gc         = (np.log(1. + c) - c / (1. + c))**(-1)
    #s          = radius * math.pi / 2.
    ##ks        = gc * (np.log(1 + c * s)) / s
    #ks         = gc * ((np.log(1. + c * s)) / s - np.log(1. + c)) + 1.
    #constant   = (2. * grav_const * mvir) / (3. * rvir)
    #vesc[radius<1]   = np.sqrt(constant * ks[radius<1]) / 1000.
    #vesc[radius>=1]  = np.sqrt(constant / s[radius>=1]) / 1000.
    #return(vesc)

def psd_escape_velocity(mvir, rvir, c, radius):
    vesc       = np.zeros(len(radius))
    grav_const = const.G.value
    mvir       = mvir * const.M_sun.value
    rvir       = rvir * const.kpc.value * 1000.
    gc         = (np.log(1. + c) - c / (1. + c))**(-1)
    s          = radius * math.pi / 2.
    #ks        = gc * (np.log(1 + c * s)) / s
    ks         = gc * ((np.log(1. + c * s)) / s) #- np.log(1. + c)) + 1.
    constant   = (2. * grav_const * mvir) / (3. * rvir) #
    #vesc[radius<1]   = np.sqrt(constant * ks[radius<1]) / 1000.
    #vesc[radius>=1]  = np.sqrt(constant / s[radius>=1]) / 1000.
    vesc       = np.sqrt(constant * ks) / 1000.
    return(vesc)









