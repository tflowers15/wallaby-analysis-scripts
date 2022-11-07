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

from functions_plot import field_mom_plot, field_pv_plot

from envelope_tracing import open_fits_file, env_trace

import warnings

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

# ====================================== Funtions =======================================

# ================================ #
# ========== Open Files ========== #
# ================================ #

def open_rms(filename):
    input_lines  = open(filename).read().split('\n')
    rms  = np.array([float(line[42:51]) for line in input_lines[8:-4]])
    return rms
  
def open_cat(filename, column):
    input_lines  = open(filename).read().split('\n')
    if len((input_lines[4].split())) == 39:
      column = column
    else:
      column = column - 1
    values  = np.array([(line.split()[column]) for line in input_lines[4:-1]])
    value = []
    for i in range(len(values)):
      if column != 52: #38
        value.append(float(values[i]))
      if column == 52: #38
        value.append(values[i])
    value = np.array(value)
    return(value)

def open_sofia_spec(filename, vel_type):
    input_lines  = open(filename).read().split('\n')
    freq  = np.array([float(line.split()[1]) for line in input_lines[23:-1]])
    flux  = np.array([float(line.split()[2]) for line in input_lines[23:-1]])
    #freq=freq/1000
    #print freq/1e6
    if vel_type == 'FREQ':
      freq = (1420.406/(freq/1e6) - 1) * const.c.to('km/s').value
    elif vel_type == 'VRAD':
      freq = (1 - (freq/1000)/const.c.to('km/s').value)*1420.406
      freq = (1420.406/(freq) - 1) * const.c.to('km/s').value
      #freq = freq/1000
    else:
      freq = freq/1000
    #print freq
    freq = np.array(freq)
    flux = np.array(flux)
    return(freq, flux)

def open_sofia_spec2(filename, vel_type):
    input_lines  = open(filename).read().split('\n')
    freq  = np.array([float(line.split()[1]) for line in input_lines[23:-1]])
    flux  = np.array([float(line.split()[2]) for line in input_lines[23:-1]])
    npix  = np.array([float(line.split()[3]) for line in input_lines[23:-1]])
    #freq=freq/1000
    #print freq/1e6
    if freq[0] > 1e8: #vel_type == 'FREQ':
      freq = (1420.406/(freq/1e6) - 1) * const.c.to('km/s').value
    #elif vel_type == 'VRAD':
    else:
      freq = (1 - (freq/1000)/const.c.to('km/s').value)*1420.406
      freq = (1420.406/(freq) - 1) * const.c.to('km/s').value
      #freq = freq/1000
    #else:
    #  freq = freq/1000
    #print freq
    freq = np.array(freq)
    flux = np.array(flux)
    npix = np.array(npix)
    return(freq, flux, npix)

def open_things_spec(filename, vel_type):
    input_lines  = open(filename).read().split('\n')
    freq  = np.array([float(line.split()[1]) for line in input_lines[3:-1]])
    flux  = np.array([float(line.split()[2]) for line in input_lines[3:-1]])
    #freq=freq/1000
    #print freq/1e6
    #if vel_type == 'FREQ':
    if freq[0] > 1e8:
      freq = (1420.406/(freq/1e6) - 1) * const.c.to('km/s').value
    #elif vel_type == 'VRAD':
    elif freq[0] < 1e8:
      freq = (1 - (freq/1000)/const.c.to('km/s').value)*1420.406
      freq = (1420.406/(freq) - 1) * const.c.to('km/s').value
      #freq = freq/1000
    #else:
    #  freq = freq/1000
    #print freq
    freq = np.array(freq)
    flux = np.array(flux)
    return(freq, flux)

def open_things_spec2(filename, vel_type):
    input_lines  = open(filename).read().split('\n')
    freq  = np.array([float(line.split()[0]) for line in input_lines[6:-1]])
    flux  = np.array([float(line.split()[1]) for line in input_lines[6:-1]])
    #freq=freq/1000
    #print freq/1e6
    #if vel_type == 'FREQ':
    if freq[0] > 1e8:
      freq = (1420.406/(freq/1e6) - 1) * const.c.to('km/s').value
    #elif vel_type == 'VRAD':
    elif freq[0] < 1e8:
      freq = (1 - (freq/1000)/const.c.to('km/s').value)*1420.406
      freq = (1420.406/(freq) - 1) * const.c.to('km/s').value
      #freq = freq/1000
    #else:
    #  freq = freq/1000
    #print freq
    freq = np.array(freq)
    flux = np.array(flux)
    return(freq, flux)

def open_param_file(filename, i):
    input_lines  = open(filename).read().split('\n')
    if len(input_lines) > 2:
      param_true   = np.array([float(line.split('\t')[i]) for line in input_lines[1:2]])
      parameter    = np.array([float(line.split('\t')[i]) for line in input_lines[2:-1]])
    else:
      param_true, parameter = 0, [0,0,0,0,0,0,0,0]
    return param_true, parameter

def open_asym_file(filename, i):
    input_lines  = open(filename).read().split('\n')
    galaxy       = np.array([line.split('\t')[0] for line in input_lines[2:-1]])
    values       = np.array([float(line.split('\t')[i]) for line in input_lines[2:-1]])
    #if galaxy[0] == 'LVHIS001':
      #for i in range(len(galaxy)):
        #if galaxy[i] == 'LVHIS043':
          #index = i
      #param_low, param_high = np.array(values[:index]), np.array(values[index+1:])
      #parameter = np.concatenate((param_low, param_high), axis = 0)
      #parameter = np.array(parameter)
    #else:
    parameter = np.array(values)
    return parameter

def open_kinemetry_file(filename, i):
    input_lines  = open(filename).read().split('\n')
    parameter    = np.array([float(line.split()[i]) for line in input_lines[0:-1]])
    return parameter

def open_position(filename):
    input_lines  = open(filename).read().split('\n')
    obj = np.array([(line.split('\t')[0]) for line in input_lines[2:-1]])
    ra  = np.array([float(line.split('\t')[1]) for line in input_lines[2:-1]])
    dec = np.array([float(line.split('\t')[2]) for line in input_lines[2:-1]])
    vsys = np.array([float(line.split('\t')[3]) for line in input_lines[2:-1]])
    return obj, ra, dec, vsys
  
def open_catalogue(filename, column):
    input_lines  = open(filename).read().split('\n')
    if column == 1:
      value  = np.array([(line.split('|')[column]) for line in input_lines[7:-2]])
      #ra     = np.array([(line.split()[0]) for line in value[7:-1]])
      #dec    = np.array([(line.split()[1]) for line in value[7:-1]])
      ra1     = np.array([(line.split()[0]) for line in value]) #[0:-1]
      ra2     = np.array([(line.split()[1]) for line in value])
      ra3     = np.array([(line.split()[2]) for line in value])
      dec1    = np.array([(line.split()[3]) for line in value])
      dec2    = np.array([(line.split()[4]) for line in value])
      dec3    = np.array([(line.split()[5]) for line in value])
      ra, dec = [],[] #np.zeros(len(ra1)), np.zeros(len(ra1))
      for i in range(len(ra1)):
        ra.append(ra1[i] + 'h' + ra2[i] + 'm' + ra3[i] + 's')
        dec.append(dec1[i] + 'd' + dec2[i] + 'm' + dec3[i] + 's')
      return ra, dec
    else:
      value  = np.array([(line.split('|')[column]) for line in input_lines[7:-2]])
      value[value=='     '] = '-999'
      value = value.astype(np.float)
      return value

def open_cat_6df_spec(filename, column):
    input_lines  = open(filename).read().split('\n')
    flag  = np.array([float(line.split('|')[7]) for line in input_lines[9:-2]])
    param = np.array([(line.split('|')[column]) for line in input_lines[9:-2]])
    value = []
    for i in range(len(flag)):
      if flag[i] < 6 and flag[i] > 2:
        value.append(np.float(param[i]))
    value = np.array(value)
    return value
  
def open_cat_evcc(filename, column):
    input_lines  = open(filename).read().split('\n')
    value = np.array([float(line.split('|')[column]) for line in input_lines[9:-2]])
    value = np.array(value)
    return value

def open_cat_sim(filename):
    input_lines  = open(filename).read().split('\n')
    ra  = np.array([float(line.split()[0]) for line in input_lines[0:-1]])
    dec = np.array([float(line.split()[1]) for line in input_lines[0:-1]])
    z   = np.array([float(line.split()[2]) for line in input_lines[0:-1]])
    h   = np.array([float(line.split()[9]) for line in input_lines[0:-1]])
    j   = np.array([float(line.split()[10]) for line in input_lines[0:-1]])
    k   = np.array([float(line.split()[11]) for line in input_lines[0:-1]])
    ra, dec, z, h, j, k = np.array(ra), np.array(dec), np.array(z), np.array(h), np.array(j), np.array(k), 
    return ra, dec, z, h, j, k

def open_vopt(filename, column):
    input_lines  = open(filename).read().split('\n')
    obj  = np.array([(line.split('|')[column]) for line in input_lines[3:-1]])
    vopt = np.array([(line.split('|')[1]) for line in input_lines[3:-1]])
    vopt[vopt=='      '] = '-999'
    #print vopt
    vopt = vopt.astype(np.float)
    vopt[vopt>12000] = np.nan
    vopt[vopt<-900]  = np.nan
    lvhis_gals = []
    vopt_all   = []
    for i in range(82):
      if i < 9:
        lvhis_gals.append('LVHIS00%i' % (i+1))
      else:
        lvhis_gals.append('LVHIS0%i' % (i+1))
    counter = 0
    for i in range(len(lvhis_gals)):
      full_mom0_file = '/Users/tflowers/ASYMMETRIES/LVHIS/SOFIA_OUTPUT/%s_mom0.fits' % lvhis_gals[i]
      if os.path.isfile(full_mom0_file) and lvhis_gals[i] != 'LVHIS043':
        vopt_all.append(np.nan)
        for j in range(len(obj)):
          if lvhis_gals[i] == obj[j].strip():
            vopt_all[counter] = vopt[j]
        counter += 1
          #else:
          #  vopt_all.append(np.nan)
    return vopt_all

def open_beam(filename):
    input_lines  = open(filename).read().split('\n')
    freq  = np.array([(line.split(',')[1]) for line in input_lines[0:-1]])
    flux  = np.array([(line.split(',')[2]) for line in input_lines[0:-1]])
    for i in range(len(freq)):
      ra  = freq[i].split(':')
      dec = flux[i].split(':')
      #print ra, dec
      freq[i] = ra[0] + 'h' + ra[1] + 'm' + ra[2] + 's'
      flux[i] = dec[0] + 'd' + dec[1] + 'm' + dec[2] + 's'
    return(freq, flux)

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
# ========== HI Mass Obs ========= #
# ================================ #
def observed_hi(flux,factor,z):
    # $\left(\frac{M_{\mathrm{HI}}}{\mathrm{M}_{\odot}}\right)=\frac{2.35\times10^5}{(1+z)^2}\left(\frac{D_{\mathrm{L}}}{\mathrm{Mpc}}\right)^2\left(\frac{S}{\mathrm{Jy\,km\,s}^{-1}}\right)$
    dist = dist_lum(z)
    #z = z_at_value(cosmo.distmod, dist)
    #flux = flux/factor
    return np.log10((2.356*10**5)*flux*dist**2/(1+z)**2)

def observed_hi_dist(flux,z):
    # $\left(\frac{M_{\mathrm{HI}}}{\mathrm{M}_{\odot}}\right)=\frac{2.35\times10^5}{(1+z)^2}\left(\frac{D_{\mathrm{L}}}{\mathrm{Mpc}}\right)^2\left(\frac{S}{\mathrm{Jy\,km\,s}^{-1}}\right)$
    #dist = dist_lum(z)
    dist = z
    #z = z_at_value(cosmo.luminosity_distance, dist, 0.00001, 0.01)
    #flux = flux/factor
    return np.log10((2.356*10**5)*flux*dist**2)#/(1+z)**2)

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
    index_com = centroid_com(data)
    #data_rotate  = scirotate(data, 180, axes=(0,1))
    #index_rotate = np.unravel_index(np.argmax(data_rotate), (hdr['NAXIS2'], hdr['NAXIS1']))
    index_com_rotate = centroid_com(data_rotate)
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
'''
def map_asymmetry_bias(infile, cubefile, maskfile):
    f1        = pyfits.open(infile)
    data, hdr = f1[0].data, f1[0].header
    #f1_rotate               = pyfits.open(infile_rotate)
    #data_rotate, hdr_rotate = f1_rotate[0].data, f1_rotate[0].header
    data_rotate      = scirotate(data, 180, axes=(0,1))
    index_com        = centroid_com(data)
    index_com_rotate = centroid_com(data_rotate)
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
'''

def map_asym_noise(infile, cubefile, maskfile):
    f1        = pyfits.open(infile)
    data, hdr = f1[0].data, f1[0].header
    data = np.pad(data, (10,10), mode='constant', constant_values=0)
    print('mom0')
    print(data)
    #f1_rotate               = pyfits.open(infile_rotate)
    #data_rotate, hdr_rotate = f1_rotate[0].data, f1_rotate[0].header
    data_rotate      = scirotate(data, 180, axes=(0,1))
    index_com        = centroid_com(data)
    index_com_rotate = centroid_com(data_rotate)
    x_diff, y_diff    = int(round(index_com[0] - index_com_rotate[0])), int(round(index_com[1] - index_com_rotate[1]))
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
    meas_asym = imdiff/(2.*flux_abs)
    #bias = compute_bias_noise(cubefile, maskfile)
    corr_asym = meas_asym #- bias/(2.*flux_abs)
    #print meas_asym, bias/(2.*flux_abs), corr_asym
    f1.close()
    return corr_asym

def compute_bias_noise(cubefile, maskfile):
    f1                        = pyfits.open(cubefile)
    data1, hdr                 = f1[0].data, f1[0].header
    data1                      = np.pad(data1, (10,10), mode='constant', constant_values=0)
    f1_mask                   = pyfits.open(maskfile)
    data_mask, hdr_mask       = f1_mask[0].data, f1_mask[0].header
    data_mask                 = np.pad(data_mask, (10,10), mode='constant', constant_values=0)
    data_mask[data_mask != 0] = 1
    print('cube')
    print(data1[0][0][1])
    #print(data1)
    print(len(data1), len(data1[0]), len(data[0]), len(data[::0]))
    data                      = data1[::0] * data_mask
    #f1_rotate               = pyfits.open(infile_rotate)
    #data_rotate, hdr_rotate = f1_rotate[0].data, f1_rotate[0].header
    data_rotate      = scirotate(data, 180, axes=(0,1))
    #index_com        = centroid_com(data)
    #index_com_rotate = centroid_com(data_rotate)
    #x_diff, y_diff    = int(round(index_com[0] - index_com_rotate[0])), int(round(index_com[1] - index_com_rotate[1]))
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
    bias = imdiff/(2.*flux_abs)
    f1.close()
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
      index_com        = centroid_com(data_mom0)
      #print (index_com)
      index_com_rotate = centroid_com(data_rotate_mom0)
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
      index_com        = centroid_com(data_mom0)
      #print (index_com)
      index_com_rotate = centroid_com(data_rotate_mom0)
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

# ====================================================================================== #
# ------------------------------------ PLOTTING ---------------------------------------- #
# ====================================================================================== #

# ================================ #
# ========= Spectra Plot ========= #
# ================================ #
def spec_plot(fig_num, sub1, sub2, sub3, velocity, flux, vel_bf, busyfit_flux, error, colour, txtstr, vsys, vsys_bf):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    if sub1*sub2 == 12:
      if sub3 > 8:
        ax1.set_xlabel(r'Velocity [km/s]')
      if sub3 == 1 or sub3 == 5 or sub3 == 9:
        ax1.set_ylabel(r'Flux [Jy]')
    if sub1*sub2 == 28:
      if sub3 > 21:
        ax1.set_xlabel(r'Velocity [km/s]')
      if sub3 == 1 or sub3 == 8 or sub3 == 15 or sub3 == 22:
        ax1.set_ylabel(r'Flux [Jy]')
    #half_vel = 4.0*len(velocity)/2.0
    #ax1.set_xlim(vsys-half_vel, vsys+half_vel)
    flux = flux/1000.0
    busyfit_flux = busyfit_flux/1000.0
    ax1.set_ylim(np.min(flux), np.max(flux))
    plt.text(0.1, 0.8, txtstr, transform=ax1.transAxes)
    #plt.axhline(0, linewidth=0.75, linestyle = ':', color = 'darkgrey')
    plt.axvline(vsys, linewidth=0.75, linestyle = '--', color = 'darkgrey', label = r'$V_{\mathrm{sys,SoFiA}}$')
    plt.axvline(vsys_bf, linewidth=0.75, linestyle = ':', color = 'black', label = r'$V_{\mathrm{sys,BF}}$')
    plt.plot(velocity, flux, linestyle = '-', color = colour, linewidth = 1.0)#, label = txtstr)
    #frequency = (1.420405751786*pow(10,9))/(velocity/C_LIGHT + 1)
    #if colour == 'peru':
    plt.plot(vel_bf, busyfit_flux, linestyle = '--', color = 'peru', linewidth = 0.75, label = 'BusyFit')
    error = np.array(error)
    plt.fill_between(velocity, flux-error, flux+error, alpha=0.5, edgecolor='none', facecolor='lightblue')
    #if sub3 == 4:
      #ax1.legend(loc='upper right', fontsize = 8.5)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    ax1.get_yaxis().set_tick_params(which='both', direction='in')
    ax1.get_xaxis().set_tick_params(which='both', direction='in')
    #return velocity, flux

# ================================ #
# ===== Example Spectrum Plot ==== #
# ================================ #
def spec_example_plot(fig_num, sub1, sub2, sub3, velocity, flux, w20, w50, v_fwm, v_width20, v_width50):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    #if sub3 > 8:
    ax1.set_xlabel(r'Velocity [km\,s$^{-1}$]')
    #if sub3 == 1 or sub3 == 5 or sub3 == 9:
    ax1.set_ylabel(r'Flux Density [Jy]')
    #half_vel = 4.0*len(velocity)/2.0
    #ax1.set_xlim(vsys-half_vel, vsys+half_vel)
    flux = flux/1000.0
    fmax = np.max(flux)
    #busyfit_flux = busyfit_flux/1000.0
    plt.text(100, 2.75, r'$S_{\mathrm{peak},1}$', fontsize = 10)
    plt.text(220, 4.05, r'$S_{\mathrm{peak},2}$', fontsize = 10)
    plt.text(160, 1.25, r'$I_{1}$', fontsize = 14)
    plt.text(250, 1.25, r'$I_{2}$', fontsize = 14)
    plt.plot([137, 160], [2.83, 2.83], linewidth=1.5, linestyle = '-', color = 'blue')
    plt.plot([262, 280], [4.13, 4.13], linewidth=1.5, linestyle = '-', color = 'blue')
    plt.plot([80, 330], [fmax/5., fmax/5.], linewidth=1, linestyle = '--', color = 'darkblue', label = r'$w_{20}$')
    plt.plot([80, 330], [fmax/2., fmax/2.], linewidth=1, linestyle = '--', color = 'peru', label = r'$w_{50}$')
    plt.axvline(v_fwm, linewidth=1, linestyle = '-', color = 'darkgreen', label = r'$V_{\mathrm{sys,fwm}}$')
    plt.axvline(v_width20, linewidth=1, linestyle = '--', color = 'darkred', label = r'$V_{\mathrm{sys},w_{20}}$')
    plt.axvline(v_width50, linewidth=1, linestyle = '-.', color = 'violet', label = r'$V_{\mathrm{sys},w_{50}}$')
    plt.plot(velocity, flux, linestyle = '-', color = 'black', linewidth = 1.0)#, label = txtstr)
    #frequency = (1.420405751786*pow(10,9))/(velocity/C_LIGHT + 1)
    #if colour == 'peru':
    #plt.plot(vel_bf, busyfit_flux, linestyle = '--', color = 'peru', linewidth = 0.75, label = 'BusyFit')
    #error = 1000.0*np.array(error)/1000.0
    vlow, vhigh = [], []
    flow, fhigh = [], []
    mlow, mhigh = [], []
    for i in range(len(velocity)):
      if velocity[i] > v_width20 - w20/2. and velocity[i] <= v_width20:
        vlow.append(velocity[i])
        flow.append(flux[i])
        mlow.append(0)
      if velocity[i] <= v_width20 + w20/2. and velocity[i] > v_width20:
        vhigh.append(velocity[i])
        fhigh.append(flux[i])
        mhigh.append(0)
    plt.fill_between(vlow, mlow, flow, alpha=0.5, edgecolor='none', facecolor='grey')
    plt.fill_between(vhigh, mhigh, fhigh, alpha=0.5, edgecolor='none', facecolor='black')
    #if sub3 == 4:
    ax1.legend(loc='upper right', fontsize = 10)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    ax1.get_yaxis().set_tick_params(which='both', direction='in')
    ax1.get_xaxis().set_tick_params(which='both', direction='in')
    #return velocity, flux

# ================================ #
# ======= Example Map Plot ======= #
# ================================ #
def map_example_plot(fig_num, sub1, sub2, sub3, model, txtstr):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    ax1.set_xlabel(r'x [pixels]')
    if sub3 == 1:
      ax1.set_ylabel(r'y [pixels]')
    else:
      ax1.set_yticklabels([])
    plt.text(5, 90, txtstr)
    image = ax1.imshow(model, origin='lower', interpolation='none', cmap='hot_r')
    image.set_clim(0,15)
    if sub3 == 3:
      cbar = plt.colorbar(image, fraction=0.05, pad=-0.05)
      cbar.set_label('Intensity')
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    ax1.get_yaxis().set_tick_params(which='both', direction='in')
    ax1.get_xaxis().set_tick_params(which='both', direction='in')
    #return velocity, flux


# ================================ #
# ===== Model Spectra Plot ======= #
# ================================ #
def spec_model_plot(fig_num, sub1, sub2, sub3, velocity, flux, vel_bf, busyfit_flux, error, colour, values, txtstr, vsys, vsys_bf):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    if sub3 == 8:
      ax1.set_xlabel(r'Velocity [km/s]')
    if sub3 == 4:
      ax1.set_ylabel(r'Flux [mJy]')
    #half_vel = 12.5*len(velocity)/3
    #ax1.set_xlim(np.min(velocity)-10, np.max(velocity)+10)
    plt.text(np.min(velocity), 1*np.max(flux)/10, values)
    #plt.axhline(0, linewidth=0.75, linestyle = ':', color = 'darkgrey')
    plt.axvline(vsys, linewidth=0.75, linestyle = '--', color = 'darkgrey')#, label = r'$V_{\mathrm{sys}}$')
    plt.axvline(vsys_bf, linewidth=0.75, linestyle = ':', color = 'black')#, label = r'$V_{\mathrm{sys,BF}}$')
    plt.plot(velocity, flux, linestyle = '-', color = colour, linewidth = 1.0, label = txtstr)
    #plt.plot(velocity, np.flip(flux,0), linestyle = '--', color = 'gray', linewidth = 1.0)
    #frequency = (1.420405751786*pow(10,9))/(velocity/C_LIGHT + 1)
    #if colour == 'peru':
    plt.plot(vel_bf, busyfit_flux, linestyle = '--', color = 'peru', linewidth = 0.75)#, label = 'BusyFit')
    error = np.array(error)
    plt.fill_between(velocity, flux-error, flux+error, alpha=0.5, edgecolor='none', facecolor='lightblue')
    #if sub3 == 3:
    ax1.legend(loc='upper right', fontsize = 8.5)
    #plt.subplots_adjust(wspace=0, hspace=0)
    ax1.get_yaxis().set_tick_params(which='both', direction='in')
    ax1.get_xaxis().set_tick_params(which='both', direction='in')
    #return velocity, flux

# ================================ #
# ===== Model Parameter Plot ===== #
# ================================ #
def asym_model_plot(fig_num, sub1, sub2, sub3, param_fit, param_true, param_str, txtstr):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    ax1.set_xlabel(param_str)
    if sub3 == 1 or sub3 == 5 or sub3 == 9 or sub3 ==13:
      ax1.set_ylabel(r'Number')
    param_fit = [x for x in param_fit if ~np.isnan(x)]
    param_fit = [x for x in param_fit if x != 0]
    if sub3 < 6:
      param_fit = np.clip(param_fit, np.median(param_fit) - 200, np.median(param_fit) + 200)
    elif sub3 > 5 and sub3 < 12:
      param_fit = np.clip(param_fit, np.median(param_fit) - 50, np.median(param_fit) + 50)
    else:
      param_fit = np.clip(param_fit, 0, 10)
    plt.axvline(param_true, linewidth=0.75, linestyle = ':', color = 'black', label = 'True')
    plt.axvline(np.mean(param_fit), linewidth=0.75, linestyle = '--', color = 'grey', label='Mean')
    plt.hist(param_fit, bins=10, color='darkblue', histtype='step', label=txtstr)
    if sub3 == 12:
      ax1.legend(loc='upper right', fontsize = 8.5)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    return np.mean(param_fit), np.std(param_fit)

def asym_hist_plot(fig_num, sub1, sub2, sub3, param_fit, bins, param_str, col, lsty, lbl):
    matplotlib.rcParams.update({'font.size': 11.5})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    ax1.set_xlabel(param_str)
    if sub3 == 1 or sub3 == 4 or sub3 == 7: #6
      ax1.set_ylabel(r'Number')
    #param_fit = [x for x in param_fit if ~np.isnan(x)]
    #plt.axvline(param_true, linewidth=0.75, linestyle = ':', color = 'black', label = 'True')
    #plt.axvline(np.mean(param_fit), linewidth=0.75, linestyle = '--', color = 'grey', label='Mean')
    hist_y, hist_x, _ = plt.hist(param_fit, bins=bins, color=col, linestyle=lsty, linewidth=1.5, histtype='step', label=lbl) #, density=True
    y_vals = ax1.get_yticks()
    #print y_vals
    #y_scaled = y_vals/y_vals[len(y_vals) - 1]
    #for i in range(len(y_scaled)):
    #  y_scaled[i] = round(y_scaled[i], 2)
    #ax1.set_yticklabels(y_scaled)
    #if param_str == r'Moment 0 Residual':
    #  ax1.set_xscale('log')
    if param_str == r'Flux Asym':
      x = []
      for i in range(50):
        x.append(1.0 + 0.01*i)
      hgauss = half_gaussian_poly(x, np.max(hist_y), 0.13, 1.0)
      #print x
      #print hgauss
      ax1.plot(x, hgauss, color=col, linestyle='-', linewidth=1)
    if sub3 == 1:
      ax1.legend(loc='upper right', fontsize = 8.5)
    plt.subplots_adjust(wspace=0.2, hspace=0.3)

def parameter_hist_plot(fig_num, sub1, sub2, sub3, param_fit, param_str, col, htype, lbl):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    ax1.set_xlabel(param_str)
    #if sub3 == 1 or sub3 == 5 or sub3 == 9 or sub3 ==13:
    if sub3 == 1 or sub3 == 4:
      ax1.set_ylabel(r'Number')
    #for i in range(len(param_fit)):
      #param_fit[i] = [x for x in param_fit[i] if ~np.isnan(x)]
    #plt.axvline(param_true, linewidth=0.75, linestyle = ':', color = 'black', label = 'True')
    #plt.axvline(np.mean(param_fit), linewidth=0.75, linestyle = '--', color = 'grey', label='Mean')
    bins = 10
    #if param_str == r'$\log(M_*/\mathrm{M}_{\odot})$':
      #bins=[6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11]
    #if param_str == r'$\log(M_{\mathrm{HI}}/\mathrm{M}_{\odot})$':
      #bins=[6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5]
    #if param_str == r'$M_{B}$':
      #bins=[-22, -21, -20, -19, -18, -17, -16, -15]
    #if param_str == r'$\log(\mathrm{SFR}/[\mathrm{M}_{\odot}\,\mathrm{yr}^{-1}])$':
      #bins=[-3.5, -3, -2.5, -2, -1.5, -1, -0.5,  0, 0.5, 1]
    #if param_str == r'$D_{25}$ [kpc]':
      #bins=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
      ##bins=[0, 10, 20, 30, 40, 50]
    #if param_str == r'$D_{\mathrm{HI}}$ [kpc]':
      #bins=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
      ##bins=[0, 10, 20, 30, 40, 50, 60]
      ##bins=[0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400]
    #if param_str == r'$\log(\mathrm{sSFR}/[\mathrm{yr}^{-1}])$':
      #bins = [-12.5, -12, -11.5, -11, -10.5, -10, -9.5, -9, -8.5]
    #if param_str == r'$\log(\mathrm{SNR})$':
      ##bins = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]
      #bins=15
    n,bins,patches = plt.hist(param_fit, bins=bins, color=col, linestyle='-', stacked=False, histtype=htype, label=lbl)
    #hatches = hsty
    #for patch_set, hatch in zip(patches, hatches):
      #for patch in patch_set.patches:
        #patch.set_hatch(hatch)
    y_vals = ax1.get_yticks()
    if sub3 == 1:
      ax1.legend(loc='upper right', fontsize = 11)
    plt.subplots_adjust(wspace=0.15, hspace=0.3)


def mstar_hist_plot(fig_num, sub1, sub2, sub3, param_fit, param_str, col, lsty, lbl):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    ax1.set_xlabel(param_str)
    #if sub3 == 1 or sub3 == 5 or sub3 == 9 or sub3 ==13:
    if sub3 == 1 or sub3 == 4:
      ax1.set_ylabel(r'Number')
    for i in range(len(param_fit)):
      param_fit[i] = [x for x in param_fit[i] if ~np.isnan(x)]
    #plt.axvline(param_true, linewidth=0.75, linestyle = ':', color = 'black', label = 'True')
    #plt.axvline(np.mean(param_fit), linewidth=0.75, linestyle = '--', color = 'grey', label='Mean')
    if param_str == r'$\log(M_*/\mathrm{M}_{\odot})$':
      bins=[6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11]
    if param_str == r'$\log(M_{\mathrm{HI}}/\mathrm{M}_{\odot})$':
      bins=[6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5]
    if param_str == r'$M_{B}$':
      bins=[-22, -21, -20, -19, -18, -17, -16, -15]
    if param_str == r'$\log(\mathrm{SFR}/[\mathrm{M}_{\odot}\,\mathrm{yr}^{-1}])$':
      bins=[-3.5, -3, -2.5, -2, -1.5, -1, -0.5,  0, 0.5, 1]
    if param_str == r'$D_{25}$ [kpc]':
      bins=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
      #bins=[0, 10, 20, 30, 40, 50]
    if param_str == r'$D_{\mathrm{HI}}$ [kpc]':
      bins=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
      #bins=[0, 10, 20, 30, 40, 50, 60]
      #bins=[0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400]
    if param_str == r'$\log(\mathrm{sSFR}/[\mathrm{yr}^{-1}])$':
      bins = [-12.5, -12, -11.5, -11, -10.5, -10, -9.5, -9, -8.5]
    if param_str == r'$\log(\mathrm{SNR})$':
      #bins = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]
      bins=15
    n,bins,patches = plt.hist(param_fit, bins=bins, color=col, linestyle=lsty, stacked=True, histtype='bar', label=lbl)
    #hatches = hsty
    #for patch_set, hatch in zip(patches, hatches):
      #for patch in patch_set.patches:
        #patch.set_hatch(hatch)
    y_vals = ax1.get_yticks()
    if sub3 == 6:
      ax1.legend(loc='upper right', fontsize = 11)
    plt.subplots_adjust(wspace=0.15, hspace=0.3)

def wallaby_hist_plot(fig_num, sub1, sub2, sub3, param_fit, param_str, col, lsty, lbl):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    ax1.set_xlabel(param_str)
    ax1.set_ylabel(r'Number')
    ax1.set_yscale('log')
    if sub3 == 1:
      bins=[-3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
      #plt.axvline(0.19, linewidth=2.5, linestyle = '-', color = 'skyblue', label = 'LVHIS')
      #plt.axvline(1.36, linewidth=2.5, linestyle = '--', color = 'goldenrod', label = 'VIVA')
      #plt.axvline(0.18, linewidth=2.5, linestyle = ':', color = 'black', label = 'HALOGAS')
      #plt.scatter(0.19, 73, color='skyblue', s=40, marker='o', label='LVHIS')
      #plt.scatter(1.36, 47, color='goldenrod', s=40, marker='s', label='VIVA')
      #plt.scatter(0.18, 18, color='black', s=40, marker='d', label='HALOGAS')
      #plt.errorbar(-1.64, 73, xerr = 0.33, color='skyblue', markersize=7, fmt='o', elinewidth=1, label = 'LVHIS')
      #plt.errorbar(0.72, 47, xerr = 0.36, color='goldenrod', markersize=7, fmt='s', elinewidth=1, label = 'VIVA')
      #plt.errorbar(-1.09, 18, xerr = 0.30, color='black', markersize=7, fmt='d', elinewidth=1, label = 'HALOGAS')
    if sub3 == 2:
      bins=[6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12] #5, 5.5, 6, 6.5, 7, 
    n,bins,patches = plt.hist(param_fit, bins=bins, color=col, linestyle=lsty, stacked=True, linewidth=3, histtype='stepfilled', label=lbl)
    #n,bins,patches = plt.hist(param_fit, bins=bins, color=col, linestyle=lsty, stacked=True, histtype='bar', label=lbl)
    y_vals = ax1.get_yticks()
    if sub3 == 1:
      ax1.legend(loc='upper right', fontsize = 10)
    plt.subplots_adjust(wspace=0.15, hspace=0.3)


# ================================ #
# ===== Model Parameter Plot ===== #
# ================================ #
def asym_comp_plot(fig_num, sub1, sub2, sub3, param_true, param_mean, param_std, txtstr):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    if sub3 > 8:
      ax1.set_xlabel('True')
    if sub3 == 1 or sub3 == 5 or sub3 == 9 or sub3 ==13:
      ax1.set_ylabel('Mean Fit')
    plt.plot([-100,100], [-100,100], color='black', linewidth=1, linestyle = '--')
    #plt.scatter(flux1, flux2, color=colour, s=20, label=lbl)
    if txtstr == r'Flux Asym':
      ax1.set_xlim(0.95, 2.5)
    ax1.set_xlim(np.min(param_true)-np.max(param_std), np.max(param_true)+np.max(param_std))
    ax1.set_ylim(np.min(param_mean)-np.max(param_std), np.max(param_mean)+np.max(param_std))
    plt.errorbar(param_true, param_mean, yerr = param_std, color='darkblue', markersize=2, fmt='o', elinewidth=0.75, label = txtstr)
    #if sub3 == 10:
    ax1.legend(loc='upper left', fontsize = 8.5)
    plt.subplots_adjust(wspace=0.25, hspace=0.2)

# ================================ #
# ===== Model Parameter Plot ===== #
# ================================ #
def sofia_bf_plot(fig_num, sub1, sub2, sub3, param_sofia, param_bf, txtstr):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    if sub3 > 3:
      ax1.set_xlabel('SoFiA')
    if sub3 == 1 or sub3 == 4:
      ax1.set_ylabel('Busy Function')
    plt.plot([-10,np.max(param_bf)], [-10,np.max(param_bf)], color='black', linewidth=1, linestyle = '--')
    #plt.scatter(flux1, flux2, color=colour, s=20, label=lbl)
    plt.scatter(param_sofia, param_bf, color='darkblue', s=5, label = txtstr)
    #if sub3 == 10:
    ax1.legend(loc='upper left', fontsize = 8.5)
    plt.subplots_adjust(wspace=0.25, hspace=0.2)

# ================================ #
# === Parameter Comparison Plot == #
# ================================ #
def param_comparison_plot(fig_num, sub1, sub2, sub3, param1, param2, param2_err, xlbl, ylbl):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    #if sub3 > 3:
    ax1.set_xlabel(xlbl)
    #if sub3 == 1 or sub3 == 4:
    ax1.set_ylabel(ylbl)
    plt.plot([-10,np.max(param1)], [-10,np.max(param1)], color='peru', linewidth=1, linestyle = '--')
    #plt.scatter(flux1, flux2, color=colour, s=20, label=lbl)
    if ylbl != r'HIPASS $S_{\mathrm{int}}$':
      plt.scatter(param1, param2, color='darkblue', s=5)
    else:
      plt.errorbar(param1, param2, yerr = param2_err, color='darkblue', markersize=2, fmt='o', elinewidth=0.75)
    #if sub3 == 10:
    #ax1.legend(loc='upper left', fontsize = 8.5)
    plt.subplots_adjust(wspace=0.45, hspace=0.3)


def pcomp_hist_plot(fig_num, sub1, sub2, sub3, param1, xlbl):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    ax1.set_xlabel(xlbl)
    ax1.set_ylabel(r'Number')
    #ax1.set_xscale('log')
    plt.hist(param1, bins=10, color='darkblue', histtype='step', linestyle='-', linewidth=1.5)
    #if sub3 == 4:
      #ax1.legend(loc='upper right', fontsize = 8.5)
    #if xlbl == r'Density [Mpc$^{-3}$]':
      #plt.subplots_adjust(wspace=0.15, hspace=0.15)
    #else:
    plt.subplots_adjust(wspace=0.3, hspace=0.3)


# ================================ #
# ===== Model Parameter Plot ===== #
# ================================ #
def asym_param_comp_plot(fig_num, sub1, sub2, sub3, param1, param2, param1_std, param2_std, txtstr1, txtstr2):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    #if sub3 > 5:
    ax1.set_xlabel(txtstr1)
    #if sub3 == 1 or sub3 == 6:
    ax1.set_ylabel(txtstr2)
    ax1.set_xlim(np.min(param1)-np.max(param1_std), np.max(param1)+np.max(param1_std))
    ax1.set_ylim(np.min(param2)-np.max(param2_std), np.max(param2)+np.max(param2_std))
    #plt.hist(param_fit, bins=10, color='darkblue', histtype='step', label=txtstr)
    plt.plot([-100,10000], [-100,10000], color='black', linewidth=1, linestyle = '--')
    #plt.scatter(flux1, flux2, color=colour, s=20, label=lbl)
    plt.errorbar(param1, param2, xerr = param1_std, yerr = param2_std, color='darkblue', markersize=2, fmt='o', elinewidth=0.75)
    #if sub3 == 10:
    #ax1.legend(loc='upper left', fontsize = 8.5)
    plt.subplots_adjust(wspace=0.25, hspace=0.2)

# ================================ #
# == Environment/Parameter Plot == #
# ================================ #
def env_param_plot(fig_num, sub1, sub2, sub3, param1, param2, param1_std, param2_std, do_mean, colour, txtstr1, txtstr2, lbl, point_size, msty):
    if sub1 == 1 and sub2 == 1:
      matplotlib.rcParams.update({'font.size': 12})
    else:
      matplotlib.rcParams.update({'font.size': 11.5})
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    if sub1 > 1 and sub2 > 1:
      if sub3 > 6: #4
        ax1.set_xlabel(txtstr1)
      if txtstr1 == r'Density [$\log(\rho_{10}/\mathrm{Mpc}^{-3})$]':
        ax1.set_xlim(-2.5, 1.5)
      if txtstr1 == r'$\log(\mathrm{SFR}/[\mathrm{M}_{\odot}\,\mathrm{yr}^{-1}])$':
        ax1.set_xlim(-4.5, 1.5)
      if txtstr1 == r'$\log(\mathrm{sSFR}/[\mathrm{yr}^{-1}])$':
        ax1.set_xlim(-12.5, -8.5)
      if txtstr1 == r'$\log(M_*/\mathrm{M}_{\odot})$':
        ax1.set_xlim(6, 11.5)
      if txtstr1 == r'$\log(M_{\mathrm{HI}}/\mathrm{M}_{\odot})$':
        ax1.set_xlim(6, 11.5)
      if txtstr1 == r'$\log(\mathrm{SNR})$':
        ax1.set_xlim(0.5, 4)
      if sub3 < 7: #6
        ax1.set_xticklabels([])
    if sub1 == 1 and (sub2 == 1 or sub2 == 2):
      ax1.set_xlabel(txtstr1)
      if txtstr1 == r'Density [$\log(\rho_{10}/\mathrm{Mpc}^{-3})$]':
        ax1.set_xlim(-2.5, 1.5)
    if sub1 == 1 and sub2 == 2:
      if sub3 == 1 and lbl == 'LVHIS' and point_size == 2.5:
        plt.text(0.05, 0.89, r'$9 \leq \log(M_*/\mathrm{M}_{\odot}) \leq 10$', transform=ax1.transAxes, fontsize=10)
      if sub3 == 2 and lbl == 'LVHIS' and point_size == 2.5:
        plt.text(0.05, 0.89, r'$9.5 \leq \log(M_*/\mathrm{M}_{\odot}) \leq 10$', transform=ax1.transAxes, fontsize=10)
    ax1.set_ylabel(txtstr2)
    if txtstr2 == r'$\langle A_{1,r/R_{25}>1}\rangle$':
      ax1.set_yticks((0, 0.2, 0.4, 0.6, 0.8))
    if txtstr2 == r'$\langle A_{1,r/R_{25}<1}\rangle$':
      ax1.set_yticks((0, 0.2, 0.4, 0.6, 0.8))
    if lbl == 'LVHIS':
      zorder = 3
    if lbl == 'VIVA':
      zorder = 2
    if lbl == 'HALOGAS':
      zorder = 1
    if lbl == 'HIPASS':
      zorder = 1
    else:
      zorder = 1
    #if lbl == r'NGC\,7232':
      #zorder = 1
    #if lbl == r'NGC\,1566':
      #zorder = 1
    #if lbl == r'IC5201':
      #zorder = 1
    #if lbl == r'LGG351':
      #zorder = 1
    #if lbl == r'ERIDANUS':
      #zorder = 1
    #if lbl
    #if txtstr1 == r'Type D=1, S=2, I=3':
      #ax1.set_xlim(0, 4)
    ##elif txtstr1 == r'Sigma [Mpc]':
    ##  ax1.set_xlim(0, 6)
    #else:
      ##ax1.set_xlim(0.05, 10)
      #ax1.set_xscale('log')
    #if txtstr2 == r'$\epsilon_{\mathrm{kin}}$' or txtstr2 == r'Spec Res':
      #ax1.set_ylim(0, 5)
    #  ax1.set_yscale('log')
    #ax1.set_ylim(0.9, 2)
    #ax1.set_xlim(np.min(param1)-np.max(param1_std), np.max(param1)+np.max(param1_std))
    #ax1.set_ylim(np.min(param2)-np.max(param2_std), np.max(param2)+np.max(param2_std))
    #plt.hist(param_fit, bins=10, color='darkblue', histtype='step', label=txtstr)
    #plt.plot([-10,20], [-10,20], color='black', linewidth=1, linestyle = '--')
    if do_mean == False:
      scatter = plt.scatter(param1, param2, color=colour, s=point_size, facecolors=colour, alpha=0.25) #'none', edgecolors=
      #, label=lbl)
    else:
      #plt.scatter(param1, param2, color=colour, s=point_size, alpha=1, label=lbl)
      ebar = plt.errorbar(param1, param2, xerr=param1_std, yerr = param2_std, color=colour, markersize=point_size, fmt=msty, alpha=1, elinewidth=0.75, label=lbl, zorder=zorder)
      if sub1 == 1 and sub2 == 1:
        ax1.legend(loc='upper left', fontsize = 11)
      #else:
      if sub3 == 1:
        ax1.legend(loc='upper left', fontsize = 8.5)
      if sub1 == 1 and sub2 == 2:
        if sub3 == 1:
          ax1.legend(loc=(0.05,0.58), fontsize = 8.5) #'center left'
    plt.subplots_adjust(wspace=0.45, hspace=0.0)


# ================================ #
# == Environment/Parameter Plot == #
# ================================ #
def param_corr_plot(fig_num, sub1, sub2, sub3, param1, param2, param2_std, do_mean, colour, txtstr1, txtstr2, lbl, point_size, msty, corr, pval):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    #if sub3 > 3:
    #if sub3 == 1 or sub3 == 10 or sub3 == 19 or sub3 == 28 or sub3 == 37 or sub3 == 46 or sub3 == 55 or sub3 == 64 or sub3 == 73:
    if sub3 == 1 or sub3 == 9 or sub3 == 17 or sub3 == 25 or sub3 == 33 or sub3 == 41 or sub3 == 49 or sub3 == 57:
      ax1.set_ylabel(txtstr2)
    else:
      ax1.set_yticklabels([])
    if sub3 > 56: #72
      ax1.set_xlabel(txtstr1)
    else:
      ax1.set_xticklabels([])
    if txtstr1 == r'Density [$\log(\rho_{10}/\mathrm{Mpc}^{-3})$]':
      ax1.set_xlim(-2.5, 1.5)
    if txtstr1 == r'$A_{\mathrm{spec}}$':
      ax1.set_xticks((0.0, 0.3))
    if txtstr1 == r'$A_{\mathrm{map}}$':
      ax1.set_xticks((0.0, 0.3, 0.6))
    if txtstr2 == r'$A_{\mathrm{map}}$':
      ax1.set_yticks((0.0, 0.3, 0.6))
    #elif txtstr1 == r'Sigma [Mpc]':
      #ax1.set_xlim(0.2, 20)
    #if sub3 == 1 or sub3 == 6:
    #if sub3 < 7:
      #ax1.set_xticklabels([])
    corr = round(corr, 2)
    if np.abs(corr) < 0.0045:
      corr = 0.0
    if lbl == 'LVHIS':
      #plt.text(1*(np.nanmax(param1)+np.nanmin(param1))/5, 4*(np.nanmax(param2)+np.nanmin(param2))/5, corr)
      if pval < 0.01:
        plt.text(0.65, 0.85, r'\textbf{%.2f}' % corr, transform=ax1.transAxes)
      else:
        plt.text(0.65, 0.85, r'%.2f' % corr, transform=ax1.transAxes)
    plt.scatter(param1, param2, color=colour, s=point_size, marker=msty, label=lbl)
    if sub3 == 1:
      ax1.legend(loc='upper left', bbox_to_anchor=(1.15, 0.95), fontsize = 11)
    plt.subplots_adjust(wspace=0.0, hspace=0.0)

# ================================ #
# == Environment Variation Plot == #
# ================================ #
def density_var_plot(fig_num, sub1, sub2, sub3, param1, param2, param2_std, colour, lbl, msty):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    #if sub3 > 3:
    #  ax1.set_xlabel(txtstr1)
    #if txtstr1 == r'Density [$\log(\rho/\mathrm{Mpc}^{-3})$]':
    ax1.set_xlim(0, 21)
    #elif txtstr1 == r'Sigma [Mpc]':
      #ax1.set_xlim(0.2, 20)
    #if sub3 == 1 or sub3 == 6:
    #if sub3 < 7:
    ax1.set_xticks([1, 3, 6, 10, 12, 20])
    if lbl == 'LVHIS':
      zorder = 2
      fill_style = 'full'
      width = 1.5
      alpha = 0.2
    if lbl == 'VIVA':
      zorder = 2
      fill_style = 'full'
      width = 1.5
      alpha = 0.2
    if lbl == 'HALOGAS':
      zorder = 2
      fill_style = 'full'
      width = 1.5
      alpha = 0.2
    ax1.set_xlabel('Nearest Neighbours')
    ax1.set_ylabel(r'Density $\log(\rho_{10}/\mathrm{Mpc}^{-3})$')
    plt.plot(param1, param2, color=colour, marker=msty, alpha=1, linewidth=width, label=lbl, fillstyle=fill_style)
    plt.fill_between(param1, param2-param2_std, param2+param2_std, alpha=alpha, edgecolor='none', zorder=zorder, facecolor=colour)
    if sub3 == 1:
      ax1.legend(loc='upper right', fontsize = 8.5)
    plt.subplots_adjust(wspace=0.38, hspace=0.0)

# ================================ #
# == Asymmetry Noise Stats Plot == #
# ================================ #
def asym_stats_plot(fig_num, sub1, sub2, sub3, noise, asym_param, asym_std, param_true, txtstr):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    if sub3 > 10:
      ax1.set_xlabel(r'Noise Level')
    ax1.set_ylabel(txtstr)
    if sub3 < 11:
      ax1.set_xticklabels([])
    #ax1.set_ylim(0, 0.39)
    #ax1.set_xlim(0, 400)
    #half_vel = 12.5*len(velocity)/3
    #ax1.set_xlim(np.min(velocity)-10, np.max(velocity)+10)
    #plt.text(np.min(velocity), 1*np.max(flux)/10, values)
    plt.axhline(param_true, linewidth=1.0, linestyle = '--', color = 'black')
    #plt.axvline(vsys, linewidth=0.75, linestyle = '--', color = 'darkgrey')#, label = r'$V_{\mathrm{sys}}$')
    #plt.axvline(vsys_bf, linewidth=0.75, linestyle = ':', color = 'black')#, label = r'$V_{\mathrm{sys,BF}}$')
    #plt.fill_between(radius, kin_param-error, kin_param+error, alpha=0.5, edgecolor='none', facecolor='lightblue')
    #plt.plot(radius, kin_param, linestyle = '-', color = 'darkblue', linewidth = 1.0)
    plt.errorbar(noise, asym_param, yerr=asym_std, marker = 'o', markersize=2, linewidth=0.5, color = 'darkblue', ls='none', label = txtstr) # facecolors='none', edgecolors='darkblue'
    plt.subplots_adjust(wspace=0.3, hspace=0)
    #ax1.legend(loc='upper right', fontsize = 11)
    ax1.get_yaxis().set_tick_params(which='both', direction='in')
    ax1.get_xaxis().set_tick_params(which='both', direction='in')
    #return velocity, flux

# ================================ #
# ======= Kinemetry Plot ========= #
# ================================ #
def kinemetry_plot(fig_num, sub1, sub2, sub3, radius, kin_param, error, txtstr, morphology):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    if sub3 == sub1*sub2:
      ax1.set_xlabel(r'Radius [arcsec]')
    ax1.set_ylabel(r'$v_{\mathrm{asym}}$')
    if sub3 != sub1*sub2:
      ax1.set_xticklabels([])
    ax1.set_ylim(0, 0.9)
    ax1.set_xlim(0, 600)
    #half_vel = 12.5*len(velocity)/3
    #ax1.set_xlim(np.min(velocity)-10, np.max(velocity)+10)
    plt.text(20, 0.5, morphology)
    #plt.axhline(0, linewidth=0.75, linestyle = ':', color = 'darkgrey')
    #plt.axvline(vsys, linewidth=0.75, linestyle = '--', color = 'darkgrey')#, label = r'$V_{\mathrm{sys}}$')
    #plt.axvline(vsys_bf, linewidth=0.75, linestyle = ':', color = 'black')#, label = r'$V_{\mathrm{sys,BF}}$')
    plt.fill_between(radius, kin_param-error, kin_param+error, alpha=0.5, edgecolor='none', facecolor='lightblue')
    plt.plot(radius, kin_param, linestyle = '-', color = 'darkblue', linewidth = 1.0)
    plt.scatter(radius, kin_param, marker = 'o', s=10, color = 'darkblue', label = txtstr) # facecolors='none', edgecolors='darkblue'
    plt.subplots_adjust(wspace=0, hspace=0)
    ax1.legend(loc='upper right', fontsize = 11)
    ax1.get_yaxis().set_tick_params(which='both', direction='in')
    ax1.get_xaxis().set_tick_params(which='both', direction='in')
    #return velocity, flux

# ================================ #
# ===== Kinemetry/Type Plot ====== #
# ================================ #
def kinemetry_type_plot(fig_num, sub1, sub2, sub3, galtype, kin_param, error_low, error_high, binned_values, txtstr):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    plt.rcParams['text.latex.unicode'] = True
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    if sub3 == sub1*sub2:
      ax1.set_xlabel(r'Barred (0)/Unbarred (2)')
    ax1.set_ylabel(r'$v_{\mathrm{asym}}$')
    if sub3 != sub1*sub2:
      ax1.set_xticklabels([])
    ax1.set_xlim(-0.5, 2.5)
    ax1.set_ylim(-0.05, 0.35)
    plt.text(2.05, -0.04, txtstr)
    plt.text(-0.25, 0.28, r'A: %.2f$\pm$%.2f' % (binned_values[0], binned_values[1]))
    plt.text(0.75, 0.28, r'AB: %.2f$\pm$%.2f' % (binned_values[2], binned_values[3]))
    plt.text(1.75, 0.28, r'B: %.2f$\pm$%.2f' % (binned_values[4], binned_values[5]))
    #plt.axhline(0, linewidth=0.75, linestyle = ':', color = 'darkgrey')
    #plt.axvline(vsys, linewidth=0.75, linestyle = '--', color = 'darkgrey')#, label = r'$V_{\mathrm{sys}}$')
    #plt.axvline(vsys_bf, linewidth=0.75, linestyle = ':', color = 'black')#, label = r'$V_{\mathrm{sys,BF}}$')
    #plt.fill_between(radius, kin_param-error, kin_param+error, alpha=0.5, edgecolor='none', facecolor='lightblue')
    #plt.plot(radius, kin_param, linestyle = '-', color = 'darkblue', linewidth = 1.0)
    #plt.scatter(radius, kin_param, marker = 'o', s=10, color = 'darkblue', label = txtstr) # facecolors='none', edgecolors='darkblue'
    lowbar = np.array(kin_param) - np.array(error_low)
    highbar = np.array(error_high) - np.array(kin_param)
    plt.errorbar(galtype, kin_param, yerr=[lowbar, highbar], marker = 'o', markersize=2, linewidth=0.5, color = 'darkblue', ls='none', label = txtstr)
    plt.subplots_adjust(wspace=0, hspace=0)
    #ax1.legend(loc='upper right', fontsize = 11)
    ax1.get_yaxis().set_tick_params(which='both', direction='in')
    ax1.get_xaxis().set_tick_params(which='both', direction='in')
    #return velocity, flux

# ================================ #
# ====== Moment 0 FD Plot ======== #
# ================================ #
def fd_mom0_plot(fig_num, sub1, sub2, sub3, radius, a_ratio, error, txtstr, a_max, a_mean, a_std):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    if sub3 == sub1*sub2:
      ax1.set_xlabel(r'Radius [$r$/$r_{\mathrm{scalelength}}$]')
    ax1.set_ylabel(r'$A_{1}$')
    if sub3 != sub1*sub2:
      ax1.set_xticklabels([])
    ax1.set_ylim(0, 2)#np.nanmax(a_ratio) + np.nanmean(error))
    ax1.set_xlim(0, 5)
    #half_vel = 12.5*len(velocity)/3
    #ax1.set_xlim(np.min(velocity)-10, np.max(velocity)+10)
    #txtstr = r'%s\\ A_{\mathrm{max}}=%.2f\\ A_{\mathrm{mean}}=%.2f\pm%.2f' % (gals[i], a_max, a_mean, a_std)
    plt.text(3, 1.75, '%s' % (txtstr))
    plt.text(3, 1.5, r'$A_{\mathrm{max}}=%.2f$' % (a_max))
    plt.text(3, 1.25, r'$A_{\mathrm{mean}}=%.2f\pm%.2f$' % (a_mean, a_std))
    #plt.axhline(0, linewidth=0.75, linestyle = ':', color = 'darkgrey')
    #plt.axvline(vsys, linewidth=0.75, linestyle = '--', color = 'darkgrey')#, label = r'$V_{\mathrm{sys}}$')
    #plt.axvline(vsys_bf, linewidth=0.75, linestyle = ':', color = 'black')#, label = r'$V_{\mathrm{sys,BF}}$')
    plt.fill_between(radius, a_ratio-error, a_ratio+error, alpha=0.5, edgecolor='none', facecolor='lightblue')
    plt.plot(radius, a_ratio, linestyle = '-', color = 'darkblue', linewidth = 1.0)
    plt.scatter(radius, a_ratio, marker = 'o', s=10, color = 'darkblue')#, label = txtstr) # facecolors='none', edgecolors='darkblue'
    plt.subplots_adjust(wspace=0, hspace=0)
    #ax1.legend(loc='upper left', fontsize = 11)
    ax1.get_yaxis().set_tick_params(which='both', direction='in')
    ax1.get_xaxis().set_tick_params(which='both', direction='in')
    #return velocity, flux


# ================================ #
# === Density/Sigma Histograms === #
# ================================ #
def density_hist_plot(fig_num, sub1, sub2, sub3, density, colour, lsty, txtstr, xlbl):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    if xlbl == r'Density [Mpc$^{-3}$]':
      if sub3 == 4:
        ax1.set_xlabel(xlbl)
      density = density[density < 20]
    else:
      ax1.set_xlabel(xlbl)
    ax1.set_ylabel(r'Number')
    #ax1.set_xscale('log')
    plt.hist(density, bins=15, color=colour, histtype='step', linestyle=lsty, linewidth=1.5, label=txtstr)
    if sub3 == 4:
      ax1.legend(loc='upper right', fontsize = 8.5)
    if xlbl == r'Density [Mpc$^{-3}$]':
      plt.subplots_adjust(wspace=0.15, hspace=0.15)
    else:
      plt.subplots_adjust(wspace=0.25, hspace=0.25)
    
# ================================ #
# ==== Sigma vs Redshift Plot ==== #
# ================================ #
def sigma_z_plot(fig_num, sub1, sub2, sub3, param1, param2, colour, msty, txtstr1, txtstr2, lbl):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    #if sub3 > 6:
    #  ax1.set_xlabel(txtstr1)
    #if sub3 == 1 or sub3 == 6:
    if sub3 != 4:
      ax1.set_xticklabels([])
    if sub3 == 4:
      ax1.set_xlabel(txtstr1)
    ax1.set_ylabel(txtstr2)
    #if txtstr2 == r'Density [Mpc$^{-3}$]':
    #ax1.set_yscale('log')
    #ax1.set_ylim(0.002, 500)
    #if lbl == 'LVHIS':
    #  ax1.set_ylim(0.0, np.max(param2)+1)
    #ax1.set_xlim(np.min(param1)-np.max(param1_std), np.max(param1)+np.max(param1_std))
    #ax1.set_ylim(np.min(param2)-np.max(param2_std), np.max(param2)+np.max(param2_std))
    #plt.hist(param_fit, bins=10, color='darkblue', histtype='step', label=txtstr)
    #plt.plot([-10,20], [-10,20], color='black', linewidth=1, linestyle = '--')
    plt.scatter(param1, param2, color=colour, s=6, marker=msty, label=lbl)
    #plt.errorbar(param1, param2, xerr = param1_std, yerr = param2_std, color='darkblue', markersize=2, fmt='o', elinewidth=0.75)
    if sub3 == 4:
      ax1.legend(loc='upper left', fontsize = 8.5)
    plt.subplots_adjust(wspace=0.0, hspace=0.0)


def density_estimation(m1, m2):
    X, Y = np.mgrid[min(m1):max(m1):100j, min(m2):max(m2):100j]                                                     
    positions = np.vstack([X.ravel(), Y.ravel()])                                                       
    values = np.vstack([m1, m2])                                                                        
    kernel = stats.gaussian_kde(values, bw_method=0.2)                                                                 
    Z = np.reshape(kernel(positions).T, X.shape)
    return X, Y, Z

# ================================ #
# ==== Sigma vs Distance Plot ==== #
# ================================ #
def sigma_dist_plot(fig_num, sub1, sub2, sub3, param1, param2, colour, msty, txtstr1, txtstr2, lbl):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    #if sub3 > 6:
    #  ax1.set_xlabel(txtstr1)
    #if sub3 == 1 or sub3 == 6:
    #if sub3 != 4:
      #ax1.set_xticklabels([])
    #if sub3 == 4:
    ax1.set_xlabel(txtstr1)
    ax1.set_ylabel(txtstr2)
    #if txtstr2 == r'Density [Mpc$^{-3}$]':
    #ax1.set_yscale('log')
    #ax1.set_ylim(0.002, 500)
    #if lbl == 'LVHIS':
    #  ax1.set_ylim(0.0, np.max(param2)+1)
    #ax1.set_xlim(np.min(param1)-np.max(param1_std), np.max(param1)+np.max(param1_std))
    #ax1.set_ylim(np.min(param2)-np.max(param2_std), np.max(param2)+np.max(param2_std))
    #plt.hist(param_fit, bins=10, color='darkblue', histtype='step', label=txtstr)
    #plt.plot([-10,20], [-10,20], color='black', linewidth=1, linestyle = '--')
    if lbl == 'WALLABY':
      #xy = np.vstack([param1, param2])
      #z  = gaussian_kde(xy)(xy)
      #idx = z.argsort()
      #x,y,z = param1[idx], param2[idx], z[idx]
      #plt.scatter(x, y, c=z, cmap='Greys', s=2, alpha=0.9, marker=msty, rasterized=True, zorder=1)
      #plt.scatter(param1, param2, c='grey', s=2.5, alpha=0.05, marker=msty, rasterized=True, zorder=1)
      nbins=20
      #k = kde.gaussian_kde(xy)
      #xi, yi = np.mgrid[param1.min():param1.max():nbins*1j, param2.min():param2.max():nbins*1j]
      #zi = k(np.vstack([xi.flatten(), yi.flatten()]))
      #print zi
      X,Y,Z = density_estimation(param1, param2)
      contour_set = plt.contour(X, Y, Z, levels=[0.0005, 0.0025, 0.01, 0.05, 0.1, 0.15, 0.2], colors='teal', zorder=2)
      print (contour_set.levels)
      #, 0.2, 0.25, 0.3, 0.35
      #plt.hist2d(param1,param2)
    else:
      plt.scatter(param1, param2, color=colour, s=25, marker=msty, label=lbl, zorder=3)
    #plt.errorbar(param1, param2, xerr = param1_std, yerr = param2_std, color='darkblue', markersize=2, fmt='o', elinewidth=0.75)
    if sub3 == 1:
      ax1.legend(loc='lower left', fontsize = 8.5, ncol=3)
    plt.subplots_adjust(wspace=0.0, hspace=0.0)

# ================================ #
# ==== Sigma vs Distance Plot ==== #
# ================================ #
def density_dist_plot(fig_num, sub1, sub2, sub3, param1, param2, colour, msty, txtstr1, txtstr2, lbl):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    #if sub3 > 6:
    #  ax1.set_xlabel(txtstr1)
    #if sub3 == 1 or sub3 == 6:
    #if sub3 != 4:
      #ax1.set_xticklabels([])
    #if sub3 == 4:
    ax1.set_xlabel(txtstr1)
    ax1.set_ylabel(txtstr2)
    #if txtstr2 == r'Density [Mpc$^{-3}$]':
    #ax1.set_yscale('log')
    #ax1.set_ylim(0.002, 500)
    #if lbl == 'LVHIS':
    #  ax1.set_ylim(0.0, np.max(param2)+1)
    #ax1.set_xlim(np.min(param1)-np.max(param1_std), np.max(param1)+np.max(param1_std))
    #ax1.set_ylim(np.min(param2)-np.max(param2_std), np.max(param2)+np.max(param2_std))
    #plt.hist(param_fit, bins=10, color='darkblue', histtype='step', label=txtstr)
    #plt.plot([-10,20], [-10,20], color='black', linewidth=1, linestyle = '--')
    #if lbl == 'WALLABY':
      ##xy = np.vstack([param1, param2])
      ##z  = gaussian_kde(xy)(xy)
      ##idx = z.argsort()
      ##x,y,z = param1[idx], param2[idx], z[idx]
      ##plt.scatter(x, y, c=z, cmap='Greys', s=2, alpha=0.9, marker=msty, rasterized=True, zorder=1)
      ##plt.scatter(param1, param2, c='grey', s=2.5, alpha=0.05, marker=msty, rasterized=True, zorder=1)
      #nbins=20
      ##k = kde.gaussian_kde(xy)
      ##xi, yi = np.mgrid[param1.min():param1.max():nbins*1j, param2.min():param2.max():nbins*1j]
      ##zi = k(np.vstack([xi.flatten(), yi.flatten()]))
      ##print zi
      #X,Y,Z = density_estimation(param1, param2)
      #contour_set = plt.contour(X, Y, Z, levels=[0.0005, 0.0025, 0.01, 0.05, 0.1, 0.15, 0.2], colors='teal', zorder=2)
      #print (contour_set.levels)
      ##, 0.2, 0.25, 0.3, 0.35
      ##plt.hist2d(param1,param2)
    #else:
    plt.scatter(param1, param2, color=colour, s=25, marker=msty, label=lbl, zorder=3)
    #plt.errorbar(param1, param2, xerr = param1_std, yerr = param2_std, color='darkblue', markersize=2, fmt='o', elinewidth=0.75)
    if sub3 == 1:
      ax1.legend(loc='lower right', fontsize = 8.5, ncol=2)
    plt.subplots_adjust(wspace=0.0, hspace=0.0)

# ================================ #
# ==== Distance vs Mstar Plot ==== #
# ================================ #
def dist_mstar_plot(fig_num, sub1, sub2, sub3, param1, param2, colour, msty, txtstr1, txtstr2, lbl):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    #if sub3 > 6:
    #  ax1.set_xlabel(txtstr1)
    #if sub3 == 1 or sub3 == 6:
    #if sub3 != 4:
      #ax1.set_xticklabels([])
    #if sub3 == 4:
    ax1.set_xlabel(txtstr1)
    ax1.set_ylabel(txtstr2)
    #if txtstr2 == r'Density [Mpc$^{-3}$]':
    #ax1.set_yscale('log')
    #ax1.set_ylim(0.002, 500)
    #if lbl == 'LVHIS':
    #  ax1.set_ylim(0.0, np.max(param2)+1)
    #ax1.set_xlim(np.min(param1)-np.max(param1_std), np.max(param1)+np.max(param1_std))
    #ax1.set_ylim(np.min(param2)-np.max(param2_std), np.max(param2)+np.max(param2_std))
    #plt.hist(param_fit, bins=10, color='darkblue', histtype='step', label=txtstr)
    #plt.plot([-10,20], [-10,20], color='black', linewidth=1, linestyle = '--')
    if lbl == 'WALLABY':
      #xy = np.vstack([param1, param2])
      #z  = gaussian_kde(xy)(xy)
      #idx = z.argsort()
      #x,y,z = param1[idx], param2[idx], z[idx]
      #plt.scatter(x, y, c=z, cmap='Greys', s=2, alpha=0.9, marker=msty, rasterized=True, zorder=1)
      #plt.scatter(param1, param2, c='grey', s=2.5, alpha=0.05, marker=msty, rasterized=True, zorder=1)
      nbins=20
      #k = kde.gaussian_kde(xy)
      #xi, yi = np.mgrid[param1.min():param1.max():nbins*1j, param2.min():param2.max():nbins*1j]
      #zi = k(np.vstack([xi.flatten(), yi.flatten()]))
      #print zi
      X,Y,Z = density_estimation(param1, param2)
      contour_set = plt.contour(X, Y, Z, levels=[0.0005, 0.0025, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3], colors='teal', zorder=2)
      print (contour_set.levels)
      #, 0.2, 0.25, 0.3, 0.35
      #plt.hist2d(param1,param2)
    else:
      plt.scatter(param1, param2, color=colour, s=25, marker=msty, label=lbl, zorder=3)
    #plt.errorbar(param1, param2, xerr = param1_std, yerr = param2_std, color='darkblue', markersize=2, fmt='o', elinewidth=0.75)
    ax1.legend(loc='upper right', fontsize = 10)
    plt.subplots_adjust(wspace=0.0, hspace=0.0)


def ratio_plot(fig_num, sub1, sub2, sub3, param1, param2, colour, txtstr1, txtstr2, lbl, txtstr):
    matplotlib.rcParams.update({'font.size': 11})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    #if sub3 > 6:
    #  ax1.set_xlabel(txtstr1)
    #if sub3 == 1 or sub3 == 6:
    #if sub3 != 6:
    #  ax1.set_xticklabels([])
    if sub3 == 2:
      ax1.set_xlabel(txtstr1)
    ax1.set_ylabel(txtstr2)
    #ax1.set_xlim(-0.0005, 0.006)
    if txtstr != False:
      plt.text(0.03, 0.85, txtstr, transform=ax1.transAxes)
    #ax1.set_ylim(0.9, 2)
    #ax1.set_xlim(np.min(param1)-np.max(param1_std), np.max(param1)+np.max(param1_std))
    #ax1.set_ylim(np.min(param2)-np.max(param2_std), np.max(param2)+np.max(param2_std))
    #plt.hist(param_fit, bins=10, color='darkblue', histtype='step', label=txtstr)
    #plt.plot([-10,20], [-10,20], color='black', linewidth=1, linestyle = '--')
    if lbl == 'Mean':
      plt.scatter(param1, param2, color=colour, s=2.5, label=lbl)
    else:
      plt.plot(param1, param2, color=colour, label=lbl)
    #plt.errorbar(param1, param2, xerr = param1_std, yerr = param2_std, color='darkblue', markersize=2, fmt='o', elinewidth=0.75)
    #if sub3 == 1:
    #  ax1.legend(loc='lower right', fontsize = 9.5)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)


# ================================= #
# ======== Smoothing Kernel ======= #
# ================================= #
knl        = np.hanning(5)
knl_sum    = np.sum(knl)
smooth_knl = knl/knl_sum

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
# ========== Beam Scaling ========= #
# ================================= #
BEAM = beam_factor(0.01062*60*60, 0.009033*60*60, 5)
#print 'ASKAP Beam Factor: %.2f' % BEAM




'''
#sql_file = base_dir + 'SB2270.bm16.group5_cat.sql'
#database = base_dir + 'example.db'

#def read_sql(filename, database):
  #conn = sqlite3.connect(database)
  #c = conn.cursor()
  ## Open and read the file as a single buffer
  #fd = open(sql_file, 'r')
  #sqlFile = fd.read()
  #fd.close()
  ## all SQL commands (split on ';')
  #sqlCommands = sqlFile.split(';')
  #sqlCommands = sqlCommands[1:]
  ##SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
  ## KEY (`id`), ) DEFAULT CHARSET=utf8  COMMENT='SoFiA source catalogue'
  ## Execute every command from the input file
  #for command in sqlCommands:
    ## This will skip and report errors
    ## For example, if the tables do not yet exist, this will skip over
    ## the DROP TABLE commands
    #try:
      #c.execute(command)
    #except OperationalError, msg:
      #print "Command skipped: ", msg
  #return conn, c

#conn, cat = read_sql(sql_file, database)

## For each of the 3 tables, query the database and print the contents
#for column in ['ra', 'dec', 'vopt', 'name']:
  ## Plug in the name of the table into SELECT * query
  #result = cat.execute("SELECT %s FROM 'SoFiA-Catalogue';" % column); #
  ## Get all rows.
  #rows = result.fetchall();
  ## \n represents an end-of-line
  #print "\n--- TABLE ", 'SoFiA-Catalogue', "\n"
  ## This will print the name of the columns, padding each name up
  ## to 22 characters. Note that comma at the end prevents new lines
  #for desc in result.description:
    #print desc[0].rjust(22, ' '),
  ##print rows[5][0]
  ## End the line with column names
  #print ""
  #for row in rows:
    #for value in row:
      ## Print each value, padding it up with ' ' to 22 characters on the right
      #print str(value).rjust(22, ' '),
      ## End the values from the row
      #print ""

#cat.close()
#conn.close()
'''
