# Libraries

import math
import numpy as np
import matplotlib
import matplotlib as mpl
#mpl.use('TkAgg')
import emcee
from matplotlib import rc
from astropy import constants as const
from astropy import units
from astropy.cosmology import Planck15 as cosmo
from astropy.coordinates import SkyCoord, match_coordinates_3d
from astropy.wcs import WCS
from astropy.io import fits
from astropy import units as u
from scipy.special import erf
from scipy.optimize import curve_fit


from collections import deque
from bisect import insort, bisect_left
from itertools import islice

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

# ================================= #
# ======== Smoothing Kernel ======= #
# ================================= #
def hann_smooth(width):
    knl        = np.hanning(width)
    knl_sum    = np.sum(knl)
    smooth_knl = knl/knl_sum
    return smooth_knl

# ================================= #
# == Stellar Mass from 6dF Colour = #
# ================================= #
def calc_lgMstar_6df(mag,Bj,Rf,z,h=cosmo.h):
    distmod = cosmo.distmod(z).value
    lgMstar = 0.48 - 0.57 * (Bj - Rf) + (3.7 - (mag - distmod)) / 2.5
    return(lgMstar)

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
    return 1.133 * (axis_maj * axis_min) / (pix * pix)

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
# ======= Mag Abs to App ========= #
# ================================ #
def mag_Mtom(Mag, dist):
    dist = dist * 10**6
    return 5. * np.log10(dist / 10.) + Mag
  
# ================================ #
# ======= Mag App to Abs ========= #
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
    frequency  = (C_LIGHT * 1000. / (wavelength[i] * 10**-10))      # Hz
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
    frequency  = (C_LIGHT * 1000. / (wavelength[i] * 10**-10))      # Hz
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
    return np.log10((2.356 * 10**5) * flux * dist**2 / (1 + z)**2)

def hi_mass_jyhz(flux,z):
    dist = dist_lum(z)
    return np.log10(49.7*flux*dist**2)

def observed_hi_dist(flux,z):
    '''
    $\left(\frac{M_{\mathrm{HI}}}{\mathrm{M}_{\odot}}\right)=\frac{2.35\times10^5}{(1+z)^2}\left(\frac{D_{\mathrm{L}}}{\mathrm{Mpc}}\right)^2\left(\frac{S}{\mathrm{Jy\,km\,s}^{-1}}\right)$
    '''
    #dist = dist_lum(z)
    dist = z
    #z = z_at_value(cosmo.luminosity_distance, dist, 0.00001, 0.01)
    #flux = flux/factor
    return np.log10((2.356 * 10**5) * flux * dist**2)#/(1+z)**2)

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
    
    a,b (float) - Fitted parameters of the stellar mass formula.
	
    mag (float) - The apparent magnitude of the model [mags].
	
    col (float) - The colour of the model [mags].
	
    MSol (float) - The absolute magnitude of the Sun in the given luminance band [mags].
	
    z (float) - The redshift of the galaxy.
	
    h (float) - The reduced Hubble's constant.
	
	return: Mstar (float) - The stellar mass [M_sol] as determined by the empirical relation in Taylor et al. 2011
	"""

    distmod = cosmo.distmod(z).value
    #distmod = distmod
    lgMstar = (a + b * col - 0.4 * mag
                + 0.4 * distmod + np.log10(1.0 / np.power(10,-0.4 * MSol))
                - np.log10(1.0 + z) - 2.0 * np.log10(cosmo.h / 0.7))
    
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
# ======= Profile Width ========== #
# ================================ #
def profile_width(nu_obs, delta_nu):
    return C_LIGHT*delta_nu/nu_obs

#def profile_width(nu_obs, delta_nu):
#    return C_LIGHT*delta_nu/nu_obs



# ================================ #
# ===== Compute NN Distances ===== #
# ================================ #
def nearest_neightbour_distances(lvhis_c, catalogue_6df, sigma):
    sum_dist = 0
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




def RunningMedian(seq, M):
    """
    Purpose: Find the median for the points in a sliding window (odd number in size) 
              as it is moved from left to right by one point at a time.
     
    Inputs:
            seq -- list containing items for which a running median (in a sliding window) 
                   is to be calculated
            M -- number of items in window (window size) -- must be an integer > 1
     
    Outputs:
         medians -- list of medians with size N - M + 1
    
    Note:
         1. The median of a finite list of numbers is the "center" value when this list
            is sorted in ascending order. 
         2. If M is an even number the two elements in the window that
            are close to the center are averaged to give the median (this
            is not by definition)
    """   
    seq = iter(seq)
    s = []   
    m = M // 2

    # Set up list s (to be sorted) and load deque with first window of seq
    s = [item for item in islice(seq,M)]    
    d = deque(s)

    # Simple lambda function to handle even/odd window sizes    
    median = lambda : s[m] if bool(M&1) else (s[m-1]+s[m])*0.5

    # Sort it in increasing order and extract the median ("center" of the sorted window)
    s.sort()    
    medians = [median()]   

    # Now slide the window by one point to the right for each new position (each pass through 
    # the loop). Stop when the item in the right end of the deque contains the last item in seq
    for item in seq:
        old = d.popleft()          # pop oldest from left
        d.append(item)             # push newest in from right
        del s[bisect_left(s, old)] # locate insertion point and then remove old 
        insort(s, item)            # insert newest such that new sort is not required        
        medians.append(median())  
    return medians




