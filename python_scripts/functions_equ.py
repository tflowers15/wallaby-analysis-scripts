import math
import os
from scipy.interpolate import griddata
from astropy.io import fits
from astropy import constants as const
from astropy import units
from astropy.cosmology import Planck15 as cosmo
from astropy.coordinates import EarthLocation, SkyCoord, ICRS
from matplotlib import rc
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as grid
import statsmodels.robust.scale as st
import scipy.integrate as integrate

import time
import numpy as np

# Calculate luminosity distance
def dist_lum(z):
    return cosmo.luminosity_distance(z).value
  
# Calculate expected HI mass
def expected_hi(a,b,diam,H0,z):
    h = H0/100
    DL = dist_lum(z)
    diam = DL*diam/60/60*math.pi/180*1000
    #print diam
    return a + b * np.log10(h * diam) - 2 * np.log10(h)
  
# Calculate HI mass from flux
def observed_hi(flux,factor,z):
    dist = dist_lum(z)
    flux = flux/factor
    #print flux
    #print (2.356*10**5)*flux*dist**2
    return np.log10((2.356*10**5)*flux*dist**2)

# Calculate HI mass from flux
def hi_mass_rms(flux,rms,factor,z):
    dist = dist_lum(z)
    flux_h = (flux+rms)/factor
    flux_l = (flux-rms)/factor
    #rms_h = np.log10((2.356*10**5)*flux_h*dist**2)
    #rms_l = np.log10((2.356*10**5)*flux_l*dist**2)
    high = (2.356*10**5)*flux_h*dist**2
    rms  = (2.356*10**5)*rms/factor*dist**2
    flux = (2.356*10**5)*flux/factor*dist**2
    rms_l = np.log10(flux - rms)
    rms_h = np.log10(flux + rms)
    return rms_l, rms_h

def dyn_mass(vel, rad):
  rad = rad * units.kiloparsec
  rad = rad.to(units.meter)
  mass = vel**2*rad/const.G.value/const.M_sun.value*pow(1000,2)
  return mass.value

def log_errors(value, error):
  return value - np.log10(10**value - 10**error)

H0 = cosmo.H(0).value
rho_crit = cosmo.critical_density(0).value*100**3/1000
#print H0
#print rho_crit


PI      = math.pi
jy_w    = pow(10,-26) # Jy to W/(m^2*Hz)
mpc_m   = 3.086*pow(10,22) # Mpc to metre
sol_lum = 3.828*pow(10,26) # Solar luminosity
w1_freq = 8.33*pow(10,13) # W1 band frequency
