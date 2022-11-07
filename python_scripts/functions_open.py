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

def open_wise(filename):
    input_lines  = open(filename).read().split('\n')
    msol      = np.array([float(line.split()[40]) for line in input_lines[1:5]])
    msol_err  = np.array([float(line.split()[41]) for line in input_lines[1:5]])
    sfr12      = np.array([float(line.split()[43]) for line in input_lines[1:5]])
    sfr12_err  = np.array([float(line.split()[44]) for line in input_lines[1:5]])
    return(msol, msol_err, sfr12, sfr12_err)
  
def open_profile(filename, column):
    input_lines  = open(filename).read().split('\n')
    msol         = np.array([float(line.split()[column]) for line in input_lines[3:-1]])
    return(msol)
  
def open_mass_model(filename, column):
    input_lines  = open(filename).read().split('\n')
    msol         = np.array([float(line.split()[column]) for line in input_lines[12:-1]])
    return(msol)

def open_rotcur(filename, column):
    input_lines  = open(filename).read().split('\n')
    vrot         = np.array([float(line.split()[column]) for line in input_lines[11:-1]])
    return(vrot)  

def open_rings(filename, param):
    input_lines  = open(filename).read().split('\n')
    rings    = np.array([float(line.split()[param]) for line in input_lines[3:-1]])
    return(rings)




H0 = cosmo.H(0).value
rho_crit = cosmo.critical_density(0).value*100**3/1000
#print H0
#print rho_crit


PI      = math.pi
jy_w    = pow(10,-26) # Jy to W/(m^2*Hz)
mpc_m   = 3.086*pow(10,22) # Mpc to metre
sol_lum = 3.828*pow(10,26) # Solar luminosity
w1_freq = 8.33*pow(10,13) # W1 band frequency
