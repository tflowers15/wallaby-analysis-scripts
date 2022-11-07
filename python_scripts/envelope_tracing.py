# Libraries

import math
import sys
import numpy as np
import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid
import astropy.io.fits as pyfits
from matplotlib import rc
from astropy.coordinates import EarthLocation, SkyCoord, ICRS
from astropy.wcs import WCS, find_all_wcs
from astropy.nddata import Cutout2D
from astropy.io import fits
from astropy.cosmology import WMAP7
from random import shuffle as sfl
from astropy.convolution import convolve, Gaussian1DKernel
from scipy import stats
from scipy.stats import norm
import re
import warnings
from astropy import constants as const
from astropy.cosmology import Planck15 as cosmo
from astropy import units as u
import matplotlib.colors as colors
from astropy.visualization.wcsaxes import SphericalCircle
#from reproject import reproject_interp
#import aplpy

warnings.simplefilter('ignore', UserWarning)


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
def open_spec(filename):
    input_lines  = open(filename).read().split('\n')
    freq  = np.array([float(line.split()[1]) for line in input_lines[37:-1]])
    flux  = np.array([float(line.split()[2]) for line in input_lines[37:-1]])
    return(freq, flux)

def open_atca_spec(filename):
    input_lines  = open(filename).read().split('\n')
    freq  = np.array([float(line.split()[1]) for line in input_lines[4:-1]])
    flux  = np.array([float(line.split()[2]) for line in input_lines[4:-1]])
    return(freq, flux)

def open_askap_spec(filename):
    input_lines  = open(filename).read().split('\n')
    freq  = np.array([float(line.split()[1]) for line in input_lines[11:61]])
    flux  = np.array([float(line.split()[2]) for line in input_lines[11:61]])
    return(freq, flux)

def open_rotcur(filename, par_i, err_i):
    input_lines  = open(filename).read().split('\n')
    radius    = np.array([float(line.split()[0]) for line in input_lines[11:-1]])
    vrot      = np.array([float(line.split()[par_i]) for line in input_lines[11:-1]])
    if err_i == 13:
      vrot_err_s  = np.array([(line.split()[err_i]) for line in input_lines[11:-1]])
      vrot_err= []
      for i in range(len(vrot_err_s)):
        #vrot_err.append(float(vrot_err_s[i][4:]))
        vrot_err.append(-1*float(re.split('-',vrot_err_s[i])[1]))
    else:
      vrot_err  = np.array([float(line.split()[err_i]) for line in input_lines[11:-1]])
    return(radius, vrot, vrot_err)

def open_sofia_spec(filename, vel_type):
    input_lines  = open(filename).read().split('\n')
    freq  = np.array([float(line.split()[1]) for line in input_lines[3:-1]])
    flux  = np.array([float(line.split()[2]) for line in input_lines[3:-1]])
    #freq=freq/1000
    if vel_type == 'FREQ':
      freq = (1420.406/(freq/1e6) - 1) * 299792.458
    elif vel_type == 'VRAD':
      freq = (1 - (freq/1000)/299792.458)*1420.406
      freq = (1420.406/(freq) - 1) * 299792.458
      #freq = freq/1000
    else:
      freq = freq/1000
    return(freq, flux)

def open_rings(filename, param):
    input_lines  = open(filename).read().split('\n')
    rings    = np.array([float(line.split()[param]) for line in input_lines[3:-1]])
    return(rings)

def open_fits_file(file_name):
  f1        = pyfits.open(file_name)
  fits_data = f1[0].data
  fits_hdr  = f1[0].header
  return(fits_data, fits_hdr)

# Calculate optical velocity
def vel_opt(freq):
  return (1420.406/(freq/1e6) - 1) * 299792.458

# Calculate luminosity distance
def dist_lum(z):
  return WMAP7.luminosity_distance(z).value

# Calculate RMS
def rms_calc(array):
  return np.sqrt(np.mean(np.square(array)))


def env_trace(infile, vsys, incl, rms_det, min_pos, max_pos, calc_vels):
  f1        = pyfits.open(infile)
  data, hdr = f1[0].data, f1[0].header
  vel_pix, vel_del, vel_val, len_val = hdr['CRPIX2'], hdr['CDELT2'], hdr['CRVAL2'], int(hdr['NAXIS2'])
  pos_pix, pos_del, pos_val, len_pos = hdr['CRPIX1'], hdr['CDELT1'], hdr['CRVAL1'], int(hdr['NAXIS1'])
  pos_low = (pos_val + pos_pix*pos_del*60.*60.)
  velocities, positions, relative_vel = [], [], []
  for j in range(len_pos):
    positions.append((pos_low-j*pos_del*60.*60.))
  if vel_val > 1e9:
    vel_low = (vel_val - (vel_pix+1)*vel_del)/1000000.
    for j in range(len_val):
      frequencies.append((vel_low+j*vel_del/1000000.))
    frequencies = np.array(frequencies)
    velocities  = ((HI_REST)/frequencies - 1)*C_LIGHT
    vsys        = ((HI_REST)/(vsys/1000.) - 1)*C_LIGHT
  else:
    vel_low = (vel_val - (vel_pix+1)*vel_del)/1000.
    for j in range(len_val):
      frequencies.append((vel_low+j*vel_del/1000.))
    frequencies = np.array(frequencies)
    velocities  = frequencies
  #vel_low = (vel_val - (vel_pix+1)*vel_del)/1000
  print (vel_low)
  #for j in range(len_val):
  #  velocities.append((vel_low+j*vel_del/1000))
  relative_vel = vsys - np.array(velocities)
  data[np.isnan(np.array(data))] = 0
  spectra = []
  for j in range(len(data[0])):
    single_spec = []
    for k in range(len(data)):
      single_spec.append(data[k][j])
    spectra.append(single_spec)
  data        = data[~np.isnan(np.array(data))]
  (mu, sigma) = norm.fit(data[len(data)-rms_det:])
  print (sigma)
  intensity_max, intensity_term, index_max = [], [], []
  vrot     = np.zeros(len(spectra))
  vrot_err = np.ones(len(spectra))*10
  uncertainty = np.sqrt(10**2+(4/np.sqrt(8*np.log(2)))**2)
  if calc_vels:
    for j in range(len(spectra)):
      if np.argmax(spectra[j]) < 10:
        spectra[j][np.argmax(spectra[j])] = 0
      intensity_max.append(np.max(spectra[j]))
      intensity_term.append(np.sqrt(0.2*intensity_max[j]**2 + (3*sigma)**2))
      index_max.append(np.argmax(spectra[j]))
      if velocities[index_max[j]] > vsys:
        for k in range(index_max[j], 0, -1):
          if spectra[j][k] < intensity_term[j] and spectra[j][k+1] > intensity_term[j]:
            vel = np.interp(intensity_term[j], [spectra[j][k], spectra[j][k+1]], [velocities[k], velocities[k+1]])
            vrot[j] = (np.abs(vel-vsys)/np.sin(incl*math.pi/180) - uncertainty)
            break
      if velocities[index_max[j]] < vsys:
        for k in range(index_max[j],len(spectra[j])):
          if index_max[j] == len(spectra[j])-1:
            break
          elif spectra[j][k] > intensity_term[j] and spectra[j][k+1] < intensity_term[j]:
            vel = np.interp(intensity_term[j], [spectra[j][k], spectra[j][k+1]], [velocities[k], velocities[k+1]])
            vrot[j] = -1*(np.abs(vel-vsys)/np.sin(incl*math.pi/180) - uncertainty)
            break
    for j in range(len(vrot)):
      if positions[j] > -20 and positions[j] < 20:
        vrot[j] = np.nan
      if positions[j] < min_pos or positions[j] > max_pos:
        vrot[j] = np.nan
  data      = data.reshape(len_val, len_pos)
  vrot_true = vrot*np.sin(incl*math.pi/180) + vsys
  return vrot, vrot_true, vrot_err, positions, velocities, data


def env_trace_field(infile, vsys, incl):#, min_pos, max_pos):
  f1        = pyfits.open(infile)
  data, hdr = f1[0].data, f1[0].header
  vel_pix, vel_del, vel_val, len_val = hdr['CRPIX2'], hdr['CDELT2'], hdr['CRVAL2'], int(hdr['NAXIS2'])
  pos_pix, pos_del, pos_val, len_pos = hdr['CRPIX1'], hdr['CDELT1'], hdr['CRVAL1'], int(hdr['NAXIS1'])
  pos_low = (pos_val + pos_pix*pos_del*60.*60.)
  velocities, frequencies, positions, relative_vel = [], [], [], []
  for j in range(len_pos):
    positions.append((pos_low-j*pos_del*60.*60.))
  #print vel_pix, vel_del, vel_val, len_val
  if vel_val > 1e9:
    vel_low = (vel_val - (vel_pix+1)*vel_del)/1000000.
    for j in range(len_val):
      frequencies.append((vel_low+j*vel_del/1000000.))
    frequencies = np.array(frequencies)
    velocities  = ((HI_REST)/frequencies - 1)*C_LIGHT
    vsys        = ((HI_REST)/(vsys/1000.) - 1)*C_LIGHT
  else:
    vel_low = (vel_val - (vel_pix+1)*vel_del)/1000.
    for j in range(len_val):
      frequencies.append((vel_low+j*vel_del/1000.))
    frequencies = np.array(frequencies)
    velocities  = frequencies
  #vel_low = (vel_val - (vel_pix+1)*vel_del)/1000
  #print vel_low
  #for j in range(len_val):
  #  velocities.append((vel_low+j*vel_del/1000))
  print (vsys)
  relative_vel = vsys - np.array(velocities)
  data[np.isnan(np.array(data))] = 0
  spectra = []
  for j in range(len(data[0])):
    single_spec = []
    for k in range(len(data)):
      single_spec.append(data[k][j])
    spectra.append(single_spec)
  single_spec = []
  for i in range(1*len(velocities)/4,3*len(velocities)/4):
    if np.abs(velocities[i] - vsys) < np.abs(velocities[i-1] - vsys):
      vel_index = i
  for j in range(len(positions)):
    single_spec.append(data[vel_index][j])
  position_offset = positions[np.argmax(single_spec)]
  data        = data[~np.isnan(np.array(data))]
  (mu, sigma) = norm.fit(data[2:len(data)/4][:len(data)/4])
  print (sigma)
  intensity_max, intensity_term, index_max = [], [], []
  vrot     = np.zeros(len(spectra))
  vrot_err = np.ones(len(spectra))*10
  vrot_true = np.zeros(len(spectra))
  uncertainty = np.sqrt(10**2+(4/np.sqrt(8*np.log(2)))**2)
  for j in range(len(spectra)):
    if np.argmax(spectra[j]) < 10:
      spectra[j][np.argmax(spectra[j])] = 0
    intensity_max.append(np.max(spectra[j]))
    intensity_term.append(np.sqrt(0.2*intensity_max[j]**2 + (3*sigma)**2))
    index_max.append(np.argmax(spectra[j]))
    if velocities[0] > velocities[len(velocities)-1]:
      if velocities[index_max[j]] > vsys:
        for k in range(index_max[j], 0, -1):
          if spectra[j][k] < intensity_term[j] and spectra[j][k+1] > intensity_term[j]:
            #vel = np.interp(intensity_term[j], [spectra[j][k], spectra[j][k+1]], [velocities[k], velocities[k+1]])
            vel = (velocities[k+1] - velocities[k])/(spectra[j][k+1] - spectra[j][k])*(intensity_term[j] - spectra[j][k]) + velocities[k]
            vrot_true[j] = vel
            vrot[j] = (np.abs(vel-vsys)/np.sin(incl*math.pi/180) - uncertainty)
            break
      if velocities[index_max[j]] < vsys:
        for k in range(index_max[j],len(spectra[j])-1):
          if index_max[j] == len(spectra[j])-1:
            break
          elif spectra[j][k] > intensity_term[j] and spectra[j][k+1] < intensity_term[j]:
            #vel = np.interp(intensity_term[j], [spectra[j][k], spectra[j][k+1]], [velocities[k], velocities[k+1]])
            vel = (velocities[k+1] - velocities[k])/(spectra[j][k+1] - spectra[j][k])*(intensity_term[j] - spectra[j][k]) + velocities[k]
            vrot_true[j] = vel
            vrot[j] = -1*(np.abs(vel-vsys)/np.sin(incl*math.pi/180) - uncertainty)
            break
    else:
      if velocities[index_max[j]] < vsys:
        for k in range(index_max[j], 0, -1):
          if spectra[j][k] < intensity_term[j] and spectra[j][k+1] > intensity_term[j]:
            #vel = np.interp(intensity_term[j], [spectra[j][k], spectra[j][k+1]], [velocities[k], velocities[k+1]])
            vel = (velocities[k+1] - velocities[k])/(spectra[j][k+1] - spectra[j][k])*(intensity_term[j] - spectra[j][k]) + velocities[k]
            vrot_true[j] = vel
            vrot[j] = (np.abs(vel-vsys)/np.sin(incl*math.pi/180) - uncertainty)
            break
      if velocities[index_max[j]] > vsys:
        for k in range(index_max[j],len(spectra[j])-1):
          if index_max[j] == len(spectra[j])-1:
            break
          elif spectra[j][k] > intensity_term[j] and spectra[j][k+1] < intensity_term[j]:
            #vel = np.interp(intensity_term[j], [spectra[j][k], spectra[j][k+1]], [velocities[k], velocities[k+1]])
            vel = (velocities[k+1] - velocities[k])/(spectra[j][k+1] - spectra[j][k])*(intensity_term[j] - spectra[j][k]) + velocities[k]
            vrot_true[j] = vel
            vrot[j] = -1*(np.abs(vel-vsys)/np.sin(incl*math.pi/180) - uncertainty)
            break
  for j in range(len(vrot)):
  #  if positions[j] > -20 and positions[j] < 20:
  #    vrot[j] = np.nan
    #if positions[j] < min_pos or positions[j] > max_pos:
    if j < len(vrot)/6. or j > 6.*len(vrot)/6.:
      vrot[j] = np.nan
      vrot_true[j] = np.nan
  #for i in range(len(positions)):
    #print velocities[index_max[i]], vsys, positions[i]
    #if np.abs(velocities[index_max[i]] - vsys) < 8:
      #print positions[i]
  data      = data.reshape(len_val, len_pos)
  #vrot_true = vrot*np.sin(incl*math.pi/180) + vsys
  #positions = np.array(positions) + position_offset
  return vrot, vrot_true, vrot_err, positions, velocities, data, position_offset

# ==================================== End Functions ====================================





