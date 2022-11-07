import math
import os
from scipy.interpolate import griddata
from astropy.io import fits
from astropy import constants as const
from astropy import units
from astropy.cosmology import Planck15 as cosmo
from astropy.coordinates import EarthLocation, SkyCoord, ICRS
from astropy.stats import median_absolute_deviation as madev
from matplotlib import rc
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as grid
import statsmodels.robust.scale as st
import scipy.integrate as integrate
import warnings
from astropy import units as u
from astropy.visualization.wcsaxes import SphericalCircle
import time
import numpy as np
#from reproject import reproject_interp
from astropy.modeling.models import Ellipse2D
from astropy.coordinates import Angle
import matplotlib.patches as mpatches
import aplpy
import astropy.io.fits as pyfits
from astropy.wcs import WCS, find_all_wcs


C_LIGHT  = const.c.to('km/s').value
H0       = cosmo.H(0).value
RHO_CRIT = cosmo.critical_density(0).value*100**3/1000
OMEGA_M  = cosmo.Om(0)
OMEGA_DE = cosmo.Ode(0)
HI_REST  = 1420.406

def mass_plot(fig_num, subfig, flux1, flux2, colour, marker, lbl, legend_true):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(subfig, facecolor = 'w')
    ax1.set_ylabel(r'Expected HI Mass $\log[\mathrm{M}_{\odot}]$')
    ax1.set_xlabel(r'Observed HI Mass $\log[\mathrm{M}_{\odot}]$')
    ax1.set_xlim(7.8, 9.3)
    ax1.set_ylim(7.8, 9.3)
    plt.plot([0,10], [0,10], color='black', linewidth=1, linestyle = '--')
    #plt.scatter(flux1, flux2, color=colour, s=20, label=lbl)
    plt.errorbar(flux1, flux2, xerr = 0.2, yerr = 0.1, color=colour, markersize=8, fmt=marker, label=lbl)
    if legend_true:
        ax1.legend(loc='lower right')
        
def sfr_plot(fig_num, subfig, flux1, flux2, colour, lbl, legend_true):
    matplotlib.rcParams.update({'font.size': 14})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(subfig, facecolor = 'w')
    ax1.set_ylabel(r'Gas Fraction ($\mathrm{M}_{\mathrm{HI}}/\mathrm{M}_{*}$)')
    ax1.set_xlabel(r'SFR12 $\log[\mathrm{M}_{\odot}/\mathrm{yr}]$')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-0.2, 0.1)
    plt.plot([0,10], [0,10], color='black', linewidth=1, linestyle = '--')
    plt.scatter(flux1, flux2, color=colour, s=20, label=lbl)
    #plt.errorbar(flux1, flux2, xerr = 0.00116, yerr = 0.00064, color=colour, markersize=4, fmt='o')
    if legend_true:
        ax1.legend(loc='upper left')

def mstar_plot(fig_num, subfig, flux1, flux2, colour, lbl, legend_true):
    matplotlib.rcParams.update({'font.size': 14})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(subfig, facecolor = 'w')
    #ax1.set_ylabel(r'$\log[\mathrm{M}_{\mathrm{HI}}/\mathrm{M}_{\odot}]$')
    ax1.set_ylabel(r'Gas Fraction ($\mathrm{M}_{\mathrm{HI}}/\mathrm{M}_{*}$)')
    ax1.set_xlabel(r'$\log[\mathrm{M}_{*}/\mathrm{M}_{\odot}]$')
    ax1.set_xlim(8, 10.5)
    #ax1.set_ylim(8, 10.5)
    ax1.set_ylim(-0.2, 0.1)
    plt.plot([0,11], [0,11], color='black', linewidth=1, linestyle = '--')
    plt.scatter(flux1, flux2, color=colour, s=20, label=lbl)
    #plt.errorbar(flux1, flux2, xerr = 0.00116, yerr = 0.00064, color=colour, markersize=4, fmt='o')
    if legend_true:
        ax1.legend(loc='upper left')

def profile_plot(fig_num, subfig, flux1, flux2, colour, marker, fsty, lbl, txtstr, legend_true):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(subfig, facecolor = 'w')
    ax1.set_ylabel(r'$\log_{10}[\Sigma/\mathrm{M}_{\odot}\mathrm{pc}^{-2}]$')
    if subfig == 211:
      ax1.set_xticklabels([])
    if subfig == 212:
      ax1.set_xlabel(r'Radius [kpc]')
    ax1.set_xlim(0.0, 25)
    ax1.set_ylim(np.min(flux2)-0.5, np.max(flux2)+0.5)
    ax1.set_ylim(-2, 2)
    #plt.plot([0,10], [0,10], color='black', linewidth=1, linestyle = '--')
    #plt.scatter(flux1, flux2, color=colour, s=20, label=lbl)
    #print pow(10,flux2)
    #print 0.2*1/pow(10,flux2)
    #err_l = 0.434*((0.1*1/pow(10,flux2))/pow(10,flux2))
    if colour == 'darkblue':
      plt.text(1, -1.75, txtstr)
    err_l = 0.434*0.4
    #err_h = np.log10(pow(10,flux2) + 0.2*1/pow(10,flux2))
    plt.errorbar(flux1, flux2, yerr = err_l, color=colour, markersize=5, fmt=marker, fillstyle=fsty, label=lbl) #xerr = 0.2*flux2, 
    if legend_true and subfig == 211:
        ax1.legend(loc='upper right')
    plt.subplots_adjust(wspace=0, hspace=0)

def model_plot(fig_num, subfig, flux1, flux2, colour, lsty, lbl, legend_true):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(subfig, facecolor = 'w')
    ax1.set_ylabel(r'Velocity [km/s]')
    ax1.set_xlabel(r'Radius [kpc]')
    ax1.set_xlim(0.0, 30)
    #ax1.set_ylim(8.0, 9.5)
    #plt.plot([0,10], [0,10], color='black', linewidth=1, linestyle = '--')
    if lbl == 'Data':
      plt.scatter(flux1, flux2, color=colour, s=15, label=lbl)
    else:
      plt.plot(flux1, flux2, color=colour, linestyle=lsty, label=lbl)
    #plt.errorbar(flux1, flux2, xerr = 0.00116, yerr = 0.00064, color=colour, markersize=4, fmt='o')
    #if legend_true:
    ax1.legend(loc='upper right')

def kinematics_plot(fig_num, subfig, freq, flux, error, colour, lsty, txtstr, lbl, min_ra, max_ra):
  matplotlib.rcParams.update({'font.size': 12})
  rc('font',**{'family':'serif','serif':['Times']})
  rc('text', usetex=True)
  ax1 = fig_num.add_subplot(subfig, facecolor = 'w')
  if subfig == 313:
    ax1.set_xlabel(r'Radius [arcsec]')
  if subfig == 311:
    ax1.set_ylabel(r'V$_{\mathrm{rot}}$ [km/s]')
    if txtstr == 'NGC 7162':
      ax1.set_ylim(0, 200)
      plt.text(min_ra + 5, 7, '(a)')
      plt.text(min_ra + 5, 180, txtstr)
    if txtstr == 'NGC 7162A':
      ax1.set_ylim(0, 175)
      plt.text(min_ra + 5, 7, '(a)')
      plt.text(min_ra + 5, 155, txtstr)
  elif subfig == 312:
    ax1.set_ylabel(r'Position Angle [deg]')
    ax1.set_ylim(np.min(flux)-10, np.max(flux)+10)
    if lbl == 'Both':
      plt.text(min_ra + 5, np.min(flux)-7, '(b)')
  else:
    ax1.set_ylabel(r'Inclination [deg]')
    if txtstr == 'NGC 7162':
      ax1.set_ylim(40, 80)
      plt.text(min_ra + 5, 43, '(c)')
      plt.axhline(67.7, linewidth=0.75, linestyle = ':', color = 'darkgrey')
    if txtstr == 'NGC 7162A':
      ax1.set_ylim(0,50)
      plt.text(min_ra + 5, 3, '(c)')
      plt.axhline(39.6, linewidth=0.75, linestyle = ':', color = 'darkgrey')
  ax1.set_xlim(min_ra, max_ra)
  #if subfig == 511 or subfig == 512 or subfig == 513 or subfig == 514:
  #  ax1.set_xticklabels([])
  if subfig == 311 or subfig == 312:
    ax1.set_xticklabels([])
  #ax1.set_xlim(2000, 2750)
  #if subfig == 311 and lbl == 'Both':
  #  plt.text(min_ra + 5, np.min(flux)+6*(np.max(flux)-np.min(flux))/8+20, txtstr)    
  #np.max(flux)-2*(np.max(flux)-np.min(flux))/10
      #textstr = r'$(%s)$ %s' % (panel, gal_name)
      #plt.text(np.min(nu), 1*(xhigh+xlow)/6, textstr)
  #plt.axhline(0, linewidth=0.75, linestyle = ':', color = 'darkgrey')
  plt.errorbar(freq, flux, yerr=error, fmt = lsty, color = colour, linewidth = 0.5, ms=5, label=lbl)
  if subfig == 311:
    ax1.legend(loc='lower right', fontsize = 10.5)
  plt.subplots_adjust(wspace=0, hspace=0)
  ax1.get_yaxis().set_tick_params(which='both', direction='in')
  ax1.get_xaxis().set_tick_params(which='both', direction='in')
  
def rotation_curve_plot(fig_num, subfig, freq, flux, error, colour, lsty, txtstr, lbl, min_ra, max_ra):
  matplotlib.rcParams.update({'font.size': 12})
  rc('font',**{'family':'serif','serif':['Times']})
  rc('text', usetex=True)
  ax1 = fig_num.add_subplot(subfig, facecolor = 'w')
  if subfig == 414:
    ax1.set_xlabel(r'Radius [arcsec]')
  ax1.set_ylabel(r'V$_{\mathrm{rot}}$ [km/s]')
  plt.text(2, 5, txtstr)
  if subfig == 411:
    ax1.set_ylim(0, 195)
  elif subfig == 412:
    ax1.set_ylim(0, 145)
  elif subfig == 413:
    ax1.set_ylim(0, 195)
  else:
    ax1.set_ylim(0, 75)
  ax1.set_xlim(0, 150)
  if subfig != 414:
    ax1.set_xticklabels([])
  plt.errorbar(freq, flux, yerr=error, fmt = lsty, color = colour, linewidth = 0.5, fillstyle='none', ms=5, label=lbl)
  if subfig == 411:
    ax1.legend(loc='lower right', fontsize = 10.5)
  plt.subplots_adjust(wspace=0, hspace=0)
  ax1.get_yaxis().set_tick_params(which='both', direction='in')
  ax1.get_xaxis().set_tick_params(which='both', direction='in')
  
def rotation_curve_plot_single(fig_num, sub1, sub2, sub3, freq, flux, error, colour, lsty, txtstr, lbl, radius_max, vrot_max):
  matplotlib.rcParams.update({'font.size': 12})
  rc('font',**{'family':'serif','serif':['Times']})
  rc('text', usetex=True)
  ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
  ax1.set_xlabel(r'Radius [arcsec]')
  ax1.set_ylabel(r'V$_{\mathrm{rot}}$ [km/s]')
  ax1.set_ylim(0, vrot_max)
  ax1.set_xlim(0, radius_max)
  plt.text(2, 5, txtstr)
  plt.errorbar(freq, flux, yerr=error, fmt = lsty, color = colour, linewidth = 0.5, fillstyle='none', ms=5, label=lbl)
  if sub3 == 5:
    ax1.legend(loc='lower right', fontsize = 10.5)
  #plt.subplots_adjust(wspace=0, hspace=0)
  ax1.get_yaxis().set_tick_params(which='both', direction='in')
  ax1.get_xaxis().set_tick_params(which='both', direction='in')

def plot_all(f1, f2, f3, colour, marker, galaxy, lbl, ubound):
  radius1, pa, pa_err = open_rotcur(f1, 8, 9)
  pa_conv = convolve(pa, smooth_knl)
  radius2, incl, incl_err = open_rotcur(f2, 10, 11)
  incl_conv = convolve(incl, smooth_knl)
  radius3, vrot, vrot_err = open_rotcur(f3, 4, 16)
  vrot_conv = convolve(vrot, smooth_knl)
  #print vrot_conv
  min_ra = 0#np.min([np.min(radius1), np.min(radius2), np.min(radius3)]) - 10
  max_ra = np.max([np.max(radius1[:ubound]), np.max(radius2[:ubound]), np.max(radius3[:ubound])]) + 10
  #print min_ra, max_ra
  kinematics_plot(fig1, 311, radius3[:ubound], vrot[:ubound], vrot_err[:ubound], colour, marker, galaxy, lbl, min_ra, max_ra)
  kinematics_plot(fig1, 312, radius1[:ubound], pa_conv[:ubound], pa_err[:ubound], colour, marker, galaxy, lbl, min_ra, max_ra)
  kinematics_plot(fig1, 313, radius2[:ubound], incl_conv[:ubound], incl_err[:ubound], colour, marker, galaxy, lbl, min_ra, max_ra)

def pv_plot(fig_num, subfig1, subfig2, subfig3, image, freq, flux, error, colour, lsty, txtstr, lbl, min_ra, max_ra, min_vel, max_vel):
  matplotlib.rcParams.update({'font.size': 24})
  rc('font',**{'family':'serif','serif':['Times']})
  rc('text', usetex=True)
  if subfig3 == 3:
    subfig_param = [0.68,0.71,0.25,0.2]
  if subfig3 == 6:
    subfig_param = [0.68,0.49,0.25,0.2]
  if subfig3 == 9:
    subfig_param = [0.68,0.27,0.25,0.2]
  if subfig3 == 12:
    subfig_param = [0.68,0.05,0.25,0.2]
  if subfig3 == 15:
    subfig_param = [0.68,0.50,0.25,0.4]
  if subfig3 == 18:
    subfig_param = [0.68,0.05,0.25,0.4]
  ax1 = fig_num.add_axes(subfig_param, facecolor = 'w')
  #ax1 = fig_num.add_subplot(subfig1, subfig2, subfig3, facecolor = 'w')
  if subfig3 == 18 or subfig3 == 12:
    ax1.set_xlabel(r'Offset [arcsec]')
  ax1.set_ylabel(r'Velocity [km/s]')
  ax1.imshow(image, origin='lower', aspect='auto', extent = (min_ra,max_ra,min_vel,max_vel), cmap='Greys')
  if subfig3 == 3:
    ax1.set_xlim(-200, 200)
  if subfig3 == 6:
    ax1.set_xlim(-200, 200)
  if subfig3 == 9:
    ax1.set_xlim(-150, 150)
  if subfig3 == 12:
    ax1.set_xlim(-100, 100)
  if subfig3 == 15:
    ax1.set_xlim(-100, 100)
  if subfig3 == 18:
    ax1.set_xlim(-100, 100)
  if subfig3 == 3 or subfig3 == 6 or subfig3 == 9 or subfig3 == 12:
    plt.errorbar(freq, flux, yerr=error, fmt = lsty, color = colour, linewidth = 0.5, ms=3, label=lbl)

def pv_plot_single(fig_num, subfig1, subfig2, subfig3, image, freq, flux, error, colour, lsty, txtstr, lbl, min_ra, max_ra, min_vel, max_vel):
  matplotlib.rcParams.update({'font.size': 12})
  rc('font',**{'family':'serif','serif':['Times']})
  rc('text', usetex=True)
  ax1 = fig_num.add_subplot(subfig1, subfig2, subfig3, facecolor = 'w')
  ax1.set_xlabel(r'Offset [arcsec]')
  ax1.set_ylabel(r'Velocity [km/s]')
  ax1.imshow(image, origin='lower', aspect='auto', extent = (min_ra,max_ra,max_vel,min_vel), cmap='Greys')
  plt.errorbar(freq, flux, yerr=error, fmt = lsty, color = colour, linewidth = 0.5, ms=3, label=lbl)

def field_pv_plot(fig_num, subfig1, subfig2, subfig3, image, vsys):
  matplotlib.rcParams.update({'font.size': 24})
  rc('font',**{'family':'serif','serif':['Times']})
  rc('text', usetex=True)
  f1        = pyfits.open(image)
  data, hdr = f1[0].data, f1[0].header
  vel_pix, vel_del, vel_val, len_val = hdr['CRPIX2'], hdr['CDELT2'], hdr['CRVAL2'], int(hdr['NAXIS2'])
  pos_pix, pos_del, pos_val, len_pos = hdr['CRPIX1'], hdr['CDELT1'], hdr['CRVAL1'], int(hdr['NAXIS1'])
  pos_low = (pos_val + pos_pix*pos_del*60*60)
  frequencies, positions = [], []
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
  for j in range(len_pos):
    positions.append((pos_low-j*pos_del*60*60))
  data      = data.reshape(len_val, len_pos)
  if subfig3 == 3:
    subfig_param = [0.68,0.71,0.25,0.2]
  if subfig3 == 6:
    subfig_param = [0.68,0.49,0.25,0.2]
  if subfig3 == 9:
    subfig_param = [0.68,0.27,0.25,0.2]
  if subfig3 == 12:
    subfig_param = [0.68,0.05,0.25,0.2]
  #if subfig3 == 15:
    #subfig_param = [0.68,0.50,0.25,0.4]
  #if subfig3 == 18:
    #subfig_param = [0.68,0.05,0.25,0.4]
  ax1 = fig_num.add_axes(subfig_param, facecolor = 'w')
  #ax1 = fig_num.add_subplot(subfig1, subfig2, subfig3, facecolor = 'w')
  #ax1 = aplpy.FITSFigure(image, figure=fig_num, subplot=subfig_param, slices=(0,1))
  if subfig3 == 18 or subfig3 == 12:
    ax1.set_xlabel(r'Offset [arcsec]')
  ax1.set_ylabel(r'Velocity [km/s]')
  #ax1.imshow(data, origin='lower', aspect='auto', cmap='Greys')
  #ax1.imshow(data, origin='lower', aspect='auto', extent = (np.min(positions), np.max(positions), np.max(velocities), np.min(velocities)), cmap='Greys')
  #if vsys + 150 > np.max(velocities):
    #velmax = np.max(velocities)
  #else:
    #velmax = vsys + 150
  #if vsys - 150 < np.min(velocities):
    #velmin = np.min(velocities)
  #else:
    #velmin = vsys - 150
  velmax = np.max(velocities)
  velmin = np.min(velocities)
  if velocities[0] > velocities[len(velocities)-1]:
    ax1.imshow(data, origin='lower', aspect='auto', extent = (np.min(positions), np.max(positions), velmax, velmin), cmap='Greys')
  else:
    ax1.imshow(data, origin='lower', aspect='auto', extent = (np.min(positions), np.max(positions), velmin, velmax), cmap='Greys')
  ax1.set_xlim(2*np.min(positions)/3, 2*np.max(positions)/3)
  y_vals   = ax1.get_yticks()
  y_lables = []
  for i in range(len(y_vals)):
    y_lables.append(int(round(y_vals[i], 0)))
  ax1.set_yticklabels(y_lables)
  f1.close()
  #if subfig3 == 3 or subfig3 == 6 or subfig3 == 9 or subfig3 == 12:
    #plt.errorbar(freq, flux, yerr=error, fmt = lsty, color = colour, linewidth = 0.5, ms=3, label=lbl)

def moment_plot(fig_num, subfig1, subfig2, subfig3, image1, image2, colour, lsty, txtstr, lbl, wcs, wcs2):
  matplotlib.rcParams.update({'font.size': 15})
  rc('font',**{'family':'serif','serif':['Times']})
  rc('text', usetex=True)
  ax1 = fig_num.add_subplot(subfig1, subfig2, subfig3, facecolor = 'w', projection=wcs)
  ax1.tick_params(direction='in')
  #ax1 = fig_num.add_subplot(subfig1, subfig2, subfig3, facecolor = 'w', projection=wcs, slices=('x', 'y', 0))
  lon = ax1.coords[0]
  lat = ax1.coords[1]
  lon.set_major_formatter('hh:mm:ss')
  lat.set_major_formatter('dd:mm')
  lon.set_separator(('h', 'm', 's'))
  lat.set_separator(('d', 'm'))
  if subfig3 == 1 or subfig3 == 4 or subfig3 == 7 or subfig3 == 10 or subfig3 == 13 or subfig3 == 16:
    ax1.set_ylabel(r'Declination')
  if subfig3 == 16 or subfig3 == 17 or subfig3 == 10 or subfig3 == 11:
    ax1.set_xlabel(r'Right Ascension')
  if txtstr == 'a) NGC 7162' or txtstr == 'b) NGC 7162A' or txtstr == 'c) ESO 288-G025' or txtstr == 'd) ESO 288-G033':
    ax1.imshow(image1, origin='lower', aspect='auto', cmap='Greys', clim=(4000,13000))
  if txtstr == 'e) AM 2159-434' or txtstr == 'f) J220338-431131':
    ax1.imshow(image1, origin='lower', aspect='auto', cmap='Greys', clim=(0,20))
  #ax1.show_beam(major=.011,minor=.011,angle=78,fill=True,color='blue')
  ax1.set_autoscale_on(False)
  #import aplpy
  #f1 = aplpy.FITSFigure('/Users/tflowers/ASKAP/DSS2_Blue.fits', figure=fig_num)
  #f1.show_beam(major=.011,minor=.011,angle=78,fill=True,color='blue')
  if subfig3 == 1 or subfig3 == 4 or subfig3 == 7 or subfig3 == 10 or subfig3 == 13 or subfig3 == 16:
    plt.text(1, 5, txtstr, fontsize=16)
    max_flux =  np.max(image2)
    if subfig3 == 4:
      max_flux = 0.8607
    print (max_flux)
    #lvls = np.array([0.01, 0.1, 0.3, 0.5, 0.7, 0.9])
    #lvls = lvls*max_flux
    min_flux = 1*10**19*(39.01*34.06)/(1.36*21*21*1.823*1000*(10**18))
    lvls = np.array([1, 5, 10, 20, 50, 70, 100])
    lvls = lvls*min_flux
    ax1.contour(image2, colors='blue', alpha=0.5, levels=lvls, transform=ax1.get_transform(wcs2))
    if subfig3 == 1:
      c = SkyCoord('21h59m27s', '-43d16m00s', frame='icrs')
    if subfig3 == 4:
      c = SkyCoord('22h00m25s', '-43d06m30s', frame='icrs')
    if subfig3 == 7:
      c = SkyCoord('21h59m27s', '-43d16m00s', frame='icrs')
    if subfig3 == 10:
      c = SkyCoord('21h59m27s', '-43d16m00s', frame='icrs')
    if subfig3 == 13:
      c = SkyCoord('21h59m27s', '-43d16m00s', frame='icrs')
    if subfig3 == 16:
      c = SkyCoord('21h59m27s', '-43d16m00s', frame='icrs')
    theta = Angle(78, 'deg')
    axis_major = 39*u.arcsec
    axis_minor = 34*u.arcsec
    print (c.ra, c.dec)
    print (axis_major.value, axis_minor.value)
    #e = Ellipse2D(amplitude=1., x_0=c.ra, y_0=c.dec, a=axis_major, b=axis_minor, theta=theta)#, transform=ax1.get_transform('icrs'))
    #e = mpatches.Ellipse((c.ra, c.dec), axis_major.value, axis_minor.value, theta.value, edgecolor='peru', facecolor='peru', transform=ax1.get_transform('icrs'))
    #e = mpatches.Ellipse((329, -43), 39*u.arcsec, 34*u.arcsec, theta, edgecolor='peru', facecolor='peru', transform=ax1.get_transform('icrs'))
    e = SphericalCircle((c.ra, c.dec), 0.011 * u.deg, edgecolor='peru', facecolor='peru', transform=ax1.get_transform('icrs'))
    ax1.add_patch(e)
  if subfig3 == 2:
    vel_range = 283
    lvls = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    lvls = lvls*vel_range/10 + 2150
    ax1.contour(image2, colors='blue', alpha=0.5, levels=lvls, transform=ax1.get_transform(wcs2))
    #ax1.imshow(image2, origin='lower', aspect='auto', cmap='RdBu_r', transform=ax1.get_transform(wcs2), clim=(2150,2450))
  if subfig3 == 5:
    vel_range = 132
    lvls = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    lvls = lvls*vel_range/10 + 2200
    ax1.contour(image2, colors='blue', alpha=0.5, levels=lvls, transform=ax1.get_transform(wcs2))
    #ax1.imshow(image2, origin='lower', aspect='auto', cmap='RdBu_r', transform=ax1.get_transform(wcs2), clim=(2200,2350))
  if subfig3 == 8:
    vel_range = 378
    lvls = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    lvls = lvls*vel_range/10 + 2300
    ax1.contour(image2, colors='blue', alpha=0.5, levels=lvls, transform=ax1.get_transform(wcs2))
    #ax1.imshow(image2, origin='lower', aspect='auto', cmap='RdBu_r', transform=ax1.get_transform(wcs2), clim=(2300,2700))
  if subfig3 == 11:
    vel_range = 128
    lvls = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    lvls = lvls*vel_range/10 + 2550
    ax1.contour(image2, colors='blue', alpha=0.5, levels=lvls, transform=ax1.get_transform(wcs2))
    #ax1.imshow(image2, origin='lower', aspect='auto', cmap='RdBu_r', transform=ax1.get_transform(wcs2), clim=(2550,2700))
  if subfig3 == 14:
    vel_range = 72
    lvls = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    lvls = lvls*vel_range/10 + 2500
    ax1.contour(image2, colors='blue', alpha=0.5, levels=lvls, transform=ax1.get_transform(wcs2))
    #ax1.imshow(image2, origin='lower', aspect='auto', cmap='RdBu_r', transform=ax1.get_transform(wcs2), clim=(2500,2600))
  if subfig3 == 17:
    vel_range = 40
    #lvls = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    lvls = np.array([1, 3, 5, 7, 9])
    lvls = lvls*vel_range/10 + 2710
    ax1.contour(image2, colors='blue', alpha=0.5, levels=lvls, transform=ax1.get_transform(wcs2))
    #ax1.imshow(image2, origin='lower', aspect='auto', cmap='RdBu_r', transform=ax1.get_transform(wcs2), clim=(2710,2760))
  plt.subplots_adjust(wspace=0.3, hspace=0.15)

def moment_plot2(fig_num, subfig1, subfig2, subfig3, image1, image2, colour, lsty, txtstr, lbl):
  matplotlib.rcParams.update({'font.size': 24})
  rc('font',**{'family':'serif','serif':['Times']})
  rc('text', usetex=True)
  #ax1 = fig_num.add_subplot(subfig1, subfig2, subfig3, facecolor = 'w')#, projection=wcs)
  if subfig3 == 1:
    subfig_param = [0.05,0.71,0.27,0.2]
  if subfig3 == 2:
    subfig_param = [0.33,0.71,0.27,0.2]
  if subfig3 == 4:
    subfig_param = [0.05,0.49,0.27,0.2]
  if subfig3 == 5:
    subfig_param = [0.33,0.49,0.27,0.2]
  if subfig3 == 7:
    subfig_param = [0.05,0.27,0.27,0.2]
  if subfig3 == 8:
    subfig_param = [0.33,0.27,0.27,0.2]
  if subfig3 == 10:
    subfig_param = [0.05,0.05,0.27,0.2]
  if subfig3 == 11:
    subfig_param = [0.33,0.05,0.27,0.2]
  if subfig3 == 13:
    subfig_param = [0.05,0.50,0.27,0.4]
  if subfig3 == 14:
    subfig_param = [0.33,0.50,0.27,0.4]
  if subfig3 == 16:
    subfig_param = [0.05,0.05,0.27,0.4]
  if subfig3 == 17:
    subfig_param = [0.33,0.05,0.27,0.4]
  ax1 = aplpy.FITSFigure(image1, figure=fig_num, subplot=subfig_param)
  #ax1 = aplpy.FITSFigure(image1, figure=fig_num, subplot=(subfig1, subfig2, subfig3))
  ax1.show_beam(major=.0108,minor=.009,angle=78,fill=True,color='peru')
  if txtstr == 'a) NGC 7162' or txtstr == 'b) NGC 7162A' or txtstr == 'c) ESO 288-G025' or txtstr == 'd) ESO 288-G033':
    #ax1.imshow(image1, origin='lower', aspect='auto', cmap='Greys', clim=(4000,13000))
    #ax1.show_colorscale(vmin=4000,vmax=13000,cmap='Greys')
    ax1.show_colorscale(vmin=0,vmax=60,cmap='Greys')
  if txtstr == 'e) AM 2159-434' or txtstr == 'f) J220338-431131':
    #ax1.imshow(image1, origin='lower', aspect='auto', cmap='Greys', clim=(0,20))
    #ax1 = aplpy.FITSFigure('/Users/tflowers/ASKAP/DSS2_Blue.fits', figure=fig_num)
    #ax1.show_beam(major=.0108,minor=.009,angle=78,fill=True,color='peru')
    ax1.show_colorscale(vmin=0,vmax=20,cmap='Greys')
  #ax1.tick_params(direction='in')
  #ax1 = fig_num.add_subplot(subfig1, subfig2, subfig3, facecolor = 'w', projection=wcs, slices=('x', 'y', 0))
  if txtstr == 'a) NGC 7162':
    position     = SkyCoord('21h59m39.1s', '-43d18m21s', frame='icrs')
    size         = u.Quantity((360, 360), u.arcsec)
    width, height = 360./60./60., 360./60./60.
  if txtstr == 'b) NGC 7162A':
    position     = SkyCoord('22h00m35.7s', '-43d08m30s', frame='icrs')
    size         = u.Quantity((360, 360), u.arcsec)
    width, height = 360./60./60., 360./60./60.
  if txtstr == 'c) ESO 288-G025':
    position     = SkyCoord('21h59m17.9s', '-43d52m01s', frame='icrs')
    size         = u.Quantity((240, 240), u.arcsec)
    width, height = 240./60./60., 240./60./60.
  if txtstr == 'd) ESO 288-G033':
    position     = SkyCoord('22h02m06.6s', '-43d16m07s', frame='icrs')
    size         = u.Quantity((240, 240), u.arcsec)
    width, height = 240./60./60., 240./60./60.
  if txtstr == 'e) AM 2159-434':
    position     = SkyCoord('22h02m50.1s', '-43d26m44s', frame='icrs')
    size         = u.Quantity((180, 180), u.arcsec)
    width, height = 240./60./60., 240./60./60.
  if txtstr == 'f) J220338-431131':
    position     = SkyCoord('22h03m38.6s', '-43d11m30s', frame='icrs')
    size         = u.Quantity((180, 180), u.arcsec)
    width, height = 240./60./60., 240./60./60.
  ax1.recenter(position.ra, position.dec, width=width, height=height)
  if subfig3 == 1 or subfig3 == 4 or subfig3 == 7 or subfig3 == 10 or subfig3 == 13 or subfig3 == 16:
    #plt.text(1, 5, txtstr, fontsize=16)
    if subfig3 == 1:
      ax1.add_label(0.20, 0.95, txtstr, relative=True)
    if subfig3 == 4:
      ax1.add_label(0.25, 0.95, txtstr, relative=True)
    if subfig3 == 7:
      ax1.add_label(0.30, 0.95, txtstr, relative=True)
    if subfig3 == 10:
      ax1.add_label(0.30, 0.95, txtstr, relative=True)
    if subfig3 == 13:
      ax1.add_label(0.25, 0.95, txtstr, relative=True)
    if subfig3 == 16:
      ax1.add_label(0.30, 0.95, txtstr, relative=True)
    min_flux = 1*10**19*(39.01*34.06)/(1.36*21*21*1.823*1000*(10**18))
    lvls = np.array([1, 5, 10, 20, 50, 70, 100])
    lvls = lvls*min_flux
    ax1.show_contour(image2, colors="blue", levels=lvls, slices=(0,1))
    #ax1.contour(image2, colors='blue', alpha=0.5, levels=lvls, transform=ax1.get_transform(wcs2))
  if subfig3 == 2:
    vel_range = 283
    lvls = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    lvls = lvls*vel_range/10 + 2150
    ax1.show_contour(image2, colors="blue", levels=lvls, slices=(0,1))
    #ax1.contour(image2, colors='blue', alpha=0.5, levels=lvls, transform=ax1.get_transform(wcs2))
    #ax1.imshow(image2, origin='lower', aspect='auto', cmap='RdBu_r', transform=ax1.get_transform(wcs2), clim=(2150,2450))
  if subfig3 == 5:
    vel_range = 132
    lvls = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    lvls = lvls*vel_range/10 + 2200
    ax1.show_contour(image2, colors="blue", levels=lvls, slices=(0,1))
    #ax1.contour(image2, colors='blue', alpha=0.5, levels=lvls, transform=ax1.get_transform(wcs2))
    #ax1.imshow(image2, origin='lower', aspect='auto', cmap='RdBu_r', transform=ax1.get_transform(wcs2), clim=(2200,2350))
  if subfig3 == 8:
    vel_range = 378
    lvls = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    lvls = lvls*vel_range/10 + 2300
    ax1.show_contour(image2, colors="blue", levels=lvls, slices=(0,1))
    #ax1.contour(image2, colors='blue', alpha=0.5, levels=lvls, transform=ax1.get_transform(wcs2))
    #ax1.imshow(image2, origin='lower', aspect='auto', cmap='RdBu_r', transform=ax1.get_transform(wcs2), clim=(2300,2700))
  if subfig3 == 11:
    vel_range = 128
    lvls = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    lvls = lvls*vel_range/10 + 2550
    ax1.show_contour(image2, colors="blue", levels=lvls, slices=(0,1))
    #ax1.contour(image2, colors='blue', alpha=0.5, levels=lvls, transform=ax1.get_transform(wcs2))
    #ax1.imshow(image2, origin='lower', aspect='auto', cmap='RdBu_r', transform=ax1.get_transform(wcs2), clim=(2550,2700))
  if subfig3 == 14:
    vel_range = 72
    lvls = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    lvls = lvls*vel_range/10 + 2500
    ax1.show_contour(image2, colors="blue", levels=lvls, slices=(0,1))
    #ax1.contour(image2, colors='blue', alpha=0.5, levels=lvls, transform=ax1.get_transform(wcs2))
    #ax1.imshow(image2, origin='lower', aspect='auto', cmap='RdBu_r', transform=ax1.get_transform(wcs2), clim=(2500,2600))
  if subfig3 == 17:
    vel_range = 40
    #lvls = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    lvls = np.array([1, 3, 5, 7, 9])
    lvls = lvls*vel_range/10 + 2710
    ax1.show_contour(image2, colors="blue", levels=lvls, slices=(0,1))
    #ax1.contour(image2, colors='blue', alpha=0.5, levels=lvls, transform=ax1.get_transform(wcs2))
    #ax1.imshow(image2, origin='lower', aspect='auto', cmap='RdBu_r', transform=ax1.get_transform(wcs2), clim=(2710,2760))
  if subfig3 == 1 or subfig3 == 4 or subfig3 == 7 or subfig3 == 10 or subfig3 == 13 or subfig3 == 16:
  #  ax1.set_ylabel(r'Declination')
    ax1.axis_labels.set_ytext('Declination')
  else:
    ax1.axis_labels.hide_y()
  if subfig3 == 16 or subfig3 == 17 or subfig3 == 10 or subfig3 == 11:
  #  ax1.set_xlabel(r'Right Ascension')
    ax1.axis_labels.set_xtext('Right Ascension')
  else:
    ax1.axis_labels.hide_x()
  if subfig3 == 2 or subfig3 == 5 or subfig3 == 8 or subfig3 == 11 or subfig3 == 14 or subfig3 == 17:
    ax1.tick_labels.hide_y()
  ax1.tick_labels.set_xformat('hh:mm:ss')
  ax1.tick_labels.set_yformat('dd:mm')
  ax1.ticks.show()
  ax1.ticks.set_xspacing(0.05)
  ax1.ticks.set_length(5)
  ax1.ticks.set_color('black')
  ax1.ticks.set_minor_frequency(1)
  #ax1.tick_labels.set_font(style='normal', variant='normal')
  ax1.tick_labels.set_style('plain')
  plt.subplots_adjust(wspace=0.25, hspace=0.15)
  #ax1.tick_labels.set_font(size= 'medium', weight= 'medium', stretch= 'normal', family= 'sans-serif', style='normal', variant='normal')

def moment_plot3(fig_num, subfig1, subfig2, subfig3, image1, image2, image3, colour, lsty, txtstr, lbl):
  matplotlib.rcParams.update({'font.size': 24})
  rc('font',**{'family':'serif','serif':['Times']})
  rc('text', usetex=True)
  #ax1 = fig_num.add_subplot(subfig1, subfig2, subfig3, facecolor = 'w')#, projection=wcs)
  if subfig3 == 1:
    subfig_param = [0.05,0.71,0.27,0.2]
  if subfig3 == 2:
    subfig_param = [0.33,0.71,0.27,0.2]
  if subfig3 == 4:
    subfig_param = [0.05,0.49,0.27,0.2]
  if subfig3 == 5:
    subfig_param = [0.33,0.49,0.27,0.2]
  if subfig3 == 7:
    subfig_param = [0.05,0.27,0.27,0.2]
  if subfig3 == 8:
    subfig_param = [0.33,0.27,0.27,0.2]
  if subfig3 == 10:
    subfig_param = [0.05,0.05,0.27,0.2]
  if subfig3 == 11:
    subfig_param = [0.33,0.05,0.27,0.2]
  if subfig3 == 13:
    subfig_param = [0.05,0.50,0.27,0.4]
  if subfig3 == 14:
    subfig_param = [0.33,0.50,0.27,0.4]
  if subfig3 == 16:
    subfig_param = [0.05,0.05,0.27,0.4]
  if subfig3 == 17:
    subfig_param = [0.33,0.05,0.27,0.4]
  #if subfig3 == 1 or subfig3 == 4 or subfig3 == 7 or subfig3 == 10 or subfig3 == 13 or subfig3 == 16:
  ax1 = aplpy.FITSFigure(image1, figure=fig_num, subplot=subfig_param)
  #if subfig3 == 2:
    #ax1 = aplpy.FITSFigure(image2, figure=fig_num, subplot=subfig_param)
    #ax1.show_colorscale(cmap='viridis')
  #if subfig3 == 5:
    #ax1 = aplpy.FITSFigure(image2, figure=fig_num, subplot=subfig_param)
    #ax1.show_colorscale(cmap='viridis')
  #if subfig3 == 8:
    #ax1 = aplpy.FITSFigure(image2, figure=fig_num, subplot=subfig_param, slices=(0,1))
    #ax1.show_colorscale(cmap='viridis')
  #if subfig3 == 11:
    #ax1 = aplpy.FITSFigure(image2, figure=fig_num, subplot=subfig_param, slices=(0,1))
    #ax1.show_colorscale(cmap='viridis')
  #if subfig3 == 14:
    #ax1 = aplpy.FITSFigure(image2, figure=fig_num, subplot=subfig_param, slices=(0,1))
    #ax1.show_colorscale(cmap='viridis')
  #if subfig3 == 17:
    #ax1 = aplpy.FITSFigure(image2, figure=fig_num, subplot=subfig_param, slices=(0,1))
    #ax1.show_colorscale(cmap='viridis')
  #ax1 = aplpy.FITSFigure(image1, figure=fig_num, subplot=(subfig1, subfig2, subfig3))
  ax1.show_beam(major=.0108,minor=.009,angle=78,fill=True,color='peru')
  if subfig3 == 1 or subfig3 == 4 or subfig3 == 10:
    ax1.show_beam(major=.0167,minor=.01,angle=10.9,fill=False,color='black',linewidth=2)
  #if subfig3 == 1 or subfig3 == 4 or subfig3 == 7 or subfig3 == 10 or subfig3 == 13 or subfig3 == 16:
  if txtstr == 'a) NGC 7162' or txtstr == 'b) NGC 7162A' or txtstr == 'c) ESO 288-G025' or txtstr == 'd) ESO 288-G033':
    #ax1.imshow(image1, origin='lower', aspect='auto', cmap='Greys', clim=(4000,13000))
    #ax1.show_colorscale(vmin=4000,vmax=13000,cmap='Greys')
    ax1.show_colorscale(vmin=0,vmax=60,cmap='Greys')
  if txtstr == 'e) AM 2159-434' or txtstr == 'f) J220338-431131':
    #ax1.imshow(image1, origin='lower', aspect='auto', cmap='Greys', clim=(0,20))
    #ax1 = aplpy.FITSFigure('/Users/tflowers/ASKAP/DSS2_Blue.fits', figure=fig_num)
    #ax1.show_beam(major=.0108,minor=.009,angle=78,fill=True,color='peru')
    ax1.show_colorscale(vmin=0,vmax=20,cmap='Greys')
  #ax1.tick_params(direction='in')
  #ax1 = fig_num.add_subplot(subfig1, subfig2, subfig3, facecolor = 'w', projection=wcs, slices=('x', 'y', 0))
  if txtstr == 'a) NGC 7162':
    position     = SkyCoord('21h59m39.1s', '-43d18m21s', frame='icrs')
    size         = u.Quantity((360, 360), u.arcsec)
    width, height = 420./60./60., 420./60./60.
  if txtstr == 'b) NGC 7162A':
    position     = SkyCoord('22h00m35.7s', '-43d08m30s', frame='icrs')
    size         = u.Quantity((360, 360), u.arcsec)
    width, height = 420./60./60., 420./60./60.
  if txtstr == 'c) ESO 288-G025':
    position     = SkyCoord('21h59m17.9s', '-43d52m01s', frame='icrs')
    size         = u.Quantity((240, 240), u.arcsec)
    width, height = 240./60./60., 240./60./60.
  if txtstr == 'd) ESO 288-G033':
    position     = SkyCoord('22h02m06.6s', '-43d16m07s', frame='icrs')
    size         = u.Quantity((240, 240), u.arcsec)
    width, height = 240./60./60., 240./60./60.
  if txtstr == 'e) AM 2159-434':
    position     = SkyCoord('22h02m50.1s', '-43d26m44s', frame='icrs')
    size         = u.Quantity((180, 180), u.arcsec)
    width, height = 240./60./60., 240./60./60.
  if txtstr == 'f) J220338-431131':
    position     = SkyCoord('22h03m38.6s', '-43d11m30s', frame='icrs')
    size         = u.Quantity((180, 180), u.arcsec)
    width, height = 240./60./60., 240./60./60.
  #if subfig3 == 1 or subfig3 == 4 or subfig3 == 7 or subfig3 == 10 or subfig3 == 13 or subfig3 == 16:
  ax1.recenter(position.ra, position.dec, width=width, height=height)
  if subfig3 == 1 or subfig3 == 4 or subfig3 == 7 or subfig3 == 10 or subfig3 == 13 or subfig3 == 16:
    #plt.text(1, 5, txtstr, fontsize=16)
    if subfig3 == 1:
      ax1.add_label(0.20, 0.95, txtstr, relative=True)
    if subfig3 == 4:
      ax1.add_label(0.25, 0.95, txtstr, relative=True)
    if subfig3 == 7:
      ax1.add_label(0.30, 0.95, txtstr, relative=True)
    if subfig3 == 10:
      ax1.add_label(0.30, 0.95, txtstr, relative=True)
    if subfig3 == 13:
      ax1.add_label(0.25, 0.95, txtstr, relative=True)
    if subfig3 == 16:
      ax1.add_label(0.30, 0.95, txtstr, relative=True)
    min_flux = 1*10**19*(39.01*34.06)/(1.36*21*21*1.823*1000*(10**18))
    lvls = np.array([1, 5, 10, 20, 50, 70, 100])
    lvls = lvls*min_flux
    if subfig3 == 1 or subfig3 == 4 or subfig3 == 10:
      atca_lvl = [1*10**19*(39.01*34.06)/(1.36*21*21*1.823*1000*(10**18))]
      ax1.show_contour(image3, colors="grey", levels=atca_lvl, linewidths=4, slices=(0,1))
    ax1.show_contour(image2, colors="blue", levels=lvls, slices=(0,1))
    #ax1.contour(image2, colors='blue', alpha=0.5, levels=lvls, transform=ax1.get_transform(wcs2))
  if subfig3 == 2:
    vel_range = 283
    vel_sys   = 2314
    lvls = np.array([-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    lvls = lvls*20 + vel_sys
    ax1.show_contour(image2, colors="blue", levels=[vel_sys], linewidths=5, returnlevels=True, slices=(0,1))
    ax1.show_contour(image2, colors="blue", levels=lvls, returnlevels=True, slices=(0,1))
    #ax1.contour(image2, colors='blue', alpha=0.5, levels=lvls, transform=ax1.get_transform(wcs2))
    #ax1.imshow(image2, origin='lower', aspect='auto', cmap='RdBu_r', transform=ax1.get_transform(wcs2), clim=(2150,2450))
  if subfig3 == 5:
    vel_range = 132
    vel_sys   = 2271
    lvls = np.array([-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    lvls = lvls*20 + vel_sys
    ax1.show_contour(image2, colors="blue", levels=[vel_sys], linewidths=5, returnlevels=True, slices=(0,1))
    ax1.show_contour(image2, colors="blue", levels=lvls, returnlevels=True, slices=(0,1))
    #plt.clabel(cs, lvls, inline=True)
    #ax1.contour(image2, colors='blue', alpha=0.5, levels=lvls, transform=ax1.get_transform(wcs2))
    #ax1.imshow(image2, origin='lower', aspect='auto', cmap='RdBu_r', transform=ax1.get_transform(wcs2), clim=(2200,2350))
  if subfig3 == 8:
    vel_range = 378
    vel_sys   = 2481
    lvls = np.array([-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    lvls = lvls*20 + vel_sys
    ax1.show_contour(image2, colors="blue", levels=[vel_sys], linewidths=5, returnlevels=True, slices=(0,1))
    ax1.show_contour(image2, colors="blue", levels=lvls, returnlevels=True, slices=(0,1))
    #ax1.clabel(cs, lvls, inline=True)
    #ax1.contour(image2, colors='blue', alpha=0.5, levels=lvls, transform=ax1.get_transform(wcs2))
    #ax1.imshow(image2, origin='lower', aspect='auto', cmap='RdBu_r', transform=ax1.get_transform(wcs2), clim=(2300,2700))
  if subfig3 == 11:
    vel_range = 128
    vel_sys   = 2641
    lvls = np.array([-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    lvls = lvls*20 + vel_sys
    ax1.show_contour(image2, colors="blue", levels=[vel_sys], linewidths=5, returnlevels=True, slices=(0,1))
    ax1.show_contour(image2, colors="blue", levels=lvls, returnlevels=True, slices=(0,1))
    #ax1.clabel(cs, lvls, inline=True)
    #ax1.contour(image2, colors='blue', alpha=0.5, levels=lvls, transform=ax1.get_transform(wcs2))
    #ax1.imshow(image2, origin='lower', aspect='auto', cmap='RdBu_r', transform=ax1.get_transform(wcs2), clim=(2550,2700))
  if subfig3 == 14:
    vel_range = 72
    vel_sys   = 2558
    lvls = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
    lvls = lvls*7 + vel_sys
    ax1.show_contour(image2, colors="blue", levels=[vel_sys], linewidths=5, returnlevels=True, slices=(0,1))
    ax1.show_contour(image2, colors="blue", levels=lvls, returnlevels=True, slices=(0,1))
    #ax1.clabel(cs, lvls, inline=True)
    #ax1.contour(image2, colors='blue', alpha=0.5, levels=lvls, transform=ax1.get_transform(wcs2))
    #ax1.imshow(image2, origin='lower', aspect='auto', cmap='RdBu_r', transform=ax1.get_transform(wcs2), clim=(2500,2600))
  if subfig3 == 17:
    vel_range = 40
    vel_sys   = 2727
    lvls = np.array([-3, -2, -1, 0, 1, 2, 3])
    #lvls = np.array([0, 1, 3, 5, 7, 9])
    lvls = lvls*7 + vel_sys
    ax1.show_contour(image2, colors="blue", levels=[vel_sys], linewidths=5, returnlevels=True, slices=(0,1))
    ax1.show_contour(image2, colors="blue", levels=lvls, returnlevels=True, slices=(0,1))
    #ax1.clabel(cs, lvls, inline=True)
    #ax1.contour(image2, colors='blue', alpha=0.5, levels=lvls, transform=ax1.get_transform(wcs2))
    #ax1.imshow(image2, origin='lower', aspect='auto', cmap='RdBu_r', transform=ax1.get_transform(wcs2), clim=(2710,2760))
  if subfig3 == 1 or subfig3 == 4 or subfig3 == 7 or subfig3 == 10 or subfig3 == 13 or subfig3 == 16:
  #  ax1.set_ylabel(r'Declination')
    ax1.axis_labels.set_ytext('Declination')
  else:
    ax1.axis_labels.hide_y()
  if subfig3 == 16 or subfig3 == 17 or subfig3 == 10 or subfig3 == 11:
  #  ax1.set_xlabel(r'Right Ascension')
    ax1.axis_labels.set_xtext('Right Ascension')
  else:
    ax1.axis_labels.hide_x()
  if subfig3 == 2 or subfig3 == 5 or subfig3 == 8 or subfig3 == 11 or subfig3 == 14 or subfig3 == 17:
    ax1.tick_labels.hide_y()
  ax1.tick_labels.set_xformat('hh:mm:ss')
  ax1.tick_labels.set_yformat('dd:mm')
  ax1.ticks.show()
  ax1.ticks.set_xspacing(0.05)
  ax1.ticks.set_length(5)
  ax1.ticks.set_color('black')
  ax1.ticks.set_minor_frequency(1)
  #ax1.tick_labels.set_font(style='normal', variant='normal')
  ax1.tick_labels.set_style('plain')
  plt.subplots_adjust(wspace=0.25, hspace=0.15)
  #ax1.tick_labels.set_font(size= 'medium', weight= 'medium', stretch= 'normal', family= 'sans-serif', style='normal', variant='normal')


def field_mom_plot(fig_num, subfig1, subfig2, subfig3, image1, image2, colour, txtstr, vel_sys):
  matplotlib.rcParams.update({'font.size': 24})
  rc('font',**{'family':'serif','serif':['Times']})
  rc('text', usetex=True)
  if subfig3 == 1:
    subfig_param = [0.05,0.71,0.27,0.2]
  if subfig3 == 2:
    subfig_param = [0.33,0.71,0.27,0.2]
  if subfig3 == 4:
    subfig_param = [0.05,0.49,0.27,0.2]
  if subfig3 == 5:
    subfig_param = [0.33,0.49,0.27,0.2]
  if subfig3 == 7:
    subfig_param = [0.05,0.27,0.27,0.2]
  if subfig3 == 8:
    subfig_param = [0.33,0.27,0.27,0.2]
  if subfig3 == 10:
    subfig_param = [0.05,0.05,0.27,0.2]
  if subfig3 == 11:
    subfig_param = [0.33,0.05,0.27,0.2]
  #if subfig3 == 13:
    #subfig_param = [0.05,0.50,0.27,0.4]
  #if subfig3 == 14:
    #subfig_param = [0.33,0.50,0.27,0.4]
  #if subfig3 == 16:
    #subfig_param = [0.05,0.05,0.27,0.4]
  #if subfig3 == 17:
    #subfig_param = [0.33,0.05,0.27,0.4]
  ax1 = aplpy.FITSFigure(image1, figure=fig_num, subplot=subfig_param, slices=(0,1))
  if subfig3 == 1 or subfig3 == 4 or subfig3 == 7 or subfig3 == 10:
    f1        = pyfits.open(image1)
    data, hdr = f1[0].data, f1[0].header
    try:
      beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], hdr['BPA']
    except KeyError:
      beam_maj = 0
      for i in range(len(hdr['HISTORY'])):
        if beam_maj == 0:
          beam_hist = str(hdr['HISTORY'][i:i+1]).split()
          for beam_i in range(len(beam_hist)):
            if beam_hist[beam_i] == 'BMAJ=':
              beam_maj = float(beam_hist[beam_i + 1])
            elif beam_hist[beam_i] == 'BMIN=':
              beam_min = float(beam_hist[beam_i + 1])
            elif beam_hist[beam_i] == 'BPA=':
              beam_pa = float(beam_hist[beam_i + 1])
    ax1.show_colorscale(vmin=0, vmax=np.max(data), cmap='Blues')
    #ax1.show_beam(major=beam_maj, minor=beam_min, angle=beam_pa, fill=False, color='peru')
    ax1.add_beam(fill=False, color='peru')
    ax1.beam.set_linewidth(2)
    ax1.beam.set_hatch('/')
  if subfig3 == 2 or subfig3 == 5 or subfig3 == 8 or subfig3 == 11:
    f1        = pyfits.open(image1)
    data, hdr = f1[0].data, f1[0].header
    #freq_min = vel_sys - 75 #5e5
    #freq_max = vel_sys + 75 #5e5
    #if txtstr == 'LVHIS005':
    #  freq_min = 25
    #  freq_max = 450
    #else:
    freq_min = np.nanmean(data) - 2.5*madev(data, ignore_nan=True) #np.nanmin(data)
    freq_max = np.nanmean(data) + 2.5*madev(data, ignore_nan=True) #np.nanmax(data)
    #if txtstr == 'LVHIS005':
    #  freq_min = 30
    #  freq_max = 440
    ax1.show_colorscale(vmin=freq_min, vmax=freq_max, cmap='RdBu_r')
  #if txtstr == 'a) NGC 7162':
    #position     = SkyCoord('21h59m39.1s', '-43d18m21s', frame='icrs')
    #size         = u.Quantity((360, 360), u.arcsec)
    #width, height = 420./60./60., 420./60./60.
  #ax1.recenter(position.ra, position.dec, width=width, height=height)
  if subfig3 == 1 or subfig3 == 4 or subfig3 == 7 or subfig3 == 10:
    #plt.text(1, 5, txtstr, fontsize=16)
    #if subfig3 == 1:
    ax1.add_label(0.20, 0.95, txtstr, relative=True)
    #if subfig3 == 4:
      #ax1.add_label(0.25, 0.95, txtstr, relative=True)
    #if subfig3 == 7:
      #ax1.add_label(0.30, 0.95, txtstr, relative=True)
    #if subfig3 == 10:
      #ax1.add_label(0.30, 0.95, txtstr, relative=True)
    #if subfig3 == 13:
    #  ax1.add_label(0.25, 0.95, txtstr, relative=True)
    #if subfig3 == 16:
      #ax1.add_label(0.30, 0.95, txtstr, relative=True)
    #min_flux = 1*10**19*(39.01*34.06)/(1.36*21*21*1.823*1000*(10**18))
    #lvls = np.array([1, 5, 10, 20, 50, 70, 100])
    #lvls = lvls*min_flux
    #ax1.show_contour(image2, colors="blue", levels=lvls, slices=(0,1))
  #if subfig3 == 2:
    #lvls = np.array([-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    #lvls = lvls*20e7 + vel_sys
    #ax1.show_contour(image2, colors="blue", levels=[vel_sys], linewidths=5, returnlevels=True, slices=(0,1))
    #ax1.show_contour(image2, colors="blue", levels=lvls, returnlevels=True, slices=(0,1))
  #if subfig3 == 5:
    #lvls = np.array([-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    #lvls = lvls*20e7 + vel_sys
    #ax1.show_contour(image2, colors="blue", levels=[vel_sys], linewidths=5, returnlevels=True, slices=(0,1))
    #ax1.show_contour(image2, colors="blue", levels=lvls, returnlevels=True, slices=(0,1))
  #if subfig3 == 8:
    #lvls = np.array([-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    #lvls = lvls*20e7 + vel_sys
    #ax1.show_contour(image2, colors="blue", levels=[vel_sys], linewidths=5, returnlevels=True, slices=(0,1))
    #ax1.show_contour(image2, colors="blue", levels=lvls, returnlevels=True, slices=(0,1))
  #if subfig3 == 11:
    #lvls = np.array([-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    #lvls = lvls*20e7 + vel_sys
    #ax1.show_contour(image2, colors="blue", levels=[vel_sys], linewidths=5, returnlevels=True, slices=(0,1))
    #ax1.show_contour(image2, colors="blue", levels=lvls, returnlevels=True, slices=(0,1))
  #if subfig3 == 14:
    #lvls = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
    #lvls = lvls*7e7 + vel_sys
    #ax1.show_contour(image2, colors="blue", levels=[vel_sys], linewidths=5, returnlevels=True, slices=(0,1))
    #ax1.show_contour(image2, colors="blue", levels=lvls, returnlevels=True, slices=(0,1))
  #if subfig3 == 17:
    #lvls = np.array([-3, -2, -1, 0, 1, 2, 3])
    #lvls = lvls*7e7 + vel_sys
    #ax1.show_contour(image2, colors="blue", levels=[vel_sys], linewidths=5, returnlevels=True, slices=(0,1))
    #ax1.show_contour(image2, colors="blue", levels=lvls, returnlevels=True, slices=(0,1))
  if subfig3 == 1 or subfig3 == 4 or subfig3 == 7 or subfig3 == 10 or subfig3 == 13 or subfig3 == 16:
  #  ax1.set_ylabel(r'Declination')
    ax1.axis_labels.set_ytext('Declination')
  else:
    ax1.axis_labels.hide_y()
  if subfig3 == 16 or subfig3 == 17 or subfig3 == 10 or subfig3 == 11:
  #  ax1.set_xlabel(r'Right Ascension')
    ax1.axis_labels.set_xtext('Right Ascension')
  else:
    ax1.axis_labels.hide_x()
  if subfig3 == 2 or subfig3 == 5 or subfig3 == 8 or subfig3 == 11 or subfig3 == 14 or subfig3 == 17:
    ax1.tick_labels.hide_y()
  ax1.tick_labels.set_xformat('hh:mm:ss')
  ax1.tick_labels.set_yformat('dd:mm')
  ax1.ticks.show()
  ax1.ticks.set_xspacing(0.05)
  ax1.ticks.set_length(5)
  ax1.ticks.set_color('black')
  ax1.ticks.set_minor_frequency(1)
  #ax1.tick_labels.set_style('plain')
  plt.subplots_adjust(wspace=0.25, hspace=0.15)
  f1.close()


def askap_mom_plot(fig_num, subfig1, subfig2, subfig3, image1, image2, colour, txtstr, vel_sys):
  matplotlib.rcParams.update({'font.size': 24})
  rc('font',**{'family':'serif','serif':['Times']})
  rc('text', usetex=True)
  if subfig3 == 1:
    subfig_param = [0.05,0.71,0.27,0.2]
  if subfig3 == 2:
    subfig_param = [0.33,0.71,0.27,0.2]
  if subfig3 == 4:
    subfig_param = [0.05,0.49,0.27,0.2]
  if subfig3 == 5:
    subfig_param = [0.33,0.49,0.27,0.2]
  if subfig3 == 7:
    subfig_param = [0.05,0.27,0.27,0.2]
  if subfig3 == 8:
    subfig_param = [0.33,0.27,0.27,0.2]
  if subfig3 == 10:
    subfig_param = [0.05,0.05,0.27,0.2]
  if subfig3 == 11:
    subfig_param = [0.33,0.05,0.27,0.2]
  #if subfig3 == 13:
    #subfig_param = [0.05,0.50,0.27,0.4]
  #if subfig3 == 14:
    #subfig_param = [0.33,0.50,0.27,0.4]
  #if subfig3 == 16:
    #subfig_param = [0.05,0.05,0.27,0.4]
  #if subfig3 == 17:
    #subfig_param = [0.33,0.05,0.27,0.4]
  #f1        = pyfits.open(image1, mode='update')
  #data, hdr = f1[0].data, f1[0].header
  ##del hdr['CTYPE4']
  ##del hdr['CRVAL4']
  ##del hdr['CDELT4']
  ##del hdr['CRPIX4']
  ##del hdr['CUNIT4']
  ##f1.flush()
  #f1.close()
  #wcs = WCS(hdr, naxis=2)
  #ax1 = fig_num.add_axes(subfig_param, facecolor = 'w', projection=wcs)
  #ax1.tick_params(direction='in')
  ##ax1 = fig_num.add_subplot(subfig1, subfig2, subfig3, facecolor = 'w', projection=wcs, slices=('x', 'y', 0))
  #lon = ax1.coords[0]
  #lat = ax1.coords[1]
  #lon.set_major_formatter('hh:mm:ss')
  #lat.set_major_formatter('dd:mm')
  #lon.set_separator(('h', 'm', 's'))
  #lat.set_separator(('d', 'm'))
  #ax1.imshow(data, origin='lower', aspect='auto', cmap='Blues', clim=(0,np.max(data)))
  #ax1.set_autoscale_on(False)
  ax1 = aplpy.FITSFigure(image1, figure=fig_num, subplot=subfig_param, dimensions=(0,1))# slices=(0,1))
  if subfig3 == 1 or subfig3 == 4 or subfig3 == 7 or subfig3 == 10:
    f1        = pyfits.open(image1)
    data, hdr = f1[0].data, f1[0].header
    try:
      beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], hdr['BPA']
    except KeyError:
      beam_maj = 0
      for i in range(len(hdr['HISTORY'])):
        if beam_maj == 0:
          beam_hist = str(hdr['HISTORY'][i:i+1]).split()
          for beam_i in range(len(beam_hist)):
            if beam_hist[beam_i] == 'BMAJ=':
              beam_maj = float(beam_hist[beam_i + 1])
            elif beam_hist[beam_i] == 'BMIN=':
              beam_min = float(beam_hist[beam_i + 1])
            elif beam_hist[beam_i] == 'BPA=':
              beam_pa = float(beam_hist[beam_i + 1])
    ax1.show_colorscale(vmin=0, vmax=np.max(data), cmap='Blues')
    #ax1.show_beam(major=beam_maj, minor=beam_min, angle=beam_pa, fill=False, color='peru')
    ax1.add_beam(major=beam_maj, minor=beam_min, angle=beam_pa, fill=False, color='peru')
    #matplotlib.patches.Ellipse((1,1), width=beam_maj, height=beam_min, angle=beam_pa, fill=False, color='peru')
    #ax1.beam.set_linewidth(2)
    #ax1.beam.set_hatch('/')
  if subfig3 == 2 or subfig3 == 5 or subfig3 == 8 or subfig3 == 11:
    f1        = pyfits.open(image1)
    data, hdr = f1[0].data, f1[0].header
    #freq_min = vel_sys - 75 #5e5
    #freq_max = vel_sys + 75 #5e5
    #if txtstr == 'LVHIS005':
    #  freq_min = 25
    #  freq_max = 450
    #else:
    freq_min = np.nanmean(data) - 2.5*madev(data, ignore_nan=True) #np.nanmin(data)
    freq_max = np.nanmean(data) + 2.5*madev(data, ignore_nan=True) #np.nanmax(data)
    #if txtstr == 'LVHIS005':
    #  freq_min = 30
    #  freq_max = 440
    ax1.show_colorscale(vmin=freq_min, vmax=freq_max, cmap='RdBu_r')
  #if txtstr == 'a) NGC 7162':
    #position     = SkyCoord('21h59m39.1s', '-43d18m21s', frame='icrs')
    #size         = u.Quantity((360, 360), u.arcsec)
    #width, height = 420./60./60., 420./60./60.
  #ax1.recenter(position.ra, position.dec, width=width, height=height)
  if subfig3 == 1 or subfig3 == 4 or subfig3 == 7 or subfig3 == 10:
    #plt.text(1, 5, txtstr, fontsize=16)
    #if subfig3 == 1:
    ax1.add_label(0.3, 0.95, txtstr, relative=True)
    #if subfig3 == 4:
      #ax1.add_label(0.25, 0.95, txtstr, relative=True)
    #if subfig3 == 7:
      #ax1.add_label(0.30, 0.95, txtstr, relative=True)
    #if subfig3 == 10:
      #ax1.add_label(0.30, 0.95, txtstr, relative=True)
    #if subfig3 == 13:
    #  ax1.add_label(0.25, 0.95, txtstr, relative=True)
    #if subfig3 == 16:
      #ax1.add_label(0.30, 0.95, txtstr, relative=True)
    min_flux = 1*10**19*(39.01*34.06)/(1.36*21*21*1.823*1000*(10**18))
    lvls = np.array([1, 5, 10, 20, 50, 70, 100])
    lvls = lvls*min_flux
    ax1.show_contour(image2, colors="blue", levels=lvls, slices=(0,1))
  #if subfig3 == 2:
    #lvls = np.array([-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    #lvls = lvls*20e7 + vel_sys
    #ax1.show_contour(image2, colors="blue", levels=[vel_sys], linewidths=5, returnlevels=True, slices=(0,1))
    #ax1.show_contour(image2, colors="blue", levels=lvls, returnlevels=True, slices=(0,1))
  #if subfig3 == 5:
    #lvls = np.array([-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    #lvls = lvls*20e7 + vel_sys
    #ax1.show_contour(image2, colors="blue", levels=[vel_sys], linewidths=5, returnlevels=True, slices=(0,1))
    #ax1.show_contour(image2, colors="blue", levels=lvls, returnlevels=True, slices=(0,1))
  #if subfig3 == 8:
    #lvls = np.array([-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    #lvls = lvls*20e7 + vel_sys
    #ax1.show_contour(image2, colors="blue", levels=[vel_sys], linewidths=5, returnlevels=True, slices=(0,1))
    #ax1.show_contour(image2, colors="blue", levels=lvls, returnlevels=True, slices=(0,1))
  #if subfig3 == 11:
    #lvls = np.array([-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    #lvls = lvls*20e7 + vel_sys
    #ax1.show_contour(image2, colors="blue", levels=[vel_sys], linewidths=5, returnlevels=True, slices=(0,1))
    #ax1.show_contour(image2, colors="blue", levels=lvls, returnlevels=True, slices=(0,1))
  #if subfig3 == 14:
    #lvls = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
    #lvls = lvls*7e7 + vel_sys
    #ax1.show_contour(image2, colors="blue", levels=[vel_sys], linewidths=5, returnlevels=True, slices=(0,1))
    #ax1.show_contour(image2, colors="blue", levels=lvls, returnlevels=True, slices=(0,1))
  #if subfig3 == 17:
    #lvls = np.array([-3, -2, -1, 0, 1, 2, 3])
    #lvls = lvls*7e7 + vel_sys
    #ax1.show_contour(image2, colors="blue", levels=[vel_sys], linewidths=5, returnlevels=True, slices=(0,1))
    #ax1.show_contour(image2, colors="blue", levels=lvls, returnlevels=True, slices=(0,1))
  if subfig3 == 1 or subfig3 == 4 or subfig3 == 7 or subfig3 == 10 or subfig3 == 13 or subfig3 == 16:
  #  ax1.set_ylabel(r'Declination')
    ax1.axis_labels.set_ytext('Declination')
  else:
    ax1.axis_labels.hide_y()
  if subfig3 == 16 or subfig3 == 17 or subfig3 == 10 or subfig3 == 11:
  #  ax1.set_xlabel(r'Right Ascension')
    ax1.axis_labels.set_xtext('Right Ascension')
  else:
    ax1.axis_labels.hide_x()
  if subfig3 == 2 or subfig3 == 5 or subfig3 == 8 or subfig3 == 11 or subfig3 == 14 or subfig3 == 17:
    ax1.tick_labels.hide_y()
  ax1.tick_labels.set_xformat('hh:mm:ss')
  ax1.tick_labels.set_yformat('dd:mm')
  ax1.ticks.show()
  ax1.ticks.set_xspacing(0.05)
  ax1.ticks.set_length(5)
  ax1.ticks.set_color('black')
  ax1.ticks.set_minor_frequency(1)
  #ax1.tick_labels.set_style('plain')
  plt.subplots_adjust(wspace=0.25, hspace=0.15)
  f1.close()


def askap_contour_plot(fig_num, subfig1, subfig2, subfig3, image1, image2, colour, txtstr, vel_sys):
  matplotlib.rcParams.update({'font.size': 22})
  rc('font',**{'family':'serif','serif':['Times']})
  rc('text', usetex=True)
  if subfig3 == 1:
    subfig_param = [0.05,0.71,0.24,0.19]
  if subfig3 == 2:
    subfig_param = [0.3,0.71,0.24,0.19]
  if subfig3 == 4:
    subfig_param = [0.05,0.49,0.24,0.19]
  if subfig3 == 5:
    subfig_param = [0.3,0.49,0.24,0.19]
  if subfig3 == 7:
    subfig_param = [0.05,0.27,0.24,0.19]
  if subfig3 == 8:
    subfig_param = [0.3,0.27,0.24,0.19]
  if subfig3 == 10:
    subfig_param = [0.05,0.05,0.24,0.19]
  if subfig3 == 11:
    subfig_param = [0.3,0.05,0.24,0.19]
  #if subfig3 == 13:
    #subfig_param = [0.05,0.50,0.27,0.4]
  #if subfig3 == 14:
    #subfig_param = [0.33,0.50,0.27,0.4]
  #if subfig3 == 16:
    #subfig_param = [0.05,0.05,0.27,0.4]
  #if subfig3 == 17:
    #subfig_param = [0.33,0.05,0.27,0.4]
  ax1 = aplpy.FITSFigure(image1, figure=fig_num, subplot=subfig_param, dimensions=(0,1))# slices=(0,1))
  if subfig3 == 1 or subfig3 == 4 or subfig3 == 7 or subfig3 == 10:
    f2        = pyfits.open(image2)
    data, hdr = f2[0].data, f2[0].header
    try:
      beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], hdr['BPA']
    except KeyError:
      beam_maj = 0
      for i in range(len(hdr['HISTORY'])):
        if beam_maj == 0:
          beam_hist = str(hdr['HISTORY'][i:i+1]).split()
          for beam_i in range(len(beam_hist)):
            if beam_hist[beam_i] == 'BMAJ=':
              beam_maj = float(beam_hist[beam_i + 1])
            elif beam_hist[beam_i] == 'BMIN=':
              beam_min = float(beam_hist[beam_i + 1])
            elif beam_hist[beam_i] == 'BPA=':
              beam_pa = float(beam_hist[beam_i + 1])
    bmaj, bmin = beam_maj*60.*60., beam_min*60.*60.
    f1        = pyfits.open(image1)
    data1, hdr1 = f1[0].data, f1[0].header
    #print(np.min(data1), np.max(data1))
    ax1.show_colorscale(vmin=4000, vmax=np.max(data1), cmap='Greys')
    ax1.add_beam(major=beam_maj, minor=beam_min, angle=beam_pa, fill=True, color='peru')
  if subfig3 == 2 or subfig3 == 5 or subfig3 == 8 or subfig3 == 11:
    f1        = pyfits.open(image1)
    data, hdr = f1[0].data, f1[0].header
    freq_min = np.nanmean(data) - 2.5*madev(data, ignore_nan=True) #np.nanmin(data)
    freq_max = np.nanmean(data) + 2.5*madev(data, ignore_nan=True) #np.nanmax(data)
    ax1.show_colorscale(vmin=freq_min, vmax=freq_max, cmap='RdBu_r')
  ra  = '%sh%sm%ss' % (txtstr[1:3], txtstr[3:5], txtstr[5:7])
  dec = '%sd%sm%ss' % (txtstr[7:10], txtstr[10:12], txtstr[12:])
  position     = SkyCoord(ra, dec, frame='icrs')
  size         = u.Quantity((360, 360), u.arcsec)
  width, height = 420./60./60., 420./60./60.
  ax1.recenter(position.ra, position.dec, width=width, height=height)
  if subfig3 == 1 or subfig3 == 4 or subfig3 == 7 or subfig3 == 10:
    #plt.text(1, 5, txtstr, fontsize=16)
    #if subfig3 == 1:
    ax1.add_label(0.35, 0.9, txtstr, relative=True)
    #if subfig3 == 4:
      #ax1.add_label(0.25, 0.95, txtstr, relative=True)
    #if subfig3 == 7:
      #ax1.add_label(0.30, 0.95, txtstr, relative=True)
    #if subfig3 == 10:
      #ax1.add_label(0.30, 0.95, txtstr, relative=True)
    #if subfig3 == 13:
    #  ax1.add_label(0.25, 0.95, txtstr, relative=True)
    #if subfig3 == 16:
      #ax1.add_label(0.30, 0.95, txtstr, relative=True)
    min_flux = 1*10**19*(bmaj*bmin)/(2.33*10**20) #1.36*21*21*1.823*1000*(10**18))
    lvls = np.array([1, 5, 10, 20, 50, 70, 100, 130])
    lvls = lvls*min_flux
    ax1.show_contour(image2, colors="blue", levels=lvls, slices=(0,1))
  #if subfig3 == 2:
    #lvls = np.array([-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    #lvls = lvls*20e7 + vel_sys
    #ax1.show_contour(image2, colors="blue", levels=[vel_sys], linewidths=5, returnlevels=True, slices=(0,1))
    #ax1.show_contour(image2, colors="blue", levels=lvls, returnlevels=True, slices=(0,1))
  #if subfig3 == 5:
    #lvls = np.array([-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    #lvls = lvls*20e7 + vel_sys
    #ax1.show_contour(image2, colors="blue", levels=[vel_sys], linewidths=5, returnlevels=True, slices=(0,1))
    #ax1.show_contour(image2, colors="blue", levels=lvls, returnlevels=True, slices=(0,1))
  #if subfig3 == 8:
    #lvls = np.array([-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    #lvls = lvls*20e7 + vel_sys
    #ax1.show_contour(image2, colors="blue", levels=[vel_sys], linewidths=5, returnlevels=True, slices=(0,1))
    #ax1.show_contour(image2, colors="blue", levels=lvls, returnlevels=True, slices=(0,1))
  #if subfig3 == 11:
    #lvls = np.array([-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    #lvls = lvls*20e7 + vel_sys
    #ax1.show_contour(image2, colors="blue", levels=[vel_sys], linewidths=5, returnlevels=True, slices=(0,1))
    #ax1.show_contour(image2, colors="blue", levels=lvls, returnlevels=True, slices=(0,1))
  #if subfig3 == 14:
    #lvls = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
    #lvls = lvls*7e7 + vel_sys
    #ax1.show_contour(image2, colors="blue", levels=[vel_sys], linewidths=5, returnlevels=True, slices=(0,1))
    #ax1.show_contour(image2, colors="blue", levels=lvls, returnlevels=True, slices=(0,1))
  #if subfig3 == 17:
    #lvls = np.array([-3, -2, -1, 0, 1, 2, 3])
    #lvls = lvls*7e7 + vel_sys
    #ax1.show_contour(image2, colors="blue", levels=[vel_sys], linewidths=5, returnlevels=True, slices=(0,1))
    #ax1.show_contour(image2, colors="blue", levels=lvls, returnlevels=True, slices=(0,1))
  if subfig3 == 1 or subfig3 == 4 or subfig3 == 7 or subfig3 == 10 or subfig3 == 13 or subfig3 == 16:
  #  ax1.set_ylabel(r'Declination')
    ax1.axis_labels.set_ytext('Declination')
  else:
    ax1.axis_labels.hide_y()
  if subfig3 == 16 or subfig3 == 17 or subfig3 == 10 or subfig3 == 11:
  #  ax1.set_xlabel(r'Right Ascension')
    ax1.axis_labels.set_xtext('Right Ascension')
  else:
    ax1.axis_labels.hide_x()
  if subfig3 == 2 or subfig3 == 5 or subfig3 == 8 or subfig3 == 11 or subfig3 == 14 or subfig3 == 17:
    ax1.tick_labels.hide_y()
  ax1.tick_labels.set_xformat('hh:mm:ss')
  ax1.tick_labels.set_yformat('dd:mm')
  ax1.ticks.show()
  ax1.ticks.set_xspacing(0.05)
  ax1.ticks.set_length(5)
  ax1.ticks.set_color('black')
  ax1.ticks.set_minor_frequency(1)
  #ax1.ticks.set_tick_direction('in')
  #ax1.tick_labels.set_style('plain')
  #plt.subplots_adjust(wspace=0.25, hspace=0.15)
  f1.close()


def spec_panel_plot(fig_num, sub1, sub2, sub3, velocity, flux, error, colour, vsys):
    matplotlib.rcParams.update({'font.size': 22})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    #ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    if sub3 == 3:
      subfig_param = [0.62,0.71,0.35,0.19]
    if sub3 == 6:
      subfig_param = [0.62,0.49,0.35,0.19]
    if sub3 == 9:
      subfig_param = [0.62,0.27,0.35,0.19]
    if sub3 == 12:
      subfig_param = [0.62,0.05,0.35,0.19]
    ax1 = fig_num.add_axes(subfig_param, facecolor = 'w')
    if sub1*sub2 == 12:
      if sub3 == 12:
        ax1.set_xlabel(r'Velocity [km\,s$^{-1}$]')
      if sub3 == 3 or sub3 == 6 or sub3 == 9 or sub3 ==12:
        ax1.set_ylabel(r'Flux [mJy]')
    #half_vel = 4.0*len(velocity)/2.0
    #ax1.set_xlim(vsys-half_vel, vsys+half_vel)
    flux = flux*1000.0
    ax1.set_ylim(np.min(flux), np.max(flux))
    #plt.text(0.1, 0.8, txtstr, transform=ax1.transAxes)
    #plt.axhline(0, linewidth=0.75, linestyle = ':', color = 'darkgrey')
    plt.axvline(vsys, linewidth=0.75, linestyle = '--', color = 'darkgrey', label = r'$V_{\mathrm{sys,SoFiA}}$')
    plt.plot(velocity, flux, linestyle = '-', color = colour, linewidth = 1.0)#, label = txtstr)
    #frequency = (1.420405751786*pow(10,9))/(velocity/C_LIGHT + 1)
    #if colour == 'peru':
    error = np.array(error)*1000.0
    plt.fill_between(velocity, flux-error, flux+error, alpha=0.5, edgecolor='none', facecolor='lightblue')
    #if sub3 == 4:
      #ax1.legend(loc='upper right', fontsize = 8.5)
    #plt.subplots_adjust(wspace=0.0, hspace=0.0)
    ax1.get_yaxis().set_tick_params(which='both', direction='in')
    ax1.get_xaxis().set_tick_params(which='both', direction='in')
    #return velocity, flux


def atca_mom_plot(fig_num, subfig1, subfig2, subfig3, image1, image2, colour, lsty, txtstr, lbl, wcs, wcs2):
  matplotlib.rcParams.update({'font.size': 15})
  rc('font',**{'family':'serif','serif':['Times']})
  rc('text', usetex=True)
  ax1 = fig_num.add_subplot(subfig1, subfig2, subfig3, facecolor = 'w', projection=wcs)
  ax1.tick_params(direction='in')
  #ax1 = fig_num.add_subplot(subfig1, subfig2, subfig3, facecolor = 'w', projection=wcs, slices=('x', 'y', 0))
  if subfig3 == 1:# or subfig3 == 4 or subfig3 == 7 or subfig3 == 10 or subfig3 == 13:
    ax1.set_ylabel(r'Declination')
  #if subfig3 == 2:# or subfig3 == 14:
  ax1.set_xlabel(r'Right Ascension')
  lon = ax1.coords[0]
  lat = ax1.coords[1]
  lon.set_major_formatter('hh:mm:ss')
  lat.set_major_formatter('dd:mm')
  lon.set_separator(('h', 'm', 's'))
  lat.set_separator(('d', 'm'))
  ax1.imshow(image1, origin='lower', aspect='auto', cmap='Greys', clim=(4000,13000))#norm=colors.Norm(vmin=4000, vmax=13000))
  ax1.set_autoscale_on(False)
  #if subfig3 == 1 or subfig3 == 4 or subfig3 == 7 or subfig3 == 10 or subfig3 == 13:
  plt.text(1, 5, txtstr, fontsize=16)
  #max_flux =  np.max(image2)
  #print max_flux
  #lvls = np.array([0.01, 0.1, 0.3, 0.5, 0.7, 0.9])
  #lvls = lvls*max_flux
  min_flux = 1*10**19*(39.01*34.06)/(1.36*21*21*1.823*1000*(10**18))
  lvls = np.array([1, 10, 30, 50, 70, 100, 130, 160])
  lvls = lvls*min_flux
  ax1.contour(image2, colors='blue', alpha=0.5, levels=lvls, transform=ax1.get_transform(wcs2))
  plt.subplots_adjust(wspace=0.3, hspace=0.15)
  
def atca_mom_plot2(fig_num, subfig1, subfig2, subfig3, image1, image2, colour, lsty, txtstr, lbl):
  matplotlib.rcParams.update({'font.size': 16})
  rc('font',**{'family':'serif','serif':['Times']})
  rc('text', usetex=True)
  ax1 = aplpy.FITSFigure(image1, figure=fig_num, subplot=(subfig1, subfig2, subfig3))
  ax1.show_beam(major=.0167,minor=.01,angle=10.9,fill=True,color='peru')
  if subfig3 == 1:# or subfig3 == 4 or subfig3 == 7 or subfig3 == 10 or subfig3 == 13:
    ax1.axis_labels.set_ytext('Declination')
  else:
    ax1.axis_labels.hide_y()
  #if subfig3 == 2:# or subfig3 == 14:
  ax1.axis_labels.set_xtext('Right Ascension')
  ax1.tick_labels.set_xformat('hh:mm:ss')
  ax1.tick_labels.set_yformat('dd:mm')
  ax1.ticks.show()
  ax1.ticks.set_xspacing(0.05)
  #ax1.show_colorscale(vmin=4000,vmax=13000,cmap='Greys')
  ax1.show_colorscale(vmin=0,vmax=60,cmap='Greys')
  if txtstr == 'a) NGC 7162':
    position     = SkyCoord('21h59m39.1s', '-43d18m21s', frame='icrs')
    size         = u.Quantity((360, 360), u.arcsec)
    width, height = 420./60./60., 420./60./60.
  if txtstr == 'b) NGC 7162A':
    position     = SkyCoord('22h00m35.7s', '-43d08m30s', frame='icrs')
    size         = u.Quantity((360, 360), u.arcsec)
    width, height = 420./60./60., 420./60./60.
  if txtstr == 'c) ESO 288-G033':
    position     = SkyCoord('22h02m06.6s', '-43d16m07s', frame='icrs')
    size         = u.Quantity((240, 240), u.arcsec)
    width, height = 300./60./60., 300./60./60.
  ax1.recenter(position.ra, position.dec, width=width, height=height)
  #ax1.imshow(image1, origin='lower', aspect='auto', cmap='Greys', clim=(4000,13000))
  #ax1.set_autoscale_on(False)
  #if subfig3 == 1 or subfig3 == 4 or subfig3 == 7 or subfig3 == 10 or subfig3 == 13:
  #plt.text(1, 5, txtstr, fontsize=16)
  min_flux = 1*10**19*(39.01*34.06)/(1.36*21*21*1.823*1000*(10**18))
  lvls = np.array([1, 10, 30, 50, 70, 100, 130, 160])
  lvls = lvls*min_flux
  if subfig3 == 1:
    ax1.add_label(0.76, 0.05, txtstr, relative=True)
  if subfig3 == 2:
    ax1.add_label(0.73, 0.05, txtstr, relative=True)
  if subfig3 == 3:
    ax1.add_label(0.72, 0.05, txtstr, relative=True)
  ax1.show_contour(image2, colors="blue", levels=lvls, slices=(0,1))
  plt.subplots_adjust(wspace=0.3, hspace=0.15)

def fits_plot(fig_num, subfig1, subfig2, subfig3, image1, colour, lsty, txtstr, wcs):
  matplotlib.rcParams.update({'font.size': 13})
  rc('font',**{'family':'serif','serif':['Times']})
  rc('text', usetex=True)
  ax1 = fig_num.add_subplot(subfig1, subfig2, subfig3, facecolor = 'w', projection=wcs)
  plt.text(8, 10, txtstr, fontsize=14)
  if subfig3 == 1 or subfig3 == 3:# or subfig3 == 4 or subfig3 == 7 or subfig3 == 10 or subfig3 == 13:
    ax1.set_ylabel(r'Declination')
  if subfig3 == 3 or subfig3 == 4:
    ax1.set_xlabel(r'Right Ascension')
  lon = ax1.coords[0]
  lat = ax1.coords[1]
  lon.set_major_formatter('hh:mm:ss')
  lat.set_major_formatter('dd:mm')
  lon.set_separator(('h', 'm', 's'))
  lat.set_separator(('d', 'm'))
  if subfig3 == 1 or subfig3 == 2:
    ax1.imshow(image1, origin='lower', aspect='auto', cmap='Greys', clim=(-200,500))
  if subfig3 == 3 or subfig3 == 4:
    ax1.imshow(image1, origin='lower', aspect='auto', cmap='Greys', clim=(-100,250))
  ax1.set_autoscale_on(False)
  plt.subplots_adjust(wspace=0.3, hspace=0.15)

def fits_plot2(fig_num, subfig1, subfig2, subfig3, image1, colour, lsty, txtstr):
  matplotlib.rcParams.update({'font.size': 16})
  rc('font',**{'family':'serif','serif':['Times']})
  rc('text', usetex=True)
  #plt.text(8, 10, txtstr, fontsize=14)
  ax1 = aplpy.FITSFigure(image1, figure=fig_num, subplot=(subfig1, subfig2, subfig3))
  #ax1.show_beam(major=.0108,minor=.009,angle=78,fill=True,color='peru')
  if subfig3 == 1 or subfig3 == 3:
    ax1.axis_labels.set_ytext('Declination')
  else:
    ax1.axis_labels.hide_y()
  if subfig3 == 4 or subfig3 == 3:
    ax1.axis_labels.set_xtext('Right Ascension')
  else:
    ax1.axis_labels.hide_x()
  ax1.tick_labels.set_xformat('hh:mm:ss')
  ax1.tick_labels.set_yformat('dd:mm')
  ax1.ticks.show()
  ax1.ticks.set_xspacing(0.05)
  ax1.show_colorscale(vmin=4000,vmax=13000,cmap='Greys')
  if subfig3 == 1 or subfig3 == 2:
    ax1.show_colorscale(vmin=-200,vmax=500,cmap='Greys')
  if subfig3 == 3 or subfig3 == 4:
    ax1.show_colorscale(vmin=-100,vmax=250,cmap='Greys')
  if subfig3 == 1:
    ax1.add_label(0.32, 0.05, txtstr, relative=True)
  if subfig3 == 2:
    ax1.add_label(0.32, 0.05, txtstr, relative=True)
  if subfig3 == 3:
    ax1.add_label(0.35, 0.05, txtstr, relative=True)
  if subfig3 == 4:
    ax1.add_label(0.35, 0.05, txtstr, relative=True)
  plt.subplots_adjust(wspace=0.25, hspace=0.07)

def spitzer_plot(fig_num, subfig1, subfig2, subfig3, image1, colour, lsty, txtstr):
  matplotlib.rcParams.update({'font.size': 16})
  rc('font',**{'family':'serif','serif':['Times']})
  rc('text', usetex=True)
  #plt.text(8, 10, txtstr, fontsize=14)
  ax1 = aplpy.FITSFigure(image1, figure=fig_num, subplot=(subfig1, subfig2, subfig3))
  #ax1.show_beam(major=.0108,minor=.009,angle=78,fill=True,color='peru')
  if subfig3 == 1 or subfig3 == 2:
    ax1.axis_labels.set_ytext('Declination')
  else:
    ax1.axis_labels.hide_y()
  if subfig3 == 2:# or subfig3 == 3:
    ax1.axis_labels.set_xtext('Right Ascension')
  else:
    ax1.axis_labels.hide_x()
  ax1.tick_labels.set_xformat('hh:mm:ss')
  ax1.tick_labels.set_yformat('dd:mm')
  ax1.ticks.show()
  ax1.ticks.set_xspacing(0.05)
  if txtstr == 'a) NGC 7162':
    ax1.show_colorscale(vmin=0,vmax=0.15,cmap='Greys')
    position     = SkyCoord('21h59m39.1s', '-43d18m21s', frame='icrs')
    size         = u.Quantity((360, 360), u.arcsec)
    width, height = 300./60./60., 300./60./60.
  if txtstr == 'b) NGC 7162A':
    ax1.show_colorscale(vmin=0,vmax=0.1,cmap='Greys')
    position     = SkyCoord('22h00m35.7s', '-43d08m30s', frame='icrs')
    size         = u.Quantity((360, 360), u.arcsec)
    width, height = 300./60./60., 300./60./60.
  ax1.recenter(position.ra, position.dec, width=width, height=height)
  #if subfig3 == 1 or subfig3 == 2:
    #ax1.show_colorscale(vmin=-200,vmax=500,cmap='Greys')
  #if subfig3 == 3 or subfig3 == 4:
    #ax1.show_colorscale(vmin=-100,vmax=250,cmap='Greys')
  if subfig3 == 1:
    ax1.add_label(0.20, 0.05, txtstr, relative=True)
  if subfig3 == 2:
    ax1.add_label(0.22, 0.05, txtstr, relative=True)
  if subfig3 == 3:
    ax1.add_label(0.35, 0.05, txtstr, relative=True)
  if subfig3 == 4:
    ax1.add_label(0.35, 0.05, txtstr, relative=True)
  plt.subplots_adjust(wspace=0.25, hspace=0.1)

def full_field_plot(fig_num, subfig1, subfig2, subfig3, image1, image2, colour, lsty, txtstr, lbl, wcs, wcs2, beams):
  matplotlib.rcParams.update({'font.size': 14})
  rc('font',**{'family':'serif','serif':['Times']})
  rc('text', usetex=True)
  ax1 = fig_num.add_subplot(subfig1, subfig2, subfig3, facecolor = 'w', projection=wcs)
  # ======== Axes Labels ======== #
  ax1.set_ylabel(r'Declination')
  ax1.set_xlabel(r'Right Ascension')
  # ======== Axes Tick Parameters ======== #
  ax1.tick_params(direction='in')
  lon = ax1.coords[0]
  lat = ax1.coords[1]
  lon.set_major_formatter('hh:mm')
  lat.set_major_formatter('dd:mm')
  lon.set_separator(('h', 'm'))
  lat.set_separator(('d', 'm'))
  if txtstr == 'a) NGC 7162':
    # ======== Add backgroud optical image ======== #
    ax1.imshow(image1, origin='lower', aspect='auto', cmap='Greys', clim=(4000,13000))
    # ======== Beam Positions ======== #
    if beams == True:
      beam_ra  = ['22h00m49.120s', '22h00m38.043s', '22h03m08.923s', '21h58m22.194s', '22h03m15.423s', '21h58m11.370s', '22h00m40.680s']
      beam_dec = ['-42d59m20.28s', '-43d53m19.67s', '-43d27m54.60s', '-42d32m16.21s', '-42d33m53.75s', '-43d26m15.67s', '-43d09m50.10s']
      for i in range(len(beam_ra)):
        c    = SkyCoord(beam_ra[i], beam_dec[i], frame='icrs')
        print (c.ra, c.dec)
        if i == 0 or i == 1:
          beam_colour = 'green'
          line_type   = '--'
          beam_size   = 0.5 * u.degree
        elif i == 6:
          beam_colour = 'peru'
          line_type   = '-'
          beam_size   = 0.4 * u.degree
        else:
          beam_colour = 'magenta'
          line_type   = ':'
          beam_size   = 0.5 * u.degree
        beam = SphericalCircle((c.ra, c.dec), beam_size, edgecolor=beam_colour, ls=line_type, facecolor='none', transform=ax1.get_transform('icrs'))
        # ======== Add Beams to image ======== #
        ax1.add_patch(beam)
  ax1.set_autoscale_on(False)
  infile   = '/Users/tflowers/NGC7162A/HIPASS/HIPASS_142.mom0.fits'
  f1        = pyfits.open(infile)
  pks_data = f1[0].data
  pks_hdr  = f1[0].header
  pks_hdr.remove('CTYPE3')
  pks_hdr.remove('CRVAL3')
  pks_hdr.remove('CDELT3')
  pks_hdr.remove('CRPIX3')
  pks_hdr.update(NAXIS=2)
  pks_wcs  = WCS(pks_hdr)
  pks_data = pks_data.reshape(int(pks_hdr['NAXIS2']), int(pks_hdr['NAXIS1']))
  lvls = [2.25] #[12.66]
  ax1.contour(pks_data, colors='purple', alpha=1, linewidths=0.75, levels=lvls, transform=ax1.get_transform(pks_wcs))
  lvls = [0.02]#lvls*max_flux
  ax1.contour(image2, colors=colour, alpha=1, linewidths=0.75, levels=lvls, transform=ax1.get_transform(wcs2))
  if txtstr == 'f) J220338-431131':
    galaxies     = ['NGC 7162', 'NGC 7162A', 'ESO 288-G025', 'ESO 288-G033', 'AM 2159-434', 'J220338-431131', 'NGC 7166']
    if beams == True:
      gal_ra       = ['21h59m20.0s', '22h00m30.0s', '21h59m00.0s', '22h05m45.0s', '22h05m55.0s', '22h07m30.0s', '22h00m30.0s']
      gal_dec      = ['-43d20m00.0s', '-43d05m00.0s', '-43d55m00.0s', '-43d21m00.0s', '-43d34m00.0s', '-43d08m00.0s', '-43d31m00.0s']
    if beams == False:
      gal_ra       = ['21h59m20.0s', '22h00m30.0s', '22h01m00.0s', '22h03m30.0s', '22h03m55.0s', '22h03m45.0s', '22h01m00.0s']
      gal_dec      = ['-43d20m00.0s', '-43d05m00.0s', '-43d54m00.0s', '-43d20m30.0s', '-43d31m00.0s', '-43d09m00.0s', '-43d28m00.0s']
    for i in range(len(galaxies)):
      c    = SkyCoord(gal_ra[i], gal_dec[i], frame='icrs')
      ra_deg,dec_deg  = float(c.ra.deg), float(c.dec.deg)
      pixels          = wcs.all_world2pix([[ra_deg, dec_deg]], 1)
      plt.text(pixels[0][0], pixels[0][1], galaxies[i], fontsize=14)
    if beams == True:
      c    = SkyCoord('22h07m30.0s', '-44d08m00.0s', frame='icrs')
      ra_deg,dec_deg  = float(c.ra.deg), float(c.dec.deg)
      pixels          = wcs.all_world2pix([[ra_deg, dec_deg]], 1)
      plt.text(pixels[0][0], pixels[0][1], 'To NGC 7232 Triplet', fontsize=14)
      plt.arrow(1000, 600, -900, -500, head_width=75, head_length=75, fc='black')
  if subfig3 == 2 or subfig3 == 5 or subfig3 == 8 or subfig3 == 11 or subfig3 == 14:
    ax1.set_yticklabels([])
  plt.subplots_adjust(wspace=0.3, hspace=0.15)


def observed_field_plot(fig_num, image1, image2, image3, colour, wcs, wcs2, beams, backgroud, labels):
  matplotlib.rcParams.update({'font.size': 14})
  rc('font',**{'family':'serif','serif':['Times']})
  rc('text', usetex=True)
  ax1 = fig_num.add_subplot(1, 1, 1, facecolor = 'w', projection=wcs)
  # ======== Axes Labels ======== #
  ax1.set_ylabel(r'Declination')
  ax1.set_xlabel(r'Right Ascension')
  # ======== Axes Tick Parameters ======== #
  ax1.tick_params(direction='in')
  lon = ax1.coords[0]
  lat = ax1.coords[1]
  lon.set_major_formatter('hh:mm')
  lat.set_major_formatter('dd:mm')
  lon.set_separator(('h', 'm'))
  lat.set_separator(('d', 'm'))
  if backgroud == True:
    # ======== Add backgroud optical image ======== #
    ax1.imshow(image1, origin='lower', aspect='auto', cmap='Greys', clim=(4000,13000))
    # ======== Beam Positions ======== #
    if beams == True:
      beam_ra  = ['22h00m49.120s', '22h00m38.043s', '22h03m08.923s', '21h58m22.194s', '22h03m15.423s', '21h58m11.370s', '22h00m40.680s']
      beam_dec = ['-42d59m20.28s', '-43d53m19.67s', '-43d27m54.60s', '-42d32m16.21s', '-42d33m53.75s', '-43d26m15.67s', '-43d09m50.10s']
      for i in range(len(beam_ra)):
        c    = SkyCoord(beam_ra[i], beam_dec[i], frame='icrs')
        print (c.ra, c.dec)
        if i == 0 or i == 1:
          beam_colour = 'green'
          line_type   = '--'
          beam_size   = 0.5 * u.degree
        elif i == 6:
          beam_colour = 'peru'
          line_type   = '-'
          beam_size   = 0.4 * u.degree
        else:
          beam_colour = 'magenta'
          line_type   = ':'
          beam_size   = 0.5 * u.degree
        beam = SphericalCircle((c.ra, c.dec), beam_size, edgecolor=beam_colour, ls=line_type, facecolor='none', transform=ax1.get_transform('icrs'))
        # ======== Add Beams to image ======== #
        ax1.add_patch(beam)
  ax1.set_autoscale_on(False)
  infile   = '/Users/tflowers/NGC7162A/HIPASS/HIPASS_142.mom0.fits'
  f1        = pyfits.open(infile)
  pks_data = f1[0].data
  pks_hdr  = f1[0].header
  pks_hdr.remove('CTYPE3')
  pks_hdr.remove('CRVAL3')
  pks_hdr.remove('CDELT3')
  pks_hdr.remove('CRPIX3')
  pks_hdr.update(NAXIS=2)
  pks_wcs  = WCS(pks_hdr)
  pks_data = pks_data.reshape(int(pks_hdr['NAXIS2']), int(pks_hdr['NAXIS1']))
  # ======== Add HIPASS contours at 2.25 Jy/beam level ======== #
  lvls = [2.25] #[12.66]
  ax1.contour(pks_data, colors='purple', alpha=1, linewidths=0.75, levels=lvls, transform=ax1.get_transform(pks_wcs))
  # ======== Add ASKAP contours at 0.02 Jy/beam level ======== #
  lvls = [0.02]#lvls*max_flux
  ax1.contour(image2, colors=colour, alpha=1, linewidths=0.75, levels=lvls, transform=ax1.get_transform(wcs2))
  if labels == True:
    # ======== Replace with your galaxy names ======== #
    galaxies     = ['NGC 7162', 'NGC 7162A', 'ESO 288-G025', 'ESO 288-G033', 'AM 2159-434', 'J220338-431131', 'NGC 7166']
    # ======== List positions for galaxy names to be displayed (use trial and error to get positions you like ======== #
    # ======== Two sets because I used different image sizes with and without displaying beams ======== #
    if beams == True:
      gal_ra       = ['21h59m20.0s', '22h00m30.0s', '21h59m00.0s', '22h05m45.0s', '22h05m55.0s', '22h07m30.0s', '22h00m30.0s']
      gal_dec      = ['-43d20m00.0s', '-43d05m00.0s', '-43d55m00.0s', '-43d21m00.0s', '-43d34m00.0s', '-43d08m00.0s', '-43d31m00.0s']
    if beams == False:
      gal_ra       = ['21h59m20.0s', '22h00m30.0s', '22h01m00.0s', '22h03m30.0s', '22h03m55.0s', '22h03m45.0s', '22h01m00.0s']
      gal_dec      = ['-43d20m00.0s', '-43d05m00.0s', '-43d54m00.0s', '-43d20m30.0s', '-43d31m00.0s', '-43d09m00.0s', '-43d28m00.0s']
    for i in range(len(galaxies)):
      c    = SkyCoord(gal_ra[i], gal_dec[i], frame='icrs')
      ra_deg,dec_deg  = float(c.ra.deg), float(c.dec.deg)
      pixels          = wcs.all_world2pix([[ra_deg, dec_deg]], 1)
      plt.text(pixels[0][0], pixels[0][1], galaxies[i], fontsize=14)
    # ======== This adds the label and arrow pointing to centre of field and NGC7232 Triplet ======== #
    if beams == True:
      c    = SkyCoord('22h07m30.0s', '-44d08m00.0s', frame='icrs')
      ra_deg,dec_deg  = float(c.ra.deg), float(c.dec.deg)
      pixels          = wcs.all_world2pix([[ra_deg, dec_deg]], 1)
      plt.text(pixels[0][0], pixels[0][1], 'To NGC 7232 Triplet', fontsize=14)
      plt.arrow(1000, 600, -900, -500, head_width=75, head_length=75, fc='black')
  plt.subplots_adjust(wspace=0.3, hspace=0.15)


# fig_num: e.g. fig1 = plt.figure(1, figsize=(6,6), facecolor = '#007D7D')
# optical_file, askap_file, hipass_file: path and file name of optical (dss) image and askap and hipass moment 0 maps, set hipass_file = False if you don't have a hipass moment 0 map
# beams: True to plot beams, False if not
# beams_a, beams_b: arrays with the ra and dec centre positions of the footprint A and B beams to plot give in format [[RA1, DEC1], [RA2,DEC2], ...] using hms,dms format in strings [['22h00m49.120s','-42d59m20.28s'], ['22h00m38.043s','-43d53m19.67s'], ...]
# background: True to plot the optical image, False if not
# labels: True to add labels to plot, False if not

def selected_beams_plot(fig_num, optical_file, askap_file, hipass_file, beams, beams_a, beams_b, background, labels):
  matplotlib.rcParams.update({'font.size': 14})
  rc('font',**{'family':'serif','serif':['Times']})
  rc('text', usetex=True)
  #optical_file   = '/Users/tflowers/ASKAP/DSS2_Blue.fits'
  f1_dss         = pyfits.open(dss_file)
  dss_data, dss_hdr = f1_dss[0].data, f1_dss[0].header
  dss_wcs      = WCS(dss_hdr)
  position     = SkyCoord('22h00m50.0s', '-43d12m00s', frame='icrs')
  size         = u.Quantity((160, 150), u.arcmin)
  dss_cutout   = Cutout2D(dss_data, position, size, dss_wcs)
  dss_cutout_wcs = dss_cutout.wcs
  ax1 = fig_num.add_subplot(1, 1, 1, facecolor = 'w', projection=dss_cutout_wcs)
  # ======== Axes Labels ======== #
  ax1.set_ylabel(r'Declination')
  ax1.set_xlabel(r'Right Ascension')
  # ======== Axes Tick Parameters ======== #
  ax1.tick_params(direction='in')
  lon = ax1.coords[0]
  lat = ax1.coords[1]
  lon.set_major_formatter('hh:mm')
  lat.set_major_formatter('dd:mm')
  lon.set_separator(('h', 'm'))
  lat.set_separator(('d', 'm'))
  if background == True:
    # ======== Add backgroud optical image ======== #
    ax1.imshow(dss_cutout, origin='lower', aspect='auto', cmap='Greys', clim=(4000,13000))
    # ======== Beam Positions ======== #
    if beams == True:
      # ======== Footprint A ======== #
      for i in range(len(beams_a)):
        c    = SkyCoord(beams_a[i][0], beams_a[i][1], frame='icrs')
        print (c.ra, c.dec)
        beam_colour = 'green'
        line_type   = '--'
        beam_size   = 0.5 * u.degree
        beam = SphericalCircle((c.ra, c.dec), beam_size, edgecolor=beam_colour, ls=line_type, facecolor='none', transform=ax1.get_transform('icrs'))
        # ======== Add Beam to image ======== #
        ax1.add_patch(beam)
      # ======== Footprint B ======== #
      for i in range(len(beams_b)):
        c    = SkyCoord(beams_b[i][0], beams_b[i][1], frame='icrs')
        print (c.ra, c.dec)
        beam_colour = 'magenta'
        line_type   = ':'
        beam_size   = 0.5 * u.degree
        beam = SphericalCircle((c.ra, c.dec), beam_size, edgecolor=beam_colour, ls=line_type, facecolor='none', transform=ax1.get_transform('icrs'))
        # ======== Add Beam to image ======== #
        ax1.add_patch(beam)
  ax1.set_autoscale_on(False)
  if hipass_file != False:
    #hipass_file   = '/Users/tflowers/NGC7162A/HIPASS/HIPASS_142.mom0.fits'
    f1_hipass     = pyfits.open(hipass_file)
    pks_data = f1_hipass[0].data
    pks_hdr  = f1_hipass[0].header
    pks_hdr.remove('CTYPE3')
    pks_hdr.remove('CRVAL3')
    pks_hdr.remove('CDELT3')
    pks_hdr.remove('CRPIX3')
    pks_hdr.update(NAXIS=2)
    pks_wcs  = WCS(pks_hdr)
    pks_data = pks_data.reshape(int(pks_hdr['NAXIS2']), int(pks_hdr['NAXIS1']))
  # ======== Add HIPASS contours at 2.25 Jy/beam level ======== #
    lvls = [2.25] #[12.66]
    ax1.contour(pks_data, colors='purple', alpha=1, linewidths=0.75, levels=lvls, transform=ax1.get_transform(pks_wcs))
  # ======== Add ASKAP contours at 0.02 Jy/beam level ======== #
  #askap_file  = indir + 'askap.comb16.linmos.imsub_%i_mom0.fits' % spec_num[i]
  f1_askap    = pyfits.open(askap_file)
  askap_data, askap_hdr = f1_askap[0].data, f1_askap[0].header
  askap_hdr.remove('CTYPE4')
  askap_hdr.remove('CRVAL4')
  askap_hdr.remove('CDELT4')
  askap_hdr.remove('CRPIX4')
  askap_hdr.update(NAXIS=2)
  askap_wcs     = WCS(askap_hdr)
  ra_pix, ra_del, ra_val, len_ra     = askap_hdr['CRPIX1'], askap_hdr['CDELT1'], askap_hdr['CRVAL1'], int(askap_hdr['NAXIS1'])
  dec_pix, dec_del, dec_val, len_dec = askap_hdr['CRPIX2'], askap_hdr['CDELT2'], askap_hdr['CRVAL2'], int(askap_hdr['NAXIS2'])
  ra_low  = (ra_val - ra_pix*ra_del*60*60)
  dec_low = (dec_val + dec_pix*dec_del*60*60)
  ra   = []
  dec  = []
  for j in range(len_ra):
    ra.append((ra_low+j*ra_del*60*60))
  for j in range(len_dec):
    dec.append((dec_low-j*dec_del*60*60))
  askap_data = askap_data.reshape(len_dec, len_ra)
  lvls = [0.02]#lvls*max_flux
  ax1.contour(askap_data, colors='darkblue', alpha=1, linewidths=0.75, levels=lvls, transform=ax1.get_transform(askap_wcs))
  if labels == True:
    # ======== Replace with your galaxy names ======== #
    galaxies     = ['NGC 7162', 'NGC 7162A', 'ESO 288-G025', 'ESO 288-G033', 'AM 2159-434', 'J220338-431131', 'NGC 7166']
    # ======== List positions for galaxy names to be displayed (use trial and error to get positions you like ======== #
    gal_ra       = ['21h59m20.0s', '22h00m30.0s', '21h59m00.0s', '22h05m45.0s', '22h05m55.0s', '22h07m30.0s', '22h00m30.0s']
    gal_dec      = ['-43d20m00.0s', '-43d05m00.0s', '-43d55m00.0s', '-43d21m00.0s', '-43d34m00.0s', '-43d08m00.0s', '-43d31m00.0s']
    for i in range(len(galaxies)):
      c    = SkyCoord(gal_ra[i], gal_dec[i], frame='icrs')
      ra_deg,dec_deg  = float(c.ra.deg), float(c.dec.deg)
      pixels          = dss_cutout_wcs.all_world2pix([[ra_deg, dec_deg]], 1)
      plt.text(pixels[0][0], pixels[0][1], galaxies[i], fontsize=14)
    # ======== This adds the label and arrow pointing to centre of field and NGC7232 Triplet ======== #
    #if beams == True:
    #  c    = SkyCoord('22h07m30.0s', '-44d08m00.0s', frame='icrs')
    #  ra_deg,dec_deg  = float(c.ra.deg), float(c.dec.deg)
    #  pixels          = dss_cutout_wcs.all_world2pix([[ra_deg, dec_deg]], 1)
    #  plt.text(pixels[0][0], pixels[0][1], 'To NGC 7232 Triplet', fontsize=14)
    #  plt.arrow(1000, 600, -900, -500, head_width=75, head_length=75, fc='black')
  plt.subplots_adjust(wspace=0.3, hspace=0.15)


def full_field_plot2(fig_num, subfig1, subfig2, subfig3, image1, image2, colour, lsty, txtstr, lbl, beams):
  matplotlib.rcParams.update({'font.size': 14})
  rc('font',**{'family':'serif','serif':['Times']})
  rc('text', usetex=True)
  #if txtstr == 'a) NGC 7162':
  ax1 = aplpy.FITSFigure(image1, figure=fig_num, subplot=(subfig1, subfig2, subfig3))
  if subfig3 == 0:
    #ax1.show_beam(major=.0108,minor=.009,angle=78,fill=True,color='peru')
    ax1.axis_labels.set_ytext('Declination')
    ax1.axis_labels.set_xtext('Right Ascension')
    ax1.tick_labels.set_xformat('hh:mm:ss')
    ax1.tick_labels.set_yformat('dd:mm')
    ax1.ticks.show()
    ax1.ticks.set_xspacing(0.05)
    position     = SkyCoord('22h00m50.0s', '-43d12m00s', frame='icrs')
    size         = u.Quantity((160, 150), u.arcmin)
    width, height = 160./60., 160./60.
    ax1.recenter(position.ra, position.dec, width=width, height=height)
    #ax1.imshow(image1, origin='lower', aspect='auto', cmap='Greys', clim=(4000,13000))
    if beams == True:
      beam_ra  = ['22h00m49.120s', '22h00m38.043s', '22h03m08.923s', '21h58m22.194s', '22h03m15.423s', '21h58m11.370s']
      beam_dec = ['-42d59m20.28s', '-43d53m19.67s', '-43d27m54.60s', '-42d32m16.21s', '-42d33m53.75s', '-43d26m15.67s']
      for i in range(len(beam_ra)):
        c    = SkyCoord(beam_ra[i], beam_dec[i], frame='icrs')
        print (c.ra, c.dec)
        if i == 0 or i == 1:
          beam_colour = 'green'
        else:
          beam_colour = 'magenta'
        #beam = SphericalCircle((c.ra, c.dec), 0.5 * u.degree, edgecolor=beam_colour, facecolor='none', transform=ax1.get_transform('icrs'))
        #ax1.add_patch(beam)
        ax1.show_circles(c.ra.deg, c.dec.deg, radius = 0.5, edgecolor=beam_colour, facecolor='none')
  #ax1.set_autoscale_on(False)
  lvls = [0.02]#lvls*max_flux
  #ax1.contour(image2, colors=colour, alpha=1, linewidths=0.75, levels=lvls, transform=ax1.get_transform(wcs2))
  ax1.show_contour(image2, colors="blue", levels=lvls, slices=(0,1))
  if txtstr == 'f) J220338-431131':
    galaxies     = ['NGC 7162', 'NGC 7162A', 'ESO 288-G025', 'ESO 288-G033', 'AM 2159-434', 'J220338-431131', 'NGC 7166']
    if beams == True:
      gal_ra       = ['21h59m20.0s', '22h00m30.0s', '21h59m00.0s', '22h05m45.0s', '22h05m55.0s', '22h07m30.0s', '22h00m30.0s']
      gal_dec      = ['-43d20m00.0s', '-43d05m00.0s', '-43d55m00.0s', '-43d21m00.0s', '-43d34m00.0s', '-43d08m00.0s', '-43d31m00.0s']
    if beams == False:
      gal_ra       = ['21h59m20.0s', '22h00m30.0s', '22h01m00.0s', '22h03m30.0s', '22h03m55.0s', '22h03m45.0s', '22h01m00.0s']
      gal_dec      = ['-43d20m00.0s', '-43d05m00.0s', '-43d54m00.0s', '-43d20m30.0s', '-43d31m00.0s', '-43d09m00.0s', '-43d28m00.0s']
    for i in range(len(galaxies)):
      c    = SkyCoord(gal_ra[i], gal_dec[i], frame='icrs')
      ra_deg,dec_deg  = float(c.ra.deg), float(c.dec.deg)
      #pixels          = wcs.all_world2pix([[ra_deg, dec_deg]], 1)
      #plt.text(pixels[0][0], pixels[0][1], galaxies[i], fontsize=14)
      ax1.add_label(c.ra.deg, c.dec.deg, galaxies[i], relative=False)
    if beams == True:
      c    = SkyCoord('22h07m30.0s', '-44d08m00.0s', frame='icrs')
      ra_deg,dec_deg  = float(c.ra.deg), float(c.dec.deg)
      #pixels          = wcs.all_world2pix([[ra_deg, dec_deg]], 1)
      #plt.text(pixels[0][0], pixels[0][1], 'To NGC 7232 Triplet', fontsize=14)
      ax1.add_label(c.ra.deg, c.dec.deg, 'To NGC 7232 Triplet', relative=False)
      plt.arrow(1000, 600, -900, -500, head_width=75, head_length=75, fc='black')
      c    = SkyCoord('22h03m30.0s', '-44d10m00.0s', frame='icrs')
      ra_deg,dec_deg  = float(c.ra.deg), float(c.dec.deg)
      ax1.show_arrows(ra_deg, dec_deg, -1, -0.9)
  if subfig3 == 2 or subfig3 == 5 or subfig3 == 8 or subfig3 == 11 or subfig3 == 14:
    ax1.set_yticklabels([])
  plt.subplots_adjust(wspace=0.3, hspace=0.15)



def ngc7232_field_plot(fig_num, subfig1, subfig2, subfig3, image1, beam_a_ra, beam_a_dec, beam_b_ra, beam_b_dec, wcs):
  matplotlib.rcParams.update({'font.size': 14})
  rc('font',**{'family':'serif','serif':['Times']})
  rc('text', usetex=True)
  ax1 = fig_num.add_subplot(subfig1, subfig2, subfig3, facecolor = 'w', projection=wcs)
  # ======== Axes Labels ======== #
  ax1.set_ylabel(r'Declination')
  ax1.set_xlabel(r'Right Ascension')
  # ======== Axes Tick Parameters ======== #
  ax1.tick_params(direction='in')
  lon = ax1.coords[0]
  lat = ax1.coords[1]
  lon.set_major_formatter('hh:mm')
  lat.set_major_formatter('dd:mm')
  lon.set_separator(('h', 'm'))
  lat.set_separator(('d', 'm'))
  # ======== Add backgroud optical image ======== #
  ax1.imshow(image1, origin='lower', aspect='auto', cmap='Greys', clim=(4000,20000))
  # ======== Beam Positions ======== #
  for i in range(len(beam_a_ra)):
    c    = SkyCoord(beam_a_ra[i], beam_a_dec[i], frame='icrs')
    beam_colour = 'green'
    line_type   = '--'
    beam_size   = 0.5 * u.degree
    beam = SphericalCircle((c.ra, c.dec), beam_size, edgecolor=beam_colour, ls=line_type, facecolor='none', transform=ax1.get_transform('icrs'))
    # ======== Add Beams to image ======== #
    ax1.add_patch(beam)
  for i in range(len(beam_b_ra)):
    c    = SkyCoord(beam_b_ra[i], beam_b_dec[i], frame='icrs')
    beam_colour = 'magenta'
    line_type   = ':'
    beam_size   = 0.5 * u.degree
    beam = SphericalCircle((c.ra, c.dec), beam_size, edgecolor=beam_colour, ls=line_type, facecolor='none', transform=ax1.get_transform('icrs'))
    # ======== Add Beams to image ======== #
    ax1.add_patch(beam)
  ax1.set_autoscale_on(False)
  plt.subplots_adjust(wspace=0.3, hspace=0.15)

def vel_model_plot(fig_num, subfig1, subfig2, subfig3, image1, colour, lsty, txtstr, lbl):
  matplotlib.rcParams.update({'font.size': 20})
  rc('font',**{'family':'serif','serif':['Times']})
  rc('text', usetex=True)
  if subfig3 == 1:
    subfig_param = [0.05,0.55,0.3,0.45]
  if subfig3 == 2:
    subfig_param = [0.305,0.55,0.3,0.45]
  if subfig3 == 3:
    subfig_param = [0.65,0.55,0.3,0.45]
  if subfig3 == 4:
    subfig_param = [0.05,0.05,0.3,0.45]
  if subfig3 == 5:
    subfig_param = [0.305,0.05,0.3,0.45]
  if subfig3 == 6:
    subfig_param = [0.65,0.05,0.3,0.45]
  ax1 = aplpy.FITSFigure(image1, figure=fig_num, subplot=subfig_param) #(subfig1, subfig2, subfig3))
  if subfig3 == 1 or subfig3 == 4:# or subfig3 == 7 or subfig3 == 10 or subfig3 == 13:
    ax1.axis_labels.set_ytext('Declination')
    ax1.show_beam(major=.0108,minor=.009,angle=78,fill=True,color='peru')
  else:
    ax1.axis_labels.hide_y()
  if subfig3 == 4 or subfig3 == 5 or subfig3 == 6:
    ax1.axis_labels.set_xtext('Right Ascension')
  else:
    ax1.axis_labels.hide_x()
  ax1.tick_labels.set_xformat('hh:mm:ss')
  ax1.tick_labels.set_yformat('dd:mm')
  ax1.ticks.show()
  ax1.ticks.set_length(5)
  ax1.ticks.set_color('black')
  ax1.ticks.set_xspacing(0.05)
  ax1.ticks.set_minor_frequency(1)
  #ax1.show_colorscale(vmin=4000,vmax=13000,cmap='Greys')
  if subfig3 == 1 or subfig3 == 2:
    ax1.show_colorscale(vmin=2160,vmax=2440,cmap='viridis')
    #if subfig3 == 2:
    ax1.add_colorbar()
    ax1.colorbar.set_location('right')
    if subfig3 == 1:
      ax1.colorbar.set_pad(-0.10)
    if subfig3 == 2:
      ax1.colorbar.set_pad(-0.15)
    ax1.colorbar.set_axis_label_text('Velocity [km/s]')
  if subfig3 == 3:
    ax1.show_colorscale(vmin=-10,vmax=40,cmap='viridis')
    ax1.add_colorbar()
    ax1.colorbar.set_location('right')
    ax1.colorbar.set_pad(-0.15)
    ax1.colorbar.set_axis_label_text('Velocity [km/s]')
  if subfig3 == 4 or subfig3 == 5:
    ax1.show_colorscale(vmin=2210,vmax=2330,cmap='viridis')
    #if subfig3 == 5:
    ax1.add_colorbar()
    ax1.colorbar.set_location('right')
    if subfig3 == 4:
      ax1.colorbar.set_pad(-0.10)
    if subfig3 == 5:
      ax1.colorbar.set_pad(-0.15)
    ax1.colorbar.set_axis_label_text('Velocity [km/s]')
  if subfig3 == 6:
    ax1.show_colorscale(vmin=-10,vmax=20,cmap='viridis')
    ax1.add_colorbar()
    ax1.colorbar.set_location('right')
    ax1.colorbar.set_pad(-0.15)
    ax1.colorbar.set_axis_label_text('Velocity [km/s]')
  if subfig3 == 2 or subfig3 == 3 or subfig3 == 5 or subfig3 == 6:
    ax1.tick_labels.hide_y()
  if txtstr == 'a) NGC 7162':
    position     = SkyCoord('21h59m39.1s', '-43d18m21s', frame='icrs')
    size         = u.Quantity((360, 360), u.arcsec)
    width, height = 420./60./60., 420./60./60.
  if txtstr == 'b) NGC 7162A':
    position     = SkyCoord('22h00m35.7s', '-43d08m30s', frame='icrs')
    size         = u.Quantity((360, 360), u.arcsec)
    width, height = 420./60./60., 420./60./60.
  if txtstr == 'c) ESO 288-G033':
    position     = SkyCoord('22h02m06.6s', '-43d16m07s', frame='icrs')
    size         = u.Quantity((240, 240), u.arcsec)
    width, height = 300./60./60., 300./60./60.
  ax1.recenter(position.ra, position.dec, width=width, height=height)
  #ax1.imshow(image1, origin='lower', aspect='auto', cmap='Greys', clim=(4000,13000))
  #ax1.set_autoscale_on(False)
  #if subfig3 == 1 or subfig3 == 4 or subfig3 == 7 or subfig3 == 10 or subfig3 == 13:
  #plt.text(1, 5, txtstr, fontsize=16)
  #min_flux = 1*10**19*(39.01*34.06)/(1.36*21*21*1.823*1000*(10**18))
  #lvls = np.array([1, 10, 30, 50, 70, 100, 130, 160])
  #lvls = lvls*min_flux
  if subfig3 == 1:
    ax1.add_label(0.76, 0.05, txtstr, relative=True)
  if subfig3 == 4:
    ax1.add_label(0.73, 0.05, txtstr, relative=True)
  #if subfig3 == 3:
  #  ax1.add_label(0.72, 0.05, txtstr, relative=True)
  #ax1.show_contour(image2, colors="blue", levels=lvls, slices=(0,1))
  #plt.subplots_adjust(wspace=0.20, hspace=0.15)

def XUV_plot(fig_num, subfig3, image1):
  matplotlib.rcParams.update({'font.size': 13})
  rc('font',**{'family':'serif','serif':['Times']})
  rc('text', usetex=True)
  if subfig3 == 1:
    subplot_params = [0.05,0.50,0.45,0.40]
  if subfig3 == 2:
    subplot_params = [0.5,0.50,0.45,0.40]
  if subfig3 == 3:
    subplot_params = [0.05,0.05,0.45,0.40]
  if subfig3 == 4:
    subplot_params = [0.5,0.05,0.45,0.40]
  ax1 = aplpy.FITSFigure(image1, figure=fig8, subplot=subplot_params)
  if subfig3 == 1 or subfig3 == 3:
    ax1.show_colorscale(vmin=0,vmax=0.01,cmap='Greys')
    ax1.axis_labels.set_ytext('Declination')
  if subfig3 == 2 or subfig3 == 4:
    ax1.show_colorscale(vmin=0,vmax=300,cmap='Greys')
    ax1.axis_labels.hide_y()
    ax1.tick_labels.hide_y()
  if subfig3 == 3 or subfig3 == 4:
    ax1.axis_labels.set_xtext('Right Ascension')
  if subfig3 == 1 or subfig3 == 2:
    ax1.axis_labels.hide_x()
  ax1.tick_labels.set_xformat('hh:mm:ss')
  ax1.tick_labels.set_yformat('dd:mm')
  ax1.ticks.show()
  ax1.ticks.set_length(5)
  ax1.ticks.set_color('black')
  #ax1.ticks.set_xspacing(0.05)
  ax1.ticks.set_minor_frequency(1)
  if subfig3 == 1 or subfig3 == 2:
    position      = SkyCoord('21h59m39.1s', '-43d18m21s', frame='icrs')
  if subfig3 == 3 or subfig3 == 4:
    position      = SkyCoord('22h00m35.7s', '-43d08m30s', frame='icrs')
  width, height = 240./60./60., 240./60./60.
  ax1.recenter(position.ra, position.dec, width=width, height=height)
  if subfig3 == 1:
    ax1.add_label(0.23, 0.05, r'a) $GALEX$ FUV', relative=True)
  if subfig3 == 2:
    ax1.add_label(0.25, 0.05, r'b) VHS $K-$band', relative=True)
  if subfig3 == 3:
    ax1.add_label(0.23, 0.05, r'c) $GALEX$ FUV', relative=True)
  if subfig3 == 4:
    ax1.add_label(0.25, 0.05, r'd) VHS $K-$band', relative=True)

H0 = cosmo.H(0).value
rho_crit = cosmo.critical_density(0).value*100**3/1000
#print H0
#print rho_crit


PI      = math.pi
jy_w    = pow(10,-26) # Jy to W/(m^2*Hz)
mpc_m   = 3.086*pow(10,22) # Mpc to metre
sol_lum = 3.828*pow(10,26) # Solar luminosity
w1_freq = 8.33*pow(10,13) # W1 band frequency
