# Libraries

import math
import cmath
import sys
import numpy as np
import datetime
import matplotlib
import matplotlib as mpl
import matplotlib.cm as cm
#mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid
import matplotlib.patches as mpatches
import astropy.io.fits as pyfits
import shutil
import emcee
import os.path
from matplotlib import rc
from matplotlib.pyplot import gca
from matplotlib.collections import PatchCollection
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
from astropy.nddata.utils import Cutout2D
from random import shuffle as sfl
from astropy.convolution import convolve, Gaussian1DKernel
from astropy.visualization.wcsaxes import SphericalCircle
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
import aplpy
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FuncFormatter
from scipy.stats import binned_statistic

import sympy as sp
from sympy.solvers import  solve
from sympy import Symbol

import sqlite3
from sqlite3 import OperationalError

#from functions_plot import field_mom_plot, field_pv_plot

#from envelope_tracing import open_fits_file, env_trace

from functions_calculations import *

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
#BEAM = beam_factor(0.01062*60*60, 0.009033*60*60, 5)
#print 'ASKAP Beam Factor: %.2f' % BEAM

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

def parameter_hist_plot(fig_num, sub1, sub2, sub3, param_fit, bins, param_str, col, htype, lbl):
    matplotlib.rcParams.update({'font.size': 11})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    ax1.set_xlabel(param_str)
    #if sub3 == 1 or sub3 == 5 or sub3 == 9 or sub3 ==13:
    if sub3 == 1 or sub3 == 5 or sub3 == 9:
    #if sub3 % sub2 == 0:
      ax1.set_ylabel(r'Number')
    #for i in range(len(param_fit)):
      #param_fit[i] = [x for x in param_fit[i] if ~np.isnan(x)]
    #plt.axvline(param_true, linewidth=0.75, linestyle = ':', color = 'black', label = 'True')
    par_mean = np.nanmedian(param_fit)
    par_std  = np.nanstd(param_fit)
    #plt.axvline(par_mean, linewidth=1, linestyle = '--', color = col)#, label='Mean')
    if lbl == 'Field':
      y = 20
    elif lbl == 'Cluster':
      y = 19
    elif lbl == 'Infall':
      y = 18
    elif lbl == 'Non-det':
      y = 17
    else:
      y = 17
    #plt.errorbar(par_mean, y, xerr=par_std, color=col, marker='o')
    #if col == 'darkblue':
      #plt.axvline(1, linewidth=1, linestyle = ':', color = 'black')#, label='Mean')
    #ax1.axvspan(par_mean-par_std, par_mean+par_std, alpha=0.1, edgecolor='none', zorder=-1, facecolor=col)
    #bins = 10
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
    n,bins,patches = plt.hist(param_fit, bins=bins, color=col, linestyle='-', cumulative=False, density=False, stacked=False, histtype=htype, label=lbl)
    #hatches = hsty
    #for patch_set, hatch in zip(patches, hatches):
      #for patch in patch_set.patches:
        #patch.set_hatch(hatch)
    y_vals = ax1.get_yticks()
    #if sub3 == 1:
      #ax1.legend(fontsize = 11)#, loc='lower right') #loc='upper right', 
    plt.subplots_adjust(wspace=0.25, hspace=0.25)


def hist_nondet_plot(fig_num, sub1, sub2, sub3, param_fit, bins, param_str, 
                     col, htype, lsty, lbl, statistic, ymean, vline, legend_loc):
    matplotlib.rcParams.update({'font.size': 11})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    #if sub3 > 6:
      #ax1.set_xlabel(param_str)
    #if sub3 == 1 or sub3 == 4 or sub3 == 7:
      #ax1.set_ylabel(r'Number')
    if sub3 > 10:
      ax1.set_xlabel(param_str)
    if sub3 == 1 or sub3 == 6 or sub3 == 11:
      ax1.set_ylabel(r'Number')
    order_array = [4, 3, 2, 1, 1, 1]
    ax1.set_yscale('log')
    ax1.set_ylim(1, ymean[4])
    #if sub2 == 2:
      #if param_str == r'$d_{\mathrm{HI}}/d_{\mathrm{opt}}$':
        #ax1.set_ylim(0,60)
        ##ymean = [15, 13, 14]
        #ymean = [55, 45, 50]
        ##ax1.set_yticks((0,4,8,12,16,20,24,28))
      #elif param_str == r'DEF$_{\mathrm{HI}}$':
        #ax1.set_ylim(0,25)
        ##ymean = [19, 17, 18]
        #ymean = [24, 20, 22]
      #else:
        #ymean = [35, 27, 31]
    #else:
      #if param_str == r'$d_{\mathrm{HI}}/d_{\mathrm{opt}}$':
        #
        #ymean = [7.5, 6.5, 7]
        #ax1.set_yticks((0,2,4,6,8))
      #elif param_str == r'DEF$_{\mathrm{HI}}$':
        #ax1.set_ylim(0,10)
        #ymean = [9, 7, 8]
        #ax1.set_yticks((0,2,4,6,8,10))
      #else:
        #ymean = [9, 7, 8]
    plt.axvline(vline[0], linewidth=1, linestyle = '--', color = 'black')
    if vline[1]:
      plt.axvline(vline[0] - 0.3, linewidth=1, linestyle = ':', color = 'black')
      plt.axvline(vline[0] + 0.3, linewidth=1, linestyle = ':', color = 'black')
    #mark = ['o', 's', 'd', 'v', 'v', 'v']
    #mark = ['s', 'd', 'o', 'v', 'v', 'v']
    mark = ['s', 'd', 'o', r'$\leftarrow$', r'$\leftarrow$', r'$\leftarrow$']
    for i in range(6):
      if len(param_fit[i]) > 0:
        if col[i] == 'darkblue':
          if param_str == r'$d_{\mathrm{HI}}/d_{\mathrm{opt}}$':
            xval = 1
            plt.axvline(xval, linewidth=1, linestyle = ':', color = 'black')
          if param_str == r'DEF$_{\mathrm{HI}}$':
            xval = 0.4
            plt.axvline(xval, linewidth=1, linestyle = ':', color = 'black')
        par_count, bin_edges, _ = binned_statistic(param_fit[i], param_fit[i], 'count', bins)
        bin_width  = np.abs(bin_edges[1] - bin_edges[0])
        xbins = bin_edges[:-1] + bin_width/2.
        if lbl[i] == 'Non-det':
          ax1.plot(xbins, par_count, color=col[i], linewidth=0.75, linestyle='--', zorder=order_array[i])
          par_count[par_count == 0] = np.nan
          ax1.plot(xbins, par_count, marker=mark[i], linestyle='none', 
                   markersize=8, color=col[i], zorder=order_array[i])
        else:
          ax1.plot(xbins, par_count, color=col[i], linewidth=0.75, zorder=order_array[i])
          par_count[par_count == 0] = np.nan
          ax1.plot(xbins, par_count, marker=mark[i], linestyle='none', 
                   markersize=4, color=col[i], label=lbl[i], zorder=order_array[i])
    for i in range(3):
      if statistic == 'median':
        par_mean = np.median(param_fit[i])
        #par_std  = np.std(param_fit[i])
        par_25th = np.abs(par_mean - np.percentile(param_fit[i], 25))
        par_75th = np.abs(par_mean - np.percentile(param_fit[i], 75))
        print(np.round(par_mean,1), np.round(par_25th,1), np.round(par_75th,1))
      if statistic == 'mean':
        par_mean = np.mean(param_fit[i])
        par_25th = np.std(param_fit[i])
        par_75th = np.std(param_fit[i])
      plt.errorbar(par_mean, ymean[i], xerr=np.array([(par_25th,par_75th)]).T, color=col[i], marker=mark[i])
    for i in range(3):
      par_mean = np.median(np.concatenate((param_fit[i], param_fit[i+3])))
      par_25th = np.abs(par_mean - np.percentile(np.concatenate((param_fit[i], param_fit[i+3])), 25))
      par_75th = np.abs(par_mean - np.percentile(np.concatenate((param_fit[i], param_fit[i+3])), 75))
      #print(np.round(par_mean,1), np.round(par_25th,1), np.round(par_75th,1))
      #par_std  = np.std(np.concatenate((param_fit[1], param_fit[3])))
      plt.plot(par_mean, ymean[i], color=col[i], marker='x')
      count_hi   = len(param_fit[i])
      count_nohi = len(param_fit[i+3])
      plt.text(ymean[3], ymean[i]-5, '%i (%i)' % (count_hi, count_nohi), color=col[i], fontsize=9)
      #, transform=ax1.transAxes
    if sub3 == 1:
      plt.text(0.8, 0.92, 'All', transform=ax1.transAxes, fontsize=10)
    if sub3 == 6:
      plt.text(0.3, 0.92, r'$\log(M_*/[\rm{M}_{\odot}])<9$', transform=ax1.transAxes, fontsize=10)
    if sub3 == 11:
      plt.text(0.3, 0.92, r'$\log(M_*/[\rm{M}_{\odot}])\geq9$', transform=ax1.transAxes, fontsize=10)
    #par_mean = np.median(np.concatenate((param_fit[1], param_fit[4])))
    ##par_std  = np.std(np.concatenate((param_fit[2], param_fit[4])))
    #plt.plot(par_mean, ymean[1], color=col[1], marker='x')
    #par_mean = np.median(np.concatenate((param_fit[2], param_fit[5])))
    ##par_std  = np.std(np.concatenate((param_fit[0], param_fit[5])))
    #plt.plot(par_mean, ymean[2], color=col[2], marker='x')
    
    y_vals = ax1.get_yticks()
    if legend_loc[0]:
      ax1.legend(fontsize = 9, loc=legend_loc[1])
    plt.subplots_adjust(wspace=0.2, hspace=0.2)


def hist_nondet_plot_portrait(fig_num, sub1, sub2, sub3, param_fit, bins, param_str, 
                              col, htype, lsty, lbl, statistic, ymean, vline, legend_loc):
    matplotlib.rcParams.update({'font.size': 11})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    #if sub3 > 6:
      #ax1.set_xlabel(param_str)
    #if sub3 == 1 or sub3 == 4 or sub3 == 7:
      #ax1.set_ylabel(r'Number')
    #if sub3 > 10:
    ax1.set_xlabel(param_str)
    if sub3 == 1 or sub3 == 4 or sub3 == 7 or sub3 == 10 or sub3 == 13:
      ax1.set_ylabel(r'Number $+1$')
    order_array = [4, 3, 2, 1, 1, 1]
    ax1.set_yscale('log')
    ax1.set_ylim(1, ymean[4])
    
    plt.axvline(vline[0], linewidth=1, linestyle = '--', color = 'black')
    if vline[1]:
      plt.axvline(vline[0] - 0.3, linewidth=1, linestyle = ':', color = 'black')
      plt.axvline(vline[0] + 0.3, linewidth=1, linestyle = ':', color = 'black')
    #mark = ['o', 's', 'd', 'v', 'v', 'v']
    #mark = ['s', 'd', 'o', 'v', 'v', 'v']
    mark = ['s', 'd', 'o', r'$\leftarrow$', r'$\leftarrow$', r'$\leftarrow$']
    for i in range(6):
      if len(param_fit[i]) > 0:
        if col[i] == 'darkblue':
          if param_str == r'$d_{\mathrm{HI}}/d_{\mathrm{opt}}$':
            xval = 1
            plt.axvline(xval, linewidth=1, linestyle = ':', color = 'black')
          if param_str == r'DEF$_{\mathrm{HI}}$':
            xval = 0.4
            plt.axvline(xval, linewidth=1, linestyle = ':', color = 'black')
        par_count, bin_edges, _ = binned_statistic(param_fit[i], param_fit[i], 'count', bins)
        bin_width  = np.abs(bin_edges[1] - bin_edges[0])
        xbins = bin_edges[:-1] + bin_width/2.
        if lbl[i] == 'Non-det':
          ax1.plot(xbins, par_count+1, color=col[i], linewidth=0.75, linestyle='--', zorder=order_array[i])
          par_count[par_count == 0] = np.nan
          ax1.plot(xbins, par_count+1, marker=mark[i], linestyle='none', 
                   markersize=8, color=col[i], zorder=order_array[i])
        else:
          ax1.plot(xbins, par_count+1, color=col[i], linewidth=0.75, zorder=order_array[i])
          par_count[par_count == 0] = np.nan
          ax1.plot(xbins, par_count+1, marker=mark[i], linestyle='none', 
                   markersize=4, color=col[i], label=lbl[i], zorder=order_array[i])
    for i in range(3):
      if statistic == 'median':
        par_mean = np.median(param_fit[i])
        par_std  = np.std(param_fit[i])
        par_25th = np.abs(par_mean - np.percentile(param_fit[i], 25))
        par_75th = np.abs(par_mean - np.percentile(param_fit[i], 75))
        print(np.round(par_mean,1), np.round(par_25th,1), np.round(par_75th,1))
        #print(np.round(par_mean,1), np.round(1.25*par_std/np.sqrt(len(param_fit[i])),1), len(param_fit[i]))
      if statistic == 'mean':
        par_mean = np.mean(param_fit[i])
        par_25th = np.std(param_fit[i])
        par_75th = np.std(param_fit[i])
      plt.errorbar(par_mean, ymean[i], xerr=np.array([(par_25th,par_75th)]).T, color=col[i], marker=mark[i])
    for i in range(3):
      par_mean = np.median(np.concatenate((param_fit[i], param_fit[i+3])))
      par_25th = np.abs(par_mean - np.percentile(np.concatenate((param_fit[i], param_fit[i+3])), 25))
      par_75th = np.abs(par_mean - np.percentile(np.concatenate((param_fit[i], param_fit[i+3])), 75))
      #print(np.round(par_mean,1), np.round(par_25th,1), np.round(par_75th,1))
      #par_std  = np.std(np.concatenate((param_fit[1], param_fit[3])))
      plt.plot(par_mean, ymean[i], color=col[i], marker='x')
      count_hi   = len(param_fit[i])
      count_nohi = len(param_fit[i+3])
      plt.text(ymean[3], ymean[i]-5, '%i (%i)' % (count_hi, count_nohi), color=col[i], fontsize=9)
      #, transform=ax1.transAxes
    if sub3 == 1:
      plt.text(0.45, 1.1, 'All', transform=ax1.transAxes, fontsize=10)
    if sub3 == 2:
      plt.text(0.3, 1.1, r'$M_*<10^9\,\rm{M}_{\odot}$', transform=ax1.transAxes, fontsize=10)
    if sub3 == 3:
      plt.text(0.3, 1.1, r'$M_*\geq10^9\,\rm{M}_{\odot}$', transform=ax1.transAxes, fontsize=10)
    plt.text(0.05, 0.92, '%s' % legend_loc[2], transform=ax1.transAxes, fontsize=10)
    #par_mean = np.median(np.concatenate((param_fit[1], param_fit[4])))
    ##par_std  = np.std(np.concatenate((param_fit[2], param_fit[4])))
    #plt.plot(par_mean, ymean[1], color=col[1], marker='x')
    #par_mean = np.median(np.concatenate((param_fit[2], param_fit[5])))
    ##par_std  = np.std(np.concatenate((param_fit[0], param_fit[5])))
    #plt.plot(par_mean, ymean[2], color=col[2], marker='x')
    if legend_loc[0]:
      x, y = 0, 0
      p3 = ax1.scatter(x, y, color=col[0], marker=mark[0],  
                        facecolor=col[0], s=30, zorder=3)
      p4 = ax1.scatter(x, y, color=col[1], marker=mark[1], 
                        facecolor=col[1], s=30, zorder=3)
      p5 = ax1.scatter(x, y, color=col[2], marker=mark[2], 
                        facecolor=col[2], s=30, zorder=3)
      p6 = ax1.scatter(x, y, color=col[3], marker=mark[3], 
                        facecolor=col[3], s=30, zorder=3)
      p7 = ax1.scatter(x, y, color=col[4], marker=mark[4], 
                        facecolor=col[4], s=30, zorder=3)
      p8 = ax1.scatter(x, y, color=col[5], marker=mark[5], 
                        facecolor=col[5], s=30, zorder=3)
      #l1 = ax1.legend([p1, p2], [txtstr[0], txtstr[1]], fontsize = 11, loc = 'upper right', ncol=1)
      l2 = ax1.legend(handles=[p3, p4, p5, p6, p7, p8],
          labels=['', '', '', 'Cluster', 'Infall', 'Field'],
          loc=legend_loc[1], ncol=2, handlelength=1.5, fontsize = 9,
          handletextpad=0.1, columnspacing=-0.2, borderpad=0.1)
    y_vals = ax1.get_yticks()
    #if legend_loc[0]:
      #ax1.legend(fontsize = 9, loc=legend_loc[1])
    plt.subplots_adjust(wspace=0.25, hspace=0.4)



def hist_nondet_single_plot(fig_num, sub1, sub2, sub3, param_fit, bins, param_str, 
                     col, htype, lsty, lbl, statistic, ymean, vline, legend_loc):
    matplotlib.rcParams.update({'font.size': 11})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    ax1.set_xlabel(param_str)
    ax1.set_ylabel(r'Number')
    order_array = [4, 3, 2, 1, 1, 1]
    ax1.set_ylim(0, ymean[4])
    plt.axvline(vline[0], linewidth=1, linestyle = '--', color = 'black')
    if vline[1]:
      plt.axvline(vline[0] - 0.3, linewidth=1, linestyle = ':', color = 'black')
      plt.axvline(vline[0] + 0.3, linewidth=1, linestyle = ':', color = 'black')
    #mark = ['o', 's', 'd', 'v', 'v', 'v']
    mark = ['s', 'd', 'o', 'v', 'v', 'v']
    for i in range(6):
      if len(param_fit[i]) > 0:
        if col[i] == 'darkblue':
          if param_str == r'$d_{\mathrm{HI}}/d_{\mathrm{opt}}$':
            xval = 1
            plt.axvline(xval, linewidth=1, linestyle = ':', color = 'black')
          if param_str == r'DEF$_{\mathrm{HI}}$':
            xval = 0.4
            plt.axvline(xval, linewidth=1, linestyle = ':', color = 'black')
        par_count, bin_edges, _ = binned_statistic(param_fit[i], param_fit[i], 'count', bins)
        bin_width  = np.abs(bin_edges[1] - bin_edges[0])
        xbins = bin_edges[:-1] + bin_width/2.
        if lbl[i] == 'Non-det':
          ax1.plot(xbins, par_count, color=col[i], linewidth=0.75, linestyle='--', zorder=order_array[i])
          par_count[par_count == 0] = np.nan
          ax1.plot(xbins, par_count, marker=mark[i], linestyle='none', 
                   markersize=4, color=col[i], zorder=order_array[i])
        else:
          ax1.plot(xbins, par_count, color=col[i], linewidth=0.75, zorder=order_array[i])
          par_count[par_count == 0] = np.nan
          ax1.plot(xbins, par_count, marker=mark[i], linestyle='none', 
                   markersize=4, color=col[i], label=lbl[i], zorder=order_array[i])
    for i in range(3):
      if statistic == 'median':
        par_mean = np.median(param_fit[i])
        #par_std  = np.std(param_fit[i])
        par_25th = np.abs(par_mean - np.percentile(param_fit[i], 25))
        par_75th = np.abs(par_mean - np.percentile(param_fit[i], 75))
        print(np.round(par_mean,2), np.round(par_25th,2), np.round(par_75th,2))
      if statistic == 'mean':
        par_mean = np.mean(param_fit[i])
        par_25th = np.std(param_fit[i])
        par_75th = np.std(param_fit[i])
      plt.errorbar(par_mean, ymean[i], xerr=np.array([(par_25th,par_75th)]).T, color=col[i], marker=mark[i])
    for i in range(3):
      par_mean = np.median(np.concatenate((param_fit[i], param_fit[i+3])))
      par_25th = np.abs(par_mean - np.percentile(np.concatenate((param_fit[i], param_fit[i+3])), 25))
      par_75th = np.abs(par_mean - np.percentile(np.concatenate((param_fit[i], param_fit[i+3])), 75))
      #print(np.round(par_mean,2), np.round(par_25th,2), np.round(par_75th,2))
      #par_std  = np.std(np.concatenate((param_fit[1], param_fit[3])))
      plt.plot(par_mean, ymean[i], color=col[i], marker='x')
      count_hi   = len(param_fit[i])
      count_nohi = len(param_fit[i+3])
      plt.text(ymean[3], ymean[i]-0.5, '%i (%i)' % (count_hi, count_nohi), color=col[i], fontsize=9)
      #, transform=ax1.transAxes
    #if sub3 == 1:
      #plt.text(0.9, 0.9, 'All', transform=ax1.transAxes, fontsize=10)
    #if sub3 == 4:
      #plt.text(0.5, 0.9, r'$\log(M_*/[\rm{M}_{\odot}])<9$', transform=ax1.transAxes, fontsize=10)
    #if sub3 == 7:
      #plt.text(0.5, 0.9, r'$\log(M_*/[\rm{M}_{\odot}])\geq9$', transform=ax1.transAxes, fontsize=10)
    y_vals = ax1.get_yticks()
    if legend_loc[0]:
      ax1.legend(fontsize = 9, loc=legend_loc[1])
    plt.subplots_adjust(wspace=0.15, hspace=0.15)


# ================================ #
# ==== Scatter Error bar Plot ==== #
# ================================ #
def scat_ebar_plot(fig_num, sub1, sub2, sub3, x, y, xerr, yerr, txtstr, xlbl, ylbl):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    #if sub3 > 8:
    ax1.set_xlabel(xlbl)
    #if sub3 == 1 or sub3 == 5 or sub3 == 9 or sub3 ==13:
    ax1.set_ylabel(ylbl)
    plt.plot([0,np.nanmax(y)], [0,np.nanmax(y)], color='grey', linewidth=1, linestyle = '--', zorder=1)
    #plt.scatter(flux1, flux2, color=colour, s=20, label=lbl)
    #ax1.set_xlim(np.nanmin(x)-np.nanmax(yerr), np.nanmax(x)+np.nanmax(yerr))
    #ax1.set_ylim(np.nanmin(y)-np.nanmax(yerr), np.nanmax(y)+np.nanmax(yerr))
    plt.errorbar(x, y, xerr = xerr, yerr = yerr, color='darkblue', markersize=2, fmt='o', elinewidth=0.75, label = txtstr, zorder=2)
    #if sub3 == 10:
    #ax1.legend(loc='upper left', fontsize = 8.5)
    plt.subplots_adjust(wspace=0.25, hspace=0.2)

# ================================ #
# ========= Scatter Plot ========= #
# ================================ #
def scat_plot(fig_num, sub1, sub2, sub3, x, y, txtstr, xlbl, ylbl, col, marker, line):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    #if sub1 == 1 and sub2 == 1:
    ax1.set_xlabel(xlbl)
    #elif sub3 > 3:
      #ax1.set_xlabel(xlbl)
    #if sub3 == 1 or sub3 == 4:
    ax1.set_ylabel(ylbl)
    if line == True:
      lmin, lmax = np.min([np.nanmin(x),np.nanmin(y)]), np.max([np.nanmax(x),np.nanmax(y)])
      #print(lmin, lmax)
      ax1.plot([lmin,lmax], [lmin,lmax], color='grey', linewidth=1, linestyle = '--', zorder=1)
    if xlbl == r'$D_{\mathrm{L}}$ [Mpc]':
      ax1.axvline(60.8, linewidth=0.75, linestyle = '--', color = 'darkgrey')#, label = r'$V_{\mathrm{sys}}$')
    #plt.scatter(flux1, flux2, color=colour, s=20, label=lbl)
    ax1.scatter(x, y, color=col, marker=marker, s=5, label = txtstr, zorder=2)
    if sub3 == 4 and (xlbl != 'WALLABY' and ylbl != 'HIPASS'):
      ax1.legend(fontsize = 8.5) #upper right loc='lower left', 
    elif xlbl == 'WALLABY' and ylbl == 'HIPASS':
      ax1.legend(loc='lower right', fontsize = 8.5)
    if sub1 == 1 and sub2 == 1:
      ax1.legend(fontsize = 8.5) #loc='upper right', 
    plt.subplots_adjust(wspace=0.25, hspace=0.25)
    
# ================================ #
# ====== Scatter Mean Plot ======= #
# ================================ #
def scat_mean_plot(fig_num, sub1, sub2, sub3, x, y, xerr, yerr, txtstr, xlbl, ylbl, col, marker, do_mean):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    #if sub1 == 1 and sub2 == 1:
    ax1.set_xlabel(xlbl)
    ax1.set_ylabel(ylbl)
    if do_mean:
      if marker == '*':
        plt.errorbar(x, y, xerr = xerr, yerr = yerr, color=col, markersize=12, 
                      fmt=marker, elinewidth=0.75, label = txtstr, zorder=2)      
      else:
        plt.errorbar(x, y, xerr = xerr, yerr = yerr, color=col, markersize=4, 
                      fmt=marker, elinewidth=0.75, zorder=2) #label = txtstr,
    else:
      if marker == '.':
        if txtstr == 'Non-det':
          ax1.scatter(x, y, color=col, marker=marker, alpha=0.8, s=2, zorder=1)#, label = txtstr)
        else:
          ax1.scatter(x, y, color=col, marker=marker, alpha=0.5, s=2, zorder=1, label = txtstr)
      else:
        if marker == 'd':
          ax1.scatter(x, y, color=col, marker=marker, edgecolor=col, facecolor=col, alpha=0.95, s=20, zorder=1) #label = txtstr, 
        else:
          ax1.scatter(x, y, color=col, marker=marker, edgecolor=col, facecolor='none', s=7.5, zorder=1,label = txtstr) 
          #ax1.scatter(x, y, color=col, marker=marker, s=7.5, zorder=1,label = txtstr) 
    if xlbl == r'$D_{\mathrm{L}}$ [Mpc]' and ylbl == r'$\log(M_{\mathrm{HI}}/\mathrm{M}_{\odot})$':
      #ax1.axhline(8.1, linewidth=0.75, linestyle = '--', color = 'black')
      distance_array = np.arange(0, 220., 5.)
      mass_limit_100     = np.log10(5. * (1. / (2.94 * 10**-4)) * distance_array**2 * 0.0016 * np.sqrt(100. * 18500.))
      mass_limit_150     = np.log10(5. * (1. / (2.94 * 10**-4)) * distance_array**2 * 0.0016 * np.sqrt(170. * 18500.))
      mass_limit_200     = np.log10(5. * (1. / (2.94 * 10**-4)) * distance_array**2 * 0.0016 * np.sqrt(300. * 18500.))
      ax1.plot(distance_array, mass_limit_100, color='black', linewidth=1, linestyle = '--', zorder=1)
      ax1.plot(distance_array, mass_limit_150, color='black', linewidth=1, linestyle = ':', zorder=1)
      ax1.plot(distance_array, mass_limit_200, color='black', linewidth=1, linestyle = '-.', zorder=1)
    if xlbl == r'$\log(M_*/\mathrm{M}_{\odot})$' and ylbl == r'$\log(M_{\mathrm{HI}}/M_*)$':
      #w50_mean    = np.array([50., 75., 150., 150., 160., 220., 220., 360., 360.])
      #mstar       = np.array([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11]) + 0.25
      #mhi_limit1   = np.log10(5. * (1. / (2.94 * 10**-4)) * 60.**2 * 0.0016 * np.sqrt(w50_mean * 18500.))
      #mhi_limit2   = np.log10(5. * (1. / (2.94 * 10**-4)) * 130.**2 * 0.0016 * np.sqrt(w50_mean * 18500.))
      
      snr_limit    = 3.5
      chan_width   = 18500.
      chan_noise   = 0.002
      #vel_width    = []
      #for i in range(len(mstar_sdss_nohi)):
        #if mstar_sdss_nohi[i] < 10:
          #vel_width.append(200.)
        #else:
          #vel_width.append(300.)
      #vel_width        = np.array(vel_width)
      vel_width   = np.array([200., 200., 200., 200., 200., 200., 300., 300., 300.])
      opt_radius  = np.array([10., 10., 10., 15., 15., 15., 25., 40., 55.])
      #hi_radius   = np.array([26., 25., 34., 40., 45., 47., 84., 84., 84.])
      mstar       = np.arange(7.25,11.75,0.5)
      hi_radius   = 8.335 * mstar - 32.254
      #hi_radius[mstar > 10.5] = 100
      beam_area   = (math.pi * 30. * 30.) / (4. * np.log(2))
      mhi_limit1  = np.round(np.log10(snr_limit / (2.92 * 10**-4) * (1. + 0.0126)**(-0.5) * 
                             61.**2 * vel_width**(0.5) * chan_width**(0.5) * 
                             chan_noise * np.sqrt(math.pi * 0.59 * hi_radius**2 / beam_area)),1)
      f_threshold = 3.5
      f_smooth    = 1. / 7.75
      sigma_cube  = 2.
      mean_mstar  = np.array([9.14, 9.44, 9.74, 10.07, 10.34, 10.65, 10.95, 11.20])
      mean_gf     = np.array([-0.242, -0.459, -0.748, -0.869, -1.175, -1.231, -1.475, -1.589])
      func_interp = interp1d(mean_mstar, mean_gf, fill_value='extrapolate')
      interp_gf   = func_interp(mstar)
      vrot_tf     = (10**mstar + 1.4 * 10**(mstar + interp_gf))**(1 / 4)
      flux_limit2 = f_threshold * f_smooth * sigma_cube * 0.002 * vrot_tf
      mhi_limit2  = np.log10(2.35 * 10**5 * (1. + 0.0126)**(-2) * 61.**2 * flux_limit2)
      #mhi_limit[4:]   = np.log10(5. * (1. / (2.94 * 10**-4)) * 120.**2 * 0.0016 * np.sqrt(w50_mean[4:] * 18500.))
      mfrac_limit1 = mhi_limit1 - mstar
      mfrac_limit2 = mhi_limit2 - mstar
      ax1.plot(mstar, mfrac_limit1, color='black', linewidth=1, linestyle = '--', zorder=1)
      #ax1.plot(mstar, mfrac_limit2, color='purple', linewidth=1, linestyle = ':', zorder=1)
      #ax1.plot([7.5,11.], [8.1-7.5,8.1-11.0], color='black', linewidth=1, linestyle = '--', zorder=1)
    if xlbl == r'$\log(M_*/\mathrm{M}_{\odot})$' and ylbl == r'$\log$(SFR/[M$_{\odot}$\,yr$^{-1}$])':
      mstar    = np.array([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11]) + 0.25
      #sfr_ms   = 1.02 * (mstar - 10.) * 0.54
      sfr_ms   = mstar - 0.344 * (mstar - 9.) - 9.822
      sigma_ms = 0.088 * (mstar - 9.) + 0.188
      ax1.plot(mstar, sfr_ms, color='black', linewidth=1, linestyle = '--', zorder=2)
      ax1.plot(mstar, sfr_ms+0.3, color='black', linewidth=1, linestyle = ':', zorder=2)
      ax1.plot(mstar, sfr_ms-0.3, color='black', linewidth=1, linestyle = ':', zorder=2)
      #plt.fill_between(mstar, sfr_ms-sigma_ms, sfr_ms+sigma_ms, alpha=0.1, edgecolor='none', zorder=-1, facecolor='grey')
      #ax1.plot(mstar, mfrac_limit2, color='black', linewidth=1, linestyle = ':', zorder=1)
      #sfr_ms_cluver = 0.93 * mstar - 9.08
      #ax1.plot(mstar, sfr_ms_cluver, color='purple', linewidth=2, linestyle = '--', zorder=2)
      #plt.fill_between(mstar, sfr_ms_cluver-0.29, sfr_ms_cluver+0.29, alpha=0.05, edgecolor='none', zorder=-1, facecolor='purple')
      ax1.set_ylim(-3,1.75)
    if xlbl == r'$\log(M_*/\mathrm{M}_{\odot})$' and ylbl == r'$\log$(SFR$_{W3}$/[M$_{\odot}$\,yr$^{-1}$])':
      mstar    = np.array([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11]) + 0.25
      sfr_ms_w3   = 1.00 * (mstar - 10.)  - 0.24
      ax1.plot(mstar, sfr_ms_w3, color='black', linewidth=1, linestyle = '--', zorder=2)
      ax1.plot(mstar, sfr_ms_w3+0.3, color='black', linewidth=1, linestyle = ':', zorder=2)
      ax1.plot(mstar, sfr_ms_w3-0.3, color='black', linewidth=1, linestyle = ':', zorder=2)
      ax1.set_ylim(-3,1.75)
    if xlbl == r'$\log(M_*/\mathrm{M}_{\odot})$' and ylbl == r'$\log$(SFR$_{W4}$/[M$_{\odot}$\,yr$^{-1}$])':
      mstar    = np.array([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11]) + 0.25
      sfr_ms_w4   = 0.82 * (mstar - 10.)  - 0.27
      ax1.plot(mstar, sfr_ms_w4, color='black', linewidth=1, linestyle = '--', zorder=2)
      ax1.plot(mstar, sfr_ms_w4+0.3, color='black', linewidth=1, linestyle = ':', zorder=2)
      ax1.plot(mstar, sfr_ms_w4-0.3, color='black', linewidth=1, linestyle = ':', zorder=2)
      ax1.set_ylim(-3,1.75)
    if xlbl == r'$\log(M_*/\mathrm{M}_{\odot})$' and ylbl == r'$\log$(sSFR/[yr$^{-1}$])':
      mstar    = np.array([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11]) + 0.25
      #sfr_ms   = 1.02 * (mstar - 10.) * 0.54
      sfr_ms   = -0.344 * (mstar - 9.) - 9.822
      #sigma_ms = 0.088 * (mstar - 9.) + 0.188
      ax1.plot(mstar, sfr_ms, color='black', linewidth=1, linestyle = '--', zorder=2)
      ax1.plot(mstar, sfr_ms+0.3, color='black', linewidth=1, linestyle = ':', zorder=2)
      ax1.plot(mstar, sfr_ms-0.3, color='black', linewidth=1, linestyle = ':', zorder=2)
      #plt.fill_between(mstar, sfr_ms-sigma_ms, sfr_ms+sigma_ms, alpha=0.1, edgecolor='none', zorder=2, facecolor='grey')
      #ax1.plot(mstar, mfrac_limit2, color='black', linewidth=1, linestyle = ':', zorder=1)
      ax1.set_ylim(-12,-8.5)
    if xlbl == r'$\log(M_*/\mathrm{M}_{\odot})$' and ylbl == r'$\log$(sSFR$_{W3}$/[yr$^{-1}$])':
      mstar    = np.array([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11]) + 0.25
      sfr_ms_w3   = 1.00 * (mstar - 10.)  - 0.24 - mstar
      ax1.plot(mstar, sfr_ms_w3, color='black', linewidth=1, linestyle = '--', zorder=2)
      ax1.plot(mstar, sfr_ms_w3+0.3, color='black', linewidth=1, linestyle = ':', zorder=2)
      ax1.plot(mstar, sfr_ms_w3-0.3, color='black', linewidth=1, linestyle = ':', zorder=2)
      ax1.set_ylim(-12,-8.5)
    if xlbl == r'$\log(M_*/\mathrm{M}_{\odot})$' and ylbl == r'$\log$(sSFR$_{W4}$/[yr$^{-1}$])':
      mstar    = np.array([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11]) + 0.25
      sfr_ms_w4   = 0.82 * (mstar - 10.)  - 0.27 - mstar
      ax1.plot(mstar, sfr_ms_w4, color='black', linewidth=1, linestyle = '--', zorder=2)
      ax1.plot(mstar, sfr_ms_w4+0.3, color='black', linewidth=1, linestyle = ':', zorder=2)
      ax1.plot(mstar, sfr_ms_w4-0.3, color='black', linewidth=1, linestyle = ':', zorder=2)
      ax1.set_ylim(-12,-8.5)
    if xlbl == r'$\log(M_*/\mathrm{M}_{\odot})$':
      ax1.set_xlim(7, 12)
    if ylbl == r'$\mathrm{DEF}_{\mathrm{HI}}$':
      ax1.set_ylim(-1, 2)
    if ylbl == r'$\log(M_{\mathrm{HI}}/M_*)$':
      ax1.set_ylim(-2.75, 1.75)
    if xlbl == r'$\log(M_{\mathrm{HI}}/\mathrm{M}_{\odot})$' and ylbl == r'$\log(d_{\mathrm{HI}}/\mathrm{kpc})$':
      ax1.set_xlim(7, 10.7)
      ax1.set_ylim(0, 2.3)
      mhi    = np.arange(7,10.8,0.1)
      dhi    = 0.506 * mhi - 3.293
      ax1.plot(mhi, dhi, color='black', linestyle='-', linewidth=0.75, zorder=2)
      ax1.plot(mhi, dhi-3.*0.06, color='black', linestyle='--', linewidth=0.75, zorder=2)
      ax1.plot(mhi, dhi+3.*0.06, color='black', linestyle='--', linewidth=0.75, zorder=2)
      #plt.fill_between(mhi, dhi-3.*0.06, dhi+3.*0.06, alpha=0.2, edgecolor='none', zorder=-1, facecolor='grey')
    if xlbl == r'$\log(d_{\mathrm{opt}}/\mathrm{kpc})$' and ylbl == r'$\log(d_{\mathrm{HI}}/\mathrm{kpc})$':
      d25    = np.arange(-0.5,2.1,0.1)
      dhi    = 1. * d25 + 0.23
      dhi1    = 0.97 * d25 + 0.11
      dhi2    = 1.03 * d25 + 0.35
      ax1.plot(d25, dhi, color='purple', linestyle='-', linewidth=0.75, zorder=2)
      ax1.plot(d25, dhi1, color='purple', linestyle='--', linewidth=0.75, zorder=2)
      ax1.plot(d25, dhi2, color='purple', linestyle='--', linewidth=0.75, zorder=2)
    if xlbl == r'$d_{\mathrm{\mathrm{opt}}}/\mathrm{kpc}$' and ylbl == r'$d_{\mathrm{HI}}/d_{\mathrm{opt}}$':
      ax1.axhline(np.nanmedian(y), color=col, linestyle='--', linewidth=0.75, zorder=-1)
    if xlbl == r'$\mathrm{DEF}_{\mathrm{HI}}$' and ylbl == r'$\log(d_{\mathrm{HI}}/d_{\mathrm{opt}})$':
      ax1.axhline(np.log10(2), color='black', linestyle='--', linewidth=0.75, zorder=-1)
      ax1.axhline(0, color='black', linestyle='-.', linewidth=0.75, zorder=-1)
      ax1.axvline(0.4, color='black', linestyle=':', linewidth=0.75, zorder=-1)
    if xlbl == 'SFR [W3]' or xlbl == 'SFR [43]' or xlbl == 'SFR [NUV+MIR]':
      ax1.set_xlim(-4, 1.25)
      ax1.set_ylim(-4, 1.25)
      ax1.plot([-5, 1], [-5, 1], color='black', linestyle='-', linewidth=0.75, zorder=2)
    if xlbl == r'$r$-band radius [arcsec]':
      #ax1.plot([0, 150], [0, 150], color='black', linestyle='-', linewidth=0.75, zorder=2)
      ax1.plot([0, 120], [0, 120], color='black', linestyle='-', linewidth=0.75, zorder=2)
    if xlbl == 'W1' or xlbl == 'W2' or xlbl == 'W3' or xlbl == 'W4':
      #ax1.set_xlim(-4, 1.25)
      #ax1.set_ylim(-4, 1.25)
      ax1.plot([np.nanmin(x), np.nanmax(x)], [np.nanmin(x), np.nanmax(x)], color='black', linestyle='-', linewidth=0.75, zorder=2)
    if ylbl == 'HI Detected Fraction':
      ax1.set_ylim(0,1.1)
    #ax1.set_xlim(7, 10.7)
    #ax1.set_ylim(-0.75, 1.75)
    if sub3 == 3 and (xlbl != 'WALLABY' and ylbl != 'HIPASS'):
      ax1.legend(fontsize = 8.5, ncol=1) #upper right loc='lower left', 
    elif xlbl == 'WALLABY' and ylbl == 'HIPASS':
      ax1.legend(loc='lower right', fontsize = 8.5)
    if sub1 == 1 and sub2 == 1:
      ax1.legend(fontsize = 8.5) #loc='upper right', 
    if sub1 == 1 and sub2 == 2 and sub3 == 1:
      ax1.legend(fontsize = 8.5) #loc='upper right', 
    #if sub1 == 1 and sub2 == 2:# and sub3 == 1:
      #ax1.legend(fontsize = 8.5) #loc='upper right', 
    plt.subplots_adjust(wspace=0.35, hspace=0.35)


# ================================ #
# ====== Scatter Mean Plot ======= #
# ================================ #
def gas_fraction_plot(fig_num, subfig, data, txtstr, xlbl, ylbl, col, marker):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(subfig[0], subfig[1], subfig[2], facecolor = 'w')
    #if sub1 == 1 and sub2 == 1:
    ax1.set_xlabel(xlbl)
    ax1.set_ylabel(ylbl)
    x = data[0]
    y = data[1]
    if marker == 'o':
      ax1.scatter(x, y, color=col, marker=marker, edgecolor=col, facecolor=col, 
                  s=30, zorder=2)#, label = txtstr)
    elif marker == 'X':
      ax1.scatter(x, y, color=col, marker=marker, edgecolor='none', facecolor=col, 
                  s=25, zorder=1)
    elif marker == r'$\downarrow$':
      ax1.scatter(x, y, color=col, marker=marker, edgecolor=col, facecolor=col, 
                  s=45, zorder=1)#, label = txtstr)
    elif marker == r'$\uparrow$':
      ax1.scatter(x, y, color=col, marker=marker, edgecolor=col, facecolor=col, 
                  s=25, zorder=1)
    elif marker == 'd':
      ax1.scatter(x, y, color=col, marker=marker, edgecolor='white', facecolor=col, 
                  s=140, linewidth=1, zorder=1)#, label = txtstr)
    elif marker == '.':
      ax1.scatter(x, y, color=col, marker=marker, alpha=0.5, s=10, zorder=1)#, label = txtstr)
    if ((xlbl == r'$\log(M_*/[\mathrm{M}_{\odot}])$' and ylbl == r'$\log(M_{\mathrm{HI}}/M_*)$') or 
        (xlbl == r'$\log(M_*/\mathrm{M}_{\odot})$' and ylbl == r'$\log(f_{\rm{HI}})$')):
      snr_limit    = 5.
      chan_width   = 18518.
      chan_noise   = 0.002
      vel_width   = np.array([200., 200., 200., 200., 200., 200., 300., 300., 300.])
      #opt_radius  = np.array([10., 10., 10., 15., 15., 15., 25., 40., 55.])
      #hi_radius   = np.array([26., 25., 34., 40., 45., 47., 84., 84., 84.])
      mstar       = np.arange(7.25,11.75,0.5)
      #hi_radius   = 8.335 * mstar - 32.254
      hi_radius   = 10**(0.105 * mstar + 0.662)
      #hi_radius[mstar > 10.5] = 100
      beam_area   = (math.pi * 30. * 30.) / (4. * np.log(2))
      mhi_limit1  = np.round(np.log10(snr_limit / (2.92 * 10**-4) * (1. + 0.0126)**(-0.5) * 
                             61.**2 * vel_width**(0.5) * chan_width**(0.5) * 
                             chan_noise * np.sqrt(math.pi * 0.59 * hi_radius**2 / beam_area)),1)
      
      #mhi_limit[4:]   = np.log10(5. * (1. / (2.94 * 10**-4)) * 120.**2 * 0.0016 * np.sqrt(w50_mean[4:] * 18500.))
      mfrac_limit1 = mhi_limit1 - mstar
      ax1.plot(mstar, mfrac_limit1, color='black', linewidth=1, linestyle = '--', zorder=1)
      gfms_const        = [-0.53, -0.07]
      main_seq_gf_hi    = gfms_const[0] * (mstar - 9.) + gfms_const[1]
      ax1.plot(mstar, main_seq_gf_hi, color='grey', linewidth=1, linestyle = '--', zorder=2)
      ax1.plot(mstar, main_seq_gf_hi+0.3, color='grey', linewidth=1, linestyle = ':', zorder=2)
      ax1.plot(mstar, main_seq_gf_hi-0.3, color='grey', linewidth=1, linestyle = ':', zorder=2)
    #if xlbl == r'$\log(M_*/\mathrm{M}_{\odot})$':
      #ax1.set_xlim(7, 12)
    ax1.set_xlim(7.2,11.3)
    #if ylbl == r'$\log(M_{\mathrm{HI}}/M_*)$':
    ax1.set_ylim(-2.75, 1.75)
    #p1, = ax1.scatter(x, y, color=col, marker=marker, edgecolor=col, facecolor=col, 
                  #s=15, zorder=2)#, label = txtstr)
    #p2, = ax1.scatter(x, y, color=col, marker=marker, edgecolor=col, facecolor=col, 
                  #s=25, zorder=1)#, label = txtstr)
    #ax1.legend([(p1, p2)], [txtstr], scatterpoints=2, fontsize = 8.5) #loc='upper right',
    #if marker == 'o':
    if len(x) == 1:
      p1 = ax1.scatter(x, y, color=col[0], marker=marker[0], s=10, zorder=1)
      p2 = ax1.scatter(x, y, color=col[1], marker=marker[1], edgecolor='white', 
                       facecolor=col[1],  linewidth=1, s=80, zorder=2)
      p3 = ax1.scatter(x, y, color=col[2], marker=marker[2], edgecolor=col[2], 
                       facecolor=col[2], s=30, zorder=3)
      p4 = ax1.scatter(x, y, color=col[3], marker=marker[3], edgecolor=col[3], 
                       facecolor=col[3], s=30, zorder=3)
      p5 = ax1.scatter(x, y, color=col[4], marker=marker[4], edgecolor=col[4], 
                       facecolor=col[4], s=30, zorder=3)
      p6 = ax1.scatter(x, y, color=col[5], marker=marker[5], edgecolor=col[5], 
                       facecolor=col[5], s=30, zorder=3)
      p7 = ax1.scatter(x, y, color=col[6], marker=marker[6], edgecolor=col[6], 
                       facecolor=col[6], s=30, zorder=3)
      p8 = ax1.scatter(x, y, color=col[7], marker=marker[7], edgecolor=col[7], 
                       facecolor=col[7], s=30, zorder=3)
      l1 = ax1.legend([p1, p2], [txtstr[0], txtstr[1]], fontsize = 11, loc = 'upper right', ncol=1)
      l2 = ax1.legend(handles=[p3, p4, p5, p6, p7, p8],
          labels=['', '', '', 'Cluster', 'Infall', 'Field'],
          loc='lower left', ncol=2, handlelength=3, fontsize = 11,
          handletextpad=0, columnspacing=-1.5)
      gca().add_artist(l1)
    plt.subplots_adjust(wspace=0.35, hspace=0.35)

# ================================ #
# ====== Scatter Mean Plot ======= #
# ================================ #
def sfr_ms_plot(fig_num, subfig, data, linear_fit, txtstr, xlbl, ylbl, col, marker):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(subfig[0], subfig[1], subfig[2], facecolor = 'w')
    #if sub1 == 1 and sub2 == 1:
    ax1.set_xlabel(xlbl)
    ax1.set_ylabel(ylbl)
    x = data[0]
    y = data[1]
    if marker == 'o':
      ax1.scatter(x, y, color=col, marker=marker, edgecolor=col, facecolor=col, 
                  s=30, zorder=2, label = txtstr)
    elif marker == 's':
      ax1.scatter(x, y, color=col, marker=marker, edgecolor=col, facecolor='none', 
                  s=30, zorder=2, label = txtstr, linewidth=1.5)
    elif marker == 'X':
      ax1.scatter(x, y, color=col, marker=marker, edgecolor='none', facecolor=col, 
                  s=45, zorder=1)
    elif marker == r'$\downarrow$':
      ax1.scatter(x, y, color=col, marker=marker, edgecolor=col, facecolor=col, 
                  s=45, zorder=1)
    elif marker == r'$\uparrow$':
      ax1.scatter(x, y, color=col, marker=marker, edgecolor=col, facecolor=col, 
                  s=40, zorder=1)
    elif marker == '.':
      ax1.scatter(x, y, color=col, marker=marker, alpha=0.5, s=10, zorder=1)
    #else:
      #ax1.scatter(x, y, color=col, marker=marker, edgecolor=col, facecolor=col, s=20, zorder=1)

    mstar    = np.arange(7.25, 11.5, 0.25)
    sfr_ms   = linear_fit[0] * mstar + linear_fit[1]
    ax1.plot(mstar, sfr_ms, color='grey', linewidth=1, linestyle = '--', zorder=2)
    ax1.plot(mstar, sfr_ms+0.3, color='grey', linewidth=1, linestyle = ':', zorder=2)
    ax1.plot(mstar, sfr_ms-0.3, color='grey', linewidth=1, linestyle = ':', zorder=2)
    #sfr_ms   = 0.656 * mstar - 6.726
    #ax1.plot(mstar, sfr_ms, color='purple', linewidth=1, linestyle = '--', zorder=2)
    ax1.set_xlim(7.2,11.3)
    ax1.set_ylim(-3.0,1.2)
    #ax1.legend(fontsize = 11, ncol=1) #upper right loc='lower left', 
    if len(x) == 1:
      if x == [0] and y == [0]:
        p1 = ax1.scatter(x, y, color=col[0], marker=marker[0], s=10, zorder=1)
        #print(col)
        #print(marker)
        p2 = ax1.scatter(x, y, color=col[1], marker=marker[1], edgecolor=col[1], 
                        facecolor=col[1], s=30, zorder=3)
        p3 = ax1.scatter(x, y, color=col[2], marker=marker[2], edgecolor=col[2], 
                        facecolor=col[2], s=30, zorder=3)
        p4 = ax1.scatter(x, y, color=col[3], marker=marker[3], edgecolor=col[3], 
                        facecolor=col[3], s=30, zorder=3)
        p5 = ax1.scatter(x, y, color=col[4], marker=marker[4], edgecolor=col[4], 
                        facecolor=col[4], s=30, zorder=3)
        p6 = ax1.scatter(x, y, color=col[5], marker=marker[5], edgecolor=col[5], 
                        facecolor=col[5], s=30, zorder=3)
        p7 = ax1.scatter(x, y, color=col[6], marker=marker[6], edgecolor=col[6], 
                        facecolor=col[6], s=30, zorder=3)
        p8 = ax1.scatter(x, y, color=col[7], marker=marker[7], edgecolor='none', 
                        facecolor=col[7], s=30, zorder=3)
        p9 = ax1.scatter(x, y, color=col[8], marker=marker[8], edgecolor='none', 
                        facecolor=col[8], s=30, zorder=3)
        p10 = ax1.scatter(x, y, color=col[9], marker=marker[9], edgecolor='none', 
                        facecolor=col[9], s=30, zorder=3)
        p11 = ax1.scatter(x, y, color=col[10], marker=marker[10], edgecolor=col[10], 
                        facecolor='none', s=30, zorder=3, linewidth=1.5)
        p12 = ax1.scatter(x, y, color=col[11], marker=marker[11], edgecolor=col[11], 
                        facecolor='none', s=30, zorder=3, linewidth=1.5)
        p13 = ax1.scatter(x, y, color=col[12], marker=marker[12], edgecolor=col[12], 
                        facecolor='none', s=30, zorder=3, linewidth=1.5)
        p14 = ax1.scatter(x, y, color=col[13], marker=marker[13], edgecolor=col[13], 
                        facecolor=col[13], s=30, zorder=3)
        p15 = ax1.scatter(x, y, color=col[14], marker=marker[14], edgecolor=col[14], 
                        facecolor=col[14], s=30, zorder=3)
        p16 = ax1.scatter(x, y, color=col[15], marker=marker[15], edgecolor=col[15], 
                        facecolor=col[15], s=30, zorder=3)
        p17 = ax1.scatter(x, y, color=col[16], marker=marker[16], edgecolor='none', 
                        facecolor=col[16], s=30, zorder=3)
        p18 = ax1.scatter(x, y, color=col[17], marker=marker[17], edgecolor='none', 
                        facecolor=col[17], s=30, zorder=3)
        p19 = ax1.scatter(x, y, color=col[18], marker=marker[18], edgecolor='none', 
                        facecolor=col[18], s=30, zorder=3)
        p20 = ax1.scatter(x, y, color='black', marker=r'$\downarrow$', edgecolor='black', 
                        facecolor='black', s=30, zorder=3)
        p21 = ax1.scatter(x, y, color='black', marker='X', edgecolor='none', 
                        facecolor='black', s=30, zorder=3)
        #l1 = ax1.legend([p1, p20, p21], [txtstr[0], r'SFR$_{\rm{max}}$', r'SFR$_{\rm{min}}$'], fontsize = 11, loc = 'lower right', ncol=1)
        #l2 = ax1.legend(handles=[p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19],
            #labels=['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'Cluster', 'Infall', 'Field'],
            #loc='upper left', ncol=6, handlelength=3, fontsize = 11,
            #handletextpad=0, columnspacing=-1.75)
        l1 = ax1.legend([p1, p20], [txtstr[0], r'SFR$_{\rm{max}}$'], fontsize = 11, loc = 'lower right', ncol=1)
        l2 = ax1.legend(handles=[p2, p3, p4, p5, p6, p7, p11, p12, p13, p14, p15, p16],
            labels=['', '', '', '', '', '', '', '', '', 'Cluster', 'Infall', 'Field'],
            loc='upper left', ncol=4, handlelength=3, fontsize = 11,
            handletextpad=0, columnspacing=-1.75)
        gca().add_artist(l1)
    plt.subplots_adjust(wspace=0.35, hspace=0.35)



# ================================ #
# ====== Scatter Mean Plot ======= #
# ================================ #
def sfms_gfms_plot(fig_num, subfig, data, txtstr, xlbl, ylbl, mark_array):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(subfig[0], subfig[1], subfig[2], facecolor = 'w')
    #if sub1 == 1 and sub2 == 1:
    ax1.set_xlabel(xlbl)
    ax1.set_ylabel(ylbl)
    x = data[0]
    y = data[1]
    if mark_array[0] == 'o':
      ax1.scatter(x, y, color=mark_array[1], marker=mark_array[0], edgecolor=mark_array[2], facecolor=mark_array[3], 
                  s=30, zorder=2)#, label = txtstr)
    #elif marker == 'X':
      #ax1.scatter(x, y, color=col, marker=marker, edgecolor='none', facecolor=col, 
                  #s=25, zorder=1)
    elif mark_array[0] == r'$\downarrow$':
      ax1.scatter(x, y, color=mark_array[1], marker=mark_array[0], edgecolor=mark_array[2], facecolor=mark_array[3], 
                  s=45, zorder=2)#, label = txtstr)
    #elif marker == r'$\uparrow$':
      #ax1.scatter(x, y, color=col, marker=marker, edgecolor=col, facecolor=col, 
                  #s=25, zorder=1)
    #elif marker == 'd':
      #ax1.scatter(x, y, color=col, marker=marker, edgecolor='white', facecolor=col, 
                  #s=140, linewidth=1, zorder=1)#, label = txtstr)
    #elif marker == '.':
      #ax1.scatter(x, y, color=col, marker=marker, alpha=0.5, s=10, zorder=1)#, label = txtstr)
    #plt.plot([-1, 1], [-1, 1], color='grey', linestyle='--', zorder=0)
    plt.axvline(0, color = 'black', linestyle = '-', linewidth = 1, zorder = 0)
    plt.axvline(0.3, color = 'black', linestyle = ':', linewidth = 1, zorder = 0)
    plt.axvline(-0.3, color = 'black', linestyle = ':', linewidth = 1, zorder = 0)
    plt.axhline(0, color = 'black', linestyle = '-', linewidth = 1, zorder = 0)
    plt.axhline(0.3, color = 'black', linestyle = ':', linewidth = 1, zorder = 0)
    plt.axhline(-0.3, color = 'black', linestyle = ':', linewidth = 1, zorder = 0)
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    #if xlbl == r'$\log(M_*/\mathrm{M}_{\odot})$':
      #ax1.set_xlim(7, 12)
    #ax1.set_xlim(7.2,11.3)
    #if ylbl == r'$\log(M_{\mathrm{HI}}/M_*)$':
    #ax1.set_ylim(-2.75, 1.75)
    #p1, = ax1.scatter(x, y, color=col, marker=marker, edgecolor=col, facecolor=col, 
                  #s=15, zorder=2)#, label = txtstr)
    #p2, = ax1.scatter(x, y, color=col, marker=marker, edgecolor=col, facecolor=col, 
                  #s=25, zorder=1)#, label = txtstr)
    #ax1.legend([(p1, p2)], [txtstr], scatterpoints=2, fontsize = 8.5) #loc='upper right',
    #if marker == 'o':
    if len(mark_array) == 2:
      #p1 = ax1.scatter(x, y, color=col[0], marker=marker[0], s=10, zorder=1)
      #p2 = ax1.scatter(x, y, color=col[1], marker=marker[1], edgecolor='white', 
                       #facecolor=col[1],  linewidth=1, s=80, zorder=2)
      col    = mark_array[0]
      marker = mark_array[1]
      p3 = ax1.scatter(x, y, color=col[2], marker=marker[2], edgecolor=col[2], 
                       facecolor=col[2], s=30, zorder=3)
      p4 = ax1.scatter(x, y, color=col[3], marker=marker[3], edgecolor=col[3], 
                       facecolor=col[3], s=30, zorder=3)
      p5 = ax1.scatter(x, y, color=col[4], marker=marker[4], edgecolor=col[4], 
                       facecolor=col[4], s=30, zorder=3)
      p9 = ax1.scatter(x, y, color=col[2], marker=marker[2], edgecolor=col[2], 
                       facecolor='none', s=30, zorder=3)
      p10 = ax1.scatter(x, y, color=col[3], marker=marker[3], edgecolor=col[3], 
                       facecolor='none', s=30, zorder=3)
      p11 = ax1.scatter(x, y, color=col[4], marker=marker[4], edgecolor=col[4], 
                       facecolor='none', s=30, zorder=3)
      p6 = ax1.scatter(x, y, color=col[5], marker=marker[5], edgecolor=col[5], 
                       facecolor=col[5], s=30, zorder=3)
      p7 = ax1.scatter(x, y, color=col[6], marker=marker[6], edgecolor=col[6], 
                       facecolor=col[6], s=30, zorder=3)
      p8 = ax1.scatter(x, y, color=col[7], marker=marker[7], edgecolor=col[7], 
                       facecolor=col[7], s=30, zorder=3)
      #l1 = ax1.legend([p1, p2], [txtstr[0], txtstr[1]], fontsize = 11, loc = 'upper right', ncol=1)
      l2 = ax1.legend(handles=[p3, p4, p5, p9, p10, p11, p6, p7, p8],
          labels=['', '', '', '', '', '', 'Cluster', 'Infall', 'Field'],
          loc='upper left', ncol=3, handlelength=1.5, fontsize = 11,
          handletextpad=0.1, columnspacing=-0.2, borderpad=0.3)
      #gca().add_artist(l1)
    plt.subplots_adjust(wspace=0.35, hspace=0.35)


# ================================ #
# ====== Scatter Mean Plot ======= #
# ================================ #
def sfr_ms_fit_plot(fig_num, subfig, data, linear_fit, txtstr, xlbl, ylbl, col, marker, do_j2017):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(subfig[0], subfig[1], subfig[2], facecolor = 'w')
    #if sub1 == 1 and sub2 == 1:
    ax1.set_xlabel(xlbl)
    ax1.set_ylabel(ylbl)
    x = data[0]
    y = data[1]
    if marker == 'o':
      ax1.scatter(x, y, color=col, marker=marker, edgecolor=col, facecolor='none', 
                  s=10, zorder=2, label = txtstr)
    elif marker == 'X':
      ax1.scatter(x, y, color=col, marker=marker, edgecolor='none', facecolor=col, 
                  s=10, zorder=1)
    elif marker == r'$\downarrow$':
      ax1.scatter(x, y, color=col, marker=marker, edgecolor=col, facecolor=col, 
                  s=10, zorder=1)
    elif marker == '.':
      ax1.scatter(x, y, color=col, marker=marker, s=5, zorder=1)
    else:
      ax1.scatter(x, y, color=col, marker=marker, edgecolor=col, facecolor=col, s=10, zorder=1)

    mstar    = np.arange(7.25, 11.5, 0.25)
    sfr_ms   = linear_fit[0] * mstar + linear_fit[1]
    ax1.plot(mstar, sfr_ms, color='black', linewidth=1, linestyle = '--', zorder=2)
    ax1.plot(mstar, sfr_ms+0.3, color='black', linewidth=1, linestyle = ':', zorder=2)
    ax1.plot(mstar, sfr_ms-0.3, color='black', linewidth=1, linestyle = ':', zorder=2)
    if do_j2017:
      sfr_ms   = 0.656 * mstar - 6.726
      ax1.plot(mstar, sfr_ms, color='red', linewidth=1, linestyle = '--', zorder=2)
    ax1.set_xlim(7.2,11.3)
    ax1.set_ylim(-3.1,1.1)
    ax1.legend(fontsize = 8.5, ncol=1) #upper right loc='lower left', 
    plt.subplots_adjust(wspace=0.35, hspace=0.35)


# ================================ #
# ====== Scatter Mean Plot ======= #
# ================================ #
def ssfr_ms_plot(fig_num, subfig, data, linear_fit, txtstr, xlbl, ylbl, col, marker):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(subfig[0], subfig[1], subfig[2], facecolor = 'w')
    #if sub1 == 1 and sub2 == 1:
    ax1.set_xlabel(xlbl)
    ax1.set_ylabel(ylbl)
    x = data[0]
    y = data[1]
    if marker == '.':
      if txtstr == 'Non-det':
        ax1.scatter(x, y, color=col, marker=marker, alpha=0.8, s=2, zorder=1)#, label = txtstr)
      else:
        ax1.scatter(x, y, color=col, marker=marker, alpha=0.8, s=2, zorder=1, label = txtstr)
    else:
      if marker == 'o':
        ax1.scatter(x, y, color=col, marker=marker, edgecolor=col, facecolor='none', 
                    alpha=0.95, s=7.5, zorder=1,label = txtstr)
      else:
        ax1.scatter(x, y, color=col, marker=marker, edgecolor='none', facecolor=col, alpha=0.95, s=7.5, zorder=1)

    mstar    = np.array([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11]) + 0.25
    sfr_ms   = linear_fit[0] * mstar + linear_fit[1]
    ax1.plot(mstar, sfr_ms, color='purple', linewidth=1, linestyle = '--', zorder=2)
    ax1.plot(mstar, sfr_ms+0.3, color='purple', linewidth=1, linestyle = ':', zorder=2)
    ax1.plot(mstar, sfr_ms-0.3, color='purple', linewidth=1, linestyle = ':', zorder=2)
    #sfr_ms   = 0.656 * mstar - 6.816
    #ax1.plot(mstar, sfr_ms, color='black', linewidth=1, linestyle = '--', zorder=2)
    #ax1.set_ylim(-3,1.75)
    if subfig[2] == 3:
      ax1.legend(fontsize = 8.5, ncol=1) #upper right loc='lower left', 
    plt.subplots_adjust(wspace=0.35, hspace=0.35)


# ================================ #
# ====== Scatter Mean Plot ======= #
# ================================ #
def hi_detected_fraction_plot(fig_num, subfig, x, y, yerr, xlbl, ylbl, col, marker, lbl):
    matplotlib.rcParams.update({'font.size': 14})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(subfig[0], subfig[1], subfig[2], facecolor = 'w')
    ax1.set_xlabel(xlbl)
    ax1.set_ylabel(ylbl)
    if len(x) > 1:
      plt.errorbar(x, y, yerr = yerr, color=col, markersize=8, 
                  fmt=marker, elinewidth=1.5, zorder=2, label = lbl)
    else:
      plt.axhline(y, color = col, linestyle = '--', linewidth = 2.5, zorder = 1, label = lbl)
      plt.axhline(y-yerr, color = col, linestyle = ':', linewidth = 1.5, zorder = 1)
      plt.axhline(y+yerr, color = col, linestyle = ':', linewidth = 1.5, zorder = 1)
    ax1.set_ylim(0,1.1)
    ax1.legend(fontsize = 12) #loc='upper right', 
    plt.subplots_adjust(wspace=0.35, hspace=0.35)


# ================================ #
# ====== Scatter Mean Plot ======= #
# ================================ #
def scat_col_plot(fig_num, sub1, sub2, sub3, x, y, z, txtstr, xlbl, ylbl, cmap, marker):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    #if sub1 == 1 and sub2 == 1:
    ax1.set_xlabel(xlbl)
    ax1.set_ylabel(ylbl)
    #scat_ax = ax1.scatter(x, y, c=z, cmap=cmap, marker=marker, facecolor='none', alpha=1, s=15, vmin=0, vmax=4, zorder=1, label = txtstr)
    if xlbl == r'$R_{\mathrm{proj}}/R_{\mathrm{vir}}$' and ylbl == r'$\mathrm{DEF}_{\mathrm{HI}}$':
      scat_ax = ax1.scatter(x, y, c=z, cmap=cmap, marker=marker, facecolor='none', alpha=1, s=15, vmin=-0.9, vmax=0.8, zorder=1, label = txtstr)
    elif xlbl == r'$\log(M_*/\mathrm{M}_{\odot})$' and (ylbl == r'$\log$(SFR/[M$_{\odot}$\,yr$^{-1}$])' or ylbl == r'$\log$(sSFR/[yr$^{-1}$])'):
      scat_ax = ax1.scatter(x, y, c=z, cmap=cmap, marker=marker, facecolor='none', alpha=1, s=15, vmin=-1, vmax=0.8, zorder=1, label = txtstr)
    else:
      scat_ax = ax1.scatter(x, y, c=z, cmap=cmap, marker=marker, facecolor='none', alpha=1, s=15, vmin=8, vmax=10.5, zorder=1, label = txtstr)
    scat_ax.set_facecolor('none')
    if txtstr == 'Field' and sub3 == 3:
      cbar = plt.colorbar(scat_ax, fraction=0.1, pad=0.01) #cax=cbar_ax, 
    if txtstr == 'Cluster' and (sub3 == 4 or sub3 == 6):
      cbar = plt.colorbar(scat_ax, fraction=0.1, pad=0.01) #cax=cbar_ax, 
    if sub1 == 1 and sub2 == 1:
      cbar = plt.colorbar(scat_ax, fraction=0.1, pad=0.01, label = r'$\log(d_{\mathrm{HI}}/d_{25})$')
      #cbar.set_label(r'$\log(d_{\mathrm{HI}}/d_{25})$')
    if sub1 == 1 and sub2 == 2 and txtstr == 'Cluster':
      cbar = plt.colorbar(scat_ax, fraction=0.1, pad=0.01)
    #cbar.set_clim(0,2000)
    #clbl = r'$\Delta\mathrm{c}z$ [km\,s$^{-1}$]'
    #cbar.set_label(clbl)
    if ylbl == r'$d_{\mathrm{HI}}/d_{\mathrm{opt}}$':
      ax1.axhline(1, color='black', linestyle='--', linewidth=0.75, zorder=-1)
      ax1.set_ylim(0, 7)
    if xlbl == r'$\log(M_*/\mathrm{M}_{\odot})$':
      ax1.set_xlim(8, 12)
    if xlbl == r'$\log(M_*/\mathrm{M}_{\odot})$' and ylbl == r'$\log$(SFR/[M$_{\odot}$\,yr$^{-1}$])':
      mstar    = np.array([8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5]) + 0.25
      #sfr_ms   = 1.02 * (mstar - 10.) * 0.54
      sfr_ms   = mstar - 0.344 * (mstar - 9.) - 9.822
      sigma_ms = 0.088 * (mstar - 9.) + 0.188
      ax1.plot(mstar, sfr_ms, color='black', linewidth=1, linestyle = '--', zorder=2)
      #plt.fill_between(mstar, sfr_ms-sigma_ms, sfr_ms+sigma_ms, alpha=0.1, edgecolor='none', zorder=-1, facecolor='grey')
      #ax1.plot(mstar, mfrac_limit2, color='black', linewidth=1, linestyle = ':', zorder=1)
      sfr_ms_cluver = 0.93 * mstar - 9.08
      ax1.plot(mstar, sfr_ms_cluver, color='purple', linewidth=2, linestyle = '--', zorder=2)
      #plt.fill_between(mstar, sfr_ms_cluver-0.29, sfr_ms_cluver+0.29, alpha=0.05, edgecolor='none', zorder=-1, facecolor='purple')
      ax1.set_ylim(-2.5,1)
    if xlbl == r'$\log(M_*/\mathrm{M}_{\odot})$' and ylbl == r'$\log$(sSFR/[yr$^{-1}$])':
      mstar    = np.array([8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5]) + 0.25
      #sfr_ms   = 1.02 * (mstar - 10.) * 0.54
      sfr_ms   = -0.344 * (mstar - 9.) - 9.822
      sigma_ms = 0.088 * (mstar - 9.) + 0.188
      ax1.plot(mstar, sfr_ms, color='black', linewidth=1, linestyle = '--', zorder=2)
      #plt.fill_between(mstar, sfr_ms-sigma_ms, sfr_ms+sigma_ms, alpha=0.1, edgecolor='none', zorder=2, facecolor='grey')
      #ax1.plot(mstar, mfrac_limit2, color='black', linewidth=1, linestyle = ':', zorder=1)
      ax1.set_ylim(-11.6,-9.3)
    if sub3 == 3 and (xlbl != 'WALLABY' and ylbl != 'HIPASS'):
      ax1.legend(fontsize = 8.5, ncol=2) #upper right loc='lower left', 
    elif xlbl == 'WALLABY' and ylbl == 'HIPASS':
      ax1.legend(loc='lower right', fontsize = 8.5)
    if sub1 == 1 and sub2 == 1:
      ax1.legend(fontsize = 8.5) #loc='upper right', 
    plt.subplots_adjust(wspace=0.35, hspace=0.35)


# ================================ #
# ====== Scatter Mean Plot ======= #
# ================================ #
def scat_col_simple_plot(fig_num, sub1, sub2, sub3, x, y, z, xlbl, ylbl, zlbl, cmap, marker):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    #if sub1 == 1 and sub2 == 1:
    ax1.set_xlabel(xlbl)
    ax1.set_ylabel(ylbl)
    #scat_ax = ax1.scatter(x, y, c=z, cmap=cmap, marker=marker, facecolor='none', alpha=1, s=15, vmin=0, vmax=4, zorder=1, label = txtstr)
    scat_ax = ax1.scatter(x, y, c=z, cmap=cmap, marker=marker, facecolor='none', alpha=1, s=15, zorder=1)
    #vmin=8, vmax=10.5,
    scat_ax.set_facecolor('none')
    cbar = plt.colorbar(scat_ax, fraction=0.1, pad=0.01, label = zlbl) #cax=cbar_ax, 
    #ax1.legend(fontsize = 8.5) #loc='upper right', 
    plt.subplots_adjust(wspace=0.5, hspace=0.35)

# ================================ #
# ====== Scatter Mean Plot ======= #
# ================================ #
def scat_col_oto_plot(fig_num, sub1, sub2, sub3, x, y, z, txtstr, xlbl, ylbl, cmap, marker):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    #if sub1 == 1 and sub2 == 1:
    ax1.set_xlabel(xlbl)
    ax1.set_ylabel(ylbl)
    scat_ax = ax1.scatter(x, y, c=z, cmap=cmap, marker=marker, facecolor='none', alpha=1, s=15, vmin=7, vmax=11.5, zorder=1, label = txtstr)
    scat_ax.set_facecolor('none')
    if sub3 == 2 and txtstr == 'WALLABY':
      cbar = plt.colorbar(scat_ax, fraction=0.1, pad=0.01)
    ax1.set_xlim(-4, 1.5)
    ax1.set_ylim(-4, 1.5)
    ax1.plot([-4, 1.5], [-4, 1.5], linestyle='--', color = 'black', zorder = -1)
    ax1.plot([-4, 1.5], [-4.3, 1.2], linestyle=':', color = 'black', zorder = -1)
    ax1.plot([-4, 1.5], [-3.7, 1.8], linestyle=':', color = 'black', zorder = -1)
    if sub3 == 3 and (xlbl != 'WALLABY' and ylbl != 'HIPASS'):
      ax1.legend(fontsize = 8.5, ncol=2) #upper right loc='lower left', 
    elif xlbl == 'WALLABY' and ylbl == 'HIPASS':
      ax1.legend(loc='lower right', fontsize = 8.5)
    if sub1 == 1 and sub2 == 1:
      ax1.legend(fontsize = 8.5) #loc='upper right', 
    plt.subplots_adjust(wspace=0.35, hspace=0.35)


# ================================ #
# ========= Scatter Plot ========= #
# ================================ #
def barolo_vrot_plot(fig_num, sub1, sub2, sub3, x, y, txtstr, xlbl, ylbl, col, marker, lbl):
    matplotlib.rcParams.update({'font.size': 10})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    #if sub1 == 1 and sub2 == 1:
    #ax1.set_xlabel(xlbl)
    plt.text(0.4, 0.05, txtstr, transform=ax1.transAxes, fontsize=9)
    if sub3 > 15:
      ax1.set_xlabel(xlbl)
    if sub3 == 1 or sub3 == 6 or sub3 == 11 or sub3 == 16 or sub3 == 21:
      ax1.set_ylabel(ylbl)
    ax1.scatter(x, y, color=col, marker=marker, s=7, label = lbl, zorder=2)
    if np.nanmax(y) < 75:
      ymax = np.nanmax(y) + 25
    else:
      ymax = np.nanmax(y) + 75
    if lbl == r'4 Tristan':
      ax1.set_ylim(0, ymax)
    ax1.set_xlim(0, None)
    if sub3 == 5:
      ax1.legend(bbox_to_anchor=(-2., -4.55, -1.5, .102), loc='center', ncol=2, fontsize = 11)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)


# ================================ #
# ========= Scatter Plot ========= #
# ================================ #
def scat_mag_derivative_plot(fig_num, subfig, data, fit, txtstr, xlbl, ylbl, col, marker):
    matplotlib.rcParams.update({'font.size': 10})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(subfig[0], subfig[1], subfig[2], facecolor = 'w')
    #if sub1 == 1 and sub2 == 1:
    #ax1.set_xlabel(xlbl)
    plt.text(0.05, 0.9, txtstr, transform=ax1.transAxes, fontsize=9)
    ax1.set_xlabel(xlbl)
    if subfig[2] == 1:
      ax1.set_ylabel(ylbl)
    ax1.set_ylim(np.nanmean(data[1])-0.5, np.nanmean(data[1])+0.5)
    ax1.scatter(data[0], data[1], color=col[0], marker=marker, edgecolor=col[0], facecolor='none', s=5, zorder=2)
    ax1.plot(fit[0], fit[1], color=col[1], linewidth = 1, zorder=1)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

# ================================ #
# ====== Growth Curve Plot ======= #
# ================================ #
def growth_curve_plot(fig_num, subfig, data, axlbls, radius, txtstr, txtpos):
    matplotlib.rcParams.update({'font.size': 10})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(subfig[0], subfig[1], subfig[2], facecolor = 'w')
    ax1.set_xlabel(axlbls[0])
    ax1.set_ylabel(axlbls[1])
    plt.scatter(data[0], data[1], color='darkblue', s=5, edgecolor='darkblue', facecolor='none')
    plt.scatter(data[0], data[2], color='darkred', s=5, edgecolor='darkred', facecolor='none')
    plt.axvline(radius, color = 'black', linestyle = ':', linewidth = 1, zorder = 0)
    #if txtpos == 'upper':
      #plt.text(0.25, 0.9, txtstr, transform=ax1.transAxes)
      #plt.axhline(23.5, color = 'grey', linestyle = '--', linewidth=1, zorder = 0)
    #if txtpos == 'lower':
      #plt.text(0.25, 0.1, txtstr, transform=ax1.transAxes)
    plt.gca().invert_yaxis()
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    ax1.get_yaxis().set_tick_params(which='both', direction='in')
    ax1.get_xaxis().set_tick_params(which='both', direction='in')


# ================================ #
# ========= Scatter Plot ========= #
# ================================ #
def scat_mag_derivative_plot(fig_num, subfig, data, fit, txtstr, xlbl, ylbl, col, marker):
    matplotlib.rcParams.update({'font.size': 10})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(subfig[0], subfig[1], subfig[2], facecolor = 'w')
    #if sub1 == 1 and sub2 == 1:
    #ax1.set_xlabel(xlbl)
    plt.text(0.05, 0.9, txtstr, transform=ax1.transAxes, fontsize=9)
    ax1.set_xlabel(xlbl)
    if subfig[2] == 1:
      ax1.set_ylabel(ylbl)
    ax1.set_ylim(np.nanmean(data[1])-0.5, np.nanmean(data[1])+0.5)
    ax1.scatter(data[0], data[1], color=col[0], marker=marker, edgecolor=col[0], facecolor='none', s=5, zorder=2)
    ax1.plot(fit[0], fit[1], color=col[1], linewidth = 1, zorder=1)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)


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


# ================================ #
# ======= Opt/Contour Plot ======= #
# ================================ #
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
  ax1 = aplpy.FITSFigure(image1, figure=fig_num, subplot=subfig_param, dimensions=(0,1))# slices=(0,1))
  if subfig3 == 1 or subfig3 == 4 or subfig3 == 7 or subfig3 == 10:
    f2        = pyfits.open(image2)
    data, hdr = f2[0].data, f2[0].header
    #if txtstr == 'J103729-261901':
      #print(hdr)
    pixels        = np.max((hdr['NAXIS1'], hdr['NAXIS2']))
    arcmin_per_pixel = np.abs(hdr['CDELT2']) * 60.
    box_side      = int(np.ceil(arcmin_per_pixel * pixels))
    try:
      if txtstr == 'J103729-261901':
        beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
      else:
        beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], hdr['BPA']
      #print(beam_maj, beam_min)
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
    pixels        = np.max((hdr['NAXIS1'], hdr['NAXIS2']))
    arcmin_per_pixel = np.abs(hdr['CDELT2']) * 60.
    box_side      = int(np.ceil(arcmin_per_pixel * pixels))
    freq_min = np.nanmean(data) - 2.5*madev(data, ignore_nan=True) #np.nanmin(data)
    freq_max = np.nanmean(data) + 2.5*madev(data, ignore_nan=True) #np.nanmax(data)
    ax1.show_colorscale(vmin=freq_min, vmax=freq_max, cmap='RdBu_r')
  ra  = '%sh%sm%ss' % (txtstr[1:3], txtstr[3:5], txtstr[5:7])
  dec = '%sd%sm%ss' % (txtstr[7:10], txtstr[10:12], txtstr[12:])
  position     = SkyCoord(ra, dec, frame='icrs')
  #size         = u.Quantity((360, 360), u.arcsec)
  #if txtstr == 'J103725-251916' or txtstr == 'J104142-284653':
    #width, height = 660./60./60., 660./60./60. # 420
  #else:
    #width, height = 420./60./60., 420./60./60. # 420a
  width, height = box_side/60., box_side/60.
  #print(position)
  ax1.recenter(position.ra, position.dec, width=width, height=height)
  if subfig3 == 1 or subfig3 == 4 or subfig3 == 7 or subfig3 == 10:
    ax1.add_label(0.35, 0.9, txtstr, relative=True)
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

# ================================ #
# ======= Opt/Contour Plot ======= #
# ================================ #
def askap_opt_contour_plot(fig_num, subfig1, subfig2, subfig3, image1, image2, colour, txtstr, txt_array):
  matplotlib.rcParams.update({'font.size': 15})
  rc('font',**{'family':'serif','serif':['Times']})
  rc('text', usetex=True)
  ax1 = aplpy.FITSFigure(image1, figure=fig_num, subplot=(subfig1, subfig2, subfig3), dimensions=(0,1))# slices=(0,1))
  if subfig3 == 1 or subfig3 == 4 or subfig3 == 7 or subfig3 == 10:
    f1        = pyfits.open(image1)
    data1, hdr1 = f1[0].data, f1[0].header
    #print(np.nanmin(data1), np.nanmax(data1))
    #ax1.show_colorscale(vmin=0, vmax=1500, cmap='Greys')
    ax1.show_colorscale(vmin=5, pmax=99.75, stretch='arcsinh', cmap='Greys') #np.nanmax(data1)/75.
    f2        = pyfits.open(image2)
    data, hdr = f2[0].data, f2[0].header
    #if txtstr == 'J103729-261901':
      #print(hdr)
    pixels        = np.max((hdr['NAXIS1'], hdr['NAXIS2']))
    arcmin_per_pixel = np.abs(hdr['CDELT2']) * 60.
    box_side      = int(np.ceil(arcmin_per_pixel * pixels))
  if subfig3 == 2 or subfig3 == 5 or subfig3 == 8 or subfig3 == 11:
    f2        = pyfits.open(image2)
    data, hdr = f2[0].data, f2[0].header
    #if txtstr == 'J103729-261901':
      #print(hdr)
    pixels        = np.max((hdr['NAXIS1'], hdr['NAXIS2']))
    arcmin_per_pixel = np.abs(hdr['CDELT2']) * 60.
    box_side      = int(np.ceil(arcmin_per_pixel * pixels))
    #print(txtstr)
    #if txtstr == '103545-284609':
      #print(hdr['BMAJ'], hdr['BMIN'])
    try:
      if txtstr == 'J103729-261901':
        beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
      elif txtstr == 'J103726-261843':
        beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
      elif txtstr == 'J103406-270617':
        beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
      elif txtstr == 'J103542-284604':
        beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
      elif txtstr == 'J103545-284609':
        beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
      elif txtstr == 'J103537-284607':
        beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
      else:
        beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], hdr['BPA']
      #print(beam_maj, beam_min)
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
    #ax1.show_colorscale(vmin=np.min(data1), vmax=np.max(data1), cmap='Greys')
    #ax1.show_colorscale(vmin=0, vmax=2500, cmap='Greys')
    ax1.show_colorscale(vmin=5, pmax=99.75, stretch='arcsinh', cmap='Greys') #np.nanmax(data1)/75.
    ax1.add_beam(major=beam_maj, minor=beam_min, angle=beam_pa, fill=True, color='peru')
  #if subfig3 == 2 or subfig3 == 5 or subfig3 == 8 or subfig3 == 11:
  if subfig3 == 3 or subfig3 == 6 or subfig3 == 9 or subfig3 == 12:
    f1               = pyfits.open(image1)
    data, hdr        = f1[0].data, f1[0].header
    f2               = pyfits.open(image2)
    data2, hdr2      = f2[0].data, f2[0].header
    pixels           = np.max((hdr2['NAXIS1'], hdr2['NAXIS2']))
    arcmin_per_pixel = np.abs(hdr2['CDELT2']) * 60.
    box_side         = int(np.ceil(arcmin_per_pixel * pixels))
    freq_min = np.nanmean(data) - 2.5*madev(data, ignore_nan=True) #np.nanmin(data)
    freq_max = np.nanmean(data) + 2.5*madev(data, ignore_nan=True) #np.nanmax(data)
    ax1.show_colorscale(vmin=freq_min, vmax=freq_max, cmap='RdBu_r')
    try:
      if txtstr == 'J103729-261901':
        beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
      elif txtstr == 'J103726-261843':
        beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
      elif txtstr == 'J103406-270617':
        beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
      elif txtstr == 'J103542-284604':
        beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
      elif txtstr == 'J103545-284609':
        beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
      elif txtstr == 'J103537-284607':
        beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
      else:
        beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], hdr['BPA']
      #print(beam_maj, beam_min)
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
    ax1.add_beam(major=beam_maj, minor=beam_min, angle=beam_pa, fill=True, color='peru')
  #ra  = '%sh%sm%ss' % (txtstr[1:3], txtstr[3:5], txtstr[5:7])
  #dec = '%sd%sm%ss' % (txtstr[7:10], txtstr[10:12], txtstr[12:])
  #print(ra, dec)
  position     = SkyCoord(txt_array[9]*u.deg, txt_array[10]*u.deg, frame='icrs')
  #size         = u.Quantity((360, 360), u.arcsec)
  #if txtstr == 'J103725-251916' or txtstr == 'J104142-284653':
    #width, height = 660./60./60., 660./60./60. # 420
  #else:
    #width, height = 420./60./60., 420./60./60. # 420a
  width, height = box_side/60. - box_side/3./60., box_side/60. - box_side/3./60.
  #print(position)
  #print(width, height)
  ax1.recenter(position.ra, position.dec, width=width, height=height)
  #if subfig3 == 1 or subfig3 == 4 or subfig3 == 7 or subfig3 == 10:
  if subfig3 == 1 or subfig3 == 4 or subfig3 == 7 or subfig3 == 10:
    ax1.add_label(0.4, 0.9, txtstr, relative=True)
  if subfig3 == 3 or subfig3 == 6 or subfig3 == 9 or subfig3 == 12:
    size_ratio = txt_array[7]/txt_array[11]
    ax1.add_label(1.5, 0.95, '%s (%i)' % (txtstr, txt_array[12]), relative=True)
    ax1.add_label(1.5, 0.83, r'$v_{\mathrm{sys}}=%.0f$' % txt_array[0], relative=True)
    ax1.add_label(1.5, 0.71, r'$\log(M_{\mathrm{HI}})=%.1f$' % txt_array[1], relative=True)
    ax1.add_label(1.5, 0.59, r'$\log(M_{\mathrm{*}})=%.1f$' % txt_array[2], relative=True)
    ax1.add_label(1.5, 0.47, r'$\log(\mathrm{SFR})=%.2f$' % txt_array[3], relative=True)
    #ax1.add_label(1.4, 0.15, r'$d_{\mathrm{HI,3\sigma}}=%.0f$' % txt_array[4], relative=True)
    ax1.add_label(1.5, 0.35, r'$d_{\mathrm{HI}}/d_{25}=%.0f/%.0f\,\,(%.1f)$' % (txt_array[7], txt_array[11], size_ratio), relative=True)
    #ax1.add_label(1.45, 0.23, r'$d_{25}=%.0f$' % txt_array[11], relative=True)
    #ax1.add_label(1.45, 0.23, r'$\mathrm{HI}/25=%.1f$' % size_ratio, relative=True)
    ax1.add_label(1.5, 0.23, r'$\mathrm{DEF}_{\mathrm{HI}}=%.2f$' % txt_array[13], relative=True)
  if subfig3 == 2 or subfig3 == 5 or subfig3 == 8 or subfig3 == 11:
    min_flux = 1. * 10**19 * (bmaj * bmin) / (2.33 * 10**20) #1.36*21*21*1.823*1000*(10**18))
    #lvls = np.array([5, 10, 20, 50, 70, 100, 130])
    lvls = np.array([5, 20, 50, 100])
    lvls = lvls*min_flux
    ax1.show_contour(image2, colors='darkred', levels=lvls, slices=(0,1))
    min_flux = 1. * (bmaj * bmin) / (2.12) #1.36*21*21*1.823*1000*(10**18))
    #lvls = np.array([5, 10, 20, 50, 70, 100, 130])
    #lvls = lvls*min_flux
    ax1.show_contour(image2, colors='red', linewidths=4, levels=[min_flux], slices=(0,1))
    #print(txt_array[4], txt_array[5], txt_array[6])
    #ax1.show_ellipses(position.ra, position.dec, width=txt_array[4]/3600., height=txt_array[5]/3600.,
                      #angle=txt_array[6], facecolor='none', edgecolor='violet', ls='-', zorder=2, linewidth=2,
                      #coords_frame='world')
    ax1.show_ellipses(position.ra, position.dec, width=txt_array[7]/3600., height=txt_array[8]/3600., angle=txt_array[6], 
                      facecolor='none', edgecolor='blue', ls='--', zorder=2, linewidth=3, coords_frame='world')
  #if subfig3 == 3 or subfig3 == 6 or subfig3 == 9 or subfig3 == 12:
    #min_flux = 1. * 10**19 * (bmaj*bmin) / (2.33 * 10**20) #1.36*21*21*1.823*1000*(10**18))
    #lvls = np.array([5, 10, 20, 50, 70, 100, 130])
    #lvls = lvls*min_flux
    #ax1.show_contour(image2, colors="blue", levels=lvls, slices=(0,1))
  if subfig3 == 1 or subfig3 == 4 or subfig3 == 7 or subfig3 == 10 or subfig3 == 13 or subfig3 == 16:
  #  ax1.set_ylabel(r'Declination')
    #ax1.axis_labels.set_ytext('Declination')
    dummy=0
  else:
    ax1.tick_labels.hide_y()
  ax1.axis_labels.hide_y()
  #if subfig3 == 16 or subfig3 == 17 or subfig3 == 10 or subfig3 == 11:
  ##  ax1.set_xlabel(r'Right Ascension')
    ##ax1.axis_labels.set_xtext('Right Ascension')
    #dummy=0
  #else:
  #ax1.axis_labels.hide_x()
  #if subfig3 == 2 or subfig3 == 5 or subfig3 == 8 or subfig3 == 11 or subfig3 == 14 or subfig3 == 17:
    #ax1.tick_labels.hide_y()
  ax1.tick_labels.set_xformat('hh:mm:ss')
  ax1.tick_labels.set_yformat('dd:mm')
  ax1.ticks.show()
  ax1.ticks.set_xspacing(0.05)
  ax1.ticks.set_length(5)
  ax1.ticks.set_color('black')
  ax1.ticks.set_minor_frequency(1)
  #ax1.ticks.set_tick_direction('in')
  #ax1.tick_labels.set_style('plain')
  plt.subplots_adjust(wspace=0.01, hspace=0.15)
  #f1.close()


# ================================ #
# ======= Opt/Contour Plot ======= #
# ================================ #
def askap_opt_spec_plot(fig_num, subfig1, subfig2, subfig3, image1, image2, colour, txtstr, txt_array):
  matplotlib.rcParams.update({'font.size': 15})
  rc('font',**{'family':'serif','serif':['Times']})
  rc('text', usetex=True)
  if type(image1) == str:
    ax1 = aplpy.FITSFigure(image1, figure=fig_num, subplot=(subfig1, subfig2, subfig3), dimensions=(0,1))# slices=(0,1))
    if subfig3 == 1 or subfig3 == 5 or subfig3 == 9 or subfig3 == 13:
      f1        = pyfits.open(image1)
      data1, hdr1 = f1[0].data, f1[0].header
      ax1.show_colorscale(vmin=5, pmax=99.75, stretch='arcsinh', cmap='Greys')
      f2        = pyfits.open(image2)
      data, hdr = f2[0].data, f2[0].header
      pixels        = np.max((hdr['NAXIS1'], hdr['NAXIS2']))
      arcmin_per_pixel = np.abs(hdr['CDELT2']) * 60.
      box_side      = int(np.ceil(arcmin_per_pixel * pixels))
    if subfig3 == 2 or subfig3 == 6 or subfig3 == 10 or subfig3 == 14:
      f2        = pyfits.open(image2)
      data, hdr = f2[0].data, f2[0].header
      pixels        = np.max((hdr['NAXIS1'], hdr['NAXIS2']))
      arcmin_per_pixel = np.abs(hdr['CDELT2']) * 60.
      box_side      = int(np.ceil(arcmin_per_pixel * pixels))
      try:
        if txtstr == 'J103729-261901':
          beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
        elif txtstr == 'J103726-261843':
          beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
        elif txtstr == 'J103406-270617':
          beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
        elif txtstr == 'J103542-284604':
          beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
        elif txtstr == 'J103545-284609':
          beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
        elif txtstr == 'J103537-284607':
          beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
        else:
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
      ax1.show_colorscale(vmin=5, pmax=99.75, stretch='arcsinh', cmap='Greys')
      ax1.add_beam(major=beam_maj, minor=beam_min, angle=beam_pa, fill=True, color='peru')
    if subfig3 == 3 or subfig3 == 7 or subfig3 == 11 or subfig3 == 15:
      f1               = pyfits.open(image1)
      data, hdr        = f1[0].data, f1[0].header
      f2               = pyfits.open(image2)
      data2, hdr2      = f2[0].data, f2[0].header
      pixels           = np.max((hdr2['NAXIS1'], hdr2['NAXIS2']))
      arcmin_per_pixel = np.abs(hdr2['CDELT2']) * 60.
      box_side         = int(np.ceil(arcmin_per_pixel * pixels))
      freq_min = np.nanmean(data) - 2.5*madev(data, ignore_nan=True) #np.nanmin(data)
      freq_max = np.nanmean(data) + 2.5*madev(data, ignore_nan=True) #np.nanmax(data)
      ax1.show_colorscale(vmin=freq_min, vmax=freq_max, cmap='RdBu_r')
      try:
        if txtstr == 'J103729-261901':
          beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
        elif txtstr == 'J103726-261843':
          beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
        elif txtstr == 'J103406-270617':
          beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
        elif txtstr == 'J103542-284604':
          beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
        elif txtstr == 'J103545-284609':
          beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
        elif txtstr == 'J103537-284607':
          beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
        else:
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
      ax1.add_beam(major=beam_maj, minor=beam_min, angle=beam_pa, fill=True, color='peru')
    position     = SkyCoord(txt_array[9]*u.deg, txt_array[10]*u.deg, frame='icrs')
    width, height = box_side/60. - box_side/3./60., box_side/60. - box_side/3./60.
    ax1.recenter(position.ra, position.dec, width=width, height=height)
    if subfig3 == 1 or subfig3 == 5 or subfig3 == 9 or subfig3 == 13:
      ax1.add_label(0.4, 0.9, txtstr, relative=True)
    if subfig3 == 3 or subfig3 == 7 or subfig3 == 11 or subfig3 == 15:
      size_ratio = txt_array[7]/txt_array[11]
      ax1.add_label(3.3, 0.95, '%s (%i)' % (txtstr, txt_array[12]), relative=True)
      ax1.add_label(3.3, 0.83, r'$v_{\mathrm{sys}}=%.0f$' % txt_array[0], relative=True)
      ax1.add_label(3.3, 0.71, r'$\log(M_{\mathrm{HI}})=%.1f$' % txt_array[1], relative=True)
      ax1.add_label(3.3, 0.59, r'$\log(M_{\mathrm{*}})=%.1f$' % txt_array[2], relative=True)
      ax1.add_label(3.3, 0.47, r'$\log(\mathrm{SFR})=%.2f$' % txt_array[3], relative=True)
      #ax1.add_label(1.4, 0.15, r'$d_{\mathrm{HI,3\sigma}}=%.0f$' % txt_array[4], relative=True)
      ax1.add_label(3.3, 0.35, r'$d_{\mathrm{HI}}/d_{25}=%.0f/%.0f\,\,(%.1f)$' % (txt_array[7], txt_array[11], size_ratio), relative=True)
      #ax1.add_label(1.45, 0.23, r'$d_{25}=%.0f$' % txt_array[11], relative=True)
      #ax1.add_label(1.45, 0.23, r'$\mathrm{HI}/25=%.1f$' % size_ratio, relative=True)
      ax1.add_label(3.3, 0.23, r'$\mathrm{DEF}_{\mathrm{HI}}=%.2f$' % txt_array[13], relative=True)
    if subfig3 == 2 or subfig3 == 6 or subfig3 == 10 or subfig3 == 14:
      min_flux = 1. * 10**19 * (bmaj * bmin) / (2.33 * 10**20) #1.36*21*21*1.823*1000*(10**18))
      #lvls = np.array([5, 10, 20, 50, 70, 100, 130])
      lvls = np.array([5, 20, 50, 100])
      lvls = lvls*min_flux
      ax1.show_contour(image2, colors='darkred', levels=lvls, slices=(0,1))
      min_flux = 1. * (bmaj * bmin) / (2.12) #1.36*21*21*1.823*1000*(10**18))
      ax1.show_contour(image2, colors='red', linewidths=4, levels=[min_flux], slices=(0,1))
      #print(txt_array[4], txt_array[5], txt_array[6])
      #ax1.show_ellipses(position.ra, position.dec, width=txt_array[4]/3600., height=txt_array[5]/3600.,
                        #angle=txt_array[6], facecolor='none', edgecolor='violet', ls='-', zorder=2, linewidth=2,
                        #coords_frame='world')
      ax1.show_ellipses(position.ra, position.dec, width=txt_array[7]/3600., height=txt_array[8]/3600., 
                        angle=txt_array[6], facecolor='none', edgecolor='blue', ls='--', 
                        zorder=2, linewidth=3, coords_frame='world')
    if subfig3 == 1 or subfig3 == 5 or subfig3 == 9 or subfig3 == 13:
      dummy=0
    else:
      ax1.tick_labels.hide_y()
    ax1.axis_labels.hide_y()
    ax1.tick_labels.set_xformat('hh:mm:ss')
    ax1.tick_labels.set_yformat('dd:mm')
    ax1.ticks.show()
    ax1.ticks.set_xspacing(0.05)
    ax1.ticks.set_length(5)
    ax1.ticks.set_color('black')
    ax1.ticks.set_minor_frequency(1)
    plt.subplots_adjust(wspace=0.02, hspace=0.15)
  else:
    ax2 = fig_num.add_subplot(subfig1, subfig2, subfig3, facecolor = 'w')
    ax2.set_xlabel(r'Velocity [km\,s$^{-1}$]')
    ax2.set_ylabel(r'Flux [mJy]')
    velocity = image1[0]
    flux     = image1[1] * 1000.0
    error    = np.array(image1[2]) * 1000.0
    ax2.set_ylim(-1, np.nanmax(flux))
    ax2.set_xlim(np.nanmin(velocity)-10, np.nanmax(velocity)+10)
    ax2.axvline(txt_array[0], linewidth=0.75, linestyle = '--', color = 'darkgrey')
    ax2.plot(velocity, flux, linestyle = '-', color = 'darkblue', linewidth = 1.0)
    ax2.fill_between(velocity, flux-error, flux+error, alpha=0.5, edgecolor='none', facecolor='lightblue')
    ax2.tick_params(axis='both', direction='in')
    ax2.tick_params(axis='y', right=True, left=False, labelright=True, labelleft=False)
    ax2.yaxis.set_label_position('right')
    plt.subplots_adjust(wspace=0.02, hspace=0.15)
  #ax1.ticks.set_tick_direction('in')
  #ax1.tick_labels.set_style('plain')
  #plt.subplots_adjust(wspace=0.08, hspace=0.15)
  #f1.close()


# ================================ #
# ======= Opt/Contour Plot ======= #
# ================================ #
def fp_askap_opt_spec_plot(fig_num, subfig1, subfig2, subfig3, image1, image2, colour, txt_array):
  matplotlib.rcParams.update({'font.size': 15})
  rc('font',**{'family':'serif','serif':['Times']})
  rc('text', usetex=True)
  if type(image1) == str:
    ax1 = aplpy.FITSFigure(image1, figure=fig_num, subplot=(subfig1, subfig2, subfig3), dimensions=(0,1))# slices=(0,1))
    if subfig3 == 1 or subfig3 == 5 or subfig3 == 9 or subfig3 == 13:
      f1        = pyfits.open(image1)
      data1, hdr1 = f1[0].data, f1[0].header
      wcs         = WCS(hdr1)
      ax1.show_colorscale(vmin=5, pmax=99.75, stretch='arcsinh', cmap='Greys')
      f2        = pyfits.open(image2)
      data, hdr = f2[0].data, f2[0].header
      pixels        = np.max((hdr['NAXIS1'], hdr['NAXIS2'])) + 5
      arcmin_per_pixel = np.abs(hdr['CDELT2']) * 60.
      box_side      = int(np.ceil(arcmin_per_pixel * pixels))
    if subfig3 == 2 or subfig3 == 6 or subfig3 == 10 or subfig3 == 14:
      f2        = pyfits.open(image2)
      data, hdr = f2[0].data, f2[0].header
      pixels        = np.max((hdr['NAXIS1'], hdr['NAXIS2'])) + 5
      arcmin_per_pixel = np.abs(hdr['CDELT2']) * 60.
      box_side      = int(np.ceil(arcmin_per_pixel * pixels))
      try:
        if txt_array[0] == 'J103729-261901':
          beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
        elif txt_array[0] == 'J103726-261843':
          beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
        elif txt_array[0] == 'J103406-270617':
          beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
        elif txt_array[0] == 'J103542-284604':
          beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
        elif txt_array[0] == 'J103545-284609':
          beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
        elif txt_array[0] == 'J103537-284607':
          beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
        else:
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
      ax1.show_colorscale(vmin=5, pmax=99.75, stretch='arcsinh', cmap='Greys')
      ax1.add_beam(major=beam_maj, minor=beam_min, angle=beam_pa, fill=True, color='peru')
    if subfig3 == 3 or subfig3 == 7 or subfig3 == 11 or subfig3 == 15:
      f1               = pyfits.open(image1)
      data, hdr        = f1[0].data, f1[0].header
      f2               = pyfits.open(image2)
      data2, hdr2      = f2[0].data, f2[0].header
      pixels           = np.max((hdr2['NAXIS1'], hdr2['NAXIS2'])) + 5
      arcmin_per_pixel = np.abs(hdr2['CDELT2']) * 60.
      box_side         = int(np.ceil(arcmin_per_pixel * pixels))
      freq_min = np.nanmean(data) - 2.5*madev(data, ignore_nan=True) #np.nanmin(data)
      freq_max = np.nanmean(data) + 2.5*madev(data, ignore_nan=True) #np.nanmax(data)
      #ax1.show_colorscale(vmin=freq_min, vmax=freq_max, cmap='RdBu_r')
      ax1.show_colorscale(pmin=5, pmax=95, cmap='RdBu_r')
      try:
        if txt_array[0] == 'J103729-261901':
          beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
        elif txt_array[0] == 'J103726-261843':
          beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
        elif txt_array[0] == 'J103406-270617':
          beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
        elif txt_array[0] == 'J103542-284604':
          beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
        elif txt_array[0] == 'J103545-284609':
          beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
        elif txt_array[0] == 'J103537-284607':
          beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
        else:
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
      ax1.add_beam(major=beam_maj, minor=beam_min, angle=beam_pa, fill=True, color='peru')
    position      = SkyCoord(txt_array[3]*u.deg, txt_array[4]*u.deg, frame='icrs')
    width, height = box_side/60. - box_side/3./60., box_side/60. - box_side/3./60.
    ax1.recenter(position.ra, position.dec, width=width, height=height)
    #if subfig3 == 1 or subfig3 == 5 or subfig3 == 9 or subfig3 == 13:
      #ax1.add_label(0.4, 0.9, txtstr, relative=True)
    if subfig3 == 3 or subfig3 == 7 or subfig3 == 11 or subfig3 == 15:
      size_ratio = txt_array[6]/txt_array[9]
      size_ratio_r27 = txt_array[6]/txt_array[12]
      ax1.add_label(3.3, 0.95, '%s (%i)' % (txt_array[0], txt_array[1]), relative=True)
      ax1.add_label(3.3, 0.83, r'$v_{\mathrm{sys}}=%.0f$' % txt_array[2], relative=True)
      ax1.add_label(3.3, 0.71, r'$\log(M_{\mathrm{HI}})=%.1f$' % txt_array[5], relative=True)
      ax1.add_label(3.3, 0.59, r'$\log(M_{\mathrm{*}})=%.1f$' % txt_array[11], relative=True)
      #ax1.add_label(3.3, 0.47, r'$\log(\mathrm{SFR})=%.2f$' % txt_array[3], relative=True)
      ##ax1.add_label(1.4, 0.15, r'$d_{\mathrm{HI,3\sigma}}=%.0f$' % txt_array[4], relative=True)
      ax1.add_label(3.3, 0.47, r'$d_{\mathrm{HI}}/d_{b25}=%.0f/%.0f\,\,(%.1f)$' % (txt_array[6], txt_array[9], size_ratio), relative=True)
      ax1.add_label(3.3, 0.35, r'$d_{\mathrm{HI}}/d_{r23}=%.0f/%.0f\,\,(%.1f)$' % (txt_array[6], txt_array[12], size_ratio_r27), relative=True)
      ##ax1.add_label(1.45, 0.23, r'$d_{25}=%.0f$' % txt_array[11], relative=True)
      ##ax1.add_label(1.45, 0.23, r'$\mathrm{HI}/25=%.1f$' % size_ratio, relative=True)
      ax1.add_label(3.3, 0.23, r'$\mathrm{DEF}_{\mathrm{HI}}=%.2f$' % txt_array[10], relative=True)
    if subfig3 == 2 or subfig3 == 6 or subfig3 == 10 or subfig3 == 14:
      min_flux = 1. * 10**19 * (bmaj * bmin) * math.pi / 4 / np.log(2) / (2.33 * 10**20) #1.36*21*21*1.823*1000*(10**18))
      #lvls = np.array([5, 10, 20, 50, 70, 100, 130])
      lvls = np.array([5, 20, 50, 100])
      lvls = lvls*min_flux
      ax1.show_contour(image2, colors='darkred', levels=lvls, slices=(0,1))
      min_flux = 1. * (bmaj * bmin) * math.pi / 4 / np.log(2) / (2.12) / np.cos(np.arcsin(np.sqrt(1-(txt_array[7]/txt_array[6])**2))) #1.36*21*21*1.823*1000*(10**18))
      #print(min_flux)
      ax1.show_contour(image2, colors='red', linewidths=4, levels=[min_flux], slices=(0,1))
      #min_flux = 1. * (bmaj * bmin) / (2.12) #1.36*21*21*1.823*1000*(10**18))
      #print(min_flux)
      #ax1.show_contour(image2, colors='green', linewidths=4, levels=[min_flux], slices=(0,1))
      #print(txt_array[4], txt_array[5], txt_array[6])
      #ax1.show_ellipses(position.ra, position.dec, width=txt_array[4]/3600., height=txt_array[5]/3600.,
                        #angle=txt_array[6], facecolor='none', edgecolor='violet', ls='-', zorder=2, linewidth=2,
                        #coords_frame='world')
      ax1.show_ellipses(position.ra, position.dec, width=txt_array[6]/3600., height=txt_array[7]/3600., 
                        angle=txt_array[8], facecolor='none', edgecolor='blue', ls='--', 
                        zorder=2, linewidth=3, coords_frame='world')
    if subfig3 == 1 or subfig3 == 5 or subfig3 == 9 or subfig3 == 13:
      racen, deccen = wcs.all_pix2world(txt_array[14], txt_array[15], 0)
      pos_ps        = SkyCoord(racen*u.deg, deccen*u.deg, frame='icrs')
      ax1.show_ellipses(pos_ps.ra, pos_ps.dec, width=txt_array[12]/3600., height=txt_array[13]/3600., 
                        angle=txt_array[16], facecolor='none', edgecolor='magenta', ls='-', 
                        zorder=2, linewidth=3, coords_frame='world')
    if subfig3 == 1 or subfig3 == 5 or subfig3 == 9 or subfig3 == 13:
      dummy=0
    else:
      ax1.tick_labels.hide_y()
    ax1.axis_labels.hide_y()
    ax1.tick_labels.set_xformat('hh:mm:ss')
    ax1.tick_labels.set_yformat('dd:mm')
    ax1.ticks.show()
    ax1.ticks.set_xspacing(0.05)
    ax1.ticks.set_length(5)
    ax1.ticks.set_color('black')
    ax1.ticks.set_minor_frequency(1)
    plt.subplots_adjust(wspace=0.02, hspace=0.15)
  else:
    ax2 = fig_num.add_subplot(subfig1, subfig2, subfig3, facecolor = 'w')
    ax2.set_xlabel(r'Velocity [km\,s$^{-1}$]')
    ax2.set_ylabel(r'Flux [mJy]')
    velocity = image1[0]
    flux     = image1[1] * 1000.0
    error    = np.array(image1[2]) * 1000.0
    ax2.set_ylim(-1, np.nanmax(flux))
    ax2.set_xlim(np.nanmin(velocity)-10, np.nanmax(velocity)+10)
    ax2.axvline(txt_array[2], linewidth=0.75, linestyle = '--', color = 'darkgrey')
    ax2.plot(velocity, flux, linestyle = '-', color = 'darkblue', linewidth = 1.0)
    ax2.fill_between(velocity, flux-error, flux+error, alpha=0.5, edgecolor='none', facecolor='lightblue')
    ax2.tick_params(axis='both', direction='in')
    ax2.tick_params(axis='y', right=True, left=False, labelright=True, labelleft=False)
    ax2.yaxis.set_label_position('right')
    plt.subplots_adjust(wspace=0.02, hspace=0.15)
  #ax1.ticks.set_tick_direction('in')
  #ax1.tick_labels.set_style('plain')
  #plt.subplots_adjust(wspace=0.08, hspace=0.15)
  #f1.close()
  
  
# ================================ #
# ======= Opt/Contour Plot ======= #
# ================================ #
def single_askap_map(fig_num, subfig1, subfig2, subfig3, image1, image2, colour, txt_array):
  matplotlib.rcParams.update({'font.size': 15})
  rc('font',**{'family':'serif','serif':['Times']})
  rc('text', usetex=True)
  ax1 = aplpy.FITSFigure(image1, figure=fig_num, subplot=(subfig1, subfig2, subfig3), dimensions=(0,1))# slices=(0,1))
    
  f2        = pyfits.open(image2)
  data, hdr = f2[0].data, f2[0].header
  pixels        = np.max((hdr['NAXIS1'], hdr['NAXIS2'])) + 5
  arcmin_per_pixel = np.abs(hdr['CDELT2']) * 60.
  box_side      = int(np.ceil(arcmin_per_pixel * pixels))
  try:
    if txt_array[0] == 'J103729-261901':
      beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
    elif txt_array[0] == 'J103726-261843':
      beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
    elif txt_array[0] == 'J103406-270617':
      beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
    elif txt_array[0] == 'J103542-284604':
      beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
    elif txt_array[0] == 'J103545-284609':
      beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
    elif txt_array[0] == 'J103537-284607':
      beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
    else:
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
  ax1.show_colorscale(vmin=5, pmax=99.75, stretch='arcsinh', cmap='Greys')
  ax1.add_beam(major=beam_maj, minor=beam_min, angle=beam_pa, fill=True, color='peru')
  
  position      = SkyCoord(txt_array[1]*u.deg, txt_array[2]*u.deg, frame='icrs')
  width, height = box_side/60. - box_side/4./60., box_side/60. - box_side/4./60.
  ax1.recenter(position.ra, position.dec, width=width, height=height)
  
  min_flux = 1. * 10**19 * (bmaj * bmin) * math.pi / 4 / np.log(2) / (2.33 * 10**20) #1.36*21*21*1.823*1000*(10**18))
  #lvls = np.array([5, 10, 20, 50, 70, 100, 130])
  lvls = np.array([5, 20, 50, 100])
  lvls = lvls*min_flux
  ax1.show_contour(image2, colors='darkred', levels=lvls, slices=(0,1))
  min_flux = 1. * (bmaj * bmin) * math.pi / 4 / np.log(2) / (2.12) / np.cos(np.arcsin(np.sqrt(1-(txt_array[3])**2))) #1.36*21*21*1.823*1000*(10**18))
  
  ax1.show_contour(image2, colors='red', linewidths=4, levels=[min_flux], slices=(0,1))
  
  ax1.add_label(0.35, 0.9, '%.2f\t%.2f\t%.2f' % (txt_array[4], txt_array[5], txt_array[6]), relative=True)
  ax1.add_label(-1, 0.9, '%.2f' % (txt_array[7]), relative=True)
  ax1.add_label(-1, 0.75, '%.2f' % (txt_array[8]), relative=True)
  ax1.add_label(-1, 0.6, '%.2f' % (txt_array[9]), relative=True)
  ax1.add_label(-1, 0.45, '%.2f' % (txt_array[10]), relative=True)
  
  ax1.tick_labels.hide_y()
  ax1.axis_labels.hide_y()
  ax1.tick_labels.set_xformat('hh:mm:ss')
  ax1.tick_labels.set_yformat('dd:mm')
  ax1.ticks.show()
  ax1.ticks.set_xspacing(0.05)
  ax1.ticks.set_length(5)
  ax1.ticks.set_color('black')
  ax1.ticks.set_minor_frequency(1)
  plt.subplots_adjust(wspace=0.02, hspace=0.15)
  

# ================================ #
# ======= Opt/Contour Plot ======= #
# ================================ #
def galaxy_plate_orig_plot(fig_num, sfig, image1, image2, txt_array):
  matplotlib.rcParams.update({'font.size': 12})
  rc('font',**{'family':'serif','serif':['Times']})
  rc('text', usetex=True)
  if type(image1) == str:
    #ax1 = aplpy.FITSFigure(image1, figure=fig_num, subplot=(sfig[0], sfig[1], sfig[2]), dimensions=(0,1))# slices=(0,1))
    ax1 = aplpy.FITSFigure(image1, figure=fig_num, subplot=[sfig[3], sfig[4], sfig[5], sfig[6]], dimensions=(0,1))
    f1        = pyfits.open(image1)
    data1, hdr1 = f1[0].data, f1[0].header
    wcs         = WCS(hdr1)
    f2        = pyfits.open(image2)
    data, hdr = f2[0].data, f2[0].header
    pixels        = np.max((hdr['NAXIS1'], hdr['NAXIS2'])) + 5
    arcmin_per_pixel = np.abs(hdr['CDELT2']) * 60.
    box_side      = int(np.ceil(arcmin_per_pixel * pixels))
    try:
      if txt_array[0] == 'J103729-261901':
        beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
      elif txt_array[0] == 'J103726-261843':
        beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
      elif txt_array[0] == 'J103406-270617':
        beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
      elif txt_array[0] == 'J103542-284604':
        beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
      elif txt_array[0] == 'J103545-284609':
        beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
      elif txt_array[0] == 'J103537-284607':
        beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
      else:
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
    #ax1.show_colorscale(vmin=5, pmax=99.75, stretch='arcsinh', cmap='Greys')
    if txt_array[13] == 'Cluster':
      beam_col = 'darkblue'
    if txt_array[13] == 'Infall':
      beam_col = 'magenta'
    if txt_array[13] == 'Field':
      beam_col = 'peru'
    ax1.add_beam(major=beam_maj, minor=beam_min, angle=beam_pa, fill=True, color=beam_col)
    ax1.beam.set_corner('top right')
    pos_ra  = txt_array[1]
    pos_dec = txt_array[2]
    position      = SkyCoord(pos_ra*u.deg, pos_dec*u.deg, frame='icrs')
    width, height = box_side/60. - box_side/3./60., box_side/60. - box_side/3./60.
    ax1.recenter(position.ra, position.dec, width=width, height=height)
    
    
    if sfig[0] == 1 and sfig[1] == 2:
      min_flux = 1. * 10**19 * (bmaj * bmin) * math.pi / 4 / np.log(2) / (2.33 * 10**20) 
      lvls = np.array([5, 20, 50, 100])
      lvls = lvls*min_flux
      ax1.show_contour(image2, colors='violet', levels=lvls, slices=(0,1))
    hi_major = txt_array[3]
    hi_minor = txt_array[4]
    min_flux = (1. * (bmaj * bmin) * math.pi / 4 / np.log(2) / (2.12) / 
                np.cos(np.arcsin(np.sqrt(1 - (hi_minor / hi_major)**2))))
    
    ax1.show_contour(image2, colors='purple', linewidths=2, levels=[min_flux], slices=(0,1))
    hi_width  = txt_array[3] / 3600.
    hi_height = txt_array[4] / 3600.
    hi_angle  = txt_array[5]
    #ax1.show_ellipses(position.ra, position.dec, width=hi_width, height=hi_height, 
                      #angle=hi_angle, facecolor='none', edgecolor='blue', ls='--', 
                      #zorder=2, linewidth=1.5, coords_frame='world')
    #if sfig[2] == 1 or sfig[2] == 5 or sfig[2] == 9 or sfig[2] == 13:
    if sfig[2] % 2 != 0:
      ax1.show_colorscale(vmin=5, pmax=99.75, stretch='arcsinh', cmap='Greys')
      ax1.add_label(0.4, 0.9, r'\bf{%s}' % (txt_array[0]), relative=True)
      #ax1.add_label(0.8, 0.05, '%s' % (txt_array[13]), relative=True)
      if sfig[2] < 7:
        ax1.add_label(0.5, 1.05, r'$r$-band', relative=True)
      #opt_ra_pix  = txt_array[6]
      #opt_dec_pix = txt_array[7]
      #racen, deccen = wcs.all_pix2world(opt_ra_pix, opt_dec_pix, 0)
      racen  = txt_array[6]
      deccen = txt_array[7]
      pos_ps        = SkyCoord(racen*u.deg, deccen*u.deg, frame='icrs')
      opt_width  = txt_array[9] / 3600.
      opt_height = txt_array[10] / 3600.
      #print(opt_width, opt_height)
      opt_angle  = txt_array[8]
      ax1.show_ellipses(pos_ps.ra, pos_ps.dec, width=opt_width, height=opt_height, 
                        angle=opt_angle, facecolor='none', edgecolor='red', ls='-', 
                        zorder=2, linewidth=2, coords_frame='world')
      nuv_width  = txt_array[11] / 3600.
      nuv_height = txt_array[12] / 3600.
      #print(nuv_width, nuv_height)
      nuv_angle  = txt_array[8]
      ax1.show_ellipses(pos_ps.ra, pos_ps.dec, width=nuv_width, height=nuv_height, 
                        angle=nuv_angle, facecolor='none', edgecolor='blue', ls='--', 
                        zorder=2, linewidth=1.5, coords_frame='world')
      ax1.add_label(0.75, 0.05, '%f' % txt_array[14], relative=True)
    #if sfig[2] == 2 or sfig[2] == 6 or sfig[2] == 10 or sfig[2] == 14:
    if sfig[2] % 2 == 0:
      ax1.show_colorscale(pmin=50, pmax=99.75, stretch='arcsinh', cmap='Greys')
      if sfig[2] < 7:
        ax1.add_label(0.5, 1.05, 'NUV', relative=True)
      #opt_ra_pix  = txt_array[6]
      #opt_dec_pix = txt_array[7]
      #racen, deccen = wcs.all_pix2world(opt_ra_pix, opt_dec_pix, 0)
      racen  = txt_array[6]
      deccen = txt_array[7]
      pos_ps        = SkyCoord(racen*u.deg, deccen*u.deg, frame='icrs')
      nuv_width  = txt_array[11] / 3600.
      nuv_height = txt_array[12] / 3600.
      #print(nuv_width, nuv_height)
      nuv_angle  = txt_array[8]
      ax1.show_ellipses(pos_ps.ra, pos_ps.dec, width=nuv_width, height=nuv_height, 
                        angle=nuv_angle, facecolor='none', edgecolor='blue', ls='--', 
                        zorder=2, linewidth=1.5, coords_frame='world')
    if sfig[2] == 1 or sfig[2] == 7 or sfig[2] == 13:
      ax1.add_label(0.05, 0.05, r'\textbf{%s}' % (txt_array[13]), color=beam_col, fontsize=16, relative=True)
    #if sfig[2] == 1 or sfig[2] == 5 or sfig[2] == 9 or sfig[2] == 13:
    #if sfig[2] % 2 != 0:
      #dummy=0
    #else:
    #if (sfig[2] == 1 or sfig[2] == 2 or sfig[2] == 7 or sfig[2] == 8 or sfig[2] == 13 or sfig[2] == 14 or 
        #sfig[2] == 5 or sfig[2] == 6 or sfig[2] == 11 or sfig[2] == 12 or sfig[2] == 17 or sfig[2] == 18):
      #ax1.frame.set_linewidth(2)
    #if txt_array[13] == 'Cluster':
      #ax1.frame.set_linewidth(2.5)
      #ax1.frame.set_color('darkblue')
    #if txt_array[13] == 'Infall':
      #ax1.frame.set_linewidth(2.5)
      #ax1.frame.set_color('magenta')
    #if txt_array[13] == 'Field':
      #ax1.frame.set_linewidth(2.5)
      #ax1.frame.set_color('peru')
    ax1.tick_labels.hide_y()
    ax1.axis_labels.hide_y()
    ax1.tick_labels.hide_x()
    ax1.axis_labels.hide_x()
    #ax1.tick_labels.set_xformat('hh:mm:ss')
    #ax1.tick_labels.set_yformat('dd:mm')
    ax1.ticks.hide()
    #ax1.ticks.set_xspacing(0.05)
    #ax1.ticks.set_length(5)
    #ax1.ticks.set_color('black')
    #ax1.ticks.set_minor_frequency(1)
    plt.subplots_adjust(wspace=0.1, hspace=0.05)


# ================================ #
# ======= Opt/Contour Plot ======= #
# ================================ #
def galaxy_plate_plot(fig_num, sfig, image1, image2, txt_array):
  matplotlib.rcParams.update({'font.size': 12})
  rc('font',**{'family':'serif','serif':['Times']})
  rc('text', usetex=True)
  if type(image1) == str:
    #ax1 = aplpy.FITSFigure(image1, figure=fig_num, subplot=(sfig[0], sfig[1], sfig[2]), dimensions=(0,1))# slices=(0,1))
    ax1 = aplpy.FITSFigure(image1, figure=fig_num, subplot=[sfig[3], sfig[4], sfig[5], sfig[6]], dimensions=(0,1))
    f1        = pyfits.open(image1)
    data1, hdr1 = f1[0].data, f1[0].header
    wcs         = WCS(hdr1)
    f2        = pyfits.open(image2)
    data, hdr = f2[0].data, f2[0].header
    pixels        = np.max((hdr['NAXIS1'], hdr['NAXIS2'])) + 5
    arcmin_per_pixel = np.abs(hdr['CDELT2']) * 60.
    box_side      = int(np.ceil(arcmin_per_pixel * pixels))
    try:
      if txt_array[0] == 'J103729-261901':
        beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
      elif txt_array[0] == 'J103726-261843':
        beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
      elif txt_array[0] == 'J103406-270617':
        beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
      elif txt_array[0] == 'J103542-284604':
        beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
      elif txt_array[0] == 'J103545-284609':
        beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
      elif txt_array[0] == 'J103537-284607':
        beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
      else:
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
    #ax1.show_colorscale(vmin=5, pmax=99.75, stretch='arcsinh', cmap='Greys')
    if txt_array[13] == 'Cluster':
      beam_col = 'darkblue'
    if txt_array[13] == 'Infall':
      beam_col = 'mediumvioletred'
    if txt_array[13] == 'Field':
      beam_col = 'peru'
    ax1.add_beam(major=beam_maj, minor=beam_min, angle=beam_pa, fill=True, color=beam_col)
    ax1.beam.set_corner('top right')
    pos_ra  = txt_array[1]
    pos_dec = txt_array[2]
    position      = SkyCoord(pos_ra*u.deg, pos_dec*u.deg, frame='icrs')
    width, height = box_side/60. - box_side/3./60., box_side/60. - box_side/3./60.
    ax1.recenter(position.ra, position.dec, width=width, height=height)
    
    
    if sfig[0] == 1 and sfig[1] == 2:
      min_flux = 1. * 10**19 * (bmaj * bmin) * math.pi / 4 / np.log(2) / (2.33 * 10**20) 
      lvls = np.array([5, 20, 50, 100])
      lvls = lvls*min_flux
      ax1.show_contour(image2, colors='violet', levels=lvls, slices=(0,1))
    hi_major = txt_array[3]
    hi_minor = txt_array[4]
    min_flux = (1. * (bmaj * bmin) * math.pi / 4 / np.log(2) / (2.12) / 
                np.cos(np.arcsin(np.sqrt(1 - (hi_minor / hi_major)**2))))
    
    ax1.show_contour(image2, colors='purple', linewidths=2, levels=[min_flux], slices=(0,1))
    hi_width  = txt_array[3] / 3600.
    hi_height = txt_array[4] / 3600.
    hi_angle  = txt_array[5]
    #ax1.show_ellipses(position.ra, position.dec, width=hi_width, height=hi_height, 
                      #angle=hi_angle, facecolor='none', edgecolor='blue', ls='--', 
                      #zorder=2, linewidth=1.5, coords_frame='world')
    #if sfig[2] == 1 or sfig[2] == 5 or sfig[2] == 9 or sfig[2] == 13:
    if sfig[2] % 2 != 0:
      ax1.show_colorscale(vmin=5, pmax=99.75, stretch='arcsinh', cmap='Greys')
      ax1.add_label(0.4, 0.9, r'\bf{%s}' % (txt_array[0]), relative=True)
      #ax1.add_label(0.8, 0.05, '%s' % (txt_array[13]), relative=True)
      if sfig[2] < 7:
        ax1.add_label(0.5, 1.05, r'$r$-band', relative=True)
      #opt_ra_pix  = txt_array[6]
      #opt_dec_pix = txt_array[7]
      #racen, deccen = wcs.all_pix2world(opt_ra_pix, opt_dec_pix, 0)
      racen  = txt_array[6]
      deccen = txt_array[7]
      pos_ps        = SkyCoord(racen*u.deg, deccen*u.deg, frame='icrs')
      opt_width  = txt_array[9] / 3600.
      opt_height = txt_array[10] / 3600.
      #print(opt_width, opt_height)
      opt_angle  = txt_array[8]
      ax1.show_ellipses(pos_ps.ra, pos_ps.dec, width=opt_width, height=opt_height, 
                        angle=opt_angle, facecolor='none', edgecolor='red', ls='-', 
                        zorder=2, linewidth=2, coords_frame='world')
      nuv_width  = txt_array[11] / 3600.
      nuv_height = txt_array[12] / 3600.
      #print(nuv_width, nuv_height)
      nuv_angle  = txt_array[8]
      #ax1.show_ellipses(pos_ps.ra, pos_ps.dec, width=nuv_width, height=nuv_height, 
                        #angle=nuv_angle, facecolor='none', edgecolor='blue', ls='--', 
                        #zorder=2, linewidth=1.5, coords_frame='world')
      ax1.add_label(0.9, 0.05, r'\bf{%0.1f}' % np.round(txt_array[14],1), fontsize=16, relative=True)
    #if sfig[2] == 2 or sfig[2] == 6 or sfig[2] == 10 or sfig[2] == 14:
    if sfig[2] % 2 == 0:
      ax1.show_colorscale(pmin=60, pmax=99.8, stretch='arcsinh', smooth=3, cmap='Greys')
      if sfig[2] < 7:
        ax1.add_label(0.5, 1.05, 'NUV', relative=True)
      #opt_ra_pix  = txt_array[6]
      #opt_dec_pix = txt_array[7]
      #racen, deccen = wcs.all_pix2world(opt_ra_pix, opt_dec_pix, 0)
      racen  = txt_array[6]
      deccen = txt_array[7]
      pos_ps        = SkyCoord(racen*u.deg, deccen*u.deg, frame='icrs')
      nuv_width  = txt_array[11] / 3600.
      nuv_height = txt_array[12] / 3600.
      #print(nuv_width, nuv_height)
      nuv_angle  = txt_array[8]
      ax1.show_ellipses(pos_ps.ra, pos_ps.dec, width=nuv_width, height=nuv_height, 
                        angle=nuv_angle, facecolor='none', edgecolor='blue', ls='--', 
                        zorder=2, linewidth=1.5, coords_frame='world')
    if sfig[2] == 1 :
      ax1.add_label(0.2, 0.05, r'\textbf{%s}' % (txt_array[13]), color=beam_col, fontsize=16, relative=True)
    if sfig[2] == 7:
      ax1.add_label(0.2, 0.05, r'\textbf{%s}' % (txt_array[13]), color=beam_col, fontsize=16, relative=True)
    if sfig[2] == 13:
      ax1.add_label(0.25, 0.05, r'\textbf{%s}' % (txt_array[13]), color=beam_col, fontsize=16, relative=True)
    ax1.tick_labels.hide_y()
    ax1.axis_labels.hide_y()
    ax1.tick_labels.hide_x()
    ax1.axis_labels.hide_x()
    ax1.ticks.hide()
    plt.subplots_adjust(wspace=0.1, hspace=0.05)
  

# ================================ #
# ====== 6dF Panstarrs Plot ====== #
# ================================ #
def panstarrs_6df_plot(fig_num, subfig1, subfig2, subfig3, image1, txt_array):
  matplotlib.rcParams.update({'font.size': 15})
  rc('font',**{'family':'serif','serif':['Times']})
  rc('text', usetex=True)
  if type(image1) == str:
    ax1 = aplpy.FITSFigure(image1, figure=fig_num, subplot=(subfig1, subfig2, subfig3), dimensions=(0,1))# slices=(0,1))
    f1          = pyfits.open(image1)
    data1, hdr1 = f1[0].data, f1[0].header
    wcs         = WCS(hdr1)
    ax1.show_colorscale(vmin=5, pmax=99.75, stretch='arcsinh', cmap='Greys')
    ax1.add_label(0.4, 0.9, '%s' % (txt_array[0]), relative=True)
    ax1.add_label(0.25, 0.8, r'$v_{\mathrm{sys}}=%.0f$' % txt_array[2], relative=True)
    ax1.add_label(0.2, 0.05, r'$d_{r23}=%.0f$' % (txt_array[6]), relative=True)
    ax1.add_label(0.9, 0.05, '%i' % (txt_array[1]), relative=True)
    if txt_array[10] == 1:
      gtype = 'L'
    elif txt_array[10] == 2:
      gtype = 'E'
    elif txt_array[10] == 3:
      gtype = 'S0'
    else:
      gtype = 'U'
    ax1.add_label(0.9, 0.15, '%s' % (gtype), relative=True)
    #racen, deccen = wcs.all_pix2world(txt_array[3], txt_array[4], 0)
    #pos_ps        = SkyCoord(racen*u.deg, deccen*u.deg, frame='icrs')
    pos_ps        = SkyCoord(txt_array[3]*u.deg, txt_array[4]*u.deg, frame='icrs')
    ax1.show_ellipses(pos_ps.ra, pos_ps.dec, width=txt_array[6]/3600., height=txt_array[7]/3600., 
                      angle=txt_array[5], facecolor='none', edgecolor='magenta', ls='-', 
                      zorder=2, linewidth=1.5, coords_frame='world')
    ax1.axis_labels.hide_y()
    ax1.axis_labels.hide_x()
    ax1.tick_labels.set_xformat('hh:mm:ss')
    ax1.tick_labels.set_yformat('dd:mm')
    ax1.ticks.show()
    ax1.ticks.set_xspacing(0.05)
    ax1.ticks.set_length(5)
    ax1.ticks.set_color('black')
    ax1.ticks.set_minor_frequency(1)
    plt.subplots_adjust(wspace=0.02, hspace=0.15)


# ================================ #
# ====== Opt Residual Plot ======= #
# ================================ #
def opt_resdiual_plot(fig_num, subfig1, subfig2, subfig3, image1):
  matplotlib.rcParams.update({'font.size': 12})
  rc('font',**{'family':'serif','serif':['Times']})
  rc('text', usetex=True)
  if type(image1) == str:
    ax1 = aplpy.FITSFigure(image1, figure=fig_num, subplot=(subfig1, subfig2, subfig3), dimensions=(0,1))# slices=(0,1))
    f1        = pyfits.open(image1)
    ax1.show_colorscale(pmax=99.85, stretch='arcsinh', cmap='Greys') #pmin=0.5, 
    ax1.axis_labels.hide_y()
    ax1.axis_labels.hide_x()
    ax1.tick_labels.set_xformat('hh:mm:ss')
    ax1.tick_labels.set_yformat('dd:mm')
    ax1.ticks.show()
    ax1.ticks.set_xspacing(0.05)
    ax1.ticks.set_length(5)
    ax1.ticks.set_color('black')
    ax1.ticks.set_minor_frequency(1)
    plt.subplots_adjust(wspace=0.25, hspace=0.15)



# ================================ #
# ======= Opt/Contour Plot ======= #
# ================================ #
def askap_opt_resolved(fig_num, subfig1, subfig2, subfig3, image1, image2, colour, txtstr, txt_array):
  matplotlib.rcParams.update({'font.size': 15})
  rc('font',**{'family':'serif','serif':['Times']})
  rc('text', usetex=True)
  f1        = pyfits.open(image1)
  hdr1 = f1[0].header
  ax1 = aplpy.FITSFigure(image1, figure=fig_num, subplot=(subfig1, subfig2, subfig3), dimensions=(0,1))
  f2        = pyfits.open(image2)
  data, hdr = f2[0].data, f2[0].header
  pixels        = np.max((hdr['NAXIS1'], hdr['NAXIS2']))
  arcmin_per_pixel = np.abs(hdr['CDELT2']) * 60.
  box_side      = int(np.ceil(arcmin_per_pixel * pixels))
  try:
    if txtstr == 'J103729-261901':
      beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
    elif txtstr == 'J103726-261843':
      beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
    elif txtstr == 'J103406-270617':
      beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
    elif txtstr == 'J103542-284604':
      beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
    elif txtstr == 'J103545-284609':
      beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
    elif txtstr == 'J103537-284607':
      beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
    else:
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
  ax1.show_colorscale(vmin=5, pmax=99.75, stretch='arcsinh', cmap='Greys')
  ax1.add_beam(major=beam_maj, minor=beam_min, angle=beam_pa, fill=True, color='peru')
  position     = SkyCoord(txt_array[0]*u.deg, txt_array[1]*u.deg, frame='icrs')
  if txtstr == 'J103702-273359' or txtstr == 'J104059-270456':
    width, height = box_side/60. - box_side/5./60., box_side/60. - box_side/5./60.
  else:
    width, height = box_side/60. - box_side/3.5/60., box_side/60. - box_side/3.5/60.
  ax1.recenter(position.ra, position.dec, width=width, height=height)
  ax1.add_label(0.35, 0.9, txtstr, relative=True, fontsize = 12)
  if txt_array[4] == 0.6:
    txt_array[4] = 0.5
  ax1.add_label(0.85, 0.9, '(%.1f)' % txt_array[4], relative=True, fontsize = 12)
  min_flux = 1. * 10**19 * (bmaj * bmin) / (2.33 * 10**20) #1.36*21*21*1.823*1000*(10**18))
  #lvls = np.array([5, 10, 20, 50, 70, 100, 130])
  lvls = np.array([5, 20, 50])#, 100])
  lvls = lvls*min_flux
  ax1.show_contour(image2, colors='blue', levels=lvls, slices=(0,1))
  if txtstr == 'J103702-273359':
    xw = 7. * hdr1['NAXIS2'] / 10.
    yw = 3.1 * hdr1['NAXIS1'] / 10.
    fc = 'none'
  elif txtstr == 'J104059-270456':
    xw = 8. * hdr1['NAXIS2'] / 10.
    yw = 1.75 * hdr1['NAXIS1'] / 10.
    fc = txt_array[3]
  elif txtstr == 'J103335-272717':
    xw = 8.5 * hdr1['NAXIS2'] / 10.
    yw = 1.25 * hdr1['NAXIS1'] / 10.
    fc = 'none'
  elif txtstr == 'J103523-281855':
    xw = 8.5 * hdr1['NAXIS2'] / 10.
    yw = 1.25 * hdr1['NAXIS1'] / 10.
    fc = 'none'
  else:
    xw = 9. * hdr1['NAXIS2'] / 10.
    yw = hdr1['NAXIS1'] / 10.
    fc = 'none'
  print(xw, yw, txt_array[2], txt_array[3])
  ax1.show_markers(xw, yw, marker = txt_array[2], s=150, 
                   edgecolor = txt_array[3], facecolor = fc, linewidth = 3, coords_frame='pixel')
  #min_flux = 1. * (bmaj * bmin) / (2.12) #1.36*21*21*1.823*1000*(10**18))
  #ax1.show_contour(image2, colors='red', linewidths=4, levels=[min_flux], slices=(0,1))
  #if subfig3 == 1 or subfig3 == 5 or subfig3 == 9 or subfig3 == 13:
    #dummy=0
  #else:
  ax1.tick_labels.hide_y()
  ax1.axis_labels.hide_y()
  ax1.tick_labels.hide_x()
  ax1.axis_labels.hide_x()
  #ax1.tick_labels.set_xformat('hh:mm:ss')
  #ax1.tick_labels.set_yformat('dd:mm')
  ax1.ticks.hide()
  #ax1.ticks.set_xspacing(0.05)
  #ax1.ticks.set_length(5)
  #ax1.ticks.set_color('black')
  #ax1.ticks.set_minor_frequency(1)
  plt.subplots_adjust(wspace=0.02, hspace=0.02)
  #ax1.ticks.set_tick_direction('in')
  #ax1.tick_labels.set_style('plain')
  #plt.subplots_adjust(wspace=0.08, hspace=0.15)
  #f1.close()


# ================================ #
# ======= Opt/Contour Plot ======= #
# ================================ #
def opt_mom0_plot(fig_num, subfig1, subfig2, subfig3, image1, image2, colour, txtstr):
    matplotlib.rcParams.update({'font.size': 16})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    if subfig3 == 1:
      subfig_param = [0.05,0.5,0.9,0.45]
    if subfig3 == 2:
      subfig_param = [0.05,0.05,0.9,0.45]
    ax1 = aplpy.FITSFigure(image1, figure=fig_num, subplot=subfig_param, dimensions=(0,1))# slices=(0,1))
    #if subfig3 == 1:
    #f1        = pyfits.open(image1)
    #print(f1[0].header)
    f2        = pyfits.open(image2)
    data, hdr = f2[0].data, f2[0].header
    #print(f2[0].header)
    #if txtstr == 'J103729-261901':
      #print(hdr)
    pixels        = np.max((hdr['NAXIS1'], hdr['NAXIS2']))
    arcmin_per_pixel = np.abs(hdr['CDELT2']) * 60.
    box_side      = int(np.ceil(arcmin_per_pixel * pixels))
    try:
      if txtstr == 'J103729-261901':
        beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
      else:
        beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], hdr['BPA']
      #print(beam_maj, beam_min)
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
    bmaj, bmin  = beam_maj*60.*60., beam_min*60.*60.
    f1          = pyfits.open(image1)
    data1, hdr1 = f1[0].data, f1[0].header
    ax1.show_colorscale(vmin=0, vmax=1200, cmap='Greys')
    ax1.add_beam(major=beam_maj, minor=beam_min, angle=beam_pa, fill=True, color='peru')
    ra  = '%sh%sm%ss' % (txtstr[1:3], txtstr[3:5], txtstr[5:7])
    dec = '%sd%sm%ss' % (txtstr[7:10], txtstr[10:12], txtstr[12:])
    position     = SkyCoord(ra, dec, frame='icrs')
    width, height = box_side/60., box_side/60.
    ax1.recenter(position.ra, position.dec, width=width, height=height)
    if subfig3 == 1:
      print(image2)
      min_flux = 1. * 10**19 * (bmaj * bmin) / (2.33 * 10**20) #1.36*21*21*1.823*1000*(10**18))
      lvls = np.array([5, 10, 20, 50, 70])#, 100, 130])
      lvls = lvls*min_flux
      ax1.show_contour(image2, colors="blue", levels=lvls, slices=(0,1))
    if subfig3 == 2:
      #print(image2)
      #vel_range = 400
      #lvls = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
      ##lvls = lvls*vel_range/10 + 4740
      #lvls = 4940 - lvls*vel_range/10
      #lvls = HI_REST * 1e6 / (lvls / C_LIGHT + 1.)
      ##print(lvls)
      #ax1.show_contour(image2, colors='blue', levels=lvls)
      #vel_range = 132
      vel_sys   = 4740
      #lvls = np.flip(np.array([-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
      lvls = np.flip(np.array([-8, -6, -4, -2, 0, 2, 4, 6, 8, 10]))
      lvls = lvls*20 + vel_sys
      print(lvls)
      lvls = HI_REST * 1e6 / (lvls / C_LIGHT + 1.)
      vel_sys   = HI_REST * 1e6 / (vel_sys / C_LIGHT + 1.)
      ax1.show_contour(data=image2, colors="blue", levels=[vel_sys], linewidths=5, returnlevels=True)#, slices=(0,1))
      ax1.show_contour(data=image2, colors="blue", levels=lvls, returnlevels=True)#, slices=(0,1))
    #ax1.show_rectangles(xw=position.ra.deg+0.05, yw=position.dec.deg+0.05, width=6./3600., height=240./3600., angle=217, color='red', linestyle='--')
    ax1.show_rectangles(xw=330, yw=1730, width=24, height=1000, angle=217, coords_frame='pixel', color='red', linestyle='--', linewidth=2)
    ax1.add_scalebar(0.0094, r'10\,kpc', color='darkred', corner='bottom right')
    print(position.ra.deg, position.dec.deg)
    ax1.axis_labels.set_ytext('Declination')
    ax1.axis_labels.set_xtext('Right Ascension')
    ax1.tick_labels.set_xformat('hh:mm:ss')
    ax1.tick_labels.set_yformat('dd:mm')
    ax1.ticks.show()
    ax1.ticks.set_xspacing(0.05)
    ax1.ticks.set_length(5)
    ax1.ticks.set_color('black')
    ax1.ticks.set_minor_frequency(1)
    f1.close()

# ================================ #
# ===== Moment/Spectrum Plot ===== #
# ================================ #
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
    if colour == 'darkblue':
      error_colour = 'lightblue'
    elif colour == 'peru':
      error_colour = 'sandybrown'
    else:
      error_colour = 'lightgrey'
    plt.fill_between(velocity, flux-error, flux+error, alpha=0.5, edgecolor='none', facecolor=error_colour)
    #if sub3 == 4:
      #ax1.legend(loc='upper right', fontsize = 8.5)
    #plt.subplots_adjust(wspace=0.0, hspace=0.0)
    ax1.get_yaxis().set_tick_params(which='both', direction='in')
    ax1.get_xaxis().set_tick_params(which='both', direction='in')
    #return velocity, flux


# ================================ #
# ===== Moment/Spectrum Plot ===== #
# ================================ #
def spec_askap_hipass_plot(fig_num, sub1, sub2, sub3, velocity, flux, error, colour, vsys, txtstr):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    ax1.set_xlabel(r'Velocity [km\,s$^{-1}$]')
    ax1.set_ylabel(r'Flux Density [mJy]')
    flux = flux*1000.0
    
    plt.axvline(vsys, linewidth=0.75, linestyle = '--', color = 'darkgrey', label = r'$V_{\mathrm{sys,SoFiA}}$')
    if colour == 'darkblue':
      plt.text(0.05, 0.9, txtstr, fontsize=16, transform=ax1.transAxes)
      error_colour = 'lightblue'
      zorder = 2
      alpha = 1
      lwidth = 1.5
    elif colour == 'peru':
      error_colour = 'sandybrown'
      zorder = 1
      alpha = 0.8
      lwidth = 0.75
    else:
      error_colour = 'lightgrey'
      zorder = 1
      alpha = 0.8
      lwidth = 0.75
    plt.plot(velocity, flux, linestyle = '-', color = colour, linewidth = lwidth, alpha=alpha, zorder=zorder)#, label = txtstr)
    error = np.array(error)*1000.0
    plt.fill_between(velocity, flux-error, flux+error, alpha=alpha/1.5, edgecolor='none', facecolor=error_colour, zorder=zorder)
    #ax1.set_ylim(-1, None)
    ax1.get_yaxis().set_tick_params(which='both', direction='in')
    ax1.get_xaxis().set_tick_params(which='both', direction='in')
    

# ================================ #
# ======= Sky/HI/OPT Plot ======== #
# ================================ #
def field_hi_opt_plot(fig_num, sub1, sub2, sub3, image, ra, dec, cpar, fc, mark, wcs, cluster, lbl):
    #field_hi_opt_plot(fig_num, sub1, sub2, sub3, image, ra, dec, colour, fc, mark, wcs, cluster, lbl):
    matplotlib.rcParams.update({'font.size': 14})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w', projection=wcs)
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
    if image != False:
      if cluster == 'Hydra':
        cluster_centre    = np.array([159.174167, -27.524444])
      if cluster == 'Virgo':
        cluster_centre    = np.array([187.705930, 12.391123]) #186.6333, 12.7233
      if cluster == 'Background':
        cluster_centre    = np.array([159.174167, -27.524444])
      if cluster == 'Eridanus':
        cluster_centre    = np.array([53.750, -22.0])
      f1_dss            = pyfits.open(image)
      dss_data, dss_hdr = f1_dss[0].data, f1_dss[0].header
      dss_wcs           = WCS(dss_hdr)
      ax1.set_xlim(-0.5, dss_data.shape[1] - 0.5)
      ax1.set_ylim(-0.5, dss_data.shape[0] - 0.5)
      # ======== Add backgroud optical image ======== #
      #ax1.imshow(dss_data, origin='lower', aspect='auto', cmap='Greys', clim=(4000,13000))
      #ax1 = aplpy.FITSFigure(image, figure=fig_num, subplot=(sub1, sub2, sub3), dimensions=(0,1))# slices=(0,1))
      # ======== Rvir Positions ======== #
      c               = SkyCoord(cluster_centre[0]*u.degree, cluster_centre[1]*u.degree, frame='icrs')
      #print (c.ra, c.dec)
      beam_colour = 'grey'
      line_type   = '--'
      if cluster == 'Hydra':
        beam_size   = 1.4 * u.degree
      if cluster == 'Virgo':
        beam_size   = 6.0 * u.degree
      if cluster == 'Background':
        beam_size   = 0.8 * u.degree
      if cluster == 'Eridanus':
        beam_size   = 2.5 * u.degree
      rvir        = SphericalCircle((c.ra, c.dec), beam_size, edgecolor=beam_colour, 
                                    ls=line_type, facecolor='none', transform=ax1.get_transform('icrs'))
      # ======== Add Rvir to image ======== #
      ax1.add_patch(rvir)
      ax1.scatter(cluster_centre[0], cluster_centre[1], s=180, marker = '*', 
            facecolor='grey', transform=ax1.get_transform('icrs'))
    else:
      #ax1.set_autoscale_on(False)
      #scat_ax = plt.scatter(ra, dec, s=15, facecolors=fc, edgecolors=colour, marker=mark, 
                            #transform=ax1.get_transform('icrs'), label=lbl)
      array_colours = np.empty(len(cpar), dtype="object")
      array_colours[cpar<3300] = 'blue'
      array_colours[(cpar>=3300) & (cpar<4300)] = 'green'
      array_colours[cpar>=4300] = 'red'
      #array_fcolours = np.empty(len(cpar), dtype="object")
      #if fc == False
      #print(array_colours)
      #if cpar < 3500:
        #colour = 'blue'
      #elif cpar > 3500 and cpar < 5000:
        #colour = 'green'
      #else:
        #colour = 'red'
      #print(np.nanmin(cpar), np.nanmax(cpar))
      #print(len(ra))
      for i in range(len(ra)):
        if fc == False:
          fcolour = 'none'
        else:
          fcolour = array_colours[i]
        #print(ra[i], dec[i], array_colours[i])
        scat_ax = plt.scatter(ra[i],dec[i],s=20, marker=mark, edgecolors=array_colours[i], 
                              facecolors=fcolour, transform=ax1.get_transform('icrs'))#, label=lbl)
      plt.scatter(ra[0],0,s=20, marker=mark, edgecolors=array_colours[i], 
                              facecolors=fcolour, transform=ax1.get_transform('icrs'), label=lbl)
      if lbl == 'WALLABY':
        plt.scatter(ra[0],0,s=20, marker=mark, c='blue', 
                              transform=ax1.get_transform('icrs'), label=r'$<3300$\,km\,s$^{-1}$')
        plt.scatter(ra[0],0,s=20, marker=mark, c='green', 
                              transform=ax1.get_transform('icrs'), label=r'3300--4300\,km\,s$^{-1}$')
        plt.scatter(ra[0],0,s=20, marker=mark, c='red', 
                              transform=ax1.get_transform('icrs'), label=r'$\geq 4300$\,km\,s$^{-1}$')
        #if lbl == 'WALLABY':
          #cbar_ax = fig_num.add_axes([0.91, 0.11, 0.02, 0.77])
          #cbar = plt.colorbar(scat_ax, cax=cbar_ax, fraction=0.1, pad=0.01)
          #cbar.set_clim(np.nanmin(cpar),np.nanmax(cpar))
          #clbl = r'$V_{\mathrm{sys}}$ [km\,s$^{-1}$]'
          #cbar.set_label(clbl)
      ax1.legend(loc='upper right', fontsize = 9, ncol=3)#, transform=ax1.get_transform('icrs'))
      #else:
        #galaxy = mpatches.Ellipse((ra, dec), scale_factor*axmaj, scale_factor*axmin, pa, 
                                  #color=colour, fill=False, transform=ax1.get_transform('icrs'))
        #ax1.add_patch(galaxy)

# ================================ #
# ======= Sky/HI/OPT Plot ======== #
# ================================ #
def sky_position_plot(fig_num, subfig, image, ra, dec, cpar, marker_par, wcs, cluster, txtstr, lbl):
    #field_hi_opt_plot(fig_num, sub1, sub2, sub3, image, ra, dec, colour, fc, mark, wcs, cluster, lbl):
    matplotlib.rcParams.update({'font.size': 16})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(subfig[0], subfig[1], subfig[2], facecolor = 'w', projection=wcs)
    # ======== Axes Labels ======== #
    ax1.set_ylabel(r'Declination')
    ax1.set_xlabel(r'Right Ascension')
    # ======== Axes Tick Parameters ======== #
    ax1.tick_params(direction='in')
    lon = ax1.coords[0]
    lat = ax1.coords[1]
    lon.set_major_formatter('hh:mm')
    lat.set_major_formatter('dd')
    lon.set_separator((r'$^{\rm{h}}$', r'$^{\rm{m}}$'))
    lat.set_separator((r'$^{\circ}$', 'm'))
    if image != False:
      if cluster == 'Hydra':
        cluster_centre    = np.array([159.174167, -27.524444])
      f1_dss            = pyfits.open(image)
      dss_data, dss_hdr = f1_dss[0].data, f1_dss[0].header
      dss_wcs           = WCS(dss_hdr)
      ax1.set_xlim(-0.5, dss_data.shape[1] - 0.5)
      ax1.set_ylim(-0.5, dss_data.shape[0] - 0.5)
      # ======== Add backgroud optical image ======== #
      #ax1.imshow(dss_data, origin='lower', aspect='auto', cmap='Greys', clim=(4000,13000))
      #ax1 = aplpy.FITSFigure(image, figure=fig_num, subplot=(sub1, sub2, sub3), dimensions=(0,1))# slices=(0,1))
      # ======== Rvir Positions ======== #
      c               = SkyCoord(cluster_centre[0]*u.degree, cluster_centre[1]*u.degree, frame='icrs')
      #print (c.ra, c.dec)
      beam_colour = 'grey'
      line_type   = '--'
      if cluster == 'Hydra':
        beam_size   = 1.4 * u.degree
      if cluster == 'Virgo':
        beam_size   = 6.0 * u.degree
      if cluster == 'Background':
        beam_size   = 0.8 * u.degree
      if cluster == 'Eridanus':
        beam_size   = 2.5 * u.degree
      rvir        = SphericalCircle((c.ra, c.dec), beam_size, edgecolor=beam_colour, linewidth=2,
                                    ls=line_type, facecolor='none', transform=ax1.get_transform('icrs'))
      # ======== Add Rvir to image ======== #
      ax1.add_patch(rvir)
      ax1.scatter(cluster_centre[0], cluster_centre[1], s=180, marker = '*', 
            facecolor='grey', transform=ax1.get_transform('icrs'))
    else:
      #ax1.set_autoscale_on(False)
      #scat_ax = plt.scatter(ra, dec, s=15, facecolors=fc, edgecolors=colour, marker=mark, 
                            #transform=ax1.get_transform('icrs'), label=lbl)
      #array_colours = np.empty(len(cpar), dtype="object")
      #array_colours[cpar<3300] = 'blue'
      #array_colours[(cpar>=3300) & (cpar<4300)] = 'green'
      #array_colours[cpar>=4300] = 'red'
      #array_fcolours = np.empty(len(cpar), dtype="object")
      #if fc == False
      #print(array_colours)
      #if cpar < 3500:
        #colour = 'blue'
      #elif cpar > 3500 and cpar < 5000:
        #colour = 'green'
      #else:
        #colour = 'red'
      #print(np.nanmin(cpar), np.nanmax(cpar))
      #print(len(ra))
      for i in range(len(ra)):
        if marker_par[0] == 'o':
          zorder = 3
        else:
          zorder = 1
        scat_ax = plt.scatter(ra[i],dec[i],s=45, marker=marker_par[0], edgecolors=marker_par[1], 
                              facecolors=marker_par[2], linewidth=2, transform=ax1.get_transform('icrs'),
                              zorder=zorder)#, label=lbl)
        if txtstr != False:
          plt.text(ra[i], dec[i], txtstr, fontsize=8, transform=ax1.get_transform('icrs'))
      if lbl != False:
        plt.scatter(ra[0],0,s=45, marker=marker_par[0], edgecolors=marker_par[1], 
                                facecolors=marker_par[2], linewidth=2, transform=ax1.get_transform('icrs'), label=lbl)
      #if lbl == 'WALLABY':
        #plt.scatter(ra[0],0,s=20, marker=mark, c='blue', 
                              #transform=ax1.get_transform('icrs'), label=r'$<3300$\,km\,s$^{-1}$')
        #plt.scatter(ra[0],0,s=20, marker=mark, c='green', 
                              #transform=ax1.get_transform('icrs'), label=r'3300--4300\,km\,s$^{-1}$')
        #plt.scatter(ra[0],0,s=20, marker=mark, c='red', 
                              #transform=ax1.get_transform('icrs'), label=r'$\geq 4300$\,km\,s$^{-1}$')
        #if lbl == 'WALLABY':
          #cbar_ax = fig_num.add_axes([0.91, 0.11, 0.02, 0.77])
          #cbar = plt.colorbar(scat_ax, cax=cbar_ax, fraction=0.1, pad=0.01)
          #cbar.set_clim(np.nanmin(cpar),np.nanmax(cpar))
          #clbl = r'$V_{\mathrm{sys}}$ [km\,s$^{-1}$]'
          #cbar.set_label(clbl)
      ax1.legend(loc='upper right', fontsize = 14, ncol=3)#, transform=ax1.get_transform('icrs'))
      #else:
        #galaxy = mpatches.Ellipse((ra, dec), scale_factor*axmaj, scale_factor*axmin, pa, 
                                  #color=colour, fill=False, transform=ax1.get_transform('icrs'))
        #ax1.add_patch(galaxy)

# ================================ #
# ======= Sky/HI/OPT Plot ======== #
# ================================ #
def hi_id_opt_plot(fig_num, sub1, sub2, sub3, image, ra, dec, cpar, gal_id, fc, mark, wcs, cluster, lbl):
    #field_hi_opt_plot(fig_num, sub1, sub2, sub3, image, ra, dec, colour, fc, mark, wcs, cluster, lbl):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w', projection=wcs)
    # ======== Axes Labels ======== #
    ax1.set_ylabel(r'Declination')
    ax1.set_xlabel(r'Right Ascension')
    # ======== Axes Tick Parameters ======== #
    ax1.tick_params(direction='in')
    lon = ax1.coords[0]
    lat = ax1.coords[1]
    lon.set_major_formatter('hh:mm')
    lat.set_major_formatter('dd') #:mm
    lon.set_separator(('h', 'm'))
    lat.set_separator(('d')) #, 'm'
    if image != False:
      if cluster == 'Hydra':
        cluster_centre    = np.array([159.174167, -27.524444])
      if cluster == 'Virgo':
        cluster_centre    = np.array([187.705930, 12.391123]) #186.6333, 12.7233
      if cluster == 'Background':
        cluster_centre    = np.array([159.174167, -27.524444])
      if cluster == 'Eridanus':
        cluster_centre    = np.array([53.750, -22.0])
      f1_dss            = pyfits.open(image)
      dss_data, dss_hdr = f1_dss[0].data, f1_dss[0].header
      dss_wcs           = WCS(dss_hdr)
      ax1.set_xlim(-0.5, dss_data.shape[1] - 0.5)
      ax1.set_ylim(-0.5, dss_data.shape[0] - 0.5)
      #print(dss_data.shape[1] - 0.5, dss_data.shape[0] - 0.5)
      # ======== Rvir Positions ======== #
      c               = SkyCoord(cluster_centre[0]*u.degree, cluster_centre[1]*u.degree, frame='icrs')
      beam_colour = 'grey'
      line_type   = '--'
      if cluster == 'Hydra':
        beam_size   = 1.4 * u.degree
      if cluster == 'Virgo':
        beam_size   = 6.0 * u.degree
      if cluster == 'Background':
        beam_size   = 0.8 * u.degree
      if cluster == 'Eridanus':
        beam_size   = 2.5 * u.degree
      rvir        = SphericalCircle((c.ra, c.dec), beam_size, edgecolor=beam_colour, 
                                    ls=line_type, facecolor='none', transform=ax1.get_transform('icrs'))
      # ======== Add Rvir to image ======== #
      ax1.add_patch(rvir)
      ax1.scatter(cluster_centre[0], cluster_centre[1], s=180, marker = '*', 
            facecolor='grey', transform=ax1.get_transform('icrs'))
    else:
      array_colours = np.empty(len(cpar), dtype="object")
      array_colours[cpar<3300] = 'purple'
      array_colours[(cpar>=3300) & (cpar<4300)] = 'blue'
      array_colours[(cpar>=4300) & (cpar<7000)] = 'green'
      array_colours[(cpar>=7000) & (cpar<9000)] = 'orange'
      array_colours[(cpar>=9000) & (cpar<11500)] = 'red'
      array_colours[cpar>=11500] = 'magenta'
      for i in range(len(ra)):
        if fc == False:
          fcolour = 'none'
          #scat_ax = plt.scatter(ra[i], dec[i], c=cpar[i], cmap='rainbow', vmin=np.nanmin(cpar), vmax=np.nanmax(cpar), s=15, marker=mark, 
                                #facecolors='none', transform=ax1.get_transform('icrs'))
        else:
          fcolour = array_colours[i]
          #fcolour = 'rainbow'
          #scat_ax = plt.scatter(ra[i], dec[i], c=cpar[i], cmap='rainbow', vmin=np.nanmin(cpar), vmax=np.nanmax(cpar), s=15, marker=mark, 
                                #transform=ax1.get_transform('icrs'))
        scat_ax = plt.scatter(ra[i],dec[i],s=15, marker=mark, edgecolors=array_colours[i], 
                              facecolors=fcolour, transform=ax1.get_transform('icrs'))#, label=lbl)
        #scat_ax = plt.scatter(ra[i], dec[i], c=cpar[i], cmap='rainbow', s=15, marker=mark, 
                              #facecolors=fcolour, transform=ax1.get_transform('icrs'))#, label=lbl) edgecolors='rainbow', 
        if len(gal_id) > 1:
          plt.text(ra[i], dec[i], int(gal_id[i]), fontsize=8, transform=ax1.get_transform('icrs'))
      plt.scatter(ra[0],0,s=15, marker=mark, edgecolors=array_colours[i], 
                              facecolors=fcolour, transform=ax1.get_transform('icrs'))#, label=lbl)
      if lbl == 'WALLABY':
        if len(array_colours[array_colours == 'purple']) > 0:
          plt.scatter(ra[0],0,s=10, marker=mark, c='purple', 
                                transform=ax1.get_transform('icrs'), label=r'$<3300$') #\,km\,s$^{-1}$
        if len(array_colours[array_colours == 'blue']) > 0:
          plt.scatter(ra[0],0,s=10, marker=mark, c='blue', 
                                transform=ax1.get_transform('icrs'), label=r'3300--4300')
        if len(array_colours[array_colours == 'green']) > 0:
          plt.scatter(ra[0],0,s=10, marker=mark, c='green', 
                                transform=ax1.get_transform('icrs'), label=r'4300--7000')
        if len(array_colours[array_colours == 'orange']) > 0:
          plt.scatter(ra[0],0,s=10, marker=mark, c='orange', 
                                transform=ax1.get_transform('icrs'), label=r'7000--9000')
        if len(array_colours[array_colours == 'red']) > 0:
          plt.scatter(ra[0],0,s=10, marker=mark, c='red', 
                                transform=ax1.get_transform('icrs'), label=r'9000--11500')
        if len(array_colours[array_colours == 'magenta']) > 0:
          plt.scatter(ra[0],0,s=10, marker=mark, c='magenta', 
                                transform=ax1.get_transform('icrs'), label=r'$\geq 11500$')
      ax1.legend(fontsize = 8, ncol=1)#, transform=ax1.get_transform('icrs')) loc='upper right', 


# ================================ #
# ======= Sky/Param Plot ========= #
# ================================ #
def hydra_field_plot(fig_num, sub1, sub2, sub3, image, ra, dec, axmaj, axmin, pa, cpar, colour, clbl, wcs, cluster):
    matplotlib.rcParams.update({'font.size': 14})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w', projection=wcs)
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
    if image != False:
      if cluster == 'Hydra':
        cluster_centre    = np.array([159.174167, -27.524444])
      if cluster == 'Virgo':
        cluster_centre    = np.array([187.705930, 12.391123]) #186.6333, 12.7233
      if cluster == 'Background':
        cluster_centre    = np.array([159.174167, -27.524444])
      if cluster == 'Eridanus':
        cluster_centre    = np.array([53.750, -22.0])
      f1_dss            = pyfits.open(image)
      dss_data, dss_hdr = f1_dss[0].data, f1_dss[0].header
      dss_wcs           = WCS(dss_hdr)
      ax1.set_xlim(-0.5, dss_data.shape[1] - 0.5)
      ax1.set_ylim(-0.5, dss_data.shape[0] - 0.5)
      # ======== Add backgroud optical image ======== #
      #ax1.imshow(dss_data, origin='lower', aspect='auto', cmap='Greys', clim=(4000,13000))
      #ax1 = aplpy.FITSFigure(image, figure=fig_num, subplot=(sub1, sub2, sub3), dimensions=(0,1))# slices=(0,1))
      # ======== Rvir Positions ======== #
      c               = SkyCoord(cluster_centre[0]*u.degree, cluster_centre[1]*u.degree, frame='icrs')
      #print (c.ra, c.dec)
      beam_colour = 'black'
      line_type   = '--'
      if cluster == 'Hydra':
        beam_size   = 1.4 * u.degree
      if cluster == 'Virgo':
        beam_size   = 6.0 * u.degree
      if cluster == 'Background':
        beam_size   = 0.8 * u.degree
      if cluster == 'Eridanus':
        beam_size   = 2.5 * u.degree
      rvir        = SphericalCircle((c.ra, c.dec), beam_size, edgecolor=beam_colour, 
                                    ls=line_type, facecolor='none', transform=ax1.get_transform('icrs'))
      # ======== Add Rvir to image ======== #
      ax1.add_patch(rvir)
      ax1.scatter(cluster_centre[0], cluster_centre[1], s=120, marker = '*', 
            edgecolor='black', facecolor='black', transform=ax1.get_transform('icrs'))
    else:
      ax1.set_autoscale_on(False)
      if cluster == 'Hydra':
        scale_factor = 5.
      elif cluster == 'Virgo':
        scale_factor = 7.5
      else:
        scale_factor = 7.5
      if colour == False:
        cpar = np.array(cpar)
        #if clbl ==  r'$A_{\mathrm{flux}}$':
        m = (1. - 0.) / (np.nanmax(cpar) - np.nanmin(cpar))
        b = 0. - m * np.nanmin(cpar)
        s = [m * a + b for a in cpar] #a / max(cpar)
        #else:
          #m = (1. - 0.) / (np.nanmax(cpar) - np.nanmin(cpar))
          #b = 0. - m * np.nanmin(cpar)
          #s = [m * a + b for a in cpar] #a / max(cpar)
        colours = [cm.jet(color) for color in s]
        for i in range(len(ra)):
          galaxy = mpatches.Ellipse((ra[i], dec[i]), scale_factor*axmaj[i], scale_factor*axmin[i], pa[i], 
                                  color=colours[i], fill=False, transform=ax1.get_transform('icrs'))
          ax1.add_patch(galaxy)
          #print(galaxy)
        if clbl == r'$A_{\mathrm{flux}}$':
          if cluster == 'Hydra':
            size = 90
          if cluster == 'Virgo':
            size = 60
          if cluster == 'Background':
            size = 120
          if cluster == 'Eridanus':
            size = 60
          plt.scatter(ra[cpar>1.26], dec[cpar>1.26], s=size, color='hotpink', 
                      marker='d', transform=ax1.get_transform('icrs'))
        scat_ax = plt.scatter(ra,dec,s=0, c=cpar, cmap='jet', facecolors='none', transform=ax1.get_transform('icrs'))
        cbar_ax = fig_num.add_axes([0.91, 0.11, 0.02, 0.77])
        cbar = plt.colorbar(scat_ax, cax=cbar_ax, fraction=0.1, pad=0.01)
        cbar.set_clim(np.nanmin(cpar),np.nanmax(cpar))
        cbar.set_label(clbl)
      else:
        galaxy = mpatches.Ellipse((ra, dec), scale_factor*axmaj, scale_factor*axmin, pa, 
                                  color=colour, fill=False, transform=ax1.get_transform('icrs'))
        ax1.add_patch(galaxy)



# ================================ #
# ==== Sky/Opt/HI/Xray Plot ====== #
# ================================ #
def sky_opt_hi_plot(fig_num, sub1, sub2, sub3, image1, image2, wcs):
    matplotlib.rcParams.update({'font.size': 10})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    #ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w', projection=wcs)
    
    #if image1 != False:
    cluster_centre    = np.array([159.174167, -27.524444])
    f1_dss            = pyfits.open(image1)
    dss_data, dss_hdr = f1_dss[0].data, f1_dss[0].header
    dss_wcs           = WCS(dss_hdr)
    #ax1.set_xlim(-0.5, dss_data.shape[1] - 0.5)
    #ax1.set_ylim(-0.5, dss_data.shape[0] - 0.5)
    # ======== Add backgroud optical image ======== #
    #ax1.imshow(dss_data, origin='lower', aspect='auto', cmap='Greys', clim=(4000,13000))
    ax1 = aplpy.FITSFigure(image1, figure=fig_num, subplot=(sub1, sub2, sub3), dimensions=(0,1))# slices=(0,1))
    #if iteration == 0:
    ax1.show_grayscale(vmin=4000,vmax=14000,invert=True)
    # ======== Rvir Positions ======== #
    c               = SkyCoord(cluster_centre[0]*u.degree, cluster_centre[1]*u.degree, frame='icrs')
    #print (c.ra, c.dec)
    beam_colour = 'black'
    line_type   = '--'
    beam_size   = 1.35 * u.degree #1.48
    ax1.show_circles(cluster_centre[0], cluster_centre[1], beam_size, facecolor='none', edgecolor=beam_colour, ls=line_type)
    #rvir        = SphericalCircle((c.ra, c.dec), beam_size, edgecolor=beam_colour, 
                                  #ls=line_type, facecolor='none')#, transform=ax1.get_transform('icrs'))
    # ======== Add Rvir to image ======== #
    #ax1.add_patch(rvir)
    ax1.show_markers(cluster_centre[0], cluster_centre[1], s=120, marker = '*', edgecolor='black', facecolor='black')
    #ax1.scatter(cluster_centre[0], cluster_centre[1], s=120, marker = '*', 
          #edgecolor='black', facecolor='black')#, transform=ax1.get_transform('icrs'))
    #else:
    #ax1.set_autoscale_on(False)
    #xray_lvl = [0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.0015]
    #ax1.show_contour(image3, colors="magenta", levels=xray_lvl, slices=(0,1), smooth=3, linewidths=0.5)
    #askap_lvl = [2*10**19*(30.*30.)/(1.36*21*21*1.823*1000*(10**18))]
    askap_lvl = [5 * 10**19 * (30. * 30.) / (2.33 * 10**20)]
    for i in range(len(image2)):
      #print(image2[i])
      imbase = '/Users/tflowers/WALLABY/Hydra_DR1/WALLABY_PS_Hya_DR1_source_products/'
      if (image2[i] == imbase + 'WALLABY_PS_Hya_DR1_J104059-270456_moment0_new.fits' or 
      image2[i] == imbase + 'WALLABY_PS_Hya_DR1_J103725-251916_moment0_new.fits' or 
      image2[i] == imbase + 'WALLABY_PS_Hya_DR1_J104142-284653_moment0_new.fits' or
      image2[i] == imbase + 'WALLABY_PS_Hya_DR1_J104016-274630_moment0_new.fits' or 
      image2[i] == imbase + 'WALLABY_PS_Hya_DR1_J103523-281855_moment0_new.fits' or 
      image2[i] == imbase + 'WALLABY_PS_Hya_DR1_J104309-300301_moment0_new.fits' or 
      image2[i] == imbase + 'WALLABY_PS_Hya_DR1_J104311-261500_moment0_new.fits' or 
      image2[i] == imbase + 'WALLABY_PS_Hya_DR1_J103702-273359_moment0_new.fits' or 
      image2[i] == imbase + 'WALLABY_PS_Hya_DR1_J103335-272717_moment0_new.fits' or 
      image2[i] == imbase + 'WALLABY_PS_Hya_DR1_J104339-285157_moment0_new.fits' or
      image2[i] == imbase + 'WALLABY_PS_Hya_DR1_J103818-285307_moment0_new.fits' or
      image2[i] == imbase + 'WALLABY_PS_Hya_DR1_J103651-260227_moment0_new.fits' or
      image2[i] == imbase + 'WALLABY_PS_Hya_DR1_J104000-292445_moment0_new.fits' or
      image2[i] == imbase + 'WALLABY_PS_Hya_DR1_J103359-301003_moment0_new.fits' or
      image2[i] == imbase + 'WALLABY_PS_Hya_DR1_J103244-283639_moment0_new.fits'):
      #image2[i] == imbase + 'WALLABY_PS_Hya_DR1_J103436-273900_moment0_new.fits' or
      #image2[i] == imbase + 'WALLABY_PS_Hya_DR1_J103918-265030_moment0_new.fits' or
      #image2[i] == imbase + 'WALLABY_PS_Hya_DR1_J103653-270311_moment0_new.fits'):
        ax1.show_contour(image2[i], colors="red", levels=askap_lvl, slices=(0,1), linewidths=0.75)
      else:
        ax1.show_contour(image2[i], colors="blue", levels=askap_lvl, slices=(0,1), linewidths=0.75)
    # ======== Galaxy Position ======== #
    gal_centre  = np.array([160.2458, -27.0822])
    c           = SkyCoord(gal_centre[0]*u.degree, gal_centre[1]*u.degree, frame='icrs')
    beam_colour = 'peru'
    line_type   = '-.'
    beam_size   = 0.35 * u.degree
    ax1.show_circles(gal_centre[0], gal_centre[1], beam_size, facecolor='none', edgecolor=beam_colour, ls=line_type)
    # ======== Axes Labels ======== #
    #ax1.set_ylabel(r'Declination')
    #ax1.set_xlabel(r'Right Ascension')
    # ======== Axes Tick Parameters ======== #
    #ax1.tick_params(direction='in')
    #lon = ax1.coords[0]
    #lat = ax1.coords[1]
    #lon.set_major_formatter('hh:mm')
    #lat.set_major_formatter('dd:mm')
    #lon.set_separator(('h', 'm'))
    #lat.set_separator(('d', 'm'))


# ================================ #
# ==== Sky/Opt/HI/Xray Plot ====== #
# ================================ #
def sky_opt_hi_xray_plot(fig_num, sub1, sub2, sub3, image1, image2, image3, wcs):
    matplotlib.rcParams.update({'font.size': 14})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    #ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w', projection=wcs)
    
    #if image1 != False:
    cluster_centre    = np.array([159.174167, -27.524444])
    f1_dss            = pyfits.open(image1)
    dss_data, dss_hdr = f1_dss[0].data, f1_dss[0].header
    dss_wcs           = WCS(dss_hdr)
    #ax1.set_xlim(-0.5, dss_data.shape[1] - 0.5)
    #ax1.set_ylim(-0.5, dss_data.shape[0] - 0.5)
    # ======== Add backgroud optical image ======== #
    #ax1.imshow(dss_data, origin='lower', aspect='auto', cmap='Greys', clim=(4000,13000))
    ax1 = aplpy.FITSFigure(image1, figure=fig_num, subplot=(sub1, sub2, sub3), dimensions=(0,1))# slices=(0,1))
    #if iteration == 0:
    ax1.show_grayscale(vmin=4000,vmax=13000,invert=True)
    # ======== Rvir Positions ======== #
    c               = SkyCoord(cluster_centre[0]*u.degree, cluster_centre[1]*u.degree, frame='icrs')
    #print (c.ra, c.dec)
    beam_colour = 'black'
    line_type   = '--'
    beam_size   = 1.4 * u.degree
    ax1.show_circles(cluster_centre[0], cluster_centre[1], beam_size, facecolor='none', edgecolor=beam_colour, ls=line_type)
    #rvir        = SphericalCircle((c.ra, c.dec), beam_size, edgecolor=beam_colour, 
                                  #ls=line_type, facecolor='none')#, transform=ax1.get_transform('icrs'))
    # ======== Add Rvir to image ======== #
    #ax1.add_patch(rvir)
    ax1.show_markers(cluster_centre[0], cluster_centre[1], s=120, marker = '*', edgecolor='black', facecolor='black')
    #ax1.scatter(cluster_centre[0], cluster_centre[1], s=120, marker = '*', 
          #edgecolor='black', facecolor='black')#, transform=ax1.get_transform('icrs'))
    #else:
    #ax1.set_autoscale_on(False)
    xray_lvl = [0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.0015]
    ax1.show_contour(image3, colors="magenta", levels=xray_lvl, slices=(0,1), smooth=3, linewidths=0.5)
    askap_lvl = [2*10**19*(30.*30.)/(1.36*21*21*1.823*1000*(10**18))]
    for i in range(len(image2)):
      ax1.show_contour(image2[i], colors="blue", levels=askap_lvl, slices=(0,1), linewidths=0.4)
    # ======== Axes Labels ======== #
    #ax1.set_ylabel(r'Declination')
    #ax1.set_xlabel(r'Right Ascension')
    # ======== Axes Tick Parameters ======== #
    #ax1.tick_params(direction='in')
    #lon = ax1.coords[0]
    #lat = ax1.coords[1]
    #lon.set_major_formatter('hh:mm')
    #lat.set_major_formatter('dd:mm')
    #lon.set_separator(('h', 'm'))
    #lat.set_separator(('d', 'm'))



# ================================ #
# ==== Sky/Opt/HI/Xray Plot ====== #
# ================================ #
def sky_opt_hi_xray2_plot(fig_num, sub1, sub2, sub3, image1, image2, image3, image4, wcs):
    matplotlib.rcParams.update({'font.size': 14})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    #ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w', projection=wcs)
    
    #if image1 != False:
    cluster_centre    = np.array([159.174167, -27.524444])
    f1_dss            = pyfits.open(image1)
    dss_data, dss_hdr = f1_dss[0].data, f1_dss[0].header
    dss_wcs           = WCS(dss_hdr)
    #ax1.set_xlim(-0.5, dss_data.shape[1] - 0.5)
    #ax1.set_ylim(-0.5, dss_data.shape[0] - 0.5)
    # ======== Add backgroud optical image ======== #
    #ax1.imshow(dss_data, origin='lower', aspect='auto', cmap='Greys', clim=(4000,13000))
    ax1 = aplpy.FITSFigure(image1, figure=fig_num, subplot=(sub1, sub2, sub3), dimensions=(0,1))# slices=(0,1))
    #if iteration == 0:
    ax1.show_grayscale(vmin=4000,vmax=13000,invert=True)
    # ======== Rvir Positions ======== #
    c               = SkyCoord(cluster_centre[0]*u.degree, cluster_centre[1]*u.degree, frame='icrs')
    #print (c.ra, c.dec)
    beam_colour = 'black'
    line_type   = '--'
    beam_size   = 1.4 * u.degree
    ax1.show_circles(cluster_centre[0], cluster_centre[1], beam_size, facecolor='none', edgecolor=beam_colour, ls=line_type)
    #rvir        = SphericalCircle((c.ra, c.dec), beam_size, edgecolor=beam_colour, 
                                  #ls=line_type, facecolor='none')#, transform=ax1.get_transform('icrs'))
    # ======== Add Rvir to image ======== #
    #ax1.add_patch(rvir)
    ax1.show_markers(cluster_centre[0], cluster_centre[1], s=120, marker = '*', edgecolor='black', facecolor='black')
    #ax1.scatter(cluster_centre[0], cluster_centre[1], s=120, marker = '*', 
          #edgecolor='black', facecolor='black')#, transform=ax1.get_transform('icrs'))
    #else:
    #ax1.set_autoscale_on(False)
    xray_lvl = [0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.0015]
    ax1.show_contour(image3, colors="magenta", levels=xray_lvl, slices=(0,1), smooth=3, linewidths=0.5)
    xray_lvl = [0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.0015]
    ax1.show_contour(image4, colors="peru", levels=xray_lvl, slices=(0,1), smooth=3, linewidths=0.5)
    askap_lvl = [2*10**19*(30.*30.)/(1.36*21*21*1.823*1000*(10**18))]
    for i in range(len(image2)):
      ax1.show_contour(image2[i], colors="blue", levels=askap_lvl, slices=(0,1), linewidths=0.4)
    # ======== Axes Labels ======== #
    #ax1.set_ylabel(r'Declination')
    #ax1.set_xlabel(r'Right Ascension')
    # ======== Axes Tick Parameters ======== #
    #ax1.tick_params(direction='in')
    #lon = ax1.coords[0]
    #lat = ax1.coords[1]
    #lon.set_major_formatter('hh:mm')
    #lat.set_major_formatter('dd:mm')
    #lon.set_separator(('h', 'm'))
    #lat.set_separator(('d', 'm'))



# ================================ #
# ===== Surface Density Plot ===== #
# ================================ #
def profile_plot(fig_num, subfig, flux1, flux2, errors, colour, marker, fsty, lbl, txtstr, legend_true):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(subfig, facecolor = 'w')
    ax1.set_ylabel(r'$\log_{10}[\Sigma/\mathrm{M}_{\odot}\mathrm{pc}^{-2}]$')
    if subfig == 211:
      ax1.set_xticklabels([])
    if subfig == 212 or subfig == 111:
      ax1.set_xlabel(r'Radius [kpc]')
    ax1.set_xlim(0.0, 50.)
    #ax1.set_ylim(np.min(flux2)-0.5, np.max(flux2)+0.5)
    #ax1.set_ylim(-2, 2)
    #plt.plot([0,10], [0,10], color='black', linewidth=1, linestyle = '--')
    #plt.scatter(flux1, flux2, color=colour, s=20, label=lbl)
    #print pow(10,flux2)
    #print 0.2*1/pow(10,flux2)
    #err_l = 0.434*((0.1*1/pow(10,flux2))/pow(10,flux2))
    #if colour == 'darkblue':
      #plt.text(0.05, 0.05, txtstr, transform=ax1.transAxes)
    #print 10**(flux2+0.2)-10**flux2
    #if lbl == 'gas':
    #  err_l = 0.434*(errors/10**flux2) + 0.2
    #else:
    #  err_l = 0.434*errors/10**flux2  #(10**(flux2+0.2)-10**flux2)/10**flux2  #0.4#0.434*0.4*0.6
    for i in range(len(flux2)):
      if 10**flux2[i] < 0:
        flux2[i] = np.log10(0.05)
    for i in range(len(errors)):
      if errors[i] < 0:
        errors[i] = np.abs(errors[i])
    el = flux2*0
    for i in range(len(flux2)):
      if 10**flux2[i] - errors[i] < 0:
        el[i] = 1
      else:
        el[i]    = flux2[i] - np.log10(10**flux2[i] - errors[i])
    eu    = -flux2 + np.log10(10**flux2 + errors)
    #err_h = np.log10(pow(10,flux2) + pow(10,0.2*flux2))
    plt.errorbar(flux1, flux2, yerr = [el,eu], color=colour, markersize=5, fmt=marker, fillstyle=fsty, label=lbl) #xerr = 0.2*flux2, 
    if legend_true and (subfig == 211 or subfig == 111):
        ax1.legend(loc='upper right')
    plt.subplots_adjust(wspace=0, hspace=0)
    ax1.get_yaxis().set_tick_params(which='both', direction='in')
    ax1.get_xaxis().set_tick_params(which='both', direction='in')


# ================================ #
# === Surface Brightness Plot ==== #
# ================================ #
def sb_profile_plot(fig_num, sub1, sub2, sub3, flux1, flux2, errors, model1, model2, radius, txtstr, txtpos):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    #ax1.set_ylabel(r'$\log_{10}[\Sigma/\mathrm{M}_{\odot}\mathrm{pc}^{-2}]$')
    #if subfig == 211:
      #ax1.set_xticklabels([])
    #if subfig == 212 or subfig == 111:
      #ax1.set_xlabel(r'Radius [kpc]')
    #ax1.set_xlim(0.0, 50.)
    ax1.set_ylim(np.nanmin(flux2)-0.2, np.nanmax(flux2)+0.2)
    #ax1.set_ylim(-2, 2)
    plt.plot(model1, model2, color='peru', linewidth=1, linestyle = '-')
    #plt.scatter(flux1, flux2, color='darkblue', s=15, edgecolor='darkblue', facecolor='none')
    plt.axvline(radius, color = 'black', linestyle = ':', linewidth = 1, zorder = 0)
    if txtpos == 'upper':
      #plt.text(0.25, 0.9, txtstr, transform=ax1.transAxes)
      plt.axhline(23.5, color = 'grey', linestyle = '--', linewidth=1, zorder = 0)
    #if txtpos == 'lower':
      #plt.text(0.25, 0.1, txtstr, transform=ax1.transAxes)
    if sub3 < 5:
      plt.gca().invert_yaxis()
    #el = flux2*0
    #for i in range(len(flux2)):
      #if 10**flux2[i] - errors[i] < 0:
        #el[i] = 1
      #else:
        #el[i]    = flux2[i] - np.log10(10**flux2[i] - errors[i])
    #eu    = -flux2 + np.log10(10**flux2 + errors)
    ##err_h = np.log10(pow(10,flux2) + pow(10,0.2*flux2))
    el = np.abs(flux2 - errors[0])
    eu = np.abs(flux2 - errors[1])
    plt.errorbar(flux1, flux2, yerr = [el,eu], color='darkblue', markersize=3, fmt='o', linewidth=0.75, fillstyle='none')
    #if legend_true and (subfig == 211 or subfig == 111):
        #ax1.legend(loc='upper right')
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    ax1.get_yaxis().set_tick_params(which='both', direction='in')
    ax1.get_xaxis().set_tick_params(which='both', direction='in')


def sb_profile_radius_plot(fig_num, sub1, sub2, sub3, flux1, flux2, radius_iso, radius_eff):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    ax1.set_ylabel(r'$\log_{10}[\Sigma/\mathrm{M}_{\odot}\mathrm{pc}^{-2}]$')
    ax1.set_xlabel(r'Radius [arcsec]')
    ax1.set_ylim(np.nanmin(flux2)-0.2, np.nanmax(flux2)+0.2)
    plt.axvline(radius_iso, color = 'peru', linestyle = '--', linewidth = 1.5, zorder = 0)
    plt.axvline(radius_eff, color = 'mediumvioletred', linestyle = '-.', linewidth = 1.5, zorder = 0)
    ax1.plot(flux1, flux2, color='darkblue', linewidth=2, linestyle='-')
    plt.gca().invert_yaxis()
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    ax1.get_yaxis().set_tick_params(which='both', direction='in')
    ax1.get_xaxis().set_tick_params(which='both', direction='in')

def sb_profile_radius_plot2(fig_num, sub1, sub2, sub3, flux1, flux2, radius_iso, radius_eff, colour):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    ax1.set_ylabel(r'$\log_{10}[\Sigma/\mathrm{M}_{\odot}\mathrm{pc}^{-2}]$')
    ax1.set_xlabel(r'Radius [arcsec]')
    ax1.set_ylim(np.nanmin(flux2)-0.2, np.nanmax(flux2)+0.2)
    plt.axvline(radius_iso, color = 'darkgrey', linestyle = '--', linewidth = 1.5, zorder = 0)
    plt.axvline(radius_eff, color = 'mediumvioletred', linestyle = '-.', linewidth = 1.5, zorder = 0)
    ax1.plot(flux1, flux2, color=colour, linewidth=1.5, linestyle='-')
    #if sub3 == 1:
    plt.gca().invert_yaxis()
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    ax1.get_yaxis().set_tick_params(which='both', direction='in')
    ax1.get_xaxis().set_tick_params(which='both', direction='in')

def sb_profile_radius_plot3(fig_num, sub1, sub2, sub3, flux1, flux2, radius_array, col_array):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    ax1.set_ylabel(r'$\log_{10}[\Sigma/\mathrm{M}_{\odot}\mathrm{pc}^{-2}]$')
    ax1.set_xlabel(r'Radius [arcsec]')
    ax1.set_ylim(np.nanmin(flux2)-0.2, np.nanmax(flux2)+0.2)
    for i in range(len(radius_array)):
      plt.axvline(radius_array[i], color = col_array[i], linestyle = '--', linewidth = 1.5, zorder = 0)
    #plt.axvline(radius_eff, color = 'mediumvioletred', linestyle = '-.', linewidth = 1.5, zorder = 0)
    ax1.plot(flux1, flux2, color='darkblue', linewidth=2, linestyle='-')
    plt.gca().invert_yaxis()
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    ax1.get_yaxis().set_tick_params(which='both', direction='in')
    ax1.get_xaxis().set_tick_params(which='both', direction='in')

# ================================ #
# === Surface Brightness Plot ==== #
# ================================ #
def surfden_profile_plot(fig_num, sub1, sub2, sub3, flux1, flux2, radius_iso1, radius_iso, radius_eff1, radius_eff):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    ax1.set_ylabel(r'$\Sigma/\mathrm{M}_{\odot}\mathrm{pc}^{-2}$')
    #if subfig == 211:
      #ax1.set_xticklabels([])
    #if subfig == 212 or subfig == 111:
    ax1.set_xlabel(r'Radius [arcsec]')
    #ax1.set_xlim(0.0, 400.)
    #ax1.set_ylim(np.nanmin(flux2)-0.2, np.nanmax(flux2)+0.2)
    #ax1.set_ylim(-2, 2)
    #plt.plot(model1, model2, color='peru', linewidth=1, linestyle = '-')
    plt.axvline(radius_iso, color = 'peru', linestyle = '--', linewidth = 1.5, zorder = 0)
    plt.axvline(radius_iso1, color = 'sandybrown', linestyle = '--', linewidth = 1, zorder = 0)
    plt.axvline(radius_eff, color = 'mediumvioletred', linestyle = '-.', linewidth = 1.5, zorder = 0)
    plt.axvline(radius_eff1, color = 'violet', linestyle = '-.', linewidth = 1, zorder = 0)
    plt.axhline(1, color = 'black', linestyle = ':', linewidth=1, zorder = 0)
    ax1.plot(flux1, flux2, color='darkblue', linewidth=2, linestyle='-')
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    ax1.get_yaxis().set_tick_params(which='both', direction='in')
    ax1.get_xaxis().set_tick_params(which='both', direction='in')

# ================================ #
# === Surface Brightness Plot ==== #
# ================================ #
def mock_surfden_profile_plot(fig_num, subfig, data, colours, lstyles, label):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(subfig[0], subfig[1], subfig[2], facecolor = 'w')
    ax1.set_ylabel(r'$\Sigma/\mathrm{M}_{\odot}\mathrm{pc}^{-2}$')
    #if subfig == 211:
      #ax1.set_xticklabels([])
    #if subfig == 212 or subfig == 111:
    ax1.set_xlabel(r'Radius [arcsec]')
    #ax1.set_xlim(0.0, 400.)
    #ax1.set_ylim(np.nanmin(flux2)-0.2, np.nanmax(flux2)+0.2)
    #ax1.set_ylim(-2, 2)
    #plt.plot(model1, model2, color='peru', linewidth=1, linestyle = '-')
    plt.axvline(data[1], color = colours[1], linestyle = lstyles[1], linewidth = 1.5, zorder = 0)
    plt.axvline(data[2], color = colours[2], linestyle = lstyles[2], linewidth = 1.5, zorder = 0)
    plt.axhline(1, color = 'black', linestyle = ':', linewidth=1, zorder = 0)
    ax1.plot(data[0][0], data[0][1], color=colours[0], linewidth=2, linestyle=lstyles[0], label=label)
    ax1.legend(loc='upper right')
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    ax1.get_yaxis().set_tick_params(which='both', direction='in')
    ax1.get_xaxis().set_tick_params(which='both', direction='in')

# ================================ #
# === Surface Brightness Plot ==== #
# ================================ #
def hisd_profile_plot(fig_num, sub1, sub2, sub3, data, axlbls, col, lwidth, zorder, txtstr):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    #ax1.set_ylim(np.nanmin(flux2)-0.2, np.nanmax(flux2)+0.2)
    #ax1.set_xlim(0,2.5) #2.5
    ax1.set_xlim(-1.3,0.5)
    if axlbls[1] == r'$\log(\Sigma_{\rm{HI}}/[\rm{M}_{\odot}\,\rm{pc}^{-2}])$':
      ax1.set_ylim(-1.5,1.2)
    if axlbls[1] == r'$\Sigma_r$ [mag/arcsec$^{2}$]':
      ax1.set_ylim(17.5,27.5)
      ax1.invert_yaxis()
    if axlbls[1] == r'$\log(M_*/[\rm{M}_{\odot}\,\rm{pc}^{-2}])$':
      ax1.set_ylim(-2.5,7)
    ax1.set_xlabel(axlbls[0])
    ax1.set_ylabel(axlbls[1])
    #ax1.set_xlabel(r'$r/R_{\rm{HI}}$')
    #ax1.set_ylabel(r'$\log(\Sigma_{\rm{HI}}/[\rm{M}_{\odot}\,\rm{pc}^{-2}])$')
    if col == 'magenta':
      linestyle = '--'
    else:
      linestyle = '-'
    if lwidth == 0.75:
      alpha = 0.5
    else:
      alpha = 1
    if txtstr != False:
      ax1.plot(data[0], data[1], color=col, linewidth=lwidth, 
              linestyle=linestyle, alpha=alpha, zorder=zorder, label=txtstr)
    else:
      ax1.plot(data[0], data[1], color=col, linewidth=lwidth, 
              linestyle=linestyle, alpha=alpha, zorder=zorder)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    ax1.legend(ncol = 2, loc='lower left', fontsize=10)
    ax1.get_yaxis().set_tick_params(which='both', direction='in')
    ax1.get_xaxis().set_tick_params(which='both', direction='in')


# ================================ #
# === Surface Brightness Plot ==== #
# ================================ #
def profile_plot10(fig_num, sub1, sub2, sub3, flux1, flux2, col, lwidth):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    #ax1.set_ylim(np.nanmin(flux2)-0.2, np.nanmax(flux2)+0.2)
    #ax1.set_xlim(0,2.5)
    #ax1.set_ylim(-1.5,1.2)
    ax1.set_xlabel(r'$r/R_{\rm{HI}}$')
    ax1.set_ylabel(r'$\log(\Sigma_{\rm{HI}}/[\rm{M}_{\odot}\,\rm{pc}^{-2}])$')
    plt.plot(flux1, flux2, color=col, linewidth=lwidth, linestyle = '-')
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    ax1.get_yaxis().set_tick_params(which='both', direction='in')
    ax1.get_xaxis().set_tick_params(which='both', direction='in')


# ================================ #
# ======= Mass Model Plot ======== #
# ================================ #
def model_plot(fig_num, subfig, flux1, flux2, flux2_err, colour, lsty, lbl, txtstr, legend_true):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(subfig, facecolor = 'w')
    ax1.set_ylabel(r'Velocity [km\,s$^{-1}$]')
    #if subfig == 211:
      #ax1.set_xticklabels([])
    #if subfig == 212:
    ax1.set_xlabel(r'Radius [kpc]')
    ax1.set_xlim(0.0, 50.)#np.nanmax(flux1)+10.)
    #ax1.set_ylim(8.0, 9.5)
    #plt.plot([0,10], [0,10], color='black', linewidth=1, linestyle = '--')
    plt.axhline(0, linewidth=0.75, linestyle = ':', color = 'darkgrey')
    if lbl == 'Data':
      #plt.scatter(flux1, flux2, color=colour, s=15, label=lbl)
      plt.errorbar(flux1, flux2, yerr = flux2_err, color=colour, markersize=5, fmt='o', label=lbl)
      #plt.text(0.05, 0.9, txtstr, transform=ax1.transAxes)
    else:
      plt.plot(flux1, flux2, color=colour, linestyle=lsty, label=lbl)
    #plt.errorbar(flux1, flux2, xerr = 0.00116, yerr = 0.00064, color=colour, markersize=4, fmt='o')
    #if subfig == 212:
    ax1.legend(loc='upper right')
    plt.subplots_adjust(wspace=0, hspace=0)
    ax1.get_yaxis().set_tick_params(which='both', direction='in')
    ax1.get_xaxis().set_tick_params(which='both', direction='in')


def rotcur_plot(fig_num, subfig, x, y, y_err, y_mean, ylbl):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(subfig, facecolor = 'w')
    ax1.set_ylabel(ylbl)
    if subfig != 313:
      ax1.set_xticklabels([])
    if subfig == 313:
      ax1.set_xlabel(r'Radius [arcsec]')
    ax1.set_xlim(0.0, 95.)#np.nanmax(flux1)+10.)
    if subfig == 311:
      ax1.set_ylim(0, 220)
    #if subfig == 412:
      #ax1.set_ylim(-6, 29)
    if subfig == 312:
      ax1.set_ylim(53.1, 61)
    if subfig == 313:
      ax1.set_ylim(202, 226)
    if ylbl == r'$v_{\mathrm{rot}}$ [km\,s$^{-1}$]' or ylbl == r'$\sigma_{\mathrm{gas}}$ [km\,s$^{-1}$]':
      if y_mean == True:
        plt.errorbar(x, y, yerr = y_err, color='darkblue', markersize=7, fmt='o', fillstyle='none', mew=1.5, zorder=2)#, label=lbl)
      if y_mean == False:
        plt.errorbar(x, y, yerr = y_err, color='peru', markersize=4, fmt='o', mew=1.5, zorder=1)#, label=lbl) fillstyle='none'
    else:
      plt.errorbar(x, y, yerr = y_err, color='peru', markersize=4, fmt='o', mew=1.5, zorder=1)
      #plt.scatter(x, y, color='peru', marker='o', s=24)#, label=lbl)
      plt.axhline(y_mean, linewidth=1.5, linestyle = '-', color = 'darkblue')
      plt.fill_between([0,95], y_mean-2., y_mean+2., alpha=0.15, edgecolor='none', facecolor='darkblue')
    #ax1.legend(loc='upper right')
    plt.subplots_adjust(wspace=0, hspace=0)
    ax1.get_yaxis().set_tick_params(which='both', direction='in')
    ax1.get_xaxis().set_tick_params(which='both', direction='in')


def colden_cut_plot(fig_num, subfig, x, y):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(subfig, facecolor = 'w')
    x_sw = x[x<=0]
    y_sw = y[x<=0]
    x_ne = x[x>=0]
    y_ne = y[x>=0]
    ax1.plot(-1.*x_sw, y_sw, color='darkblue', linestyle='-', markersize=3, label = 'SW') #marker='o', 
    ax1.plot(x_ne, y_ne, color='peru', linestyle='--', markersize=3, label = 'NE') # marker='s', 
    #plt.text(0.05, 0.9, 'SW', transform=ax1.transAxes, fontsize=14)
    #plt.text(0.9, 0.9, 'NE', transform=ax1.transAxes, fontsize=14)
    ax1.set_xlim(0,200)
    ax1.set_xlabel(r'Offset [arcsec]')
    ax1.set_ylabel(r'$N_{\mathrm{HI}}$ [$10^{20}$\,cm$^{-2}$]')
    ax2 = ax1.twiny()
    ax1Xs = ax1.get_xticks()
    ax2Xs = []
    for X in ax1Xs:
      ax2Xs.append(round(61000. * math.atan(X * math.pi / 180. / 3600.),1))
    ax2.set_xticks(ax1Xs)
    ax2.set_xbound(ax1.get_xbound())
    ax2.set_xticklabels(ax2Xs)
    ax2.set_xlabel(r'Offset [kpc]')
    #print(ax2Xs)
    ax1.legend(loc='upper right')
    plt.subplots_adjust(wspace=0, hspace=0)
    ax1.get_yaxis().set_tick_params(which='both', direction='in')
    ax1.get_xaxis().set_tick_params(which='both', direction='in')
    ax2.get_xaxis().set_tick_params(which='both', direction='in')


# ================================ #
# ====== Force Balance Plot ====== #
# ================================ #
def force_balance_plot(fig_num, subfig, x, y, colour, lsty, lbl, txtstr, legtitle):
    matplotlib.rcParams.update({'font.size': 12}) 
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(subfig, facecolor = 'w')
    ax1.set_ylabel(r'$\log_{10}(P_{\mathrm{grav}}/P_{\mathrm{ram}})$')
    if subfig == 211:
      ax1.set_xticklabels([])
    #if subfig == 212:
    ax1.set_xlabel(r'Radius [kpc]')
    ax1.set_xlim(0, np.max(x) + 1.)
    ax1.set_ylim(-2.5, 1.8)
    #ax1.set_ylim(-1.1, 5.1)
    if lsty == '-':
      plt.text(0.65, 0.9, txtstr, transform=ax1.transAxes)
    plt.axhline(0, linewidth=0.75, linestyle = '--', color = 'black', alpha=0.65, zorder=1)
    plt.axvline(25, linewidth=0.75, linestyle = ':', color = 'black', alpha=0.65, zorder=1)
    if colour == 'darkblue':
      lwidth = 2.0
    else:
      lwidth = 2.0 
    if colour != 'lightgrey':
      plt.plot(x, y, color=colour, linestyle=lsty, label=lbl, linewidth=lwidth, zorder=3)
    else:
      plt.plot(x, y, color=colour, linestyle=lsty, linewidth=1, zorder=2)
    ax1.legend(loc='lower left', title=legtitle, fontsize=9.5, title_fontsize=9.5)#, ncol=2)
    plt.subplots_adjust(wspace=0, hspace=0)
    ax1.get_yaxis().set_tick_params(which='both', direction='in')
    ax1.get_xaxis().set_tick_params(which='both', direction='in')



# ================================ #
# ====== Force Balance Plot ====== #
# ================================ #
def grav_potential_plot(fig_num, subfig, x, y, colour, lsty, xlbl, ylbl):
    matplotlib.rcParams.update({'font.size': 12}) 
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(subfig, facecolor = 'w')
    ax1.set_ylabel(ylbl)
    #if subfig == 211:
      #ax1.set_xticklabels([])
    #if subfig == 212:
    ax1.set_xlabel(xlbl)
    #ax1.set_xlim(0, 60)
    #ax1.set_ylim(-6, 2)
    #if lsty == '-':
      #plt.text(0.65, 0.9, txtstr, transform=ax1.transAxes)
    #plt.axhline(0, linewidth=0.75, linestyle = '--', color = 'darkgrey', zorder=1)
    #plt.axvline(30, linewidth=0.75, linestyle = ':', color = 'darkgrey', zorder=1)
    plt.plot(x, y, color=colour, linestyle=lsty)#, label=lbl, linewidth=0.9, zorder=2)
    #ax1.legend(loc='lower left', title=legtitle, fontsize=9.5, title_fontsize=9.5, ncol=2)
    plt.subplots_adjust(wspace=0.20, hspace=0.20)
    ax1.get_yaxis().set_tick_params(which='both', direction='in')
    ax1.get_xaxis().set_tick_params(which='both', direction='in')



# ================================ #
# ======= IGM Density Plot ======= #
# ================================ #
def igm_density_profile_plot(fig_num, subfig, x, y, yerr, colour, lbl):
    matplotlib.rcParams.update({'font.size': 12}) 
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(subfig, facecolor = 'w')
    ax1.set_ylabel(r'$n_{\mathrm{IGM}}$ [cm$^{-3}$]')
    ax1.set_xlabel(r'Radius [Mpc]')
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_xlim(0.0008, 5)
    ax1.set_ylim(0.000005, 0.1)
    #labels = [item.get_text() for item in ax1.get_xticklabels()]
    #labels[0] = 0.001
    #labels[1] = 0.01
    #labels[2] = 0.1
    #labels[3] = 1
    #ax1.set_xticklabels([0.001, 0.01, 0.1, 1])
    formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))
    ax1.xaxis.set_major_formatter(formatter)
    #if lsty == '-':
      #plt.text(0.65, 0.9, txtstr, transform=ax1.transAxes)
    #plt.axhline(0, linewidth=0.75, linestyle = '--', color = 'darkgrey', zorder=1)
    #plt.axvline(30, linewidth=0.75, linestyle = ':', color = 'darkgrey', zorder=1)
    if lbl == r'\textit{Chandra} X-ray':
      sym = 'o'
    if lbl == r'HI Ram Pressure':
      sym = 's'
    if lbl == r'Average Profile ($z<0.1$)':
      sym = 'd'
    if lbl == r'HI Ram Pressure':
      #plt.scatter(x, y, color=colour, s=12, marker=sym, label=lbl, zorder=2)#, linestyle=lsty)#, label=lbl, linewidth=0.9, zorder=2)
      plt.errorbar(x, y, xerr = yerr, color=colour, markersize=4.5, fmt=sym, elinewidth=0.85, label = lbl, zorder=2)
    if lbl == r'\textit{Chandra} X-ray':
      plt.scatter(x, y, color=colour, s=10, marker=sym, label=lbl, zorder=2)
      plt.fill_between(x, y-yerr, y+yerr, alpha=0.25, edgecolor='none', facecolor=colour)
      #plt.errorbar(x, y, yerr = yerr, color=colour, markersize=2, fmt='o', elinewidth=0.75, label = lbl, zorder=2)
    if lbl == r'Average Profile ($z<0.1$)' or lbl == r'Average Profile ($1.2<z<1.9$)':
      plt.plot(x, y, color=colour, linestyle='--', linewidth=1.5, label=lbl, zorder=1)
      plt.fill_between(x, y-yerr, y+yerr, alpha=0.25, edgecolor='none', facecolor=colour)
    ax1.legend(loc='lower left', fontsize=11)#, title=legtitle, fontsize=9.5, title_fontsize=9.5)#, ncol=2)
    plt.subplots_adjust(wspace=0.20, hspace=0.20)
    ax1.get_yaxis().set_tick_params(which='both', direction='in')
    ax1.get_xaxis().set_tick_params(which='both', direction='in')


# ================================ #
# ===== IGM Beta Density Plot ==== #
# ================================ #
def igm_beta_density_profile_plot(fig_num, subfig, x, y, yerr, colour, lbl):
    matplotlib.rcParams.update({'font.size': 12}) 
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(subfig, facecolor = 'w')
    ax1.set_ylabel(r'$n_{\mathrm{IGM}}$ [cm$^{-3}$]')
    ax1.set_xlabel(r'Radius [Mpc]')
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_xlim(0.008, 5)
    ax1.set_ylim(0.000005, 0.1)
    #labels = [item.get_text() for item in ax1.get_xticklabels()]
    #labels[0] = 0.001
    #labels[1] = 0.01
    #labels[2] = 0.1
    #labels[3] = 1
    #ax1.set_xticklabels([0.001, 0.01, 0.1, 1])
    formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))
    ax1.xaxis.set_major_formatter(formatter)
    if lbl == r'HI Ram Pressure':
      #sym = 's'
      sym = '|'
    if lbl == r'HI Ram Pressure':
      plt.errorbar(x, y, xerr = yerr, xlolims=True, color=colour, markersize=8, fmt=sym, elinewidth=1, label = lbl, zorder=2)
    if lbl == r'$\beta$-model':
      plt.plot(x, y, color=colour, linestyle='--', linewidth=1.5, label=lbl, zorder=1)
      plt.fill_between(x, y-yerr, y+yerr, alpha=0.25, edgecolor='none', facecolor=colour)
    if lbl == r'\textit{Chandra} X-ray':
      sym = 'o'
      plt.scatter(x, y, color=colour, s=10, marker=sym, label=lbl, zorder=2)
    ax1.legend(loc='lower left', fontsize=11)#, title=legtitle, fontsize=9.5, title_fontsize=9.5)#, ncol=2)
    plt.subplots_adjust(wspace=0.20, hspace=0.20)
    ax1.get_yaxis().set_tick_params(which='both', direction='in')
    ax1.get_xaxis().set_tick_params(which='both', direction='in')


# ================================ #
# === Phase Space Diagram Plot === #
# ================================ #
def phase_space_diagram_plot(fig_num, subfig, x, y, xgal, ygal, colour, lbl):
    matplotlib.rcParams.update({'font.size': 12}) 
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(subfig, facecolor = 'w')
    ax1.set_ylabel(r'$\Delta v/\sigma$')
    ax1.set_xlabel(r'$r/R_{200}$')
    #ax1.set_yscale('log')
    #ax1.set_xscale('log')
    ax1.set_xlim(0, 1.8)
    ax1.set_ylim(0, 2.85)
    #if lsty == '-':
      #plt.text(0.65, 0.9, txtstr, transform=ax1.transAxes)
    #plt.axhline(0, linewidth=0.75, linestyle = '--', color = 'darkgrey', zorder=1)
    #plt.axvline(30, linewidth=0.75, linestyle = ':', color = 'darkgrey', zorder=1)
    plt.plot(x, y, color=colour, linestyle='-', linewidth=1.5, label=lbl, zorder=1)
    ax1.scatter(xgal, ygal, s=60, color='peru', zorder=5)
    plt.fill_between(x, 0*y, y, alpha=0.5, edgecolor='none', facecolor='lightblue')
    plt.plot([0, 1.2], [1.5, 0], color='black', linestyle='--', linewidth=1.5, label=lbl, zorder=1)
    plt.fill_between([0, 1.2], [0, 0], [1.5, 0], alpha=1, edgecolor='none', facecolor='lightgrey')
    plt.text(0.05, 0.10, 'Virialised', transform=ax1.transAxes)
    plt.text(0.60, 0.10, 'Recent Infall\n (Gravitationally Bound)', transform=ax1.transAxes)
    #plt.text(0.55, 0.75, 'Recent Infall', transform=ax1.transAxes) #\n (Not Gravitationally Bound)
    plt.text(0.55, 0.75, 'Not Gravitationally Bound', transform=ax1.transAxes) #\n (Not Gravitationally Bound)
    plt.text(0.45, 0.53, r'ESO\,501-G075', transform=ax1.transAxes)
    #ax1.legend(loc='lower left', fontsize=11)#, title=legtitle, fontsize=9.5, title_fontsize=9.5)#, ncol=2)
    plt.subplots_adjust(wspace=0.20, hspace=0.20)
    ax1.get_yaxis().set_tick_params(which='both', direction='in')
    ax1.get_xaxis().set_tick_params(which='both', direction='in')


# ================================ #
# === Phase Space Diagram Plot === #
# ================================ #
def psd_asym_gal_plot(fig_num, subfig, x, y, xgal, ygal, colour, msty, lbl):
    matplotlib.rcParams.update({'font.size': 12}) 
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(subfig, facecolor = 'w')
    ax1.set_ylabel(r'$\Delta v/\sigma$')
    ax1.set_xlabel(r'$r/R_{200}$')
    #ax1.set_yscale('log')
    #ax1.set_xscale('log')
    ax1.set_xlim(0, 1.8)
    ax1.set_ylim(0, 2.85)
    #if lsty == '-':
      #plt.text(0.65, 0.9, txtstr, transform=ax1.transAxes)
    #plt.axhline(0, linewidth=0.75, linestyle = '--', color = 'darkgrey', zorder=1)
    #plt.axvline(30, linewidth=0.75, linestyle = ':', color = 'darkgrey', zorder=1)
    if lbl != 'none':
      if lbl == 'J104059-270456':
        plt.plot(x, y, color='black', linestyle='-', linewidth=1.5, zorder=1) #label=lbl, 
        plt.fill_between(x, 0*y, y, alpha=1, edgecolor='none', facecolor='lightgrey')
        plt.plot([0, 1.2], [1.5, 0], color='black', linestyle='--', linewidth=1.5, zorder=1) #label=lbl, 
        plt.fill_between([0, 1.2], [0, 0], [1.5, 0], alpha=1, edgecolor='none', facecolor='darkgrey')
        plt.text(0.05, 0.10, 'Virialised', transform=ax1.transAxes)
        plt.text(0.60, 0.10, 'Recent Infall\n (Gravitationally Bound)', transform=ax1.transAxes)
        ax1.scatter(xgal, ygal, s=80, marker=msty, color=colour, zorder=5, label=lbl, facecolor=colour, linewidth=2)
      else:
        ax1.scatter(xgal, ygal, s=80, marker=msty, color=colour, zorder=5, label=lbl, facecolor='none', linewidth=2)
    else:
      ax1.scatter(xgal, ygal, s=5, marker=msty, color=colour, edgecolor='none', zorder=4, linewidth=2)
    #plt.text(0.55, 0.75, 'Recent Infall', transform=ax1.transAxes) #\n (Not Gravitationally Bound)
    #plt.text(0.55, 0.75, 'Not Gravitationally Bound', transform=ax1.transAxes) #\n (Not Gravitationally Bound)
    #plt.text(0.45, 0.53, r'ESO\,501-G075', transform=ax1.transAxes)
    ax1.legend(fontsize=9)#, title=legtitle, fontsize=9.5, title_fontsize=9.5)#, ncol=2) loc='lower left', 
    plt.subplots_adjust(wspace=0.20, hspace=0.20)
    ax1.get_yaxis().set_tick_params(which='both', direction='in')
    ax1.get_xaxis().set_tick_params(which='both', direction='in')

# ================================ #
# === Phase Space Diagram Plot === #
# ================================ #
def psd_cluster_plot(fig_num, subfig, xgal, ygal, marker, col_scat, facecolor, label, escape_curve):
    matplotlib.rcParams.update({'font.size': 14}) 
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(subfig, facecolor = 'w')
    ax1.set_ylabel(r'$\Delta v/\sigma_{\rm{disp}}$')
    ax1.set_xlabel(r'$r/R_{200}$')
    #ax1.set_yscale('log')
    #ax1.set_xscale('log')
    ax1.set_xlim(0, 5.75)
    ax1.set_ylim(0, 4.5)
    #if lsty == '-':
      #plt.text(0.65, 0.9, txtstr, transform=ax1.transAxes)
    #plt.axhline(0, linewidth=0.75, linestyle = '--', color = 'darkgrey', zorder=1)
    #plt.axvline(30, linewidth=0.75, linestyle = ':', color = 'darkgrey', zorder=1)
    plt.plot(escape_curve[0], escape_curve[1], color=escape_curve[2], 
             linestyle=escape_curve[3], linewidth=1, zorder=1)
    if label != False:
      ax1.scatter(xgal, ygal, marker=marker, color=col_scat, facecolor=facecolor, edgecolor=col_scat, s=18, zorder=3, label=label)
    else:
      ax1.scatter(xgal, ygal, marker=marker, color=col_scat, facecolor=facecolor, edgecolor=col_scat, s=18, zorder=3)
    #plt.fill_between(x, 0*y, y, alpha=0.5, edgecolor='none', facecolor='lightblue')
    #plt.plot([0, 1.2], [1.5, 0], color='grey', linestyle='--', linewidth=1.5, zorder=1)
    plt.plot([0, 0.5], [2.0, 0], color='red', linestyle='-', linewidth=1, zorder=1)
    #plt.plot([0, 0.4, 0.6], [2.5, 2.5, 1.5], color='green', linestyle='-', linewidth=1, zorder=1)
    #plt.plot([0.15, 0.9, 1.1], [1.5, 1.5, 0.5], color='orange', linestyle='-', linewidth=1, zorder=1)
    #plt.plot([0.4, 1.4, 1.5], [0.5, 0.5, 0], color='pink', linestyle='-', linewidth=1, zorder=1)
    #plt.plot([0.4, 3], [2.5, 1], color='blue', linestyle='-', linewidth=1, zorder=1)
    #plt.fill_between([0, 1.2], [0, 0], [1.5, 0], alpha=1, edgecolor='none', facecolor='lightgrey')
    #plt.text(0.05, 0.10, 'Virialised', transform=ax1.transAxes)
    #plt.text(0.60, 0.10, 'Recent Infall\n (Gravitationally Bound)', transform=ax1.transAxes)
    #plt.text(0.55, 0.75, 'Recent Infall\n (Not Gravitationally Bound)', transform=ax1.transAxes)
    #plt.text(0.40, 0.53, r'ESO\,501-G075', transform=ax1.transAxes)
    #ax1.legend(fontsize=12, ncol=1)#, title=legtitle, fontsize=9.5, title_fontsize=9.5)#, ncol=2)
    if label == 'Cluster':
      x, y = -10, -10
      sym_array = ['o', 'o', 'o', 's', 's', 's']
      col_array = ['darkblue', 'mediumvioletred', 'peru', 'lightblue', 'violet', 'sandybrown']
      pt_array  = ['Cluster', 'Infall', 'Field', 'Cluster', 'Infall', 'Field']
      p3 = ax1.scatter(x, y, color=col_array[0], marker=sym_array[0], edgecolor=col_array[0], 
                       facecolor=col_array[0], s=18, zorder=3)
      p4 = ax1.scatter(x, y, color=col_array[1], marker=sym_array[1], edgecolor=col_array[1], 
                       facecolor=col_array[1], s=18, zorder=3)
      p5 = ax1.scatter(x, y, color=col_array[2], marker=sym_array[2], edgecolor=col_array[2], 
                       facecolor=col_array[2], s=18, zorder=3)
      p6 = ax1.scatter(x, y, color=col_array[3], marker=sym_array[3], edgecolor=col_array[3], 
                       facecolor='none', s=18, zorder=3)
      p7 = ax1.scatter(x, y, color=col_array[4], marker=sym_array[4], edgecolor=col_array[4], 
                       facecolor='none', s=18, zorder=3)
      p8 = ax1.scatter(x, y, color=col_array[5], marker=sym_array[5], edgecolor=col_array[5], 
                       facecolor='none', s=18, zorder=3)
      #l1 = ax1.legend([p1, p2], [txtstr[0], txtstr[1]], fontsize = 11, loc = 'upper right', ncol=1)
      l2 = ax1.legend(handles=[p3, p4, p5, p6, p7, p8],
          labels=['', '', '', 'Cluster', 'Infall', 'Field'],
          loc='upper center', ncol=2, handlelength=2, fontsize = 12,
          handletextpad=0, columnspacing=-0.5, borderpad=0.2)
    
    plt.subplots_adjust(wspace=0.20, hspace=0.20)
    ax1.get_yaxis().set_tick_params(which='both', direction='in')
    ax1.get_xaxis().set_tick_params(which='both', direction='in')


def VHS_contour_plot(fig_num, subfig3, image1, image2):
    matplotlib.rcParams.update({'font.size': 15})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    if subfig3 == 1:
      subplot_params = [0.05,0.05,0.45,0.95]
    if subfig3 == 2:
      subplot_params = [0.5,0.05,0.45,0.95]
    ax1 = aplpy.FITSFigure(image1, figure=fig_num, subplot=subplot_params)
    if subfig3 == 1 or subfig3 == 3:
      ax1.show_colorscale(vmin=0,vmax=2000,cmap='Greys')
      ax1.axis_labels.set_ytext('Declination')
    if subfig3 == 2 or subfig3 == 4:
      ax1.show_colorscale(vmin=0,vmax=2000,cmap='Greys')
      ax1.axis_labels.hide_y()
      ax1.tick_labels.hide_y()
    #if subfig3 == 3 or subfig3 == 4:
    ax1.axis_labels.set_xtext('Right Ascension')
    #if subfig3 == 1 or subfig3 == 2:
      #ax1.axis_labels.hide_x()
    ax1.tick_labels.set_xformat('hh:mm:ss')
    ax1.tick_labels.set_yformat('dd:mm')
    ax1.ticks.show()
    ax1.ticks.set_length(5)
    ax1.ticks.set_color('black')
    ax1.ticks.set_xspacing(0.06)
    ax1.ticks.set_minor_frequency(1)
    position      = SkyCoord('10h40m59s', '-27d04m56s', frame='icrs')
    #if subfig3 == 1 or subfig3 == 3:
    width, height = 300./60./60., 300./60./60. #240./60./60., 240./60./60.
    #if subfig3 == 2 or subfig3 == 4:
      #width, height = 420./60./60., 420./60./60. #240./60./60., 240./60./60.
    ax1.recenter(position.ra, position.dec, width=width, height=height)
    #askap_lvl = [5.*10**19*(39.01*34.06)/(1.36*21*21*1.823*1000*(10**18))]
    askap_lvl = [5. * 10**19 * (30. * 30.) / (2.33 * 10**20)]
    print(askap_lvl)
    ax1.show_contour(image2, colors="blue", levels=askap_lvl, slices=(0,1))
    if subfig3 == 1:
      ax1.add_label(0.2, 0.05, r'VHS $J-$band', relative=True)
    if subfig3 == 2:
      ax1.add_label(0.2, 0.05, r'VHS $K-$band', relative=True)
    #if subfig3 == 3:
      #ax1.add_label(0.32, 0.05, r'c) $GALEX$ FUV', relative=True)
    #if subfig3 == 4:
      #ax1.add_label(0.33, 0.05, r'd) VHS $K-$band', relative=True)
      
      
def VHS_GALEX_contour_plot(fig_num, subfig3, image1, image2, image3):
    matplotlib.rcParams.update({'font.size': 15})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    if subfig3 == 1:
      subplot_params = [0.05,0.05,0.45,0.95]
    if subfig3 == 2:
      subplot_params = [0.5,0.05,0.45,0.95]
    ax1 = aplpy.FITSFigure(image1, figure=fig_num, subplot=subplot_params)
    if subfig3 == 1 or subfig3 == 3:
      ax1.show_colorscale(vmin=0,vmax=2000,cmap='Greys')
      ax1.axis_labels.set_ytext('Declination')
    if subfig3 == 2 or subfig3 == 4:
      ax1.show_colorscale(vmin=0,vmax=2000,cmap='Greys')
      ax1.axis_labels.hide_y()
      ax1.tick_labels.hide_y()
    #if subfig3 == 3 or subfig3 == 4:
    ax1.axis_labels.set_xtext('Right Ascension')
    #if subfig3 == 1 or subfig3 == 2:
      #ax1.axis_labels.hide_x()
    ax1.tick_labels.set_xformat('hh:mm:ss')
    ax1.tick_labels.set_yformat('dd:mm')
    ax1.ticks.show()
    ax1.ticks.set_length(5)
    ax1.ticks.set_color('black')
    ax1.ticks.set_xspacing(0.06)
    ax1.ticks.set_minor_frequency(1)
    position      = SkyCoord('10h40m59s', '-27d04m56s', frame='icrs')
    #if subfig3 == 1 or subfig3 == 3:
    width, height = 300./60./60., 300./60./60. #240./60./60., 240./60./60.
    #if subfig3 == 2 or subfig3 == 4:
      #width, height = 420./60./60., 420./60./60. #240./60./60., 240./60./60.
    ax1.recenter(position.ra, position.dec, width=width, height=height)
    askap_lvl = [5.*10**19*(39.01*34.06)/(1.36*21*21*1.823*1000*(10**18))]
    print(askap_lvl)
    ax1.show_contour(image2, colors="blue", levels=askap_lvl, slices=(0,1))
    galex_lvl = [0.0075, 0.01, 0.02, 0.03, 0.04]
    ax1.show_contour(image3, colors="magenta", levels=galex_lvl, slices=(0,1), smooth=3, linewidths=1)
    if subfig3 == 1:
      ax1.add_label(0.2, 0.05, r'VHS $J-$band', relative=True)
    if subfig3 == 2:
      ax1.add_label(0.2, 0.05, r'VHS $K-$band', relative=True)
    #if subfig3 == 3:
      #ax1.add_label(0.32, 0.05, r'c) $GALEX$ FUV', relative=True)
    #if subfig3 == 4:
      #ax1.add_label(0.33, 0.05, r'd) VHS $K-$band', relative=True)


def VHS_GALEX_continuum_contour_plot(fig_num, subfig3, image1, image2, image3, image4):
    matplotlib.rcParams.update({'font.size': 15})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    if subfig3 == 1:
      subplot_params = [0.05,0.05,0.45,0.95]
    if subfig3 == 2:
      subplot_params = [0.5,0.05,0.45,0.95]
    ax1 = aplpy.FITSFigure(image1, figure=fig_num, subplot=subplot_params)
    if subfig3 == 1 or subfig3 == 3:
      ax1.show_colorscale(vmin=0,vmax=2000,cmap='Greys')
      ax1.axis_labels.set_ytext('Declination')
    if subfig3 == 2 or subfig3 == 4:
      ax1.show_colorscale(vmin=0,vmax=2000,cmap='Greys')
      ax1.axis_labels.hide_y()
      ax1.tick_labels.hide_y()
    #if subfig3 == 3 or subfig3 == 4:
    ax1.axis_labels.set_xtext('Right Ascension')
    #if subfig3 == 1 or subfig3 == 2:
      #ax1.axis_labels.hide_x()
    ax1.tick_labels.set_xformat('hh:mm:ss')
    ax1.tick_labels.set_yformat('dd:mm')
    ax1.ticks.show()
    ax1.ticks.set_length(5)
    ax1.ticks.set_color('black')
    ax1.ticks.set_xspacing(0.06)
    ax1.ticks.set_minor_frequency(1)
    position      = SkyCoord('10h40m59s', '-27d04m56s', frame='icrs')
    #if subfig3 == 1 or subfig3 == 3:
    width, height = 300./60./60., 300./60./60. #240./60./60., 240./60./60.
    #if subfig3 == 2 or subfig3 == 4:
      #width, height = 420./60./60., 420./60./60. #240./60./60., 240./60./60.
    ax1.recenter(position.ra, position.dec, width=width, height=height)
    askap_lvl = [5.*10**19*(39.01*34.06)/(1.36*21*21*1.823*1000*(10**18))]
    #print(askap_lvl)
    ax1.show_contour(image2, colors="blue", levels=askap_lvl, slices=(0,1))
    if subfig3 == 1:
      #galex_lvl = [0.0075, 0.01, 0.02, 0.03, 0.04]
      galex_lvl = [0.0075, 0.02, 0.04]
      ax1.show_contour(image3, colors="magenta", levels=galex_lvl, slices=(0,1), smooth=3, linewidths=1.5)
    if subfig3 == 2:
      galex_lvl = [0.0015, 0.0025, 0.005, 0.0075]
      ax1.show_contour(image3, colors="magenta", levels=galex_lvl, slices=(0,1), smooth=5, linewidths=1.5)
    #contiuum_lvl = [0.000008, 0.00002, 0.00004, 0.00006, 0.00008, 0.0001, 0.00012]
    contiuum_lvl = [0.000008, 0.00002, 0.00006, 0.0001]
    ax1.show_contour(image4, colors="green", levels=contiuum_lvl, slices=(0,1), smooth=5, linewidths=2, linestyles='--')
    if subfig3 == 1:
      ax1.add_label(0.2, 0.05, r'\textit{GALEX} NUV', relative=True)
    if subfig3 == 2:
      ax1.add_label(0.2, 0.05, r'\textit{GALEX} FUV', relative=True)
    #if subfig3 == 3:
      #ax1.add_label(0.32, 0.05, r'c) $GALEX$ FUV', relative=True)
    #if subfig3 == 4:
      #ax1.add_label(0.33, 0.05, r'd) VHS $K-$band', relative=True)


def VHS_FUV_continuum_contour_plot(fig_num, subfig3, image1, image2, image3, image4):
    matplotlib.rcParams.update({'font.size': 15})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    if subfig3 == 1:
      subplot_params = [0.05,0.05,0.95,0.95]
    #if subfig3 == 2:
      #subplot_params = [0.5,0.05,0.45,0.95]
    ax1 = aplpy.FITSFigure(image1, figure=fig_num, subplot=subplot_params)
    if subfig3 == 1 or subfig3 == 3:
      ax1.show_colorscale(vmin=0,vmax=2000,cmap='Greys')
      ax1.axis_labels.set_ytext('Declination')
    if subfig3 == 2 or subfig3 == 4:
      ax1.show_colorscale(vmin=0,vmax=2000,cmap='Greys')
      ax1.axis_labels.hide_y()
      ax1.tick_labels.hide_y()
    #if subfig3 == 3 or subfig3 == 4:
    ax1.axis_labels.set_xtext('Right Ascension')
    #if subfig3 == 1 or subfig3 == 2:
      #ax1.axis_labels.hide_x()
    ax1.tick_labels.set_xformat('hh:mm:ss')
    ax1.tick_labels.set_yformat('dd:mm')
    ax1.ticks.show()
    ax1.ticks.set_length(5)
    ax1.ticks.set_color('black')
    ax1.ticks.set_xspacing(0.06)
    ax1.ticks.set_minor_frequency(1)
    position      = SkyCoord('10h40m59s', '-27d04m56s', frame='icrs')
    #if subfig3 == 1 or subfig3 == 3:
    width, height = 300./60./60., 300./60./60. #240./60./60., 240./60./60.
    #if subfig3 == 2 or subfig3 == 4:
      #width, height = 420./60./60., 420./60./60. #240./60./60., 240./60./60.
    ax1.recenter(position.ra, position.dec, width=width, height=height)
    askap_lvl = [5. * 10**19 * (30. * 30.) / (2.33 * 10**20)]  #(1.36*21*21*1.823*1000*(10**18))]
    #print(askap_lvl)
    ax1.show_contour(image2, colors="blue", levels=askap_lvl, slices=(0,1))
    #if subfig3 == 1:
      ##galex_lvl = [0.0075, 0.01, 0.02, 0.03, 0.04]
      #galex_lvl = [0.0075, 0.02, 0.04]
      #ax1.show_contour(image3, colors="magenta", levels=galex_lvl, slices=(0,1), smooth=3, linewidths=1.5)
    #if subfig3 == 2:
    galex_lvl = [0.0015, 0.0025, 0.005, 0.0075]
    ax1.show_contour(image3, colors="magenta", levels=galex_lvl, slices=(0,1), smooth=5, linewidths=1.5)
    #contiuum_lvl = [0.000008, 0.00002, 0.00004, 0.00006, 0.00008, 0.0001, 0.00012]
    contiuum_lvl = [0.000008, 0.00002, 0.00006, 0.0001]
    ax1.show_contour(image4, colors="green", levels=contiuum_lvl, slices=(0,1), smooth=5, linewidths=2, linestyles='--')
    #if subfig3 == 1:
      #ax1.add_label(0.2, 0.05, r'\textit{GALEX} NUV', relative=True)
    #if subfig3 == 2:
      #ax1.add_label(0.2, 0.05, r'\textit{GALEX} FUV', relative=True)
    #if subfig3 == 3:
      #ax1.add_label(0.32, 0.05, r'c) $GALEX$ FUV', relative=True)
    #if subfig3 == 4:
      #ax1.add_label(0.33, 0.05, r'd) VHS $K-$band', relative=True)


def GALEX_contour_plot(fig_num, subfig3, image1, image2):
    matplotlib.rcParams.update({'font.size': 15})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    if subfig3 == 1:
      subplot_params = [0.05,0.05,0.45,0.95]
    if subfig3 == 2:
      subplot_params = [0.5,0.05,0.45,0.95]
    ax1 = aplpy.FITSFigure(image1, figure=fig_num, subplot=subplot_params)
    if subfig3 == 1 or subfig3 == 3:
      ax1.show_colorscale(vmin=0,vmax=0.03,cmap='Greys')
      ax1.axis_labels.set_ytext('Declination')
    if subfig3 == 2 or subfig3 == 4:
      ax1.show_colorscale(vmin=0,vmax=0.01,cmap='Greys')
      ax1.axis_labels.hide_y()
      ax1.tick_labels.hide_y()
    #if subfig3 == 3 or subfig3 == 4:
    ax1.axis_labels.set_xtext('Right Ascension')
    #if subfig3 == 1 or subfig3 == 2:
      #ax1.axis_labels.hide_x()
    ax1.tick_labels.set_xformat('hh:mm:ss')
    ax1.tick_labels.set_yformat('dd:mm')
    ax1.ticks.show()
    ax1.ticks.set_length(5)
    ax1.ticks.set_color('black')
    ax1.ticks.set_xspacing(0.06)
    ax1.ticks.set_minor_frequency(1)
    position      = SkyCoord('10h40m59s', '-27d04m56s', frame='icrs')
    #if subfig3 == 1 or subfig3 == 3:
    width, height = 300./60./60., 300./60./60. #240./60./60., 240./60./60.
    #if subfig3 == 2 or subfig3 == 4:
      #width, height = 420./60./60., 420./60./60. #240./60./60., 240./60./60.
    ax1.recenter(position.ra, position.dec, width=width, height=height)
    askap_lvl = [5.*10**19*(39.01*34.06)/(1.36*21*21*1.823*1000*(10**18))]
    print(askap_lvl)
    ax1.show_contour(image2, colors="blue", levels=askap_lvl, slices=(0,1))
    galex_lvl = [0.008, 0.01, 0.02, 0.03, 0.04]
    ax1.show_contour(image1, colors="magenta", levels=galex_lvl, slices=(0,1), smooth=3, linewidths=1)
    if subfig3 == 1:
      ax1.add_label(0.2, 0.05, r'$NUV$', relative=True)
    if subfig3 == 2:
      ax1.add_label(0.2, 0.05, r'$FUV$', relative=True)
    #if subfig3 == 3:
      #ax1.add_label(0.32, 0.05, r'c) $GALEX$ FUV', relative=True)
    #if subfig3 == 4:
      #ax1.add_label(0.33, 0.05, r'd) VHS $K-$band', relative=True)



def VHS_xray_contour_plot(fig_num, subfig3, image1, image2, image3, image4):#, image5, image6):
    matplotlib.rcParams.update({'font.size': 15})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    if subfig3 == 1:
      subplot_params = [0.05,0.05,0.45,0.95]
    if subfig3 == 2:
      subplot_params = [0.5,0.05,0.45,0.95]
    ax1 = aplpy.FITSFigure(image1, figure=fig_num, subplot=subplot_params)
    if subfig3 == 1 or subfig3 == 3:
      ax1.show_colorscale(vmin=0,vmax=2000,cmap='Greys')
      ax1.axis_labels.set_ytext('Declination')
    if subfig3 == 2 or subfig3 == 4:
      ax1.show_colorscale(vmin=0,vmax=2000,cmap='Greys')
      ax1.axis_labels.hide_y()
      ax1.tick_labels.hide_y()
    #if subfig3 == 3 or subfig3 == 4:
    ax1.axis_labels.set_xtext('Right Ascension')
    #if subfig3 == 1 or subfig3 == 2:
      #ax1.axis_labels.hide_x()
    ax1.tick_labels.set_xformat('hh:mm:ss')
    ax1.tick_labels.set_yformat('dd:mm')
    ax1.ticks.show()
    ax1.ticks.set_length(5)
    ax1.ticks.set_color('black')
    ax1.ticks.set_xspacing(0.06)
    ax1.ticks.set_minor_frequency(1)
    position      = SkyCoord('10h37m49s', '-27d07m15s', frame='icrs')
    #if subfig3 == 1 or subfig3 == 3:
    width, height = 480./60./60., 480./60./60. #240./60./60., 240./60./60.
    #if subfig3 == 2 or subfig3 == 4:
      #width, height = 420./60./60., 420./60./60. #240./60./60., 240./60./60.
    ax1.recenter(position.ra, position.dec, width=width, height=height)
    xray_lvl = [0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.0015]
    ax1.show_contour(image3, colors="magenta", levels=xray_lvl, slices=(0,1), smooth=3, linewidths=1)
    xray_lvl = [0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.002]
    ax1.show_contour(image4, colors="peru", levels=xray_lvl, slices=(0,1), smooth=1, linewidths=0.9)
    #xray_lvl = [0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.002]
    #ax1.show_contour(image5, colors="green", levels=xray_lvl, slices=(0,1), smooth=1, linewidths=0.9)
    #xray_lvl = [0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.002]
    #ax1.show_contour(image6, colors="red", levels=xray_lvl, slices=(0,1), smooth=1, linewidths=0.9)
    askap_lvl = [5.*10**19*(39.01*34.06)/(1.36*21*21*1.823*1000*(10**18))]
    print(askap_lvl)
    ax1.show_contour(image2, colors="blue", levels=askap_lvl, slices=(0,1))
    if subfig3 == 1:
      ax1.add_label(0.2, 0.05, r'VHS $J-$band', relative=True)
    if subfig3 == 2:
      ax1.add_label(0.2, 0.05, r'VHS $K-$band', relative=True)
    #if subfig3 == 3:
      #ax1.add_label(0.32, 0.05, r'c) $GALEX$ FUV', relative=True)
    #if subfig3 == 4:
      #ax1.add_label(0.33, 0.05, r'd) VHS $K-$band', relative=True)



def ring_map_plot(fig_num, sub1, sub2, sub3, image1, image2):
    matplotlib.rcParams.update({'font.size': 15})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = aplpy.FITSFigure(image1, figure=fig_num, subplot=(sub1, sub2, sub3))
    #if subfig3 == 1 or subfig3 == 3:
    ax1.show_colorscale(vmin=0,vmax=2000,cmap='Greys')
    ax1.axis_labels.set_ytext('Declination')
    #if subfig3 == 2 or subfig3 == 4:
      #ax1.show_colorscale(vmin=0,vmax=2000,cmap='Greys')
      #ax1.axis_labels.hide_y()
      #ax1.tick_labels.hide_y()
    #if subfig3 == 3 or subfig3 == 4:
    ax1.axis_labels.set_xtext('Right Ascension')
    #if subfig3 == 1 or subfig3 == 2:
      #ax1.axis_labels.hide_x()
    ax1.tick_labels.set_xformat('hh:mm:ss')
    ax1.tick_labels.set_yformat('dd:mm')
    ax1.ticks.show()
    ax1.ticks.set_length(5)
    ax1.ticks.set_color('black')
    ax1.ticks.set_xspacing(0.06)
    ax1.ticks.set_minor_frequency(1)
    #position      = SkyCoord('10h40m59s', '-27d04m56s', frame='icrs')
    position      = SkyCoord('10h40m59.07s', '-27d05m02.5s', frame='icrs')
    #if subfig3 == 1 or subfig3 == 3:
    width, height = 300./60./60., 300./60./60. #240./60./60., 240./60./60.
    #if subfig3 == 2 or subfig3 == 4:
      #width, height = 420./60./60., 420./60./60. #240./60./60., 240./60./60.
    ax1.recenter(position.ra, position.dec, width=width, height=height)
    #askap_lvl = [5.*10**19*(30.*30.)/(1.36*21.*21.*1.823*1000*(10**18))]
    askap_lvl = [5. * 10**19 * (30. * 30.) / (2.33 * 10**20)]
    print(askap_lvl)
    ax1.show_contour(image2, colors="darkblue", levels=askap_lvl, slices=(0,1), linewidths=1.5, zorder=1)
    #for i in range(30):
      #a = (5. + i * 5.) / 3600.
      #b = a * np.sqrt(1 - np.sin(55. * math.pi / 180.)**2)
      #ax1.show_ellipses(position.ra.value, position.dec.value, width=a, height=b, angle=125., 
                      #facecolor='none', edgecolor='peru', ls='-')
    for i in range(9):
      a = 2. * (7.5 + i * 15.) / 3600.
      b = a * np.sqrt(1 - np.sin(55. * math.pi / 180.)**2)
      ax1.show_ellipses(position.ra.value, position.dec.value, width=a, height=b, angle=125., 
                      facecolor='none', edgecolor='peru', ls='-', zorder=2, coords_frame='world')
    #if subfig3 == 1:
      #ax1.add_label(0.2, 0.05, r'VHS $J-$band', relative=True)
    #if subfig3 == 2:
      #ax1.add_label(0.2, 0.05, r'VHS $K-$band', relative=True)
    #if subfig3 == 3:
      #ax1.add_label(0.32, 0.05, r'c) $GALEX$ FUV', relative=True)
    #if subfig3 == 4:
      #ax1.add_label(0.33, 0.05, r'd) VHS $K-$band', relative=True)


def pv_plot_single(fig_num, subfig1, subfig2, subfig3, image, freq, flux, error, colour, lsty, txtstr, lbl, min_ra, max_ra, min_vel, max_vel):
  matplotlib.rcParams.update({'font.size': 12})
  rc('font',**{'family':'serif','serif':['Times']})
  rc('text', usetex=True)
  ax1 = fig_num.add_subplot(subfig1, subfig2, subfig3, facecolor = 'w')
  ax1.set_xlabel(r'Offset [arcsec]')
  ax1.set_ylabel(r'Velocity [km/s]')
  ax1.imshow(image, origin='lower', aspect='auto', extent = (min_ra,max_ra,max_vel,min_vel), cmap='Greys')
  plt.plot(freq, flux, fmt = lsty, color = colour, linewidth = 0.5, ms=3, label=lbl) #yerr=error,








