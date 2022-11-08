# ============================================================ #
# ===== Basic plots for analysis of HI scaling relations ===== #
# ============================================================ #

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
from astropy.cosmology import z_at_value, WMAP7
from astropy.coordinates import EarthLocation, SkyCoord, ICRS, Angle, match_coordinates_3d, Distance
#from astropy.coordinates import EarthLocation, SkyCoord, FK5, ICRS, match_coordinates_3d, Distance
from astropy.wcs import WCS, find_all_wcs
from astropy.io import fits
from astropy import units as u
from astropy.modeling.models import Ellipse2D, Disk2D, Gaussian2D, Sersic1D
from astropy.stats import median_absolute_deviation as madev
from astropy.stats import binom_conf_interval
from astropy.modeling.rotations import Rotation2D
from astropy.io.votable import parse_single_table
from astropy.convolution import convolve, Gaussian1DKernel
from astropy.table import Table, join, Column, vstack
from random import shuffle as sfl
#from photutils.centroids import centroid_com
import scipy
from scipy import stats
from scipy.special import erf
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.ndimage.interpolation import rotate as scirotate
from scipy.stats import binned_statistic, linregress, ks_2samp

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



def scatter_outlier_plot(fig_num, sub1, sub2, sub3, x, y, txtstr, xlbl, ylbl, marker, do_legend):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    #if sub1 == 1 and sub2 == 1:
    ax1.set_xlabel(xlbl)
    ax1.set_ylabel(ylbl)
    ax1.scatter(x, y, color=marker[0], marker=marker[1], 
                edgecolor=marker[0], facecolor=marker[2], s=marker[3], zorder=1, label = txtstr)
    if do_legend:
      ax1.legend(fontsize = 9) #loc='upper right', 
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

def scatter_error_plot(fig_num, sub1, sub2, sub3, x, y, error, txtstr, xlbl, ylbl, marker, do_legend):
    matplotlib.rcParams.update({'font.size': 14})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    #if sub1 == 1 and sub2 == 1:
    ax1.set_xlabel(xlbl)
    ax1.set_ylabel(ylbl)
    ax1.errorbar(x, y, yerr=error, linestyle='none', color=marker[0], marker=marker[1], 
                mec=marker[0], mfc=marker[2], ms=marker[3], zorder=1, label = txtstr)
    if do_legend:
      ax1.legend(fontsize = 12) #loc='upper right', 
    plt.subplots_adjust(wspace=0.3, hspace=0.3)


def scat_col_simple_plot2(fig_num, sub1, sub2, sub3, x, y, z, labels, marker, do_legend, do_cbar):
    matplotlib.rcParams.update({'font.size': 14})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    #if sub1 == 1 and sub2 == 1:
    txtstr = labels[0]
    xlbl   = labels[1]
    ylbl   = labels[2]
    zlbl   = labels[3]
    vmin   = marker[4]
    vmax   = marker[5]
    ax1.set_xlabel(xlbl)
    ax1.set_ylabel(ylbl)
    #scat_ax = ax1.scatter(x, y, c=z, cmap=cmap, marker=marker, facecolor='none', alpha=1, s=15, vmin=0, vmax=4, zorder=1, label = txtstr)
    scat_ax = ax1.scatter(x, y, c=z, cmap=marker[0], marker=marker[1], 
                          facecolor=marker[2], alpha=1, s=marker[3], 
                          vmin=vmin, vmax=vmax, zorder=1, label = txtstr)
                          #vmin=-1.5, vmax=1.6, zorder=1, label = txtstr)
    #vmin=0.1, vmax=1.6, zorder=1, label = txtstr)
    #vmin=8, vmax=10.5,
    #scat_ax.set_facecolor('none')
    if do_cbar:
      cbar = plt.colorbar(scat_ax, fraction=0.1, pad=0.01, label = zlbl) #cax=cbar_ax, 
    if do_legend:
      ax1.legend(fontsize = 12) #loc='upper right', 
    if do_cbar == False:
      scat_ax.set_facecolor('none')
    plt.subplots_adjust(wspace=0.5, hspace=0.35)


def scat_col_simple_plot3(fig_num, sub1, sub2, sub3, x, y, z, labels, marker, do_legend, do_cbar):
    matplotlib.rcParams.update({'font.size': 14})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    #if sub1 == 1 and sub2 == 1:
    txtstr = labels[0]
    xlbl   = labels[1]
    ylbl   = labels[2]
    zlbl   = labels[3]
    vmin   = marker[4]
    vmax   = marker[5]
    ax1.set_xlabel(xlbl)
    ax1.set_ylabel(ylbl)
    #scat_ax = ax1.scatter(x, y, c=z, cmap=cmap, marker=marker, facecolor='none', alpha=1, s=15, vmin=0, vmax=4, zorder=1, label = txtstr)
    scat_ax = ax1.scatter(x, y, c=z, cmap=marker[0], marker=marker[1], 
                          facecolor=marker[2], alpha=1, s=marker[3], 
                          vmin=vmin, vmax=vmax, zorder=1, label = txtstr)
                          #vmin=-1.5, vmax=1.6, zorder=1, label = txtstr)
    #vmin=0.1, vmax=1.6, zorder=1, label = txtstr)
    #vmin=8, vmax=10.5,
    #scat_ax.set_facecolor('none')
    if do_cbar:
      cax  = fig_num.add_axes([0.92, 0.11, 0.025, 0.77])
      cbar = plt.colorbar(scat_ax, cax=cax, fraction=0.1, pad=0.01, label = zlbl)#, ticks=[0.66, 1, 2, 3, 4, 5]) #cax=cbar_ax,
      cbar.ax.set_yticklabels(['2/3', '1', '2', '3', '4', r'$>5$'])
    if do_legend:
      ax1.legend(fontsize = 12) #loc='upper right', 
    plt.subplots_adjust(wspace=0.5, hspace=0.35)
    
def scat_col_simple_plot4(fig_num, sub1, sub2, sub3, x, y, z, labels, marker, do_legend, do_cbar):
    matplotlib.rcParams.update({'font.size': 14})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    #if sub1 == 1 and sub2 == 1:
    txtstr = labels[0]
    xlbl   = labels[1]
    ylbl   = labels[2]
    zlbl   = labels[3]
    vmin   = marker[4]
    vmax   = marker[5]
    ax1.set_xlabel(xlbl)
    ax1.set_ylabel(ylbl)
    #scat_ax = ax1.scatter(x, y, c=z, cmap=cmap, marker=marker, facecolor='none', alpha=1, s=15, vmin=0, vmax=4, zorder=1, label = txtstr)
    x = x[np.argsort(z)]
    y = y[np.argsort(z)]
    z = z[np.argsort(z)]
    scat_ax = ax1.scatter(x, y, c=z, cmap=marker[0], marker=marker[1], 
                          facecolor=marker[2], alpha=1, s=marker[3], 
                          vmin=vmin, vmax=vmax, zorder=1, label = txtstr)
                          #vmin=-1.5, vmax=1.6, zorder=1, label = txtstr)
    #vmin=0.1, vmax=1.6, zorder=1, label = txtstr)
    #vmin=8, vmax=10.5,
    #scat_ax.set_facecolor('none')
    if do_cbar:
      cax  = fig_num.add_axes([0.66, 0.545, 0.025, 0.33])
      cbar = plt.colorbar(scat_ax, cax=cax, fraction=0.05, pad=0.05, label = zlbl)#, ticks=[0.66, 1, 2, 3, 4, 5]) #cax=cbar_ax,
      cbar.ax.set_yticklabels(['2/3', '1', '2', '3', '4', r'$>5$'])
    if do_legend:
      ax1.legend(fontsize = 12) #loc='upper right', 
    plt.subplots_adjust(wspace=0.5, hspace=0.35)

def scatter_radial_profiles(input_dirs, input_fnames, output_fnames, data_to_plot):
    dir_hydra           = input_dirs[0]
    dir_ps_hydra        = dir_hydra + 'MULTIWAVELENGTH/PANSTARRS/'
    
    dir_ngc4636         = input_dirs[1]
    dir_ps_ngc4636      = dir_ngc4636 + 'MULTIWAVELENGTH/PANSTARRS/'
    
    fits_rp             = dir_hydra + 'PROFILES/' + input_fnames[0]
    hdu_rp              = fits.open(fits_rp)
    data_hi_rp_hydra    = hdu_rp[1].data
    
    fits_rp             = dir_ngc4636 + 'PROFILES/' + input_fnames[0]
    hdu_rp              = fits.open(fits_rp)
    data_hi_rp_ngc4636  = hdu_rp[1].data
    
    join_hi_rp          = join(data_hi_rp_hydra, data_hi_rp_ngc4636, keys='RADIUS')
    
    radii_hi            = join_hi_rp[data_hi_rp_hydra.columns.names[0]]
    
    
    fits_rp             = dir_ps_hydra + 'PROFILES_BKGDSUB/' + input_fnames[1]
    hdu_rp              = fits.open(fits_rp)
    data_r_rp_hydra     = hdu_rp[1].data
    
    fits_rp             = dir_ps_ngc4636 + 'PROFILES_BKGDSUB/' + input_fnames[1]
    hdu_rp              = fits.open(fits_rp)
    data_r_rp_ngc4636   = hdu_rp[1].data
    
    join_r_rp           = join(data_r_rp_hydra, data_r_rp_ngc4636, keys='RADIUS')
    
    radii_r             = join_r_rp[data_r_rp_hydra.columns.names[0]]
    
    profiles_hi = []
    for i in range(len(galaxies)):
      profiles_hi.append(join_hi_rp[galaxies[i]])
    
    profiles_hi = np.array(profiles_hi)
    
    profiles_r = []
    for i in range(len(galaxies)):
      profiles_r.append(join_r_rp[galaxies[i]])
    
    profiles_r = np.array(profiles_r)
    
    print(len(profiles_r))
    
    xpar        = data_to_plot[0]
    ypar        = data_to_plot[1]
    
    fig2        = plt.figure(2, figsize=(6, 6))
    
    subsample   = (data_to_plot[2] > data_to_plot[3]) & np.isfinite(xpar) & np.isfinite(ypar) #& (ypar < 10)
    #data_array  = [xpar[subsample], np.log10(ypar[subsample])]
    data_array   = [xpar[subsample], ypar[subsample]]
    data_array2  = [xpar[subsample & (np.array(data_to_plot[6])[2])], 
                    ypar[subsample & (np.array(data_to_plot[6])[2])]]
    # | np.array(data_to_plot[6])[5]
    data_array3  = [xpar[subsample & (np.array(data_to_plot[6])[0] | np.array(data_to_plot[6])[1])], 
                    ypar[subsample & (np.array(data_to_plot[6])[0] | np.array(data_to_plot[6])[1])]]
    
    
    profile_hi_sub = profiles_hi[subsample]
    profile_r_sub  = profiles_r[subsample]
    
    #profile_hi_sub = profiles_hi[subsample & (np.array(data_to_plot[6])[2])]
    #profile_r_sub  = profiles_r[subsample & (np.array(data_to_plot[6])[2])]
    
    result = linregress(data_array2)
    result3 = linregress(data_array)
    
    if data_to_plot[4] == r'$\log(M_*/[\mathrm{M}_{\odot}])$':
      xfit   = np.arange(6.5, 11.5, 0.25)
    elif data_to_plot[4] == r'$\log(\mu_*/[\rm{M}_{\odot}\,\rm{kpc}^{-2}])$':
      xfit   = np.arange(6, 9.5, 0.25)
    elif data_to_plot[4] == 'Radius':
      #xfit   = np.arange(0, 40, 0.25)
      xfit   = np.arange(0, 2, 0.25)
    else:
      xfit   = np.arange(6.5, 11.5, 0.25)
    yfit   = result[0] * xfit + result[1]
    dfit   = np.std(result[0] * data_array2[0] + result[1] - data_array2[1])
    offset = data_array[1] - (result[0] * data_array[0] + result[1])
    
    print(np.round(result,2), np.round(dfit, 2), len(data_array2[0]), 10**np.nanmean(data_array2[1]))
    
    yfit3   = result3[0] * xfit + result3[1]
    dfit3   = np.std(result3[0] * data_array[0] + result3[1] - data_array[1])
    #offset = data_array[1] - (result[0] * data_array[0] + result[1])
    
    print(np.round(result3,2), np.round(dfit3, 2), len(data_array[0]), 10**np.nanmean(data_array[1]))
    
    profile_hi_ul = profile_hi_sub[((offset) < -1*dfit)]
    profile_hi_ll = profile_hi_sub[((offset) > 1*dfit)]
    
    profile_r_ul  = profile_r_sub[((offset) < -1*dfit)]
    profile_r_ll  = profile_r_sub[((offset) > 1*dfit)]
    
    xlbl = r'$\log(r/R_{\rm{iso,HI}})$'
    ylbl = r'$\log(\Sigma_{\rm{HI}}/[\rm{M}_{\odot}\,\rm{pc}^{-2}])$'
    
    radii_hi = np.log10(radii_hi)
    
    #for i in range(len(profile_hi_ul)):
      #hisd_profile_plot(fig2, 2, 1, 1, [radii_hi, np.log10(profile_hi_ul[i])], 
                        #[xlbl, ylbl], 'magenta', 0.75, 4)
    
    #for i in range(len(profile_hi_ll)):
      #hisd_profile_plot(fig2, 2, 1, 1, [radii_hi, np.log10(profile_hi_ll[i])], 
                        #[xlbl, ylbl], 'black', 0.75, 3)
    
    #hisd_profile_plot(fig2, 2, 1, 1, [radii_hi, np.log10(np.nanmean(profiles_hi, axis=0))], 
                      #[xlbl, ylbl], 'blue', 4, 4)
    
    for i in range(len(profile_hi_ul)):
      if i == 0:
        legend = 'Below Relation'
      else:
        legend = False
      hisd_profile_plot(fig2, 2, 1, 1, [radii_hi, np.log10(profile_hi_ul[i])], 
                        [xlbl, ylbl], 'darkblue', 0.75, 4, legend)
    
    for i in range(len(profile_hi_ll)):
      if i == 0:
        legend = 'Above Relation'
      else:
        legend = False
      hisd_profile_plot(fig2, 2, 1, 1, [radii_hi, np.log10(profile_hi_ll[i])], 
                        [xlbl, ylbl], 'peru', 0.75, 3, legend)
    
    hisd_profile_plot(fig2, 2, 1, 1, [radii_hi, np.log10(np.nanmean(profiles_hi, axis=0))], 
                      [xlbl, ylbl], 'black', 4, 4, 'Mean Profile')
    
    
    xlbl = r'$\log(r/R_{\rm{iso,r}})$'
    ylbl = r'$\Sigma_r$ [mag/arcsec$^{2}$]'
    
    radii_r = np.log10(radii_r)
    
    #for i in range(len(profile_r_ul)):
      #hisd_profile_plot(fig2, 2, 1, 2, [radii_r, profile_r_ul[i]], 
                        #[xlbl, ylbl], 'magenta', 0.75, 4)
    
    #for i in range(len(profile_r_ll)):
      #hisd_profile_plot(fig2, 2, 1, 2, [radii_r, profile_r_ll[i]], 
                        #[xlbl, ylbl], 'black', 0.75, 3)
    
    #hisd_profile_plot(fig2, 2, 1, 2, [radii_r, np.nanmean(profiles_r, axis=0)], 
                      #[xlbl, ylbl], 'blue', 4, 4)
    
    for i in range(len(profile_r_ul)):
      if i == 0:
        legend = 'Below Relation'
      else:
        legend = False
      hisd_profile_plot(fig2, 2, 1, 2, [radii_r, profile_r_ul[i]], 
                        [xlbl, ylbl], 'darkblue', 0.75, 4, legend)
    
    for i in range(len(profile_r_ll)):
      if i == 0:
        legend = 'Above Relation'
      else:
        legend = False
      hisd_profile_plot(fig2, 2, 1, 2, [radii_r, profile_r_ll[i]], 
                        [xlbl, ylbl], 'peru', 0.75, 3, legend)
    
    hisd_profile_plot(fig2, 2, 1, 2, [radii_r, np.nanmean(profiles_r, axis=0)], 
                      [xlbl, ylbl], 'black', 4, 4, 'Mean Profile')
    
    #plt.show()
    plot_name = phase1_dir + 'PLOTS/PROFILES/SIZE_RATIO/' + output_fnames[0]
    plt.savefig(plot_name, bbox_inches = 'tight', dpi = 100)
    plt.close()
    
    fig4 = plt.figure(14, figsize=(6, 4))#, facecolor = '#007D7D')
    
    xlbl    = data_to_plot[4]
    ylbl    = data_to_plot[5]
    
    colour  = ['grey', 'darkblue', 'peru', 'green']
    poplbl  = [r'$|$Offset$|$ $<1.2\sigma$', r'Offset $<-1.2\sigma$', r'Offset $>1.2\sigma$']
    
    row, column = 1, 1
    
    print(len(data_array[0]))
    
    print('m, c, rval, pval, stderr')
    
    #subsample_b   = [(offset < 1.2*dfit) & (offset > -1.2*dfit), (offset < -1.2*dfit), (offset > 1.2*dfit)]
    
    subsample_b   = [(offset < 1.*dfit) & (offset > -1.*dfit), (offset < -1.*dfit), (offset > 1.*dfit)]
    
    gal_subsample = galaxies[subsample]
    
    for i in range(3):
      if i == 0:
        legend = True
      else:
        legend = False
      #scatter_outlier_plot(fig4, row, column, 1, 
                           #data_array[0][subsample_b[i] & (data_to_plot[2][subsample] > 45)], 
                           #data_array[1][subsample_b[i] & (data_to_plot[2][subsample] > 45)],
                    #r'$>3$ beams', xlbl, ylbl, [colour[i],  'o', colour[i], 25], legend)
      
      #scatter_outlier_plot(fig4, row, column, 1, 
                           #data_array[0][subsample_b[i] & (data_to_plot[2][subsample] < 45)], 
                           #data_array[1][subsample_b[i] & (data_to_plot[2][subsample] < 45)],
                    #r'$<3$ beams', xlbl, ylbl, [colour[i],  'o', 'none', 25], legend)
      
      environment = np.array(data_to_plot[6])[2]
      scatter_outlier_plot(fig4, row, column, 1, 
                           data_array[0][subsample_b[i] & (data_to_plot[2][subsample] > 45) & (environment[subsample])], 
                           data_array[1][subsample_b[i] & (data_to_plot[2][subsample] > 45) & (environment[subsample])],
                    r'$>3$ beams', xlbl, ylbl, ['black',  'o', 'black', 25], legend)
      
      scatter_outlier_plot(fig4, row, column, 1, 
                           data_array[0][subsample_b[i] & (data_to_plot[2][subsample] < 45) & (environment[subsample])], 
                           data_array[1][subsample_b[i] & (data_to_plot[2][subsample] < 45) & (environment[subsample])],
                    r'$<3$ beams', xlbl, ylbl, ['black',  'o', 'none', 25], legend)
      
      do_environments = False
      if do_environments:
        legend = False
        environment = np.array(data_to_plot[6])[1]
        scatter_outlier_plot(fig4, row, column, 1, 
                            data_array[0][subsample_b[i] & (data_to_plot[2][subsample] > 45) & (environment[subsample])], 
                            data_array[1][subsample_b[i] & (data_to_plot[2][subsample] > 45) & (environment[subsample])],
                      r'$>3$ beams', xlbl, ylbl, [colour[i],  'd', colour[i], 15], legend)
        
        scatter_outlier_plot(fig4, row, column, 1, 
                            data_array[0][subsample_b[i] & (data_to_plot[2][subsample] < 45) & (environment[subsample])], 
                            data_array[1][subsample_b[i] & (data_to_plot[2][subsample] < 45) & (environment[subsample])],
                      r'$<3$ beams', xlbl, ylbl, [colour[i],  'd', 'none', 15], legend)
        
        environment = np.array(data_to_plot[6])[3]
        scatter_outlier_plot(fig4, row, column, 1, 
                            data_array[0][subsample_b[i] & (data_to_plot[2][subsample] > 45) & (environment[subsample])], 
                            data_array[1][subsample_b[i] & (data_to_plot[2][subsample] > 45) & (environment[subsample])],
                      r'$>3$ beams', xlbl, ylbl, [colour[i],  'd', colour[i], 15], legend)
        
        scatter_outlier_plot(fig4, row, column, 1, 
                            data_array[0][subsample_b[i] & (data_to_plot[2][subsample] < 45) & (environment[subsample])], 
                            data_array[1][subsample_b[i] & (data_to_plot[2][subsample] < 45) & (environment[subsample])],
                      r'$<3$ beams', xlbl, ylbl, [colour[i],  'd', 'none', 15], legend)
        
        environment = np.array(data_to_plot[6])[4]
        scatter_outlier_plot(fig4, row, column, 1, 
                            data_array[0][subsample_b[i] & (data_to_plot[2][subsample] > 45) & (environment[subsample])], 
                            data_array[1][subsample_b[i] & (data_to_plot[2][subsample] > 45) & (environment[subsample])],
                      r'$>3$ beams', xlbl, ylbl, [colour[i],  'd', colour[i], 15], legend)
        
        scatter_outlier_plot(fig4, row, column, 1, 
                            data_array[0][subsample_b[i] & (data_to_plot[2][subsample] < 45) & (environment[subsample])], 
                            data_array[1][subsample_b[i] & (data_to_plot[2][subsample] < 45) & (environment[subsample])],
                      r'$<3$ beams', xlbl, ylbl, [colour[i],  'd', 'none', 15], legend)
        
        environment = np.array(data_to_plot[6])[5]
        scatter_outlier_plot(fig4, row, column, 1, 
                            data_array[0][subsample_b[i] & (data_to_plot[2][subsample] > 45) & (environment[subsample])], 
                            data_array[1][subsample_b[i] & (data_to_plot[2][subsample] > 45) & (environment[subsample])],
                      r'$>3$ beams', xlbl, ylbl, [colour[i],  'd', colour[i], 15], legend)
        
        scatter_outlier_plot(fig4, row, column, 1, 
                            data_array[0][subsample_b[i] & (data_to_plot[2][subsample] < 45) & (environment[subsample])], 
                            data_array[1][subsample_b[i] & (data_to_plot[2][subsample] < 45) & (environment[subsample])],
                      r'$<3$ beams', xlbl, ylbl, [colour[i],  'd', 'none', 15], legend)
        
        environment = np.array(data_to_plot[6])[0]
        scatter_outlier_plot(fig4, row, column, 1, 
                            data_array[0][subsample_b[i] & (data_to_plot[2][subsample] > 45) & (environment[subsample])], 
                            data_array[1][subsample_b[i] & (data_to_plot[2][subsample] > 45) & (environment[subsample])],
                      r'$>3$ beams', xlbl, ylbl, [colour[i],  'd', colour[i], 15], legend)
        
        scatter_outlier_plot(fig4, row, column, 1, 
                            data_array[0][subsample_b[i] & (data_to_plot[2][subsample] < 45) & (environment[subsample])], 
                            data_array[1][subsample_b[i] & (data_to_plot[2][subsample] < 45) & (environment[subsample])],
                      r'$<3$ beams', xlbl, ylbl, [colour[i],  'd', 'none', 15], legend)
      
      #if len(data_to_plot) == 7:
        #if i == 0:
          #legend = True
        #else:
          #legend = False
        #environment = np.array(data_to_plot[6])[4]
        #scatter_outlier_plot(fig4, row, column, 1, 
                           #data_array[0][subsample_b[i] & (environment[subsample])], 
                           #data_array[1][subsample_b[i] & (environment[subsample])],
                    #'NGC4636', xlbl, ylbl, ['darkred',  'o', 'darkred', 10], legend)
        #environment = np.array(data_to_plot[6])[5]
        #scatter_outlier_plot(fig4, row, column, 1, 
                           #data_array[0][subsample_b[i] & (environment[subsample])], 
                           #data_array[1][subsample_b[i] & (environment[subsample])],
                    #'Field (N)', xlbl, ylbl, ['red',  'o', 'red', 10], legend)
        #environment = np.array(data_to_plot[6])[0]
        #scatter_outlier_plot(fig4, row, column, 1, 
                           #data_array[0][subsample_b[i] & (environment[subsample])], 
                           #data_array[1][subsample_b[i] & (environment[subsample])],
                    #'Cluster', xlbl, ylbl, ['blue',  'o', 'blue', 20], legend)
        #environment = np.array(data_to_plot[6])[1]
        #scatter_outlier_plot(fig4, row, column, 1, 
                           #data_array[0][subsample_b[i] & (environment[subsample])], 
                           #data_array[1][subsample_b[i] & (environment[subsample])],
                    #'Infall', xlbl, ylbl, ['violet',  'o', 'violet', 20], legend)
        #environment = np.array(data_to_plot[6])[2]
        #scatter_outlier_plot(fig4, row, column, 1, 
                           #data_array[0][subsample_b[i] & (environment[subsample])], 
                           #data_array[1][subsample_b[i] & (environment[subsample])],
                    #'Field (H)', xlbl, ylbl, ['peru',  'o', 'peru', 40], legend)
        ##environment = np.array(data_to_plot[6])[3]
        ##scatter_outlier_plot(fig4, row, column, 1, 
                           ##data_array[0][subsample_b[i] & (environment[subsample])], 
                           ##data_array[1][subsample_b[i] & (environment[subsample])],
                    ##'Background', xlbl, ylbl, ['black',  'o', 'black', 25], legend)
        
        
      
      result = linregress(data_array)
      #print(np.round(result,2))
      
      ax1 = fig4.add_subplot(row, column, 1, facecolor = 'w')
      
      if i == 0:
        #ax1.text(0.8, 0.05, '%.2f' % result[2], transform=ax1.transAxes, fontsize=12)
        ax1.plot(xfit, yfit, color=colour[0], linewidth=1, linestyle = '--', zorder=1)
        ax1.fill_between(xfit, yfit-dfit, yfit+dfit, alpha=0.25, edgecolor='none', zorder=-1, facecolor=colour[0])
        #ax1.plot(xfit, yfit3, color=colour[0], linewidth=1, linestyle = ':', zorder=1)
      
      ax1.set_ylim(-0.3,0.9)
      
    plot_name = phase1_dir + 'PLOTS/PROPERTIES/SIZE_RATIO/' + output_fnames[1]
    plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()


def scatter_radial_profiles_optical(input_dirs, input_fnames, output_fnames, data_to_plot):
    dir_hydra        = input_dirs[0]
    dir_ps_hydra     = dir_hydra + 'MULTIWAVELENGTH/PANSTARRS/'
    
    dir_ngc4636      = input_dirs[1]
    dir_ps_ngc4636   = dir_ngc4636 + 'MULTIWAVELENGTH/PANSTARRS/'
    
    fits_rp          = dir_ps_hydra + 'PROFILES_BKGDSUB/' + input_fnames[0]
    hdu_rp           = fits.open(fits_rp)
    data_rp_hydra    = hdu_rp[1].data
    
    fits_rp          = dir_ps_ngc4636 + 'PROFILES_BKGDSUB/' + input_fnames[0]
    hdu_rp           = fits.open(fits_rp)
    data_rp_ngc4636  = hdu_rp[1].data
    
    join_rp          = join(data_rp_hydra, data_rp_ngc4636, keys='RADIUS')
    
    radii            = join_rp[data_rp_hydra.columns.names[0]]
    
    profiles_r = []
    for i in range(len(galaxies)):
      profiles_r.append(join_rp[galaxies[i]])
    
    profiles_r = np.array(profiles_r)
    
    fig2        = plt.figure(2, figsize=(6, 4))
    
    xpar        = data_to_plot[0]
    ypar        = data_to_plot[1]
    #ypar        = radius_r_iso25_kpc
    #ypar        = radius_total_kpc  # radius_opt_kpc
    
    if data_to_plot[3] == 30:
      subsample   = (data_to_plot[2] > data_to_plot[3]) & np.isfinite(xpar) & np.isfinite(ypar)
    else:
      subsample   = np.isfinite(xpar) & np.isfinite(ypar)
    data_array  = [xpar[subsample], np.log10(ypar[subsample])]
    
    profile_sub = profiles_r[subsample]
    
    print(len(profiles_r), len(profile_sub))
    
    result = linregress(data_array)
    print(np.round(result,2))
    
    if np.nanmin(data_array[0] < 0):
      xfit   = np.arange(-23, -13, 0.25)
    else:
      xfit   = np.arange(6.5, 11.5, 0.25)
    
    
    yfit   = result[0] * xfit + result[1]
    dfit   = np.std(result[0] * data_array[0] + result[1] - data_array[1])
    offset = result[0] * data_array[0] + result[1] - data_array[1]
    
    #print(dfit)
    print(np.nanmean(np.abs(result[0] * data_array[0] + result[1] - data_array[1])), dfit)
    
    profile_ul = profile_sub[((offset) > 1.2*dfit)]
    profile_ll = profile_sub[((offset) < -1.2*dfit)]
    
    xlbl = r'$\log(r/R_{\rm{iso,r}})$'
    #xlbl = r'$r/R_{\rm{iso,r}}$'
    ylbl = r'$\Sigma_r$ [mag/arcsec$^{2}$]'
    
    #for i in range(len(profiles_r)):
      #hisd_profile_plot(fig2, 1, 1, 1, [radii, profiles_r[i]], 
                        #[r'$r/R_{\rm{opt}}$', r'$\Sigma_r$ [mag/arcsec$^{2}$]'], 'peru', 0.25, 1)
    
    radii = np.log10(radii)
    
    for i in range(len(profile_ul)):
      if i == 0:
        legend = 'Below Relation'
      else:
        legend = False
      hisd_profile_plot(fig2, 1, 1, 1, [radii, profile_ul[i]], 
                        [xlbl, ylbl], 'darkblue', 0.75, 4, legend)
    
    for i in range(len(profile_ll)):
      if i == 0:
        legend = 'Above Relation'
      else:
        legend = False
      hisd_profile_plot(fig2, 1, 1, 1, [radii, profile_ll[i]], 
                        [xlbl, ylbl], 'peru', 0.75, 3, legend)
    
    hisd_profile_plot(fig2, 1, 1, 1, [radii, np.nanmean(profiles_r, axis=0)], 
                      [xlbl, ylbl], 'black', 4, 4, 'Mean Profile')
    
    #sersic_profile = np.log(21.5) - radii**1
    sersic_model   = Sersic1D(amplitude=10**(-23.5/2.5), r_eff=0.6, n=1)
    sersic_radius  = np.arange(0, 3, .01)
    
    sersic_profile = sersic_model(sersic_radius)
    
    hisd_profile_plot(fig2, 1, 1, 1, [np.log10(sersic_radius), -2.5*np.log10(sersic_profile)], 
                      [xlbl, ylbl], 'magenta', 4, 4, r'Sersic $N=1$')
    
    sersic_model   = Sersic1D(amplitude=10**(-23.5/2.5), r_eff=0.6, n=2)
    sersic_radius  = np.arange(0, 3, .01)
    
    sersic_profile = sersic_model(sersic_radius)
    
    hisd_profile_plot(fig2, 1, 1, 1, [np.log10(sersic_radius), -2.5*np.log10(sersic_profile)], 
                      [xlbl, ylbl], 'magenta', 3, 4, r'Sersic $N=2$')
    
    sersic_model   = Sersic1D(amplitude=10**(-23.5/2.5), r_eff=0.6, n=4)
    sersic_radius  = np.arange(0, 3, .01)
    
    sersic_profile = sersic_model(sersic_radius)
    
    hisd_profile_plot(fig2, 1, 1, 1, [np.log10(sersic_radius), -2.5*np.log10(sersic_profile)], 
                      [xlbl, ylbl], 'magenta', 2, 4, r'Sersic $N=4$')
    
    #plt.show()
    plot_name = phase1_dir + 'PLOTS/PAPER3/' + output_fnames[0]
    plt.savefig(plot_name, bbox_inches = 'tight', dpi = 100)
    plt.close()
    
    fig4 = plt.figure(14, figsize=(5, 5))#, facecolor = '#007D7D')
    
    xlbl         = data_to_plot[4]
    ylbl         = data_to_plot[5]
    
    colour  = ['grey', 'darkblue', 'peru', 'green']
    poplbl  = [r'$|$Offset$|$ $<1.2\sigma$', r'Offset $<-1.2\sigma$', r'Offset $>1.2\sigma$']
    
    
    row, column = 1, 1
    
    print('m, c, rval, pval, stderr')
    
    #data_array = [xpar[subsample], np.log10(ypar[subsample])]
    
    #xfit   = np.arange(-23.5, -15.5, 0.25)
    #yfit = result[0] * xfit + result[1]
    #dfit = np.std(result[0] * data_array[0] + result[1] - data_array[1])
    #offset = result[0] * data_array[0] + result[1] - data_array[1]
    
    #print(np.nanmean(np.abs(result[0] * data_array[0] + result[1] - data_array[1])), dfit)
    
    subsample_b = [(offset < 1.2*dfit) & (offset > -1.2*dfit),
                  (offset > 1.2*dfit),
                  (offset < -1.2*dfit)]
    
    for i in range(3):
      #scat_mean_plot(fig4, row, column, 1, data_array[0][subsample_b[i]], data_array[1][subsample_b[i]], False, False, 
                    #poplbl[i], xlbl, ylbl, colour[i],  'o', False)
      if i == 0:
        legend = True
      else:
        legend = False
        
      scatter_outlier_plot(fig4, row, column, 1, 
                           data_array[0][subsample_b[i] & (data_to_plot[2][subsample] > 45)], 
                           data_array[1][subsample_b[i] & (data_to_plot[2][subsample] > 45)],
                    r'$>3$ beams', xlbl, ylbl, [colour[i],  'o', colour[i], 25], legend)
      
      scatter_outlier_plot(fig4, row, column, 1, 
                           data_array[0][subsample_b[i] & (data_to_plot[2][subsample] < 45)], 
                           data_array[1][subsample_b[i] & (data_to_plot[2][subsample] < 45)],
                    r'$<3$ beams', xlbl, ylbl, [colour[i],  'o', 'none', 25], legend)
      
      
      result = linregress(data_array)
      print(np.round(result,2))
      
      ax1 = fig4.add_subplot(row, column, 1, facecolor = 'w')
      
      if i == 0:
        #ax1.text(0.8, 0.05, '%.2f' % result[2], transform=ax1.transAxes, fontsize=12)
        ax1.plot(xfit, yfit, color=colour[0], linewidth=1, linestyle = '--', zorder=1)
        ax1.fill_between(xfit, yfit-dfit, yfit+dfit, alpha=0.25, edgecolor='none', zorder=-1, facecolor=colour[0])
      
      
    plot_name = phase1_dir + 'PLOTS/PAPER3/' + output_fnames[1]
    plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()


def hist_percentile_plot(fig_num, subfig, data, colour, lsty, percentile, xlbl):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(subfig[0], subfig[1], subfig[2], facecolor = 'w')
    ax1.set_xlabel(xlbl)
    ax1.set_ylabel(r'Number')
    #ax1.set_xscale('log')
    plt.hist(data, bins=10, color=colour, histtype='step', linestyle=lsty, linewidth=1.5)
    data_median = np.nanmedian(data)
    data_p20    = np.abs(data_median - np.percentile(data[~np.isnan(data)], percentile))
    data_p80    = np.abs(data_median - np.percentile(data[~np.isnan(data)], 100 - percentile))
    #print(data_median, data_p20, data_p80)
    ax1.axvline(data_median, linewidth=0.75, linestyle = '--', color = 'grey')
    ax1.axvline(data_median - data_p20, linewidth=0.75, linestyle = ':', color = 'grey')
    ax1.axvline(data_median + data_p80, linewidth=0.75, linestyle = ':', color = 'grey')
    #ax1.legend(loc='upper right', fontsize = 8.5)
    plt.subplots_adjust(wspace=0.15, hspace=0.15)
    return(data_median, data_p20, data_p80)
    


# ================================ #
# ====== Scatter Mean Plot ======= #
# ================================ #
def size_mass_relation_plot(fig_num, sub1, sub2, sub3, data_array, txtstr, xlbl, ylbl, col_array, marker):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    ax1.set_xlabel(xlbl)
    ax1.set_ylabel(ylbl)
    col  = col_array[0]
    ecol = col_array[1]
    fcol = col_array[2]
    ax1.scatter(data_array[0], data_array[1], color=col, marker=marker, 
                edgecolor=ecol, facecolor=fcol, s=15, zorder=1, label = txtstr) 
    
    result = np.round(linregress(data_array),3)
    xfit  = np.arange(6.5, 11.5, 0.25)
    yfit  = (result[0] * xfit + result[1])
    dfit  = (np.std(result[0] * data_array[0] + result[1] - data_array[1]))
    offset = data_array[1] - (result[0] * data_array[0] + result[1])
    print(len(offset[offset<0]), len(offset[offset>0]))
    #ax1.text(0.8, 0.05, '%.2f' % result[2], transform=ax1.transAxes, fontsize=12, color=col)
    
    ax1.plot(xfit, yfit, color=col, linewidth=1, linestyle = '--', zorder=1)
    ax1.fill_between(xfit, yfit-dfit, yfit+dfit, alpha=0.25, 
                     edgecolor='none', zorder=-1, facecolor=col)
    
    print(len(data_array[0]), np.round(result,3), np.round(dfit, 2))
    
    #m, b = np.polyfit(data_array[0], data_array[1], 1)
    #xfit  = np.arange(6.5, 11.5, 0.25)
    #yfit  = (m * xfit + b)
    #dfit  = (np.std(m * data_array[0] + b - data_array[1]))
    #offset = data_array[1] - (m * data_array[0] + b)
    #print(m, b, len(offset[offset<0]), len(offset[offset>0]))
    
    ax1.legend(fontsize = 12)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    return(offset)



# ================================= #
# ========== Axes Labels ========== #
# ================================= #
lbl_czhi   = r'c$z_{\mathrm{HI}}$ [km\,s$^{-1}$]'
lbl_czopt  = r'c$z_{\mathrm{opt}}$ [km\,s$^{-1}$]'
lbl_sint   = r'$\log(S_{\mathrm{int}}/\mathrm{Jy})$'
lbl_mstar  = r'$\log(M_*/\mathrm{M}_{\odot})$'
lbl_mhi    = r'$\log(M_{\mathrm{HI}}/\mathrm{M}_{\odot})$'
lbl_mustar = r'$\log(\mu_*/[\rm{M}_{\odot}\,\rm{kpc}^{-2}])$'
lbl_muhi   = r'$\Sigma_{\rm{HI}}/[\rm{M}_{\odot}\,\rm{pc}^{-2}]$'
lbl_hidef  = r'$\mathrm{DEF}_{\mathrm{HI}}$'
lbl_dhi    = r'$\log(d_{\mathrm{HI}}/\mathrm{kpc})$'
lbl_d25    = r'$d_{\mathrm{\mathrm{opt}}}/\mathrm{kpc}$'
lbl_hifrac = r'$\log(M_{\mathrm{HI}}/M_*)$'
lbl_dlum   = r'$D_{\mathrm{L}}$ [Mpc]'
lbl_aflux  = r'$A_{\mathrm{flux}}$' 
lbl_aspec  = r'$A_{\mathrm{spec}}$'
lbl_dvsys  = r'$\Delta V_{\mathrm{sys}}$ [km\,s$^{-1}$]'
lbl_w20    = r'$w_{20}$ [km\,s$^{-1}$]'
lbl_sep    = r'$R_{\mathrm{proj}}/R_{\mathrm{vir}}$'
lbl_sfr    = r'$\log$(SFR/[M$_{\odot}$\,yr$^{-1}$])'
lbl_nuvr   = r'$\mathrm{NUV}-r$ [Mag]'
lbl_ssfr   = r'$\log$(sSFR/yr$^{-1}$)'
lbl_sratio = r'$d_{\mathrm{HI}}/d_{\mathrm{opt}}$'
lbl_cindex = r'$R_{90}/R_{50}$'
lbl_tdep   = r'$\tau_{\rm{dep}}$/[Gyr]'
lbl_sfe    = r'$\log$(SFE/[yr$^{-1}$])'

# ================================= #
# ========= File Strings ========== #
# ================================= #
#basedir       = '/Users/tflowers/WALLABY/Hydra_DR2/'
#panstarrs_dir = basedir + 'MULTIWAVELENGTH/PANSTARRS/'
#sofia_dir     = basedir + 'SOFIA_DR2/'
plot_dir      = '/Users/tflowers/WALLABY/PHASE2/PLOTS/'


#f_meas = np.array([3, 3.5, 4, 4.5, 5, 5.5, 6])

#f_corr = f_meas - 0.0285 * f_meas**3 + 0.439 * f_meas**2 - 2.294 * f_meas + 4.097

#print(f_corr)
#print(10**f_meas / 10**f_corr)

# ================================= #
# =========== Switches ============ #
# ================================= #
do_open_catalogues   = True
do_get_properties    = True


do_wang2016          = False
do_xgass             = True

do_main_sequence     = False
do_hi_fraction_mstar = False
do_hi_fraction_ssfr  = False
do_hi_fraction_nuvr  = False
do_hi_fraction_dsfms = True

do_hifrac_mstar      = False

do_sample_select     = False

do_scatter_plots     = False

do_hism_scatter      = False

do_sratio_profiles   = False
do_radial_profiles   = False

do_paper_plots       = False

do_radial_hi_profiles = False
do_true_radial_profiles = False

# ================================= #
# ============= Hydra ============= #
# ================================= #
if do_open_catalogues:
  do_save_table_calc_par = False
  
  #tr_i                     = 4
  survey_phase_list        = ['PHASE1', 'PHASE1', 'PHASE1', 'PHASE2', 'PHASE2', 'PHASE2']
  team_release_list        = ['Hydra_DR1', 'Hydra_DR2', 'NGC4636_DR1', 'NGC4808_DR1', 'NGC5044_DR1', 'NGC5044_DR2']
  
  basedir                  = '/Users/tflowers/WALLABY/PHASE2/NGC5044_DR1/'
  fits_dr1                 = basedir + 'SOFIA/' + 'NGC5044_DR1_catalogue.fits'
  basedir                  = '/Users/tflowers/WALLABY/PHASE2/NGC5044_DR2/'
  fits_dr2                 = basedir + 'SOFIA/' + 'NGC5044_DR2_catalogue.fits'
  
  hdu_dr1                  = fits.open(fits_dr1)
  data_dr1                 = hdu_dr1[1].data
  hdu_dr2                  = fits.open(fits_dr2)
  data_dr2                 = hdu_dr2[1].data
  
  join_ngc5044             = join(data_dr1, data_dr2, keys='name', join_type='inner')
  
  galaxies_both5044        = join_ngc5044['name']
  
  print(len(join_ngc5044))
  
  for tr_i in range(1,6):
    survey_phase             = survey_phase_list[tr_i]
    team_release             = team_release_list[tr_i]
    
    # ================================= #
    # ========= File Strings ========== #
    # ================================= #
    basedir        = '/Users/tflowers/WALLABY/%s/%s/' % (survey_phase, team_release)
    sofia_dir      = basedir + 'SOFIA/'
    parameter_dir  = basedir + 'PARAMETERS/'
    
    fits_sofia     = sofia_dir + '%s_catalogue.fits' % team_release
    fits_flags     = sofia_dir + '%s_catalogue_flags.fits' % team_release
    fits_params    = parameter_dir + '%s_derived_galaxy_properties.fits' % team_release
    fits_sfrs      = parameter_dir + '%s_derived_galaxy_sfrs.fits' % team_release
    fits_panstarrs = parameter_dir + '%s_panstarrs_photometry.fits' % team_release
    #fits_hiopt     = parameter_dir + '%s_hi_optical_disc.fits' % team_release
    
    hdu_sofia      = fits.open(fits_sofia)
    data_sofia     = hdu_sofia[1].data
    
    hdu_flags      = fits.open(fits_flags)
    data_flags     = hdu_flags[1].data
    
    hdu_params     = fits.open(fits_params)
    data_params    = hdu_params[1].data
    
    hdu_sfrs       = fits.open(fits_sfrs)
    data_sfrs      = hdu_sfrs[1].data
    
    hdu_panstarrs  = fits.open(fits_panstarrs)
    data_panstarrs = hdu_panstarrs[1].data
    
    join1          = join(data_sofia, data_flags, join_type='left')
    join2          = join(join1, data_params, join_type='left')
    join3          = join(join2, data_sfrs, join_type='left')
    data_join      = join(join3, data_panstarrs, join_type='left')
    
    if tr_i == 1:
      data_all    = data_join
    else:
      data_all    = vstack([data_all, data_join], join_type='outer')
  
  #print(data_all.columns)
  
  data_mask      = data_all[(data_all['flag_src_class'] == 1) & (data_all['flag_opt_fit'] == 0)]
  
  print(len(data_mask))
  
  galaxy_names   = data_mask['name']
  release        = data_mask['team_release']
  counter        = 0
  
  #print(galaxy_names[0], galaxies_both5044[0], release[0])
  
  for i in range(len(galaxy_names)):
    for j in range(len(galaxies_both5044)):
      if galaxy_names[i] == galaxies_both5044[j] and release[i] == 'NGC 5044 TR1':
        #print(galaxy_names[i], galaxies_both5044[j], release[i])
        data_mask.remove_row(counter)
        counter -= 1
    counter += 1
  
  print(len(data_mask))
  
  
if do_get_properties:
  do_save_table_calc_par = False
  gal_name = []
  for i in range(len(data_mask['name'])):
    split_name = data_mask['name'][i][8:]
    gal_name.append(split_name)
  
  basedir           = '/Users/tflowers/WALLABY/%s/%s/' % (survey_phase_list[1], team_release_list[1])
  dataprod_dir      = basedir + 'SOFIA/%s_source_products/' % team_release_list[1]
  galaxy_dir        = 'WALLABY_' + gal_name[0] + '/'
  fits_file         = dataprod_dir + galaxy_dir + 'WALLABY_' + gal_name[0] + '_cube.fits.gz'
  f1                = pyfits.open(fits_file)
  data, hdr         = f1[0].data, f1[0].header
  if hdr['CTYPE3'] == 'FREQ':
    chan_width = np.abs((HI_REST / (hdr['CRVAL3']/1e6)) - (HI_REST / ((hdr['CRVAL3']-hdr['CDELT3'])/1e6))) * C_LIGHT
    chan_width_hz = hdr['CDELT3']
  else:
    chan_width = np.abs(hdr['CDELT3']/1000.)
  beam_maj, beam_min, pix_scale  = hdr['BMAJ'], hdr['BMIN'], np.abs(hdr['CDELT1'])
  f1.close()
  
  BEAM                 = beam_factor(beam_maj*3600., beam_min*3600., pix_scale*3600.)
  
  redshift_hi          = (HI_REST / (data_mask['freq'] / 1e6)) - 1.
  galaxies             = np.array(gal_name)
  sofia_ra             = data_mask['ra']
  sofia_dec            = data_mask['dec']
  sofia_vsys           = redshift_hi * C_LIGHT
  sofia_sint           = data_mask['f_sum'] * chan_width / chan_width_hz
  sofia_rms            = data_mask['rms'] * 1000.
  sofia_snr            = data_mask['f_sum'] / data_mask['err_f_sum']
  sofia_kinpa          = data_mask['kin_pa']
  sofia_w20            = data_mask['w20'] / chan_width_hz * chan_width
  sofia_w50            = data_mask['w50'] / chan_width_hz * chan_width
  sofia_ell_maj        = data_mask['ell_maj'] * pix_scale * 3600.
  sofia_ell_min        = data_mask['ell_min'] * pix_scale * 3600.
  sofia_ellpa          = data_mask['ell_pa']
  #sofia_id             = data_mask['wal_id']
  
  sint_meas            = np.log10(np.array(data_mask['f_sum']))
  
  redshift             = np.array(data_mask['REDSHIFT'])
  distance             = np.array(data_mask['DISTANCE'])
  mstar                = np.array(data_mask['lgMSTAR_SDSS_25'])
  
  mhi                  = np.array(data_mask['lgMHI_CORRECT'])
  mhi_optdisc          = np.array(data_mask['lgMHI_OPTICAL_DISC_CORRECT_ISO'])
  hifrac               = np.array(data_mask['HIFRAC_CORRECT'])
  hifrac_optdisc       = np.array(data_mask['HIFRAC_OPTICAL_DISC_CORRECT_ISO'])
  
  radius_r_iso25       = np.array(data_mask['RADIUS_R_ISO25'])
  radius_r_iso26       = np.array(data_mask['RADIUS_R_ISO26'])
  radius_r_50          = np.array(data_mask['RADIUS_R_50'])
  radius_r_90          = np.array(data_mask['RADIUS_R_90'])
  axis_ratio_opt       = np.array(data_mask['SEG_BA'])
  
  radius_hi_iso        = np.array(data_mask['RADIUS_HI_ISO_CORR'])
  radius_hi_eff        = np.array(data_mask['RADIUS_HI_EFF_CORR'])
  radius_error         = np.array(data_mask['RADIUS_ISO_ERR'])
  axis_ratio_hi        = np.array(data_mask['AXIS_RATIO_BA'])
  
  radius_nuv_iso       = np.array(data_mask['RADIUS_NUV_ISO'])
  nuvr                 = np.array(data_mask['NUV-R'])
  
  surfden_hi_iso       = np.array(data_mask['SURFACE_DENSITY_HI_ISO_CORR'])
  surfden_hi_eff       = np.array(data_mask['SURFACE_DENSITY_HI_EFF_CORR'])
  
  hi_r_sratio          = np.array(data_mask['HI_R25_SIZE_RATIO'])
  nuv_r_sratio         = np.array(data_mask['NUV_R25_SIZE_RATIO'])
  hi_nuv_sratio        = np.array(data_mask['HI_NUV_SIZE_RATIO'])
  
  sfr                  = np.array(data_mask['SFR_NUV+MIR'])
  sfr_uplim            = data_mask['SFR_UPLIM']
  ssfr                 = np.array(data_mask['SSFR'])
  no_galex             = data_mask['NO_GALEX_COVERAGE_FLAG']
  
  #sfr_w3aper           = np.array(data_mask['SFR_NUV+MIR_W3_APERTURE'])
  #sfr_w4aper           = np.array(data_mask['SFR_NUV+MIR_W4_APERTURE'])
  
  w3_uplim             = data_mask['W3_UPPERLIM']
  w4_uplim             = data_mask['W4_UPPERLIM']
  
  ap_mag_r             = np.array(data_mask['R_MAG_PS25'])
  
  nuv_mag              = np.array(data_mask['NUV_MAG'])
  
  team_release         = np.array(data_mask['team_release'])
  
  cindex               = radius_r_90 / radius_r_50
  
  hi_incl              = np.arccos(axis_ratio_hi) * 180. / math.pi
  
  #print(sfr[galaxies == 'J103939-280552'])
  
  sfr[np.isnan(sfr)]   = np.array(data_mask['SFR_NUV+W3'][np.isnan(sfr)])
  
  #print(data_mask['SFR NUV+W3'][galaxies == 'J103939-280552'])
  
  ssfr[np.isnan(ssfr)] = sfr[np.isnan(ssfr)] - mstar[np.isnan(ssfr)]
  
  radius_nuv_iso[np.isnan(nuv_mag)] = np.nan
  hi_nuv_sratio[np.isnan(nuv_mag)]  = np.nan
  nuv_r_sratio[np.isnan(nuv_mag)]   = np.nan
  
  #radius_nuv_iso[nuv_mag == 20.0]   = np.nan
  #hi_nuv_sratio[nuv_mag == 20.0]    = np.nan
  #nuv_r_sratio[nuv_mag == 20.0]     = np.nan
  
  hi_nuv_sratio[radius_nuv_iso < 10]  = np.nan
  nuv_r_sratio[radius_nuv_iso < 10]   = np.nan
  radius_nuv_iso[radius_nuv_iso < 10] = np.nan
  
  radius_r_50_kpc     = distance * np.arctan(radius_r_50 / 3600. * math.pi / 180.) * 1000.
  radius_r_90_kpc     = distance * np.arctan(radius_r_90 / 3600. * math.pi / 180.) * 1000.
  
  radius_r_iso25_kpc  = distance * np.arctan(radius_r_iso25 / 3600. * math.pi / 180.) * 1000.
  radius_r_iso26_kpc  = distance * np.arctan(radius_r_iso26 / 3600. * math.pi / 180.) * 1000.
  
  radius_nuv_iso_kpc  = distance * np.arctan(radius_nuv_iso / 3600. * math.pi / 180.) * 1000.
  
  radius_hi_iso_kpc   = distance * np.arctan(radius_hi_iso / 3600. * math.pi / 180.) * 1000.
  radius_hi_eff_kpc   = distance * np.arctan(radius_hi_eff / 3600. * math.pi / 180.) * 1000.
  
  radius_hi_err_kpc   = radius_hi_iso_kpc * radius_error / radius_hi_iso
  
  
  sfms_const        = [0.656, -6.726]
  sfms              = sfms_const[0] * mstar + sfms_const[1]
  delta_sfms        = sfr - sfms
  
  gfms_const        = [-0.53, -0.07]
  gfms              = gfms_const[0] * (mstar - 9.) + gfms_const[1]
  delta_gfms        = hifrac - gfms
  
  abs_mag_r         = mag_mtoM(ap_mag_r, distance)
  #abs_mag_r        = mag_mtoM(ap_mag_r, dist_lum(redshift_hi))
  
  mu_star           = np.log10(10**mstar / (2. * math.pi * radius_r_50_kpc**2))
  
  mu_hi             = 10**mhi / (math.pi * (radius_hi_iso_kpc * 1000.)**2)
  
  
  mh2_total         = 0.75 * ssfr + 6.24 + mstar
  h2onhi_total      = mh2_total - mhi
  
  do_save_table_paper =  False
  if do_save_table_paper:
    subsample  = ((radius_hi_iso > 20) & np.isfinite(mhi) & np.isfinite(mstar) & 
                  np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  np.isfinite(surfden_hi_iso) & np.isfinite(mu_hi) & 
                  np.isfinite(radius_r_iso25_kpc) & np.isfinite(radius_r_50_kpc) & 
                  np.isfinite(radius_hi_eff_kpc) &
                  np.isfinite(radius_hi_iso_kpc) & (mstar > 6.5))
    
    table_str  = phase1_dir + 'PARAMETERS/PAPER3/detection_properties_latex.fits'
    os.system('rm -rf %s' % table_str)
      
    #tdata = [galaxies[subsample], 
             #team_release[subsample],
             #distance[subsample],
             #mhi[subsample], 
             #radius_hi_eff_kpc[subsample], 
             #radius_hi_iso_kpc[subsample],
             #surfden_hi_eff[subsample],
             #surfden_hi_iso[subsample],
             #mstar[subsample], 
             #radius_r_50_kpc[subsample], 
             #radius_r_iso25_kpc[subsample],
             #mu_star[subsample], 
             #nuvr[subsample], 
             #ssfr[subsample], 
             #delta_sfms[subsample], 
             #sfr_uplim[subsample]]
    
    tdata = [galaxies[subsample], 
             team_release[subsample],
             np.round(distance[subsample],2),
             np.round(mhi[subsample],2), 
             np.round(radius_hi_eff_kpc[subsample],2), 
             np.round(radius_hi_iso_kpc[subsample],2),
             np.round(surfden_hi_eff[subsample],2),
             np.round(surfden_hi_iso[subsample],2),
             np.round(mstar[subsample],2), 
             np.round(radius_r_50_kpc[subsample],2), 
             np.round(radius_r_iso25_kpc[subsample],2),
             np.round(mu_star[subsample],2), 
             np.round(nuvr[subsample],2), 
             np.round(ssfr[subsample],2), 
             np.round(delta_sfms[subsample],2), 
             sfr_uplim[subsample]]
    
    tcols = ('NAME', 
             'TEAM_RELEASE',
             'DISTANCE', 
             'lgMHI', 
             'RADIUS_HI_50_KPC',
             'RADIUS_HI_ISO_KPC',
             'MUHI_50',
             'MUHI_ISO',
             'lgMSTAR', 
             'RADIUS_R_50_KPC',
             'RADIUS_R_ISO_KPC',
             'lgMUSTAR', 
             'NUV-R', 
             'SSFR', 
             'DELTA_SFMS',
             'SSFR_UPPER_LIMIT')
    
    t = Table(tdata, names=tcols)
    t.write(table_str, format = 'fits')


subsample  = ((radius_hi_eff > 20) & np.isfinite(mhi) & np.isfinite(mstar) & 
                  np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  np.isfinite(radius_r_iso25_kpc) & np.isfinite(radius_r_50_kpc) & 
                  np.isfinite(radius_hi_iso_kpc) & np.isfinite(radius_hi_eff_kpc) & (mstar > 6.5))


print(len(galaxies), 
      len(galaxies[np.isfinite(radius_hi_iso)]), 
      len(galaxies[np.isfinite(mstar)]),
      len(galaxies[np.isfinite(sfr)]),
      len(galaxies[np.isfinite(mstar) & np.isfinite(sfr)]),
      len(galaxies[(radius_hi_iso > 20) & np.isfinite(mstar) & np.isfinite(sfr)]),
      len(galaxies[(radius_hi_iso > 30) & np.isfinite(mstar) & np.isfinite(sfr)]),
      len(galaxies[subsample & (radius_r_iso25>20)]))


#print(np.nanmean(radius_hi_iso[subsample]/radius_r_iso25[subsample]),
      #np.nanmean(radius_hi_eff[subsample]/radius_r_iso25[subsample]))

#print(np.nanmean(radius_error), 
      #np.nanmedian(radius_error), 
      #np.nanmean(radius_error / radius_hi_iso), 
      #np.nanmedian(radius_error / radius_hi_iso))

#print(len(np.isfinite(radius_error[radius_error<15])) / len(np.isfinite(radius_error)))

# ================================= #
# ============ Wang2016 =========== #
# ================================= #
if do_wang2016:
  wang2016_file = '/Users/tflowers/WALLABY/Hydra_DR1/CATALOGUES/wang2016_table.txt'
  wang2016_dhi, wang2016_mhi, wang2016_d25 = np.genfromtxt(wang2016_file, skip_header=17, usecols=(1,2,7), unpack=True)
  wang2016_dhi = np.log10(wang2016_dhi)
  
  wang2016_survey = np.genfromtxt(wang2016_file, skip_header=17, usecols=(8), dtype=str, unpack=True)
  
  broeils1997_file = '/Users/tflowers/WALLABY/Hydra_DR1/CATALOGUES/Broeils1997_table.txt'
  broeils1997_dhi, broeils1997_mhi, broeils1997_d25 = np.genfromtxt(broeils1997_file, skip_header=1, usecols=(2,5,3), unpack=True)
  broeils1997_mhi = np.log10(broeils1997_mhi * 10.**10)
  broeils1997_dhi = np.log10(broeils1997_dhi)
  
  broeils1997_survey = ['B97' for x in range(len(broeils1997_dhi))]
  
  wang2016_dhi    = np.concatenate((wang2016_dhi, broeils1997_dhi))
  wang2016_mhi    = np.concatenate((wang2016_mhi, broeils1997_mhi))
  wang2016_d25    = np.concatenate((wang2016_d25, broeils1997_d25))
  wang2016_survey = np.concatenate((wang2016_survey, broeils1997_survey))
  
  print(np.nanmean(10**wang2016_dhi / wang2016_d25))

# ================================= #
# ============= xGASS ============= #
# ================================= #
if do_xgass:
  #cat_file      = basedir + dir_hydra + 'CATALOGUES/' + 'xGASS_representative_sample.txt'
  #xgass_mstar, xgass_mhi, xgass_sfr, xgass_nuvr, xgass_flag = np.genfromtxt(cat_file, usecols=(10, 31, 23, 21, 26), unpack=True)
  #xgass_mstar[xgass_flag == 4] = np.nan
  #xgass_mhi[xgass_flag == 4]   = np.nan
  #xgass_sfr[xgass_flag == 4]   = np.nan
  #xgass_nuvr[xgass_flag == 4]  = np.nan
  
  fits_xgass     = '/Users/tflowers/WALLABY/Hydra_DR2/' + 'CATALOGUES/' + 'xGASS_representative_sample.fits'
  
  # ======== xGASS PARAMETERS ======== #
  hdu_xgass      = fits.open(fits_xgass)
  data_xgass     = hdu_xgass[1].data
  
  xgass_mstar    = data_xgass['lgMstar']
  xgass_mhi      = data_xgass['lgMHI']
  xgass_sfr      = data_xgass['SFR_best']
  xgass_nuvr     = data_xgass['NUVr']
  xgass_mustar   = data_xgass['lgmust']
  xgass_cindex   = data_xgass['CINDX']
  xgass_hifrac   = data_xgass['lgGF']
  xgass_src      = data_xgass['HIsrc']
  
  #xgass_mhi_exp = np.log10(10 ** (8.0857 - 0.857 * xgass_mstar) * 10**xgass_mstar)
  #xgass_hidef   = xgass_mhi - xgass_mhi_exp
  #xgass_hifrac  = xgass_mhi - xgass_mstar
  xgass_nuvr[xgass_nuvr<-50] = np.nan
  xgass_sfr  = np.log10(xgass_sfr)
  xgass_ssfr = xgass_sfr - xgass_mstar
  #print(xgass_sfr)
  
  xgass_hifrac_plane     = -0.250 * xgass_nuvr - 0.240 * xgass_mustar + 2.083
  xgass_hifrac_nuvr      = -0.250 * xgass_nuvr
  
  xgass_mstar_table      = np.array([np.array([9.14, 9.44, 9.74, 10.07, 10.34, 10.65, 10.95, 11.20]),
                                     np.array([-0.242, -0.459, -0.748, -0.869, -1.175, -1.231, -1.475, -1.589]),
                                     np.array([0.053, 0.067, 0.069, 0.042, 0.037, 0.036, 0.033, 0.044])])
  xgass_ssfr_table       = np.array([np.array([-11.97, -11.42, -10.79, -10.19, -9.72]),
                                     np.array([-1.633, -1.442, -1.109, -0.539, -0.063]),
                                     np.array([0.022, 0.032, 0.034, 0.033, 0.041])])
  xgass_nuvr_table       = np.array([np.array([1.62, 2.25, 2.98, 3.79, 4.59, 5.44, 6.04]),
                                     np.array([0.174, -0.090, -0.593, -0.987, -1.281, -1.514, -1.672]),
                                     np.array([0.050, 0.028, 0.030, 0.033, 0.040, 0.026, 0.020])])
  
  func = interp1d(xgass_mstar_table[0], xgass_mstar_table[1], fill_value='extrapolate')
  
  hifrac_lim = func(xgass_mstar)
  
  ssfrms     = -0.344 * (xgass_mstar - 9.) - 9.822
  ssfrms_err = 0.088 * (xgass_mstar - 9.) + 0.188
  
  xgass_obs_lims = (xgass_hifrac > hifrac_lim) & (xgass_ssfr > ssfrms - 2*ssfrms_err)
  
  xpar = [xgass_mstar, xgass_mustar, xgass_ssfr, xgass_nuvr]
  ypar = xgass_hifrac
  
  bins = [np.arange(9, 11.25, 0.25),
          np.arange(7.5, 10, 0.25), 
          np.arange(-11, -9, 0.25),
          np.arange(1, 4, 0.25)]
  
  #for i in range(len(xpar)):
    ##if i == 0 :
    #xpar_i = xpar[i][(xgass_src < 4) & np.isfinite(xpar[i]) & (xgass_obs_lims)]
    #ypar_i = ypar[(xgass_src < 4) & np.isfinite(xpar[i]) & (xgass_obs_lims)]
    ##if i > 0:
      ##xpar_i = xpar[i][(xgass_src < 4) & np.isfinite(xpar[i])]
      ##ypar_i = ypar[(xgass_src < 4) & np.isfinite(xpar[i])]
    #print(np.round(stats.pearsonr(xpar_i, ypar_i)[0], 2), stats.pearsonr(xpar_i, ypar_i)[1])
    
    #par_mean, bin_edges, _ = binned_statistic(xpar_i, 
                                              #ypar_i,
                                              #np.median,
                                              #bins[i])    
    #print(np.round(par_mean, 2))
  



# ================================= #
# ======= SFR Main Sequence ======= #
# ================================= #
if do_main_sequence:
  fig1 = plt.figure(1, figsize=(5, 4))
    
  xlbl         = lbl_mstar
  ylbl         = lbl_sfr
  
  sfms_const = [0.656, -6.726]
  
  scatter_outlier_plot(fig1, 1, 1, 1, xgass_mstar, xgass_ssfr, 
                       'xGASS', xlbl, ylbl, ['darkgrey', '.', 'darkgrey', 3], False)
  
  xpar    = mstar
  ypar    = ssfr
  limpar  = sfr_uplim
  colour  = ['darkblue', 'peru', 'mediumvioletred', 'green', 'magenta']
  poplbl  = ['Hydra TR2', 'NGC 4636 TR1', 'NGC 4808 TR1', 'NGC 5044 TR1', 'NGC 5044 TR2']
  
  for j in range(len(poplbl)):
    subsample  = (team_release == poplbl[j]) & (limpar == 0) & (no_galex == 0)
    data_array = [xpar[subsample], ypar[subsample]]
    #sfr_ms_plot(fig1, [1, 1, 1], data_array, sfms_const, 
                #poplbl[j], xlbl, ylbl, colour[j], 'o')
    scatter_outlier_plot(fig1, 1, 1, 1, data_array[0], data_array[1], 
                         poplbl[j], xlbl, ylbl, [colour[j], 'o', colour[j], 10], True)
  
  for j in range(len(poplbl)):
    subsample  = (team_release == poplbl[j]) & (limpar == 1) & (no_galex == 0)
    data_array = [xpar[subsample], ypar[subsample]]
    #sfr_ms_plot(fig1, [1, 1, 1], data_array, sfms_const, 
                #poplbl[j], xlbl, ylbl, colour[j], r'$\downarrow$')
    scatter_outlier_plot(fig1, 1, 1, 1, data_array[0], data_array[1], 
                         poplbl[j], xlbl, ylbl, [colour[j], r'$\downarrow$', colour[j], 10], False)
    
    #subsample  = pop_hi_array[j] & no_galex
    #data_array = [xpar[subsample], ypar[subsample]]
    #sfr_ms_plot(fig1, [1, 1, 1], data_array, sfms_const, 
                #poplbl[j], xlbl, ylbl, colour[j], 'X')
  
  ax1 = fig1.add_subplot(1, 1, 1, facecolor = 'w')
    
  x = np.arange(7, 12, 0.1)
  #y = 0.656 * x - 6.816
  y = -0.344 * (x - 9.) - 9.822
  ax1.plot(x, y, color='black', linewidth=2.5, linestyle = '--', zorder=1)
  
  
  #sym_array = ['.', 'o', 'o', 'o', r'$\downarrow$', r'$\downarrow$', r'$\downarrow$', 'X', 'X', 'X', 's', 's', 's', r'$\downarrow$', r'$\downarrow$', r'$\downarrow$', 'X', 'X', 'X', r'$\downarrow$', 'X']
  #col_array = ['darkgrey', 'darkblue', 'magenta', 'peru',  'darkblue', 'magenta', 'peru',  'darkblue', 'magenta', 'peru', 'skyblue', 'lightpink', 'sandybrown', 'skyblue', 'lightpink', 'sandybrown', 'skyblue', 'lightpink', 'sandybrown', 'black', 'black']
  #pt_array  = ['xGASS', 'Cluster', 'Infall', 'Field', 'Cluster', 'Infall', 'Field', 'Cluster', 'Infall', 'Field', 'Upper Limit', 'No GALEX']
  
  #sfr_ms_plot(fig1, [1, 1, 1], [[0], [0]], sfms_const, pt_array, xlbl, ylbl, col_array, sym_array)
  
  #sfr_ms   = 0.656 * mstar - 6.816
  plot_name = plot_dir + 'SCALING_RELATIONS/pilot_sfms.pdf'
  plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
  plt.clf()
  #plt.show()

# ================================= #
# ======== HI Gas Fraction ======== #
# ================================= #
if do_hi_fraction_mstar:
  do_all_coloured     = False
  do_optdisc_coloured = False
  do_optdisc_sub      = True
  
  if do_all_coloured:
    fig1 = plt.figure(1, figsize=(5, 4))
      
    xlbl         = lbl_mstar
    ylbl         = lbl_hifrac
    
    scatter_outlier_plot(fig1, 1, 1, 1, xgass_mstar, xgass_hifrac, 
                        'xGASS', xlbl, ylbl, ['darkgrey', '.', 'darkgrey', 3], False)
    
    xpar    = mstar
    ypar    = hifrac
    limpar  = sfr_uplim
    colour  = ['darkblue', 'peru', 'mediumvioletred', 'green', 'magenta']
    poplbl  = ['Hydra TR2', 'NGC 4636 TR1', 'NGC 4808 TR1', 'NGC 5044 TR1', 'NGC 5044 TR2']
    
    for j in range(len(poplbl)):
      subsample  = (team_release == poplbl[j])
      data_array = [xpar[subsample], ypar[subsample]]
      scatter_outlier_plot(fig1, 1, 1, 1, data_array[0], data_array[1], 
                          poplbl[j], xlbl, ylbl, [colour[j], 'o', colour[j], 5], True)
    
    ax1 = fig1.add_subplot(1, 1, 1, facecolor = 'w')
    x, y, z = xgass_mstar_table[0], xgass_mstar_table[1], xgass_mstar_table[2]
    scatter_error_plot(fig1, 1, 1, 1, x, y, [z, z],
                      'xGASS Medians', xlbl, ylbl, ['black', 'd', 'black', 7], False)
    
    ax1.set_xlim(5.5,11.5)
    ax1.set_ylim(-2.4,2.3)
    
    plot_name = plot_dir + 'SCALING_RELATIONS/pilot_hifrac_mstar_all.pdf'
    plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()
  
  if do_optdisc_coloured:
    fig1 = plt.figure(2, figsize=(5, 4))
      
    xlbl         = lbl_mstar
    ylbl         = lbl_hifrac
    
    scatter_outlier_plot(fig1, 1, 1, 1, xgass_mstar, xgass_hifrac, 
                        'xGASS', xlbl, ylbl, ['darkgrey', '.', 'darkgrey', 3], False)
    
    xpar    = mstar
    ypar    = hifrac_optdisc
    limpar  = sfr_uplim
    colour  = ['darkblue', 'peru', 'mediumvioletred', 'green', 'magenta']
    poplbl  = ['Hydra TR2', 'NGC 4636 TR1', 'NGC 4808 TR1', 'NGC 5044 TR1', 'NGC 5044 TR2']
    
    for j in range(len(poplbl)):
      subsample  = (team_release == poplbl[j]) & (radius_r_iso25 > 15)
      data_array = [xpar[subsample], ypar[subsample]]
      scatter_outlier_plot(fig1, 1, 1, 1, data_array[0], data_array[1], 
                          poplbl[j], xlbl, ylbl, [colour[j], 'o', colour[j], 5], True)
    
    ax1 = fig1.add_subplot(1, 1, 1, facecolor = 'w')
    x, y, z = xgass_mstar_table[0], xgass_mstar_table[1], xgass_mstar_table[2]
    scatter_error_plot(fig1, 1, 1, 1, x, y, [z, z],
                      'xGASS Medians', xlbl, ylbl, ['black', 'd', 'black', 7], False)
    
    ax1.set_xlim(5.5,11.5)
    ax1.set_ylim(-2.4,2.3)
    
    plot_name = plot_dir + 'SCALING_RELATIONS/pilot_hifrac_mstar_optdisc_rmaj.pdf'
    plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()
    
    fig1 = plt.figure(3, figsize=(5, 4))
      
    xlbl         = lbl_mstar
    ylbl         = lbl_hifrac
    
    scatter_outlier_plot(fig1, 1, 1, 1, xgass_mstar, xgass_hifrac, 
                        'xGASS', xlbl, ylbl, ['darkgrey', '.', 'darkgrey', 3], False)
    
    xpar    = mstar
    ypar    = hifrac_optdisc
    limpar  = sfr_uplim
    colour  = ['darkblue', 'peru', 'mediumvioletred', 'green', 'magenta']
    poplbl  = ['Hydra TR2', 'NGC 4636 TR1', 'NGC 4808 TR1', 'NGC 5044 TR1', 'NGC 5044 TR2']
    
    for j in range(len(poplbl)):
      subsample  = (team_release == poplbl[j]) & (radius_r_iso25 * axis_ratio_opt > 15)
      data_array = [xpar[subsample], ypar[subsample]]
      scatter_outlier_plot(fig1, 1, 1, 1, data_array[0], data_array[1], 
                          poplbl[j], xlbl, ylbl, [colour[j], 'o', colour[j], 5], True)
    
    ax1 = fig1.add_subplot(1, 1, 1, facecolor = 'w')
    x, y, z = xgass_mstar_table[0], xgass_mstar_table[1], xgass_mstar_table[2]
    scatter_error_plot(fig1, 1, 1, 1, x, y, [z, z],
                      'xGASS Medians', xlbl, ylbl, ['black', 'd', 'black', 7], False)
    
    ax1.set_xlim(5.5,11.5)
    ax1.set_ylim(-2.4,2.3)
    
    plot_name = plot_dir + 'SCALING_RELATIONS/pilot_hifrac_mstar_optdisc_rmin.pdf'
    plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()
  
  if do_optdisc_sub:
    fig1 = plt.figure(4, figsize=(8, 3.5))
     
    xlbl    = lbl_mstar
    ylbl    = lbl_hifrac #r'$\log(M_{\mathrm{HI,total}}/M_*)$'
    
    #scatter_outlier_plot(fig1, 1, 1, 1, xgass_mstar, xgass_hifrac, 
                        #'xGASS', xlbl, ylbl, ['darkgrey', '.', 'darkgrey', 3], False)
    
    xpar    = mstar
    ypar    = hifrac
    
    bins                   = np.arange(7.5, 11.5, 0.5)
    par_mean_parent, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins_parent                  = bin_edges[:-1] + bin_width/2.
    p20_parent, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80_parent, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    subsample  = (radius_r_iso25 * axis_ratio_opt > 15)
    
    xpar    = mstar[subsample]
    ypar    = hifrac[subsample]
    limpar  = sfr_uplim[subsample]
    
    scatter_outlier_plot(fig1, 1, 2, 1, xpar, ypar, 
                        r'$M_{\mathrm{HI,total}}$', xlbl, ylbl, ['skyblue', 'o', 'skyblue', 5], False)
    
    bins                   = np.arange(7.5, 11.5, 0.5)
    par_mean_all, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins_all                  = bin_edges[:-1] + bin_width/2.
    p20_all, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80_all, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    #print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    #print(np.round(stats.pearsonr(xpar[xpar>9], ypar[xpar>9])[0], 2), stats.pearsonr(xpar[xpar>9], ypar[xpar>9])[1])
    
    ax1 = fig1.add_subplot(1, 2, 1, facecolor = 'w')
    x, y, z = xgass_mstar_table[0], xgass_mstar_table[1], xgass_mstar_table[2]
    scatter_error_plot(fig1, 1, 2, 1, x, y, [z, z],
                      'xGASS', xlbl, ylbl, ['darkgrey', 'd', 'darkgrey', 7], True)
    
    scatter_error_plot(fig1, 1, 2, 1, xbins_parent, par_mean_parent, 
                      [par_mean_parent - p20_parent, p80_parent - par_mean_parent],
                      'All', xlbl, ylbl, ['mediumvioletred', 'x', 'mediumvioletred', 5], True)
    
    scatter_error_plot(fig1, 1, 2, 1, xbins_all, par_mean_all, [par_mean_all - p20_all, p80_all - par_mean_all],
                       'Median', xlbl, ylbl, ['darkblue', 's', 'darkblue', 5], True)
    
    ax1.set_xlim(5.5,11.5)
    ax1.set_ylim(-2.4,2.3)
    
    #subsample  = (radius_r_iso25 * axis_ratio_opt > 15)
    
    xpar    = mstar[subsample]
    ypar    = hifrac_optdisc[subsample]
    limpar  = sfr_uplim[subsample]
    
    #ylbl    = r'$\log(M_{\mathrm{HI,inner}}/M_*)$'
    
    scatter_outlier_plot(fig1, 1, 2, 2, xpar, ypar, 
                        r'$M_{\mathrm{HI,inner}}$', xlbl, ylbl, ['wheat', 'o', 'wheat', 5], True)
    
    bins                   = np.arange(7.5, 11.5, 0.5)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    #print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    #print(np.round(stats.pearsonr(xpar[xpar>9], ypar[xpar>9])[0], 2), stats.pearsonr(xpar[xpar>9], ypar[xpar>9])[1])
    
    scatter_error_plot(fig1, 1, 2, 2, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                       'Median', xlbl, ylbl, ['peru', 's', 'peru', 5], True)
    
    ax1 = fig1.add_subplot(1, 2, 2, facecolor = 'w')
    x, y, z = xgass_mstar_table[0], xgass_mstar_table[1], xgass_mstar_table[2]
    scatter_error_plot(fig1, 1, 2, 2, x, y, [z, z],
                      'xGASS', xlbl, ylbl, ['darkgrey', 'd', 'darkgrey', 7], False)
    
    scatter_error_plot(fig1, 1, 2, 2, xbins_parent, par_mean_parent, 
                      [par_mean_parent - p20_parent, p80_parent - par_mean_parent],
                      'All', xlbl, ylbl, ['mediumvioletred', 'x', 'mediumvioletred', 5], False)
    
    scatter_error_plot(fig1, 1, 2, 2, xbins_all, par_mean_all, [par_mean_all - p20_all, p80_all - par_mean_all],
                       'Median', xlbl, ylbl, ['darkblue', 's', 'darkblue', 5], False)
    
    ax1.text(0.05, 0.05, r"$N=%i$" % len(xpar), 
               transform=ax1.transAxes, fontsize=11)
    
    ax1.set_xlim(5.5,11.5)
    ax1.set_ylim(-2.4,2.3)
    
    plot_name = plot_dir + 'SCALING_RELATIONS/pilot_hifrac_mstar_optdisc.pdf'
    plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()


# ================================= #
# ======== HI Gas Fraction ======== #
# ================================= #
if do_hi_fraction_ssfr:
  do_all_coloured     = False
  do_optdisc_coloured = False
  do_optdisc_sub      = True
  
  if do_all_coloured:
    fig1 = plt.figure(1, figsize=(5, 4))
      
    xlbl         = lbl_ssfr
    ylbl         = lbl_hifrac
    
    scatter_outlier_plot(fig1, 1, 1, 1, xgass_ssfr, xgass_hifrac, 
                        'xGASS', xlbl, ylbl, ['darkgrey', '.', 'darkgrey', 3], False)
    
    xpar    = ssfr
    ypar    = hifrac
    limpar  = sfr_uplim
    colour  = ['darkblue', 'peru', 'mediumvioletred', 'green', 'magenta']
    poplbl  = ['Hydra TR2', 'NGC 4636 TR1', 'NGC 4808 TR1', 'NGC 5044 TR1', 'NGC 5044 TR2']
    
    for j in range(len(poplbl)):
      subsample  = (team_release == poplbl[j])
      data_array = [xpar[subsample], ypar[subsample]]
      scatter_outlier_plot(fig1, 1, 1, 1, data_array[0], data_array[1], 
                          poplbl[j], xlbl, ylbl, [colour[j], 'o', colour[j], 5], True)
    
    ax1 = fig1.add_subplot(1, 1, 1, facecolor = 'w')
    x, y, z = xgass_mstar_table[0], xgass_mstar_table[1], xgass_mstar_table[2]
    scatter_error_plot(fig1, 1, 1, 1, x, y, [z, z],
                      'xGASS Medians', xlbl, ylbl, ['black', 'd', 'black', 7], False)
    
    ax1.set_xlim(-13, -8.5)
    ax1.set_ylim(-2.4,2.3)
    
    plot_name = plot_dir + 'SCALING_RELATIONS/pilot_hifrac_ssfr_all.pdf'
    plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()
  
  if do_optdisc_coloured:
    fig1 = plt.figure(2, figsize=(5, 4))
      
    xlbl         = lbl_ssfr
    ylbl         = lbl_hifrac
    
    scatter_outlier_plot(fig1, 1, 1, 1, xgass_ssfr, xgass_hifrac, 
                        'xGASS', xlbl, ylbl, ['darkgrey', '.', 'darkgrey', 3], False)
    
    xpar    = ssfr
    ypar    = hifrac_optdisc
    limpar  = sfr_uplim
    colour  = ['darkblue', 'peru', 'mediumvioletred', 'green', 'magenta']
    poplbl  = ['Hydra TR2', 'NGC 4636 TR1', 'NGC 4808 TR1', 'NGC 5044 TR1', 'NGC 5044 TR2']
    
    for j in range(len(poplbl)):
      subsample  = (team_release == poplbl[j]) & (radius_r_iso25 > 15)
      data_array = [xpar[subsample], ypar[subsample]]
      scatter_outlier_plot(fig1, 1, 1, 1, data_array[0], data_array[1], 
                          poplbl[j], xlbl, ylbl, [colour[j], 'o', colour[j], 5], True)
    
    ax1 = fig1.add_subplot(1, 1, 1, facecolor = 'w')
    x, y, z = xgass_ssfr_table[0], xgass_ssfr_table[1], xgass_ssfr_table[2]
    scatter_error_plot(fig1, 1, 1, 1, x, y, [z, z],
                      'xGASS Medians', xlbl, ylbl, ['black', 'd', 'black', 7], False)
    
    ax1.set_xlim(-13, -8.5)
    ax1.set_ylim(-2.4,2.3)
    
    plot_name = plot_dir + 'SCALING_RELATIONS/pilot_hifrac_ssfr_optdisc_rmaj.pdf'
    plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()

    
    fig1 = plt.figure(3, figsize=(5, 4))
      
    #xlbl         = lbl_mstar
    #ylbl         = lbl_hifrac
    
    scatter_outlier_plot(fig1, 1, 1, 1, xgass_ssfr, xgass_hifrac, 
                        'xGASS', xlbl, ylbl, ['darkgrey', '.', 'darkgrey', 3], False)
    
    xpar    = ssfr
    ypar    = hifrac_optdisc
    limpar  = sfr_uplim
    colour  = ['darkblue', 'peru', 'mediumvioletred', 'green', 'magenta']
    poplbl  = ['Hydra TR2', 'NGC 4636 TR1', 'NGC 4808 TR1', 'NGC 5044 TR1', 'NGC 5044 TR2']
    
    print(len(xpar), 
          len(xpar[(radius_r_iso25 > 15)]), 
          len(xpar[(radius_r_iso25 * axis_ratio_opt > 15)]))
    
    print(np.nanmax(ypar[(radius_r_iso25 * axis_ratio_opt > 15) & np.isfinite(ypar)]))
    
    for j in range(len(poplbl)):
      subsample  = (team_release == poplbl[j]) & (radius_r_iso25 * axis_ratio_opt > 15)
      data_array = [xpar[subsample], ypar[subsample]]
      scatter_outlier_plot(fig1, 1, 1, 1, data_array[0], data_array[1], 
                          poplbl[j], xlbl, ylbl, [colour[j], 'o', colour[j], 5], True)
    
    ax1 = fig1.add_subplot(1, 1, 1, facecolor = 'w')
    x, y, z = xgass_ssfr_table[0], xgass_ssfr_table[1], xgass_ssfr_table[2]
    scatter_error_plot(fig1, 1, 1, 1, x, y, [z, z],
                      'xGASS Medians', xlbl, ylbl, ['black', 'd', 'black', 7], False)
    
    ax1.set_xlim(-13, -8.5)
    ax1.set_ylim(-2.4,2.3)
    
    plot_name = plot_dir + 'SCALING_RELATIONS/pilot_hifrac_ssfr_optdisc_rmin.pdf'
    plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()
  
  if do_optdisc_sub:
    fig1 = plt.figure(4, figsize=(8, 3.5))
      
    xlbl         = lbl_ssfr
    ylbl         = lbl_hifrac
    
    #scatter_outlier_plot(fig1, 1, 1, 1, xgass_mstar, xgass_hifrac, 
                        #'xGASS', xlbl, ylbl, ['darkgrey', '.', 'darkgrey', 3], False)
    
    xpar    = ssfr
    ypar    = hifrac
    
    bins                   = np.arange(-11, -8.5, 0.5)
    par_mean_parent, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins_parent                  = bin_edges[:-1] + bin_width/2.
    p20_parent, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80_parent, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    subsample  = (radius_r_iso25 * axis_ratio_opt > 15)
    
    xpar    = ssfr[subsample]
    ypar    = hifrac[subsample]
    limpar  = sfr_uplim[subsample]
    
    scatter_outlier_plot(fig1, 1, 2, 1, xpar, ypar, 
                        r'$M_{\mathrm{HI,total}}$', xlbl, ylbl, ['skyblue', 'o', 'skyblue', 5], True)
    
    bins                   = np.arange(-11, -8.5, 0.5)
    par_mean_all, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins_all                  = bin_edges[:-1] + bin_width/2.
    p20_all, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80_all, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    #print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    #print(np.round(stats.pearsonr(xpar[xpar>9], ypar[xpar>9])[0], 2), stats.pearsonr(xpar[xpar>9], ypar[xpar>9])[1])
    
    ax1 = fig1.add_subplot(1, 2, 1, facecolor = 'w')
    x, y, z = xgass_ssfr_table[0], xgass_ssfr_table[1], xgass_ssfr_table[2]
    scatter_error_plot(fig1, 1, 2, 1, x, y, [z, z],
                      'xGASS', xlbl, ylbl, ['darkgrey', 'd', 'darkgrey', 7], True)
    
    scatter_error_plot(fig1, 1, 2, 1, xbins_parent, par_mean_parent, 
                      [par_mean_parent - p20_parent, p80_parent - par_mean_parent],
                      'All', xlbl, ylbl, ['mediumvioletred', 'x', 'mediumvioletred', 5], True)
    
    scatter_error_plot(fig1, 1, 2, 1, xbins_all, par_mean_all, [par_mean_all - p20_all, p80_all - par_mean_all],
                       'Medians', xlbl, ylbl, ['darkblue', 's', 'darkblue', 5], True)
    
    ax1.set_xlim(-13, -8.5)
    ax1.set_ylim(-2.4,2.3)
    
    #subsample  = (radius_r_iso25 * axis_ratio_opt > 15)
    
    xpar    = ssfr[subsample]
    ypar    = hifrac_optdisc[subsample]
    limpar  = sfr_uplim[subsample]
    
    scatter_outlier_plot(fig1, 1, 2, 2, xpar, ypar, 
                        r'$M_{\mathrm{HI,inner}}$', xlbl, ylbl, ['wheat', 'o', 'wheat', 5], True)
    
    bins                   = np.arange(-11, -8.5, 0.5)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    #print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    #print(np.round(stats.pearsonr(xpar[xpar>9], ypar[xpar>9])[0], 2), stats.pearsonr(xpar[xpar>9], ypar[xpar>9])[1])
    
    scatter_error_plot(fig1, 1, 2, 2, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                       'Medians', xlbl, ylbl, ['peru', 's', 'peru', 5], True)
    
    ax1 = fig1.add_subplot(1, 2, 2, facecolor = 'w')
    x, y, z = xgass_ssfr_table[0], xgass_ssfr_table[1], xgass_ssfr_table[2]
    scatter_error_plot(fig1, 1, 2, 2, x, y, [z, z],
                      'xGASS', xlbl, ylbl, ['darkgrey', 'd', 'darkgrey', 7], False)
    
    scatter_error_plot(fig1, 1, 2, 2, xbins_parent, par_mean_parent, 
                       [par_mean_parent - p20_parent, p80_parent - par_mean_parent],
                       'All', xlbl, ylbl, ['mediumvioletred', 'x', 'mediumvioletred', 5], False)
    
    scatter_error_plot(fig1, 1, 2, 2, xbins_all, par_mean_all, [par_mean_all - p20_all, p80_all - par_mean_all],
                       'Medians', xlbl, ylbl, ['darkblue', 's', 'darkblue', 5], False)
    
    scatter_error_plot(fig1, 1, 2, 2, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                       'Medians', xlbl, ylbl, ['peru', 's', 'peru', 5], False)
    
    
    
    ax1.text(0.05, 0.05, r"$N=%i$" % len(xpar), 
               transform=ax1.transAxes, fontsize=11)
    
    ax1.set_xlim(-13, -8.5)
    ax1.set_ylim(-2.4,2.3)
    
    plot_name = plot_dir + 'SCALING_RELATIONS/pilot_hifrac_ssfr_optdisc.pdf'
    plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()


# ================================= #
# ======== HI Gas Fraction ======== #
# ================================= #
if do_hi_fraction_nuvr:
  do_all_coloured     = False
  do_optdisc_coloured = False
  do_optdisc_sub      = True
  
  if do_all_coloured:
    fig1 = plt.figure(1, figsize=(5, 4))
      
    xlbl         = lbl_nuvr
    ylbl         = lbl_hifrac
    
    scatter_outlier_plot(fig1, 1, 1, 1, xgass_ssfr, xgass_hifrac, 
                        'xGASS', xlbl, ylbl, ['darkgrey', '.', 'darkgrey', 3], False)
    
    xpar    = nuvr
    ypar    = hifrac
    limpar  = sfr_uplim
    colour  = ['darkblue', 'peru', 'mediumvioletred', 'green', 'magenta']
    poplbl  = ['Hydra TR2', 'NGC 4636 TR1', 'NGC 4808 TR1', 'NGC 5044 TR1', 'NGC 5044 TR2']
    
    for j in range(len(poplbl)):
      subsample  = (team_release == poplbl[j])
      data_array = [xpar[subsample], ypar[subsample]]
      scatter_outlier_plot(fig1, 1, 1, 1, data_array[0], data_array[1], 
                          poplbl[j], xlbl, ylbl, [colour[j], 'o', colour[j], 5], True)
    
    ax1 = fig1.add_subplot(1, 1, 1, facecolor = 'w')
    x, y, z = xgass_nuvr_table[0], xgass_nuvr_table[1], xgass_nuvr_table[2]
    scatter_error_plot(fig1, 1, 1, 1, x, y, [z, z],
                      'xGASS Medians', xlbl, ylbl, ['black', 'd', 'black', 7], False)
    
    ax1.set_xlim(0, 5)
    ax1.set_ylim(-2.4,2.3)
    
    plot_name = plot_dir + 'SCALING_RELATIONS/pilot_hifrac_nuvr_all.pdf'
    plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()
  
  if do_optdisc_coloured:
    fig1 = plt.figure(2, figsize=(5, 4))
      
    xlbl         = lbl_nuvr
    ylbl         = lbl_hifrac
    
    scatter_outlier_plot(fig1, 1, 1, 1, xgass_ssfr, xgass_hifrac, 
                        'xGASS', xlbl, ylbl, ['darkgrey', '.', 'darkgrey', 3], False)
    
    xpar    = nuvr
    ypar    = hifrac_optdisc
    limpar  = sfr_uplim
    colour  = ['darkblue', 'peru', 'mediumvioletred', 'green', 'magenta']
    poplbl  = ['Hydra TR2', 'NGC 4636 TR1', 'NGC 4808 TR1', 'NGC 5044 TR1', 'NGC 5044 TR2']
    
    for j in range(len(poplbl)):
      subsample  = (team_release == poplbl[j]) & (radius_r_iso25 > 15)
      data_array = [xpar[subsample], ypar[subsample]]
      scatter_outlier_plot(fig1, 1, 1, 1, data_array[0], data_array[1], 
                          poplbl[j], xlbl, ylbl, [colour[j], 'o', colour[j], 5], True)
    
    ax1 = fig1.add_subplot(1, 1, 1, facecolor = 'w')
    x, y, z = xgass_nuvr_table[0], xgass_nuvr_table[1], xgass_nuvr_table[2]
    scatter_error_plot(fig1, 1, 1, 1, x, y, [z, z],
                      'xGASS Medians', xlbl, ylbl, ['black', 'd', 'black', 7], False)
    
    ax1.set_xlim(0, 5)
    ax1.set_ylim(-2.4,2.3)
    
    plot_name = plot_dir + 'SCALING_RELATIONS/pilot_hifrac_nuvr_optdisc_rmaj.pdf'
    plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()

    
    fig1 = plt.figure(3, figsize=(5, 4))
      
    #xlbl         = lbl_mstar
    #ylbl         = lbl_hifrac
    
    scatter_outlier_plot(fig1, 1, 1, 1, xgass_ssfr, xgass_hifrac, 
                        'xGASS', xlbl, ylbl, ['darkgrey', '.', 'darkgrey', 3], False)
    
    xpar    = nuvr
    ypar    = hifrac_optdisc
    limpar  = sfr_uplim
    colour  = ['darkblue', 'peru', 'mediumvioletred', 'green', 'magenta']
    poplbl  = ['Hydra TR2', 'NGC 4636 TR1', 'NGC 4808 TR1', 'NGC 5044 TR1', 'NGC 5044 TR2']
    
    print(len(xpar), 
          len(xpar[(radius_r_iso25 > 15)]), 
          len(xpar[(radius_r_iso25 * axis_ratio_opt > 15)]))
    
    print(np.nanmax(ypar[(radius_r_iso25 * axis_ratio_opt > 15) & np.isfinite(ypar)]))
    
    for j in range(len(poplbl)):
      subsample  = (team_release == poplbl[j]) & (radius_r_iso25 * axis_ratio_opt > 15)
      data_array = [xpar[subsample], ypar[subsample]]
      scatter_outlier_plot(fig1, 1, 1, 1, data_array[0], data_array[1], 
                          poplbl[j], xlbl, ylbl, [colour[j], 'o', colour[j], 5], True)
    
    ax1 = fig1.add_subplot(1, 1, 1, facecolor = 'w')
    x, y, z = xgass_nuvr_table[0], xgass_nuvr_table[1], xgass_nuvr_table[2]
    scatter_error_plot(fig1, 1, 1, 1, x, y, [z, z],
                      'xGASS Medians', xlbl, ylbl, ['black', 'd', 'black', 7], False)
    
    ax1.set_xlim(0, 5)
    ax1.set_ylim(-2.4,2.3)
    
    plot_name = plot_dir + 'SCALING_RELATIONS/pilot_hifrac_nuvr_optdisc_rmin.pdf'
    plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()
  
  if do_optdisc_sub:
    fig1 = plt.figure(4, figsize=(8, 3.5))
      
    xlbl         = lbl_nuvr
    ylbl         = lbl_hifrac
    
    #scatter_outlier_plot(fig1, 1, 1, 1, xgass_mstar, xgass_hifrac, 
                        #'xGASS', xlbl, ylbl, ['darkgrey', '.', 'darkgrey', 3], False)
    
    xpar    = nuvr
    ypar    = hifrac
    
    bins                   = np.arange(1, 4, 0.5)
    par_mean_parent, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins_parent                  = bin_edges[:-1] + bin_width/2.
    p20_parent, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80_parent, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    subsample  = (radius_r_iso25 * axis_ratio_opt > 15)
    
    xpar    = nuvr[subsample]
    ypar    = hifrac[subsample]
    limpar  = sfr_uplim[subsample]
    
    scatter_outlier_plot(fig1, 1, 2, 1, xpar, ypar, 
                        r'$M_{\mathrm{HI,total}}$', xlbl, ylbl, ['skyblue', 'o', 'skyblue', 5], True)
    
    bins                   = np.arange(1, 4, 0.5)
    par_mean_all, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins_all                  = bin_edges[:-1] + bin_width/2.
    p20_all, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80_all, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    #print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    #print(np.round(stats.pearsonr(xpar[xpar>9], ypar[xpar>9])[0], 2), stats.pearsonr(xpar[xpar>9], ypar[xpar>9])[1])
    
    ax1 = fig1.add_subplot(1, 2, 1, facecolor = 'w')
    x, y, z = xgass_nuvr_table[0], xgass_nuvr_table[1], xgass_nuvr_table[2]
    scatter_error_plot(fig1, 1, 2, 1, x, y, [z, z],
                      'xGASS', xlbl, ylbl, ['darkgrey', 'd', 'darkgrey', 7], True)
    
    scatter_error_plot(fig1, 1, 2, 1, xbins_parent, par_mean_parent, 
                      [par_mean_parent - p20_parent, p80_parent - par_mean_parent],
                      'All', xlbl, ylbl, ['mediumvioletred', 'x', 'mediumvioletred', 5], True)
    
    scatter_error_plot(fig1, 1, 2, 1, xbins_all, par_mean_all, [par_mean_all - p20_all, p80_all - par_mean_all],
                       'Medians', xlbl, ylbl, ['darkblue', 's', 'darkblue', 5], True)
    
    ax1.set_xlim(0, 5)
    ax1.set_ylim(-2.4,2.3)
    
    #subsample  = (radius_r_iso25 * axis_ratio_opt > 15)
    
    xpar    = nuvr[subsample]
    ypar    = hifrac_optdisc[subsample]
    limpar  = sfr_uplim[subsample]
    
    scatter_outlier_plot(fig1, 1, 2, 2, xpar, ypar, 
                        r'$M_{\mathrm{HI,inner}}$', xlbl, ylbl, ['wheat', 'o', 'wheat', 5], True)
    
    bins                   = np.arange(1, 4, 0.5)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    #print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    #print(np.round(stats.pearsonr(xpar[xpar>9], ypar[xpar>9])[0], 2), stats.pearsonr(xpar[xpar>9], ypar[xpar>9])[1])
    
    scatter_error_plot(fig1, 1, 2, 2, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                       'Medians', xlbl, ylbl, ['peru', 's', 'peru', 5], True)
    
    ax1 = fig1.add_subplot(1, 2, 2, facecolor = 'w')
    x, y, z = xgass_nuvr_table[0], xgass_nuvr_table[1], xgass_nuvr_table[2]
    scatter_error_plot(fig1, 1, 2, 2, x, y, [z, z],
                      'xGASS', xlbl, ylbl, ['darkgrey', 'd', 'darkgrey', 7], False)
    
    scatter_error_plot(fig1, 1, 2, 2, xbins_parent, par_mean_parent, 
                       [par_mean_parent - p20_parent, p80_parent - par_mean_parent],
                       'All', xlbl, ylbl, ['mediumvioletred', 'x', 'mediumvioletred', 5], False)
    
    scatter_error_plot(fig1, 1, 2, 2, xbins_all, par_mean_all, [par_mean_all - p20_all, p80_all - par_mean_all],
                       'Medians', xlbl, ylbl, ['darkblue', 's', 'darkblue', 5], False)
    
    scatter_error_plot(fig1, 1, 2, 2, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                       'Medians', xlbl, ylbl, ['peru', 's', 'peru', 5], False)
    
    
    
    ax1.text(0.05, 0.05, r"$N=%i$" % len(xpar), 
               transform=ax1.transAxes, fontsize=11)
    
    ax1.set_xlim(0, 5)
    ax1.set_ylim(-2.4,2.3)
    
    plot_name = plot_dir + 'SCALING_RELATIONS/pilot_hifrac_nuvr_optdisc.pdf'
    plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()


# ================================= #
# ======== HI Gas Fraction ======== #
# ================================= #
if do_hi_fraction_dsfms:
  do_all_coloured     = False
  do_optdisc_coloured = False
  do_optdisc_sub      = True
  
  if do_all_coloured:
    fig1 = plt.figure(1, figsize=(5, 4))
      
    xlbl         = r'$\Delta$ SFMS'
    ylbl         = lbl_hifrac
    
    scatter_outlier_plot(fig1, 1, 1, 1, xgass_ssfr, xgass_hifrac, 
                        'xGASS', xlbl, ylbl, ['darkgrey', '.', 'darkgrey', 3], False)
    
    xpar    = delta_sfms
    ypar    = hifrac
    limpar  = sfr_uplim
    colour  = ['darkblue', 'peru', 'mediumvioletred', 'green', 'magenta']
    poplbl  = ['Hydra TR2', 'NGC 4636 TR1', 'NGC 4808 TR1', 'NGC 5044 TR1', 'NGC 5044 TR2']
    
    for j in range(len(poplbl)):
      subsample  = (team_release == poplbl[j])
      data_array = [xpar[subsample], ypar[subsample]]
      scatter_outlier_plot(fig1, 1, 1, 1, data_array[0], data_array[1], 
                          poplbl[j], xlbl, ylbl, [colour[j], 'o', colour[j], 5], True)
    
    ax1 = fig1.add_subplot(1, 1, 1, facecolor = 'w')
    x, y, z = xgass_nuvr_table[0], xgass_nuvr_table[1], xgass_nuvr_table[2]
    scatter_error_plot(fig1, 1, 1, 1, x, y, [z, z],
                      'xGASS Medians', xlbl, ylbl, ['black', 'd', 'black', 7], False)
    
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2.4,2.3)
    
    plot_name = plot_dir + 'SCALING_RELATIONS/pilot_hifrac_dsfms_all.pdf'
    plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()
  
  if do_optdisc_coloured:
    fig1 = plt.figure(2, figsize=(5, 4))
      
    xlbl         = r'$\Delta$ SFMS'
    ylbl         = lbl_hifrac
    
    scatter_outlier_plot(fig1, 1, 1, 1, xgass_ssfr, xgass_hifrac, 
                        'xGASS', xlbl, ylbl, ['darkgrey', '.', 'darkgrey', 3], False)
    
    xpar    = delta_sfms
    ypar    = hifrac_optdisc
    limpar  = sfr_uplim
    colour  = ['darkblue', 'peru', 'mediumvioletred', 'green', 'magenta']
    poplbl  = ['Hydra TR2', 'NGC 4636 TR1', 'NGC 4808 TR1', 'NGC 5044 TR1', 'NGC 5044 TR2']
    
    for j in range(len(poplbl)):
      subsample  = (team_release == poplbl[j]) & (radius_r_iso25 > 15)
      data_array = [xpar[subsample], ypar[subsample]]
      scatter_outlier_plot(fig1, 1, 1, 1, data_array[0], data_array[1], 
                          poplbl[j], xlbl, ylbl, [colour[j], 'o', colour[j], 5], True)
    
    ax1 = fig1.add_subplot(1, 1, 1, facecolor = 'w')
    x, y, z = xgass_nuvr_table[0], xgass_nuvr_table[1], xgass_nuvr_table[2]
    scatter_error_plot(fig1, 1, 1, 1, x, y, [z, z],
                      'xGASS Medians', xlbl, ylbl, ['black', 'd', 'black', 7], False)
    
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2.4,2.3)
    
    plot_name = plot_dir + 'SCALING_RELATIONS/pilot_hifrac_dsfms_optdisc_rmaj.pdf'
    plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()

    
    fig1 = plt.figure(3, figsize=(5, 4))
      
    #xlbl         = lbl_mstar
    #ylbl         = lbl_hifrac
    
    scatter_outlier_plot(fig1, 1, 1, 1, xgass_ssfr, xgass_hifrac, 
                        'xGASS', xlbl, ylbl, ['darkgrey', '.', 'darkgrey', 3], False)
    
    xpar    = delta_sfms
    ypar    = hifrac_optdisc
    limpar  = sfr_uplim
    colour  = ['darkblue', 'peru', 'mediumvioletred', 'green', 'magenta']
    poplbl  = ['Hydra TR2', 'NGC 4636 TR1', 'NGC 4808 TR1', 'NGC 5044 TR1', 'NGC 5044 TR2']
    
    print(len(xpar), 
          len(xpar[(radius_r_iso25 > 15)]), 
          len(xpar[(radius_r_iso25 * axis_ratio_opt > 15)]))
    
    print(np.nanmax(ypar[(radius_r_iso25 * axis_ratio_opt > 15) & np.isfinite(ypar)]))
    
    for j in range(len(poplbl)):
      subsample  = (team_release == poplbl[j]) & (radius_r_iso25 * axis_ratio_opt > 15)
      data_array = [xpar[subsample], ypar[subsample]]
      scatter_outlier_plot(fig1, 1, 1, 1, data_array[0], data_array[1], 
                          poplbl[j], xlbl, ylbl, [colour[j], 'o', colour[j], 5], True)
    
    ax1 = fig1.add_subplot(1, 1, 1, facecolor = 'w')
    x, y, z = xgass_nuvr_table[0], xgass_nuvr_table[1], xgass_nuvr_table[2]
    scatter_error_plot(fig1, 1, 1, 1, x, y, [z, z],
                      'xGASS Medians', xlbl, ylbl, ['black', 'd', 'black', 7], False)
    
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2.4,2.3)
    
    plot_name = plot_dir + 'SCALING_RELATIONS/pilot_hifrac_dsfms_optdisc_rmin.pdf'
    plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()
  
  if do_optdisc_sub:
    fig1 = plt.figure(4, figsize=(8, 3.5))
      
    xlbl         = r'$\Delta$ SFMS'
    ylbl         = lbl_hifrac
    
    #scatter_outlier_plot(fig1, 1, 1, 1, xgass_mstar, xgass_hifrac, 
                        #'xGASS', xlbl, ylbl, ['darkgrey', '.', 'darkgrey', 3], False)
    
    xpar    = delta_sfms
    ypar    = hifrac
    
    bins                   = np.arange(-1, 1.5, 0.5)
    par_mean_parent, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins_parent                  = bin_edges[:-1] + bin_width/2.
    p20_parent, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80_parent, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    subsample  = (radius_r_iso25 * axis_ratio_opt > 15)
    
    xpar    = delta_sfms[subsample]
    ypar    = hifrac[subsample]
    limpar  = sfr_uplim[subsample]
    
    scatter_outlier_plot(fig1, 1, 2, 1, xpar, ypar, 
                        r'$M_{\mathrm{HI,total}}$', xlbl, ylbl, ['skyblue', 'o', 'skyblue', 5], True)
    
    bins                   = np.arange(-1, 1.5, 0.5)
    par_mean_all, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins_all                  = bin_edges[:-1] + bin_width/2.
    p20_all, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80_all, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    #print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    #print(np.round(stats.pearsonr(xpar[xpar>9], ypar[xpar>9])[0], 2), stats.pearsonr(xpar[xpar>9], ypar[xpar>9])[1])
    
    ax1 = fig1.add_subplot(1, 2, 1, facecolor = 'w')
    #x, y, z = xgass_nuvr_table[0], xgass_nuvr_table[1], xgass_nuvr_table[2]
    #scatter_error_plot(fig1, 1, 2, 1, x, y, [z, z],
                      #'xGASS', xlbl, ylbl, ['darkgrey', 'd', 'darkgrey', 7], True)
    
    scatter_error_plot(fig1, 1, 2, 1, xbins_parent, par_mean_parent, 
                      [par_mean_parent - p20_parent, p80_parent - par_mean_parent],
                      'All', xlbl, ylbl, ['mediumvioletred', 'x', 'mediumvioletred', 5], True)
    
    scatter_error_plot(fig1, 1, 2, 1, xbins_all, par_mean_all, [par_mean_all - p20_all, p80_all - par_mean_all],
                       'Medians', xlbl, ylbl, ['darkblue', 's', 'darkblue', 5], True)
    
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2.4,2.3)
    
    #subsample  = (radius_r_iso25 * axis_ratio_opt > 15)
    
    xpar    = delta_sfms[subsample]
    ypar    = hifrac_optdisc[subsample]
    limpar  = sfr_uplim[subsample]
    
    scatter_outlier_plot(fig1, 1, 2, 2, xpar, ypar, 
                        r'$M_{\mathrm{HI,inner}}$', xlbl, ylbl, ['wheat', 'o', 'wheat', 5], True)
    
    bins                   = np.arange(-1, 1.5, 0.5)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    #print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    #print(np.round(stats.pearsonr(xpar[xpar>9], ypar[xpar>9])[0], 2), stats.pearsonr(xpar[xpar>9], ypar[xpar>9])[1])
    
    scatter_error_plot(fig1, 1, 2, 2, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                       'Medians', xlbl, ylbl, ['peru', 's', 'peru', 5], True)
    
    ax1 = fig1.add_subplot(1, 2, 2, facecolor = 'w')
    #x, y, z = xgass_nuvr_table[0], xgass_nuvr_table[1], xgass_nuvr_table[2]
    #scatter_error_plot(fig1, 1, 2, 2, x, y, [z, z],
                      #'xGASS', xlbl, ylbl, ['darkgrey', 'd', 'darkgrey', 7], False)
    
    scatter_error_plot(fig1, 1, 2, 2, xbins_parent, par_mean_parent, 
                       [par_mean_parent - p20_parent, p80_parent - par_mean_parent],
                       'All', xlbl, ylbl, ['mediumvioletred', 'x', 'mediumvioletred', 5], False)
    
    scatter_error_plot(fig1, 1, 2, 2, xbins_all, par_mean_all, [par_mean_all - p20_all, p80_all - par_mean_all],
                       'Medians', xlbl, ylbl, ['darkblue', 's', 'darkblue', 5], False)
    
    scatter_error_plot(fig1, 1, 2, 2, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                       'Medians', xlbl, ylbl, ['peru', 's', 'peru', 5], False)
    
    
    
    ax1.text(0.05, 0.05, r"$N=%i$" % len(xpar), 
               transform=ax1.transAxes, fontsize=11)
    
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2.4,2.3)
    
    plot_name = plot_dir + 'SCALING_RELATIONS/pilot_hifrac_dsfms_optdisc.pdf'
    plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()


# ========== HI Gas Fraction vs Stellar Mass =========== #
'''
if do_hifrac_mstar:
  fig4 = plt.figure(5, figsize=(5, 4))#, facecolor = '#007D7D')
  xlbl         = lbl_mstar
  ylbl         = lbl_hifrac
  
  xpar    = mstar
  #ypar    = hifrac
  ypar    = hifrac_optdisc
  #limpar  = [w3_uplim, w4_uplim, w34_uplim]
  limpar  = no_galex
  colour  = ['darkblue', 'peru']
  poplbl  = ['Hydra', 'NGC4636']
  
  gas_fraction_plot(fig4, [1, 1, 1], [xgass_mstar, xgass_hifrac], 'xGASS', xlbl, ylbl, 'darkgrey',  '.')
  #scat_mean_plot(fig4, 1, 1, 1, , False, False, 
                  #'xGASS', xlbl, ylbl, 'lightgrey',  '.', False)
  
  gas_fraction_plot(fig4, [1, 1, 1], xgass_mstar_table, 'xGASS Median', xlbl, ylbl, 'limegreen',  'd')
  #scat_mean_plot(fig4, 1, 1, 1, xgass_mstar_table[0], xgass_mstar_table[1], False, False, 
                  #'xGASS', xlbl, ylbl, 'red',  'd', False)
  
  for i in range(2):
    subsample  = (wallaby_field == poplbl[i]) & np.isfinite(xpar) & np.isfinite(ypar) & (radius_r_iso25 > 30)
    data_array = [xpar[subsample], ypar[subsample]]
    #sfr_ms_plot(fig1, [1, 1, 1], data_array, sfms_const, 
                #poplbl[j], xlbl, ylbl, colour[j], 'o')
    gas_fraction_plot(fig4, [1, 1, 1], data_array, poplbl[i], xlbl, ylbl, colour[i],  'o')
    #scat_mean_plot(fig4, 1, 1, 1, data_array[0], data_array[1], False, False, 
                  #poplbl[i], xlbl, ylbl, colour[i],  'o', False)
  
  #sfr_ms   = 0.656 * mstar - 6.816
  plot_name = phase1_dir + 'PLOTS/PROPERTIES/SCATTER/hifrac_mstar_optdisc_resolved.pdf'
  plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
  plt.clf()
  #plt.show()
'''

if do_hifrac_mstar:
  fig4 = plt.figure(5, figsize=(6, 5))#, facecolor = '#007D7D')
  xlbl         = lbl_mstar
  ylbl         = lbl_hifrac
  
  xpar    = mstar
  ypar    = hifrac
  #ypar    = hifrac_optdisc
  colour  = ['darkblue', 'peru']
  poplbl  = ['Hydra', 'NGC4636']
  
  #gas_fraction_plot(fig4, [1, 1, 1], [xgass_mstar, xgass_hifrac], 'xGASS', xlbl, ylbl, 'darkgrey',  '.')
  
  #gas_fraction_plot(fig4, [1, 1, 1], xgass_mstar_table, 'xGASS Median', xlbl, ylbl, 'limegreen',  'd')
  
  subsample  = np.isfinite(xpar) & np.isfinite(ypar) & (radius_hi_iso > 30)
  #data_array = [xpar[subsample], ypar[subsample]]
  #gas_fraction_plot(fig4, [1, 1, 1], data_array, 'WALLABY', xlbl, ylbl, 'darkblue', 'o')
  
  #data_array = [xpar[subsample & pop_hi_array[3]], ypar[subsample & pop_hi_array[3]]]
  #gas_fraction_plot(fig4, [1, 1, 1], data_array, 'Field', xlbl, ylbl, 'green', 'o')
  #data_array = [xpar[subsample & pop_hi_array[0]], ypar[subsample & pop_hi_array[0]]]
  #gas_fraction_plot(fig4, [1, 1, 1], data_array, 'Cluster', xlbl, ylbl, 'darkblue', 'o')
  #data_array = [xpar[subsample & pop_hi_array[1]], ypar[subsample & pop_hi_array[1]]]
  #gas_fraction_plot(fig4, [1, 1, 1], data_array, 'Infall', xlbl, ylbl, 'mediumvioletred', 'o')
  #data_array = [xpar[subsample & pop_hi_array[2]], ypar[subsample & pop_hi_array[2]]]
  #gas_fraction_plot(fig4, [1, 1, 1], data_array, 'Field', xlbl, ylbl, 'peru', 'o')
  
  scatter_outlier_plot(fig4, 1, 1, 1, xgass_mstar, xgass_hifrac,
                       'xGASS', xlbl, ylbl, ['darkgrey',  'o', 'darkgrey', 2], True)
  scatter_outlier_plot(fig4, 1, 1, 1, xgass_mstar_table[0], xgass_mstar_table[1],
                       'xGASS Median', xlbl, ylbl, ['black',  'd', 'black', 50], True)
  
  #radius_r_iso25 = np.tan(10**(0.31 * mstar - 1.94) / distance / 1000.) * (180. / math.pi) * 3600.
  
  iso_size_ratio = np.log10(radius_hi_iso / radius_r_iso25)
  
  subsample   = ((radius_hi_iso > 30) & np.isfinite(mstar) & 
                     np.isfinite(np.log10(radius_hi_iso/radius_r_iso25)))
  
  #data_array = [xpar[subsample & (iso_size_ratio > 0.4)], ypar[subsample & (iso_size_ratio > 0.4)]]
  #scatter_outlier_plot(fig4, 1, 1, 1, data_array[0], data_array[1],
                       #'High', xlbl, ylbl, ['darkblue',  'o', 'darkblue', 25], True)
  #data_array = [xpar[subsample & (iso_size_ratio < 0.4) & (iso_size_ratio > 0.2)], 
                #ypar[subsample & (iso_size_ratio < 0.4) & (iso_size_ratio > 0.2)]]
  #scatter_outlier_plot(fig4, 1, 1, 1, data_array[0], data_array[1],
                       #'Middle', xlbl, ylbl, ['peru',  'o', 'peru', 25], True)
  #data_array = [xpar[subsample & (iso_size_ratio < 0.2) & (iso_size_ratio > 0.0)], 
                #ypar[subsample & (iso_size_ratio < 0.2) & (iso_size_ratio > 0.0)]]
  #scatter_outlier_plot(fig4, 1, 1, 1, data_array[0], data_array[1],
                       #'Low', xlbl, ylbl, ['mediumvioletred',  'o', 'mediumvioletred', 25], True)
  
  
  #data_array = [xpar[subsample & pop_hi_array[4]], ypar[subsample & pop_hi_array[4]]]
  #scatter_outlier_plot(fig4, 1, 1, 1, data_array[0], data_array[1],
                       #'NGC 4636', xlbl, ylbl, ['green',  'o', 'green', 25], True)
  #data_array = [xpar[subsample & pop_hi_array[0]], ypar[subsample & pop_hi_array[0]]]
  #scatter_outlier_plot(fig4, 1, 1, 1, data_array[0], data_array[1],
                       #'Cluster', xlbl, ylbl, ['blue',  'o', 'blue', 25], True)
  #data_array = [xpar[subsample & pop_hi_array[1]], ypar[subsample & pop_hi_array[1]]]
  #scatter_outlier_plot(fig4, 1, 1, 1, data_array[0], data_array[1],
                       #'Infall', xlbl, ylbl, ['mediumvioletred',  'o', 'mediumvioletred', 25], True)
  #data_array = [xpar[subsample & pop_hi_array[2]], ypar[subsample & pop_hi_array[2]]]
  #scatter_outlier_plot(fig4, 1, 1, 1, data_array[0], data_array[1],
                       #'Field', xlbl, ylbl, ['peru',  'o', 'peru', 25], True)
  #data_array = [xpar[subsample & pop_hi_array[3]], ypar[subsample & pop_hi_array[3]]]
  #scatter_outlier_plot(fig4, 1, 1, 1, data_array[0], data_array[1],
                       #'Background', xlbl, ylbl, ['teal',  'o', 'teal', 25], True)
  
  
  #[(radius_hi_iso[subsample] > 45)]
  #scatter_outlier_plot(fig4, 1, 1, 1, 
                       #data_array[0][subsample_b[i] & (data_to_plot[2][subsample] < 45)], 
                       #data_array[1][subsample_b[i] & (data_to_plot[2][subsample] < 45)],
                       #r'$<3$ beams', xlbl, ylbl, [colour[i],  'o', 'none', 25], legend)
  
  xlbl        = lbl_mstar
  ylbl        = lbl_hifrac
  zlbl        = r'$\log(R_{\rm{iso,HI}}/R_{\rm{iso25,r}})$'
  
  xpar    = mstar[subsample]
  ypar    = hifrac[subsample]
  zpar    = iso_size_ratio[subsample]
  scat_col_simple_plot2(fig4, 1, 1, 1, 
                        xpar[radius_hi_iso[subsample] > 45], 
                        ypar[radius_hi_iso[subsample] > 45], 
                        zpar[radius_hi_iso[subsample] > 45], 
                        [r'$>3$ beams', xlbl, ylbl, zlbl], 
                        ['rainbow_r', 'o', 'rainbow_r', 25, -0.2, 0.9], True, True)

  scat_col_simple_plot2(fig4, 1, 1, 1, 
                        xpar[radius_hi_iso[subsample] < 45], 
                        ypar[radius_hi_iso[subsample] < 45], 
                        zpar[radius_hi_iso[subsample] < 45], 
                        [r'$<3$ beams', xlbl, ylbl, zlbl], 
                        ['rainbow_r', 'o', 'none', 25, -0.2, 0.9], True, False)
  
  #sfr_ms   = 0.656 * mstar - 6.816
  plot_name = phase1_dir + 'PLOTS/PAPER3/hifrac_mstar.pdf'
  plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
  plt.clf()
  #plt.show()
  
  fig6 = plt.figure(6, figsize=(6, 5))
  
  zlbl        = lbl_mstar
  xlbl        = lbl_hifrac
  ylbl        = r'$\log(R_{\rm{iso,HI}}/R_{\rm{iso25,r}})$'
  
  xpar    = hifrac[subsample]
  ypar    = iso_size_ratio[subsample]
  zpar    = mstar[subsample]
  scat_col_simple_plot2(fig6, 1, 1, 1, 
                        xpar[radius_hi_iso[subsample] > 45], 
                        ypar[radius_hi_iso[subsample] > 45], 
                        zpar[radius_hi_iso[subsample] > 45], 
                        [r'$>3$ beams', xlbl, ylbl, zlbl], 
                        ['rainbow', 'o', 'rainbow', 25, 7, 11], True, True)

  scat_col_simple_plot2(fig6, 1, 1, 1, 
                        xpar[radius_hi_iso[subsample] < 45], 
                        ypar[radius_hi_iso[subsample] < 45], 
                        zpar[radius_hi_iso[subsample] < 45], 
                        [r'$<3$ beams', xlbl, ylbl, zlbl], 
                        ['rainbow', 'o', 'none', 25, 7, 11], True, False)
  
  bins                   = np.arange(-1.5, 1.75, 0.25)
  par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
  bin_width              = np.abs(bin_edges[1] - bin_edges[0])
  xbins                  = bin_edges[:-1] + bin_width/2.
  
  p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
  p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
  
  scatter_error_plot(fig6, 1, 1, 1, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                        'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], True)
  
  #sfr_ms   = 0.656 * mstar - 6.816
  plot_name = phase1_dir + 'PLOTS/PAPER3/sratio_hifrac_mstar.pdf'
  plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
  plt.clf()
  
  fig7 = plt.figure(7, figsize=(6, 5))
  
  zlbl        = lbl_mhi
  xlbl        = lbl_hifrac
  ylbl        = r'$\log(R_{\rm{iso,HI}}/R_{\rm{iso25,r}})$'
  
  xpar    = hifrac[subsample]
  ypar    = iso_size_ratio[subsample]
  zpar    = mhi[subsample]
  scat_col_simple_plot2(fig7, 1, 1, 1, 
                        xpar[radius_hi_iso[subsample] > 45], 
                        ypar[radius_hi_iso[subsample] > 45], 
                        zpar[radius_hi_iso[subsample] > 45], 
                        [r'$>3$ beams', xlbl, ylbl, zlbl], 
                        ['rainbow_r', 'o', 'rainbow_r', 25, 7, 11], True, True)

  scat_col_simple_plot2(fig7, 1, 1, 1, 
                        xpar[radius_hi_iso[subsample] < 45], 
                        ypar[radius_hi_iso[subsample] < 45], 
                        zpar[radius_hi_iso[subsample] < 45], 
                        [r'$<3$ beams', xlbl, ylbl, zlbl], 
                        ['rainbow_r', 'o', 'none', 25, 7, 11], True, False)
  
  bins                   = np.arange(-1.5, 1.75, 0.25)
  par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
  bin_width              = np.abs(bin_edges[1] - bin_edges[0])
  xbins                  = bin_edges[:-1] + bin_width/2.
  
  p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
  p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
  
  scatter_error_plot(fig7, 1, 1, 1, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                        'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], True)
  
  #sfr_ms   = 0.656 * mstar - 6.816
  plot_name = phase1_dir + 'PLOTS/PAPER3/sratio_hifrac_mhi.pdf'
  plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
  plt.clf()
  
  fig8 = plt.figure(8, figsize=(6, 5))
  
  xlbl        = lbl_mhi
  ylbl        = lbl_hifrac
  zlbl        = r'$\log(R_{\rm{iso,HI}}/R_{\rm{iso25,r}})$'
  
  xpar    = mhi[subsample]
  ypar    = hifrac[subsample]
  zpar    = iso_size_ratio[subsample]
  scat_col_simple_plot2(fig8, 1, 1, 1, 
                        xpar[radius_hi_iso[subsample] > 45], 
                        ypar[radius_hi_iso[subsample] > 45], 
                        zpar[radius_hi_iso[subsample] > 45], 
                        [r'$>3$ beams', xlbl, ylbl, zlbl], 
                        ['rainbow_r', 'o', 'rainbow_r', 25, -0.2, 0.9], True, True)

  scat_col_simple_plot2(fig8, 1, 1, 1, 
                        xpar[radius_hi_iso[subsample] < 45], 
                        ypar[radius_hi_iso[subsample] < 45], 
                        zpar[radius_hi_iso[subsample] < 45], 
                        [r'$<3$ beams', xlbl, ylbl, zlbl], 
                        ['rainbow_r', 'o', 'none', 25, -0.2, 0.9], True, False)
  
  #sfr_ms   = 0.656 * mstar - 6.816
  plot_name = phase1_dir + 'PLOTS/PAPER3/hifrac_mhi.pdf'
  plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
  plt.clf()
  
  fig9 = plt.figure(9, figsize=(6, 5))
  
  zlbl        = r'$\log(j_{\rm{HI}}/[\rm{km/s}])$'
  xlbl        = lbl_hifrac
  ylbl        = r'$\log(R_{\rm{iso,HI}}/R_{\rm{iso25,r}})$'
  
  xpar    = hifrac[subsample]
  ypar    = iso_size_ratio[subsample]
  zpar    = jhi[subsample]
  scat_col_simple_plot2(fig9, 1, 1, 1, 
                        xpar[radius_hi_iso[subsample] > 45], 
                        ypar[radius_hi_iso[subsample] > 45], 
                        zpar[radius_hi_iso[subsample] > 45], 
                        [r'$>3$ beams', xlbl, ylbl, zlbl], 
                        ['rainbow_r', 'o', 'rainbow_r', 25, 2, 4.5], True, True)

  scat_col_simple_plot2(fig9, 1, 1, 1, 
                        xpar[radius_hi_iso[subsample] < 45], 
                        ypar[radius_hi_iso[subsample] < 45], 
                        zpar[radius_hi_iso[subsample] < 45], 
                        [r'$<3$ beams', xlbl, ylbl, zlbl], 
                        ['rainbow_r', 'o', 'none', 25, 2, 4.5], True, False)
  
  bins                   = np.arange(-1.5, 1.75, 0.25)
  par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
  bin_width              = np.abs(bin_edges[1] - bin_edges[0])
  xbins                  = bin_edges[:-1] + bin_width/2.
  
  p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
  p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
  
  scatter_error_plot(fig9, 1, 1, 1, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                        'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], True)
  
  #sfr_ms   = 0.656 * mstar - 6.816
  plot_name = phase1_dir + 'PLOTS/PAPER3/sratio_hifrac_jhi.pdf'
  plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
  plt.clf()
  
  
  nuv_size_ratio = np.log10(radius_nuv_iso / radius_r_iso23)
  subsample   = ((radius_hi_iso > 30) & np.isfinite(mstar) & 
                     np.isfinite(nuv_size_ratio)) # & (wallaby_field == 'NGC4636')
  
  fig10 = plt.figure(10, figsize=(6, 5))
  
  xlbl        = r'$\log(R_{\rm{iso,NUV}}/R_{\rm{iso23.5,r}})$'
  ylbl        = lbl_hifrac
  zlbl        = lbl_mhi
  
  xpar    = nuv_size_ratio[subsample]
  ypar    = hifrac[subsample]
  zpar    = mhi[subsample]
  scat_col_simple_plot2(fig10, 1, 1, 1, 
                        xpar[radius_hi_iso[subsample] > 45], 
                        ypar[radius_hi_iso[subsample] > 45], 
                        zpar[radius_hi_iso[subsample] > 45], 
                        [r'$>3$ beams', xlbl, ylbl, zlbl], 
                        ['rainbow_r', 'o', 'rainbow_r', 25, 7, 11], True, True)

  scat_col_simple_plot2(fig10, 1, 1, 1, 
                        xpar[radius_hi_iso[subsample] < 45], 
                        ypar[radius_hi_iso[subsample] < 45], 
                        zpar[radius_hi_iso[subsample] < 45], 
                        [r'$<3$ beams', xlbl, ylbl, zlbl], 
                        ['rainbow_r', 'o', 'none', 25, 7, 11], True, False)
  
  bins                   = np.arange(-0.2, 0.6, 0.1)
  par_mean, bin_edges, _ = binned_statistic(xpar[radius_hi_iso[subsample] > 45], ypar[radius_hi_iso[subsample] > 45], np.nanmedian, bins)
  bin_width              = np.abs(bin_edges[1] - bin_edges[0])
  xbins                  = bin_edges[:-1] + bin_width/2.
  
  p20, _, _ = binned_statistic(xpar[radius_hi_iso[subsample] > 45], ypar[radius_hi_iso[subsample] > 45], lambda y: np.percentile(y, 20), bins)
  p80, _, _ = binned_statistic(xpar[radius_hi_iso[subsample] > 45], ypar[radius_hi_iso[subsample] > 45], lambda y: np.percentile(y, 80), bins)
  
  scatter_error_plot(fig10, 1, 1, 1, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                        'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], True)
  
  par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
  bin_width              = np.abs(bin_edges[1] - bin_edges[0])
  xbins                  = bin_edges[:-1] + bin_width/2.
  
  p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
  p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
  
  scatter_error_plot(fig10, 1, 1, 1, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                        'Medians', xlbl, ylbl, ['grey', 's', 'none', 7.5], False)
  
  ax1 = fig10.add_subplot(1, 1, 1, facecolor = 'w')
  ax1.set_xlim(-0.7,0.6)
  ax1.set_ylim(-4,1.5)
  
  #sfr_ms   = 0.656 * mstar - 6.816
  plot_name = phase1_dir + 'PLOTS/PAPER3/sratio_nuv_hifrac_mhi.pdf'
  plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
  plt.clf()
  
  #nuv_size_ratio = np.log10(radius_nuv_iso / radius_r_iso23)
  #subsample   = ((radius_hi_iso > 30) & np.isfinite(mu_star) & 
                      #np.isfinite(nuv_size_ratio))

  #fig10 = plt.figure(10, figsize=(6, 5))

  #zlbl        = lbl_mustar
  #xlbl        = lbl_hifrac
  #ylbl        = r'$\log(R_{\rm{iso,NUV}}/R_{\rm{iso25,r}})$'

  #xpar    = hifrac[subsample]
  #ypar    = iso_size_ratio[subsample]
  #zpar    = mu_star[subsample]
  #scat_col_simple_plot2(fig10, 1, 1, 1, 
                        #xpar[radius_hi_iso[subsample] > 45], 
                        #ypar[radius_hi_iso[subsample] > 45], 
                        #zpar[radius_hi_iso[subsample] > 45], 
                        #[r'$>3$ beams', xlbl, ylbl, zlbl], 
                        #['rainbow_r', 'o', 'rainbow_r', 25, 5.75, 9], True, True)

  #scat_col_simple_plot2(fig10, 1, 1, 1, 
                        #xpar[radius_hi_iso[subsample] < 45], 
                        #ypar[radius_hi_iso[subsample] < 45], 
                        #zpar[radius_hi_iso[subsample] < 45], 
                        #[r'$<3$ beams', xlbl, ylbl, zlbl], 
                        #['rainbow_r', 'o', 'none', 25, 5.75, 9], True, False)

  #bins                   = np.arange(-1.5, 1.75, 0.5)
  #par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
  #bin_width              = np.abs(bin_edges[1] - bin_edges[0])
  #xbins                  = bin_edges[:-1] + bin_width/2.

  #p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
  #p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)

  #scatter_error_plot(fig10, 1, 1, 1, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                        #'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], True)
  
  ##sfr_ms   = 0.656 * mstar - 6.816
  #plot_name = phase1_dir + 'PLOTS/PAPER3/sratio_hifrac_mustar.pdf'
  #plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
  #plt.clf()
  
  
  
# ========== Sample Selection =========== #
if do_sample_select:
  fig1 = plt.figure(1, figsize=(8, 5))
  xpar    = [mstar, mstar, mstar, mstar, mstar, mstar]
  ypar    = [hifrac, ssfr, hifrac, ssfr, hifrac, ssfr]
  colour  = ['peru', 'darkblue']
  
  xlbl    = [lbl_mstar, lbl_mstar, lbl_mstar, lbl_mstar, lbl_mstar, lbl_mstar]
  ylbl    = [lbl_hifrac, lbl_ssfr, lbl_hifrac, lbl_ssfr, lbl_hifrac, lbl_ssfr]
  
  row, column = 2, 2
  
  for i in range(4):
    if ylbl[i] == lbl_hifrac:
      scatter_outlier_plot(fig1, row, column, i+1, xgass_mstar, xgass_hifrac, 
                         'xGASS', xlbl[i], ylbl[i], ['darkgrey', '.', 'darkgrey', 3], False)
    if ylbl[i] == lbl_ssfr:
      scatter_outlier_plot(fig1, row, column, i+1, xgass_mstar, xgass_ssfr, 
                         'xGASS', xlbl[i], ylbl[i], ['darkgrey', '.', 'darkgrey', 3], False)
      
    #if ylbl[i] == lbl_hifrac:
      #scatter_outlier_plot(fig1, row, column, i+1, 
                           #xgass_mstar[xgass_obs_lims], xgass_hifrac[xgass_obs_lims], 
                         #'xGASS', xlbl[i], ylbl[i], ['black', '.', 'black', 3], False)
    #if ylbl[i] == lbl_ssfr:
      #scatter_outlier_plot(fig1, row, column, i+1, 
                           #xgass_mstar[xgass_obs_lims], xgass_ssfr[xgass_obs_lims], 
                         #'xGASS', xlbl[i], ylbl[i], ['black', '.', 'black', 3], False)
    
    finite_pars  = (np.isfinite(mstar) & np.isfinite(hifrac) & 
                    np.isfinite(nuvr) & np.isfinite(ssfr) &
                    np.isfinite(radius_hi_iso) & np.isfinite(radius_hi_eff) & 
                    np.isfinite(radius_r_iso25) & np.isfinite(radius_r_50) & (mstar > 6.5))
    
    #finite_pars  = (np.isfinite(mstar) & np.isfinite(hifrac) & np.isfinite(mu_hi) & 
                    #np.isfinite(nuvr) & np.isfinite(ssfr) &
                    #np.isfinite(radius_hi_iso_kpc) & np.isfinite(radius_hi_eff) & 
                    #np.isfinite(radius_r_iso25_kpc) & np.isfinite(radius_r_50) & (mstar > 6.5))
    
    #subsample  = ((radius_hi_iso > 20) & np.isfinite(mhi) & np.isfinite(mstar) &
                  #np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  #np.isfinite(surfden_hi_iso) & np.isfinite(mu_hi) & 
                  #np.isfinite(radius_r_iso25_kpc) & 
                  #np.isfinite(radius_hi_iso_kpc) & (mstar > 6.5))
    
    #np.isnan(ssfr)
    
    #if i < 2:
      #subsample  = finite_pars
      ##subsample  = (np.isfinite(mstar) & np.isfinite(hifrac) & np.isfinite(ssfr))
    #if i > 1 and i < 4:
    if i < 2:
      subsample  = (radius_hi_iso > 20) & finite_pars
    if i > 1:
      subsample  = (radius_hi_eff > 20) & finite_pars
    if i == 1:
      plot_legend = True
    else:
      plot_legend = False
    
    #print(len(xpar[i][subsample]), 
          #len(xpar[i][subsample & (wallaby_field == 'Hydra')]), 
          #len(xpar[i][subsample & (wallaby_field == 'NGC4636')]))
    
    print(len(xpar[i][subsample & (mstar < 9)]), 
          len(xpar[i][subsample & (mstar < 9) & (wallaby_field == 'Hydra')]), 
          len(xpar[i][subsample & (mstar < 9) & (wallaby_field == 'NGC4636')]))
    
    print(len(xpar[i][subsample & (mstar > 9)]), 
          len(xpar[i][subsample & (mstar > 9) & (wallaby_field == 'Hydra')]), 
          len(xpar[i][subsample & (mstar > 9) & (wallaby_field == 'NGC4636')]))
    
    if ylbl[i] == lbl_hifrac:
      scatter_outlier_plot(fig1, row, column, i+1, 
                          xpar[i][subsample & (wallaby_field == 'Hydra')], 
                          ypar[i][subsample & (wallaby_field == 'Hydra')], 
                          'Hydra', xlbl[i], ylbl[i], ['darkblue', 'o', 'darkblue', 15], plot_legend)
      
      scatter_outlier_plot(fig1, row, column, i+1, 
                          xpar[i][subsample & (wallaby_field == 'NGC4636')], 
                          ypar[i][subsample & (wallaby_field == 'NGC4636')], 
                          'NGC4636', xlbl[i], ylbl[i], ['peru', 'o', 'peru', 15], plot_legend)
    
    if ylbl[i] == lbl_ssfr:
      scatter_outlier_plot(fig1, row, column, i+1, 
                          xpar[i][subsample & (wallaby_field == 'Hydra') & (~sfr_uplim)], 
                          ypar[i][subsample & (wallaby_field == 'Hydra') & (~sfr_uplim)], 
                          'Hydra', xlbl[i], ylbl[i], ['darkblue', 'o', 'darkblue', 15], plot_legend)
      
      scatter_outlier_plot(fig1, row, column, i+1, 
                          xpar[i][subsample & (wallaby_field == 'NGC4636') & (~sfr_uplim)], 
                          ypar[i][subsample & (wallaby_field == 'NGC4636') & (~sfr_uplim)], 
                          'NGC4636', xlbl[i], ylbl[i], ['peru', 'o', 'peru', 15], plot_legend)
      
      scatter_outlier_plot(fig1, row, column, i+1, 
                          xpar[i][subsample & (wallaby_field == 'Hydra') & (sfr_uplim)], 
                          ypar[i][subsample & (wallaby_field == 'Hydra') & (sfr_uplim)], 
                          'Hydra', xlbl[i], ylbl[i], ['darkblue', r'$\downarrow$', 'darkblue', 40], False)
      
      scatter_outlier_plot(fig1, row, column, i+1, 
                          xpar[i][subsample & (wallaby_field == 'NGC4636') & (sfr_uplim)], 
                          ypar[i][subsample & (wallaby_field == 'NGC4636') & (sfr_uplim)], 
                          'NGC4636', xlbl[i], ylbl[i], ['peru', r'$\downarrow$', 'peru', 40], False)
      
      
      
      
    
    if ylbl[i] == lbl_hifrac:
      x, y, z = xgass_mstar_table[0], xgass_mstar_table[1], xgass_mstar_table[2]
      scatter_error_plot(fig1, row, column, i+1, x, y, [z, z],
                         'Medians', xlbl[i], ylbl[i], ['black', 'd', 'black', 7.5], False)
    
    ax1 = fig1.add_subplot(row, column, i+1, facecolor = 'w')
    
    if ylbl[i] == lbl_ssfr:
      x = np.arange(7, 12, 0.1)
      y = -0.344 * (x - 9.) - 9.822
      z = 0.088 * (x - 9.) + 0.188
      #scatter_error_plot(fig1, row, column, i+1, x, y, [z, z],
                         #'Medians', xlbl[i], ylbl[i], ['black', 's', 'black', 7.5], False)
      ax1.plot(x, y, color='black', linewidth=2.5, linestyle = '--', zorder=1)
      #ax1.plot(x, y+z, color='magenta', linewidth=2, linestyle = ':', zorder=1)
      #ax1.plot(x, y-z, color='magenta', linewidth=2, linestyle = ':', zorder=1)
      ax1.fill_between(x[x>8.9], y[x>8.9]-z[x>8.9], y[x>8.9]+z[x>8.9], alpha=0.3, 
                        edgecolor='none', zorder=3, facecolor='black')
    
    
    #if i == 0:
      #ax1.text(0.05, 0.05, 'All (%i)' % len(xpar[i][subsample]), 
               #transform=ax1.transAxes, fontsize=10)
      #ax1.text(0.75, 0.75, '%i (H)\n%i (N)' % (len(xpar[i][subsample & (wallaby_field == 'Hydra')]), 
          #len(xpar[i][subsample & (wallaby_field == 'NGC4636')])), 
               #transform=ax1.transAxes, fontsize=10)
    if i == 0:
      ax1.text(0.05, 0.05, r"$R_{\rm{iso,HI}}>20''$ (%i)" % len(xpar[i][subsample]), 
               transform=ax1.transAxes, fontsize=11)
      ax1.text(0.65, 0.82, '%i (Hydra)\n%i (NGC4636)' % (len(xpar[i][subsample & (wallaby_field == 'Hydra')]), 
          len(xpar[i][subsample & (wallaby_field == 'NGC4636')])), 
               transform=ax1.transAxes, fontsize=11)
    if i == 2:
      ax1.text(0.05, 0.05, r"$R_{\rm{50,HI}}>20''$ (%i)" % len(xpar[i][subsample]), 
               transform=ax1.transAxes, fontsize=11)
      ax1.text(0.65, 0.82, '%i (Hydra)\n%i (NGC4636)' % (len(xpar[i][subsample & (wallaby_field == 'Hydra')]), 
          len(xpar[i][subsample & (wallaby_field == 'NGC4636')])), 
               transform=ax1.transAxes, fontsize=11)
    
    if ylbl[i] == lbl_hifrac:
      ax1.set_xlim(7,11.5)
      ax1.set_ylim(-2, 1.5)
    if ylbl[i] == lbl_ssfr:
      ax1.set_xlim(7, 11.5)
      ax1.set_ylim(-12.5, -8.5)
  
  plt.subplots_adjust(wspace=0.25, hspace=0.17)
  
  plot_name = phase1_dir + 'PLOTS/PAPER3/sample_selection20.pdf'
  plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
  plt.clf()
  
  subsample = (radius_hi_iso > 30) & finite_pars
  
  iso_sratio = radius_hi_iso[subsample] / radius_r_iso25[subsample]
  
  print(np.round([np.nanmedian(iso_sratio), np.nanmean(iso_sratio), 
        np.nanmean(iso_sratio) + np.nanstd(iso_sratio),
        np.nanmean(iso_sratio) - np.nanstd(iso_sratio)], 1))
  
  subsample = (radius_hi_eff > 30) & finite_pars
  
  iso_sratio = radius_hi_iso[subsample] / radius_r_iso25[subsample]
  
  print(np.round([np.nanmedian(iso_sratio), np.nanmean(iso_sratio), 
        np.nanmean(iso_sratio) + np.nanstd(iso_sratio),
        np.nanmean(iso_sratio) - np.nanstd(iso_sratio)], 1))
  
  subsample = (radius_hi_eff > 30) & finite_pars
  
  eff_sratio = radius_hi_eff[subsample] / radius_r_50[subsample]
  
  print(np.round([np.nanmedian(eff_sratio), np.nanmean(eff_sratio), 
        np.nanmean(eff_sratio) + np.nanstd(eff_sratio),
        np.nanmean(eff_sratio) - np.nanstd(eff_sratio)], 1))


# ================================= #
# ======== Scatter Plots ========== #
# ================================= #
if do_scatter_plots:
  do_iso_eff            = False
  do_iso_mstar          = False
  do_iso_mstar_sratio   = False
  do_iso_mstar_hifrac   = False
  do_iso_muhi_ssfr      = False
  do_iso_muhi_mstar     = False
  do_iso_muhi_nuvr      = False
  do_iso_muhi_mustar    = False
  do_iso_muhi_cindex    = False
  do_iso_sratio_ssfr    = False
  do_iso_sratio_mstar   = False
  do_iso_sratio_nuvr    = False
  do_iso_sratio_mustar  = True
  do_iso_sratio_cindex  = False
  do_delta_size         = False
  do_opt_cindex         = False
  do_muhi_radius_ratio  = False
  
  def linear_func(x, m, b):
      return m * x + b
    
  fig4 = plt.figure(5, figsize=(7, 7))#, facecolor = '#007D7D')
  
  
  if do_iso_eff:
    #xlbl         = lbl_mstar
    #ylbl         = r'Radius [kpc]'
    
    xpar    = [mstar, mstar, mhi, mhi]
    ypar    = [radius_r_iso25_kpc, radius_r_50_kpc, radius_hi_iso_kpc, radius_hi_eff_kpc]
    limpar  = no_galex
    colour  = ['peru', 'sandybrown', 'darkblue', 'royalblue']
    poplbl  = ['Cluster', 'Infall', 'Field']
    
    xlbl    = [lbl_mstar, lbl_mstar, lbl_mhi, lbl_mhi]
    
    ylbl    = [r'$\log(R_{\rm{iso25,r}}/[\rm{kpc}])$', r'$\log(R_{\rm{50,r}}/[\rm{kpc}])$', 
               r'$\log(R_{\rm{iso,HI}}/[\rm{kpc}])$', r'$\log(R_{\rm{50,HI}}/[\rm{kpc}])$']
    
    row, column = 2, 2
    
    print('m, c, rval, pval, stderr')
    
    #yfit = []
    #dfit = []
    
    fig5 = plt.figure(55, figsize=(9, 6))
    
    #subsample  = ((radius_hi_iso > 30) & np.isfinite(mhi) & np.isfinite(mstar) & 
                  #np.isfinite(xpar[i]) & np.isfinite(ypar[i]) & (mstar > 6.5))
      
    subsample  = ((radius_hi_eff > 20) & np.isfinite(mhi) & np.isfinite(mstar) & 
                  np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  np.isfinite(radius_r_iso25_kpc) & np.isfinite(radius_r_50_kpc) & 
                  np.isfinite(radius_hi_iso_kpc) & np.isfinite(radius_hi_eff_kpc) & (mstar > 6.5))
    
    #subsample  = ((radius_hi_iso > 20) & np.isfinite(mhi) & np.isfinite(mstar) &
                  #np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  #np.isfinite(surfden_hi_iso) & np.isfinite(mu_hi) & 
                  #np.isfinite(radius_r_iso25_kpc) & 
                  #np.isfinite(radius_hi_iso_kpc) & (mstar > 6.5))
    
    # =============== #
    data_array = [xpar[1][subsample], np.log10(ypar[1][subsample])]
    print(len(data_array[0]))
    offset_r_eff = size_mass_relation_plot(fig5, 2, 2, 1, data_array, 'Effective', 
                                           xlbl[0], r'$\log(R_{\rm{r}}/\rm{kpc})$', 
                                           [colour[1], colour[1], 'none'], 'o')
    
    # =============== #
    data_array = [xpar[0][subsample], np.log10(ypar[0][subsample])]
    offset_r_iso = size_mass_relation_plot(fig5, 2, 2, 1, data_array, 'Isophotal', 
                                           xlbl[0], r'$\log(R_{\rm{r}}/\rm{kpc})$', 
                                           [colour[0], colour[0], colour[0]], 'o')
    
    ax1 = fig5.add_subplot(2, 2, 1, facecolor = 'w')
      
    ax1.set_xlim(6.8,11.2)
    ax1.set_ylim(-0.2,1.9)
    
    # =============== #
    data_array = [xpar[3][subsample], np.log10(ypar[3][subsample])]
    offset_h_eff = size_mass_relation_plot(fig5, 2, 2, 2, data_array, 'Effective', 
                                           xlbl[2], r'$\log(R_{\rm{HI}}/\rm{kpc})$', 
                                           [colour[3], colour[3], 'none'], 'o')
    
    # =============== #
    data_array = [xpar[2][subsample], np.log10(ypar[2][subsample])]
    offset_h_iso = size_mass_relation_plot(fig5, 2, 2, 2, data_array, 'Isodensity', 
                                           xlbl[2], r'$\log(R_{\rm{HI}}/\rm{kpc})$', 
                                           [colour[2], colour[2], colour[2]], 'o')
    
    
    ax1 = fig5.add_subplot(2, 2, 2, facecolor = 'w')
    
    a_16, b_16 = 0.506, -3.293
    #a_16, b_16 = 0.51, -3.29
    d_16       = 0.06
    xfit       = np.arange(7, 11.25, 0.25)
    yfit_16    = a_16 * xfit + b_16 - np.log10(2.)
    print(a_16, b_16 - np.log10(2.))
    ax1.plot(xfit, yfit_16, color='magenta', linewidth=1, linestyle = '--', zorder=1)
    #ax1.fill_between(xfit, yfit_16-d_16, yfit_16+d_16, alpha=0.5, 
                     #edgecolor='none', zorder=3, facecolor='grey')
    
    ax1.set_xlim(6.8,11.2)
    ax1.set_ylim(-0.2,1.9)
    
    #plt.show()
    #plot_name = phase1_dir + 'PLOTS/PAPER3/size_mass_histar_iso.pdf'
    #fig5.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    #plt.clf()
    
    #fig6 = plt.figure(6, figsize=(9, 3.5))
    
    offset  = [offset_r_iso, offset_r_eff, offset_h_iso, offset_h_eff]
    
    #xpar    = radius_hi_iso_kpc
    #ypar    = offset
    
    xlbl    = r'$\log(R_{\rm{iso,r}}/\rm{kpc})$'
    ylbl    = 'Offset [dex]'
    
    data_array = [np.log10(ypar[0][subsample]), offset[1]]
    scatter_outlier_plot(fig5, 2, 2, 3, 
                         data_array[0], data_array[1], 
                         'Effective', xlbl, ylbl, ['sandybrown', 'o', 'none', 25], False)
    
    data_array = [np.log10(ypar[0][subsample]), offset[0]]
    scatter_outlier_plot(fig5, 2, 2, 3, 
                         data_array[0], data_array[1], 
                         'Isodensity', xlbl, ylbl, ['peru', 'o', 'peru', 25], False)
    
    ax1 = fig5.add_subplot(2, 2, 3, facecolor = 'w')
    ax1.axhline(0, color = 'grey', linestyle = '--', zorder = -1)
    #ax1.set_xlim(-0.5, 0.5)
    ax1.set_ylim(-0.5, 0.5)
    
    xlbl    = r'$\log(R_{\rm{iso,HI}}/\rm{kpc})$'
    
    data_array = [np.log10(ypar[2][subsample]), offset[3]]
    scatter_outlier_plot(fig5, 2, 2, 4, 
                         data_array[0], data_array[1], 
                         'Effective', xlbl, ylbl, ['royalblue', 'o', 'none', 25], False)
    
    data_array = [np.log10(ypar[2][subsample]), offset[2]]
    scatter_outlier_plot(fig5, 2, 2, 4, 
                         data_array[0], data_array[1], 
                         'Isodensity', xlbl, ylbl, ['darkblue', 'o', 'darkblue', 25], False)
    
    ax1 = fig5.add_subplot(2, 2, 4, facecolor = 'w')
    ax1.axhline(0, color = 'grey', linestyle = '--', zorder = -1)
    #ax1.set_xlim(-0.5, 0.5)
    ax1.set_ylim(-0.5, 0.5)
    
    plot_name = phase1_dir + 'PLOTS/PAPER3/size_mass_residual_eff20.pdf'
    plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()
    
  
  if do_iso_mstar:
    #xlbl         = lbl_mstar
    #ylbl         = r'Radius [kpc]'
    
    xpar    = [mstar, mstar]
    ypar    = [radius_r_iso25_kpc, radius_hi_iso_kpc]
    limpar  = no_galex
    colour  = ['peru', 'darkblue']
    poplbl  = ['Cluster', 'Infall', 'Field']
    
    xlbl    = [lbl_mstar, lbl_mstar]
    
    ylbl    = [r'$\log(R_{\rm{iso25,r}}/[\rm{kpc}])$', 
               r'$\log(R_{\rm{iso,HI}}/[\rm{kpc}])$']
    
    print('m, c, rval, pval, stderr')
    
    #yfit = []
    #dfit = []
    
    fig5 = plt.figure(55, figsize=(6, 6))
    
    #subsample  = ((radius_hi_iso > 30) & np.isfinite(mhi) & np.isfinite(mstar) & 
                  #np.isfinite(xpar[i]) & np.isfinite(ypar[i]) & (mstar > 6.5))
      
    subsample  = ((radius_hi_iso > 30) & np.isfinite(mhi) & np.isfinite(mstar) & 
                  np.isfinite(radius_r_iso25_kpc) & 
                  np.isfinite(radius_hi_iso_kpc) & (mstar > 6.5))
    
    
    # =============== #
    #data_array = [xpar[1][subsample], np.log10(ypar[1][subsample])]
    #print(len(data_array[0]))
    #size_mass_relation_plot(fig5, 1, 2, 1, data_array, 'Effective', 
                            #xlbl[0], r'$\log(R_{\rm{r}}/[\rm{kpc}])$', 
                            #[colour[1], colour[1], 'none'], 'o')
    
    # =============== #
    data_array = [xpar[0][subsample], np.log10(ypar[0][subsample])]
    size_mass_relation_plot(fig5, 1, 1, 1, data_array, r'$r$-band', 
                            xlbl[0], r'$\log(R_{\rm{iso}}/[\rm{kpc}])$', 
                            [colour[0], colour[0], colour[0]], 'o')
    
    # =============== #
    #data_array = [xpar[3][subsample], np.log10(ypar[3][subsample])]
    #size_mass_relation_plot(fig5, 1, 2, 2, data_array, 'Effective', 
                            #xlbl[2], r'$\log(R_{\rm{HI}}/[\rm{kpc}])$', 
                            #[colour[3], colour[3], 'none'], 'o')
    
    # =============== #
    data_array = [xpar[1][subsample], np.log10(ypar[1][subsample])]
    size_mass_relation_plot(fig5, 1, 1, 1, data_array, 'HI', 
                            xlbl[1], r'$\log(R_{\rm{iso}}/[\rm{kpc}])$', 
                            [colour[1], colour[1], 'none'], 'o')
    
    #plt.show()
    plot_name = phase1_dir + 'PLOTS/PAPER3/size_mstar.pdf'
    fig5.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()
  
  if do_iso_mstar_sratio:
    print('m, c, rval, pval, stderr')
    
    fig5 = plt.figure(55, figsize=(9, 4))
    
    #subsample  = ((radius_hi_iso > 30) & np.isfinite(mhi) & np.isfinite(mstar) & 
                  #np.isfinite(xpar[i]) & np.isfinite(ypar[i]) & (mstar > 6.5))
      
    subsample  = ((radius_hi_iso > 30) & np.isfinite(mhi) & np.isfinite(mstar) & 
                  np.isfinite(radius_r_iso25_kpc) & 
                  np.isfinite(radius_hi_iso_kpc) & (mstar > 6.5))
    
    iso_size_ratio = np.log10(radius_hi_iso / radius_r_iso25)
    
    xlbl    = lbl_mstar
    ylbl    = r'$\log(R_{\rm{iso}}/[\rm{kpc}])$'
    zlbl    = r'$\log(R_{\rm{iso,HI}}/R_{\rm{iso25,r}})$'
    
    xpar    = mstar[subsample]
    ypar    = np.log10(radius_r_iso25_kpc[subsample])
    zpar    = iso_size_ratio[subsample]
    
    scat_col_simple_plot2(fig5, 1, 2, 1, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'$r$-band', xlbl, ylbl, zlbl], 
                          ['rainbow_r', 'o', 'rainbow_r', 25, -0.2, 0.8], True, True)
    
    xpar    = mstar[subsample]
    ypar    = np.log10(radius_hi_iso_kpc[subsample])
    zpar    = iso_size_ratio[subsample]

    scat_col_simple_plot2(fig5, 1, 2, 2, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'HI', xlbl, ylbl, zlbl],
                          ['rainbow_r', 'o', 'rainbow_r', 25, -0.2, 0.8], True, True)
    
    #plt.show()
    plot_name = phase1_dir + 'PLOTS/PAPER3/size_mstar_sratio.pdf'
    fig5.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()
    
  if do_iso_mstar_hifrac:
    print('m, c, rval, pval, stderr')
    
    fig5 = plt.figure(56, figsize=(9, 4))
    
    #subsample  = ((radius_hi_iso > 30) & np.isfinite(mhi) & np.isfinite(mstar) & 
                  #np.isfinite(xpar[i]) & np.isfinite(ypar[i]) & (mstar > 6.5))
      
    subsample  = ((radius_hi_iso > 30) & np.isfinite(mhi) & np.isfinite(mstar) & 
                  np.isfinite(radius_r_iso25_kpc) & 
                  np.isfinite(radius_hi_iso_kpc) & (mstar > 6.5))
    
    iso_size_ratio = np.log10(radius_hi_iso / radius_r_iso25)
    
    xlbl    = lbl_mstar
    ylbl    = r'$\log(R_{\rm{iso}}/[\rm{kpc}])$'
    zlbl    = lbl_hifrac
    
    xpar    = mstar[subsample]
    ypar    = np.log10(radius_r_iso25_kpc[subsample])
    zpar    = hifrac[subsample]
    
    scat_col_simple_plot2(fig5, 1, 2, 1, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'$r$-band', xlbl, ylbl, zlbl], 
                          ['rainbow_r', 'o', 'rainbow_r', 25, -1.5, 1.6], True, True)
    
    xpar    = mstar[subsample]
    ypar    = np.log10(radius_hi_iso_kpc[subsample])
    zpar    = hifrac[subsample]

    scat_col_simple_plot2(fig5, 1, 2, 2, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'HI', xlbl, ylbl, zlbl],
                          ['rainbow_r', 'o', 'rainbow_r', 25, -1.5, 1.6], True, True)
    
    #plt.show()
    plot_name = phase1_dir + 'PLOTS/PAPER3/size_mstar_hifrac.pdf'
    fig5.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()
  
  if do_iso_muhi_ssfr:
    fig5 = plt.figure(56, figsize=(13, 4))
      
    subsample  = ((radius_hi_iso > 20) & np.isfinite(mhi) & np.isfinite(mstar) &
                  np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  np.isfinite(surfden_hi_iso) & np.isfinite(mu_hi) & 
                  np.isfinite(radius_r_iso25_kpc) & 
                  np.isfinite(radius_hi_iso_kpc) & (mstar > 6.5))
    
    iso_size_ratio = np.log10(radius_hi_iso / radius_r_iso25)
    
    xlbl    = lbl_ssfr
    ylbl    = r'$\log(\mu_{\rm{HI}}/[\rm{M_{\odot}\,pc^{-2}}])$'
    zlbl    = 'Beams' #lbl_mstar 
    
    xpar    = ssfr[subsample]
    ypar    = np.log10(surfden_hi_iso[subsample])
    zpar    = radius_hi_iso[subsample]/30.
    
    print(len(xpar))
    
    cmap = plt.get_cmap('rainbow', 5)
    
    scat_col_simple_plot3(fig5, 1, 3, 1, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'ISO', xlbl, ylbl, zlbl], 
                          [cmap, 'o', cmap, 20, 0, 5], True, False)
    
    bins                   = np.arange(-10.5, -8.75, 0.25)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 1, 3, 1, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    ax1 = fig5.add_subplot(1, 3, 1, facecolor = 'w')
    ax1.set_xlim(-11.25, -8.5)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    
    subsample  = ((radius_hi_eff > 20) & np.isfinite(mhi) & np.isfinite(mstar) &
                  np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  np.isfinite(surfden_hi_iso) & np.isfinite(mu_hi) & 
                  np.isfinite(radius_r_iso25_kpc) & 
                  np.isfinite(radius_hi_iso_kpc) & (mstar > 6.5))
    
    iso_size_ratio = np.log10(radius_hi_iso / radius_r_iso25)
    
    xlbl    = lbl_ssfr
    ylbl    = r'$\log(\mu_{\rm{HI}}/[\rm{M_{\odot}\,pc^{-2}}])$'
    zlbl    = 'Beams' #
    
    xpar    = ssfr[subsample]
    ypar    = np.log10(surfden_hi_iso[subsample])
    zpar    = radius_hi_iso[subsample]/30.
    
    print(len(xpar))
    
    scat_col_simple_plot3(fig5, 1, 3, 2, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'ISO', xlbl, ylbl, zlbl], 
                          [cmap, 'o', cmap, 20, 0, 5], True, False)
    
    #bins                   = np.arange(-11, -8.75, 0.25)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 1, 3, 2, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    ax1 = fig5.add_subplot(1, 3, 2, facecolor = 'w')
    ax1.set_xlim(-11.25, -8.5)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    
    subsample  = ((radius_hi_eff > 20) & np.isfinite(mhi) & np.isfinite(mstar) &
                  np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  np.isfinite(surfden_hi_eff) & np.isfinite(mu_hi) & 
                  np.isfinite(radius_r_iso25_kpc) & 
                  np.isfinite(radius_hi_iso_kpc) & (mstar > 6.5))
    
    
    xpar    = ssfr[subsample]
    ypar    = np.log10(surfden_hi_eff[subsample])
    zpar    = radius_hi_eff[subsample]/30.
    
    print(len(xpar))
    
    scat_col_simple_plot3(fig5, 1, 3, 3, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'EFF', xlbl, ylbl, zlbl],
                          [cmap, 'o', cmap, 20, 0, 5], True, True)
    
    #bins                   = np.arange(-11, -8.75, 0.25)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 1, 3, 3, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    ax1 = fig5.add_subplot(1, 3, 3, facecolor = 'w')
    ax1.set_xlim(-11.25, -8.5)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    plot_name = phase1_dir + 'PLOTS/PAPER3/muhi_ssfr_log20.pdf'
    fig5.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()
  
  if do_iso_muhi_mstar:
    fig5 = plt.figure(56, figsize=(13, 4))
      
    subsample  = ((radius_hi_iso > 20) & np.isfinite(mhi) & np.isfinite(mstar) & 
                  np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  np.isfinite(surfden_hi_iso) & np.isfinite(mu_hi) & 
                  np.isfinite(radius_r_iso25_kpc) & 
                  np.isfinite(radius_hi_iso_kpc) & (mstar > 6.5))
    
    iso_size_ratio = np.log10(radius_hi_iso / radius_r_iso25)
    
    xlbl    = lbl_mstar
    ylbl    = r'$\log(\mu_{\rm{HI}}/[\rm{M_{\odot}\,pc^{-2}}])$'
    zlbl    = 'Beams'
    
    xpar    = mstar[subsample]
    #xpar    = delta_sfms[subsample]
    ypar    = np.log10(surfden_hi_iso[subsample])
    zpar    = radius_hi_iso[subsample]/30.
    
    print(len(xpar))
    
    cmap = plt.get_cmap('rainbow', 5)
    
    scat_col_simple_plot3(fig5, 1, 3, 1, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'ISO', xlbl, ylbl, zlbl], 
                          [cmap, 'o', cmap, 20, 0, 5], True, False)
    
    bins                   = np.arange(7.5, 11.25, 0.5)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 1, 3, 1, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    ax1 = fig5.add_subplot(1, 3, 1, facecolor = 'w')
    #ax1.set_xlim(-11.25, -8.5)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    
    subsample  = ((radius_hi_eff > 20) & np.isfinite(mhi) & np.isfinite(mstar) &
                  np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  np.isfinite(surfden_hi_iso) & np.isfinite(mu_hi) & 
                  np.isfinite(radius_r_iso25_kpc) & 
                  np.isfinite(radius_hi_iso_kpc) & (mstar > 6.5))
    
    iso_size_ratio = np.log10(radius_hi_iso / radius_r_iso25)
    
    xlbl    = lbl_mstar
    #xlbl    = r'$\Delta$\,SFMS [dex]'
    ylbl    = r'$\log(\mu_{\rm{HI}}/[\rm{M_{\odot}\,pc^{-2}}])$'
    zlbl    = 'Beams'
    
    xpar    = mstar[subsample]
    ypar    = np.log10(surfden_hi_iso[subsample])
    zpar    = radius_hi_iso[subsample]/30.
    
    print(len(xpar))
    
    scat_col_simple_plot3(fig5, 1, 3, 2, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'ISO', xlbl, ylbl, zlbl], 
                          [cmap, 'o', cmap, 20, 0, 5], True, False)
    
    bins                   = np.arange(7.5, 11.25, 0.5)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 1, 3, 2, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    ax1 = fig5.add_subplot(1, 3, 2, facecolor = 'w')
    #ax1.set_xlim(-11.25, -8.5)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    subsample  = ((radius_hi_eff > 20) & np.isfinite(mhi) & np.isfinite(mstar) &
                  np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  np.isfinite(surfden_hi_eff) & np.isfinite(mu_hi) & 
                  np.isfinite(radius_r_iso25_kpc) & 
                  np.isfinite(radius_hi_iso_kpc) & (mstar > 6.5))
    
    #eff_size_ratio = np.log10(radius_hi_eff / radius_r_50)
    
    xpar    = mstar[subsample]
    ypar    = np.log10(surfden_hi_eff[subsample])
    zpar    = radius_hi_eff[subsample]/30.
    
    print(len(xpar))
    
    scat_col_simple_plot3(fig5, 1, 3, 3, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'EFF', xlbl, ylbl, zlbl],
                          [cmap, 'o', cmap, 20, 0, 5], True, True)
    
    bins                   = np.arange(7.5, 11.25, 0.5)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 1, 3, 3, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    ax1 = fig5.add_subplot(1, 3, 3, facecolor = 'w')
    #ax1.set_xlim(-11.25, -8.5)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    #plt.show()
    plot_name = phase1_dir + 'PLOTS/PAPER3/muhi_mstar_log20.pdf'
    fig5.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()
  
  if do_iso_muhi_nuvr:
    fig5 = plt.figure(56, figsize=(13, 4))
    
    subsample  = ((radius_hi_iso > 20) & np.isfinite(mhi) & np.isfinite(mstar) &
                  np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  np.isfinite(surfden_hi_iso) & np.isfinite(mu_hi) & 
                  np.isfinite(radius_r_iso25_kpc) & 
                  np.isfinite(radius_hi_iso_kpc) & (mstar > 6.5))
    
    iso_size_ratio = np.log10(radius_hi_iso / radius_r_iso25)
    
    xlbl    = lbl_nuvr
    ylbl    = r'$\log(\mu_{\rm{HI}}/[\rm{M_{\odot}\,pc^{-2}}])$'
    zlbl    = 'Beams' #lbl_mstar
    
    xpar    = nuvr[subsample]
    ypar    = np.log10(surfden_hi_iso[subsample])
    zpar    = radius_hi_iso[subsample]/30.
    
    print(len(xpar))
    
    cmap = plt.get_cmap('rainbow', 5)
    
    scat_col_simple_plot3(fig5, 1, 3, 1, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'ISO', xlbl, ylbl, zlbl], 
                          [cmap, 'o', cmap, 20, 0, 5], True, False)
    
    bins                   = np.arange(0.7, 3.5, 0.5)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 1, 3, 1, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    ax1 = fig5.add_subplot(1, 3, 1, facecolor = 'w')
    ax1.set_xlim(-0.1, 6)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    subsample  = ((radius_hi_eff > 20) & np.isfinite(mhi) & np.isfinite(mstar) &
                  np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  np.isfinite(surfden_hi_iso) & np.isfinite(mu_hi) & 
                  np.isfinite(radius_r_iso25_kpc) & 
                  np.isfinite(radius_hi_iso_kpc) & (mstar > 6.5))
    
    iso_size_ratio = np.log10(radius_hi_iso / radius_r_iso25)
    
    xlbl    = lbl_nuvr
    ylbl    = r'$\log(\mu_{\rm{HI}}/[\rm{M_{\odot}\,pc^{-2}}])$'
    zlbl    = 'Beams' #lbl_mstar 
    
    xpar    = nuvr[subsample]
    ypar    = np.log10(surfden_hi_iso[subsample])
    zpar    = radius_hi_iso[subsample]/30. #mstar[subsample]
    
    print(len(xpar))
    
    scat_col_simple_plot3(fig5, 1, 3, 2, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'ISO', xlbl, ylbl, zlbl], 
                          [cmap, 'o', cmap, 20, 0, 5], True, False)
    
    #bins                   = np.arange(0.5, 4.5, 0.5)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 1, 3, 2, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    ax1 = fig5.add_subplot(1, 3, 2, facecolor = 'w')
    ax1.set_xlim(-0.1, 6)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    #zlbl    = r'$\log(R_{\rm{eff,HI}}/R_{\rm{eff,r}})$'
    
    subsample  = ((radius_hi_eff > 20) & np.isfinite(mhi) & np.isfinite(mstar) &
                  np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  np.isfinite(surfden_hi_eff) & np.isfinite(mu_hi) & 
                  np.isfinite(radius_r_iso25_kpc) & 
                  np.isfinite(radius_hi_iso_kpc) & (mstar > 6.5))
    
    #eff_size_ratio = np.log10(radius_hi_eff / radius_r_50)
    
    xpar    = nuvr[subsample]
    #xpar    = delta_sfms[subsample]
    ypar    = np.log10(surfden_hi_eff[subsample])
    #ypar    = mu_mean[subsample]
    zpar    = radius_hi_eff[subsample]/30. #mstar[subsample]
    
    print(len(xpar))
    
    scat_col_simple_plot3(fig5, 1, 3, 3, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'EFF', xlbl, ylbl, zlbl],
                          [cmap, 'o', cmap, 20, 0, 5], True, True)
    
    #bins                   = np.arange(0, 4, 0.25)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 1, 3, 3, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    ax1 = fig5.add_subplot(1, 3, 3, facecolor = 'w')
    ax1.set_xlim(-0.1, 6)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    plot_name = phase1_dir + 'PLOTS/PAPER3/muhi_nuvr_log20.pdf'
    fig5.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()
    
  if do_iso_muhi_mustar:
    fig5 = plt.figure(56, figsize=(13, 4))
      
    subsample  = ((radius_hi_iso > 20) & np.isfinite(mhi) & np.isfinite(mu_star) & 
                  np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  np.isfinite(surfden_hi_iso) & np.isfinite(mu_hi) & 
                  np.isfinite(radius_r_iso25_kpc) & 
                  np.isfinite(radius_hi_iso_kpc) & (mstar > 6.5))
    
    iso_size_ratio = np.log10(radius_hi_iso / radius_r_iso25)
    
    xlbl    = lbl_mustar
    #xlbl    = r'$\Delta$\,SFMS [dex]'
    ylbl    = r'$\log(\mu_{\rm{HI}}/[\rm{M_{\odot}\,pc^{-2}}])$'
    zlbl    = 'Beams'
    
    xpar    = mu_star[subsample]
    #xpar    = delta_sfms[subsample]
    ypar    = np.log10(surfden_hi_iso[subsample])
    zpar    = radius_hi_iso[subsample]/30.
    
    print(len(xpar))
    
    cmap = plt.get_cmap('rainbow', 5)
    
    scat_col_simple_plot3(fig5, 1, 3, 1, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'ISO', xlbl, ylbl, zlbl], 
                          [cmap, 'o', cmap, 20, 0, 5], True, False)
    
    bins                   = np.arange(6.2, 9, 0.5)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 1, 3, 1, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    ax1 = fig5.add_subplot(1, 3, 1, facecolor = 'w')
    ax1.set_xlim(5.5, 9)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    subsample  = ((radius_hi_eff > 20) & np.isfinite(mhi) & np.isfinite(mu_star) & 
                  np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  np.isfinite(surfden_hi_iso) & np.isfinite(mu_hi) & 
                  np.isfinite(radius_r_iso25_kpc) & 
                  np.isfinite(radius_hi_iso_kpc) & (mstar > 6.5))
    
    iso_size_ratio = np.log10(radius_hi_iso / radius_r_iso25)
    
    xlbl    = lbl_mustar
    #xlbl    = r'$\Delta$\,SFMS [dex]'
    ylbl    = r'$\log(\mu_{\rm{HI}}/[\rm{M_{\odot}\,pc^{-2}}])$'
    zlbl    = 'Beams'
    
    xpar    = mu_star[subsample]
    ypar    = np.log10(surfden_hi_iso[subsample])
    zpar    = radius_hi_iso[subsample]/30.
    
    print(len(xpar))
    
    scat_col_simple_plot3(fig5, 1, 3, 2, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'ISO', xlbl, ylbl, zlbl], 
                          [cmap, 'o', cmap, 20, 0, 5], True, False)
    
    #bins                   = np.arange(5.5, 9, 0.5)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 1, 3, 2, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    ax1 = fig5.add_subplot(1, 3, 2, facecolor = 'w')
    ax1.set_xlim(5.5, 9)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    subsample  = ((radius_hi_eff > 20) & np.isfinite(mhi) & np.isfinite(mu_star) & 
                  np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  np.isfinite(surfden_hi_eff) & np.isfinite(mu_hi) & 
                  np.isfinite(radius_r_iso25_kpc) & 
                  np.isfinite(radius_hi_iso_kpc) & (mstar > 6.5))
    
    xpar    = mu_star[subsample]
    ypar    = np.log10(surfden_hi_eff[subsample])
    zpar    = radius_hi_eff[subsample]/30.
    
    print(len(xpar))
    
    scat_col_simple_plot3(fig5, 1, 3, 3, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'EFF', xlbl, ylbl, zlbl],
                          [cmap, 'o', cmap, 20, 0, 5], True, True)
    
    #bins                   = np.arange(0, 4, 0.25)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 1, 3, 3, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    ax1 = fig5.add_subplot(1, 3, 3, facecolor = 'w')
    ax1.set_xlim(5.5, 9)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    #plt.show()
    plot_name = phase1_dir + 'PLOTS/PAPER3/muhi_mustar_log20.pdf'
    fig5.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()
  
  if do_iso_muhi_cindex:
    fig5 = plt.figure(56, figsize=(13, 4))
      
    subsample  = ((radius_hi_iso > 20) & np.isfinite(mhi) & np.isfinite(cindex) & 
                  np.isfinite(surfden_hi_iso) & np.isfinite(mu_hi) & 
                  np.isfinite(radius_r_iso25_kpc) & 
                  np.isfinite(radius_hi_iso_kpc) & (mstar > 6.5))
    
    iso_size_ratio = np.log10(radius_hi_iso / radius_r_iso25)
    
    xlbl    = r'$R_{\rm{90,r}}/R_{\rm{50,r}}$'
    ylbl    = r'$\log(\mu_{\rm{HI}}/[\rm{M_{\odot}\,pc^{-2}}])$'
    zlbl    = 'Beams'
    
    xpar    = cindex[subsample]
    #xpar    = delta_sfms[subsample]
    ypar    = np.log10(surfden_hi_iso[subsample])
    zpar    = radius_hi_iso[subsample]/30.
    
    print(len(xpar))
    
    scat_col_simple_plot3(fig5, 1, 3, 1, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'ISO', xlbl, ylbl, zlbl], 
                          ['rainbow', 'o', 'rainbow', 20, 0, 5], True, False)
    
    bins                   = np.arange(1.5, 4, 0.25)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 1, 3, 1, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], True)
    
    ax1 = fig5.add_subplot(1, 3, 1, facecolor = 'w')
    ax1.set_xlim(1.5, 4)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    subsample  = ((radius_hi_eff > 20) & np.isfinite(mhi) & np.isfinite(cindex) & 
                  np.isfinite(surfden_hi_iso) & np.isfinite(mu_hi) & 
                  np.isfinite(radius_r_iso25_kpc) & 
                  np.isfinite(radius_hi_iso_kpc) & (mstar > 6.5))
    
    iso_size_ratio = np.log10(radius_hi_iso / radius_r_iso25)
    
    xlbl    = r'$R_{\rm{90,r}}/R_{\rm{50,r}}$'
    ylbl    = r'$\log(\mu_{\rm{HI}}/[\rm{M_{\odot}\,pc^{-2}}])$'
    zlbl    = 'Beams'
    
    xpar    = cindex[subsample]
    ypar    = np.log10(surfden_hi_iso[subsample])
    zpar    = radius_hi_iso[subsample]/30.
    
    print(len(xpar))
    
    scat_col_simple_plot3(fig5, 1, 3, 2, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'ISO', xlbl, ylbl, zlbl], 
                          ['rainbow', 'o', 'rainbow', 20, 0, 5], True, False)
    
    bins                   = np.arange(1.5, 4, 0.25)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 1, 3, 2, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], True)
    
    ax1 = fig5.add_subplot(1, 3, 2, facecolor = 'w')
    ax1.set_xlim(1.5, 4)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    subsample  = ((radius_hi_eff > 20) & np.isfinite(mhi) & np.isfinite(cindex) & 
                  np.isfinite(surfden_hi_eff) & np.isfinite(mu_hi) & 
                  np.isfinite(radius_r_iso25_kpc) & 
                  np.isfinite(radius_hi_iso_kpc) & (mstar > 6.5))
    
    xpar    = cindex[subsample]
    ypar    = np.log10(surfden_hi_eff[subsample])
    zpar    = radius_hi_eff[subsample]/30.
    
    print(len(xpar))
    
    scat_col_simple_plot3(fig5, 1, 3, 3, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'EFF', xlbl, ylbl, zlbl],
                          ['rainbow', 'o', 'rainbow', 20, 0, 5], True, True)
    
    #bins                   = np.arange(0, 4, 0.25)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 1, 3, 3, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], True)
    
    ax1 = fig5.add_subplot(1, 3, 3, facecolor = 'w')
    ax1.set_xlim(1.5, 4)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    #plt.show()
    plot_name = phase1_dir + 'PLOTS/PAPER3/muhi_cindex_log20.pdf'
    fig5.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()
  
  if do_iso_sratio_ssfr:
    fig5 = plt.figure(56, figsize=(13, 4))
    
    iso_size_ratio = np.log10(radius_hi_iso / radius_r_iso25)
      
    subsample  = ((radius_hi_iso > 20) & np.isfinite(mhi) & np.isfinite(mstar) &
                  np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  np.isfinite(iso_size_ratio) & np.isfinite(radius_r_iso25) & 
                  np.isfinite(radius_hi_iso) & (mstar > 6.5))
    
    xlbl    = lbl_ssfr
    ylbl    = r'$\log(R_{\rm{iso,HI}}/R_{\rm{iso25,r}})$'
    zlbl    = 'Beams'
    
    xpar    = ssfr[subsample]
    #xpar    = delta_sfms[subsample]
    ypar    = iso_size_ratio[subsample]
    zpar    = radius_hi_iso[subsample]/30.
    
    print(len(xpar))
    
    cmap = plt.get_cmap('rainbow', 5)
    
    scat_col_simple_plot3(fig5, 1, 3, 1, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'ISO', xlbl, ylbl, zlbl], 
                          [cmap, 'o', cmap, 20, 0, 5], True, False)
    
    bins                   = np.arange(-10.5, -8.75, 0.25)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 1, 3, 1, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    ax1 = fig5.add_subplot(1, 3, 1, facecolor = 'w')
    ax1.set_xlim(-11.25, -8.5)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    iso_size_ratio = np.log10(radius_hi_iso / radius_r_iso25)
    
    subsample  = ((radius_hi_eff > 20) & np.isfinite(mhi) & np.isfinite(mstar) &
                  np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  np.isfinite(iso_size_ratio) & np.isfinite(radius_r_iso25) & 
                  np.isfinite(radius_hi_iso) & (mstar > 6.5))
    
    xlbl    = lbl_ssfr
    ylbl    = r'$\log(R_{\rm{iso,HI}}/R_{\rm{iso25,r}})$'
    zlbl    = 'Beams'
    
    xpar    = ssfr[subsample]
    #xpar    = delta_sfms[subsample]
    ypar    = iso_size_ratio[subsample]
    zpar    = radius_hi_iso[subsample]/30.
    
    print(len(xpar))
    
    scat_col_simple_plot3(fig5, 1, 3, 2, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'ISO', xlbl, ylbl, zlbl], 
                          [cmap, 'o', cmap, 20, 0, 5], True, False)
    
    #bins                   = np.arange(-11, -8.75, 0.25)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 1, 3, 2, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    ax1 = fig5.add_subplot(1, 3, 2, facecolor = 'w')
    ax1.set_xlim(-11.25, -8.5)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    #zlbl    = r'$\log(R_{\rm{eff,HI}}/R_{\rm{eff,r}})$'
    
    eff_size_ratio = np.log10(radius_hi_iso / radius_r_50)
    
    subsample  = ((radius_hi_eff > 20) & np.isfinite(mhi) & np.isfinite(mstar) &
                  np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  np.isfinite(eff_size_ratio) & np.isfinite(radius_r_50) & 
                  np.isfinite(radius_hi_eff) & (mstar > 6.5))
    
    
    xlbl    = lbl_ssfr
    ylbl    = r'$\log(R_{\rm{eff,HI}}/R_{\rm{eff,r}})$'
    zlbl    = 'Beams'
    
    xpar    = ssfr[subsample]
    ypar    = eff_size_ratio[subsample]
    zpar    = radius_hi_eff[subsample]/30.
    
    print(len(xpar))
    
    scat_col_simple_plot3(fig5, 1, 3, 3, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'EFF', xlbl, ylbl, zlbl],
                          [cmap, 'o', cmap, 20, 0, 5], True, True)
    
    #bins                   = np.arange(-11, -8.75, 0.25)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 1, 3, 3, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    ax1 = fig5.add_subplot(1, 3, 3, facecolor = 'w')
    ax1.set_xlim(-11.25, -8.5)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    plot_name = phase1_dir + 'PLOTS/PAPER3/sratio_ssfr_log20_iso.pdf'
    fig5.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()
  
  if do_iso_sratio_mstar:
    fig5 = plt.figure(56, figsize=(13, 4))
    
    subsample  = ((radius_hi_iso > 20) & np.isfinite(mhi) & np.isfinite(mstar) &
                  np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  np.isfinite(radius_r_iso25) & 
                  np.isfinite(radius_hi_iso) & (mstar > 6.5))
    
    iso_size_ratio = np.log10(radius_hi_iso / radius_r_iso25)
    
    xlbl    = lbl_mstar
    ylbl    = r'$\log(R_{\rm{iso,HI}}/R_{\rm{iso25,r}})$'
    zlbl    = 'Beams'
    
    xpar    = mstar[subsample]
    #xpar    = delta_sfms[subsample]
    ypar    = iso_size_ratio[subsample]
    zpar    = radius_hi_iso[subsample]/30.
    
    print(len(xpar))
    
    cmap = plt.get_cmap('rainbow', 5)
    
    scat_col_simple_plot3(fig5, 1, 3, 1, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'ISO', xlbl, ylbl, zlbl], 
                          [cmap, 'o', cmap, 20, 0, 5], True, False)
    
    bins                   = np.arange(7.5, 11.25, 0.5)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 1, 3, 1, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    ax1 = fig5.add_subplot(1, 3, 1, facecolor = 'w')
    #ax1.set_xlim(-11.25, -8.5)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    
    subsample  = ((radius_hi_eff > 20) & np.isfinite(mhi) & np.isfinite(mstar) &
                  np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  np.isfinite(radius_r_iso25) & 
                  np.isfinite(radius_hi_iso) & (mstar > 6.5))
    
    iso_size_ratio = np.log10(radius_hi_iso / radius_r_iso25)
    
    xlbl    = lbl_mstar
    ylbl    = r'$\log(R_{\rm{iso,HI}}/R_{\rm{iso25,r}})$'
    zlbl    = 'Beams'
    
    xpar    = mstar[subsample]
    #xpar    = delta_sfms[subsample]
    ypar    = iso_size_ratio[subsample]
    zpar    = radius_hi_iso[subsample]/30.
    
    print(len(xpar))
    
    scat_col_simple_plot3(fig5, 1, 3, 2, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'ISO', xlbl, ylbl, zlbl], 
                          [cmap, 'o', cmap, 20, 0, 5], True, False)
    
    bins                   = np.arange(7.5, 11.25, 0.5)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 1, 3, 2, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    ax1 = fig5.add_subplot(1, 3, 2, facecolor = 'w')
    #ax1.set_xlim(-11.25, -8.5)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    #zlbl    = r'$\log(R_{\rm{eff,HI}}/R_{\rm{eff,r}})$'
    
    subsample  = ((radius_hi_eff > 20) & np.isfinite(mhi) & np.isfinite(mstar) &
                  np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  np.isfinite(radius_r_50) & 
                  np.isfinite(radius_hi_eff) & (mstar > 6.5))
    
    eff_size_ratio = np.log10(radius_hi_iso / radius_r_50)
    
    xlbl    = lbl_mstar
    ylbl    = r'$\log(R_{\rm{eff,HI}}/R_{\rm{eff,r}})$'
    zlbl    = 'Beams'
    
    xpar    = mstar[subsample]
    #xpar    = delta_sfms[subsample]
    ypar    = eff_size_ratio[subsample]
    #ypar    = mu_mean[subsample]
    zpar    = radius_hi_eff[subsample]/30.
    
    print(len(xpar))
    
    scat_col_simple_plot3(fig5, 1, 3, 3, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'EFF', xlbl, ylbl, zlbl],
                          [cmap, 'o', cmap, 20, 0, 5], True, True)
    
    bins                   = np.arange(7.5, 11.25, 0.5)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 1, 3, 3, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    ax1 = fig5.add_subplot(1, 3, 3, facecolor = 'w')
    #ax1.set_xlim(-11.25, -8.5)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    #plt.show()
    plot_name = phase1_dir + 'PLOTS/PAPER3/sratio_mstar_log20_iso.pdf'
    fig5.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()
  
  if do_iso_sratio_nuvr:
    fig5 = plt.figure(56, figsize=(13, 4))
    
    subsample  = ((radius_hi_iso > 20) & np.isfinite(mhi) & np.isfinite(mstar) &
                  np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  np.isfinite(radius_r_iso25) & 
                  np.isfinite(radius_hi_iso) & (mstar > 6.5))
    
    iso_size_ratio = np.log10(radius_hi_iso / radius_r_iso25)
    
    xlbl    = lbl_nuvr
    ylbl    = r'$\log(R_{\rm{iso,HI}}/R_{\rm{iso25,r}})$'
    zlbl    = 'Beams' #lbl_mstar #r'$\log(R_{\rm{iso,HI}}/R_{\rm{iso25,r}})$'
    
    xpar    = nuvr[subsample]
    #xpar    = delta_sfms[subsample]
    ypar    = iso_size_ratio[subsample]
    zpar    = radius_hi_iso[subsample]/30. #mstar[subsample]
    
    print(len(xpar))
    
    cmap = plt.get_cmap('rainbow', 5)
    
    scat_col_simple_plot3(fig5, 1, 3, 1, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'ISO', xlbl, ylbl, zlbl], 
                          [cmap, 'o', cmap, 20, 0, 5], True, False)
    
    bins                   = np.arange(0.5, 3.5, 0.5)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 1, 3, 1, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    ax1 = fig5.add_subplot(1, 3, 1, facecolor = 'w')
    ax1.set_xlim(-0.1, 6)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    subsample  = ((radius_hi_eff > 20) & np.isfinite(mhi) & np.isfinite(mstar) &
                  np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  np.isfinite(radius_r_iso25) & 
                  np.isfinite(radius_hi_iso) & (mstar > 6.5))
    
    iso_size_ratio = np.log10(radius_hi_iso / radius_r_iso25)
    
    xlbl    = lbl_nuvr
    ylbl    = r'$\log(R_{\rm{iso,HI}}/R_{\rm{iso25,r}})$'
    zlbl    = 'Beams' #lbl_mstar #r'$\log(R_{\rm{iso,HI}}/R_{\rm{iso25,r}})$'
    
    xpar    = nuvr[subsample]
    #xpar    = delta_sfms[subsample]
    ypar    = iso_size_ratio[subsample]
    zpar    = radius_hi_iso[subsample]/30. #mstar[subsample]
    
    print(len(xpar))
    
    scat_col_simple_plot3(fig5, 1, 3, 2, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'ISO', xlbl, ylbl, zlbl], 
                          [cmap, 'o', cmap, 20, 0, 5], True, False)
    
    #bins                   = np.arange(0.5, 4.5, 0.5)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 1, 3, 2, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    ax1 = fig5.add_subplot(1, 3, 2, facecolor = 'w')
    ax1.set_xlim(-0.1, 6)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    #zlbl    = r'$\log(R_{\rm{eff,HI}}/R_{\rm{eff,r}})$'
    
    subsample  = ((radius_hi_eff > 20) & np.isfinite(mhi) & np.isfinite(mstar) &
                  np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  np.isfinite(radius_r_50) & 
                  np.isfinite(radius_hi_eff) & (mstar > 6.5))
    
    eff_size_ratio = np.log10(radius_hi_iso / radius_r_50)
    
    xlbl    = lbl_nuvr
    ylbl    = r'$\log(R_{\rm{eff,HI}}/R_{\rm{eff,r}})$'
    zlbl    = 'Beams' #lbl_mstar #r'$\log(R_{\rm{iso,HI}}/R_{\rm{iso25,r}})$'
    
    xpar    = nuvr[subsample]
    #xpar    = delta_sfms[subsample]
    ypar    = eff_size_ratio[subsample]
    #ypar    = mu_mean[subsample]
    zpar    = radius_hi_eff[subsample]/30. #mstar[subsample]
    
    print(len(xpar))
    
    scat_col_simple_plot3(fig5, 1, 3, 3, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'EFF', xlbl, ylbl, zlbl],
                          [cmap, 'o', cmap, 20, 0, 5], True, True)
    
    #bins                   = np.arange(0, 4, 0.25)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 1, 3, 3, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    ax1 = fig5.add_subplot(1, 3, 3, facecolor = 'w')
    ax1.set_xlim(-0.1, 6)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    #plt.show()
    plot_name = phase1_dir + 'PLOTS/PAPER3/sratio_nuvr_log20_iso.pdf'
    #plot_name = phase1_dir + 'PLOTS/PAPER3/size_dsfms_muhi.pdf'
    fig5.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()
  
  if do_iso_sratio_mustar:
    fig5 = plt.figure(56, figsize=(13, 4))
    
    subsample  = ((radius_hi_iso > 20) & np.isfinite(mhi) & np.isfinite(mu_star) & np.isfinite(mstar) &
                  np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  np.isfinite(radius_r_iso25) & 
                  np.isfinite(radius_hi_iso) & (mstar > 6.5))
    
    iso_size_ratio = np.log10(radius_hi_iso / radius_r_iso25)
    
    xlbl    = lbl_mustar
    ylbl    = r'$\log(R_{\rm{iso,HI}}/R_{\rm{iso25,r}})$'
    zlbl    = 'Beams'
    
    xpar    = mu_star[subsample]
    ypar    = iso_size_ratio[subsample]
    zpar    = radius_hi_iso[subsample]/30.
    
    print(len(xpar))
    
    cmap = plt.get_cmap('rainbow', 5)
    
    scat_col_simple_plot3(fig5, 1, 3, 1, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'ISO', xlbl, ylbl, zlbl], 
                          [cmap, 'o', cmap, 20, 0, 5], True, False)
    
    bins                   = np.arange(6.2, 9, 0.5)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 1, 3, 1, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    ax1 = fig5.add_subplot(1, 3, 1, facecolor = 'w')
    ax1.set_xlim(5.5, 9)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    subsample  = ((radius_hi_eff > 20) & np.isfinite(mhi) & np.isfinite(mu_star) & np.isfinite(mstar) &
                  np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  np.isfinite(radius_r_iso25) & 
                  np.isfinite(radius_hi_iso) & (mstar > 6.5))
    
    iso_size_ratio = np.log10(radius_hi_iso / radius_r_iso25)
    
    xlbl    = lbl_mustar
    ylbl    = r'$\log(R_{\rm{iso,HI}}/R_{\rm{iso25,r}})$'
    zlbl    = 'Beams'
    
    xpar    = mu_star[subsample]
    #xpar    = delta_sfms[subsample]
    ypar    = iso_size_ratio[subsample]
    zpar    = radius_hi_iso[subsample]/30.
    
    print(len(xpar))
    
    scat_col_simple_plot3(fig5, 1, 3, 2, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'ISO', xlbl, ylbl, zlbl], 
                          [cmap, 'o', cmap, 20, 0, 5], True, False)
    
    #bins                   = np.arange(6, 9, 0.5)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 1, 3, 2, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    ax1 = fig5.add_subplot(1, 3, 2, facecolor = 'w')
    ax1.set_xlim(5.5, 9)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    #zlbl    = r'$\log(R_{\rm{eff,HI}}/R_{\rm{eff,r}})$'
    
    subsample  = ((radius_hi_eff > 20) & np.isfinite(mhi) & np.isfinite(mu_star) & np.isfinite(mstar) &
                  np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  np.isfinite(radius_r_50) & 
                  np.isfinite(radius_hi_eff) & (mstar > 6.5))
    
    eff_size_ratio = np.log10(radius_hi_iso / radius_r_50)
    
    xlbl    = lbl_mustar
    ylbl    = r'$\log(R_{\rm{eff,HI}}/R_{\rm{eff,r}})$'
    zlbl    = 'Beams'
    
    xpar    = mu_star[subsample]
    #xpar    = delta_sfms[subsample]
    ypar    = eff_size_ratio[subsample]
    #ypar    = mu_mean[subsample]
    zpar    = radius_hi_eff[subsample]/30.
    
    print(len(xpar))
    
    scat_col_simple_plot3(fig5, 1, 3, 3, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'EFF', xlbl, ylbl, zlbl],
                          [cmap, 'o', cmap, 20, 0, 5], True, True)
    
    #bins                   = np.arange(0, 4, 0.25)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 1, 3, 3, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    ax1 = fig5.add_subplot(1, 3, 3, facecolor = 'w')
    ax1.set_xlim(5.5, 9)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    #plt.show()
    plot_name = phase1_dir + 'PLOTS/PAPER3/sratio_mustar_log20_iso.pdf'
    #plot_name = phase1_dir + 'PLOTS/PAPER3/size_dsfms_muhi.pdf'
    fig5.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()
  
  if do_iso_sratio_cindex:
    fig5 = plt.figure(56, figsize=(13, 4))
    
    subsample  = ((radius_hi_iso > 20) & np.isfinite(mhi) & np.isfinite(cindex) & 
                  np.isfinite(radius_r_iso25) & 
                  np.isfinite(radius_hi_iso) & (mstar > 6.5))
    
    iso_size_ratio = np.log10(radius_hi_iso / radius_r_iso25)
    
    xlbl    = r'$R_{\rm{90,r}}/R_{\rm{50,r}}$'
    ylbl    = r'$\log(R_{\rm{iso,HI}}/R_{\rm{iso25,r}})$'
    zlbl    = 'Beams'
    
    xpar    = cindex[subsample]
    ypar    = iso_size_ratio[subsample]
    zpar    = radius_hi_iso[subsample]/30.
    
    print(len(xpar))
    
    scat_col_simple_plot3(fig5, 1, 3, 1, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'ISO', xlbl, ylbl, zlbl], 
                          ['rainbow', 'o', 'rainbow', 20, 0, 5], True, False)
    
    bins                   = np.arange(1.5, 4, 0.25)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 1, 3, 1, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], True)
    
    ax1 = fig5.add_subplot(1, 3, 1, facecolor = 'w')
    ax1.set_xlim(1.5, 4)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    subsample  = ((radius_hi_eff > 20) & np.isfinite(mhi) & np.isfinite(cindex) & 
                  np.isfinite(radius_r_iso25) & 
                  np.isfinite(radius_hi_iso) & (mstar > 6.5))
    
    iso_size_ratio = np.log10(radius_hi_iso / radius_r_iso25)
    
    xlbl    = r'$R_{\rm{90,r}}/R_{\rm{50,r}}$'
    ylbl    = r'$\log(R_{\rm{iso,HI}}/R_{\rm{iso25,r}})$'
    zlbl    = 'Beams'
    
    xpar    = cindex[subsample]
    #xpar    = delta_sfms[subsample]
    ypar    = iso_size_ratio[subsample]
    zpar    = radius_hi_iso[subsample]/30.
    
    print(len(xpar))
    
    scat_col_simple_plot3(fig5, 1, 3, 2, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'ISO', xlbl, ylbl, zlbl], 
                          ['rainbow', 'o', 'rainbow', 20, 0, 5], True, False)
    
    bins                   = np.arange(1.5, 4, 0.25)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 1, 3, 2, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], True)
    
    ax1 = fig5.add_subplot(1, 3, 2, facecolor = 'w')
    ax1.set_xlim(1.5, 4)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    #zlbl    = r'$\log(R_{\rm{eff,HI}}/R_{\rm{eff,r}})$'
    
    subsample  = ((radius_hi_eff > 20) & np.isfinite(mhi) & np.isfinite(cindex) & 
                  np.isfinite(radius_r_50) & 
                  np.isfinite(radius_hi_eff) & (mstar > 6.5))
    
    eff_size_ratio = np.log10(radius_hi_eff / radius_r_50)
    
    xlbl    = r'$R_{\rm{90,r}}/R_{\rm{50,r}}$'
    ylbl    = r'$\log(R_{\rm{eff,HI}}/R_{\rm{eff,r}})$'
    zlbl    = 'Beams'
    
    xpar    = cindex[subsample]
    #xpar    = delta_sfms[subsample]
    ypar    = eff_size_ratio[subsample]
    #ypar    = mu_mean[subsample]
    zpar    = radius_hi_eff[subsample]/30.
    
    print(len(xpar))
    
    scat_col_simple_plot3(fig5, 1, 3, 3, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'EFF', xlbl, ylbl, zlbl],
                          ['rainbow', 'o', 'rainbow', 20, 0, 5], True, True)
    
    #bins                   = np.arange(0, 4, 0.25)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 1, 3, 3, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], True)
    
    ax1 = fig5.add_subplot(1, 3, 3, facecolor = 'w')
    ax1.set_xlim(1.5, 4)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    #plt.show()
    plot_name = phase1_dir + 'PLOTS/PAPER3/sratio_cindex_log20.pdf'
    #plot_name = phase1_dir + 'PLOTS/PAPER3/size_dsfms_muhi.pdf'
    fig5.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()
  
  if do_delta_size:
    fig4 = plt.figure(6, figsize=(6, 8))
    
    xpar    = [mstar, mhi]
    #xpar    = [mu_star, surfden_hi_eff]
    ypar    = [(radius_r_iso25 - radius_r_50) / radius_r_iso25, 
               (radius_hi_iso - radius_hi_eff) / radius_hi_iso]
    colour  = ['peru', 'darkblue']
    
    xlbl    = [lbl_mstar, lbl_mhi]
    #xlbl    = [lbl_mustar, lbl_muhi]
    ylbl    = [r'$(R_{\rm{iso,r}} - R_{\rm{eff,r}})/R_{\rm{iso,r}}$', 
               r'$(R_{\rm{iso,HI}} - R_{\rm{eff,HI}})/R_{\rm{iso,HI}}$']
    
    row, column = 2, 1
    
    for i in range(2):
      #if i == 0 or i == 2:
      subsample  = (radius_hi_eff > 15) & np.isfinite(xpar[i]) & np.isfinite(ypar[i]) & (ypar[i] > -1)
      gal_subsample = galaxies[subsample]
      iso_subsample = radius_hi_iso[subsample]
      eff_subsample = radius_hi_eff[subsample]
      #if i == 1 or i == 3:
        #subsample  = (radius_hi_eff > 30) & np.isfinite(xpar[i]) & np.isfinite(ypar[i]) & (ypar[i] < 10)
      #subsample  = (radius_hi_iso > 30) & np.isfinite(xpar[i]) & np.isfinite(ypar[i])
      #data_array = [xpar[i][subsample], np.log10(ypar[i][subsample])]
      data_array = [xpar[i][subsample], ypar[i][subsample]]
      #scat_mean_plot(fig4, row, column, i+1, data_array[0], data_array[1], False, False, 
                     #'WALLABY', xlbl[i], ylbl[i], colour[i],  'o', False)
      scatter_outlier_plot(fig4, row, column, i+1, 
                           data_array[0][iso_subsample > 45], data_array[1][iso_subsample > 45], 
                           r'$>3$ beams', xlbl[i], ylbl[i], [colour[i], 'o', colour[i], 25], True)
      scatter_outlier_plot(fig4, row, column, i+1, 
                           data_array[0][iso_subsample < 45], data_array[1][iso_subsample < 45], 
                           r'$<3$ beams', xlbl[i], ylbl[i], [colour[i], 'o', 'none', 25], True)
      #print(gal_subsample[data_array[1] < 0])
      #print(iso_subsample[data_array[1] < 0])
      #print(eff_subsample[data_array[1] < 0])
      
    plot_name = phase1_dir + 'PLOTS/PAPER3/delta_radius_mass3.pdf'
    plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()
    
  if do_opt_cindex:
    fig4 = plt.figure(6, figsize=(6, 3.5))
    
    xpar    = mstar
    ypar    = cindex
    
    xlbl    = lbl_mstar
    ylbl    = r'$R_{\rm{90,r}}/R_{\rm{50,r}}$'
    
    print(galaxies[(cindex < 3.5) & (cindex > 3)])
    
    subsample  = (radius_hi_iso > 15) & np.isfinite(xpar) & np.isfinite(ypar)
    data_array = [xpar[subsample], ypar[subsample]]
    scatter_outlier_plot(fig4, 1, 1, 1, 
                          data_array[0], data_array[1], 
                          'WALLABY', xlbl, ylbl, ['peru', 'o', 'peru', 25], True)
      
    plot_name = phase1_dir + 'PLOTS/PAPER3/rband_cindex.pdf'
    plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()

  if do_muhi_radius_ratio:
    fig4 = plt.figure(6, figsize=(6, 3))
    
    xpar    = radius_hi_iso / radius_hi_eff
    ypar    = surfden_hi_iso / surfden_hi_eff
    colour  = ['peru', 'darkblue']
    
    xlbl    = r'$R_{\rm{iso,HI}}/R_{\rm{eff,HI}}$'
    ylbl    = r'$\mu_{\rm{iso,HI}}/\mu_{\rm{eff,HI}}$'
    
    row, column = 1, 2
    
    subsample  = (radius_hi_iso > 30) & np.isfinite(xpar) & np.isfinite(ypar)
    scatter_outlier_plot(fig4, row, column, 1, 
                          xpar[subsample], ypar[subsample], 
                          r'ISO $>2$ beams', xlbl, ylbl, ['darkblue', 'o', 'darkblue', 25], True)
    
    ax1 = fig4.add_subplot(1, 2, 1, facecolor = 'w')
    ax1.set_xlim(0.5, 3)
    ax1.set_ylim(0.4, 2)
    
    subsample  = (radius_hi_eff > 30) & np.isfinite(xpar) & np.isfinite(ypar)
    scatter_outlier_plot(fig4, row, column, 2, 
                          xpar[subsample], ypar[subsample], 
                          r'EFF $>2$ beams', xlbl, ylbl, ['skyblue', 'o', 'skyblue', 25], True)
    
    ax1 = fig4.add_subplot(1, 2, 2, facecolor = 'w')
    ax1.set_xlim(0.5, 3)
    ax1.set_ylim(0.4, 2)
    
    plot_name = phase1_dir + 'PLOTS/PAPER3/muhi_radius_ratio.pdf'
    plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()


# ================================= #
# ===== SRATIO Profile Plots ====== #
# ================================= #
if do_sratio_profiles:
  do_mstar     = True
  #do_mhi       = False
  #do_mustar    = False
  
  do_log       = True
  
  if do_mstar:
    do_iso_iso25_mstar  = False
    do_eff_eff_mstar    = False
    do_iso_eff_mstar    = False
    do_eff_iso25_mstar  = False
    do_iso_iso25_radius = False
    do_eff_eff_radius   = False
    do_col_hifrac       = False
    do_mass_colratio    = False
    do_col_ssfr         = False
    do_col_nuv          = False
    do_col_dsfms        = False
    do_iso_ssfr         = False
    do_iso_nuvr         = True
    radius_limit        = 30
    if do_iso_iso25_mstar:
      input_dirs    = ['/Users/tflowers/WALLABY/Hydra_DR2/', '/Users/tflowers/WALLABY/NGC4636_DR1/']
      input_fnames  = ['all_normalised_hi.fits', 'all_normalised_rband25.fits']
      if do_log:
        output_fnames = ['MSTAR/LOG/iso_iso25_mstar_profiles_log_iso3.pdf', 'MSTAR/LOG/iso_iso25_mstar_log_iso3.pdf']
        #output_fnames = ['MSTAR/LOG/iso_iso25_mstar_profiles_log_env_iso.pdf', 
        #                 'MSTAR/LOG/iso_iso25_mstar_log_env_iso.pdf']
        xlbl          = lbl_mstar
        ylbl          = r'$\log(R_{\rm{iso,HI}}/R_{\rm{iso25,r}})$'
        data_to_plot  = [mstar, np.log10(radius_hi_iso/radius_r_iso25), radius_hi_iso, radius_limit, xlbl, ylbl,
                         pop_hi_array]
      else:
        output_fnames = ['MSTAR/iso_iso25_mstar_profiles.pdf', 'MSTAR/iso_iso25_mstar.pdf']
        xlbl          = lbl_mstar
        ylbl          = r'$R_{\rm{iso,HI}}/R_{\rm{iso25,r}}$'
        data_to_plot  = [mstar, radius_hi_iso/radius_r_iso25, radius_hi_iso, radius_limit, xlbl, ylbl]

      scatter_radial_profiles(input_dirs, input_fnames, output_fnames, data_to_plot)
    
    if do_eff_eff_mstar:
      input_dirs    = ['/Users/tflowers/WALLABY/Hydra_DR2/', '/Users/tflowers/WALLABY/NGC4636_DR1/']
      input_fnames  = ['all_normalised_hi.fits', 'all_normalised_rband25.fits']
      if do_log:
        output_fnames = ['MSTAR/LOG/eff_eff_mstar_profiles_log_iso3.pdf', 'MSTAR/LOG/eff_eff_mstar_log_iso3.pdf']
        #output_fnames = ['MSTAR/LOG/eff_eff_mstar_profiles_log_env_iso.pdf', 
        #                 'MSTAR/LOG/eff_eff_mstar_log_env_iso.pdf']
        xlbl          = lbl_mstar
        ylbl          = r'$\log(R_{\rm{eff,HI}}/R_{\rm{eff,r}})$'
        data_to_plot  = [mstar, np.log10(radius_hi_eff/radius_r_50), radius_hi_eff, radius_limit, xlbl, ylbl,
                         pop_hi_array]
      else:
        output_fnames = ['MSTAR/eff_eff_mstar_profiles.pdf', 'MSTAR/eff_eff_mstar.pdf']
        xlbl          = lbl_mstar
        ylbl          = r'$R_{\rm{eff,HI}}/R_{\rm{eff,r}}$'
        data_to_plot  = [mstar, radius_hi_eff/radius_r_50, radius_hi_eff, radius_limit, xlbl, ylbl]

      scatter_radial_profiles(input_dirs, input_fnames, output_fnames, data_to_plot)
    
    if do_iso_eff_mstar:
      input_dirs    = ['/Users/tflowers/WALLABY/Hydra_DR2/', '/Users/tflowers/WALLABY/NGC4636_DR1/']
      input_fnames  = ['all_normalised_hi.fits', 'all_normalised_rband23.fits']
      if do_log:
        output_fnames = ['MSTAR/LOG/iso_eff_mstar_profiles_log.pdf', 'MSTAR/LOG/iso_eff_mstar_log.pdf']
        xlbl          = lbl_mstar
        ylbl          = r'$\log(R_{\rm{iso,HI}}/R_{\rm{eff,r}})$'
        data_to_plot  = [mstar, np.log10(radius_hi_iso/radius_r_50), radius_hi_eff, radius_limit, xlbl, ylbl]
      else:
        output_fnames = ['MSTAR/iso_eff_mstar_profiles.pdf', 'MSTAR/iso_eff_mstar.pdf']
        xlbl          = lbl_mstar
        ylbl          = r'$\log(R_{\rm{iso,HI}}/R_{\rm{eff,r}})$'
        data_to_plot  = [mstar, radius_hi_iso/radius_r_50, radius_hi_eff, radius_limit, xlbl, ylbl]

      scatter_radial_profiles(input_dirs, input_fnames, output_fnames, data_to_plot)
      
    if do_eff_iso25_mstar:
      input_dirs    = ['/Users/tflowers/WALLABY/Hydra_DR2/', '/Users/tflowers/WALLABY/NGC4636_DR1/']
      input_fnames  = ['all_normalised_hi.fits', 'all_normalised_rband25.fits']
      if do_log:
        output_fnames = ['MSTAR/LOG/eff_iso25_mstar_profiles_log.pdf', 'MSTAR/LOG/eff_iso25_mstar_log.pdf']
        xlbl          = lbl_mstar
        ylbl          = r'$\log(R_{\rm{eff,HI}}/R_{\rm{iso25,r}})$'
        data_to_plot  = [mstar, np.log10(radius_hi_eff/radius_r_iso25), radius_hi_eff, radius_limit, xlbl, ylbl]
      else:
        output_fnames = ['MSTAR/eff_iso25_mstar_profiles.pdf', 'MSTAR/eff_iso25_mstar.pdf']
        xlbl          = lbl_mstar
        ylbl          = r'$R_{\rm{eff,HI}}/R_{\rm{iso25,r}}$'
        data_to_plot  = [mstar, radius_hi_eff/radius_r_iso25, radius_hi_eff, radius_limit, xlbl, ylbl]

      scatter_radial_profiles(input_dirs, input_fnames, output_fnames, data_to_plot)
      
    if do_iso_iso25_radius:
      input_dirs    = ['/Users/tflowers/WALLABY/Hydra_DR2/', '/Users/tflowers/WALLABY/NGC4636_DR1/']
      input_fnames  = ['all_normalised_hi.fits', 'all_normalised_rband25.fits']
      if do_log:
        #output_fnames = ['MSTAR/LOG/iso_iso25_mstar_profiles_log.pdf', 'MSTAR/LOG/iso_iso25_mstar_log.pdf']
        output_fnames = ['MSTAR/LOG/iso_iso25_radius_profiles_log_env.pdf', 'MSTAR/LOG/iso_iso25_radius_log_env.pdf']
        xlbl          = 'Radius'
        ylbl          = r'$\log(R_{\rm{iso,HI}}/R_{\rm{iso25,r}})$'
        data_to_plot  = [np.log10(radius_r_iso25_kpc), np.log10(radius_hi_iso/radius_r_iso25), 
                         radius_hi_iso, radius_limit, xlbl, ylbl,
                         pop_hi_array]
      else:
        output_fnames = ['MSTAR/iso_iso25_mstar_profiles.pdf', 'MSTAR/iso_iso25_mstar.pdf']
        xlbl          = lbl_mstar
        ylbl          = r'$R_{\rm{iso,HI}}/R_{\rm{iso25,r}}$'
        data_to_plot  = [mstar, radius_hi_iso/radius_r_iso25, radius_hi_iso, radius_limit, xlbl, ylbl]

      scatter_radial_profiles(input_dirs, input_fnames, output_fnames, data_to_plot)
    
    if do_eff_eff_radius:
      input_dirs    = ['/Users/tflowers/WALLABY/Hydra_DR2/', '/Users/tflowers/WALLABY/NGC4636_DR1/']
      input_fnames  = ['all_normalised_hi.fits', 'all_normalised_rband23.fits']
      if do_log:
        #output_fnames = ['MSTAR/LOG/eff_eff_mstar_profiles_log.pdf', 'MSTAR/LOG/eff_eff_mstar_log.pdf']
        output_fnames = ['MSTAR/LOG/eff_eff_radius_profiles_log_env.pdf', 'MSTAR/LOG/eff_eff_radius_log_env.pdf']
        xlbl          = 'Radius'
        ylbl          = r'$\log(R_{\rm{eff,HI}}/R_{\rm{eff,r}})$'
        data_to_plot  = [radius_r_50_kpc, np.log10(radius_hi_eff/radius_r_50), 
                         radius_hi_eff, radius_limit, xlbl, ylbl,
                         pop_hi_array]
      else:
        output_fnames = ['MSTAR/eff_eff_mstar_profiles.pdf', 'MSTAR/eff_eff_mstar.pdf']
        xlbl          = lbl_mstar
        ylbl          = r'$R_{\rm{eff,HI}}/R_{\rm{eff,r}}$'
        data_to_plot  = [mstar, radius_hi_eff/radius_r_50, radius_hi_eff, radius_limit, xlbl, ylbl]

      scatter_radial_profiles(input_dirs, input_fnames, output_fnames, data_to_plot)
      
    if do_col_hifrac:
      subsample   = ((radius_hi_eff > radius_limit) & np.isfinite(mstar) & 
                     np.isfinite(np.log10(radius_hi_iso/radius_r_iso25)))# & pop_hi_array[2])
      xpar        = mstar[subsample]
      ypar        = np.log10(radius_hi_iso[subsample]/radius_r_iso25[subsample])
      zpar        = hifrac[subsample]
      xlbl        = lbl_mstar
      ylbl        = r'$\log(R_{\rm{iso,HI}}/R_{\rm{iso25,r}})$'
      zlbl        = lbl_hifrac
      fig4 = plt.figure(44, figsize=(6, 4))
      scat_col_simple_plot2(fig4, 1, 1, 1, 
                            xpar[radius_hi_iso[subsample] > 45], 
                            ypar[radius_hi_iso[subsample] > 45], 
                            zpar[radius_hi_iso[subsample] > 45], 
                            [r'$>3$ beams', xlbl, ylbl, zlbl], 
                            ['rainbow_r', 'o', 'rainbow_r', 25, -1.5, 1.6], True, True)
    
      scat_col_simple_plot2(fig4, 1, 1, 1, 
                            xpar[radius_hi_iso[subsample] < 45], 
                            ypar[radius_hi_iso[subsample] < 45], 
                            zpar[radius_hi_iso[subsample] < 45], 
                            [r'$<3$ beams', xlbl, ylbl, zlbl], 
                            ['rainbow_r', 'o', 'none', 25, -1.5, 1.6], True, False)
      
      bins                   = np.arange(7, 11.5, 0.5)
      par_mean_iso, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
      bin_width              = np.abs(bin_edges[1] - bin_edges[0])
      xbins                  = bin_edges[:-1] + bin_width/2.
      p20_iso, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
      p80_iso, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
      
      scatter_error_plot(fig4, 1, 1, 1, xbins, par_mean_iso, [par_mean_iso - p20_iso, p80_iso - par_mean_iso],
                            'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], True)
      #scatter_outlier_plot(fig4, 1, 1, 1, xbins, p20,
                            #'Medians', xlbl, ylbl, ['black', 'X', 'black', 40], False)
      #scatter_outlier_plot(fig4, 1, 1, 1, xbins, p80,
                            #'Medians', xlbl, ylbl, ['black', 'X', 'black', 40], False)
      
      ax1 = fig4.add_subplot(1, 1, 1, facecolor = 'w')
      ax1.set_ylim(-0.3,0.9)
      
      #plot_name = phase1_dir + 'PLOTS/PAPER3/iso_mstar_col.pdf'
      plot_name = phase1_dir + 'PLOTS/PAPER3/iso_mstar_col_e30.pdf'
      plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
      plt.clf()
      
      subsample   = ((radius_hi_eff > radius_limit) & np.isfinite(mstar) & 
                     np.isfinite(np.log10(radius_hi_eff/radius_r_50)))# & pop_hi_array[2])
      xpar        = mstar[subsample]
      ypar        = np.log10(radius_hi_eff[subsample]/radius_r_50[subsample])
      zpar        = hifrac[subsample]
      xlbl        = lbl_mstar
      ylbl        = r'$\log(R_{\rm{eff,HI}}/R_{\rm{eff,r}})$'
      zlbl        = lbl_hifrac
      fig4 = plt.figure(45, figsize=(6, 4))
      scat_col_simple_plot2(fig4, 1, 1, 1, 
                            xpar[radius_hi_eff[subsample] > 45], 
                            ypar[radius_hi_eff[subsample] > 45], 
                            zpar[radius_hi_eff[subsample] > 45], 
                            [r'$>3$ beams', xlbl, ylbl, zlbl], 
                            ['rainbow_r', 'o', 'rainbow_r', 25, -1.5, 1.6], True, True)
    
      scat_col_simple_plot2(fig4, 1, 1, 1, 
                            xpar[radius_hi_eff[subsample] < 45], 
                            ypar[radius_hi_eff[subsample] < 45], 
                            zpar[radius_hi_eff[subsample] < 45], 
                            [r'$<3$ beams', xlbl, ylbl, zlbl], 
                            ['rainbow_r', 'o', 'none', 25, -1.5, 1.6], True, False)
      
      bins                   = np.arange(7, 11.5, 0.5)
      par_mean_eff, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
      bin_width              = np.abs(bin_edges[1] - bin_edges[0])
      xbins                  = bin_edges[:-1] + bin_width/2.
      p20_eff, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
      p80_eff, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
      
      scatter_error_plot(fig4, 1, 1, 1, xbins, par_mean_eff, [par_mean_eff - p20_eff, p80_eff - par_mean_eff],
                            'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], True)
      
      scatter_error_plot(fig4, 1, 1, 1, xbins, par_mean_iso, [par_mean_iso - p20_iso, p80_iso - par_mean_iso],
                            'Medians', xlbl, ylbl, ['grey', 's', 'grey', 7.5], False)
      
      ax1 = fig4.add_subplot(1, 1, 1, facecolor = 'w')
      ax1.set_ylim(-0.3,0.9)
      
      #plot_name = phase1_dir + 'PLOTS/PAPER3/eff_mstar_field_col.pdf'
      plot_name = phase1_dir + 'PLOTS/PAPER3/eff_mstar_col.pdf'
      plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
      plt.clf()
      
    if do_mass_colratio:
      subsample   = ((radius_hi_iso > 30) & np.isfinite(mstar) & 
                    np.isfinite(np.log10(radius_hi_iso/radius_r_iso25)))# & pop_hi_array[2])
      xpar        = mstar[subsample]
      ypar        = mhi[subsample]
      zpar        = np.log10(radius_hi_iso[subsample]/radius_r_iso25[subsample])
      xlbl        = lbl_mstar
      ylbl        = lbl_mhi
      zlbl        = r'$\log(R_{\rm{iso,HI}}/R_{\rm{iso25,r}})$'
      
      print(len(xpar))
      
      fig4 = plt.figure(46, figsize=(6, 4))
      scat_col_simple_plot2(fig4, 1, 1, 1, 
                            xpar[radius_hi_iso[subsample] > 45], 
                            ypar[radius_hi_iso[subsample] > 45], 
                            zpar[radius_hi_iso[subsample] > 45], 
                            [r'$>3$ beams', xlbl, ylbl, zlbl], 
                            ['rainbow_r', 'o', 'rainbow_r', 25, -0.2, 0.9], True, True)
    
      scat_col_simple_plot2(fig4, 1, 1, 1, 
                            xpar[radius_hi_iso[subsample] < 45], 
                            ypar[radius_hi_iso[subsample] < 45], 
                            zpar[radius_hi_iso[subsample] < 45], 
                            [r'$<3$ beams', xlbl, ylbl, zlbl], 
                            ['rainbow_r', 'o', 'none', 25, -0.2, 0.9], True, False)
      
      bins                   = np.arange(7, 11.5, 0.5)
      par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
      bin_width              = np.abs(bin_edges[1] - bin_edges[0])
      xbins                  = bin_edges[:-1] + bin_width/2.
      
      scatter_outlier_plot(fig4, 1, 1, 1, xbins, par_mean,
                            'Medians', xlbl, ylbl, ['black', 's', 'black', 40], True)
      
      ax1 = fig4.add_subplot(1, 1, 1, facecolor = 'w')
      
      ax1.set_xlim(6,11.5)
      ax1.set_ylim(7.5,11)
      x_array = np.arange(6,12.25, 0.25)
      ax1.plot(x_array, x_array+1, color='grey', linewidth=2, linestyle='--', zorder=-1)
      ax1.plot(x_array, x_array+0.5, color='grey', linewidth=1.75, linestyle='--', zorder=-1)
      ax1.plot(x_array, x_array, color='grey', linewidth=1.5, linestyle='--', zorder=-1)
      ax1.plot(x_array, x_array-0.5, color='grey', linewidth=1.25, linestyle='--', zorder=-1)
      ax1.plot(x_array, x_array-1, color='grey', linewidth=1, linestyle='--', zorder=-1)
      #ax1.plot(x_array, x_array-1.5, color='grey', linewidth=0.75, linestyle='--', zorder=-1)
      
      plot_name = phase1_dir + 'PLOTS/PAPER3/mhi_mstar_isoratio.pdf'
      plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
      plt.clf()
      
      
      subsample   = ((radius_hi_eff > 30) & np.isfinite(mstar) & 
                    np.isfinite(np.log10(radius_hi_eff/radius_r_50)))# & pop_hi_array[2])
      xpar        = mstar[subsample]
      ypar        = mhi[subsample]
      zpar        = np.log10(radius_hi_eff[subsample]/radius_r_50[subsample])
      xlbl        = lbl_mstar
      ylbl        = lbl_mhi
      zlbl        = r'$\log(R_{\rm{eff,HI}}/R_{\rm{eff,r}})$'
      
      print(len(xpar))
      
      fig4 = plt.figure(47, figsize=(6, 4))
      scat_col_simple_plot2(fig4, 1, 1, 1, 
                            xpar[radius_hi_eff[subsample] > 45], 
                            ypar[radius_hi_eff[subsample] > 45], 
                            zpar[radius_hi_eff[subsample] > 45], 
                            [r'$>3$ beams', xlbl, ylbl, zlbl], 
                            ['rainbow_r', 'o', 'rainbow_r', 25, -0.2, 0.9], True, True)
    
      scat_col_simple_plot2(fig4, 1, 1, 1, 
                            xpar[radius_hi_eff[subsample] < 45], 
                            ypar[radius_hi_eff[subsample] < 45], 
                            zpar[radius_hi_eff[subsample] < 45], 
                            [r'$<3$ beams', xlbl, ylbl, zlbl], 
                            ['rainbow_r', 'o', 'none', 25, -0.2, 0.9], True, False)
      
      bins                   = np.arange(7, 11.5, 0.5)
      par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
      bin_width              = np.abs(bin_edges[1] - bin_edges[0])
      xbins                  = bin_edges[:-1] + bin_width/2.

      scatter_outlier_plot(fig4, 1, 1, 1, xbins, par_mean,
                            'Medians', xlbl, ylbl, ['black', 's', 'black', 40], True)
      
      ax1 = fig4.add_subplot(1, 1, 1, facecolor = 'w')
      
      ax1.set_xlim(6,11.5)
      ax1.set_ylim(7.5,11)
      x_array = np.arange(6,12.25, 0.25)
      ax1.plot(x_array, x_array+1, color='grey', linewidth=2, linestyle='--', zorder=-1)
      ax1.plot(x_array, x_array+0.5, color='grey', linewidth=1.75, linestyle='--', zorder=-1)
      ax1.plot(x_array, x_array, color='grey', linewidth=1.5, linestyle='--', zorder=-1)
      ax1.plot(x_array, x_array-0.5, color='grey', linewidth=1.25, linestyle='--', zorder=-1)
      ax1.plot(x_array, x_array-1, color='grey', linewidth=1, linestyle='--', zorder=-1)
      #ax1.plot(x_array, x_array-1.5, color='grey', linewidth=0.75, linestyle='--', zorder=-1)
      
      plot_name = phase1_dir + 'PLOTS/PAPER3/mhi_mstar_effratio.pdf'
      plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
      plt.clf()
      
      
      subsample   = ((radius_hi_iso > 30) & np.isfinite(mu_star) & 
                    np.isfinite(np.log10(radius_hi_iso/radius_r_iso25)))# & pop_hi_array[2])
      xpar        = mu_star[subsample]
      ypar        = np.log10(radius_hi_iso[subsample]/radius_r_iso25[subsample])
      zpar        = hifrac[subsample]
      xlbl        = lbl_mustar
      ylbl        = r'$\log(R_{\rm{iso,HI}}/R_{\rm{iso25,r}})$'
      zlbl        = lbl_hifrac
      
      print(len(xpar))
      
      fig4 = plt.figure(48, figsize=(6, 4))
      scat_col_simple_plot2(fig4, 1, 1, 1, 
                            xpar[radius_hi_iso[subsample] > 45], 
                            ypar[radius_hi_iso[subsample] > 45], 
                            zpar[radius_hi_iso[subsample] > 45], 
                            [r'$>3$ beams', xlbl, ylbl, zlbl], 
                            ['rainbow_r', 'o', 'rainbow_r', 25, -1.5, 1.6], True, True)
    
      scat_col_simple_plot2(fig4, 1, 1, 1, 
                            xpar[radius_hi_iso[subsample] < 45], 
                            ypar[radius_hi_iso[subsample] < 45], 
                            zpar[radius_hi_iso[subsample] < 45], 
                            [r'$<3$ beams', xlbl, ylbl, zlbl], 
                            ['rainbow_r', 'o', 'none', 25, -1.5, 1.6], True, False)
      
      bins                   = np.arange(6, 9.25, 0.4)
      par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
      bin_width              = np.abs(bin_edges[1] - bin_edges[0])
      xbins                  = bin_edges[:-1] + bin_width/2.
      p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
      p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
      
      scatter_error_plot(fig4, 1, 1, 1, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                            'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], True)
      #scatter_outlier_plot(fig4, 1, 1, 1, xbins, par_mean,
                            #'Medians', xlbl, ylbl, ['black', 's', 'black', 40], True)
      
      ax1 = fig4.add_subplot(1, 1, 1, facecolor = 'w')
      
      #ax1.set_xlim(6,11.5)
      #ax1.set_ylim(7.5,11)
      #x_array = np.arange(6,12.25, 0.25)
      #ax1.plot(x_array, x_array+1, color='grey', linewidth=2, linestyle='--', zorder=-1)
      #ax1.plot(x_array, x_array+0.5, color='grey', linewidth=1.75, linestyle='--', zorder=-1)
      #ax1.plot(x_array, x_array, color='grey', linewidth=1.5, linestyle='--', zorder=-1)
      #ax1.plot(x_array, x_array-0.5, color='grey', linewidth=1.25, linestyle='--', zorder=-1)
      #ax1.plot(x_array, x_array-1, color='grey', linewidth=1, linestyle='--', zorder=-1)
      #ax1.plot(x_array, x_array-1.5, color='grey', linewidth=0.75, linestyle='--', zorder=-1)
      
      plot_name = phase1_dir + 'PLOTS/PAPER3/sratio_mustar_hifrac.pdf'
      plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
      plt.clf()
      
      
      
      #test_mass = np.arange(7,12,1)
      #test_r25  = 0.31 * test_mass - 1.94
      #test_rhi  = 0.27 * test_mass - 1.24
      #test_rhih = 0.51 * test_mass - 3.59
      #test_mhi  = (3.59 + test_rhi) / 0.51
      #print(test_r25)
      #print(test_rhi)
      #print(test_rhih)
      #print(test_mhi)
      #print(test_mhi - test_mass)
      #print(test_rhi - test_r25)
      
    if do_col_ssfr:
      subsample   = ((radius_hi_iso > radius_limit) & np.isfinite(mstar) & 
                     np.isfinite(np.log10(radius_hi_iso/radius_r_iso25)))# & pop_hi_array[2])
      xpar        = mstar[subsample]
      ypar        = np.log10(radius_hi_iso[subsample]/radius_r_iso25[subsample])
      zpar        = ssfr[subsample]
      xlbl        = lbl_mstar
      ylbl        = r'$\log(R_{\rm{iso,HI}}/R_{\rm{iso25,r}})$'
      zlbl        = lbl_ssfr
      fig4 = plt.figure(44, figsize=(6, 4))
      scat_col_simple_plot2(fig4, 1, 1, 1, 
                            xpar[radius_hi_iso[subsample] > 45], 
                            ypar[radius_hi_iso[subsample] > 45], 
                            zpar[radius_hi_iso[subsample] > 45], 
                            [r'$>3$ beams', xlbl, ylbl, zlbl], 
                            ['rainbow_r', 'o', 'rainbow_r', 25, -11, -8.75], True, True) # -2.5, 1.5
    
      scat_col_simple_plot2(fig4, 1, 1, 1, 
                            xpar[radius_hi_iso[subsample] < 45], 
                            ypar[radius_hi_iso[subsample] < 45], 
                            zpar[radius_hi_iso[subsample] < 45], 
                            [r'$<3$ beams', xlbl, ylbl, zlbl], 
                            ['rainbow_r', 'o', 'none', 25, -11, -8.75], True, False)
      
      bins                   = np.arange(7, 11.5, 0.5)
      par_mean_iso, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
      bin_width              = np.abs(bin_edges[1] - bin_edges[0])
      xbins                  = bin_edges[:-1] + bin_width/2.
      p20_iso, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
      p80_iso, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
      
      scatter_error_plot(fig4, 1, 1, 1, xbins, par_mean_iso, [par_mean_iso - p20_iso, p80_iso - par_mean_iso],
                            'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], True)
      #scatter_outlier_plot(fig4, 1, 1, 1, xbins, p20,
                            #'Medians', xlbl, ylbl, ['black', 'X', 'black', 40], False)
      #scatter_outlier_plot(fig4, 1, 1, 1, xbins, p80,
                            #'Medians', xlbl, ylbl, ['black', 'X', 'black', 40], False)
      
      ax1 = fig4.add_subplot(1, 1, 1, facecolor = 'w')
      ax1.set_ylim(-0.3,0.9)
      
      plot_name = phase1_dir + 'PLOTS/PAPER3/iso_mstar_ssfr.pdf'
      #plot_name = phase1_dir + 'PLOTS/PAPER3/iso_mstar_ssfr_e30.pdf'
      plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
      plt.clf()
      
      subsample   = ((radius_hi_eff > radius_limit) & np.isfinite(mstar) & 
                     np.isfinite(np.log10(radius_hi_eff/radius_r_50)))# & pop_hi_array[2])
      xpar        = mstar[subsample]
      ypar        = np.log10(radius_hi_eff[subsample]/radius_r_50[subsample])
      zpar        = ssfr[subsample]
      xlbl        = lbl_mstar
      ylbl        = r'$\log(R_{\rm{eff,HI}}/R_{\rm{eff,r}})$'
      zlbl        = lbl_ssfr
      fig4 = plt.figure(45, figsize=(6, 4))
      scat_col_simple_plot2(fig4, 1, 1, 1, 
                            xpar[radius_hi_eff[subsample] > 45], 
                            ypar[radius_hi_eff[subsample] > 45], 
                            zpar[radius_hi_eff[subsample] > 45], 
                            [r'$>3$ beams', xlbl, ylbl, zlbl], 
                            ['rainbow_r', 'o', 'rainbow_r', 25, -11, -8.75], True, True)
    
      scat_col_simple_plot2(fig4, 1, 1, 1, 
                            xpar[radius_hi_eff[subsample] < 45], 
                            ypar[radius_hi_eff[subsample] < 45], 
                            zpar[radius_hi_eff[subsample] < 45], 
                            [r'$<3$ beams', xlbl, ylbl, zlbl], 
                            ['rainbow_r', 'o', 'none', 25, -11, -8.75], True, False)
      
      bins                   = np.arange(7, 11.5, 0.5)
      par_mean_eff, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
      bin_width              = np.abs(bin_edges[1] - bin_edges[0])
      xbins                  = bin_edges[:-1] + bin_width/2.
      p20_eff, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
      p80_eff, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
      
      scatter_error_plot(fig4, 1, 1, 1, xbins, par_mean_eff, [par_mean_eff - p20_eff, p80_eff - par_mean_eff],
                            'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], True)
      
      scatter_error_plot(fig4, 1, 1, 1, xbins, par_mean_iso, [par_mean_iso - p20_iso, p80_iso - par_mean_iso],
                            'Medians', xlbl, ylbl, ['grey', 's', 'grey', 7.5], False)
      
      ax1 = fig4.add_subplot(1, 1, 1, facecolor = 'w')
      ax1.set_ylim(-0.3,0.9)
      
      #plot_name = phase1_dir + 'PLOTS/PAPER3/eff_mstar_field_col.pdf'
      plot_name = phase1_dir + 'PLOTS/PAPER3/eff_mstar_ssfr.pdf'
      plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
      plt.clf()
      
    if do_col_nuv:
      subsample   = ((radius_hi_iso > radius_limit) & np.isfinite(mstar) & 
                     np.isfinite(np.log10(radius_hi_iso/radius_r_iso25)) & 
                     np.isfinite(np.log10(radius_hi_iso/radius_nuv_iso)))# & pop_hi_array[2])
      xpar        = mstar[subsample]
      #ypar        = np.log10(radius_hi_iso[subsample]/radius_r_iso25[subsample])
      ypar        = np.log10(radius_hi_iso[subsample]/radius_nuv_iso[subsample])
      zpar        = np.log10(radius_nuv_iso[subsample]/radius_r_iso25[subsample])
      xlbl        = lbl_mstar
      ylbl        = r'$\log(R_{\rm{iso,HI}}/R_{\rm{iso25,r}})$'
      zlbl        = r'$\log(R_{\rm{iso,nuv}}/R_{\rm{iso25,r}})$'
      fig4 = plt.figure(44, figsize=(6, 4))
      scat_col_simple_plot2(fig4, 1, 1, 1, 
                            xpar[radius_hi_iso[subsample] > 45], 
                            ypar[radius_hi_iso[subsample] > 45], 
                            zpar[radius_hi_iso[subsample] > 45], 
                            [r'$>3$ beams', xlbl, ylbl, zlbl], 
                            ['rainbow_r', 'o', 'rainbow_r', 25, -0.5, 0.2], True, True) # -2.5, 1.5
    
      scat_col_simple_plot2(fig4, 1, 1, 1, 
                            xpar[radius_hi_iso[subsample] < 45], 
                            ypar[radius_hi_iso[subsample] < 45], 
                            zpar[radius_hi_iso[subsample] < 45], 
                            [r'$<3$ beams', xlbl, ylbl, zlbl], 
                            ['rainbow_r', 'o', 'none', 25, -0.5, 0.2], True, False)
      
      bins                   = np.arange(7, 11.5, 0.5)
      par_mean_iso, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
      bin_width              = np.abs(bin_edges[1] - bin_edges[0])
      xbins                  = bin_edges[:-1] + bin_width/2.
      p20_iso, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
      p80_iso, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
      
      scatter_error_plot(fig4, 1, 1, 1, xbins, par_mean_iso, [par_mean_iso - p20_iso, p80_iso - par_mean_iso],
                            'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], True)
      #scatter_outlier_plot(fig4, 1, 1, 1, xbins, p20,
                            #'Medians', xlbl, ylbl, ['black', 'X', 'black', 40], False)
      #scatter_outlier_plot(fig4, 1, 1, 1, xbins, p80,
                            #'Medians', xlbl, ylbl, ['black', 'X', 'black', 40], False)
      
      ax1 = fig4.add_subplot(1, 1, 1, facecolor = 'w')
      ax1.set_ylim(-0.3,0.9)
      
      plot_name = phase1_dir + 'PLOTS/PAPER3/iso_mstar_nuv.pdf'
      #plot_name = phase1_dir + 'PLOTS/PAPER3/iso_mstar_ssfr_e30.pdf'
      plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
      plt.clf()
      
      subsample   = ((radius_hi_eff > radius_limit) & np.isfinite(mstar) & 
                     np.isfinite(np.log10(radius_hi_eff/radius_r_50)))# & pop_hi_array[2])
      xpar        = mstar[subsample]
      ypar        = np.log10(radius_hi_eff[subsample]/radius_r_50[subsample])
      zpar        = np.log10(radius_hi_eff[subsample]/radius_nuv_iso[subsample])
      xlbl        = lbl_mstar
      ylbl        = r'$\log(R_{\rm{eff,HI}}/R_{\rm{eff,r}})$'
      zlbl        = r'$\log(R_{\rm{eff,HI}}/R_{\rm{iso,nuv}})$'
      fig4 = plt.figure(45, figsize=(6, 4))
      scat_col_simple_plot2(fig4, 1, 1, 1, 
                            xpar[radius_hi_eff[subsample] > 45], 
                            ypar[radius_hi_eff[subsample] > 45], 
                            zpar[radius_hi_eff[subsample] > 45], 
                            [r'$>3$ beams', xlbl, ylbl, zlbl], 
                            ['rainbow_r', 'o', 'rainbow_r', 25, -0.25, 0.75], True, True)
    
      scat_col_simple_plot2(fig4, 1, 1, 1, 
                            xpar[radius_hi_eff[subsample] < 45], 
                            ypar[radius_hi_eff[subsample] < 45], 
                            zpar[radius_hi_eff[subsample] < 45], 
                            [r'$<3$ beams', xlbl, ylbl, zlbl], 
                            ['rainbow_r', 'o', 'none', 25, -0.25, 0.75], True, False)
      
      bins                   = np.arange(7, 11.5, 0.5)
      par_mean_eff, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
      bin_width              = np.abs(bin_edges[1] - bin_edges[0])
      xbins                  = bin_edges[:-1] + bin_width/2.
      p20_eff, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
      p80_eff, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
      
      scatter_error_plot(fig4, 1, 1, 1, xbins, par_mean_eff, [par_mean_eff - p20_eff, p80_eff - par_mean_eff],
                            'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], True)
      
      scatter_error_plot(fig4, 1, 1, 1, xbins, par_mean_iso, [par_mean_iso - p20_iso, p80_iso - par_mean_iso],
                            'Medians', xlbl, ylbl, ['grey', 's', 'grey', 7.5], False)
      
      ax1 = fig4.add_subplot(1, 1, 1, facecolor = 'w')
      ax1.set_ylim(-0.3,0.9)
      
      #plot_name = phase1_dir + 'PLOTS/PAPER3/eff_mstar_field_col.pdf'
      plot_name = phase1_dir + 'PLOTS/PAPER3/eff_mstar_nuv.pdf'
      plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
      plt.clf()
    
    if do_col_dsfms:
      subsample   = ((radius_hi_iso > radius_limit) & np.isfinite(delta_sfms) & 
                     np.isfinite(np.log10(radius_hi_iso/radius_r_iso25)))# & pop_hi_array[2])
      xpar        = delta_sfms[subsample]
      ypar        = np.log10(radius_hi_iso[subsample]/radius_r_iso25[subsample])
      zpar        = surfden_hi_eff[subsample]
      xlbl        = r'$\Delta$\,SFMS [dex]'
      ylbl        = r'$\log(R_{\rm{iso,HI}}/R_{\rm{iso25,r}})$'
      zlbl        = r'$\mu_{\rm{HI}}/[\rm{M_{\odot}\,pc^{-2}}]$'
      fig4 = plt.figure(44, figsize=(6, 4))
      #scat_col_simple_plot2(fig4, 1, 1, 1, 
                            #xpar[radius_hi_iso[subsample] > 45], 
                            #ypar[radius_hi_iso[subsample] > 45], 
                            #zpar[radius_hi_iso[subsample] > 45], 
                            #[r'$>3$ beams', xlbl, ylbl, zlbl], 
                            #['rainbow_r', 'o', 'rainbow_r', 25, 0, 9], True, True) # -2.5, 1.5
    
      #scat_col_simple_plot2(fig4, 1, 1, 1, 
                            #xpar[radius_hi_iso[subsample] < 45], 
                            #ypar[radius_hi_iso[subsample] < 45], 
                            #zpar[radius_hi_iso[subsample] < 45], 
                            #[r'$<3$ beams', xlbl, ylbl, zlbl], 
                            #['rainbow_r', 'o', 'none', 25, 0, 9], True, False)
      
      scat_col_simple_plot2(fig4, 1, 1, 1, 
                            xpar[~sfr_uplim[subsample]], 
                            ypar[~sfr_uplim[subsample]], 
                            zpar[~sfr_uplim[subsample]], 
                            [r'$>3$ beams', xlbl, ylbl, zlbl], 
                            ['rainbow_r', 'o', 'rainbow_r', 25, 0, 9], True, True) # -2.5, 1.5
    
      scat_col_simple_plot2(fig4, 1, 1, 1, 
                            xpar[sfr_uplim[subsample]], 
                            ypar[sfr_uplim[subsample]], 
                            zpar[sfr_uplim[subsample]], 
                            [r'$<3$ beams', xlbl, ylbl, zlbl], 
                            ['rainbow_r', r'$\leftarrow$', 'rainbow_r', 25, 0, 9], True, False)
      
      
      bins                   = np.arange(-1, 1.2, 0.2)
      par_mean_iso, bin_edges, _ = binned_statistic(xpar[~sfr_uplim[subsample]], ypar[~sfr_uplim[subsample]], np.nanmedian, bins)
      par_count, _, _            = binned_statistic(xpar, ypar, 'count', bins)
      bin_width              = np.abs(bin_edges[1] - bin_edges[0])
      xbins                  = bin_edges[:-1] + bin_width/2.
      p20_iso, _, _ = binned_statistic(xpar[~sfr_uplim[subsample]], ypar[~sfr_uplim[subsample]], lambda y: np.percentile(y, 20), bins)
      p80_iso, _, _ = binned_statistic(xpar[~sfr_uplim[subsample]], ypar[~sfr_uplim[subsample]], lambda y: np.percentile(y, 80), bins)
      
      print(par_mean_iso, par_mean_iso - p20_iso, par_count)
      
      scatter_error_plot(fig4, 1, 1, 1, xbins, par_mean_iso, [par_mean_iso - p20_iso, p80_iso - par_mean_iso],
                            'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], True)
      #scatter_outlier_plot(fig4, 1, 1, 1, xbins, p20,
                            #'Medians', xlbl, ylbl, ['black', 'X', 'black', 40], False)
      #scatter_outlier_plot(fig4, 1, 1, 1, xbins, p80,
                            #'Medians', xlbl, ylbl, ['black', 'X', 'black', 40], False)
      
      ax1 = fig4.add_subplot(1, 1, 1, facecolor = 'w')
      ax1.set_ylim(-0.3,0.9)
      
      plot_name = phase1_dir + 'PLOTS/PAPER3/iso_mstar_dsfms.pdf'
      #plot_name = phase1_dir + 'PLOTS/PAPER3/iso_mstar_ssfr_e30.pdf'
      plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
      plt.clf()
      
      subsample   = ((radius_hi_eff > radius_limit) & np.isfinite(mstar) & 
                     np.isfinite(np.log10(radius_hi_eff/radius_r_50)))# & pop_hi_array[2])
      xpar        = delta_sfms[subsample]
      ypar        = np.log10(radius_hi_eff[subsample]/radius_r_50[subsample])
      zpar        = surfden_hi_eff[subsample]
      xlbl        = r'$\Delta$\,SFMS [dex]'
      ylbl        = r'$\log(R_{\rm{eff,HI}}/R_{\rm{eff,r}})$'
      zlbl        = lbl_ssfr
      fig4 = plt.figure(45, figsize=(6, 4))
      scat_col_simple_plot2(fig4, 1, 1, 1, 
                            xpar[radius_hi_eff[subsample] > 45], 
                            ypar[radius_hi_eff[subsample] > 45], 
                            zpar[radius_hi_eff[subsample] > 45], 
                            [r'$>3$ beams', xlbl, ylbl, zlbl], 
                            ['rainbow_r', 'o', 'rainbow_r', 25, 0, 10], True, True)
    
      scat_col_simple_plot2(fig4, 1, 1, 1, 
                            xpar[radius_hi_eff[subsample] < 45], 
                            ypar[radius_hi_eff[subsample] < 45], 
                            zpar[radius_hi_eff[subsample] < 45], 
                            [r'$<3$ beams', xlbl, ylbl, zlbl], 
                            ['rainbow_r', 'o', 'none', 25, 0, 10], True, False)
      
      #bins                   = np.arange(-0.75, 1, 0.5)
      par_mean_eff, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
      bin_width              = np.abs(bin_edges[1] - bin_edges[0])
      xbins                  = bin_edges[:-1] + bin_width/2.
      p20_eff, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
      p80_eff, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
      
      scatter_error_plot(fig4, 1, 1, 1, xbins, par_mean_eff, [par_mean_eff - p20_eff, p80_eff - par_mean_eff],
                            'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], True)
      
      scatter_error_plot(fig4, 1, 1, 1, xbins, par_mean_iso, [par_mean_iso - p20_iso, p80_iso - par_mean_iso],
                            'Medians', xlbl, ylbl, ['grey', 's', 'grey', 7.5], False)
      
      ax1 = fig4.add_subplot(1, 1, 1, facecolor = 'w')
      ax1.set_ylim(-0.3,0.9)
      
      #plot_name = phase1_dir + 'PLOTS/PAPER3/eff_mstar_field_col.pdf'
      plot_name = phase1_dir + 'PLOTS/PAPER3/eff_mstar_dsfms.pdf'
      plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
      plt.clf()

    if do_iso_ssfr:
      subsample   = ((radius_hi_iso > radius_limit) & np.isfinite(mstar) & 
                     np.isfinite(np.log10(radius_hi_iso/radius_r_iso25)))# & pop_hi_array[2])
      zpar        = surfden_hi_iso[subsample]
      ypar        = np.log10(radius_hi_iso[subsample]/radius_r_iso25[subsample])
      xpar        = ssfr[subsample]
      zlbl        = r'$\mu_{\rm{HI}}/[\rm{M_{\odot}\,pc^{-2}}]$'
      ylbl        = r'$\log(R_{\rm{iso,HI}}/R_{\rm{iso25,r}})$'
      xlbl        = lbl_ssfr
      fig4 = plt.figure(44, figsize=(6, 4))
      #scat_col_simple_plot2(fig4, 1, 1, 1, 
                            #xpar[radius_hi_iso[subsample] > 45], 
                            #ypar[radius_hi_iso[subsample] > 45], 
                            #zpar[radius_hi_iso[subsample] > 45], 
                            #[r'$>3$ beams', xlbl, ylbl, zlbl], 
                            #['rainbow_r', 'o', 'rainbow_r', 25, 1, 4.5], True, True) # -2.5, 1.5
    
      #scat_col_simple_plot2(fig4, 1, 1, 1, 
                            #xpar[radius_hi_iso[subsample] < 45], 
                            #ypar[radius_hi_iso[subsample] < 45], 
                            #zpar[radius_hi_iso[subsample] < 45], 
                            #[r'$<3$ beams', xlbl, ylbl, zlbl], 
                            #['rainbow_r', 'o', 'none', 25, 1, 4.5], True, False)
      
      scat_col_simple_plot2(fig4, 1, 1, 1, 
                            xpar[~sfr_uplim[subsample]], 
                            ypar[~sfr_uplim[subsample]], 
                            zpar[~sfr_uplim[subsample]], 
                            ['SFR', xlbl, ylbl, zlbl], 
                            ['rainbow_r', 'o', 'rainbow_r', 25, 1, 4.5], True, True) # -2.5, 1.5
    
      scat_col_simple_plot2(fig4, 1, 1, 1, 
                            xpar[sfr_uplim[subsample]], 
                            ypar[sfr_uplim[subsample]], 
                            zpar[sfr_uplim[subsample]], 
                            ['SFR UPLIM', xlbl, ylbl, zlbl], 
                            ['rainbow_r', r'$\leftarrow$', 'rainbow_r', 25, 1, 4.5], True, False)
      
      bins                   = np.arange(-10.6, -8.8, 0.2)
      par_mean_iso, bin_edges, _ = binned_statistic(xpar[~sfr_uplim[subsample]], ypar[~sfr_uplim[subsample]], np.nanmedian, bins)
      bin_width              = np.abs(bin_edges[1] - bin_edges[0])
      xbins                  = bin_edges[:-1] + bin_width/2.
      p20_iso, _, _ = binned_statistic(xpar[~sfr_uplim[subsample]], ypar[~sfr_uplim[subsample]], lambda y: np.percentile(y, 20), bins)
      p80_iso, _, _ = binned_statistic(xpar[~sfr_uplim[subsample]], ypar[~sfr_uplim[subsample]], lambda y: np.percentile(y, 80), bins)
      
      print(par_mean_iso)
      
      scatter_error_plot(fig4, 1, 1, 1, xbins, par_mean_iso, [par_mean_iso - p20_iso, p80_iso - par_mean_iso],
                            'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], True)
      
      par_mean_iso, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
      bin_width              = np.abs(bin_edges[1] - bin_edges[0])
      xbins                  = bin_edges[:-1] + bin_width/2.
      p20_iso, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
      p80_iso, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
      
      print(par_mean_iso)
      
      scatter_error_plot(fig4, 1, 1, 1, xbins, par_mean_iso, [par_mean_iso - p20_iso, p80_iso - par_mean_iso],
                            'Medians', xlbl, ylbl, ['grey', 's', 'none', 7.5], False)
      
      ax1 = fig4.add_subplot(1, 1, 1, facecolor = 'w')
      ax1.set_ylim(-0.3,0.9)
      
      plot_name = phase1_dir + 'PLOTS/PAPER3/iso_ratio_ssfr.pdf'
      #plot_name = phase1_dir + 'PLOTS/PAPER3/iso_mstar_ssfr_e30.pdf'
      plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
      plt.clf()
      
      subsample   = ((radius_hi_eff > radius_limit) & np.isfinite(mstar) & 
                     np.isfinite(np.log10(radius_hi_eff/radius_r_50)))# & pop_hi_array[2])
      zpar        = surfden_hi_eff[subsample]
      ypar        = np.log10(radius_hi_eff[subsample]/radius_r_50[subsample])
      xpar        = ssfr[subsample]
      zlbl        = r'$\mu_{\rm{HI}}/[\rm{M_{\odot}\,pc^{-2}}]$'
      ylbl        = r'$\log(R_{\rm{eff,HI}}/R_{\rm{eff,r}})$'
      xlbl        = lbl_ssfr
      fig4 = plt.figure(45, figsize=(6, 4))
      scat_col_simple_plot2(fig4, 1, 1, 1, 
                            xpar[radius_hi_eff[subsample] > 45], 
                            ypar[radius_hi_eff[subsample] > 45], 
                            zpar[radius_hi_eff[subsample] > 45], 
                            [r'$>3$ beams', xlbl, ylbl, zlbl], 
                            ['rainbow_r', 'o', 'rainbow_r', 25, 0, 10], True, True)
    
      scat_col_simple_plot2(fig4, 1, 1, 1, 
                            xpar[radius_hi_eff[subsample] < 45], 
                            ypar[radius_hi_eff[subsample] < 45], 
                            zpar[radius_hi_eff[subsample] < 45], 
                            [r'$<3$ beams', xlbl, ylbl, zlbl], 
                            ['rainbow_r', 'o', 'none', 25, 0, 10], True, False)
      
      #bins                   = np.arange(-10.5, -8.5, 0.5)
      par_mean_eff, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
      bin_width              = np.abs(bin_edges[1] - bin_edges[0])
      xbins                  = bin_edges[:-1] + bin_width/2.
      p20_eff, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
      p80_eff, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
      
      scatter_error_plot(fig4, 1, 1, 1, xbins, par_mean_eff, [par_mean_eff - p20_eff, p80_eff - par_mean_eff],
                            'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], True)
      
      scatter_error_plot(fig4, 1, 1, 1, xbins, par_mean_iso, [par_mean_iso - p20_iso, p80_iso - par_mean_iso],
                            'Medians', xlbl, ylbl, ['grey', 's', 'grey', 7.5], False)
      
      ax1 = fig4.add_subplot(1, 1, 1, facecolor = 'w')
      ax1.set_ylim(-0.3,0.9)
      
      #plot_name = phase1_dir + 'PLOTS/PAPER3/eff_mstar_field_col.pdf'
      plot_name = phase1_dir + 'PLOTS/PAPER3/eff_ratio_ssfr.pdf'
      plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
      plt.clf()
      
      
    if do_iso_nuvr:
      subsample   = ((radius_hi_iso > radius_limit) & np.isfinite(mstar) & np.isfinite(nuvr) & 
                     np.isfinite(np.log10(radius_hi_iso/radius_r_iso25)))# & pop_hi_array[2])
      zpar        = mstar[subsample]
      ypar        = np.log10(radius_hi_iso[subsample]/radius_r_iso25[subsample])
      xpar        = nuvr[subsample]
      zlbl        = lbl_mstar
      ylbl        = r'$\log(R_{\rm{iso,HI}}/R_{\rm{iso25,r}})$'
      xlbl        = lbl_nuvr
      fig4 = plt.figure(44, figsize=(6, 4))
      #scat_col_simple_plot2(fig4, 1, 1, 1, 
                            #xpar[radius_hi_iso[subsample] > 45], 
                            #ypar[radius_hi_iso[subsample] > 45], 
                            #zpar[radius_hi_iso[subsample] > 45], 
                            #[r'$>3$ beams', xlbl, ylbl, zlbl], 
                            #['rainbow_r', 'o', 'rainbow_r', 25, 1, 4.5], True, True) # -2.5, 1.5
    
      #scat_col_simple_plot2(fig4, 1, 1, 1, 
                            #xpar[radius_hi_iso[subsample] < 45], 
                            #ypar[radius_hi_iso[subsample] < 45], 
                            #zpar[radius_hi_iso[subsample] < 45], 
                            #[r'$<3$ beams', xlbl, ylbl, zlbl], 
                            #['rainbow_r', 'o', 'none', 25, 1, 4.5], True, False)
      
      scat_col_simple_plot2(fig4, 1, 1, 1, 
                            xpar, 
                            ypar, 
                            zpar, 
                            ['SFR', xlbl, ylbl, zlbl], 
                            ['rainbow', 'o', 'rainbow', 25, 7, 11.5], True, True) # -2.5, 1.5
    
      #scat_col_simple_plot2(fig4, 1, 1, 1, 
                            #xpar[sfr_uplim[subsample]], 
                            #ypar[sfr_uplim[subsample]], 
                            #zpar[sfr_uplim[subsample]], 
                            #['SFR UPLIM', xlbl, ylbl, zlbl], 
                            #['rainbow_r', r'$\leftarrow$', 'rainbow_r', 25, 1, 4.5], True, False)
      
      bins                   = np.arange(0, 4.5, 0.5)
      par_mean_iso, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
      bin_width              = np.abs(bin_edges[1] - bin_edges[0])
      xbins                  = bin_edges[:-1] + bin_width/2.
      p20_iso, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
      p80_iso, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
      
      print(par_mean_iso)
      
      scatter_error_plot(fig4, 1, 1, 1, xbins, par_mean_iso, [par_mean_iso - p20_iso, p80_iso - par_mean_iso],
                            'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
      
      ax1 = fig4.add_subplot(1, 1, 1, facecolor = 'w')
      ax1.set_ylim(-0.3,0.9)
      
      plot_name = phase1_dir + 'PLOTS/PAPER3/iso_ratio_nuvr.pdf'
      #plot_name = phase1_dir + 'PLOTS/PAPER3/iso_mstar_ssfr_e30.pdf'
      plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
      plt.clf()
      
      subsample   = ((radius_hi_eff > radius_limit) & np.isfinite(mstar) & 
                     np.isfinite(np.log10(radius_hi_eff/radius_r_50)))# & pop_hi_array[2])
      zpar        = mstar[subsample]
      ypar        = np.log10(radius_hi_eff[subsample]/radius_r_50[subsample])
      xpar        = nuvr[subsample]
      zlbl        = lbl_mstar
      ylbl        = r'$\log(R_{\rm{eff,HI}}/R_{\rm{eff,r}})$'
      xlbl        = lbl_nuvr
      fig4 = plt.figure(45, figsize=(6, 4))
      scat_col_simple_plot2(fig4, 1, 1, 1, 
                            xpar, 
                            ypar, 
                            zpar, 
                            [r'$>3$ beams', xlbl, ylbl, zlbl], 
                            ['rainbow', 'o', 'rainbow', 25, 7, 11.5], True, True)
    
      #scat_col_simple_plot2(fig4, 1, 1, 1, 
                            #xpar[radius_hi_eff[subsample] < 45], 
                            #ypar[radius_hi_eff[subsample] < 45], 
                            #zpar[radius_hi_eff[subsample] < 45], 
                            #[r'$<3$ beams', xlbl, ylbl, zlbl], 
                            #['rainbow_r', 'o', 'none', 25, 0, 10], True, False)
      
      #bins                   = np.arange(-10.5, -8.5, 0.5)
      par_mean_eff, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
      bin_width              = np.abs(bin_edges[1] - bin_edges[0])
      xbins                  = bin_edges[:-1] + bin_width/2.
      p20_eff, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
      p80_eff, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
      
      scatter_error_plot(fig4, 1, 1, 1, xbins, par_mean_eff, [par_mean_eff - p20_eff, p80_eff - par_mean_eff],
                            'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], True)
      
      scatter_error_plot(fig4, 1, 1, 1, xbins, par_mean_iso, [par_mean_iso - p20_iso, p80_iso - par_mean_iso],
                            'Medians', xlbl, ylbl, ['grey', 's', 'grey', 7.5], False)
      
      ax1 = fig4.add_subplot(1, 1, 1, facecolor = 'w')
      ax1.set_ylim(-0.3,0.9)
      
      #plot_name = phase1_dir + 'PLOTS/PAPER3/eff_mstar_field_col.pdf'
      plot_name = phase1_dir + 'PLOTS/PAPER3/eff_ratio_nuvr.pdf'
      plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
      plt.clf()


# ========== Radial Profiles =========== #
if do_hism_scatter:
  do_hism_plot     = True
  do_offset_hist   = True
  do_offset_jhi    = True
  do_mh2_offset    = False
  do_error_offset  = True
  do_size_offset   = False
  do_mass_offset   = False
  do_h2_jhi_offset = False
  do_h2_jbar_plot  = False
  
  if do_hism_plot:
    #subsample   = (radius_hi_iso > 30) & np.isfinite(mhi) & np.isfinite(radius_hi_iso_kpc)
    subsample  = ((radius_hi_iso > 30) & np.isfinite(mhi) & np.isfinite(mstar) & 
                  np.isfinite(radius_r_iso25_kpc) & 
                  np.isfinite(radius_hi_iso_kpc) & (mstar > 6.5))
    data_array  = [mhi[subsample], np.log10(radius_hi_iso_kpc[subsample])]
    
    result = linregress(data_array)
    #result = np.round(linregress(data_array),2)
    print(np.round(result,3))
        
    xfit   = np.arange(6.5, 11.5, 0.25)
    yfit   = result[0] * xfit + result[1]
    dfit   = np.std(result[0] * data_array[0] + result[1] - data_array[1])
    offset = result[0] * data_array[0] + result[1] - data_array[1]
    
    fig1 = plt.figure(11, figsize=(5, 5))#, facecolor = '#007D7D')
    
    xlbl         = lbl_mhi
    ylbl         = r'Radius [kpc]'
      
    xpar    = mhi
    ypar    = radius_hi_iso_kpc
    limpar  = no_galex
    colour  = ['grey', 'darkblue', 'peru', 'green']
    poplbl  = [r'$|$Offset$|$ $<1.2\sigma$', r'Offset $<-1.2\sigma$', r'Offset $>1.2\sigma$']
    
    ylbl    = r'$\log(R_{\rm{iso,HI}}/[\rm{kpc}])$'
    
    row, column = 1, 1
    
    
    
    #subsample  = (radius_hi_iso > 30) & np.isfinite(mstar) & np.isfinite(xpar) & np.isfinite(ypar)
    
    data_array = [xpar[subsample], np.log10(ypar[subsample])]
    
    print(len(data_array[0]))
    
    print('m, c, rval, pval, stderr')
    
    xfit = np.arange(7.5, 10.75, 0.25)
    yfit = result[0] * xfit + result[1]
    dfit = np.std(result[0] * data_array[0] + result[1] - data_array[1])
    offset = result[0] * data_array[0] + result[1] - data_array[1]
    
    subsample_b = [(offset < 1.2*dfit) & (offset > -1.2*dfit) & (data_array[0] > 6),
                  (offset > 1.2*dfit) & (data_array[0] > 6),
                  (offset < -1.2*dfit) & (data_array[0] > 6)]
    
    #subsample_b = [(offset < 2*dfit) & (offset > -2*dfit) & (data_array[0] > 6),
                  #(offset > 2*dfit) & (data_array[0] > 6),
                  #(offset < -2*dfit) & (data_array[0] > 6)]
    
    for i in range(3):
      #scat_mean_plot(fig1, row, column, 1, data_array[0][subsample_b[i]], data_array[1][subsample_b[i]], False, False, 
                    #poplbl[i], xlbl, ylbl, colour[i],  'o', False)
      if i == 0:
        legend = True
      else:
        legend = False
      #scat_mean_plot(fig1, row, column, 1, 
                           #data_array[0][subsample_b[i] & (radius_hi_iso[subsample] > 45)], 
                           #data_array[1][subsample_b[i] & (radius_hi_iso[subsample] > 45)],
                           #False, False,
                    #poplbl[i], xlbl, ylbl, colour[i], 'o', False)
      
      #scat_mean_plot(fig1, row, column, 1, 
                           #data_array[0][subsample_b[i] & (radius_hi_iso[subsample] < 45)], 
                           #data_array[1][subsample_b[i] & (radius_hi_iso[subsample] < 45)],
                           #False, False,
                    #poplbl[i], xlbl, ylbl, colour[i], '.', False)
      
      scatter_outlier_plot(fig1, row, column, 1, 
                           data_array[0][subsample_b[i] & (radius_hi_iso[subsample] > 45)], 
                           data_array[1][subsample_b[i] & (radius_hi_iso[subsample] > 45)],
                           r'$>3$ beams', xlbl, ylbl, [colour[i], 'o', colour[i], 25], legend)
      
      scatter_outlier_plot(fig1, row, column, 1, 
                           data_array[0][subsample_b[i] & (radius_hi_iso[subsample] < 45)], 
                           data_array[1][subsample_b[i] & (radius_hi_iso[subsample] < 45)],
                           r'$<3$ beams', xlbl, ylbl, [colour[i], 'o', 'none', 25], legend)
      
      result = linregress(data_array)
      print(np.round(result,2))
      
      ax1 = fig1.add_subplot(row, column, 1, facecolor = 'w')

      if i == 0:
        #ax1.text(0.8, 0.05, '%.2f' % result[2], transform=ax1.transAxes, fontsize=12)
        ax1.plot(xfit, yfit, color=colour[0], linewidth=1, linestyle = '--', zorder=1)
        ax1.fill_between(xfit, yfit-dfit, yfit+dfit, alpha=0.25, edgecolor='none', zorder=-1, facecolor=colour[0])
      
    plot_name = phase1_dir + 'PLOTS/PROPERTIES/HISM_SCATTER/hi_iso_vs_mhi.pdf'
    plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()
    
    #fig1 = plt.figure(1, figsize=(3, 3))
    #xlbl         = lbl_mhi
    #ylbl         = r'Radius [kpc]'
    #zlbl         = r'$\theta$ [degrees]'
    #scat_col_simple_plot(fig1, 1, 1, 1, data_array[0], data_array[1], hi_incl[subsample], 
                         #xlbl, ylbl, zlbl, 'viridis', 'o')
    
    #ax1 = fig1.add_subplot(1, 1, 1, facecolor = 'w')
    #ax1.plot(xfit, yfit, color='black', linewidth=1, linestyle = '--', zorder=1)
    
    #plot_name = phase1_dir + 'PLOTS/PROPERTIES/HISM_SCATTER/hi_iso_vs_mhi_incl.pdf'
    #plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    #plt.clf()
  
  
    #fig10 = plt.figure(10, figsize=(3, 3))
    #xlbl         = lbl_mhi
    #ylbl         = r'Radius [kpc]'
    #zlbl         = r'Percent Error'
    #scat_col_simple_plot(fig10, 1, 1, 1, data_array[0], data_array[1], 
                         #radius_error[subsample]/radius_hi_iso[subsample]*100., 
                         #xlbl, ylbl, zlbl, 'viridis', 'o')
    
    #ax1 = fig10.add_subplot(1, 1, 1, facecolor = 'w')
    #ax1.plot(xfit, yfit, color='black', linewidth=1, linestyle = '--', zorder=1)
    
    #plot_name = phase1_dir + 'PLOTS/PROPERTIES/HISM_SCATTER/hi_iso_vs_mhi_error.pdf'
    #plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    #plt.clf()
  
  if do_offset_hist:
    fig3 = plt.figure(3, figsize=(6, 6))
    
    diff_ver = data_array[1] - (result[0] * data_array[0] + result[1])
    diff_hor = data_array[0] - (data_array[1] - result[1]) / result[0]
    
    median_ver, p20_ver, p80_ver = hist_percentile_plot(fig3, [2,2,1], diff_ver, 'darkblue', 
                                                        '-', 20, r'$\Delta R_{\rm{HI}}$ [dex]')
    median_hor, p20_hor, p80_hor = hist_percentile_plot(fig3, [2,2,2], diff_hor, 'darkblue', 
                                                        '-', 20, r'$\Delta M_{\rm{HI}}$ [dex]')
    
    outlier_cut = [(diff_ver > median_ver - p20_ver) & (diff_ver < median_ver + p80_ver),
                  (diff_ver < median_ver - p20_ver),
                  (diff_ver > median_ver + p80_ver)]
    
    poplbl = ['Main Sample', r'$<20^{\rm{th}}$ Percentile', r'$>20^{\rm{th}}$ Percentile']
    
    for i in range(3):
      scat_mean_plot(fig3, 2, 2, 3, data_array[0][outlier_cut[i]], data_array[1][outlier_cut[i]], False, False, 
                    poplbl[i], xlbl, ylbl, colour[i],  'o', False)
      
    ax1 = fig3.add_subplot(2, 2, 3, facecolor = 'w')
    ax1.text(0.8, 0.05, '%.2f' % result[2], transform=ax1.transAxes, fontsize=12)
    ax1.plot(xfit, yfit, color=colour[0], linewidth=1, linestyle = '--', zorder=1)
    ax1.fill_between(xfit, yfit-dfit, yfit+dfit, alpha=0.25, edgecolor='none', zorder=-1, facecolor=colour[0])
    
    outlier_cut = [(diff_hor > median_hor - p20_hor) & (diff_hor < median_hor + p80_hor),
                  (diff_hor < median_hor - p20_hor),
                  (diff_hor > median_hor + p80_hor)]
    
    for i in range(3):
      scat_mean_plot(fig3, 2, 2, 4, data_array[0][outlier_cut[i]], data_array[1][outlier_cut[i]], False, False, 
                    poplbl[i], xlbl, ylbl, colour[i],  'o', False)
      
    ax1 = fig3.add_subplot(2, 2, 4, facecolor = 'w')
    ax1.text(0.8, 0.05, '%.2f' % result[2], transform=ax1.transAxes, fontsize=12)
    ax1.plot(xfit, yfit, color=colour[0], linewidth=1, linestyle = '--', zorder=1)
    ax1.fill_between(xfit, yfit-dfit, yfit+dfit, alpha=0.25, edgecolor='none', zorder=-1, facecolor=colour[0])
    
    plot_name = phase1_dir + 'PLOTS/PROPERTIES/HISM_SCATTER/vertical_horizontal_offsets.pdf'
    plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()
  
  
  if do_offset_jhi:
    fig3 = plt.figure(3, figsize=(6, 6))
    
    sub_finite = np.isfinite(mbar) & np.isfinite(jbar)
    
    data_bar = [mbar[subsample & sub_finite], jbar[subsample & sub_finite]]
    
    result_bar = linregress([data_bar[0], data_bar[1]])
    #print(np.round(result,3))
    xfit_bar   = np.arange(7.5, 12, 0.25)
    yfit_bar   = result_bar[0] * xfit_bar + result_bar[1]
    dfit_bar   = np.std(result_bar[0] * data_bar[0] + result_bar[1] - data_bar[1])
    offset_bar = result_bar[0] * data_bar[0] + result_bar[1] - data_bar[1]
    
    sub_finite = np.isfinite(mhi) & np.isfinite(jbar)
    
    data_hi = [mhi[subsample & sub_finite], jhi[subsample & sub_finite]]
    
    result_hi = linregress([data_hi[0], data_hi[1]])
    #print(np.round(result,3))
    xfit_hi   = np.arange(7.5, 11, 0.25)
    yfit_hi   = result_hi[0] * xfit_hi + result_hi[1]
    dfit_hi   = np.std(result_hi[0] * data_hi[0] + result_hi[1] - data_hi[1])
    offset_hi = result_hi[0] * data_hi[0] + result_hi[1] - data_hi[1]
    
    diff_jbar = data_bar[1] - (result_bar[0] * data_bar[0] + result_bar[1])
    diff_jhi  = data_hi[1] - (result_hi[0] * data_hi[0] + result_hi[1])
    
    
    
    median_bar, p20_bar, p80_bar = hist_percentile_plot(fig3, [2,2,1], diff_jbar, 'darkblue', 
                                                        '-', 20, r'$\Delta j_{\rm{bar}}$ [dex]')
    median_hi, p20_hi, p80_hi    = hist_percentile_plot(fig3, [2,2,2], diff_jhi, 'darkblue', 
                                                        '-', 20, r'$\Delta j_{\rm{HI}}$ [dex]')
    
    outlier_cut_bar = [(diff_jbar > median_bar - p20_bar) & (diff_jbar < median_bar + p80_bar),
                       (diff_jbar < median_bar - p20_bar),
                       (diff_jbar > median_bar + p80_bar)]
    
    xlbl = r'$\log(M_{\rm{bar}}/[\rm{M}_{\odot}])$'
    ylbl = r'$\log(j_{\rm{bar}}/[\rm{km/s}])$'
    
    for i in range(3):
      scat_mean_plot(fig3, 2, 2, 3, data_bar[0][outlier_cut_bar[i]], data_bar[1][outlier_cut_bar[i]], False, False, 
                    poplbl[i], xlbl, ylbl, colour[i],  'o', False)
      
      #result = linregress(data_array)
      #print(np.round(result,2))
      
    ax1 = fig3.add_subplot(2, 2, 3, facecolor = 'w')
    ax1.text(0.8, 0.05, '%.2f' % result_bar[2], transform=ax1.transAxes, fontsize=12)
    ax1.plot(xfit_bar, yfit_bar, color=colour[0], linewidth=1, linestyle = '--', zorder=1)
    ax1.fill_between(xfit_bar, yfit_bar-dfit_bar, yfit_bar+dfit_bar, alpha=0.25, 
                     edgecolor='none', zorder=-1, facecolor=colour[0])
    
    outlier_cut_hi = [(diff_jhi > median_hi - p20_hi) & (diff_jhi < median_hi + p80_hi),
                      (diff_jhi < median_hi - p20_hi),
                      (diff_jhi > median_hi + p80_hi)]
    
    xlbl = r'$\log(M_{\rm{HI}}/[\rm{M}_{\odot}])$'
    ylbl = r'$\log(j_{\rm{HI}}/[\rm{km/s}])$'
    
    for i in range(3):
      scat_mean_plot(fig3, 2, 2, 4, data_hi[0][outlier_cut_hi[i]], data_hi[1][outlier_cut_hi[i]], False, False, 
                    poplbl[i], xlbl, ylbl, colour[i],  'o', False)
      
      #result = linregress(data_array)
      #print(np.round(result,2))
      
    ax1 = fig3.add_subplot(2, 2, 4, facecolor = 'w')
    ax1.text(0.8, 0.05, '%.2f' % result_hi[2], transform=ax1.transAxes, fontsize=12)
    ax1.plot(xfit_hi, yfit_hi, color=colour[0], linewidth=1, linestyle = '--', zorder=1)
    ax1.fill_between(xfit_hi, yfit_hi-dfit_hi, yfit_hi+dfit_hi, alpha=0.25, 
                     edgecolor='none', zorder=-1, facecolor=colour[0])
    
    plot_name = phase1_dir + 'PLOTS/PROPERTIES/HISM_SCATTER/jbar_jhi_offsets.pdf'
    plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()
    
  
  if do_mh2_offset:
    fig4 = plt.figure(4, figsize=(6, 3))
    
    xlbl = r'$\Delta R_{\rm{HI}}$ [dex]'
    ylbl = r'$\log(M_{\rm{H2}}/M_{\rm{HI,disc}})$'
    zlbl = 'Number of Beams'
    
    scat_mean_plot(fig4, 1, 2, 1, diff_ver, h2onhi[subsample], False, False, 
                    'WALLABY', xlbl, ylbl, 'darkblue', 'o', False)
    
    #scat_col_simple_plot(fig4, 1, 2, 1, diff_ver, h2onhi[subsample], 
                        #radius_r_iso23[subsample]/15., 
                        #xlbl, ylbl, zlbl, 'viridis', 'o')
    
    #[np.isfinite(h2onhi[subsample])]
    
    ax1 = fig4.add_subplot(1, 2, 1, facecolor = 'w')
    #ax1.axvline(median_ver, linewidth=0.75, linestyle = '--', color = 'grey')
    ax1.axvline(median_ver - p20_ver, linewidth=1, linestyle = '--', color = 'grey')
    ax1.axvline(median_ver + p80_ver, linewidth=1, linestyle = '--', color = 'grey')
    ax1.set_xlim(-0.25,0.25)
    ax1.set_ylim(-2.2,0.5)
    
    xlbl = r'$\Delta M_{\rm{HI}}$ [dex]'
    ylbl = r'$\log(M_{\rm{H2}}/M_{\rm{HI,disc}})$'
    zlbl = r'$\log(N_{\rm{beams}})$'
    #scat_mean_plot(fig4, 1, 2, 2, diff_hor, h2onhi[subsample], False, False, 
                    #'WALLABY', xlbl, ylbl, 'darkblue', 'o', False)
    
    scat_col_simple_plot(fig4, 1, 2, 2, diff_hor, h2onhi[subsample], 
                        np.log10(radius_r_iso23[subsample]/15.), 
                        xlbl, ylbl, zlbl, 'rainbow', 'o')
    
    #scat_col_simple_plot(fig4, 1, 2, 2, diff_hor, h2onhi[subsample], 
                        #np.log10(radius_r_iso23[subsample]*axis_ratio_hi[subsample]/15.), 
                        #xlbl, ylbl, zlbl, 'rainbow', 'o')
    
    ax1 = fig4.add_subplot(1, 2, 2, facecolor = 'w')
    #ax1.axvline(median_hor, linewidth=0.75, linestyle = '--', color = 'grey')
    ax1.axvline(median_hor - p20_hor, linewidth=1, linestyle = '--', color = 'grey')
    ax1.axvline(median_hor + p80_hor, linewidth=1, linestyle = '--', color = 'grey')
    ax1.set_xlim(-0.25,0.25)
    ax1.set_ylim(-2.2,0.5)
    
    plot_name = phase1_dir + 'PLOTS/PROPERTIES/HISM_SCATTER/mh2_offset_bmaj.pdf'
    plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()
  
  
  if do_error_offset:
    fig4 = plt.figure(44, figsize=(6, 5))
    
    xlbl = r'$\Delta R_{\rm{HI}}$ [dex]'
    ylbl = r'$\sigma_{R_{\rm{HI}}}$ [dex]'
    zlbl = r'$\log(N_{\mathrm{beam}})$'
    
    #scat_mean_plot(fig4, 1, 1, 1, diff_ver, 
                   #np.log10(radius_hi_iso_kpc[subsample] + radius_hi_err_kpc[subsample]) - np.log10(radius_hi_iso_kpc[subsample]), False, False, 
                    #'WALLABY', xlbl, ylbl, 'darkblue', 'o', False)
    
    lg_rsigma = np.log10(radius_hi_iso_kpc + radius_hi_err_kpc) - np.log10(radius_hi_iso_kpc)
    
    #scat_col_simple_plot(fig4, 1, 1, 1, diff_ver[(radius_hi_iso[subsample] < 45)], 
                   #lg_rsigma[subsample][(radius_hi_iso[subsample] < 45)], 
                        #np.log10(radius_hi_iso[subsample][(radius_hi_iso[subsample] < 45)]/15.), 
                        #xlbl, ylbl, zlbl, 'rainbow', '.')
    
    scat_col_simple_plot2(fig4, 1, 1, 1, 
                          diff_ver[radius_hi_iso[subsample] > 45], 
                          lg_rsigma[subsample][radius_hi_iso[subsample] > 45], 
                          np.log10(radius_hi_iso[subsample][radius_hi_iso[subsample] > 45]/15.), 
                          [r'$>3$ beams', xlbl, ylbl, zlbl], ['rainbow', 'o', 'rainbow', 25], True, True)
    
    scat_col_simple_plot2(fig4, 1, 1, 1, 
                          diff_ver[radius_hi_iso[subsample] < 45], 
                          lg_rsigma[subsample][radius_hi_iso[subsample] < 45], 
                          np.log10(radius_hi_iso[subsample][radius_hi_iso[subsample] < 45]/15.), 
                          [r'$<3$ beams', xlbl, ylbl, zlbl], ['rainbow', 'o', 'none', 25], True, False)
    
    #scat_mean_plot(fig4, 1, 1, 1, diff_ver[(radius_hi_iso[subsample] < 45)], 
                   #lg_rsigma[subsample][(radius_hi_iso[subsample] < 45)], False, False, 
                    #'WALLABY', xlbl, ylbl, 'darkblue', '.', False)
    
    print(galaxies[subsample][(lg_rsigma[subsample] < 0.04) & (diff_ver < -0.04)])
    
    #[np.isfinite(h2onhi[subsample])]
    
    ax1 = fig4.add_subplot(1, 1, 1, facecolor = 'w')
    #ax1.axvline(median_ver, linewidth=0.75, linestyle = '--', color = 'grey')
    #ax1.axvline(median_ver - p20_ver, linewidth=1, linestyle = '--', color = 'grey')
    #ax1.axvline(median_ver + p80_ver, linewidth=1, linestyle = '--', color = 'grey')
    ax1.axvline(-0.04, linewidth=1, linestyle = '--', color = 'grey')
    ax1.axvline(0.04, linewidth=1, linestyle = '--', color = 'grey')
    ax1.set_xlim(-0.15,0.15)
    ax1.set_ylim(0,0.16)
    
    #xlbl = r'$\Delta M_{\rm{HI}}$ [dex]'
    #ylbl = r'$\sigma_{R_{\rm{HI}}}$ [dex]'
    #zlbl = r'$\log(N_{\rm{beams}})$'
    #scat_mean_plot(fig4, 1, 2, 2, diff_hor, 
                   #np.log10(radius_hi_iso_kpc[subsample] + radius_hi_err_kpc[subsample]) - np.log10(radius_hi_iso_kpc[subsample]), False, False, 
                    #'WALLABY', xlbl, ylbl, 'darkblue', 'o', False)
    
    ##scat_col_simple_plot(fig4, 1, 2, 2, diff_hor, h2onhi[subsample], 
                        ##np.log10(radius_r_iso23[subsample]/15.), 
                        ##xlbl, ylbl, zlbl, 'rainbow', 'o')
    
    ##scat_col_simple_plot(fig4, 1, 2, 2, diff_hor, h2onhi[subsample], 
                        ##np.log10(radius_r_iso23[subsample]*axis_ratio_hi[subsample]/15.), 
                        ##xlbl, ylbl, zlbl, 'rainbow', 'o')
    
    #ax1 = fig4.add_subplot(1, 2, 2, facecolor = 'w')
    ##ax1.axvline(median_hor, linewidth=0.75, linestyle = '--', color = 'grey')
    #ax1.axvline(median_hor - p20_hor, linewidth=1, linestyle = '--', color = 'grey')
    #ax1.axvline(median_hor + p80_hor, linewidth=1, linestyle = '--', color = 'grey')
    #ax1.set_xlim(-0.25,0.25)
    #ax1.set_ylim(-2.2,0.5)
    
    plot_name = phase1_dir + 'PLOTS/PROPERTIES/HISM_SCATTER/err_offset_beams.pdf'
    plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()
  
  
  if do_size_offset:
    fig4 = plt.figure(41, figsize=(6, 3))
    
    xlbl = r'$\Delta R_{\rm{HI}}$ [dex]'
    ylbl = r'$\Delta j_{\rm{bar}}$ [dex]'
    zlbl = r'$\log(R_{\rm{iso,HI}}/[\rm{kpc}])$'
    
    scat_mean_plot(fig4, 1, 2, 1, diff_ver[np.isfinite(jbar[subsample])], diff_jbar, False, False, 
                    'WALLABY', xlbl, ylbl, 'darkblue', 'o', False)
    
    #scat_col_simple_plot(fig4, 1, 2, 1, diff_ver[np.isfinite(jbar[subsample])], diff_jbar, 
                         #np.log10(radius_hi_iso_kpc[subsample][np.isfinite(jbar[subsample])]), 
                         #xlbl, ylbl, zlbl, 'viridis', 'o')
    
    ax1 = fig4.add_subplot(1, 2, 1, facecolor = 'w')
    #ax1.axvline(median_ver, linewidth=0.75, linestyle = '--', color = 'grey')
    ax1.axvline(median_ver - p20_ver, linewidth=1, linestyle = '--', color = 'grey')
    ax1.axvline(median_ver + p80_ver, linewidth=1, linestyle = '--', color = 'grey')
    
    #ax1.axhline(median_bar, linewidth=0.75, linestyle = '--', color = 'grey')
    ax1.axhline(median_bar - p20_bar, linewidth=1, linestyle = '--', color = 'grey')
    ax1.axhline(median_bar + p80_bar, linewidth=1, linestyle = '--', color = 'grey')
    ax1.set_xlim(-0.15,0.15)
    ax1.set_ylim(-0.5,0.5)
    
    xlbl = r'$\Delta R_{\rm{HI}}$ [dex]'
    ylbl = r'$\Delta j_{\rm{HI}}$ [dex]'
    #zlbl = r'$\log(R_{\rm{iso,HI}}/[\rm{kpc}])$'
    #zlbl = r'$\sigma_{R_{\rm{HI}}}$ [dex]'
    zlbl = r'$\log(M_{\rm{HI}}/[\rm{M}_{\odot}])$'
    
    #scat_mean_plot(fig4, 1, 2, 2, diff_ver[np.isfinite(jhi[subsample])], diff_jhi, False, False, 
                    #'WALLABY', xlbl, ylbl, 'darkblue', 'o', False)
    
    #scat_col_simple_plot(fig4, 1, 2, 2, diff_ver[np.isfinite(jhi[subsample])], diff_jhi, 
                         #np.log10(radius_hi_iso_kpc[subsample][np.isfinite(jhi[subsample])]), 
                         #xlbl, ylbl, zlbl, 'rainbow', 'o')
    
    #scat_col_simple_plot(fig4, 1, 2, 2, diff_ver[np.isfinite(jhi[subsample])], diff_jhi, 
                         #np.log10(radius_hi_iso_kpc[subsample][np.isfinite(jhi[subsample])] + radius_hi_err_kpc[subsample][np.isfinite(jhi[subsample])]) - np.log10(radius_hi_iso_kpc[subsample][np.isfinite(jhi[subsample])]), 
                         #xlbl, ylbl, zlbl, 'rainbow', 'o')
    
    scat_col_simple_plot(fig4, 1, 2, 2, diff_ver[np.isfinite(jhi[subsample])], diff_jhi, 
                          mhi[subsample][np.isfinite(jhi[subsample])], 
                          xlbl, ylbl, zlbl, 'rainbow', 'o')
    
    ax1 = fig4.add_subplot(1, 2, 2, facecolor = 'w')
    #ax1.axvline(median_ver, linewidth=0.75, linestyle = '--', color = 'grey')
    ax1.axvline(median_ver - p20_ver, linewidth=1, linestyle = '--', color = 'grey')
    ax1.axvline(median_ver + p80_ver, linewidth=1, linestyle = '--', color = 'grey')
    
    ax1.axvline(-0.04, linewidth=1, linestyle = '-', color = 'black')
    ax1.axvline(0.04, linewidth=1, linestyle = '-', color = 'black')
    
    #ax1.axhline(median_hi, linewidth=0.75, linestyle = '--', color = 'grey')
    ax1.axhline(median_hi - p20_hi, linewidth=1, linestyle = '--', color = 'grey')
    ax1.axhline(median_hi + p80_hi, linewidth=1, linestyle = '--', color = 'grey')
    ax1.set_xlim(-0.15,0.15)
    ax1.set_ylim(-0.5,0.5)
    
    plot_name = phase1_dir + 'PLOTS/PROPERTIES/HISM_SCATTER/size_v_offset_mhi.pdf'
    plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()
    
  if do_mass_offset:
    fig4 = plt.figure(4, figsize=(6, 3))
    
    xlbl = r'$\Delta M_{\rm{HI}}$ [dex]'
    ylbl = r'$\Delta j_{\rm{bar}}$ [dex]'
    
    scat_mean_plot(fig4, 1, 2, 1, diff_hor[np.isfinite(jbar[subsample])], diff_jbar, False, False, 
                    'WALLABY', xlbl, ylbl, 'darkblue', 'o', False)
    
    ax1 = fig4.add_subplot(1, 2, 1, facecolor = 'w')
    #ax1.axvline(median_ver, linewidth=0.75, linestyle = '--', color = 'grey')
    ax1.axvline(median_hor - p20_hor, linewidth=1, linestyle = '--', color = 'grey')
    ax1.axvline(median_hor + p80_hor, linewidth=1, linestyle = '--', color = 'grey')
    
    #ax1.axhline(median_bar, linewidth=0.75, linestyle = '--', color = 'grey')
    ax1.axhline(median_bar - p20_bar, linewidth=1, linestyle = '--', color = 'grey')
    ax1.axhline(median_bar + p80_bar, linewidth=1, linestyle = '--', color = 'grey')
    ax1.set_xlim(-0.25,0.25)
    ax1.set_ylim(-0.5,0.5)
    
    xlbl = r'$\Delta M_{\rm{HI}}$ [dex]'
    ylbl = r'$\Delta j_{\rm{HI}}$ [dex]'
    
    scat_mean_plot(fig4, 1, 2, 2, diff_hor[np.isfinite(jhi[subsample])], diff_jhi, False, False, 
                    'WALLABY', xlbl, ylbl, 'darkblue', 'o', False)
    
    ax1 = fig4.add_subplot(1, 2, 2, facecolor = 'w')
    #ax1.axvline(median_ver, linewidth=0.75, linestyle = '--', color = 'grey')
    ax1.axvline(median_hor - p20_hor, linewidth=1, linestyle = '--', color = 'grey')
    ax1.axvline(median_hor + p80_hor, linewidth=1, linestyle = '--', color = 'grey')
    
    #ax1.axhline(median_hi, linewidth=0.75, linestyle = '--', color = 'grey')
    ax1.axhline(median_hi - p20_hi, linewidth=1, linestyle = '--', color = 'grey')
    ax1.axhline(median_hi + p80_hi, linewidth=1, linestyle = '--', color = 'grey')
    ax1.set_xlim(-0.25,0.25)
    ax1.set_ylim(-0.5,0.5)
    
    plot_name = phase1_dir + 'PLOTS/PROPERTIES/HISM_SCATTER/mass_v_offset.pdf'
    plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()
  
  if do_h2_jhi_offset:
    fig6 = plt.figure(6, figsize=(4, 4))
    
    xlbl = r'$\Delta j_{\rm{HI}}$ [dex]'
    ylbl = r'$\log(M_{\rm{H2}}/M_{\rm{HI,disc}})$'
    
    scat_mean_plot(fig6, 1, 1, 1, diff_jhi, h2onhi[subsample][np.isfinite(jbar[subsample])], False, False, 
                    'WALLABY', xlbl, ylbl, 'darkblue', 'o', False)
    
    ax1 = fig6.add_subplot(1, 1, 1, facecolor = 'w')
    ##ax1.axvline(median_ver, linewidth=0.75, linestyle = '--', color = 'grey')
    #ax1.axvline(median_hor - p20_hor, linewidth=1, linestyle = '--', color = 'grey')
    #ax1.axvline(median_hor + p80_hor, linewidth=1, linestyle = '--', color = 'grey')
    
    #ax1.axhline(median_bar, linewidth=0.75, linestyle = '--', color = 'grey')
    ax1.axvline(median_hi - p20_hi, linewidth=1, linestyle = '--', color = 'grey')
    ax1.axvline(median_hi + p80_hi, linewidth=1, linestyle = '--', color = 'grey')
    ax1.set_xlim(-0.5,0.5)
    ax1.set_ylim(-2.2,0.5)
    
    plot_name = phase1_dir + 'PLOTS/PROPERTIES/HISM_SCATTER/mh2_v_jhi_offset.pdf'
    plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()
  
  
  if do_h2_jbar_plot:
    gal_subsample        = galaxies[subsample]
    ra_subsample         = sofia_ra[subsample]
    dec_subsample        = sofia_dec[subsample]
    ba_subsample         = axis_ratio_hi[subsample]
    err_subsample        = radius_hi_err_kpc[subsample]
    r_subsample          = radius_hi_iso_kpc[subsample]
    riso_subsample       = radius_hi_iso[subsample]
    mstar_subsample      = mstar[subsample]
    sfr_subsample        = sfr[subsample]
    ssfr_subsample       = ssfr[subsample]
    h2onhi_subsample     = h2onhi[subsample]
    mbar_subsample       = mbar[subsample]
    jbar_subsample       = jbar[subsample]
    fatm_subsample       = fatm[subsample]
    qstab_subsample      = qstab[subsample]
    envcl_subsample      = environ_class[subsample]
    h2onhi_tot_subsample = h2onhi_total[subsample]
    mhi_subsample        = mhi[subsample]
    jhi_subsample        = jhi[subsample]
    
    fig2 = plt.figure(2, figsize=(6, 6))#, facecolor = '#007D7D')
    xlbl = lbl_mstar
    ylbl = r'$\log(M_{\rm{H2}}/M_{\rm{HI}})$'
    for i in range(3):
      scat_mean_plot(fig2, 2, 2, 1, mstar_subsample[subsample_b[i]], h2onhi_subsample[subsample_b[i]], False, False, 
                    poplbl[i], xlbl, ylbl, colour[i], 'o', False)
    
    xfit = [9.18, 9.53, 9.88, 10.22, 10.59, 10.91, 11.20]
    yfit = [-0.903, -0.865, -0.660, -0.704, -0.662, -0.530, -0.578]
    ax1 = fig2.add_subplot(2, 2, 1, facecolor = 'w')
    ax1.plot(xfit, yfit, color='blue', linewidth=2, linestyle = '--', zorder=1)
      
    xlbl = r'$\log(M_{\rm{bar}}/[\rm{M}_{\odot}])$'
    ylbl = r'$\log(j_{\rm{bar}}/[\rm{km/s}])$'
    for i in range(3):
      scat_mean_plot(fig2, 2, 2, 2, mbar_subsample[subsample_b[i]], jbar_subsample[subsample_b[i]], False, False, 
                    poplbl[i], xlbl, ylbl, colour[i], 'o', False)
    
    xfit = np.arange(7.25, 12, 0.25)
    yfit = 0.66 * xfit - 3.1
    ax1 = fig2.add_subplot(2, 2, 2, facecolor = 'w')
    ax1.plot(xfit, yfit, color='grey', linewidth=2, linestyle = '--', zorder=2)
    
    sub_finite = np.isfinite(mbar_subsample) & np.isfinite(jbar_subsample)
    
    result = linregress([mbar_subsample[sub_finite], jbar_subsample[sub_finite]])
    print(np.round(result,3))
    xfit   = np.arange(7.5, 12, 0.25)
    yfit   = result[0] * xfit + result[1]
    dfit   = np.std(result[0] * mbar_subsample[sub_finite] + result[1] - jbar_subsample[sub_finite])
    offset = result[0] * mbar_subsample[sub_finite] + result[1] - jbar_subsample[sub_finite]
    print(dfit)
    ax1.plot(xfit, yfit, color='mediumvioletred', linewidth=2, linestyle = '--', zorder=2)
    
    xlbl = r'$\log(M_{\rm{HI}}/[\rm{M}_{\odot}])$'
    ylbl = r'$\log(j_{\rm{HI}}/[\rm{km/s}])$'
    for i in range(3):
      scat_mean_plot(fig2, 2, 2, 3, mhi_subsample[subsample_b[i]], jhi_subsample[subsample_b[i]], False, False, 
                    poplbl[i], xlbl, ylbl, colour[i], 'o', False)
    
    xfit = np.arange(7.5, 11, 0.25)
    yfit = 0.66 * xfit - 3.1
    ax1 = fig2.add_subplot(2, 2, 3, facecolor = 'w')
    ax1.plot(xfit, yfit, color='grey', linewidth=2, linestyle = '--', zorder=2)
    
    sub_finite = np.isfinite(mhi_subsample) & np.isfinite(jhi_subsample)
    
    result = linregress([mhi_subsample[sub_finite], jhi_subsample[sub_finite]])
    print(np.round(result,3))
    xfit   = np.arange(7.5, 11, 0.25)
    yfit   = result[0] * xfit + result[1]
    dfit   = np.std(result[0] * mhi_subsample[sub_finite] + result[1] - jhi_subsample[sub_finite])
    offset = result[0] * mhi_subsample[sub_finite] + result[1] - jhi_subsample[sub_finite]
    print(dfit)
    ax1.plot(xfit, yfit, color='mediumvioletred', linewidth=2, linestyle = '--', zorder=2)
    
    #xlbl = r'$q$'
    #ylbl = r'$f_{\rm{atm}}$'
    #for i in range(3):
      #scat_mean_plot(fig4, 2, 2, 4, np.log10(qstab_subsample[subsample_b[i]]), 
                    #np.log10(fatm_subsample[subsample_b[i]]), False, False, 
                    #poplbl[i], xlbl, ylbl, colour[i], 'o', False)
      
    plot_name = phase1_dir + 'PLOTS/PROPERTIES/HISM_SCATTER/mh2_jbar_scatter.pdf'
    plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()
  
  
  
  
  
# ================================= #
# ======== Scatter Plots ========== #
# ================================= #
if do_paper_plots:
  do_iso_eff            = False
  do_iso_mstar          = False
  do_iso_mstar_sratio   = False
  do_iso_mstar_hifrac   = False
  
  do_iso_muhi           = False
  do_iso_sratio         = True
  do_eff_muhi_nuvr      = False
  do_eff_sratio_nuvr    = False
  do_eff_sratio_mustar  = False
  
  do_delta_size         = False
  
  do_iso_muhi_ssfr      = False
  do_iso_muhi_mstar     = False
  do_iso_muhi_dsfms     = False
  do_iso_muhi_cindex    = False
  do_iso_sratio_ssfr    = False
  do_iso_sratio_dsfms   = False
  
  def linear_func(x, m, b):
      return m * x + b
    
  fig4 = plt.figure(5, figsize=(7, 7))#, facecolor = '#007D7D')
  
  
  if do_iso_eff:
    #xlbl         = lbl_mstar
    #ylbl         = r'Radius [kpc]'
    
    xpar    = [mstar, mstar, mhi, mhi]
    ypar    = [radius_r_iso25_kpc, radius_r_50_kpc, radius_hi_iso_kpc, radius_hi_eff_kpc]
    limpar  = no_galex
    colour  = ['peru', 'sandybrown', 'darkblue', 'royalblue']
    poplbl  = ['Cluster', 'Infall', 'Field']
    
    xlbl    = [lbl_mstar, lbl_mstar, lbl_mhi, lbl_mhi]
    
    ylbl    = [r'$\log(R_{\rm{iso25,r}}/[\rm{kpc}])$', r'$\log(R_{\rm{50,r}}/[\rm{kpc}])$', 
               r'$\log(R_{\rm{iso,HI}}/[\rm{kpc}])$', r'$\log(R_{\rm{50,HI}}/[\rm{kpc}])$']
    
    row, column = 2, 2
    
    print('m, c, rval, pval, stderr')
    
    #yfit = []
    #dfit = []
    
    fig5 = plt.figure(55, figsize=(9, 6))
    
    #subsample  = ((radius_hi_iso > 30) & np.isfinite(mhi) & np.isfinite(mstar) & 
                  #np.isfinite(xpar[i]) & np.isfinite(ypar[i]) & (mstar > 6.5))
      
    subsample  = ((radius_hi_eff > 20) & np.isfinite(mhi) & np.isfinite(mstar) & 
                  np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  np.isfinite(radius_r_iso25_kpc) & np.isfinite(radius_r_50_kpc) & 
                  np.isfinite(radius_hi_iso_kpc) & np.isfinite(radius_hi_eff_kpc) & (mstar > 6.5))
    
    #subsample  = ((radius_hi_iso > 20) & np.isfinite(mhi) & np.isfinite(mstar) &
                  #np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  #np.isfinite(surfden_hi_iso) & np.isfinite(mu_hi) & 
                  #np.isfinite(radius_r_iso25_kpc) & 
                  #np.isfinite(radius_hi_iso_kpc) & (mstar > 6.5))
    
    # =============== #
    data_array = [xpar[1][subsample], np.log10(ypar[1][subsample])]
    print(len(data_array[0]))
    offset_r_eff = size_mass_relation_plot(fig5, 2, 2, 1, data_array, 'Effective', 
                                           xlbl[0], r'$\log(R_{\rm{r}}/\rm{kpc})$', 
                                           [colour[1], colour[1], 'none'], 'o')
    
    # =============== #
    data_array = [xpar[0][subsample], np.log10(ypar[0][subsample])]
    offset_r_iso = size_mass_relation_plot(fig5, 2, 2, 1, data_array, 'Isophotal', 
                                           xlbl[0], r'$\log(R_{\rm{r}}/\rm{kpc})$', 
                                           [colour[0], colour[0], colour[0]], 'o')
    
    ax1 = fig5.add_subplot(2, 2, 1, facecolor = 'w')
      
    ax1.set_xlim(6.8,11.2)
    ax1.set_ylim(-0.2,1.9)
    
    # =============== #
    data_array = [xpar[3][subsample], np.log10(ypar[3][subsample])]
    offset_h_eff = size_mass_relation_plot(fig5, 2, 2, 2, data_array, 'Effective', 
                                           xlbl[2], r'$\log(R_{\rm{HI}}/\rm{kpc})$', 
                                           [colour[3], colour[3], 'none'], 'o')
    
    # =============== #
    data_array = [xpar[2][subsample], np.log10(ypar[2][subsample])]
    offset_h_iso = size_mass_relation_plot(fig5, 2, 2, 2, data_array, 'Isodensity', 
                                           xlbl[2], r'$\log(R_{\rm{HI}}/\rm{kpc})$', 
                                           [colour[2], colour[2], colour[2]], 'o')
    
    
    ax1 = fig5.add_subplot(2, 2, 2, facecolor = 'w')
    
    a_16, b_16 = 0.506, -3.293
    #a_16, b_16 = 0.51, -3.29
    d_16       = 0.06
    xfit       = np.arange(7, 11.25, 0.25)
    yfit_16    = a_16 * xfit + b_16 - np.log10(2.)
    print(a_16, b_16 - np.log10(2.))
    ax1.plot(xfit, yfit_16, color='magenta', linewidth=1, linestyle = '--', zorder=1)
    #ax1.fill_between(xfit, yfit_16-d_16, yfit_16+d_16, alpha=0.5, 
                     #edgecolor='none', zorder=3, facecolor='grey')
    
    ax1.set_xlim(6.8,11.2)
    ax1.set_ylim(-0.2,1.9)
    
    #plt.show()
    #plot_name = phase1_dir + 'PLOTS/PAPER3/size_mass_histar_iso.pdf'
    #fig5.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    #plt.clf()
    
    #fig6 = plt.figure(6, figsize=(9, 3.5))
    
    offset  = [offset_r_iso, offset_r_eff, offset_h_iso, offset_h_eff]
    
    #xpar    = radius_hi_iso_kpc
    #ypar    = offset
    
    xlbl    = lbl_mstar #r'$\log(R_{\rm{iso,r}}/\rm{kpc})$'
    ylbl    = r'$\Delta R_{\rm{r}}$ [dex]'
    
    data_array = [xpar[0][subsample], offset[1]]
    scatter_outlier_plot(fig5, 2, 2, 3, 
                         data_array[0], data_array[1], 
                         'Effective', xlbl, ylbl, ['sandybrown', 'o', 'none', 25], False)
    
    data_array = [xpar[0][subsample], offset[0]]
    scatter_outlier_plot(fig5, 2, 2, 3, 
                         data_array[0], data_array[1], 
                         'Isodensity', xlbl, ylbl, ['peru', 'o', 'peru', 25], False)
    
    ax1 = fig5.add_subplot(2, 2, 3, facecolor = 'w')
    ax1.axhline(0, color = 'grey', linestyle = '--', zorder = -1)
    #ax1.set_xlim(-0.5, 0.5)
    ax1.set_xlim(6.8,11.2)
    ax1.set_ylim(-0.5, 0.5)
    
    xlbl    = lbl_mhi #r'$\log(R_{\rm{iso,HI}}/\rm{kpc})$'
    ylbl    = r'$\Delta R_{\rm{HI}}$ [dex]'
    
    data_array = [xpar[2][subsample], offset[3]]
    scatter_outlier_plot(fig5, 2, 2, 4, 
                         data_array[0], data_array[1], 
                         'Effective', xlbl, ylbl, ['royalblue', 'o', 'none', 25], False)
    
    data_array = [xpar[2][subsample], offset[2]]
    scatter_outlier_plot(fig5, 2, 2, 4, 
                         data_array[0], data_array[1], 
                         'Isodensity', xlbl, ylbl, ['darkblue', 'o', 'darkblue', 25], False)
    
    ax1 = fig5.add_subplot(2, 2, 4, facecolor = 'w')
    ax1.axhline(0, color = 'grey', linestyle = '--', zorder = -1)
    #ax1.set_xlim(-0.5, 0.5)
    ax1.set_xlim(6.8,11.2)
    ax1.set_ylim(-0.5, 0.5)
    
    plot_name = phase1_dir + 'PLOTS/PAPER3/size_mass_residual_eff20_fitting.pdf'
    plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()
    
  
  if do_iso_mstar:
    #xlbl         = lbl_mstar
    #ylbl         = r'Radius [kpc]'
    
    xpar    = [mstar, mstar]
    ypar    = [radius_r_iso25_kpc, radius_hi_iso_kpc]
    limpar  = no_galex
    colour  = ['peru', 'darkblue']
    poplbl  = ['Cluster', 'Infall', 'Field']
    
    xlbl    = [lbl_mstar, lbl_mstar]
    
    ylbl    = [r'$\log(R_{\rm{iso25,r}}/[\rm{kpc}])$', 
               r'$\log(R_{\rm{iso,HI}}/[\rm{kpc}])$']
    
    print('m, c, rval, pval, stderr')
    
    #yfit = []
    #dfit = []
    
    fig5 = plt.figure(55, figsize=(6, 6))
    
    #subsample  = ((radius_hi_iso > 30) & np.isfinite(mhi) & np.isfinite(mstar) & 
                  #np.isfinite(xpar[i]) & np.isfinite(ypar[i]) & (mstar > 6.5))
      
    subsample  = ((radius_hi_iso > 30) & np.isfinite(mhi) & np.isfinite(mstar) & 
                  np.isfinite(radius_r_iso25_kpc) & 
                  np.isfinite(radius_hi_iso_kpc) & (mstar > 6.5))
    
    
    # =============== #
    #data_array = [xpar[1][subsample], np.log10(ypar[1][subsample])]
    #print(len(data_array[0]))
    #size_mass_relation_plot(fig5, 1, 2, 1, data_array, 'Effective', 
                            #xlbl[0], r'$\log(R_{\rm{r}}/[\rm{kpc}])$', 
                            #[colour[1], colour[1], 'none'], 'o')
    
    # =============== #
    data_array = [xpar[0][subsample], np.log10(ypar[0][subsample])]
    size_mass_relation_plot(fig5, 1, 1, 1, data_array, r'$r$-band', 
                            xlbl[0], r'$\log(R_{\rm{iso}}/[\rm{kpc}])$', 
                            [colour[0], colour[0], colour[0]], 'o')
    
    # =============== #
    #data_array = [xpar[3][subsample], np.log10(ypar[3][subsample])]
    #size_mass_relation_plot(fig5, 1, 2, 2, data_array, 'Effective', 
                            #xlbl[2], r'$\log(R_{\rm{HI}}/[\rm{kpc}])$', 
                            #[colour[3], colour[3], 'none'], 'o')
    
    # =============== #
    data_array = [xpar[1][subsample], np.log10(ypar[1][subsample])]
    size_mass_relation_plot(fig5, 1, 1, 1, data_array, 'HI', 
                            xlbl[1], r'$\log(R_{\rm{iso}}/[\rm{kpc}])$', 
                            [colour[1], colour[1], 'none'], 'o')
    
    #plt.show()
    plot_name = phase1_dir + 'PLOTS/PAPER3/size_mstar.pdf'
    fig5.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()
  
  if do_iso_mstar_sratio:
    print('m, c, rval, pval, stderr')
    
    fig5 = plt.figure(55, figsize=(9, 4))
    
    #subsample  = ((radius_hi_iso > 30) & np.isfinite(mhi) & np.isfinite(mstar) & 
                  #np.isfinite(xpar[i]) & np.isfinite(ypar[i]) & (mstar > 6.5))
      
    subsample  = ((radius_hi_iso > 30) & np.isfinite(mhi) & np.isfinite(mstar) & 
                  np.isfinite(radius_r_iso25_kpc) & 
                  np.isfinite(radius_hi_iso_kpc) & (mstar > 6.5))
    
    iso_size_ratio = np.log10(radius_hi_iso / radius_r_iso25)
    
    xlbl    = lbl_mstar
    ylbl    = r'$\log(R_{\rm{iso}}/[\rm{kpc}])$'
    zlbl    = r'$\log(R_{\rm{iso,HI}}/R_{\rm{iso25,r}})$'
    
    xpar    = mstar[subsample]
    ypar    = np.log10(radius_r_iso25_kpc[subsample])
    zpar    = iso_size_ratio[subsample]
    
    scat_col_simple_plot2(fig5, 1, 2, 1, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'$r$-band', xlbl, ylbl, zlbl], 
                          ['rainbow_r', 'o', 'rainbow_r', 25, -0.2, 0.8], True, True)
    
    xpar    = mstar[subsample]
    ypar    = np.log10(radius_hi_iso_kpc[subsample])
    zpar    = iso_size_ratio[subsample]

    scat_col_simple_plot2(fig5, 1, 2, 2, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'HI', xlbl, ylbl, zlbl],
                          ['rainbow_r', 'o', 'rainbow_r', 25, -0.2, 0.8], True, True)
    
    #plt.show()
    plot_name = phase1_dir + 'PLOTS/PAPER3/size_mstar_sratio.pdf'
    fig5.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()
    
  if do_iso_mstar_hifrac:
    print('m, c, rval, pval, stderr')
    
    fig5 = plt.figure(56, figsize=(9, 4))
    
    #subsample  = ((radius_hi_iso > 30) & np.isfinite(mhi) & np.isfinite(mstar) & 
                  #np.isfinite(xpar[i]) & np.isfinite(ypar[i]) & (mstar > 6.5))
      
    subsample  = ((radius_hi_iso > 30) & np.isfinite(mhi) & np.isfinite(mstar) & 
                  np.isfinite(radius_r_iso25_kpc) & 
                  np.isfinite(radius_hi_iso_kpc) & (mstar > 6.5))
    
    iso_size_ratio = np.log10(radius_hi_iso / radius_r_iso25)
    
    xlbl    = lbl_mstar
    ylbl    = r'$\log(R_{\rm{iso}}/[\rm{kpc}])$'
    zlbl    = lbl_hifrac
    
    xpar    = mstar[subsample]
    ypar    = np.log10(radius_r_iso25_kpc[subsample])
    zpar    = hifrac[subsample]
    
    scat_col_simple_plot2(fig5, 1, 2, 1, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'$r$-band', xlbl, ylbl, zlbl], 
                          ['rainbow_r', 'o', 'rainbow_r', 25, -1.5, 1.6], True, True)
    
    xpar    = mstar[subsample]
    ypar    = np.log10(radius_hi_iso_kpc[subsample])
    zpar    = hifrac[subsample]

    scat_col_simple_plot2(fig5, 1, 2, 2, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'HI', xlbl, ylbl, zlbl],
                          ['rainbow_r', 'o', 'rainbow_r', 25, -1.5, 1.6], True, True)
    
    #plt.show()
    plot_name = phase1_dir + 'PLOTS/PAPER3/size_mstar_hifrac.pdf'
    fig5.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()
  
  if do_iso_muhi:
    fig5 = plt.figure(59, figsize=(11, 8))
    
    subsample  = ((radius_hi_iso > 20) & np.isfinite(mhi) & np.isfinite(mstar) & 
                  np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  np.isfinite(surfden_hi_iso) & np.isfinite(mu_hi) & 
                  np.isfinite(radius_r_iso25_kpc) & np.isfinite(radius_r_50_kpc) & 
                  np.isfinite(radius_hi_eff_kpc) &
                  np.isfinite(radius_hi_iso_kpc) & (mstar > 6.5))
    
    iso_size_ratio = np.log10(radius_hi_iso / radius_r_iso25)
    
    xlbl    = lbl_mstar
    ylbl    = r'$\log(\mu_{\rm{iso,HI}}/[\rm{M_{\odot}\,pc^{-2}}])$'
    zlbl    = 'Radius / Beams'
    
    xpar    = mstar[subsample]
    #xpar    = delta_sfms[subsample]
    ypar    = np.log10(surfden_hi_iso[subsample])
    zpar    = radius_hi_iso[subsample]/30.
    
    print(len(xpar))
    
    print(len(mstar[subsample & (radius_hi_iso > 60) & (mstar < 9.5)])/len(mstar[subsample & (mstar < 9.5)]))
    print(len(mstar[subsample & (radius_hi_iso > 60) & (mstar > 9.5)])/len(mstar[subsample & (mstar > 9.5)]))
    
    #cmap = plt.get_cmap('rainbow', 5)
    cmap = (mpl.colors.ListedColormap(['silver', 'silver', 'tab:cyan', 'dodgerblue', 'navy']))
    
    scat_col_simple_plot4(fig5, 2, 3, 1, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'Isophotal', xlbl, ylbl, zlbl], 
                          [cmap, 'o', cmap, 20, 0, 5], False, False)
    
    bins                   = np.arange(7.5, 11.25, 0.5)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    print(np.round(stats.pearsonr(xpar[xpar>9], ypar[xpar>9])[0], 2), stats.pearsonr(xpar[xpar>9], ypar[xpar>9])[1])
    
    scatter_error_plot(fig5, 2, 3, 1, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    ax1 = fig5.add_subplot(2, 3, 1, facecolor = 'w')
    #ax1.set_xlim(-11.25, -8.5)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    ax1.text(7, 0.9, 'Isodensity', fontsize = 14)
    ax1.text(0.7, 0.9, r'$%.2f$' % np.round(stats.pearsonr(xpar, ypar)[0], 2), 
             fontsize = 14, transform=ax1.transAxes)
    
    xlbl    = lbl_mustar
    #xlbl    = r'$\Delta$\,SFMS [dex]'
    ylbl    = r'$\log(\mu_{\rm{iso,HI}}/[\rm{M_{\odot}\,pc^{-2}}])$'
    zlbl    = 'Radius / Beams'
    
    xpar    = mu_star[subsample]
    #xpar    = delta_sfms[subsample]
    ypar    = np.log10(surfden_hi_iso[subsample])
    zpar    = radius_hi_iso[subsample]/30.
    
    print(len(xpar))
    
    #cmap = plt.get_cmap('rainbow', 5)
    
    scat_col_simple_plot4(fig5, 2, 3, 2, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'Isophotal', xlbl, ylbl, zlbl], 
                          [cmap, 'o', cmap, 20, 0, 5], False, True)
    
    bins                   = np.arange(6.2, 9, 0.5)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    print(np.round(stats.pearsonr(xpar[xpar>7], ypar[xpar>7])[0], 2), stats.pearsonr(xpar[xpar>7], ypar[xpar>7])[1])
    
    scatter_error_plot(fig5, 2, 3, 2, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    ax1 = fig5.add_subplot(2, 3, 2, facecolor = 'w')
    ax1.set_xlim(5.5, 9)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    ax1.text(0.7, 0.9, r'$%.2f$' % np.round(stats.pearsonr(xpar, ypar)[0], 2), 
             fontsize = 14, transform=ax1.transAxes)
    
    xlbl    = lbl_ssfr
    ylbl    = r'$\log(\mu_{\rm{iso,HI}}/[\rm{M_{\odot}\,pc^{-2}}])$'
    zlbl    = 'Radius / Beams' #lbl_mstar 
    
    xpar    = ssfr[subsample & (sfr_uplim == True)]
    ypar    = np.log10(surfden_hi_iso[subsample & (sfr_uplim == True)])
    zpar    = radius_hi_iso[subsample & (sfr_uplim == True)]/30.
    
    print(len(xpar))
    
    #cmap = plt.get_cmap('rainbow', 5)
    
    scat_col_simple_plot4(fig5, 2, 3, 4, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'Isophotal', xlbl, ylbl, zlbl], 
                          [cmap, r'$\leftarrow$', cmap, 60, 0, 5], False, False)
    
    xpar    = ssfr[subsample & (sfr_uplim == False)]
    ypar    = np.log10(surfden_hi_iso[subsample & (sfr_uplim == False)])
    zpar    = radius_hi_iso[subsample & (sfr_uplim == False)]/30.
    
    print(len(xpar))
    
    #cmap = plt.get_cmap('rainbow', 5)
    
    scat_col_simple_plot4(fig5, 2, 3, 4, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'Isophotal', xlbl, ylbl, zlbl], 
                          [cmap, 'o', cmap, 20, 0, 5], False, False)
    
    bins                   = np.arange(-10.5, -8.75, 0.25)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    #scatter_error_plot(fig5, 2, 3, 4, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          #'Medians', xlbl, ylbl, ['red', 'd', 'red', 7.5], False)
    
    
    
    xpar    = ssfr[subsample]
    ypar    = np.log10(surfden_hi_iso[subsample])
    zpar    = radius_hi_iso[subsample]/30.
    
    print(len(xpar))
    
    bins                   = np.arange(-10.5, -8.75, 0.25)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 2, 3, 4, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    ax1 = fig5.add_subplot(2, 3, 4, facecolor = 'w')
    ax1.set_xlim(-11.25, -8.5)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    ax1.text(0.7, 0.9, r'$%.2f$' % np.round(stats.pearsonr(xpar, ypar)[0], 2), 
             fontsize = 14, transform=ax1.transAxes)
    
    xlbl    = lbl_nuvr
    ylbl    = r'$\log(\mu_{\rm{iso,HI}}/[\rm{M_{\odot}\,pc^{-2}}])$'
    zlbl    = 'Radius / Beams' #lbl_mstar
    
    xpar    = nuvr[subsample]
    ypar    = np.log10(surfden_hi_iso[subsample])
    zpar    = radius_hi_iso[subsample]/30.
    
    print(len(xpar))
    
    #cmap = plt.get_cmap('rainbow', 5)
    
    scat_col_simple_plot4(fig5, 2, 3, 5, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'Isophotal', xlbl, ylbl, zlbl], 
                          [cmap, 'o', cmap, 20, 0, 5], False, True)
    
    bins                   = np.arange(0.7, 3.5, 0.5)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 2, 3, 5, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    ax1 = fig5.add_subplot(2, 3, 5, facecolor = 'w')
    ax1.set_xlim(-0.1, 6)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    ax1.text(0.7, 0.9, r'$%.2f$' % np.round(stats.pearsonr(xpar, ypar)[0], 2), 
             fontsize = 14, transform=ax1.transAxes)
    
    xlbl    = r'$\Delta\,\rm{SFMS}$'
    #xlbl    = r'$\Delta$\,SFMS [dex]'
    ylbl    = r'$\log(\mu_{\rm{isoHI}}/[\rm{M_{\odot}\,pc^{-2}}])$'
    zlbl    = 'Radius / Beams'
    
    xpar    = delta_sfms[subsample & (sfr_uplim == True)]
    ypar    = np.log10(surfden_hi_iso[subsample & (sfr_uplim == True)])
    zpar    = radius_hi_iso[subsample & (sfr_uplim == True)]/30.
    
    print(len(xpar))
    
    scat_col_simple_plot4(fig5, 2, 3, 6, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'ISO', xlbl, ylbl, zlbl], 
                          [cmap, r'$\leftarrow$', cmap, 60, 0, 5], False, False)
    
    xpar    = delta_sfms[subsample & (sfr_uplim == False)]
    ypar    = np.log10(surfden_hi_iso[subsample & (sfr_uplim == False)])
    zpar    = radius_hi_iso[subsample & (sfr_uplim == False)]/30.
    
    print(len(xpar))
    
    scat_col_simple_plot4(fig5, 2, 3, 6, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'ISO', xlbl, ylbl, zlbl], 
                          [cmap, 'o', cmap, 20, 0, 5], False, False)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    
    
    xpar    = delta_sfms[subsample]
    ypar    = np.log10(surfden_hi_iso[subsample])
    zpar    = radius_hi_iso[subsample]/30.
    
    print(len(xpar))
    
    bins                   = np.arange(-0.7, 1, 0.3)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 2, 3, 6, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    ax1 = fig5.add_subplot(2, 3, 6, facecolor = 'w')
    ax1.set_xlim(-1, 1.25)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    ax1.text(0.7, 0.9, r'$%.2f$' % np.round(stats.pearsonr(xpar, ypar)[0], 2), 
             fontsize = 14, transform=ax1.transAxes)
    
    plot_name = phase1_dir + 'PLOTS/PAPER3/iso_muhi_log20.pdf'
    fig5.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()
  
  if do_iso_sratio:
    fig5 = plt.figure(56, figsize=(11, 8))
    
    iso_size_ratio = np.log10(radius_hi_iso / radius_r_iso25)
      
    subsample  = ((radius_hi_iso > 20) & np.isfinite(mhi) & np.isfinite(mstar) & 
                  np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  np.isfinite(surfden_hi_iso) & np.isfinite(mu_hi) & 
                  np.isfinite(radius_r_iso25_kpc) & np.isfinite(radius_r_50_kpc) & 
                  np.isfinite(radius_hi_eff_kpc) &
                  np.isfinite(radius_hi_iso_kpc) & (mstar > 6.5))
    
    
    xlbl    = lbl_mstar
    ylbl    = r'$\log(R_{\rm{iso,HI}}/R_{\rm{iso,r}})$'
    zlbl    = 'Radius / Beams'
    
    xpar    = mstar[subsample]
    #xpar    = delta_sfms[subsample]
    ypar    = iso_size_ratio[subsample]
    zpar    = radius_hi_iso[subsample]/30.
    
    print(len(xpar))
    
    #cmap = plt.get_cmap('rainbow', 4)
    
    cmap = (mpl.colors.ListedColormap(['silver', 'silver', 'tab:cyan', 'dodgerblue', 'navy']))
    #.with_extremes(over='0.25', under='0.75'))
    
    scat_col_simple_plot4(fig5, 2, 3, 1, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'ISO', xlbl, ylbl, zlbl], 
                          [cmap, 'o', cmap, 20, 0, 5], False, False)
    
    bins                   = np.arange(7.5, 11.25, 0.5)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 2, 3, 1, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    ax1 = fig5.add_subplot(2, 3, 1, facecolor = 'w')
    #ax1.set_xlim(-11.25, -8.5)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    ax1.text(7, 0.9, 'Isodensity', fontsize = 14)
    ax1.text(0.7, 0.9, r'$%.2f$' % np.round(stats.pearsonr(xpar, ypar)[0], 2), 
             fontsize = 14, transform=ax1.transAxes)
    
    xlbl    = lbl_mustar
    ylbl    = r'$\log(R_{\rm{iso,HI}}/R_{\rm{iso,r}})$'
    zlbl    = 'Radius / Beams'
    
    xpar    = mu_star[subsample]
    ypar    = iso_size_ratio[subsample]
    zpar    = radius_hi_iso[subsample]/30.
    
    print(len(xpar))
    
    #cmap = plt.get_cmap('rainbow', 5)
    
    scat_col_simple_plot4(fig5, 2, 3, 2, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'ISO', xlbl, ylbl, zlbl], 
                          [cmap, 'o', cmap, 20, 0, 5], False, True)
    
    bins                   = np.arange(6.2, 9, 0.5)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 2, 3, 2, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    ax1 = fig5.add_subplot(2, 3, 2, facecolor = 'w')
    ax1.set_xlim(5.5, 9)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    ax1.text(0.7, 0.9, r'$%.2f$' % np.round(stats.pearsonr(xpar, ypar)[0], 2), 
             fontsize = 14, transform=ax1.transAxes)
    
    xlbl    = lbl_ssfr
    ylbl    = r'$\log(R_{\rm{iso,HI}}/R_{\rm{iso,r}})$'
    zlbl    = 'Radius / Beams'
    
    xpar    = ssfr[subsample & (sfr_uplim == True)]
    ypar    = iso_size_ratio[subsample & (sfr_uplim == True)]
    zpar    = radius_hi_iso[subsample & (sfr_uplim == True)]/30.
    
    print(len(xpar))
    
    #cmap = plt.get_cmap('rainbow', 5)
    
    scat_col_simple_plot4(fig5, 2, 3, 4, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'ISO', xlbl, ylbl, zlbl], 
                          [cmap, r'$\leftarrow$', cmap, 60, 0, 5], False, False)
    
    xpar    = ssfr[subsample & (sfr_uplim == False)]
    ypar    = iso_size_ratio[subsample & (sfr_uplim == False)]
    zpar    = radius_hi_iso[subsample & (sfr_uplim == False)]/30.
    
    print(len(xpar))
    
    scat_col_simple_plot4(fig5, 2, 3, 4, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'ISO', xlbl, ylbl, zlbl], 
                          [cmap, 'o', cmap, 20, 0, 5], False, False)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    xpar    = ssfr[subsample]
    ypar    = iso_size_ratio[subsample]
    zpar    = radius_hi_iso[subsample]/30.
    
    print(len(xpar))
    
    bins                   = np.arange(-10.5, -8.75, 0.25)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 2, 3, 4, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    ax1 = fig5.add_subplot(2, 3, 4, facecolor = 'w')
    ax1.set_xlim(-11.25, -8.5)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    ax1.text(0.7, 0.9, r'$%.2f$' % np.round(stats.pearsonr(xpar, ypar)[0], 2), 
             fontsize = 14, transform=ax1.transAxes)
    
    xlbl    = lbl_nuvr
    ylbl    = r'$\log(R_{\rm{iso,HI}}/R_{\rm{iso,r}})$'
    zlbl    = 'Radius / Beams' #lbl_mstar #r'$\log(R_{\rm{iso,HI}}/R_{\rm{iso25,r}})$'
    
    xpar    = nuvr[subsample]
    #xpar    = delta_sfms[subsample]
    ypar    = iso_size_ratio[subsample]
    zpar    = radius_hi_iso[subsample]/30. #mstar[subsample]
    
    print(len(xpar))
    
    #cmap = plt.get_cmap('rainbow', 5)
    
    scat_col_simple_plot4(fig5, 2, 3, 5, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'ISO', xlbl, ylbl, zlbl], 
                          [cmap, 'o', cmap, 20, 0, 5], False, False)
    
    bins                   = np.arange(0.5, 3.5, 0.5)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 2, 3, 5, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    ax1 = fig5.add_subplot(2, 3, 5, facecolor = 'w')
    ax1.set_xlim(-0.1, 6)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    ax1.text(0.7, 0.9, r'$%.2f$' % np.round(stats.pearsonr(xpar, ypar)[0], 2), 
             fontsize = 14, transform=ax1.transAxes)
    
    xlbl    = r'$\Delta\,\rm{SFMS}$'
    ylbl    = r'$\log(R_{\rm{iso,HI}}/R_{\rm{iso,r}})$'
    zlbl    = 'Radius / Beams'
    
    xpar    = delta_sfms[subsample & (sfr_uplim == True)]
    ypar    = iso_size_ratio[subsample & (sfr_uplim == True)]
    zpar    = radius_hi_iso[subsample & (sfr_uplim == True)]/30.
    
    print(len(xpar))
    
    scat_col_simple_plot4(fig5, 2, 3, 6, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'ISO', xlbl, ylbl, zlbl], 
                          [cmap, r'$\leftarrow$', cmap, 60, 0, 5], False, False)
    
    xpar    = delta_sfms[subsample & (sfr_uplim == False)]
    ypar    = iso_size_ratio[subsample & (sfr_uplim == False)]
    zpar    = radius_hi_iso[subsample & (sfr_uplim == False)]/30.
    
    print(len(xpar))
    
    scat_col_simple_plot4(fig5, 2, 3, 6, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'ISO', xlbl, ylbl, zlbl], 
                          [cmap, 'o', cmap, 20, 0, 5], False, False)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    xpar    = delta_sfms[subsample]
    ypar    = iso_size_ratio[subsample]
    zpar    = radius_hi_iso[subsample]/30.
    
    print(len(xpar))
    
    bins                   = np.arange(-0.7, 1, 0.3)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 2, 3, 6, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    ax1 = fig5.add_subplot(2, 3, 6, facecolor = 'w')
    ax1.set_xlim(-1, 1.25)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    ax1.text(0.7, 0.9, r'$%.2f$' % np.round(stats.pearsonr(xpar, ypar)[0], 2), 
             fontsize = 14, transform=ax1.transAxes)
    
    plot_name = phase1_dir + 'PLOTS/PAPER3/iso_sratio_log20.pdf'
    fig5.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()
  
  if do_iso_muhi_mstar:
    fig5 = plt.figure(56, figsize=(13, 4))
      
    subsample  = ((radius_hi_iso > 20) & np.isfinite(mhi) & np.isfinite(mstar) & 
                  np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  np.isfinite(surfden_hi_iso) & np.isfinite(mu_hi) & 
                  np.isfinite(radius_r_iso25_kpc) & 
                  np.isfinite(radius_hi_iso_kpc) & (mstar > 6.5))
    
    iso_size_ratio = np.log10(radius_hi_iso / radius_r_iso25)
    
    xlbl    = lbl_mstar
    ylbl    = r'$\log(\mu_{\rm{HI}}/[\rm{M_{\odot}\,pc^{-2}}])$'
    zlbl    = 'Beams'
    
    xpar    = mstar[subsample]
    #xpar    = delta_sfms[subsample]
    ypar    = np.log10(surfden_hi_iso[subsample])
    zpar    = radius_hi_iso[subsample]/30.
    
    print(len(xpar))
    
    cmap = plt.get_cmap('rainbow', 5)
    
    scat_col_simple_plot3(fig5, 1, 3, 1, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'ISO', xlbl, ylbl, zlbl], 
                          [cmap, 'o', cmap, 20, 0, 5], True, False)
    
    bins                   = np.arange(7.5, 11.25, 0.5)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 1, 3, 1, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    ax1 = fig5.add_subplot(1, 3, 1, facecolor = 'w')
    #ax1.set_xlim(-11.25, -8.5)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    
    subsample  = ((radius_hi_eff > 20) & np.isfinite(mhi) & np.isfinite(mstar) &
                  np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  np.isfinite(surfden_hi_iso) & np.isfinite(mu_hi) & 
                  np.isfinite(radius_r_iso25_kpc) & 
                  np.isfinite(radius_hi_iso_kpc) & (mstar > 6.5))
    
    iso_size_ratio = np.log10(radius_hi_iso / radius_r_iso25)
    
    xlbl    = lbl_mstar
    #xlbl    = r'$\Delta$\,SFMS [dex]'
    ylbl    = r'$\log(\mu_{\rm{HI}}/[\rm{M_{\odot}\,pc^{-2}}])$'
    zlbl    = 'Beams'
    
    xpar    = mstar[subsample]
    ypar    = np.log10(surfden_hi_iso[subsample])
    zpar    = radius_hi_iso[subsample]/30.
    
    print(len(xpar))
    
    scat_col_simple_plot3(fig5, 1, 3, 2, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'ISO', xlbl, ylbl, zlbl], 
                          [cmap, 'o', cmap, 20, 0, 5], True, False)
    
    bins                   = np.arange(7.5, 11.25, 0.5)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 1, 3, 2, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    ax1 = fig5.add_subplot(1, 3, 2, facecolor = 'w')
    #ax1.set_xlim(-11.25, -8.5)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    subsample  = ((radius_hi_eff > 20) & np.isfinite(mhi) & np.isfinite(mstar) &
                  np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  np.isfinite(surfden_hi_eff) & np.isfinite(mu_hi) & 
                  np.isfinite(radius_r_iso25_kpc) & 
                  np.isfinite(radius_hi_iso_kpc) & (mstar > 6.5))
    
    #eff_size_ratio = np.log10(radius_hi_eff / radius_r_50)
    
    xpar    = mstar[subsample]
    ypar    = np.log10(surfden_hi_eff[subsample])
    zpar    = radius_hi_eff[subsample]/30.
    
    print(len(xpar))
    
    scat_col_simple_plot3(fig5, 1, 3, 3, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'EFF', xlbl, ylbl, zlbl],
                          [cmap, 'o', cmap, 20, 0, 5], True, True)
    
    bins                   = np.arange(7.5, 11.25, 0.5)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 1, 3, 3, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    ax1 = fig5.add_subplot(1, 3, 3, facecolor = 'w')
    #ax1.set_xlim(-11.25, -8.5)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    #plt.show()
    plot_name = phase1_dir + 'PLOTS/PAPER3/muhi_mstar_log20.pdf'
    fig5.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()
  
  if do_eff_muhi_nuvr:
    fig5 = plt.figure(56, figsize=(8, 4))
    
    subsample  = ((radius_hi_iso > 20) & np.isfinite(mhi) & np.isfinite(mstar) &
                  np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  np.isfinite(surfden_hi_iso) & np.isfinite(mu_hi) & 
                  np.isfinite(radius_r_iso25_kpc) & 
                  np.isfinite(radius_hi_iso_kpc) & (mstar > 6.5))
    
    iso_size_ratio = np.log10(radius_hi_iso / radius_r_iso25)
    
    print(len(radius_hi_iso[subsample & (radius_hi_iso < 60)]) / len(radius_hi_iso[subsample]))
    
    #xlbl    = lbl_nuvr
    #ylbl    = r'$\log(\mu_{\rm{HI}}/[\rm{M_{\odot}\,pc^{-2}}])$'
    #zlbl    = 'Beams' #lbl_mstar
    
    #xpar    = nuvr[subsample]
    #ypar    = np.log10(surfden_hi_iso[subsample])
    #zpar    = radius_hi_iso[subsample]/30.
    
    #print(len(xpar))
    
    #cmap = plt.get_cmap('rainbow', 5)
    
    #scat_col_simple_plot3(fig5, 1, 3, 1, 
                          #xpar, 
                          #ypar, 
                          #zpar, 
                          #[r'ISO', xlbl, ylbl, zlbl], 
                          #[cmap, 'o', cmap, 20, 0, 5], True, False)
    
    #bins                   = np.arange(0.7, 3.5, 0.5)
    #par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    #bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    #xbins                  = bin_edges[:-1] + bin_width/2.
    #p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    #p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    #print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    #scatter_error_plot(fig5, 1, 3, 1, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          #'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    #ax1 = fig5.add_subplot(1, 3, 1, facecolor = 'w')
    #ax1.set_xlim(-0.1, 6)
    ##ax1.set_xlim(-1.5, 1.5)
    #ax1.set_ylim(0, 1)
    
    subsample  = ((radius_hi_eff > 20) & np.isfinite(mhi) & np.isfinite(mstar) &
                  np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  np.isfinite(surfden_hi_iso) & np.isfinite(mu_hi) & 
                  np.isfinite(radius_r_iso25_kpc) & 
                  np.isfinite(radius_hi_iso_kpc) & (mstar > 6.5))
    
    print(len(radius_hi_iso[subsample & (radius_hi_eff < 60)]) / len(radius_hi_iso[subsample]))
    
    iso_size_ratio = np.log10(radius_hi_iso / radius_r_iso25)
    
    #cmap = plt.get_cmap('rainbow', 5)
    cmap = (mpl.colors.ListedColormap(['silver', 'silver', 'tab:cyan', 'dodgerblue', 'navy']))
    
    
    xlbl    = lbl_nuvr
    ylbl    = r'$\log(\mu_{\rm{iso,HI}}/[\rm{M_{\odot}\,pc^{-2}}])$'
    zlbl    = 'Radius / Beams' #lbl_mstar 
    
    xpar    = nuvr[subsample]
    ypar    = np.log10(surfden_hi_iso[subsample])
    zpar    = radius_hi_iso[subsample]/30. #mstar[subsample]
    
    print(len(xpar))
    
    scat_col_simple_plot3(fig5, 1, 2, 1, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'ISO', xlbl, ylbl, zlbl], 
                          [cmap, 'o', cmap, 20, 0, 5], False, False)
    
    bins                   = np.arange(0.7, 3.5, 0.5)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 1, 2, 1, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    ax1 = fig5.add_subplot(1, 2, 1, facecolor = 'w')
    ax1.set_xlim(-0.1, 6)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    ax1.text(0.25, 0.05, 'Isodensity', fontsize = 14)
    ax1.text(0.8, 0.9, r'$%.2f$' % np.round(stats.pearsonr(xpar, ypar)[0], 2), 
             fontsize = 14, transform=ax1.transAxes)
    
    #zlbl    = r'$\log(R_{\rm{eff,HI}}/R_{\rm{eff,r}})$'
    
    subsample  = ((radius_hi_eff > 20) & np.isfinite(mhi) & np.isfinite(mstar) &
                  np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  np.isfinite(surfden_hi_eff) & np.isfinite(mu_hi) & 
                  np.isfinite(radius_r_iso25_kpc) & 
                  np.isfinite(radius_hi_iso_kpc) & (mstar > 6.5))
    
    #eff_size_ratio = np.log10(radius_hi_eff / radius_r_50)
    
    ylbl    = r'$\log(\mu_{\rm{50,HI}}/[\rm{M_{\odot}\,pc^{-2}}])$'
    
    xpar    = nuvr[subsample]
    ypar    = np.log10(surfden_hi_eff[subsample])
    zpar    = radius_hi_eff[subsample]/30. #mstar[subsample]
    
    print(len(xpar))
    
    scat_col_simple_plot3(fig5, 1, 2, 2, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'EFF', xlbl, ylbl, zlbl],
                          [cmap, 'o', cmap, 20, 0, 5], False, True)
    
    #bins                   = np.arange(0, 4, 0.25)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 1, 2, 2, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    ax1 = fig5.add_subplot(1, 2, 2, facecolor = 'w')
    ax1.set_xlim(-0.1, 6)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    ax1.text(0.25, 0.05, 'Effective', fontsize = 14)
    ax1.text(0.8, 0.9, r'$%.2f$' % np.round(stats.pearsonr(xpar, ypar)[0], 2), 
             fontsize = 14, transform=ax1.transAxes)
    
    plot_name = phase1_dir + 'PLOTS/PAPER3/eff_muhi_nuvr_log20.pdf'
    fig5.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()
    
  if do_iso_muhi_dsfms:
    fig5 = plt.figure(56, figsize=(13, 4))
      
    subsample  = ((radius_hi_iso > 20) & np.isfinite(mhi) & np.isfinite(mu_star) & 
                  np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  np.isfinite(surfden_hi_iso) & np.isfinite(mu_hi) & 
                  np.isfinite(radius_r_iso25_kpc) & 
                  np.isfinite(radius_hi_iso_kpc) & (mstar > 6.5))
    
    iso_size_ratio = np.log10(radius_hi_iso / radius_r_iso25)
    
    xlbl    = r'$\Delta\,\rm{SFMS}$'
    #xlbl    = r'$\Delta$\,SFMS [dex]'
    ylbl    = r'$\log(\mu_{\rm{HI}}/[\rm{M_{\odot}\,pc^{-2}}])$'
    zlbl    = 'Beams'
    
    xpar    = delta_sfms[subsample]
    #xpar    = delta_sfms[subsample]
    ypar    = np.log10(surfden_hi_iso[subsample])
    zpar    = radius_hi_iso[subsample]/30.
    
    print(len(xpar))
    
    #cmap = plt.get_cmap('rainbow', 5)
    cmap = (mpl.colors.ListedColormap(['silver', 'silver', 'tab:cyan', 'dodgerblue', 'navy']))
    
    scat_col_simple_plot3(fig5, 1, 3, 1, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'ISO', xlbl, ylbl, zlbl], 
                          [cmap, 'o', cmap, 20, 0, 5], True, False)
    
    bins                   = np.arange(-0.7, 1, 0.3)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 1, 3, 1, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    ax1 = fig5.add_subplot(1, 3, 1, facecolor = 'w')
    ax1.set_xlim(-1, 1.25)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    subsample  = ((radius_hi_eff > 20) & np.isfinite(mhi) & np.isfinite(mu_star) & 
                  np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  np.isfinite(surfden_hi_iso) & np.isfinite(mu_hi) & 
                  np.isfinite(radius_r_iso25_kpc) & 
                  np.isfinite(radius_hi_iso_kpc) & (mstar > 6.5))
    
    iso_size_ratio = np.log10(radius_hi_iso / radius_r_iso25)
    
    #xlbl    = lbl_mustar
    #xlbl    = r'$\Delta$\,SFMS [dex]'
    ylbl    = r'$\log(\mu_{\rm{HI}}/[\rm{M_{\odot}\,pc^{-2}}])$'
    zlbl    = 'Beams'
    
    xpar    = delta_sfms[subsample]
    ypar    = np.log10(surfden_hi_iso[subsample])
    zpar    = radius_hi_iso[subsample]/30.
    
    print(len(xpar))
    
    scat_col_simple_plot3(fig5, 1, 3, 2, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'ISO', xlbl, ylbl, zlbl], 
                          [cmap, 'o', cmap, 20, 0, 5], True, False)
    
    #bins                   = np.arange(5.5, 9, 0.5)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 1, 3, 2, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    ax1 = fig5.add_subplot(1, 3, 2, facecolor = 'w')
    ax1.set_xlim(-1, 1.25)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    subsample  = ((radius_hi_eff > 20) & np.isfinite(mhi) & np.isfinite(mu_star) & 
                  np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  np.isfinite(surfden_hi_eff) & np.isfinite(mu_hi) & 
                  np.isfinite(radius_r_iso25_kpc) & 
                  np.isfinite(radius_hi_iso_kpc) & (mstar > 6.5))
    
    xpar    = delta_sfms[subsample]
    ypar    = np.log10(surfden_hi_eff[subsample])
    zpar    = radius_hi_eff[subsample]/30.
    
    print(len(xpar))
    
    scat_col_simple_plot3(fig5, 1, 3, 3, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'EFF', xlbl, ylbl, zlbl],
                          [cmap, 'o', cmap, 20, 0, 5], True, True)
    
    #bins                   = np.arange(0, 4, 0.25)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 1, 3, 3, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    ax1 = fig5.add_subplot(1, 3, 3, facecolor = 'w')
    ax1.set_xlim(-1, 1.25)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    #plt.show()
    plot_name = phase1_dir + 'PLOTS/PAPER3/muhi_dsfms_log20.pdf'
    fig5.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()
  
  if do_iso_muhi_cindex:
    fig5 = plt.figure(56, figsize=(13, 4))
      
    subsample  = ((radius_hi_iso > 20) & np.isfinite(mhi) & np.isfinite(cindex) & 
                  np.isfinite(surfden_hi_iso) & np.isfinite(mu_hi) & 
                  np.isfinite(radius_r_iso25_kpc) & 
                  np.isfinite(radius_hi_iso_kpc) & (mstar > 6.5))
    
    iso_size_ratio = np.log10(radius_hi_iso / radius_r_iso25)
    
    xlbl    = r'$R_{\rm{90,r}}/R_{\rm{50,r}}$'
    ylbl    = r'$\log(\mu_{\rm{HI}}/[\rm{M_{\odot}\,pc^{-2}}])$'
    zlbl    = 'Beams'
    
    xpar    = cindex[subsample]
    #xpar    = delta_sfms[subsample]
    ypar    = np.log10(surfden_hi_iso[subsample])
    zpar    = radius_hi_iso[subsample]/30.
    
    print(len(xpar))
    
    scat_col_simple_plot3(fig5, 1, 3, 1, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'ISO', xlbl, ylbl, zlbl], 
                          ['rainbow', 'o', 'rainbow', 20, 0, 5], True, False)
    
    bins                   = np.arange(1.5, 4, 0.25)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 1, 3, 1, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], True)
    
    ax1 = fig5.add_subplot(1, 3, 1, facecolor = 'w')
    ax1.set_xlim(1.5, 4)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    subsample  = ((radius_hi_eff > 20) & np.isfinite(mhi) & np.isfinite(cindex) & 
                  np.isfinite(surfden_hi_iso) & np.isfinite(mu_hi) & 
                  np.isfinite(radius_r_iso25_kpc) & 
                  np.isfinite(radius_hi_iso_kpc) & (mstar > 6.5))
    
    iso_size_ratio = np.log10(radius_hi_iso / radius_r_iso25)
    
    xlbl    = r'$R_{\rm{90,r}}/R_{\rm{50,r}}$'
    ylbl    = r'$\log(\mu_{\rm{HI}}/[\rm{M_{\odot}\,pc^{-2}}])$'
    zlbl    = 'Beams'
    
    xpar    = cindex[subsample]
    ypar    = np.log10(surfden_hi_iso[subsample])
    zpar    = radius_hi_iso[subsample]/30.
    
    print(len(xpar))
    
    scat_col_simple_plot3(fig5, 1, 3, 2, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'ISO', xlbl, ylbl, zlbl], 
                          ['rainbow', 'o', 'rainbow', 20, 0, 5], True, False)
    
    bins                   = np.arange(1.5, 4, 0.25)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 1, 3, 2, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], True)
    
    ax1 = fig5.add_subplot(1, 3, 2, facecolor = 'w')
    ax1.set_xlim(1.5, 4)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    subsample  = ((radius_hi_eff > 20) & np.isfinite(mhi) & np.isfinite(cindex) & 
                  np.isfinite(surfden_hi_eff) & np.isfinite(mu_hi) & 
                  np.isfinite(radius_r_iso25_kpc) & 
                  np.isfinite(radius_hi_iso_kpc) & (mstar > 6.5))
    
    xpar    = cindex[subsample]
    ypar    = np.log10(surfden_hi_eff[subsample])
    zpar    = radius_hi_eff[subsample]/30.
    
    print(len(xpar))
    
    scat_col_simple_plot3(fig5, 1, 3, 3, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'EFF', xlbl, ylbl, zlbl],
                          ['rainbow', 'o', 'rainbow', 20, 0, 5], True, True)
    
    #bins                   = np.arange(0, 4, 0.25)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 1, 3, 3, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], True)
    
    ax1 = fig5.add_subplot(1, 3, 3, facecolor = 'w')
    ax1.set_xlim(1.5, 4)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    #plt.show()
    plot_name = phase1_dir + 'PLOTS/PAPER3/muhi_cindex_log20.pdf'
    fig5.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()
  
  
  
  if do_iso_sratio_dsfms:
    fig5 = plt.figure(56, figsize=(13, 4))
    
    subsample  = ((radius_hi_iso > 20) & np.isfinite(mhi) & np.isfinite(mstar) &
                  np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  np.isfinite(radius_r_iso25) & 
                  np.isfinite(radius_hi_iso) & (mstar > 6.5))
    
    iso_size_ratio = np.log10(radius_hi_iso / radius_r_iso25)
    
    xlbl    = r'$\Delta\,\rm{SFMS}$'
    ylbl    = r'$\log(R_{\rm{iso,HI}}/R_{\rm{iso25,r}})$'
    zlbl    = 'Beams'
    
    xpar    = delta_sfms[subsample]
    #xpar    = delta_sfms[subsample]
    ypar    = iso_size_ratio[subsample]
    zpar    = radius_hi_iso[subsample]/30.
    
    print(len(xpar))
    
    #cmap = plt.get_cmap('rainbow', 5)
    cmap = (mpl.colors.ListedColormap(['silver', 'silver', 'tab:cyan', 'dodgerblue', 'navy']))
    
    scat_col_simple_plot3(fig5, 1, 3, 1, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'ISO', xlbl, ylbl, zlbl], 
                          [cmap, 'o', cmap, 20, 0, 5], True, False)
    
    bins                   = np.arange(-0.7, 1, 0.3)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 1, 3, 1, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    ax1 = fig5.add_subplot(1, 3, 1, facecolor = 'w')
    ax1.set_xlim(-1, 1.25)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    
    subsample  = ((radius_hi_eff > 20) & np.isfinite(mhi) & np.isfinite(mstar) &
                  np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  np.isfinite(radius_r_iso25) & 
                  np.isfinite(radius_hi_iso) & (mstar > 6.5))
    
    iso_size_ratio = np.log10(radius_hi_iso / radius_r_iso25)
    
    #xlbl    = lbl_mstar
    ylbl    = r'$\log(R_{\rm{iso,HI}}/R_{\rm{iso25,r}})$'
    zlbl    = 'Beams'
    
    xpar    = delta_sfms[subsample]
    #xpar    = delta_sfms[subsample]
    ypar    = iso_size_ratio[subsample]
    zpar    = radius_hi_iso[subsample]/30.
    
    print(len(xpar))
    
    scat_col_simple_plot3(fig5, 1, 3, 2, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'ISO', xlbl, ylbl, zlbl], 
                          [cmap, 'o', cmap, 20, 0, 5], True, False)
    
    #bins                   = np.arange(7.5, 11.25, 0.5)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 1, 3, 2, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    ax1 = fig5.add_subplot(1, 3, 2, facecolor = 'w')
    ax1.set_xlim(-1, 1.25)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    #zlbl    = r'$\log(R_{\rm{eff,HI}}/R_{\rm{eff,r}})$'
    
    subsample  = ((radius_hi_eff > 20) & np.isfinite(mhi) & np.isfinite(mstar) &
                  np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  np.isfinite(radius_r_50) & 
                  np.isfinite(radius_hi_eff) & (mstar > 6.5))
    
    eff_size_ratio = np.log10(radius_hi_eff / radius_r_50)
    
    #xlbl    = lbl_mstar
    ylbl    = r'$\log(R_{\rm{eff,HI}}/R_{\rm{eff,r}})$'
    zlbl    = 'Beams'
    
    xpar    = delta_sfms[subsample]
    #xpar    = delta_sfms[subsample]
    ypar    = eff_size_ratio[subsample]
    #ypar    = mu_mean[subsample]
    zpar    = radius_hi_eff[subsample]/30.
    
    print(len(xpar))
    
    scat_col_simple_plot3(fig5, 1, 3, 3, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'EFF', xlbl, ylbl, zlbl],
                          [cmap, 'o', cmap, 20, 0, 5], True, True)
    
    #bins                   = np.arange(7.5, 11.25, 0.5)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 1, 3, 3, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    ax1 = fig5.add_subplot(1, 3, 3, facecolor = 'w')
    ax1.set_xlim(-1, 1.25)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    #plt.show()
    plot_name = phase1_dir + 'PLOTS/PAPER3/sratio_dsfms_log20.pdf'
    fig5.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()
  
  if do_eff_sratio_nuvr:
    fig5 = plt.figure(56, figsize=(8, 4))
    
    subsample  = ((radius_hi_iso > 20) & np.isfinite(mhi) & np.isfinite(mstar) &
                  np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  np.isfinite(radius_r_iso25) & 
                  np.isfinite(radius_hi_iso) & (mstar > 6.5))
    
    iso_size_ratio = np.log10(radius_hi_iso / radius_r_iso25)
    
    #xlbl    = lbl_nuvr
    #ylbl    = r'$\log(R_{\rm{iso,HI}}/R_{\rm{iso25,r}})$'
    #zlbl    = 'Beams' #lbl_mstar #r'$\log(R_{\rm{iso,HI}}/R_{\rm{iso25,r}})$'
    
    #xpar    = nuvr[subsample]
    ##xpar    = delta_sfms[subsample]
    #ypar    = iso_size_ratio[subsample]
    #zpar    = radius_hi_iso[subsample]/30. #mstar[subsample]
    
    #print(len(xpar))
    
    #cmap = plt.get_cmap('rainbow', 5)
    
    #scat_col_simple_plot3(fig5, 1, 3, 1, 
                          #xpar, 
                          #ypar, 
                          #zpar, 
                          #[r'ISO', xlbl, ylbl, zlbl], 
                          #[cmap, 'o', cmap, 20, 0, 5], True, False)
    
    #bins                   = np.arange(0.5, 3.5, 0.5)
    #par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    #bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    #xbins                  = bin_edges[:-1] + bin_width/2.
    #p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    #p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    #print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    #scatter_error_plot(fig5, 1, 3, 1, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          #'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    #ax1 = fig5.add_subplot(1, 3, 1, facecolor = 'w')
    #ax1.set_xlim(-0.1, 6)
    ##ax1.set_xlim(-1.5, 1.5)
    #ax1.set_ylim(0, 1)
    
    subsample  = ((radius_hi_eff > 20) & np.isfinite(mhi) & np.isfinite(mstar) &
                  np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  np.isfinite(radius_r_iso25) & 
                  np.isfinite(radius_hi_iso) & (mstar > 6.5))
    
    iso_size_ratio = np.log10(radius_hi_iso / radius_r_iso25)
    
    xlbl    = lbl_nuvr
    ylbl    = r'$\log(R_{\rm{iso,HI}}/R_{\rm{iso,r}})$'
    zlbl    = 'Radius / Beams' #lbl_mstar #r'$\log(R_{\rm{iso,HI}}/R_{\rm{iso25,r}})$'
    
    xpar    = nuvr[subsample]
    #xpar    = delta_sfms[subsample]
    ypar    = iso_size_ratio[subsample]
    zpar    = radius_hi_iso[subsample]/30. #mstar[subsample]
    
    print(len(xpar))
    
    #cmap = plt.get_cmap('rainbow', 5)
    cmap = (mpl.colors.ListedColormap(['silver', 'silver', 'tab:cyan', 'dodgerblue', 'navy']))
    
    scat_col_simple_plot3(fig5, 1, 2, 1, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'ISO', xlbl, ylbl, zlbl], 
                          [cmap, 'o', cmap, 20, 0, 5], False, False)
    
    bins                   = np.arange(0.5, 3.5, 0.5)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 1, 2, 1, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    ax1 = fig5.add_subplot(1, 2, 1, facecolor = 'w')
    ax1.set_xlim(-0.1, 6)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    ax1.text(0.25, 0.05, 'Isodensity', fontsize = 14)
    ax1.text(0.8, 0.9, r'$%.2f$' % np.round(stats.pearsonr(xpar, ypar)[0], 2), 
             fontsize = 14, transform=ax1.transAxes)
    
    #zlbl    = r'$\log(R_{\rm{eff,HI}}/R_{\rm{eff,r}})$'
    
    subsample  = ((radius_hi_eff > 20) & np.isfinite(mhi) & np.isfinite(mstar) &
                  np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  np.isfinite(radius_r_50) & 
                  np.isfinite(radius_hi_eff) & (mstar > 6.5))
    
    eff_size_ratio = np.log10(radius_hi_eff / radius_r_50)
    
    xlbl    = lbl_nuvr
    ylbl    = r'$\log(R_{\rm{50,HI}}/R_{\rm{50,r}})$'
    zlbl    = 'Radius / Beams' #lbl_mstar #r'$\log(R_{\rm{iso,HI}}/R_{\rm{iso25,r}})$'
    
    xpar    = nuvr[subsample]
    #xpar    = delta_sfms[subsample]
    ypar    = eff_size_ratio[subsample]
    #ypar    = mu_mean[subsample]
    zpar    = radius_hi_eff[subsample]/30. #mstar[subsample]
    
    print(len(xpar))
    
    scat_col_simple_plot3(fig5, 1, 2, 2, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'EFF', xlbl, ylbl, zlbl],
                          [cmap, 'o', cmap, 20, 0, 5], False, True)
    
    #bins                   = np.arange(0, 4, 0.25)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 1, 2, 2, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    ax1 = fig5.add_subplot(1, 2, 2, facecolor = 'w')
    ax1.set_xlim(-0.1, 6)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    ax1.text(0.25, 0.05, 'Effective', fontsize = 14)
    ax1.text(0.8, 0.9, r'$%.2f$' % np.round(stats.pearsonr(xpar, ypar)[0], 2), 
             fontsize = 14, transform=ax1.transAxes)
    
    #plt.show()
    plot_name = phase1_dir + 'PLOTS/PAPER3/eff_sratio_nuvr_log20.pdf'
    #plot_name = phase1_dir + 'PLOTS/PAPER3/size_dsfms_muhi.pdf'
    fig5.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()
  
  if do_eff_sratio_mustar:
    fig5 = plt.figure(57, figsize=(8, 4))
    
    subsample  = ((radius_hi_iso > 20) & np.isfinite(mhi) & np.isfinite(mu_star) & np.isfinite(mstar) &
                  np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  np.isfinite(radius_r_iso25) & 
                  np.isfinite(radius_hi_iso) & (mstar > 6.5))
    
    iso_size_ratio = np.log10(radius_hi_iso / radius_r_iso25)
    
    #xlbl    = lbl_mustar
    #ylbl    = r'$\log(R_{\rm{iso,HI}}/R_{\rm{iso25,r}})$'
    #zlbl    = 'Beams'
    
    #xpar    = mu_star[subsample]
    #ypar    = iso_size_ratio[subsample]
    #zpar    = radius_hi_iso[subsample]/30.
    
    #print(len(xpar))
    
    #cmap = plt.get_cmap('rainbow', 5)
    
    #scat_col_simple_plot3(fig5, 1, 3, 1, 
                          #xpar, 
                          #ypar, 
                          #zpar, 
                          #[r'ISO', xlbl, ylbl, zlbl], 
                          #[cmap, 'o', cmap, 20, 0, 5], True, False)
    
    #bins                   = np.arange(6.2, 9, 0.5)
    #par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    #bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    #xbins                  = bin_edges[:-1] + bin_width/2.
    #p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    #p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    #print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    #scatter_error_plot(fig5, 1, 3, 1, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          #'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    #ax1 = fig5.add_subplot(1, 3, 1, facecolor = 'w')
    #ax1.set_xlim(5.5, 9)
    ##ax1.set_xlim(-1.5, 1.5)
    #ax1.set_ylim(0, 1)
    
    subsample  = ((radius_hi_eff > 20) & np.isfinite(mhi) & np.isfinite(mu_star) & np.isfinite(mstar) &
                  np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  np.isfinite(radius_r_iso25) & 
                  np.isfinite(radius_hi_iso) & (mstar > 6.5))
    
    iso_size_ratio = np.log10(radius_hi_iso / radius_r_iso25)
    
    xlbl    = lbl_mustar
    ylbl    = r'$\log(R_{\rm{iso,HI}}/R_{\rm{iso,r}})$'
    zlbl    = 'Radius / Beams'
    
    xpar    = mu_star[subsample]
    #xpar    = delta_sfms[subsample]
    ypar    = iso_size_ratio[subsample]
    zpar    = radius_hi_iso[subsample]/30.
    
    print(len(xpar))
    
    #cmap = plt.get_cmap('rainbow', 5)
    cmap = (mpl.colors.ListedColormap(['silver', 'silver', 'tab:cyan', 'dodgerblue', 'navy']))
    
    scat_col_simple_plot3(fig5, 1, 2, 1, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'ISO', xlbl, ylbl, zlbl], 
                          [cmap, 'o', cmap, 20, 0, 5], False, False)
    
    bins                   = np.arange(6.2, 9, 0.5)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 1, 2, 1, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    ax1 = fig5.add_subplot(1, 2, 1, facecolor = 'w')
    ax1.set_xlim(5.5, 9)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    ax1.text(5.8, 0.05, 'Isodensity', fontsize = 14)
    ax1.text(0.8, 0.9, r'$%.2f$' % np.round(stats.pearsonr(xpar, ypar)[0], 2), 
             fontsize = 14, transform=ax1.transAxes)
    
    #zlbl    = r'$\log(R_{\rm{eff,HI}}/R_{\rm{eff,r}})$'
    
    subsample  = ((radius_hi_eff > 20) & np.isfinite(mhi) & np.isfinite(mu_star) & np.isfinite(mstar) &
                  np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  np.isfinite(radius_r_50) & 
                  np.isfinite(radius_hi_eff) & (mstar > 6.5))
    
    eff_size_ratio = np.log10(radius_hi_eff / radius_r_50)
    
    xlbl    = lbl_mustar
    ylbl    = r'$\log(R_{\rm{50,HI}}/R_{\rm{50,r}})$'
    zlbl    = 'Radius / Beams'
    
    xpar    = mu_star[subsample]
    #xpar    = delta_sfms[subsample]
    ypar    = eff_size_ratio[subsample]
    #ypar    = mu_mean[subsample]
    zpar    = radius_hi_eff[subsample]/30.
    
    print(len(xpar))
    
    scat_col_simple_plot3(fig5, 1, 2, 2, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'EFF', xlbl, ylbl, zlbl],
                          [cmap, 'o', cmap, 20, 0, 5], False, True)
    
    
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 1, 2, 2, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    ax1 = fig5.add_subplot(1, 2, 2, facecolor = 'w')
    ax1.set_xlim(5.5, 9)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(0, 1)
    
    ax1.text(5.8, 0.05, 'Effective', fontsize = 14)
    ax1.text(0.8, 0.9, r'$%.2f$' % np.round(stats.pearsonr(xpar, ypar)[0], 2), 
             fontsize = 14, transform=ax1.transAxes)
    
    #plt.show()
    plot_name = phase1_dir + 'PLOTS/PAPER3/eff_sratio_mustar_log20.pdf'
    #plot_name = phase1_dir + 'PLOTS/PAPER3/size_dsfms_muhi.pdf'
    fig5.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()
  
  if do_delta_size:
    fig5 = plt.figure(56, figsize=(8, 4))
    
    subsample  = ((radius_hi_eff > 20) & np.isfinite(mhi) & np.isfinite(mstar) &
                  np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  np.isfinite(surfden_hi_iso) & np.isfinite(mu_hi) & 
                  np.isfinite(radius_r_iso25_kpc) & 
                  np.isfinite(radius_hi_iso_kpc) & (mstar > 6.5) & 
                  np.isfinite(radius_r_50) & np.isfinite(radius_hi_eff))
    
    xlbl    = lbl_mstar
    ylbl    = r'$(R_{\rm{iso,r}} - R_{\rm{50,r}})/R_{\rm{iso,r}}$'
    zlbl    = 'Radius / Beams'
    
    xpar    = mstar[subsample]
    ypar    = (radius_r_iso25 - radius_r_50) / radius_r_iso25
    ypar    = ypar[subsample]
    zpar    = radius_hi_iso[subsample]/30.
    
    print(len(xpar))
    
    print(len(mstar[subsample & (radius_hi_iso < 60)])/len(mstar[subsample]))
    
    #cmap = plt.get_cmap('rainbow', 5)
    cmap = (mpl.colors.ListedColormap(['silver', 'silver', 'tab:cyan', 'dodgerblue', 'navy']))
    
    scat_col_simple_plot3(fig5, 1, 2, 1, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'Isophotal', xlbl, ylbl, zlbl], 
                          [cmap, 'o', cmap, 20, 0, 5], False, False)
    
    bins                   = np.arange(7.5, 11.25, 0.5)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 1, 2, 1, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    ax1 = fig5.add_subplot(1, 2, 1, facecolor = 'w')
    #ax1.set_xlim(-11.25, -8.5)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_xlim(6.8,11.2)
    ax1.set_ylim(0, 1)
    
    #ax1.text(7, 0.9, 'Isodensity', fontsize = 14)
    ax1.text(0.8, 0.9, r'$%.2f$' % np.round(stats.pearsonr(xpar, ypar)[0], 2), 
             fontsize = 14, transform=ax1.transAxes)
    
    subsample  = ((radius_hi_eff > 20) & np.isfinite(mhi) & np.isfinite(mstar) &
                  np.isfinite(nuvr)  & np.isfinite(ssfr) & 
                  np.isfinite(surfden_hi_iso) & np.isfinite(mu_hi) & 
                  np.isfinite(radius_r_iso25_kpc) & 
                  np.isfinite(radius_hi_iso_kpc) & (mstar > 6.5) & 
                  np.isfinite(radius_r_50) & np.isfinite(radius_hi_eff))
    
    xlbl    = lbl_mhi
    ylbl    = r'$(R_{\rm{iso,HI}} - R_{\rm{50,HI}})/R_{\rm{iso,HI}}$'
    zlbl    = 'Radius / Beams'
    
    xpar    = mhi[subsample]
    ypar    = (radius_hi_iso - radius_hi_eff) / radius_hi_iso
    ypar    = ypar[subsample]
    zpar    = radius_hi_iso[subsample]/30.
    
    print(len(xpar))
    
    print(len(mstar[subsample & (radius_hi_iso < 60)])/len(mstar[subsample]))
    
    #cmap = plt.get_cmap('rainbow', 5)
    cmap = (mpl.colors.ListedColormap(['silver', 'silver', 'tab:cyan', 'dodgerblue', 'navy']))
    
    scat_col_simple_plot3(fig5, 1, 2, 2, 
                          xpar, 
                          ypar, 
                          zpar, 
                          [r'Isophotal', xlbl, ylbl, zlbl], 
                          [cmap, 'o', cmap, 20, 0, 5], False, True)
    
    bins                   = np.arange(8, 11, 0.5)
    par_mean, bin_edges, _ = binned_statistic(xpar, ypar, np.nanmedian, bins)
    bin_width              = np.abs(bin_edges[1] - bin_edges[0])
    xbins                  = bin_edges[:-1] + bin_width/2.
    p20, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 20), bins)
    p80, _, _ = binned_statistic(xpar, ypar, lambda y: np.percentile(y, 80), bins)
    
    print(np.round(stats.pearsonr(xpar, ypar)[0], 2), stats.pearsonr(xpar, ypar)[1])
    
    scatter_error_plot(fig5, 1, 2, 2, xbins, par_mean, [par_mean - p20, p80 - par_mean],
                          'Medians', xlbl, ylbl, ['black', 's', 'black', 7.5], False)
    
    ax1 = fig5.add_subplot(1, 2, 2, facecolor = 'w')
    #ax1.set_xlim(-11.25, -8.5)
    #ax1.set_xlim(-1.5, 1.5)
    ax1.set_xlim(6.8,11.2)
    ax1.set_ylim(0, 1)
    
    #ax1.text(7, 0.9, 'Isodensity', fontsize = 14)
    ax1.text(0.8, 0.9, r'$%.2f$' % np.round(stats.pearsonr(xpar, ypar)[0], 2), 
             fontsize = 14, transform=ax1.transAxes)
    
    plot_name = phase1_dir + 'PLOTS/PAPER3/delta_radius_mass.pdf'
    plt.savefig(plot_name, bbox_inches = 'tight', dpi = 1000)
    plt.clf()

