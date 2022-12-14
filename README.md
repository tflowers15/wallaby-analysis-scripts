# wallaby-analysis-scripts

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

This repository contains python scripts for measuring structural parameters from HI sources 
detected by WALLABY, downloading mutli-wavelength image cutouts at the HI source positions, 
measuring multi-wavelength photometry and deriving physical quantities (e.g. stellar/HI mass
and star formation rates).

## Documentation

These scripts are built using Python v3.7.6 and primarily use the packages Astropy v4.0 and Photutils v1.3.0. The python scripts are run from the command line, e.g. "python *.py". Controls and settings are defined within each script. These scripts require a WALLABY SoFiA source catalogue and associated data products (e.g. spectra, moment maps, cubelets, etc.) as inputs to be run.

The procedures for downloading/measuring photometry/deriving physical quantities are described in Photometry_Cookbook_vX.pdf. This document also provides a step-by-step guide showing the order and settings in which each python script should be run.

## Copyright and licence

Copyright (C) 2022 Tristan Reynolds

**wallaby-analysis-scripts** is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.