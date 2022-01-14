# CASTEP ELNES Parser

A simple python script for parsing [CASTEP](http://www.castep.org/) output files especially for ELNES calculations.

## Overview

This includes some functions for reading .cell, .castep, .bands, and .eels_mat files, calculating excitation energy, and forming core-loss spectra with Gaussian smearing.

## Requirements

- python (=>3)
- numpy

## Install

Clone this repository and run `pip install`:

``` bash
$ git clone https://github.com/nmdl-mizo/castep_elnes_parser.git
$ cd castep_elne_parser
$ pip intall .
```

or directry run `pip install`:

``` bash
$ pip install git+https://github.com/nmdl-mizo/castep_elnes_parser
```

## Usage

This simple script contains several primitive functions for dealing with CASTEP files.

### Get energies from castep files and calculate excitation energy

This script can extract energies and calculate the excitation energies from a pair of {seed_name}.castep files for the ground state (gs) and excited state (ex).
The label of the site of interest can be specified by `element` and `suffix`.

``` python
>>> !ls . # prepare two {seed_name}.castep files for ground state and excitated state
case_gs.castep    case.castep
>>> import castep_elnes_parser as cep
>>> # extract energies and calculate excitation energy for a site labelled as "C:ex"
>>> energies = cep.get_energies(filename_gs="./case_gs.castep", filename_ex="./case.castep", element="C", suffix=":ex")
```

### Form spectra from eigen values and transition matrix elements

This script can form spectra from a pair of {seed_name}.bands and {seed_name}.eels_mat files.

``` python
>>> !ls . # prepare {seed_name}.bands and {seed_name}.eels_mat
case_elnes.bands    case_elnes.eels_mat
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> import castep_elnes_parser as cep
>>> # calculate gaussian smeared spectra of Gaussian sigma 0.3 eV
>>> energies = np.arange(-4.999, 30.002, 0.001) # make an array for energies with a desired range and resolution
>>> sp = get_smeared_spectrum(energies=energies, sigma=0.3, calc_dir=".", seed_name="case_elnes") # parse and make spectra by gaussian smearing with sigma
>>> fig, ax = plt.subplots(1)
>>> ax.set_xlabel("Energy (eV)")
>>> ax.set_ylabel("Intensity (arb. u.)")
>>> ax.plot(energies, sp[0, 0], label="x") # plot a spectrum for x component of the 1st core projection
>>> ax.plot(energies, sp[0, 1], label="y") # plot a spectrum for y component of the 1st core projection
>>> ax.plot(energies, sp[0, 2], label="z") # plot a spectrum for z component of the 1st core projection
>>> ax.plot(energies, np.mean(sp[0], axis=0), label="total") # plot a total spectrum of the 1st core projection
```

## Note

This script is tested on input and output files of CASTEP version 8 and may not be incompatible to other versions.

## Reference

In all papers using the CASTEP code, you should cite:
> "First principles methods using CASTEP",
> Zeitschrift fuer Kristallographie 220(5-6) pp. 567-570 (2005)
> S. J. Clark, M. D. Segall, C. J. Pickard, P. J. Hasnip, M. J. Probert, K. Refson, M. C. Payne

If you use get_energies() for calculating excitation energy, please consider to cite:
> Mizoguchi, T.; Tanaka, I.; Gao, S.-P.; Pickard, C. J.
> "First-Principles Calculation of Spectral Features, Chemical Shift and Absolute Threshold of ELNES and XANES Using a Plane Wave Pseudopotential Method." \
> J. Phys. Condens. Matter 2009, 21 (10), 104204.

## Documentation

[Documentation generated by Sphinx](https://nmdl-mizo.github.io/castep_elnes_parser/index.html) is available.

## Author

- [Kiyou](https://github.com/kiyou)
- [Mizoguchi Lab.](https://github.com/nmdl-mizo)

## Licence

The source code is licensed MIT.
