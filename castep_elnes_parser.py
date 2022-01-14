#!/usr/bin/env python
"""
A simple python script for parsing CASTEP (http://www.castep.org/) output files especially for ELNES calculations.

This includes some functions for reading .cell, .castep, .bands, and .eels_mat files, calculating excitation energy, and forming core-loss spectra with Gaussian smearing.

Copyright (c) 2022 kiyou, nmdl-mizo
This software is released under the MIT License, see LICENSE.

Note
-----
This script is tested on input and output files of CASTEP version 8 and may not be incompatible to other versions.

In all papers using the CASTEP code, you should cite:
"First principles methods using CASTEP",
Zeitschrift fuer Kristallographie 220(5-6) pp. 567-570 (2005)
S. J. Clark, M. D. Segall, C. J. Pickard, P. J. Hasnip, M. J. Probert, K. Refson, M. C. Payne

If you use get_energies() for calculating excitation energy, please consider to cite:
Mizoguchi, T.; Tanaka, I.; Gao, S.-P.; Pickard, C. J.
"First-Principles Calculation of Spectral Features, Chemical Shift and Absolute Threshold of ELNES and XANES Using a Plane Wave Pseudopotential Method." \
J. Phys. Condens. Matter 2009, 21 (10), 104204.
"""
import os
import re
import struct
import numpy as np


def split_castep(filename):
    """
    Split .castep file into each calculation

    Running CASTEP several times yields a single .castep file with concatenated output.
    This function splits the outputs into a list of each calculation run.

    Parameters
    --------
    filename : str
        path to the .castep file

    Returns
    --------
    run_list : list
        list of lines of castep output for each run
    """
    with open(filename, "rt") as f:
        lines = f.readlines()
    castep_header = [
        ' +-------------------------------------------------+\n',
        ' |                                                 |\n',
        ' |      CCC   AA    SSS  TTTTT  EEEEE  PPPP        |\n',
        ' |     C     A  A  S       T    E      P   P       |\n',
        ' |     C     AAAA   SS     T    EEE    PPPP        |\n',
        ' |     C     A  A     S    T    E      P           |\n',
        ' |      CCC  A  A  SSS     T    EEEEE  P           |\n',
        ' |                                                 |\n',
        ' +-------------------------------------------------+\n'
    ]
    header_position_list = list()
    for l in range(len(lines) - 9):
        if lines[l: l + 9] == castep_header:
            header_position_list.append(l)
    header_position_list.append(-1)
    run_list = [lines[l_s:l_e] for l_s, l_e in zip(header_position_list[:-1], header_position_list[1:])]
    return run_list


def get_final_energy_castep(lines):
    """
    get the final energy of the system from .castep output

    Parameters
    --------
    lines : list of str
        Castep output for a calculation run.
        Can be obtained by split_castep

    Returns
    --------
    final_energy: float
        final energy in eV
    """
    for line in lines:
        if "Final energy" in line:
            final_energy = float(line.split()[-2])
            break
    else:
        raise RuntimeError("Could not find Final energy")
    return final_energy


def get_ae_energy_castep(lines, element="C", suffix=":ex"):
    """
    get the atomic energy of the all-electron (ae) calculation from .castep output

    Parameters
    --------
    lines : list of str
        Castep output for a calculation run.
        Can be obtained by split_castep
    element : str, default "C"
        element name. e.g. "C"
    suffix : str, default ":ex"
        suffix for the site name e.g. ":ex"

    Returns
    --------
    ae_energy: float
        ae energy in eV
    """
    re_ac = re.compile(f"Atomic calculation performed for {element}{suffix}")
    re_ae = re.compile("Converged in \d+ iterations to an ae energy of ([\d\.\-]+) eV")
    for i, line in enumerate(lines):
        if re_ac.search(line) is not None:
            break
    else:
        raise RuntimeError(f"Could not find atomic calculation for {element}{suffix}")
    ae_energy = float(re_ae.search(lines[i + 2]).group(1))
    return ae_energy


def get_pa_energy_castep(lines, element="C", suffix=":ex"):
    """
    get the atomic energy of the pseudo atomic (pa) calculation from .castep output

    Parameters
    --------
    lines : list of str
        Castep output for a calculation run.
        Can be obtained by split_castep
    element : str, default "C"
        element name. e.g. "C"
    suffix : str, default ":ex"
        suffix for the site name e.g. ":ex"

    Returns
    --------
    pa_energy: float
        pseudo atomic energy in eV
    """
    re_pac = re.compile(f"Pseudo atomic calculation performed for {element}{suffix}")
    re_pa = re.compile("Converged in \d+ iterations to a total energy of ([\d\.\-]+) eV")
    for i, line in enumerate(lines):
        if re_pac.search(line) is not None:
            break
    else:
        raise RuntimeError(f"Could not find pseudo atomic calculation for {element}{suffix}")
    pa_energy = float(re_pa.search(lines[i + 2]).group(1))
    return pa_energy


def get_energies(filename_gs, filename_ex, element="C", suffix=":ex", gs_split=-1, ex_split=-1):
    """
    get energies from .castep outputs

    Parameters
    --------
    lines : list of str
        Castep output for a calculation run.
        Can be obtained by split_castep
    element : str, default "C"
        element name. e.g. "C"
    suffix : str, default ":ex"
        suffix for the site name e.g. ":ex"
    gs_split : int, default -1
        index of the split used for extracting energies from ground state calculation. -1 to latest.
    ex_split : int, default -1
        index of the split used for extracting energies from excited state calculation. -1 to latest.

    Returns
    --------
    energy_dict: dict
        a dictionary wiht str keys of labels and flot values of energies in eV
            gs_final: ground state final_energy,
            ex_final: excited state final energy,
            gs_ae: ground state atomic energy in the all-electron calculation,
            ex_ae: excited state atomic energy in the all-electron calculation,
            gs_pa: ground state atomic energy in the pseudo-atomic calculation,
            ex_pa: excited state atomic energy in the pseudo-atomic calculation,
            excitation_energy: excitation energy,

    Note
    -------
    The calculation of excitation energy is based on the following paper:
    Mizoguchi, T.; Tanaka, I.; Gao, S.-P.; Pickard, C. J. \
    "First-Principles Calculation of Spectral Features, Chemical Shift and Absolute Threshold of ELNES and XANES Using a Plane Wave Pseudopotential Method." \
    J. Phys. Condens. Matter 2009, 21 (10), 104204.
    """
    lines_gs = split_castep(filename_gs)[gs_split]
    lines_ex = split_castep(filename_ex)[ex_split]
    energy_dict = {
        "gs_final": get_final_energy_castep(lines_gs),
        "ex_final": get_final_energy_castep(lines_ex),
        "gs_ae": get_ae_energy_castep(lines_gs, element, suffix),
        "ex_ae": get_ae_energy_castep(lines_ex, element, suffix),
        "gs_pa": get_pa_energy_castep(lines_gs, element, suffix),
        "ex_pa": get_pa_energy_castep(lines_ex, element, suffix),
    }
    excitation_energy = (energy_dict["ex_final"] - energy_dict["gs_final"]) + ((energy_dict["ex_ae"] - energy_dict["ex_pa"]) - (energy_dict["gs_ae"] - energy_dict["gs_pa"]))
    energy_dict["excitation_energy"] = excitation_energy
    return energy_dict


def get_coords_cell(filename):
    """
    extract species and coordinates from .cell file

    Parameters
    --------
    filename : str
        path to the .cell file

    Returns
    --------
    coords_list : list
        a list of two elements
        - a list of species
        - a numpy array of coordinations
    """
    with open(filename, "r") as f:
        while not "%BLOCK POSITIONS_FRAC" in f.readline():
            pass
        lines = list()
        while True:
            line = f.readline()
            if "%ENDBLOCK POSITIONS_FRAC" in line:
                break
            lines.append(line)
    coords_list = [line.split()[0] for line in lines], np.array([line.split()[1:] for line in lines], dtype=np.float)
    return coords_list


def read_bin_data(f, dtype, data_byte, header_byte=4, footer_byte=4):
    """
    read data from binary file
    This function reads a binary chunk from current position.

    Parameters
    --------
    f : file object
        a file object opened in read only mode.
    dtype : str
        dtype of the binary chunk
    data_byte : int
        length of the binary chunk to read in byte
    header_byte : int, default 4
        length of the header before the binary chunk to read in byte
    footer_byte : int, default 4
        length of the footer after the binary chunk to read in byte

    Returns
    --------
    data : int, str, float, or list
        a data converted by struct.unpack
    """
    f.seek(header_byte, 1)
    data = struct.unpack(dtype, f.read(data_byte))
    f.seek(footer_byte, 1)
    return data


def read_eels_mat(filename):
    """
    extract data from .eels_mat file

    Parameters
    --------
    filename : str
        path to the .eels_mat file

    Returns
    --------
    eels_mat_dict : dict
        a dictionary of the extracted data
        - tot_core_projectors
        - max_eigenvalues
        - sum_num_kpoints
        - num_spins
        - core_orbital_species
        - core_orbital_ion
        - core_orbital_n
        - core_orbital_lm
        - transition_matrix
    """
    params = [
        "tot_core_projectors",
        "max_eigenvalues",
        "sum_num_kpoints",
        "num_spins",
    ]
    co = [
        "core_orbital_species",
        "core_orbital_ion",
        "core_orbital_n",
        "core_orbital_lm",
    ]
    with open(filename, "rb") as f:
        param_chunk = {p: read_bin_data(f, ">i", 4)[0] for p in params}
        n_proj = param_chunk["tot_core_projectors"]
        if n_proj == 1:
            dt_proj = ">i"
        else:
            dt_proj = f">{n_proj}i"
        co_chunk = {p: read_bin_data(f, dt_proj, n_proj * 4) for p in co}
        mat = np.array([read_bin_data(f, '>6d', 3 * 8 * 2) for i in range(np.prod([param_chunk[p] for p in params]))])
    mat.dtype = complex
    eels_mat_dict = dict()
    for p in params:
        eels_mat_dict[p] = param_chunk[p]
    for p in co:
        eels_mat_dict[p] = co_chunk[p]
    eels_mat_dict["transition_matrix"] = mat.reshape(
        eels_mat_dict["sum_num_kpoints"],
        eels_mat_dict["num_spins"],
        eels_mat_dict["tot_core_projectors"],
        eels_mat_dict["max_eigenvalues"],
        3,
    )
    return eels_mat_dict


def read_bands(filename, output_eV=True):
    """
    extract data from .bands file

    Parameters
    --------
    filename : str
        path to the .bands file
    output_ev : bool, default True
        whether output energy in eV (True) or hartree (False)

    Returns
    --------
    bands_dict : dict
        a dictionary of the extracted data
    """
    with open(filename, "rt") as f:
        # start reading header
        nkpnt = int(f.readline().split()[-1])
        nspin = int(f.readline().split()[-1])
        nelectrons = list(map(float, f.readline().split()[-nspin:]))
        nbands = list(map(int, f.readline().split()[-nspin:]))
        efermi = list(map(float, f.readline().split()[-nspin:]))
        f.readline()
        a = list(map(float, f.readline().split()))
        b = list(map(float, f.readline().split()))
        c = list(map(float, f.readline().split()))
        # finish reading header
        nk = list()
        kpts = list()
        wk = list()
        spin = list()
        eigenvalues = list()
        for _ in range(nkpnt):
            line = f.readline().split()
            nk.append(int(line[1]))
            kpts.append(list(map(float, line[2:5])))
            wk.append(float(line[5]))
            spin_k = list()
            eigenvalues_k = list()
            for nb in nbands:
                spin_k.append(int(f.readline().split()[-1]))
                eigenvalues_k.append(list(map(float, [f.readline() for _ in range(nb)])))
            spin.append(spin_k)
            eigenvalues.append(eigenvalues_k)
    bands_dict = {
        "num_kponts": nkpnt,
        "num_spins": nspin,
        "num_electrons": nelectrons,
        "num_eigenvalues": nbands,
        "efermi": efermi,
        "lattice_vectors": np.array([a, b, c]),
        "nk": nk,
        "kpoint": kpts,
        "kpoint_weights": wk,
        "spin": spin,
        "eigenvalues": np.array(eigenvalues)
    }
    if output_eV:
        hart2eV = 27.211396132
        bands_dict["efermi"] = np.array(bands_dict["efermi"]) * hart2eV
        bands_dict["eigenvalues"] = np.array(bands_dict["eigenvalues"]) * hart2eV
    return bands_dict


def get_spectrum(calc_dir=".", seed_name="case_elnes", output_eV=True, atomic_unit=False):
    """
    get primitive spectral data from a .bands file and a .eels_mat file

    Parameters
    --------
    calc_dir : str, default "."
        path to the directory containing .bands and .eels_mat
    seed_name : str, default "case_elnes"
        seed name of the calculation
    output_eV : bool, default True
        whether output energy in eV (True) or hartree (False)
    atomic_unit : bool, default False
        whether output dynamical structure factor in the unit of Bohr radius^2 (True) or angstrom^2 (False).

    Returns
    --------
    spectrum_dict : dict
        a dictionary of the spectral data
    """
    eels_mat_dict = read_eels_mat(os.path.join(calc_dir, f"{seed_name}.eels_mat"))
    en = np.square(np.abs(eels_mat_dict["transition_matrix"]))
    if not atomic_unit:
        en *= 0.529177210903**2 # Bohr radius^-2 to Angstrom^-2
    bands_dict = read_bands(os.path.join(calc_dir, f"{seed_name}.bands"), output_eV)
    spectrum_dict = {
        "eigenvalues": bands_dict["eigenvalues"],
        "efermi": bands_dict["efermi"],
        "num_electrons": bands_dict["num_electrons"],
        "kpoint_weights": bands_dict["kpoint_weights"],
        "num_spins": bands_dict["num_spins"],
        "tot_core_projectors": eels_mat_dict["tot_core_projectors"],
        "dsf":en
    }
    return spectrum_dict


def gaussian(x, c, w, s):
    """
    gaussian function for smearing

    Parameters
    --------
    x : list or numpy array
        1d list of energies
    c : float
        center position (=mu)
    w : float
        height scaling factor, weights
    s : float
        standard deviation of gaussian smearing (=sigma)

    Returns
    --------
    numpy array
        numpy array of gaussian distribution
    """
    return w / (np.sqrt(2. * np.pi) * s) * np.exp(-((x - c) / s)**2 / 2.)


def get_smeared_spectrum(energies, sigma=0.3, calc_dir=".", seed_name="case_elnes", e_origin="eigen_value", output_eV=True, atomic_unit=False):
    """
    get gaussian smeared spectra from a .bands file and a .eels_mat file

    Parameters
    --------
    energies : list or numpy array
        1d list of energies
    sigma : float, default 0.3
        standard deviation of gaussian smearing
    calc_dir : str, default "."
        path to the directory containing .bands and .eels_mat
    seed_name : str, default "case_elnes"
        seed name of the calculation
    e_origin : str, default "eigen_value"
        set energy origin
    output_eV : bool, default True
        whether output energy in eV (True) or hartree (False)
    atomic_unit : bool, default False
        whether output dynamical structure factor in the unit of Bohr radius^2 (True) or angstrom^2 (False).

    Returns
    --------
    spectrum : numpy array
        numpy array of smeared spectra

    Examples
    --------
    >>> !ls .
    case_elnes.bands    case_elnes.eels_mat
    >>> energies = np.arange(-4.999, 30.002, 0.001) # make an array for energies
    >>> sp = get_smeared_spectrum(energies, 0.3) # parse and make spectra
    >>> fig, ax = plt.subplots(1)
    >>> ax.set_xlabel("Energy (eV)")
    >>> ax.set_ylabel("Intensity (arb. u.)")
    >>> ax.plot(energies, sp[0, 0], label="x") # plot a spectrum for x component of the 1st core projection
    >>> ax.plot(energies, sp[0, 1], label="y") # plot a spectrum for y component of the 1st core projection
    >>> ax.plot(energies, sp[0, 2], label="z") # plot a spectrum for z component of the 1st core projection
    >>> ax.plot(energies, np.mean(sp[0], axis=0), label="total") # plot a total spectrum of the 1st core projection
    """
    spectrum = get_spectrum(calc_dir, seed_name, output_eV, atomic_unit)
    if e_origin == "eigen_value":
        e_origin_value = np.min(
            [
                [
                    ev[int((ne * spectrum["num_spins"] - 1) // 2)]
                    for ev, ne in zip(ev_k, spectrum["num_electrons"])
                ]
                for ev_k in spectrum["eigenvalues"]
            ]
        )
    elif e_origin == "efermi":
        e_origin_value = spectrum["efermi"]
    sp = [
        [
            [
                [
                    kw * np.array(
                        [
                            gaussian(energies, e, w, sigma)
                            for e, w in zip(spectrum["eigenvalues"][i_kp, i_spin] - e_origin_value, spectrum["dsf"][i_kp, i_spin, i_proj, :, i_ra])
                            if e >= 0.
                        ]
                    )
                    for i_kp, kw in enumerate(spectrum["kpoint_weights"])
                ]
                for i_spin in range(spectrum["num_spins"])
            ]
            for i_ra in range(3)
        ]
        for i_proj in range(spectrum["tot_core_projectors"])
    ]
    sp = np.sum(np.array(sp), axis=(2, 3, 4))
    sp *= 2. / spectrum["num_spins"] # consider spin multiplicity
    return sp

