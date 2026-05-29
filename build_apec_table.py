#!/usr/bin/env python3
"""
Build the TNG-reference-compatible APEC emissivity table.

Adapted from Dylan Nelson's temet/cosmo/xray.py (apec_convert_tables).
Exactly reproduces his methodology:
  - H and He only in the primary (cosmic) component
  - Everything else scales with total metallicity
  - AG89 solar abundances (matching APEC's internal assumption)
  - Band selection on emitted (rest-frame) photon energies
  - Photon energies converted to OBSERVED erg/photon: E_emit / (1+z)
    so the table stores Λ(T, Z, z) = Λ_rest(T, Z) × a

Output
------
    xray_pipeline/apec_dylan.hdf5
        /redshift           (17,)               float64  redshift grid
        /temp               (n_temp,)            float64  log10 kT [keV]
        /metal              (100,)               float64  log10 Z/Z_AG89
        /emis_0.5-5.0kev    (17, n_temp, 100)   float64  erg cm³ s⁻¹

Usage
-----
    python build_apec_table.py
"""

import os
import numpy as np
import astropy.io.fits as pyfits
import h5py

# ── physical constants ────────────────────────────────────────────────────────
HC_KEV_ANG = 12.39841984       # h·c  [keV·Angstrom]
ERG_IN_KEV = 1.602176634e-9    # erg per keV

# ── paths ─────────────────────────────────────────────────────────────────────
CACHE = os.path.expanduser('~/.cache/soxs')
PATH_CONT = os.path.join(CACHE, 'apec_v3.0.9_coco.fits')
PATH_LINE = os.path.join(CACHE, 'apec_v3.0.9_line.fits')
OUT_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'xray_pipeline', 'apec_dylan.hdf5')

# ── grid config (mirrors temet exactly) ──────────────────────────────────────
N_ENERGY  = 2000
GRID      = np.logspace(-3.5, 1.5, N_ENERGY + 1)   # keV edges
GRID_MID  = (GRID[1:] + GRID[:-1]) / 2
DE        = np.diff(GRID)

REDSHIFTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
             0.8, 0.9, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]
BANDS     = [[0.5, 5.0]]        # extend to other bands here if needed

N_METAL   = 100
METAL_RANGE = [-3.5, 1.0]      # log10 Z/Z_AG89


def main():
    for p in (PATH_CONT, PATH_LINE):
        if not os.path.exists(p):
            raise FileNotFoundError(f'Missing APEC file: {p}')

    # ── continuum + pseudo-continuum ──────────────────────────────────────────
    print(f'Reading {os.path.basename(PATH_CONT)} ...')
    with pyfits.open(PATH_CONT) as f:
        temp_kev = f[1].data.field('kT')
        n_temp   = temp_kev.size
        n_atom   = f[2].data.field('Z').size

        cont   = np.zeros((n_temp, n_atom, N_ENERGY))
        pseudo = np.zeros((n_temp, n_atom, N_ENERGY))

        for i in range(n_temp):
            d   = f[2 + i].data
            nc  = d.field('N_cont');   cb = d.field('E_Cont');   c = d.field('Continuum')
            np_ = d.field('N_pseudo'); pb = d.field('E_Pseudo'); p = d.field('Pseudo')
            for j in range(n_atom):
                cont[i, j]   = np.interp(GRID_MID, cb[j, :nc[j]],  c[j, :nc[j]])  * DE
                pseudo[i, j] = np.interp(GRID_MID, pb[j, :np_[j]], p[j, :np_[j]]) * DE

    # ── line emission ─────────────────────────────────────────────────────────
    print(f'Reading {os.path.basename(PATH_LINE)} ...')
    with pyfits.open(PATH_LINE) as f:
        line = np.zeros((n_temp, n_atom, N_ENERGY))
        for i in range(n_temp):
            Z_arr   = f[2 + i].data.field('Element')
            waveang = f[2 + i].data.field('Lambda')
            emis    = f[2 + i].data.field('Epsilon')
            ekeV    = HC_KEV_ANG / waveang
            idx     = np.clip(np.searchsorted(GRID, ekeV, side='left') - 1,
                              0, N_ENERGY - 1)
            for k, zi in enumerate(Z_arr):
                if zi < n_atom:
                    line[i, zi, idx[k]] += emis[k]

    # ── H+He primary, everything else metals (Dylan's convention) ────────────
    spec_prim  = (cont[:, 0] + pseudo[:, 0] + line[:, 0] +
                  cont[:, 1] + pseudo[:, 1] + line[:, 1])   # H + He only

    spec_metal = np.zeros((n_temp, N_ENERGY))
    for i in range(2, n_atom):
        spec_metal += cont[:, i] + pseudo[:, i] + line[:, i]

    # ── 3D grid [temp, metal, energy] ─────────────────────────────────────────
    metals       = np.linspace(METAL_RANGE[0], METAL_RANGE[1], N_METAL)
    spec_grid_3d = np.zeros((n_temp, N_METAL, N_ENERGY))
    for k in range(N_METAL):
        spec_grid_3d[:, k] = spec_prim + (10.0 ** metals[k]) * spec_metal

    # ── integrate bands at each redshift, weighting by observed photon energy ─
    energy_emit = GRID_MID * ERG_IN_KEV    # erg/photon: keV × (erg/keV) = erg

    print(f'Writing {OUT_PATH} ...')
    with h5py.File(OUT_PATH, 'w') as fh:
        fh['redshift'] = np.array(REDSHIFTS)
        fh['temp']     = np.log10(temp_kev)   # log10 kT [keV]
        fh['metal']    = metals                # log10 Z/Z_AG89

        for emin, emax in BANDS:
            emis_out = np.zeros((len(REDSHIFTS), n_temp, N_METAL))
            w = np.where((GRID_MID >= emin) & (GRID_MID <= emax))[0]

            for j, z in enumerate(REDSHIFTS):
                energy_obs   = energy_emit / (1.0 + z)          # observed erg/photon
                s_loc        = spec_grid_3d * energy_obs         # erg cm³/s per EM
                emis_out[j]  = s_loc[:, :, w].sum(axis=2)

            # floor to avoid zeros in log interpolation
            floor = emis_out[emis_out > 0].min() / 100
            emis_out[emis_out <= 0] = floor

            key = f'emis_{emin:.1f}-{emax:.1f}kev'
            fh[key] = emis_out
            print(f'  [{key}]  shape={emis_out.shape}  '
                  f'range=[{emis_out.min():.2e}, {emis_out.max():.2e}]')

    print(f'Done → {OUT_PATH}')


if __name__ == '__main__':
    main()
