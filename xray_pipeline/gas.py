"""
Gas particle loading and per-particle X-ray luminosity.

TNG field conventions
---------------------
- ``GFM_Metals[:, 0..9]`` : mass fractions for H, He, C, N, O, Ne, Mg, Si, Fe,
  and all remaining metals (column 9).  TNG-specific column order.
- ``GFM_Metallicity`` : total metal mass fraction (fallback when GFM_Metals absent).
- ``InternalEnergy`` : code units (UV² = km²/s²); temperature derived from the
  mean molecular weight of a fully-ionised plasma.
- ``StarFormationRate`` : [M_sun/yr]; non-zero cells contribute zero X-ray
  luminosity, matching the TNG/Nelson+2025 convention.

Spectral model
--------------
Per-element VAPEC (Asplund et al. 2009 solar abundances), identical to the
methodology used for the TNG-Cluster postprocessing reference projections:

    Lambda(T) = Lambda_c(T)
              + Z_Fe_sol          * Lambda_other(T)
              + sum_i Z_i_sol[i]  * Lambda_metals[i](T)

where Z_i_sol = GFM_Metals[:, 2+i] / X_sol[i] for i = C,N,O,Ne,Mg,Si,Fe,
and Fe is also used as a proxy for untracked APEC metals (X/Fe=1, Nelson+2025).

Fallback
--------
When GFM_Metals is absent (some TNG-Cluster high-z snapshot chunks), the total
GFM_Metallicity is distributed in solar ratios across all metal channels.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from astropy.constants import m_p, k_B

from .constants import KPC_TO_CM
from ._il import il

# GFM_Metals column indices (TNG convention, same across TNG300/100/50/Cluster)
_COL_H   = 0
_COL_HE  = 1
_COL_C   = 2   # first metal; columns 2–8 are C, N, O, Ne, Mg, Si, Fe in order
_COL_FE  = 8
_COL_OTH = 9   # all remaining metals not individually tracked by TNG

_X_H_PRIMORDIAL = 0.76   # fallback H mass fraction when GFM_Metals is absent

# Boltzmann constant in keV/K
_K_TO_KEV = k_B.cgs.value / 1.602176634e-9   # [erg/K] / [erg/keV]

_T_MIN_K       = 1.0e6   # minimum temperature [K]

_FIELDS_FULL    = ('Coordinates', 'Masses', 'Density', 'InternalEnergy',
                    'ElectronAbundance', 'StarFormationRate', 'GFM_CoolingRate',
                    'GFM_Metals', 'GFM_Metallicity')
_FIELDS_NO_MET  = ('Coordinates', 'Masses', 'Density', 'InternalEnergy',
                    'ElectronAbundance', 'StarFormationRate', 'GFM_CoolingRate',
                    'GFM_Metallicity')
_FIELDS_MINIMAL = ('Coordinates', 'Masses', 'Density', 'InternalEnergy',
                    'ElectronAbundance', 'StarFormationRate',
                    'GFM_Metallicity')


@dataclass
class GasParticles:
    """
    All gas-particle arrays for one halo, expressed in physical units.

    Attributes
    ----------
    p_kpc      : (N, 3) ndarray   particle positions [physical kpc]
    lum_erg_s  : (N,)   ndarray   VAPEC band luminosity per particle [erg/s]
    h_kpc      : (N,)   ndarray   SPH / Voronoi cell effective radius [kpc]
    """

    p_kpc     : np.ndarray
    lum_erg_s : np.ndarray
    h_kpc     : np.ndarray


def load_gas(
    halo_id   : int,
    snap      : int,
    base_path : str,
    loader    : str,
    grp       : dict,
    h         : float,
    a         : float,
    UM        : float,
    UL        : float,
    UV        : float,
    emin_kev  : float,
    emax_kev  : float,
    verbose   : bool = True,
) -> GasParticles:
    """
    Load gas particles for one halo and compute per-particle VAPEC luminosity.

    Parameters
    ----------
    halo_id, snap, base_path : halo and simulation identifiers
    loader    : 'fof' uses ``il.snapshot.loadHalo``; 'zoom' uses
                ``il.snapshot.loadOriginalZoom``
    grp       : pre-loaded ``il.groupcat.loadSingle`` result
    h, a      : Hubble parameter and scale factor from the group catalog header
    UM, UL, UV: AREPO code units — mass [g], length [cm], velocity [cm/s]
    emin_kev, emax_kev : rest-frame energy band [keV]
    verbose   : whether to print particle counts and luminosity summary

    Returns
    -------
    GasParticles
    """
    def _load(fields):
        if loader == 'zoom':
            return il.snapshot.loadOriginalZoom(base_path, snap, halo_id,
                                                'gas', fields=list(fields))
        return il.snapshot.loadHalo(base_path, snap, halo_id,
                                    'gas', fields=list(fields))

    try:
        gas = _load(_FIELDS_FULL)
        has_metals = True
    except Exception:
        try:
            gas = _load(_FIELDS_NO_MET)
            has_metals = False
        except Exception:
            gas = _load(_FIELDS_MINIMAL)
            has_metals = False

    n_par = len(gas['Masses'])
    sfr   = gas['StarFormationRate']
    sf_mask = sfr > 0.0

    # Net cooling rate cut: in TNG/AREPO, GFM_CoolingRate stores dU/dt.
    # Negative = energy loss = net cooling (include; this is the hot ICM).
    # Positive = energy gain = net heating by UV background (exclude).
    cr          = gas.get('GFM_CoolingRate')
    has_cr      = cr is not None
    heat_mask   = (cr >= 0.0) if has_cr else np.zeros(n_par, dtype=bool)

    if verbose:
        flag = '' if has_metals else '  [GFM_Metals absent — solar-ratio fallback]'
        print(f'    {n_par:,} gas particles loaded{flag}')
        if sf_mask.any():
            print(f'    {sf_mask.sum():,} star-forming particles → lum = 0')
        if has_cr and heat_mask.any():
            print(f'    {heat_mask.sum():,} net-heated particles (GFM_CoolingRate≥0) → lum = 0')
        elif not has_cr:
            print(f'    GFM_CoolingRate absent — net-heating cut skipped')

    # ── positions: comoving ckpc/h → physical kpc ─────────────────────────────
    p_kpc = gas['Coordinates'] * (a / h)

    # ── density and mass in CGS ────────────────────────────────────────────────
    rho_cgs  = gas['Density']  * (h**2 / a**3) * UM / UL**3
    mass_cgs = gas['Masses']   * UM / h

    # ── hydrogen mass fraction ─────────────────────────────────────────────────
    if has_metals:
        X_H_par = gas['GFM_Metals'][:, _COL_H]
    else:
        X_H_par = np.full(n_par, _X_H_PRIMORDIAL, dtype=np.float32)

    # ── emission measure: n_e * n_H * V_cell ──────────────────────────────────
    n_H = X_H_par * rho_cgs / m_p.cgs.value
    x_e = gas['ElectronAbundance']
    em  = x_e * n_H**2 * (mass_cgs / rho_cgs)   # n_e * n_H * V  [cm⁻³]

    # ── temperature [K] ────────────────────────────────────────────────────────
    mu  = 4.0 / (1.0 + 3.0 * X_H_par + 4.0 * X_H_par * x_e)
    T_K = (2.0 / 3.0) * gas['InternalEnergy'] * UV**2 * mu * m_p.cgs.value / k_B.cgs.value

    cold_mask = T_K < _T_MIN_K   # T < 1×10⁶ K: zero emission

    if verbose and cold_mask.any():
        print(f'    {cold_mask.sum():,} cold particles (T<1×10⁶ K) → lum = 0')

    # ── Dylan-style APEC table lookup ─────────────────────────────────────────
    # Uses total metallicity (GFM_Metallicity / Z_solar_AG89) — H+He only in
    # primary component, everything else scales with Z.  Photon energies are
    # weighted by observed energy E_emit/(1+z), so no separate × a needed.
    from .apec import get_dylan_emissivity, Z_SOLAR_AG89

    z_snap = 1.0 / a - 1.0
    kT_keV = np.maximum(T_K * _K_TO_KEV, 1e-4)
    log_kT = np.log10(kT_keV)

    Z_met          = gas.get('GFM_Metallicity', np.zeros(n_par))
    metal_logSolar = np.log10(np.maximum(Z_met / Z_SOLAR_AG89, 1e-6))

    Lambda    = get_dylan_emissivity(log_kT, metal_logSolar, z_snap, emin_kev, emax_kev)
    lum_erg_s = em * Lambda

    # Apply all three exclusion criteria from the TNG postprocessing reference
    lum_erg_s[cold_mask]  = 0.0   # T < 1×10⁶ K
    lum_erg_s[sf_mask]    = 0.0   # star-forming
    lum_erg_s[heat_mask]  = 0.0   # net-heated (GFM_CoolingRate ≥ 0)

    if verbose:
        print(f'    lum: total={lum_erg_s.sum():.3e}  max={lum_erg_s.max():.3e} erg/s')

    # ── effective cell radius from Voronoi cell volume = mass / density ────────
    V_cell_kpc3 = (mass_cgs / rho_cgs) / KPC_TO_CM**3
    h_kpc       = (3.0 * V_cell_kpc3 / (4.0 * np.pi)) ** (1.0 / 3.0)

    return GasParticles(p_kpc=p_kpc, lum_erg_s=lum_erg_s, h_kpc=h_kpc)
