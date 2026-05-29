"""
APEC spectral emissivity tables — per-element VAPEC model.

The TableCIEModel from pyxsim is built once per unique energy band (keV) and
cached via ``functools.lru_cache``.  All subsequent halos at the same band
reuse the cached Lambda arrays without any disk I/O.

Band-integrated emissivities
-----------------------------
``Lambda_c(T)``     : continuum + H/He emissivity          [erg s⁻¹ cm³]
``Lambda_other(T)`` : emissivity from APEC-tracked metals NOT individually
                      handled (Na, Al, Ar, Ca, Cr, Mn, Ni, …) at 1 solar unit
``Lambda_metals``   : shape (7, nT), per-element emissivities for the 7 TNG-
                      tracked metals C, N, O, Ne, Mg, Si, Fe, at 1 solar unit
                      each                                 [erg s⁻¹ cm³]
``X_sol``           : (7,) Asplund et al. (2009) solar mass fractions for the
                      same 7 elements

Per-particle luminosity in gas.py
-----------------------------------
    Z_i_sol[i] = GFM_Metals[:, 2+i] / X_sol[i]     (i=0..6: C,N,O,Ne,Mg,Si,Fe)
    Z_Fe_sol   = GFM_Metals[:, 8]   / X_sol[6]      (Fe as proxy for untracked metals)

    Lambda(T) = Lambda_c(T)
              + Z_Fe_sol          * Lambda_other(T)
              + sum_i Z_i_sol[i]  * Lambda_metals[i](T)

    L_i = n_e,i * n_H,i * V_cell,i * Lambda(T_i)    [erg/s]

Abundance table
---------------
Asplund et al. (2009) — "aspl" in soxs — matching the TNG GFM reference and
the Nelson et al. (2025) TNG-Cluster X-ray methodology.
"""

import functools
import numpy as np

from .constants import KEV_TO_ERG

# The 7 metals that TNG tracks individually via GFM_Metals columns 2–8.
# H and He are handled via cosmic_spec / emission measure — not varied here.
_TNG_METALS  = ['C', 'N', 'O', 'Ne', 'Mg', 'Si', 'Fe']
_ABUND_TABLE = 'angr'   # Anders & Grevesse (1989) — matches TNG reference convention

# Internal APEC grid parameters
_APEC_N_BINS  = 500
_APEC_KT_MIN  = 0.025   # keV  — minimum temperature (≈ 2.9×10⁵ K), matches pyxsim default
_APEC_KT_MAX  = 64.0    # keV  — maximum temperature (≈ 7.4×10⁸ K)


def _solar_mass_fractions() -> np.ndarray:
    """
    Compute Asplund 2009 solar mass fractions for C, N, O, Ne, Mg, Si, Fe
    from the soxs abundance and atomic-weight tables.

    Returns
    -------
    X_sol : (7,) ndarray  mass fractions in the order C,N,O,Ne,Mg,Si,Fe
    """
    from soxs.constants import abund_tables, atomic_weights
    aspl    = abund_tables[_ABUND_TABLE]
    # sum over all 30 elements soxs tracks (index = atomic number, 1-based)
    denom   = sum(aspl[z] * atomic_weights[z] for z in range(1, 31))
    Z_nums  = [6, 7, 8, 10, 12, 14, 26]   # C,N,O,Ne,Mg,Si,Fe
    return np.array([aspl[z] * atomic_weights[z] / denom for z in Z_nums])


@functools.lru_cache(maxsize=32)
def get_apec_vapec(
    emin_kev: float,
    emax_kev: float,
    zobs    : float = 0.0,
) -> tuple:
    """
    Build and cache the band-integrated per-element VAPEC emissivity tables.

    Parameters
    ----------
    emin_kev, emax_kev : float
        Observer-frame energy band boundaries [keV].
    zobs : float
        Snapshot redshift (z = 1/a − 1).

    Returns
    -------
    log_kT        : (nT,)    ndarray   log10(kT [keV])
    Lambda_c      : (nT,)    ndarray   continuum + H/He emissivity [erg/s/cm³]
    Lambda_other  : (nT,)    ndarray   untracked-metal emissivity at 1 solar unit
    Lambda_metals : (7, nT)  ndarray   per-element [C,N,O,Ne,Mg,Si,Fe] at 1 solar unit
    X_sol         : (7,)     ndarray   Asplund 2009 solar mass fractions
    T_lo, T_hi    : float              kT range [keV]
    """
    from pyxsim.spectral_models import TableCIEModel

    sm = TableCIEModel(
        'apec', emin_kev, emax_kev,
        _APEC_N_BINS, _APEC_KT_MIN, _APEC_KT_MAX,
        binscale='log',
        var_elem=_TNG_METALS,
        abund_table=_ABUND_TABLE,
    )
    sm.prepare_spectrum(zobs=zobs)

    wt = sm.emid * KEV_TO_ERG   # [photon energy in erg] per energy bin

    Lambda_c      = (sm.cosmic_spec * wt).sum(axis=1)           # (nT,)
    Lambda_other  = (sm.metal_spec  * wt).sum(axis=1)           # (nT,)
    Lambda_metals = np.array(
        [(sm.var_spec[i] * wt).sum(axis=1) for i in range(len(_TNG_METALS))]
    )                                                            # (7, nT)

    X_sol = _solar_mass_fractions()                              # (7,)

    return (
        np.log10(sm.Tvals),
        Lambda_c,
        Lambda_other,
        Lambda_metals,
        X_sol,
        float(sm.Tvals[0]),
        float(sm.Tvals[-1]),
    )
