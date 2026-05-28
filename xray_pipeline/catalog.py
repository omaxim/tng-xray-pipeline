"""
Halo catalogue metadata extraction.

Pulls quantities from the pre-loaded group and subhalo catalogue dicts
returned by ``il.groupcat.loadSingle``.  All output quantities are in
physical (non-comoving) units.

TNG particle-type indices (AREPO convention)
--------------------------------------------
- Type 0 : gas
- Type 4 : stars (and wind particles)
- Type 5 : black holes
"""

from __future__ import annotations
from dataclasses import dataclass, asdict

import numpy as np

# AREPO GroupMassType / SubhaloMassType particle-type indices
_PTYPE_GAS   = 0
_PTYPE_STARS = 4


@dataclass
class CatalogMeta:
    """
    Physical halo properties derived from the TNG group/subhalo catalogue.

    All masses in M_sun, all lengths in physical kpc, luminosity in erg/s.
    ``L_x_r500c`` is filled by ``process_halo`` after gas loading.
    """

    halo_id        : int
    snap           : int
    sim            : str
    a              : float    # scale factor (= 1 / (1+z))

    M200c_msun     : float
    M500c_msun     : float
    M_gas_msun     : float
    R200c_kpc      : float    # physical kpc
    R500c_kpc      : float    # physical kpc

    M_star_BCG_msun: float
    M_BH_msun      : float
    BHAR_code      : float    # BH accretion rate in code units
    SFR_msun_yr    : float
    SFR_halfmass   : float

    L_x_r500c      : float = float('nan')   # filled after gas loading

    def to_dict(self) -> dict:
        """Return all fields as a flat Python dict."""
        return asdict(self)


def load_catalog_meta(
    grp    : dict,
    sub    : dict,
    h      : float,
    a      : float,
    halo_id: int,
    snap   : int,
    sim    : str,
) -> CatalogMeta:
    """
    Extract catalogue metadata from pre-loaded illustris_python dicts.

    Parameters
    ----------
    grp, sub : dicts from ``il.groupcat.loadSingle(..., haloID=...)`` and
               ``il.groupcat.loadSingle(..., subhaloID=GroupFirstSub)``
    h, a     : Hubble parameter and scale factor from the group catalogue header
    halo_id, snap, sim : identifiers stored verbatim on the returned object

    Returns
    -------
    CatalogMeta  (L_x_r500c is NaN until set by the caller)
    """
    return CatalogMeta(
        halo_id         = halo_id,
        snap            = snap,
        sim             = sim,
        a               = float(a),

        M200c_msun      = float(grp['Group_M_Crit200']              * 1e10 / h),
        M500c_msun      = float(grp['Group_M_Crit500']              * 1e10 / h),
        M_gas_msun      = float(grp['GroupMassType'][_PTYPE_GAS]    * 1e10 / h),
        R200c_kpc       = float(grp['Group_R_Crit200']              * a    / h),
        R500c_kpc       = float(grp['Group_R_Crit500']              * a    / h),

        M_star_BCG_msun = float(sub['SubhaloMassType'][_PTYPE_STARS] * 1e10 / h),
        M_BH_msun       = float(sub['SubhaloBHMass']                 * 1e10 / h),
        BHAR_code       = float(sub['SubhaloBHMdot']),
        SFR_msun_yr     = float(grp['GroupSFR']),
        SFR_halfmass    = float(sub['SubhaloSFRinHalfRad']),
    )
