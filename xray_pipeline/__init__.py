"""
xray_pipeline — intrinsic X-ray projection pipeline for TNG simulations.

Public API
----------
process_halo(halo_id, snap, base_path, loader='zoom', ...)
    Full pipeline for one halo: catalogue metadata + L_x + 3 SPH-projected images.
    Returns a HaloResult dataclass.

    loader='zoom'  uses loadOriginalZoom — required for TNG-Cluster (default)
    loader='fof'   uses loadHalo         — for TNG50/100/300 uniform boxes

HaloResult
    .meta   : CatalogMeta  — physical properties (M_sun, kpc, erg/s)
    .images : (2000, 2000, 3) float32  — log10 SB [erg/s/kpc²], three views

CatalogMeta
    Physical halo properties from the TNG group catalogue.

SimConfig, SIM_CONFIGS
    Per-simulation configuration (base_path, loader, mass_cut).

Example
-------
>>> from xray_pipeline import process_halo, SIM_CONFIGS
>>> cfg    = SIM_CONFIGS['tng-cluster']
>>> result = process_halo(halo_id=0, snap=99, base_path=cfg.base_path)
>>> print(f"log L_x = {result.meta.L_x_r500c:.3e} erg/s")
>>> print(result.images.shape)   # (2000, 2000, 3)
"""

from .pipeline   import process_halo, HaloResult
from .catalog    import CatalogMeta
from .sim_config import SimConfig, SIM_CONFIGS
from . import constants

__all__ = [
    'process_halo',
    'HaloResult',
    'CatalogMeta',
    'SimConfig',
    'SIM_CONFIGS',
    'constants',
]
