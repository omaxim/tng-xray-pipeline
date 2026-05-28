"""
Simulation configurations.

Each SimConfig encodes everything the pipeline needs to interface with one
simulation: where the snapshot data lives and how to load gas particles.

Loader strategies
-----------------
'zoom'  : ``il.snapshot.loadOriginalZoom`` — loads gas in the high-resolution
          Lagrangian region around each zoom target.  **Default for TNG-Cluster.**
          Required at high z where the proto-cluster is fragmented across multiple
          FoF groups; ensures diffuse proto-cluster gas is included, matching the
          TNG reference 'sphMap_globalZoomOrig' convention.

'fof'   : ``il.snapshot.loadHalo`` — loads all gas particles belonging to the
          FoF group.  Correct for uniform-resolution simulations (TNG50/100/300).

Adding a new simulation
-----------------------
    from xray_pipeline.sim_config import SimConfig, SIM_CONFIGS
    SIM_CONFIGS['my_sim'] = SimConfig(
        name      = 'my_sim',
        base_path = '/path/to/sim/output',
        loader    = 'zoom',   # or 'fof'
        mass_cut  = 1e13,
    )
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class SimConfig:
    """Immutable configuration for one simulation."""

    name      : str
    base_path : str
    loader    : str   = 'zoom'    # 'zoom' | 'fof'
    mass_cut  : float = 1e13      # minimum M200c [M_sun] for halo selection


SIM_CONFIGS: dict[str, SimConfig] = {
    'tng-cluster': SimConfig(
        name      = 'tng-cluster',
        base_path = '/virgotng/mpia/TNG-Cluster/L680n8192TNG/output',
        loader    = 'zoom',   # zoom sim — must use zoom loader
        mass_cut  = 1e13,
    ),
    'tng300': SimConfig(
        name      = 'tng300',
        base_path = '/virgotng/universe/IllustrisTNG/L205n2500TNG/output',
        loader    = 'fof',
        mass_cut  = 1e13,
    ),
    'tng100': SimConfig(
        name      = 'tng100',
        base_path = '/virgotng/universe/IllustrisTNG/L75n1820TNG/output',
        loader    = 'fof',
        mass_cut  = 1e13,
    ),
    'tng50': SimConfig(
        name      = 'tng50',
        base_path = '/virgotng/universe/IllustrisTNG/L35n2160TNG/output',
        loader    = 'fof',
        mass_cut  = 1e13,
    ),
}
