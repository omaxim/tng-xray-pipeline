"""
Top-level halo processing pipeline.

``process_halo`` is the single entry point: catalogue metadata, L_x within
R500c (3D sphere), and three projected surface-brightness images.

Convention
----------
All luminosities are **intrinsic rest-frame** quantities: VAPEC band evaluated
at [emin_kev, emax_kev] for every snapshot regardless of redshift (zobs=0,
no K-correction).  This matches the TNG postprocessing reference:
    partField: xray_lum_0.5-5.0kev  (rest-frame, intrinsic)
"""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from ._il      import il
from .catalog  import CatalogMeta, load_catalog_meta
from .gas      import GasParticles, load_gas
from .projection import compute_lx_sphere, project_view

import glob, os, h5py


@dataclass
class HaloResult:
    """
    All outputs from processing one halo.

    Attributes
    ----------
    meta   : CatalogMeta   physical properties and luminosity
    images : (N, N, 3) ndarray float32   log10 surface brightness [erg/s/kpc²]
             Axis 2 indexes the three projection views (xy / xz / yz planes).
    """

    meta  : CatalogMeta
    images: np.ndarray


# ── cosmological unit reading ─────────────────────────────────────────────────

def _cosmo_units(base_path: str, snap: int) -> tuple[float, float, float, float, float, float]:
    """
    Return (h, a, z, UL, UM, UV) from the group catalogue header and snapshot.

    Both a and z are read directly from the header — z is not derived from a,
    since TNG snapshots are at exact integer redshifts stored in the header.
    """
    gc      = il.groupcat.loadHeader(base_path, snap)
    pattern = os.path.join(base_path, f'snapdir_{snap:03d}', f'snap_{snap:03d}.0.hdf5')
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f'Snapshot file not found — searched: {pattern}')
    with h5py.File(matches[0], 'r') as f:
        u = dict(f['Header'].attrs)
    return (
        gc['HubbleParam'],
        gc['Time'],        # scale factor a
        gc['Redshift'],    # z, read directly — not derived from a
        u['UnitLength_in_cm'],
        u['UnitMass_in_g'],
        u['UnitVelocity_in_cm_per_s'],
    )


# ── main entry point ──────────────────────────────────────────────────────────

def process_halo(
    halo_id  : int,
    snap     : int,
    base_path: str,
    loader   : str   = 'zoom',
    sim      : str   = '',
    emin_kev : float = 0.5,
    emax_kev : float = 5.0,
    r_max    : int   = 11,
    nthreads : int   = 8,
    verbose  : bool  = True,
) -> HaloResult:
    """
    Process one halo: load gas once, compute metadata + L_x + 3 projections.

    Parameters
    ----------
    halo_id, snap, base_path : halo and simulation identifiers
    loader      : 'zoom' (default) or 'fof' — see sim_config.py
    sim         : simulation name stored verbatim in the returned metadata
    emin_kev, emax_kev : rest-frame energy band [keV]; applied at all z (no K-correction)
    r_max       : image resolution as 2^r_max pixels per side (default 11 → 2048 px)
    nthreads    : total CPU threads; split across three parallel projections
    verbose     : print progress to stdout

    Returns
    -------
    HaloResult
        ``meta.L_x_r500c`` — intrinsic rest-frame luminosity [erg/s]
        ``images``          — (2000, 2000, 3) float32  log10 erg/s/kpc²
    """
    # ── cosmological units ────────────────────────────────────────────────────
    h, a, z, UL, UM, UV = _cosmo_units(base_path, snap)

    if verbose:
        print(f'  halo_id={halo_id}  snap={snap}  z={z:.4g}  a={a:.6f}  sim={sim or base_path}')

    # ── catalogue ─────────────────────────────────────────────────────────────
    grp = il.groupcat.loadSingle(base_path, snap, haloID=halo_id)
    first_sub_id = int(grp['GroupFirstSub'])
    if first_sub_id < 0:
        raise ValueError(
            f'halo {halo_id} (snap {snap}) has no subhalos (GroupFirstSub=-1)')
    sub = il.groupcat.loadSingle(base_path, snap, subhaloID=first_sub_id)

    meta = load_catalog_meta(grp, sub, h, a, halo_id, snap, sim)

    # ── gas loading + luminosity ──────────────────────────────────────────────
    if verbose:
        print(f'  Loading gas (loader={loader}, band={emin_kev:.3f}–{emax_kev:.3f} keV) ...')

    gas = load_gas(
        halo_id, snap, base_path, loader, grp,
        h, a, UM, UL, UV,
        emin_kev, emax_kev,
        verbose=verbose,
    )

    group_pos_kpc = grp['GroupPos'] * a / h   # physical kpc

    # ── L_x: 3D sphere within R500c, intrinsic rest-frame ────────────────────
    meta.L_x_r500c = compute_lx_sphere(gas.p_kpc, gas.lum_erg_s,
                                        group_pos_kpc, meta.R500c_kpc)

    if verbose:
        print(f'  L_x(R500c) = {meta.L_x_r500c:.3e} erg/s  '
              f'(log10 = {np.log10(meta.L_x_r500c):.3f})')

    # ── projections: 3 views in parallel ─────────────────────────────────────
    nthreads_per_view = max(1, nthreads // 3)
    n_parallel        = min(3, nthreads)

    if verbose:
        print(f'  Projecting 3 views ({nthreads_per_view} threads each) ...')

    def _proj(view: int) -> tuple[int, np.ndarray]:
        img = project_view(gas, group_pos_kpc, meta.R200c_kpc,
                           view, r_max=r_max, nthreads=nthreads_per_view)
        return view, img

    with ThreadPoolExecutor(max_workers=n_parallel) as pool:
        view_images = dict(pool.map(_proj, range(3)))

    raw_stack = np.stack(
        [view_images[v].astype(np.float32) for v in range(3)],
        axis=2,
    )
    images = _centre_crop(raw_stack, 2000)

    return HaloResult(meta=meta, images=images)


def _centre_crop(img: np.ndarray, target: int) -> np.ndarray:
    """Centre-crop a (H, W, C) array to (target, target, C)."""
    h, w = img.shape[:2]
    if h == target and w == target:
        return img
    off_h = (h - target) // 2
    off_w = (w - target) // 2
    return img[off_h:off_h + target, off_w:off_w + target]
