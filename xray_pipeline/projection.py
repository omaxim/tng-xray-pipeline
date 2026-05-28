"""
X-ray surface-brightness projection and aperture luminosity.

Projection geometry
--------------------
Three orthogonal views are produced for each halo:

    View 0 : x-y plane   (depth = z axis)
    View 1 : x-z plane   (depth = y axis)
    View 2 : y-z plane   (depth = x axis)

The depth cut removes gas particles further than R200c from the halo centre
along the line of sight, matching the TNG postprocessing convention.

Image units
-----------
Output arrays are log10(surface brightness) in erg s⁻¹ kpc⁻².
Pixels with zero or negligible signal are set to log10(_SB_FLOOR) = -100.

Aperture luminosity
--------------------
``compute_lx_sphere`` integrates the true 3D sphere of radius R500c,
independent of any projected image.
"""

from __future__ import annotations

import numpy as np

from .gas import GasParticles

# Each entry (ax_h, ax_v, ax_d) maps: horizontal axis, vertical axis, depth axis
_VIEW_AXES = [
    (1, 0, 2),   # view 0: x-y plane, depth=z
    (2, 0, 1),   # view 1: x-z plane, depth=y
    (2, 1, 0),   # view 2: y-z plane, depth=x
]

_SB_FLOOR = 1e-100   # erg/s/kpc² — replaces zero/negative pixels before log10


def compute_lx_sphere(
    p_kpc      : np.ndarray,
    lum_erg_s  : np.ndarray,
    center_kpc : np.ndarray,
    radius_kpc : float,
) -> float:
    """
    Integrate luminosity for all particles within a 3D sphere.

    Parameters
    ----------
    p_kpc      : (N, 3) ndarray   particle positions [physical kpc]
    lum_erg_s  : (N,)   ndarray   per-particle luminosity [erg/s]
    center_kpc : (3,)   array_like halo centre [physical kpc]
    radius_kpc : float            sphere radius [physical kpc]

    Returns
    -------
    L_x : float   total luminosity inside the sphere [erg/s]
    """
    r2   = np.sum((p_kpc - center_kpc) ** 2, axis=1)
    mask = r2 < radius_kpc ** 2
    return float(lum_erg_s[mask].sum())


def project_view(
    gas          : GasParticles,
    group_pos_kpc: np.ndarray,
    R200c_kpc    : float,
    view         : int,
    r_max        : int  = 11,
    nthreads     : int  = 8,
    lbox_kpc     : float | None = None,
) -> np.ndarray:
    """
    Project gas particles onto a 2D image using the SPH kernel.

    Parameters
    ----------
    gas           : GasParticles   pre-loaded particle data
    group_pos_kpc : (3,) ndarray   halo centre in physical kpc
    R200c_kpc     : float          virial radius [kpc]; sets FOV and depth cut
    view          : int (0–2)      projection plane (see module docstring)
    r_max         : int            image resolution as 2^r_max pixels per side
    nthreads      : int            number of C-threads for sphviewer2
    lbox_kpc      : float or None  simulation box size [physical kpc]; passed to
                                   sphviewer2 but has no effect when periodic=False

    Returns
    -------
    log_sb : (2^r_max, 2^r_max) ndarray   log10 surface brightness [erg/s/kpc²]
    """
    import sphviewer2

    ax_h, ax_v, ax_d = _VIEW_AXES[view]
    fov_kpc = 4.0 * R200c_kpc

    cen_h = group_pos_kpc[ax_h]
    cen_v = group_pos_kpc[ax_v]
    cen_d = group_pos_kpc[ax_d]

    # Depth cut: ±R200c along the line of sight
    mask = np.abs(gas.p_kpc[:, ax_d] - cen_d) <= R200c_kpc
    p    = gas.p_kpc[mask]
    lum  = gas.lum_erg_s[mask]
    h_s  = gas.h_kpc[mask]

    _lbox = lbox_kpc if lbox_kpc is not None else 2e6   # dummy > any TNG box

    image, _ = sphviewer2.render(
        x=p[:, ax_v], y=p[:, ax_h], z=p[:, ax_d],
        h=h_s, m=lum,
        Lbox=_lbox, extent=fov_kpc,
        xc=cen_v, yc=cen_h, zc=cen_d,
        periodic=False, r_max=r_max, num_threads=nthreads,
    )

    return np.log10(np.where(image > _SB_FLOOR, image, _SB_FLOOR))
