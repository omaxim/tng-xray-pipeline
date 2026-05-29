# Investigation Handoff: TNG X-ray Pipeline vs Reference Discrepancy

## Summary

We have a working intrinsic X-ray pipeline (`tng-xray-pipeline-clean/`) that reproduces
TNG-Cluster reference projections. The main systematic offset is now largely resolved,
but a residual mass-dependent offset remains. This document is a handoff for further
investigation.

---

## Current State

Pipeline repo: `/vera/u/maoweyssi/Code/VeraWorkspace/tng-xray-pipeline-clean/`

The pipeline:
- Loads gas particles via `loader='zoom'` (same as TNG reference)
- Computes per-particle VAPEC emissivity using Dylan's APEC table (`xray_pipeline/apec_dylan.hdf5`)
- Uses APEC v3.0.9, AG89 solar abundances, total metallicity (not per-element)
- Applies photon energy weighting `× a = 1/(1+z)` (baked into the table)

### Remaining delta values after all corrections

| Halo | z | M200c | Δ log Lx (pipe − ref) |
|------|---|-------|----------------------|
| 18130773 | 1.0 | 1.2×10¹⁴ M☉ | +0.075 dex |
| 17186013 | 1.0 | 5.75×10¹³ M☉ | +0.253 dex |
| 5551172  | 4.0 | 5.5×10¹² M☉ | +0.208 dex |

**Key pattern:** offset is LARGER for LOWER-MASS clusters. This is mass-dependent,
not purely redshift-dependent.

---

## What Has Been Confirmed as NOT the Cause

- K-correction (zobs = z) — tried, makes it much worse (−1.7 dex at z=5)
- GFM_CoolingRate sign convention — confirmed correct
- T_min threshold — minor effect
- APEC table version — same (v3.0.9)
- FoF vs zoom loader — user confirmed Dylan uses zoom too; only ~0.03 dex difference
- Comoving vs physical kpc² in the reference image — reference r200c exactly matches physical R200c
- Scale factor read from header vs lookup table — already reading from header
- Abundance table (Asplund2009 vs AG89) — changing this had negligible effect
- Per-element VAPEC vs single-Z table — both give the same result once the × a correction is applied
- The `× a` correction itself — this is the big fix, reduced delta from ~1 dex to ~0.1-0.25 dex

---

## The Main Fix Already Applied

Dylan's APEC tables (`temet/cosmo/xray.py`, `apec_convert_tables()`) weight each
photon by its OBSERVED energy `E_emit / (1+z)` rather than rest-frame energy. This
makes his Lx = L_intrinsic × a. We apply the same factor in `gas.py`:

```python
lum_erg_s *= a   # match Dylan's photon energy convention (baked into apec_dylan.hdf5)
```

**Wait — with `apec_dylan.hdf5`, this line should NOT be present!** The table already
has the `1/(1+z)` factor baked in. Check `gas.py` to confirm `lum_erg_s *= a` is
absent when using `get_dylan_emissivity`.

---

## Where the Remaining ~0.1–0.25 dex Comes From

Using Dylan's exact APEC table (`apec_dylan.hdf5`) gives THE SAME result as
pyxsim + manual `× a`. So the remaining offset is NOT from the emissivity model.
Both models agree with each other but disagree with the TNG reference.

**The offset is mass-dependent, suggesting a gas particle selection or projection
geometry issue, not a spectral model issue.**

---

## Leads to Investigate

### Lead 1 (MOST LIKELY): Hydrogen number density formula

**Dylan's code** (`snap_fields_custom.py`):
```python
def nh(sim, ...):
    dens = codeDensToPhys(density, numDens=True)  # total number density
    nh = dens * 0.76  # constant hydrogen mass fraction assumed
    return nh
```

**Our code** (`gas.py`):
```python
n_H = X_H_par * rho_cgs / m_p   # per-particle X_H from GFM_Metals[:, 0]
```

Dylan uses constant X_H = 0.76; we use per-particle X_H from the simulation. For
hot ICM this is close, but check whether `codeDensToPhys(numDens=True)` converts to
number density using `m_p` or using a mean molecular weight. If Dylan divides by m_p
while we account for mean molecular weight, EM could differ.

### Lead 2: Cell volume — stored field vs mass/density

**Dylan's code**:
```python
volume_cm3 = codeVolumeToCm3(snapshotSubset("gas", "volume"))
```
He loads the stored Voronoi `Volume` field from the snapshot.

**Our code** (`gas.py`):
```python
V_cell = mass_cgs / rho_cgs
```
We derive volume from mass/density. These should be equivalent but may differ if
the stored `Volume` field uses a different unit conversion. Check:
- Is TNG's stored `Volume` in comoving or physical units?
- Does `codeVolumeToCm3` apply `× (a/h)³`?
- Our formula: `V = (M_code × UM/h) / (rho_code × h²/a³ × UM/UL³) = M_code × UL³ × a³ / (rho_code × h³)` — verify this matches the stored volume.

### Lead 3: GFM_CoolingRate masking

**Dylan's code**: Does NOT appear to apply a `GFM_CoolingRate >= 0` mask. His only
gas exclusion is the temperature cut via `temp_sfcold`.

**Our code** (`gas.py`):
```python
heat_mask = (cr >= 0.0)   # exclude UV-heated gas
lum_erg_s[heat_mask] = 0.0
```

At z=1, the fraction of heated particles is high (7.4M / 24.8M = 30% for halo 17186013!).
But these are cool particles that contribute negligible X-ray emission. Still worth checking
if Dylan includes them (they'd have low T → low Lambda → low Lx contribution).

### Lead 4: SPH projection vs reference projection method

The TNG reference projection may use a different SPH kernel or projection method
than our `sphviewer2`. If sphviewer2 spreads particle emission differently at the
image boundary (particles near the edge of the 4×R200c FOV), total Lx could differ.

**Test**: Compare `result.meta.L_x_r500c` (3D sphere, no projection) to the reference
image Lx for the same aperture. If the 3D sphere Lx matches but the projected image Lx
doesn't, the issue is in the projection step.

### Lead 5: Projection depth cut — HIGH PRIORITY

Our depth cut: `|z - z_halo| <= R200c` (±R200c along LOS, total depth 2×R200c).
Reference filename: `d=r200` — ambiguous. Two interpretations:
- **Half-depth = R200c** → total depth = 2×R200c  (matches ours ✓)
- **Full depth = R200c** → total depth = R200c, i.e., only ±R200c/2 (we'd include 2× too much!)

The second interpretation would give a constant offset regardless of redshift, but could
be mass-dependent if gas outside ±R200c/2 is more important for lower-mass halos (e.g.
because their gas fraction outside the core is higher).

**Test**: Check the reference HDF5 `box_size` attr. We know `box_size[0]` = lateral size
= 4×R200c. The `box_size[2]` (depth dimension) should tell us: is it 2×R200c or R200c?
```python
import h5py
with h5py.File('/virgotng/.../gas-xray_lum_0.5-5.0kev__2r200_d=r200.99.hdf5', 'r') as f:
    print(f['Halo_0'].attrs['box_size'])  # [lateral, lateral, depth] in ckpc/h
    r200c = f['r200c'][0]  # physical kpc
    # if depth = 2 × r200c × h/a → half-depth = R200c (our convention)
    # if depth = 1 × r200c × h/a → half-depth = R200c/2 (we include 2x too much)
```

**R200c itself is correct** — we verified `r200c_ref / r200c_phys = 1.00000` at both
snap 99 and snap 17 for multiple halos. R200c is not the issue, but the DEPTH
CONVENTION derived from it might be.

---

## How to Test

### Quick test for Lead 3 (cooling rate masking):
```python
# In gas.py, comment out heat_mask line and rerun the two z=1 halos
# Does delta decrease for halo B (5.75e13)?
```

### Quick test for Lead 4 (projection vs sphere):
```python
# Compare 3D sphere Lx to projected image Lx for both the reference and pipeline
import h5py, numpy as np
with h5py.File('results_ag89/results.hdf5', 'r') as f:
    grp = f['halos/snap050_halo17186013']
    lx_sphere_pipe = float(grp.attrs['L_x_r500c'])
    # compare to reference 3D Lx (if available)
```

### Quick test for Lead 1 (constant vs per-particle X_H):
```python
# In gas.py, replace X_H_par with np.full(n_par, 0.76) and see if delta changes
```

---

## Files to Read

- `xray_pipeline/gas.py` — main emissivity + EM calculation
- `xray_pipeline/apec.py` — Dylan's table loader (`get_dylan_emissivity`)
- `xray_pipeline/projection.py` — SPH projection
- `build_apec_table.py` — how Dylan's APEC table was built
- Dylan's code (already seen): `temet/cosmo/xray.py` — reference implementation
- Dylan's code: `temet/load/snap_fields_custom.py` — `nh`, `volume_cm3`, `temp_sfcold`

---

## Key Values

- Reference files: `/virgotng/mpia/TNG-Cluster/L680n8192TNG/postprocessing/projections/gas-xray_lum_0.5-5.0kev__2r200_d=r200.{snap}.hdf5`
- Dylan's APEC table: `xray_pipeline/apec_dylan.hdf5`
- Z_solar_AG89 = 0.023 (Dylan's value, used in metallicity normalization)
- FOV = 4×R200c, depth = ±R200c, N_PIX = 2000
- Test halos: snap=50 halo_id=17186013 (worst case, Δ=+0.25) and halo_id=18130773 (Δ=+0.07)
