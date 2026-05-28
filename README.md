# tng-xray-pipeline

Intrinsic rest-frame X-ray projection pipeline for TNG simulations.

Reproduces the TNG-Cluster postprocessing methodology: per-particle VAPEC emissivity
(APEC v3.0.9, Asplund+2009 abundances), projected onto 2000×2000 pixel images in
three orthogonal views, with SPH-kernel smoothing ("fuzz").

---

## Pipeline steps

```
snapshot  →  load gas (zoom)  →  compute T, n_e, n_H, V_cell
          →  VAPEC emissivity  →  per-particle L_x [erg/s]
          →  SPH projection    →  log₁₀ SB image [erg/s/kpc²]
```

1. **Load cosmological units** — `a`, `h`, AREPO code unit conversions from the snapshot header (not a lookup table).
2. **Load halo catalogue** — M200c, M500c, R200c, R500c, BCG stellar mass, BH mass, SFR from `GroupCat`.
3. **Load gas particles** — `zoom` loader (`loadOriginalZoom`) captures the full Lagrangian region, including diffuse proto-cluster gas at high z that the FoF group misses.
4. **Compute emission measure** — `EM = n_e × n_H × V_cell` in physical units, from IllustrisTNG fields `Density`, `Masses`, `ElectronAbundance`, `GFM_Metals`.
5. **Compute temperature** — from `InternalEnergy` and mean molecular weight; exclude star-forming particles (SFR > 0) and net-heated gas (GFM_CoolingRate ≥ 0) and cold gas (T < 10⁶ K).
6. **VAPEC emissivity** — per-element APEC tables via `pyxsim`, band-integrated to [emin, emax] keV at zobs = 0 (intrinsic rest-frame; no K-correction).
7. **SPH projection** — `sphviewer2` renders each particle with its Voronoi-cell kernel radius; output is log₁₀(SB) [erg/s/kpc²], FOV = 4 × R200c, depth = ±R200c.
8. **Aperture luminosity** — `L_x(R500c)` from a 3D sphere, independent of the projected image.

---

## Loader choice: zoom vs FoF

| Loader | IllustrisTNG function | When to use |
|--------|----------------------|-------------|
| **`zoom`** (default) | `loadOriginalZoom` | TNG-Cluster (zoom sim). Loads the full high-res Lagrangian region. Required at high z where the proto-cluster is fragmented across multiple FoF groups. |
| `fof` | `loadHalo` | TNG50/100/300 (uniform box). Loads all FoF group members. Correct when the halo is fully resolved and not a zoom target. |

Override via the `loader` argument to `process_halo()` or the `--loader` flag on `run_sample.py`.

---

## Installation

```bash
# 1. Clone
git clone https://github.com/omaxim/tng-xray-pipeline.git
cd tng-xray-pipeline

# 2. Virtual environment (recommended)
python3 -m venv xray_venv && source xray_venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. illustris_python (not on PyPI — clone alongside this repo or set env var)
git clone https://github.com/illustristng/illustris_python.git
# OR set ILLUSTRIS_PYTHON_PATH=/path/to/illustris_python in your environment
```

> **MPIA Vera cluster**: load `module anaconda/3/2023.03` and activate
> `/vera/u/<user>/Code/VeraWorkspace/XRAY_ML/xray_venv` — all dependencies are
> already installed there.

---

## Quick start

### Single halo

```python
from xray_pipeline import process_halo, SIM_CONFIGS

cfg    = SIM_CONFIGS['tng-cluster']
result = process_halo(halo_id=0, snap=99,
                      base_path=cfg.base_path,
                      sim='tng-cluster')

print(f"log L_x(R500c) = {result.meta.L_x_r500c:.3e} erg/s")
print(f"images shape   = {result.images.shape}")   # (2000, 2000, 3)
```

### Batch run (saves HDF5)

```bash
python run_sample.py \
    --sim tng-cluster \
    --n   30          \
    --seed 42         \
    --out-dir results/
```

Output: `results/results.hdf5`, `results/sample_list.csv`, `results/run.log`

### SLURM (Vera)

```bash
sbatch slurm/sample.sh results/ 30
```

---

## Output format

`results.hdf5`:

```
/config                 attrs: sim, band, seed, generated_at, …
/sample_list            (N, 3) int32  [snap, halo_id, zoom_idx]
/halos/
    snap{snap:03d}_halo{halo_id:07d}/
        images          (2000, 2000, 3) float32  log10 erg/s/kpc²
        attrs:          all CatalogMeta fields (M200c, R200c, a, z, …)
```

---

## Showcase notebook

Open `showcase.ipynb` for:
- Single-halo walkthrough
- 30-halo validation figure (pipeline vs TNG reference projections)
- ΔlogLx vs redshift scatter with empirical correction analysis
- Discussion of the known systematic offset (see below)

---

## Known systematic offset

When comparing integrated Lx to TNG-Cluster reference projections
(`gas-xray_lum_0.5-5.0kev__2r200_d=r200.{snap}.hdf5`):

| Snapshot | z   | Δ log Lx (pipe − ref) |
|----------|-----|----------------------|
| 99       | 0.0 | ≈ 0.00 dex           |
| 84       | 0.2 | ≈ +0.05 dex          |
| 50       | 1.0 | ≈ +0.25 dex          |
| 33       | 2.0 | ≈ +0.45 dex          |
| 21       | 4.0 | ≈ +0.87 dex          |
| 17       | 5.0 | ≈ +1.1 dex           |

Best empirical power-law correction: multiply Lx by `a^1.26` (i.e. divide by `(1+z)^1.26`).

**Confirmed not the cause:** K-correction, GFM_CoolingRate sign, T_min threshold,
APEC table version, FoF vs zoom loader, comoving/physical coordinate confusion.

**Leading hypothesis:** APEC version difference between this pipeline (v3.0.9) and
the version used to generate the TNG-Cluster reference projections. At high z,
clusters are cooler (~1–3 keV) and soft X-ray line emission dominates; APEC versions
differ significantly in that regime.  Contact the TNG-Cluster team for the reference
APEC version.

---

## Citation / references

- TNG-Cluster simulations: Nelson et al. (2024), [arXiv:2405.12373](https://arxiv.org/abs/2405.12373)
- X-ray methodology: Nelson et al. (2025), TNG-Cluster X-ray paper
- APEC emissivity tables: Smith et al. (2001), [atomdb.org](http://www.atomdb.org)
- Solar abundances: Asplund et al. (2009)
- pyxsim: ZuHone et al., [hea-www.cfa.harvard.edu/~jzuhone/pyxsim](https://hea-www.cfa.harvard.edu/~jzuhone/pyxsim)
