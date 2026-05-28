#!/usr/bin/env python3
"""
Run the intrinsic X-ray pipeline on a random sample of TNG-Cluster halos.

Samples uniformly from all (snap, halo_id) pairs across the 13 reference-
projection snapshots.  All results use zobs=0 (intrinsic rest-frame 0.5–5 keV,
no K-correction).

Usage
-----
    python run_sample.py --sim tng-cluster --n 30 --seed 42 --out-dir results/

Output
------
    <out-dir>/results.hdf5      images + metadata per halo
    <out-dir>/sample_list.csv   the sampled (snap, halo_id, zoom_idx) rows
    <out-dir>/run.log           full stdout log

HDF5 structure
--------------
    /config               attrs: sim, band, seed, generated_at, …
    /sample_list          (N, 3) int32  columns: snap, halo_id, zoom_idx
    /halos/
        snap{snap:03d}_halo{halo_id:07d}/
            images        (2000, 2000, 3) float32  log10 erg/s/kpc²
            attrs:        CatalogMeta fields + zoom_idx
"""

import argparse
import os
import sys
import time
import traceback
from datetime import datetime, timezone

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from xray_pipeline import process_halo, SIM_CONFIGS

# ── reference snapshots ────────────────────────────────────────────────────────

_PROJ_DIR  = ('/virgotng/mpia/TNG-Cluster/L680n8192TNG'
              '/postprocessing/projections')
_PROJ_TMPL = os.path.join(_PROJ_DIR,
             'gas-xray_lum_0.5-5.0kev__2r200_d=r200.{snap}.hdf5')
_AVAILABLE_SNAPS = [17, 21, 25, 33, 40, 50, 59, 67, 72, 78, 84, 91, 99]


# ── helpers ────────────────────────────────────────────────────────────────────

def _build_pool(snaps):
    """Return list of (snap, halo_id, zoom_idx) from the reference projection files."""
    pool = []
    for snap in snaps:
        path = _PROJ_TMPL.format(snap=snap)
        if not os.path.exists(path):
            print(f'WARNING: reference file missing for snap {snap}, skipping')
            continue
        with h5py.File(path, 'r') as fh:
            halo_ids = fh['HaloIDs'][:].astype(int)
        for zoom_idx, halo_id in enumerate(halo_ids):
            pool.append((snap, int(halo_id), zoom_idx))
    return pool


def _sample(pool, n, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=n, replace=False)
    idx.sort()
    return [pool[i] for i in idx]


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Run intrinsic X-ray pipeline on a random halo sample.')
    parser.add_argument('--sim',      default='tng-cluster',
                        help='Simulation key in SIM_CONFIGS (default: tng-cluster)')
    parser.add_argument('--n',        type=int, default=30,
                        help='Number of halos to sample (default: 30)')
    parser.add_argument('--seed',     type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--out-dir',  default='results',
                        help='Output directory (default: results/)')
    parser.add_argument('--nthreads', type=int, default=8,
                        help='CPU threads per halo (default: 8)')
    parser.add_argument('--loader',   default=None,
                        help='Override loader: zoom or fof (default: from SIM_CONFIGS)')
    parser.add_argument('--emin',     type=float, default=0.5,
                        help='Rest-frame band minimum [keV] (default: 0.5)')
    parser.add_argument('--emax',     type=float, default=5.0,
                        help='Rest-frame band maximum [keV] (default: 5.0)')
    args = parser.parse_args()

    if args.sim not in SIM_CONFIGS:
        sys.exit(f'Unknown sim "{args.sim}". Known: {list(SIM_CONFIGS.keys())}')

    cfg    = SIM_CONFIGS[args.sim]
    loader = args.loader or cfg.loader

    os.makedirs(args.out_dir, exist_ok=True)
    hdf_path = os.path.join(args.out_dir, 'results.hdf5')
    csv_path = os.path.join(args.out_dir, 'sample_list.csv')
    log_path = os.path.join(args.out_dir, 'run.log')
    log_fh   = open(log_path, 'w', buffering=1)

    def log(msg=''):
        print(msg, flush=True)
        log_fh.write(msg + '\n')

    log(f'=== run_sample  sim={args.sim}  n={args.n}  seed={args.seed} ===')
    log(f'loader={loader}  band={args.emin}–{args.emax} keV  zobs=0 (intrinsic)')
    log(f'out_dir={args.out_dir}')
    log()

    log('Building halo pool ...')
    pool   = _build_pool(_AVAILABLE_SNAPS)
    sample = _sample(pool, args.n, args.seed)
    log(f'  Pool size: {len(pool)}  Sampled: {len(sample)}')
    log()

    with open(csv_path, 'w') as fcsv:
        fcsv.write('snap,halo_id,zoom_idx\n')
        for snap, halo_id, zoom_idx in sample:
            fcsv.write(f'{snap},{halo_id},{zoom_idx}\n')

    log(f'  idx  snap  halo_id   zoom_idx')
    log(f'  ---  ----  --------  --------')
    for i, (snap, halo_id, zoom_idx) in enumerate(sample):
        log(f'  {i:3d}  {snap:4d}  {halo_id:8d}  {zoom_idx:8d}')
    log()

    # ── initialise HDF5 ────────────────────────────────────────────────────────
    with h5py.File(hdf_path, 'w') as fh:
        cg = fh.create_group('config')
        cg.attrs['sim']          = args.sim
        cg.attrs['loader']       = loader
        cg.attrs['emin_kev']     = args.emin
        cg.attrs['emax_kev']     = args.emax
        cg.attrs['seed']         = args.seed
        cg.attrs['n_halos']      = args.n
        cg.attrs['zobs']         = 0
        cg.attrs['generated_at'] = datetime.now(timezone.utc).isoformat()
        sl = np.array([(s, h, z) for s, h, z in sample],
                      dtype=[('snap','i4'),('halo_id','i4'),('zoom_idx','i4')])
        fh.create_dataset('sample_list', data=sl)
        fh.create_group('halos')

    # ── process each halo ─────────────────────────────────────────────────────
    n_done = n_failed = 0

    for i, (snap, halo_id, zoom_idx) in enumerate(sample):
        key = f'snap{snap:03d}_halo{halo_id:07d}'
        log(f'[{i+1:2d}/{args.n}]  snap={snap}  halo={halo_id}  zoom={zoom_idx}')
        t0 = time.time()

        try:
            result  = process_halo(
                halo_id   = halo_id,
                snap      = snap,
                base_path = cfg.base_path,
                loader    = loader,
                sim       = args.sim,
                emin_kev  = args.emin,
                emax_kev  = args.emax,
                nthreads  = args.nthreads,
                verbose   = True,
            )
            elapsed = time.time() - t0
            meta    = result.meta
            log(f'  z={1/meta.a-1:.3f}  elapsed={elapsed:.0f}s  '
                f'log Lx(R500c)={np.log10(max(meta.L_x_r500c, 1e30)):.2f}')

            with h5py.File(hdf_path, 'a') as fh:
                grp = fh['halos'].create_group(key)
                grp.create_dataset('images', data=result.images,
                                   compression='gzip', compression_opts=4)
                for k, v in meta.to_dict().items():
                    grp.attrs[k] = v
                grp.attrs['zoom_idx'] = zoom_idx
                grp.attrs['emin_kev'] = args.emin
                grp.attrs['emax_kev'] = args.emax
                grp.attrs['zobs']     = 0
                fh.flush()

            n_done += 1

        except Exception:
            elapsed = time.time() - t0
            log(f'  FAILED after {elapsed:.0f}s')
            log(traceback.format_exc())
            with h5py.File(hdf_path, 'a') as fh:
                grp = fh['halos'].create_group(key)
                grp.attrs['failed']   = True
                grp.attrs['snap']     = snap
                grp.attrs['halo_id']  = halo_id
                grp.attrs['zoom_idx'] = zoom_idx
                fh.flush()
            n_failed += 1

        log()

    log('=== SUMMARY ===')
    log(f'  Done:   {n_done}')
    log(f'  Failed: {n_failed}')
    log(f'  Output: {hdf_path}')

    with h5py.File(hdf_path, 'a') as fh:
        fh['config'].attrs['n_done']   = n_done
        fh['config'].attrs['n_failed'] = n_failed

    log_fh.close()


if __name__ == '__main__':
    main()
