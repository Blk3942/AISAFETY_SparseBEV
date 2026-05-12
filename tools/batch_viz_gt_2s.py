#!/usr/bin/env python3
"""
Batch-generate GT BEV visualizations for nuScenes val split,
sub-sampled every N consecutive samples within each scene.

With --step 10, takes samples at index 0, 10, 20, ... per scene.
For 150 val scenes × ~40 samples each, this yields ~600 images total.

Output: --out_dir/{sample_id:06d}.png
Uses ProcessPoolExecutor for parallel rendering.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pymysql
import pymysql.cursors

sys.path.insert(0, str(Path(__file__).parent))
from viz_gt_topo_map_bev import mysql_connect, render_figure  # noqa: E402


# ── helpers ──────────────────────────────────────────────────────────────────

def fmt_time(s: float) -> str:
    s = max(0, int(s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h {m:02d}m {sec:02d}s" if h else f"{m:02d}m {sec:02d}s"


def _p(*args, **kwargs):
    print(*args, **kwargs, flush=True)


# ── worker ────────────────────────────────────────────────────────────────────

def _worker_render(task: tuple) -> tuple:
    """Render one sample in a subprocess. Returns (sample_id, ok: bool)."""
    sid, out_path, db_cfg, render_cfg = task

    import sys, os, contextlib
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from viz_gt_topo_map_bev import mysql_connect as _connect, render_figure as _render
    from argparse import Namespace

    conn = _connect(
        db_cfg["host"], db_cfg["port"], db_cfg["user"],
        db_cfg["password"], db_cfg["database"],
    )
    rargs = Namespace(**render_cfg, out=out_path)

    with open(os.devnull, "w") as _null, contextlib.redirect_stdout(_null):
        try:
            ok = bool(_render(conn, sid, rargs))
        except Exception:
            ok = False
    try:
        conn.close()
    except Exception:
        pass
    return sid, ok


# ── sample selection ──────────────────────────────────────────────────────────

def select_every_n_samples(conn, dataroot: str, version: str, step: int = 10) -> list:
    """
    For every val scene (sorted by timestamp), keep samples at index 0, step, 2*step, ...
    Returns sorted list of sample_ids.
    """
    _p("  Loading nuScenes val split ...")
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes

    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    val_scenes: set = set(create_splits_scenes().get("val", []))
    _p(f"  Val scenes  : {len(val_scenes)}")

    token_by_name = {s["name"]: s["token"]
                     for s in nusc.scene if s["name"] in val_scenes}
    val_tokens = set(token_by_name.values())
    _p(f"  Scene tokens: {len(val_tokens)}")

    cur = conn.cursor()
    cur.execute(
        "SELECT id, scene_token, timestamp_us "
        "FROM ground_truth_sample "
        "WHERE scene_token IS NOT NULL "
        "ORDER BY scene_token, timestamp_us"
    )
    rows = cur.fetchall()
    cur.close()

    from collections import defaultdict
    scene_samples: dict = defaultdict(list)
    for r in rows:
        if r["scene_token"] in val_tokens:
            scene_samples[r["scene_token"]].append(
                (r["timestamp_us"], int(r["id"]))
            )

    selected = []
    for scene_tok, samples in scene_samples.items():
        samples.sort(key=lambda x: x[0])
        for idx, (ts, sid) in enumerate(samples):
            if idx % step == 0:
                selected.append(sid)

    selected.sort()
    return selected


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir",           default=r"E:\Data\Nuscenes\Viz2")
    ap.add_argument("--dataroot",          default=r"E:\Data\Nuscenes\Full")
    ap.add_argument("--host",              default="127.0.0.1")
    ap.add_argument("--port",             type=int, default=3306)
    ap.add_argument("--user",              default="root")
    ap.add_argument("--password",          default="")
    ap.add_argument("--database",          default="safetyai_sparsebev")
    ap.add_argument("--bev_range",        type=float, default=80.0)
    ap.add_argument("--nuscenes_version",  default="v1.0-trainval")
    ap.add_argument("--map_dataroot",      default=None)
    ap.add_argument("--basemap_dir",       default=None)
    ap.add_argument("--map_name_override", default=None)
    ap.add_argument("--step",             type=int, default=10,
                    help="Take one sample every N consecutive samples per scene (default 10).")
    ap.add_argument("--skip_existing",     action="store_true", default=True)
    ap.add_argument("--workers",          type=int, default=4,
                    help="Parallel worker processes (default 4).")
    ap.add_argument("--log_every",        type=int, default=10)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _p(f"\n{'='*64}")
    _p(f"  Output dir  : {out_dir}")
    _p(f"  Data root   : {args.dataroot}")
    _p(f"  Step        : every {args.step} samples per scene")
    _p(f"  Workers     : {args.workers}")
    _p(f"{'='*64}\n")

    conn = mysql_connect(args.host, args.port, args.user, args.password, args.database)

    sample_ids = select_every_n_samples(conn, args.dataroot, args.nuscenes_version,
                                        step=args.step)
    conn.close()

    total = len(sample_ids)
    _p(f"\n  Selected samples: {total}  (every {args.step} consecutive per scene)")

    if total == 0:
        _p("  [ERROR] No samples selected.")
        return

    if args.skip_existing:
        before = len(sample_ids)
        sample_ids = [s for s in sample_ids if not (out_dir / f"{s:06d}.png").exists()]
        _p(f"  Already done    : {before - len(sample_ids)}")
        _p(f"  To generate     : {len(sample_ids)}\n")

    todo = len(sample_ids)
    if todo == 0:
        _p("  All images already exist. Nothing to do.")
        return

    db_cfg = dict(host=args.host, port=args.port, user=args.user,
                  password=args.password, database=args.database)
    render_cfg = dict(
        dataroot=args.dataroot,
        bev_range=args.bev_range,
        nuscenes_version=args.nuscenes_version,
        map_dataroot=args.map_dataroot,
        basemap_dir=args.basemap_dir,
        map_name_override=args.map_name_override,
    )

    tasks = [
        (sid, str(out_dir / f"{sid:06d}.png"), db_cfg, render_cfg)
        for sid in sample_ids
    ]

    _p(f"{'─'*64}")
    _p(f"  Rendering {todo} images with {args.workers} workers ...")
    _p(f"  (First few images are slower while workers load NuScenes/maps)")
    _p(f"{'─'*64}\n")

    t_start = time.time()
    done = fail = 0
    last_log = 0
    W = len(str(todo))

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_worker_render, t): t[0] for t in tasks}

        for future in as_completed(futures):
            try:
                _sid, ok = future.result()
            except Exception as exc:
                ok = False
                _sid = futures[future]
                _p(f"    [FAIL] id={_sid}: {exc}")

            if ok:
                done += 1
            else:
                fail += 1

            completed = done + fail
            if completed - last_log >= args.log_every or completed == todo:
                last_log = completed
                elapsed = time.time() - t_start
                speed = completed / max(elapsed, 1e-6)
                eta = (todo - completed) / max(speed, 1e-6)
                pct = completed / todo * 100
                bw = 26
                filled = int(bw * completed / todo)
                bar = "[" + "#" * filled + "-" * (bw - filled) + "]"
                _p(
                    f"  {bar} {completed:>{W}}/{todo} {pct:5.1f}%"
                    f" | elapsed {fmt_time(elapsed)}"
                    f" | eta ~{fmt_time(eta)}"
                    f" | {speed:.2f} img/s"
                    f" | ok={done} fail={fail}"
                )

    elapsed = time.time() - t_start
    _p(f"\n{'='*64}")
    _p(f"  Done!  {done} succeeded  {fail} failed  ({todo} total)")
    _p(f"  Total time  : {fmt_time(elapsed)}")
    _p(f"  Avg speed   : {done / max(elapsed, 1):.2f} img/s")
    _p(f"  Output dir  : {out_dir}")
    _p(f"{'='*64}\n")


if __name__ == "__main__":
    main()
