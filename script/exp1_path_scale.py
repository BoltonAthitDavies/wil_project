#!/usr/bin/env python3
"""Where ORB-SLAM3's apparent path-length excess comes from: not scale.

    python3 script/exp1_path_scale.py

WHY THIS EXISTS
  vio_metrics.analyse() reports each run's estimated path length against the
  truth's, and on every ORB-SLAM3 run it comes out 1.7-5.4 % long.  Read as a
  scale bias that fitted the stereo-inertial method's fixed s = 1, and the
  chapter said so.  It is not.  The local scale -- estimated displacement over
  true displacement across each one-second window -- is 0.997-0.999, slightly
  UNDER one.  The excess is (a) the arc length of the pre-initialisation
  spike, when the estimate steps ~90 deg and 3-4 m while the truth moves a
  metre, and (b) frame-to-frame jitter, which inflates an arc length summed at
  30 Hz and vanishes when both series are sampled every second.  VINS-Fusion,
  which estimates scale with its IMU, is the control: its small excess on
  realistic_002 survives decimation, so that one is a real scale error.

PRINTS, per run
  reported   analyse()'s path/ref - 1, the figure the chapter quoted
  spike      arc length of the estimate before the published frame switch + 1 s
  after      path/ref - 1 over the rest of the run, at full rate, jump steps removed
  @1 s       the same with both series sampled every second
  local      median per-window scale
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np                                   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vio_metrics as vm                             # noqa: E402

ROOT = str(vm.ROOT)
BAGS = ["dataset_allsensor_realistic_%03d" % i for i in range(4)]
ARMS = [("ORB-SLAM3", "output_orb", "loopclosing_cpu"), ("ORB-SLAM3 GPU", "output_orb", "loopclosing_gpu"),
        ("ORB-SLAM3 (no LC)", "output_orb", "noloopclosing_cpu"), ("ORB-SLAM3 GPU (no LC)", "output_orb", "noloopclosing_gpu"),
        ("VINS-Fusion", "output_vins", "cpu")]


def decimated(t, a, b, dt):
    idx = [0]
    for i in range(1, len(t)):
        if t[i] - t[idx[-1]] >= dt:
            idx.append(i)
    idx = np.array(idx)
    return np.linalg.norm(np.diff(a[idx], axis=0), axis=1), np.linalg.norm(np.diff(b[idx], axis=0), axis=1), idx


def main():
    print("%-5s %-22s %9s %9s %9s %9s %8s" % ("bag", "arm", "reported", "spike", "after", "@1 s", "local"))
    orb_local, orb_after, orb_1s = [], [], []
    for ds in BAGS:
        gp = [c for c in vm.gt_candidates(ds) if os.path.exists(c)][0]; g = vm.load_gt(gp)
        for name, tree, run in ARMS:
            res = vm.analyse("simulation/" + ds, systems=[(name, tree, run)])
            if res is None or not res[0]:
                continue
            row = res[0][0]; reported = 100 * (row["path"] / row["ref"] - 1)
            d = os.path.join(ROOT, "output", tree, "simulation", ds, run); e = vm.load(os.path.join(d, "vio.csv"))
            t, p = e[0], e[1]; m = (t >= max(t[0], g[0][0])) & (t <= min(t[-1], g[0][-1])); t, p = t[m], p[m]
            sw = vm.world_frame_time(d) if tree == "output_orb" else None
            pre = (t < sw + 1.0) if sw is not None else (t < t[0] + 5.0)
            spike = np.linalg.norm(np.diff(p[pre], axis=0), axis=1).sum()
            t, p = t[~pre], p[~pre]; gq = vm.resample(g[0], g[1], t)
            se, sg = np.linalg.norm(np.diff(p, axis=0), axis=1), np.linalg.norm(np.diff(gq, axis=0), axis=1)
            keep = np.ones(len(se), bool); keep[list(vm.find_jumps(t, p)[0])] = False
            after = 100 * (se[keep].sum() / sg[keep].sum() - 1)
            s1e, s1g, idx = decimated(t, p, gq, 1.0)
            jumps = vm.find_jumps(t, p)[0]
            k1 = np.array([not any((j >= idx[n]) & (j < idx[n + 1]) for j in jumps) for n in range(len(idx) - 1)])
            at1 = 100 * (s1e[k1].sum() / s1g[k1].sum() - 1)
            local = float(np.median((s1e[k1] / np.maximum(s1g[k1], 1e-9))[s1g[k1] > 0.3]))
            if tree == "output_orb":
                orb_local.append(local); orb_after.append(after); orb_1s.append(at1)
            print("%-5s %-22s %+8.2f%% %7.2f m %+8.2f%% %+8.2f%% %8.4f" % (ds[-3:], name, reported, spike, after, at1, local))
    print("\nORB-SLAM3 over %d runs: local scale %.4f-%.4f ; after the spike %+.2f to %+.2f%% at full rate, %+.2f to %+.2f%% sampled every second"
          % (len(orb_local), min(orb_local), max(orb_local), min(orb_after), max(orb_after), min(orb_1s), max(orb_1s)))


if __name__ == "__main__":
    main()
