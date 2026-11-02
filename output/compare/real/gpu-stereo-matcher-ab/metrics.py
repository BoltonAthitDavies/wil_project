#!/usr/bin/env python3
"""Per-run metrics for the patch-10b node A/B (same columns as gpu_ab_metrics.csv + matcher/log-line counts)."""
import csv, statistics as st, sys, os
root = sys.argv[1]; out = sys.argv[2]
runs = [("extract_r05","cpu-bf",0.5),("match_r05","gpu-knn2",0.5),("extract_r10","cpu-bf",1.0),("match_r10","gpu-knn2",1.0)]
cols = ["run","extractor","stereo_matcher","replay_rate","pairs_received","pairs_processed","queue_dropped","drop_pct",
        "tracker_total_ms_median","extraction_stage_ms_median","pose_prediction_ms_median","local_map_ms_median",
        "inliers_median","inliers_p10","frames_ok","frames_lost","tracking_loss_events","resets","keyframes","map_points_final","poses",
        "gpu_extractor_log_lines","gpu_matcher_log_lines","shutdown"]
rows = []
for name, matcher, rate in runs:
    d = os.path.join(root, name)
    if not os.path.isfile(os.path.join(d, "run_summary.csv")): print("SKIP (no run_summary)", name); continue
    summ = {r["key"]: r["value"] for r in csv.DictReader(open(os.path.join(d, "run_summary.csv")))}
    fr = list(csv.DictReader(open(os.path.join(d, "tracking_frontend.csv"))))
    ok = [r for r in fr if r["tracking_state"] == "2"]
    def med(k): v = [float(r[k]) for r in ok if r[k] not in ("", "nan")]; return round(st.median(v), 1) if v else ""
    inl = sorted(float(r["map_matches_inliers"]) for r in ok)
    log = open(os.path.join(d, "node.log"), errors="replace").read()
    rec, proc = int(summ["stereo_pairs_received"]), int(summ["stereo_pairs_processed"])
    rows.append(dict(run=name, extractor="gpu", stereo_matcher=matcher, replay_rate=rate, pairs_received=rec, pairs_processed=proc,
        queue_dropped=summ["queue_dropped_frames"], drop_pct=round(100 * (rec - proc) / max(rec, 1), 1),
        tracker_total_ms_median=med("tracking_total_ms"), extraction_stage_ms_median=med("image_preprocessing_feature_stereo_ms"),
        pose_prediction_ms_median=med("pose_prediction_ms"), local_map_ms_median=med("local_map_tracking_ms"),
        inliers_median=int(st.median(inl)) if inl else "", inliers_p10=int(inl[len(inl) // 10]) if inl else "",
        frames_ok=len(ok), frames_lost=sum(1 for r in fr if r["tracking_state"] in ("3", "4")),
        tracking_loss_events=summ["tracking_loss_events"], resets=summ["active_map_reset_requests"], keyframes=summ["keyframes_created"],
        map_points_final=summ["map_points_in_final_map"], poses=summ["poses_written"],
        gpu_extractor_log_lines=log.count("GPU front end active"), gpu_matcher_log_lines=log.count("GPU stereo matcher active"),
        shutdown=summ["shutdown_status"]))
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
for r in rows: print({k: r[k] for k in ("run","pairs_processed","drop_pct","extraction_stage_ms_median","pose_prediction_ms_median","local_map_ms_median","tracker_total_ms_median","inliers_median","map_points_final","tracking_loss_events","gpu_matcher_log_lines","shutdown")})
