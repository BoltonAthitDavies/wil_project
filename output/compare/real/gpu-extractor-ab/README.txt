GPU ORB front end (ORB-SLAM3 fork patch 10) validation, 2026-10-28.
gpu_orb_test.cc: CPU-vs-GPU extractor harness (per-image time, keypoint/descriptor agreement) on the four PNG frames from dataset_realv5_000.
cpu_r*/gpu_r*: node runs (ab.sh), stereo-only, loopClosing 0, viewer off, rate 0.5 and 1.0; stereo_cpu.yaml / stereo_gpu.yaml differ only by ORBextractor.useGPU.
gpu_ab_metrics.csv: the table quoted in orbslam3_ros2/MEMORY_CLAUDE/integration.md section 3.5.
