GPU stereo matcher (ORB-SLAM3 fork patch 10b) validation, 2026-10-29.
Patch 10b moves the brute-force Hamming 2-NN of Frame::ComputeStereoFishEyeMatches (cv::BFMatcher knnMatch k=2,
all ~2000 x 2000 descriptors per stereo frame because the lapping area is the full 1920 px) to the GPU (GpuHamming::Knn2
in ORBextractorGPU.cc, same NVRTC module as the patch-10 extractor). Enabled when ORBextractor.useGPU is 1; the YAML key
Stereo.useGPUMatch (0/1) overrides it and is what the A/B uses.

stereo_stage_test.cc : harness. Stage timing on the realv5 frames (two extractor threads -> BF knn2 -> ratio -> triangulate)
                       and a per-query comparison of the GPU matcher with cv::BFMatcher. Result: 2003/2003 and 2005/2005 queries
                       identical (best index + both distances), identical ratio-test sets; 9.1-10.0 ms -> 0.44 ms per frame.
                       (Note: this OpenCV build's knnMatch APPENDS to the output vector; clear it between calls.)
ab2.sh, r05_*/r10_* : node ABBA runs (stereo-only, loopClosing 0, viewer off), realv5, bag player cut after 60 s of bag time
                       (timeout 120 s wall at rate 0.5, 60 s at rate 1.0), each run started only when the CPU package was <= 72 C
                       (or after 300 s). A = stereo_extract.yaml (GPU extractor, CPU matcher), B = stereo_match.yaml (GPU both).
gpu_match_abba_metrics.csv : per-run table; progress_temperatures.txt: start/end package temperature per run.
order_confounded/   : the first attempt (full bag, A then B back to back, no cooling). Every run slows 60 -> 150 ms as the
                       laptop heats past 90 C, so the second arm always looked worse; kept as the record of that effect.
                       extract_r10 (CPU matcher) segfaulted via the known stereo-only loss -> "Relocalized!!" -> crash; so did
                       r05_B1 in the ABBA set after a 95 C start. Neither crash is in the matcher path.
