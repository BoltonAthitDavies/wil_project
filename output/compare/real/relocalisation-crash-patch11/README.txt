ORB-SLAM3 fork patch (11): segfault after tracking loss on the fisheye stereo rig (exit code -11), 2026-10-29.

Root cause: ORBmatcher::SearchByProjection(Frame&, KeyFrame*, set<MapPoint*>, th, ORBdist) -- called only from
Tracking::Relocalization() -- reads pKF->mvKeysUn[i] for every matched map point. KannalaBrandt8 frames have
mDistCoef == 0, so Frame::UndistortKeyPoints() sets mvKeysUn = mvKeys (NLeft left-camera keypoints only), while
GetMapPointMatches() has NLeft + NRight entries. A map point first observed through the right camera (i >= NLeft)
reads past the vector -> garbage/NaN angle -> round() = INT_MIN -> rotHist[bin] indexes the stack -> SIGSEGV.
Nondeterministic: it only faults when the overrun crosses into unmapped memory.

crash_backtrace_before_fix_run1.log            : gdb, realv5 rate 1.0, stereo-only. SIGSEGV in that function <- Relocalization().
crash_backtrace_before_fix_attempt1_registers.log : same, with `info symbol`, registers and disassembly:
                                                 rax = 0xffffffff80000000 (bin = INT_MIN), r12 = 0x7e3 = 2019 (i >= NLeft ~ 2003),
                                                 faulting insn `mov -0x308(%rbp,%rcx,1),%rsi` = rotHist[bin].begin.
survived_before_fix_run2.log                   : same scenario, relocalised 4x and survived (the nondeterminism).
after_fix_jump_run{1,2}/                       : patched library under gdb; map built from the first 25 s of realv5, then playback
                                                 restarted at offsets 3 / 40 / 8 s so the tracker must relocalise. Run 1: lost,
                                                 "Relocalized!!", continued. Run 2: 48 frames in RECENTLY_LOST with repeated
                                                 relocalisation attempts, then a new map. 0 segfaults, graceful shutdown in both.
gdb_repro.sh / loop_repro.sh / gdb_reloc_jump.sh : the scripts (pidfile-based, no pattern kills).
Fix: NLeft-select the keypoint (mvKeys / mvKeysRight) like every other matcher variant, plus a bin range check; original line
kept as a comment; applied idempotently by build_orbslam3.sh (patch 11); integration.md section 7.10.
