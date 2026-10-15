#!/usr/bin/env bash
#
# Build ORB-SLAM3 and Pangolin out of tree, for the orbslam3_ros2 package.
#
#   ./build_orbslam3.sh                 # run every stage that still needs running
#   ./build_orbslam3.sh pangolin        # run just one stage (repeat after a fix)
#   ./build_orbslam3.sh orbslam3 vocab
#   FORCE=1 ./build_orbslam3.sh orbslam3   # redo a stage that already looks done
#
# Stages: preflight pangolin orbslam3 vocab cleanup
#
# WHY THIS EXISTS, RATHER THAN upstream's build.sh
#   build.sh hardcodes the flags we have to override. Specifically it builds
#   Thirdparty/Sophus (pointless -- Sophus here is `add_library(sophus INTERFACE)`,
#   header-only, and ORB-SLAM3 only ever puts it on the include path; building it
#   turns its tests on, which is a known failure and pure disk waste), and it
#   leaves ORB-SLAM3 on -std=c++11, which cannot parse Pangolin 0.9's headers.
#
# DISK
#   This machine has ~2.5 GB free on /. The whole build fits in roughly 1 GB, but
#   only because examples, tools and tests are off everywhere and the object trees
#   are deleted afterwards. Every stage re-checks free space first and refuses to
#   start rather than dying halfway through a link with ENOSPC.
#
# NO SUDO
#   sudo needs a password here, so nothing is apt-installed and nothing is written
#   outside $ROOT. Pangolin goes to a local prefix; ORB-SLAM3 builds in its own
#   source tree (upstream ships no install rules, and the ROS package points
#   straight at the tree).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TP="$ROOT/thirdparty"
PREFIX="$TP/install"
PANGOLIN_SRC="$TP/Pangolin"
ORB_SRC="$TP/ORB_SLAM3"

PANGOLIN_TAG="v0.9.1"
# Our own fork, on a branch that already carries the four patches below. Pinned
# to a commit rather than tracking the branch, so a rebuild months from now is
# byte-identical. Forked from UZ-SLAMLab/ORB_SLAM3 @ 4452a3c, so that upstream
# disappearing or force-pushing cannot break this build.
ORB_REPO="https://github.com/BoltonAthitDavies/ORB_SLAM3.git"
ORB_COMMIT="5c2b00f732775f78836751dba13db25463a0aaa1"

NPROC="$(nproc)"
JOBS="$(( NPROC > 8 ? 8 : NPROC ))"   # -j16 with -O3 on this codebase OOMs a 15 GB box
MIN_FREE_MB=1500

say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn] %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31m[fail] %s\033[0m\n' "$*" >&2; exit 1; }

free_mb() { df -Pm "$ROOT" | awk 'NR==2 {print $4}'; }

require_space() {
    local need="${1:-$MIN_FREE_MB}" have
    have="$(free_mb)"
    if [ "$have" -lt "$need" ]; then
        die "only ${have} MB free on $(df -P "$ROOT" | awk 'NR==2{print $6}'), need ${need} MB.
       Free some space and re-run. Big reclaim candidates:
         du -sh $ROOT/dataset/*     # 15 GB of bags
         rm -rf $ROOT/log/build_*   # old colcon logs"
    fi
    printf '    disk: %s MB free\n' "$have"
}

# ---------------------------------------------------------------- preflight ---
stage_preflight() {
    say "preflight"
    require_space
    for t in git cmake make g++ tar; do
        command -v "$t" >/dev/null || die "missing required tool: $t"
    done
    pkg-config --exists opencv4 || warn "no opencv4 in pkg-config; cmake may still find it"
    printf '    gcc     %s\n' "$(g++ -dumpversion)"
    printf '    cmake   %s\n' "$(cmake --version | head -1 | awk '{print $3}')"
    printf '    opencv  %s\n' "$(pkg-config --modversion opencv4 2>/dev/null || echo '?')"
    printf '    jobs    %s\n' "$JOBS"

    mkdir -p "$TP" "$PREFIX"
    # Keep colcon out of here. thirdparty/ has CMakeLists.txt files in it and
    # colcon would happily try to build them as workspace packages.
    touch "$TP/COLCON_IGNORE"
}

# ----------------------------------------------------------------- pangolin ---
stage_pangolin() {
    say "Pangolin $PANGOLIN_TAG"
    if [ -z "${FORCE:-}" ] && [ -f "$PREFIX/lib/libpango_display.so" ]; then
        echo "    already installed, skipping (FORCE=1 to redo)"
        return
    fi
    require_space

    if [ ! -d "$PANGOLIN_SRC/.git" ]; then
        rm -rf "$PANGOLIN_SRC"
        git clone --depth 1 --branch "$PANGOLIN_TAG" \
            https://github.com/stevenlovegrove/Pangolin.git "$PANGOLIN_SRC"
    fi

    # Everything off but the core libraries. BUILD_EXAMPLES and BUILD_TOOLS default
    # to ON upstream and together are most of the build time and disk.
    cmake -S "$PANGOLIN_SRC" -B "$PANGOLIN_SRC/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$PREFIX" \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_TOOLS=OFF \
        -DBUILD_TESTS=OFF \
        -DBUILD_PANGOLIN_PYTHON=OFF
    cmake --build "$PANGOLIN_SRC/build" -j "$JOBS"
    cmake --install "$PANGOLIN_SRC/build"

    [ -f "$PREFIX/lib/libpango_display.so" ] || die "Pangolin install produced no libpango_display.so"
    rm -rf "$PANGOLIN_SRC/build"     # ~400 MB of objects we will never need again
    echo "    installed to $PREFIX"
}

# ----------------------------------------------------------------- orbslam3 ---
patch_orbslam3() {
    local f

    # As of the ORB_COMMIT pin above these four patches are already committed on
    # the fork's `wil` branch, so every one of them is a no-op here. They are kept
    # rather than deleted: each is guarded by its own grep, so this doubles as an
    # assertion that the checkout really is the patched tree, and it still does the
    # right thing if ORB_SRC is ever pointed at a pristine upstream clone.

    # (1) `++` on a bool was deprecated in C++11 and REMOVED in C++17, and we are
    #     on C++17 because Pangolin 0.9's headers require it. Three sites.
    #
    #     `= true` is exactly what bool++ did, so this preserves behaviour: our
    #     build stays identical to every C++14 ORB-SLAM3 out there, which matters
    #     when the whole point is comparing against published results.
    #
    #     (Upstream oddity, left alone deliberately: mnFullBAIdx is declared
    #     `bool` in LoopClosing.h:226 but used as a generation counter --
    #     `int idx = mnFullBAIdx; ... if(idx!=mnFullBAIdx)` at LoopClosing.cc
    #     2300/2309. The name says "Idx". As a bool it saturates at true, so that
    #     check can only fire once. Changing it to int would alter loop-closure
    #     behaviour, so we do not.)
    #
    #     NOTE: there is a widely-copied "fix" that rewrites LoopClosing.h:51's
    #     `std::pair<KeyFrame* const, g2o::Sim3>` into `std::pair<const KeyFrame*, ...>`.
    #     Do NOT apply it. It is backwards and BREAKS the build: for std::map<Key,T>
    #     the value_type is std::pair<const Key,T>, and with Key=KeyFrame* that is
    #     `KeyFrame* const` (a const pointer), not `const KeyFrame*` (pointer to
    #     const). Upstream is already correct. Applying it yields
    #     "static assertion failed: std::map must have the same value_type as its
    #     allocator". This cost us a build cycle; leave the line alone.
    f="$ORB_SRC/src/LoopClosing.cc"
    if grep -qE '^\s*mnFullBAIdx\+\+;' "$f"; then
        sed -i 's/^\( *\)mnFullBAIdx++;/\1mnFullBAIdx = true;   \/\/ was mnFullBAIdx++: `++` on bool is illegal in C++17/' "$f"
        echo "    patched LoopClosing.cc (bool++ -> = true, 3 sites)"
    fi
    grep -qE '^\s*mnFullBAIdx\+\+;' "$f" && die "LoopClosing.cc bool++ patch did not take"

    # (2) Pangolin 0.9 headers are C++17. Upstream appends -std=c++11 to
    #     CMAKE_CXX_FLAGS, which beats any -DCMAKE_CXX_STANDARD we pass, so the
    #     flag itself has to change. Rewriting only the `set(...)` line keeps
    #     add_definitions(-DCOMPILEDWITHC11) intact -- the sources use that ifdef
    #     to pick std::chrono over the monotonic clock, and dropping it silently
    #     changes which timing path compiles.
    f="$ORB_SRC/CMakeLists.txt"
    if grep -q 'CMAKE_CXX_FLAGS} -std=c++11")' "$f"; then
        sed -i 's/CMAKE_CXX_FLAGS} -std=c++11")/CMAKE_CXX_FLAGS} -std=c++17")/' "$f"
        echo "    patched CMakeLists.txt (-std=c++11 -> -std=c++17)"
    fi

    # (3) Drop the ~20 example executables. We only need libORB_SLAM3.so, and the
    #     examples are both a chunk of disk and the place where OpenCV-4
    #     incompatibilities (CV_LOAD_IMAGE_*) actually bite. The cut is at the
    #     upstream "# Build examples" comment; everything above it defines the
    #     library and everything below is executables. Idempotent: after the first
    #     run the marker is gone and this is a no-op.
    if grep -q '^# Build examples' "$f"; then
        awk '/^# Build examples/{exit} {print}' "$f" > "$f.trimmed"
        mv "$f.trimmed" "$f"
        echo "    patched CMakeLists.txt (dropped example executables)"
    fi

    # (4) Expose the tracker so the node can read the estimated IMU velocity.
    #     Frame::GetVelocity() and Tracking::mCurrentFrame are both public, but
    #     System::mpTracker is private, so there is otherwise no way to reach it.
    #     Without this the node has to finite-difference position instead.
    f="$ORB_SRC/include/System.h"
    if ! grep -q 'GetTracker' "$f"; then
        sed -i 's|^    int GetTrackingState();|    // Added for orbslam3_ros2: reach the tracker to read the estimated\n    // IMU velocity (Frame::GetVelocity), which System otherwise hides.\n    Tracking* GetTracker() { return mpTracker; }\n\n    int GetTrackingState();|' "$f"
        grep -q 'GetTracker' "$f" || die "System.h GetTracker patch did not take"
        echo "    patched System.h       (added GetTracker accessor)"
    fi

    # (5) UPSTREAM BUG -> SIGSEGV. Tracking::PreintegrateIMU() has three early
    #     returns; the first two call mCurrentFrame.setIntegrated() before
    #     bailing, the third does not:
    #
    #         const int n = mvImuFromLastFrame.size()-1;
    #         if(n==0){ cout << "Empty IMU measurements vector!!!\n"; return; }
    #
    #     The frame is then left un-preintegrated, the caller reports
    #     "Not preintegrated measurement", and a null preintegration is
    #     dereferenced -- exit code -11.
    #
    #     Note the message lies: n is size-1, so it fires when there is exactly
    #     ONE IMU sample between consecutive frames, not zero. Guarding the
    #     node against an EMPTY vector (which we also do) is therefore not
    #     enough to avoid it.
    #
    #     Hit on dataset/dataset_real_000: images 30 Hz and IMU 200 Hz give a
    #     median of 7 samples per frame interval, but three intervals in that
    #     bag hold exactly 2 -- one dropped IMU message away from the bug.
    f="$ORB_SRC/src/Tracking.cc"
    if grep -q 'Empty IMU measurements vector' "$f" && \
       ! grep -q 'orbslam3_ros2 patch: mark the frame integrated' "$f"; then
        python3 - "$f" <<'PYEOF'
import pathlib, sys
p = pathlib.Path(sys.argv[1]); s = p.read_text()
old = '        cout << "Empty IMU measurements vector!!!\\n";\n        return;'
new = ('        cout << "Empty IMU measurements vector!!!\\n";\n'
       '        // orbslam3_ros2 patch: mark the frame integrated before bailing out.\n'
       '        mCurrentFrame.setIntegrated();\n        return;')
assert old in s, "Tracking.cc n==0 branch not found"
p.write_text(s.replace(old, new, 1))
PYEOF
        echo "    patched Tracking.cc    (setIntegrated on the n==0 early return)"
    fi

    # (6) Map::PreSave() walks mspMapPoints with a range-for while the body can
    #     erase from that same set, so saving an atlas segfaults.
    #
    #     EraseObservation() takes a point to <=2 observations -> SetBadFlag()
    #     -> Map::EraseMapPoint() -> mspMapPoints.erase(pMP). That frees the
    #     red-black-tree node the loop is standing on, and the next ++ faults
    #     inside std::_Rb_tree_increment. gdb backtrace:
    #       _Rb_tree_increment <- Map::PreSave <- Atlas::PreSave <- SaveAtlas
    #
    #     Only bites when a point really falls to <=2 obs, which is data
    #     dependent: dataset_allsensor_realistic_001 crashes every run while
    #     other sequences wrote a healthy .osa from this same build. Symptom is
    #     exit -11 at shutdown with no .osa, AFTER the node has already written
    #     its CSVs and reported shutdown_status=graceful.
    f="$ORB_SRC/src/Map.cc"
    if grep -q 'for(MapPoint\* pMPi : mspMapPoints)' "$f" && \
       ! grep -q 'spMapPointsSnapshot' "$f"; then
        python3 - "$f" <<'PYEOF'
import pathlib, sys
p = pathlib.Path(sys.argv[1]); s = p.read_text()
old = "    int nMPWithoutObs = 0;\n    for(MapPoint* pMPi : mspMapPoints)\n    {"
new = ("    int nMPWithoutObs = 0;\n"
       "    // orbslam3_ros2 patch (6): iterate a SNAPSHOT, not the live set --\n"
       "    // EraseObservation() below can erase from mspMapPoints mid-loop.\n"
       "    // Original line, kept for reference:\n"
       "    // for(MapPoint* pMPi : mspMapPoints)\n"
       "    const std::set<MapPoint*> spMapPointsSnapshot = mspMapPoints;\n"
       "    for(MapPoint* pMPi : spMapPointsSnapshot)\n    {")
assert old in s, "Map.cc PreSave loop not found"
p.write_text(s.replace(old, new, 1))
PYEOF
        echo "    patched Map.cc        (PreSave iterates a snapshot)"
    fi

    # (7) With loopClosing:0, LoopClosing::Run() reads uninitialised memory and
    #     busy-spins. Two upstream defects behind one YAML switch:
    #       a. mpCurrentKF is never initialised; its only assignment is after
    #          the !mbActiveLC early return, so the "place_recognition" log line
    #          reads heap garbage. Zero by luck on a cold heap (every OFF run
    #          logged current_keyframe_id=-1); non-zero once LoadAtlasFromFile
    #          has dirtied the heap -> SIGSEGV on the first keyframe.
    #       b. the early return sits before the queue pop, so the KF is never
    #          dequeued, CheckNewKeyFrames() stays true, the thread spins at
    #          100 % CPU writing junk CSV rows, and mpKeyFrameDB stays empty so
    #          relocalisation can never find candidates.
    #     Fix: initialise mpCurrentKF, and move the check below the pop with the
    #     same add/SetErase/return body as the three gates after it.
    f="$ORB_SRC/src/LoopClosing.cc"
    if grep -q 'mpLastCurrentKF = static_cast<KeyFrame\*>(NULL);' "$f" && \
       ! grep -q 'orbslam3_ros2 patch (7)' "$f"; then
        python3 - "$f" <<'PYEOF'
import pathlib, sys
p = pathlib.Path(sys.argv[1]); s = p.read_text()
a = "    mpLastCurrentKF = static_cast<KeyFrame*>(NULL);\n"
assert a in s, "LoopClosing ctor anchor not found"
s = s.replace(a, a + "    // orbslam3_ros2 patch (7): upstream never initialises mpCurrentKF.\n"
                     "    mpCurrentKF = static_cast<KeyFrame*>(NULL);\n", 1)
old = ("    if(!mbActiveLC)\n        return false;\n\n    {\n"
       "        unique_lock<mutex> lock(mMutexLoopQueue);\n"
       "        mpCurrentKF = mlpLoopKeyFrameQueue.front();\n"
       "        mlpLoopKeyFrameQueue.pop_front();\n")
assert old in s, "NewDetectCommonRegions early return not found"
new = ("    // orbslam3_ros2 patch (7): early return moved below the pop so the\n"
       "    // queue drains with loopClosing:0. Original lines, kept for reference:\n"
       "    // if(!mbActiveLC)\n    //     return false;\n\n    {\n"
       "        unique_lock<mutex> lock(mMutexLoopQueue);\n"
       "        mpCurrentKF = mlpLoopKeyFrameQueue.front();\n"
       "        mlpLoopKeyFrameQueue.pop_front();\n")
s = s.replace(old, new, 1)
old2 = "        mpLastMap = mpCurrentKF->GetMap();\n    }\n\n    if(mpLastMap->IsInertial()"
assert old2 in s, "pop block end not found"
new2 = ("        mpLastMap = mpCurrentKF->GetMap();\n    }\n\n"
        "    if(!mbActiveLC)\n    {\n        mpKeyFrameDB->add(mpCurrentKF);\n"
        "        mpCurrentKF->SetErase();\n        return false;\n    }\n\n    if(mpLastMap->IsInertial()")
s = s.replace(old2, new2, 1)
p.write_text(s)
PYEOF
        echo "    patched LoopClosing.cc (init mpCurrentKF; drain queue when inactive)"
    fi

    # (8) System::Shutdown() never stops the Viewer thread -- upstream left the
    #     block commented out. With use_viewer:=true, Viewer::Run() is still in
    #     cv::waitKey()'s GTK main loop when main() returns and the process dies
    #     in libgtk at exit (SIGSEGV/SIGTRAP) after all outputs are written.
    #     Re-enabled with a bounded (<= 2 s) wait.
    f="$ORB_SRC/src/System.cc"
    if grep -q '/\*if(mpViewer)' "$f" && ! grep -q 'orbslam3_ros2 patch (8)' "$f"; then
        python3 - "$f" <<'PYEOF'
import pathlib, sys
p = pathlib.Path(sys.argv[1]); s = p.read_text()
old = ("    /*if(mpViewer)\n    {\n        mpViewer->RequestFinish();\n"
       "        while(!mpViewer->isFinished())\n            usleep(5000);\n    }*/\n")
assert old in s, "System::Shutdown viewer block not found"
new = ("    // orbslam3_ros2 patch (8): stop the Viewer thread before teardown.\n"
       "    // Original lines, kept for reference:\n"
       "    // /*if(mpViewer)\n    // {\n    //     mpViewer->RequestFinish();\n"
       "    //     while(!mpViewer->isFinished())\n    //         usleep(5000);\n    // }*/\n"
       "    if(mpViewer)\n    {\n        mpViewer->RequestFinish();\n"
       "        for(int i = 0; i < 400 && !mpViewer->isFinished(); ++i)\n            usleep(5000);\n    }\n")
p.write_text(s.replace(old, new, 1))
PYEOF
        echo "    patched System.cc      (Shutdown stops the Viewer thread)"
    fi

    # (9) Companion to (8). Once the Viewer thread can exit, Pangolin's global
    #     window registry is still destroyed by the MAIN thread in _dl_fini with
    #     no current GL context -> glDeleteTextures segfaults at process exit.
    #     Destroy the windows on the viewer thread, before SetFinish().
    f="$ORB_SRC/src/Viewer.cc"
    if grep -q 'CreateWindowAndBind("ORB-SLAM3: Map Viewer"' "$f" && \
       ! grep -q 'orbslam3_ros2 patch (9)' "$f"; then
        python3 - "$f" <<'PYEOF'
import pathlib, sys
p = pathlib.Path(sys.argv[1]); s = p.read_text()
old = "        if(CheckFinish())\n            break;\n    }\n\n    SetFinish();\n"
assert s.count(old) == 1, "Viewer::Run tail not found"
new = ("        if(CheckFinish())\n            break;\n    }\n\n"
       "    // orbslam3_ros2 patch (9): destroy windows on the GL-owning thread\n"
       "    // before SetFinish(); pure insertion, no upstream line removed.\n"
       "    cv::destroyAllWindows();\n"
       "    pangolin::DestroyWindow(\"ORB-SLAM3: Map Viewer\");\n\n    SetFinish();\n")
p.write_text(s.replace(old, new, 1))
PYEOF
        echo "    patched Viewer.cc      (DestroyWindow on the viewer thread)"
    fi
}

stage_orbslam3() {
    say "ORB-SLAM3 @ ${ORB_COMMIT:0:8}"
    if [ -z "${FORCE:-}" ] && [ -f "$ORB_SRC/lib/libORB_SLAM3.so" ]; then
        echo "    already built, skipping (FORCE=1 to redo)"
        return
    fi
    require_space
    [ -f "$PREFIX/lib/libpango_display.so" ] || die "Pangolin not installed yet -- run the pangolin stage first"

    if [ ! -d "$ORB_SRC/.git" ]; then
        rm -rf "$ORB_SRC"
        # Full clone: --depth 1 cannot check out a specific commit, and we want the
        # build pinned. It is ~90 MB, most of it the vocabulary tarball.
        git clone "$ORB_REPO" "$ORB_SRC"
        git -C "$ORB_SRC" checkout --quiet "$ORB_COMMIT"
    fi

    say "  patches"
    patch_orbslam3

    # DBoW2 is linked as a prebuilt .so by path, so it must be built first and
    # separately. g2o is NOT -- the main CMakeLists add_subdirectory()s it.
    # Sophus is header-only (add_library(sophus INTERFACE)) and is never built.
    say "  Thirdparty/DBoW2"
    cmake -S "$ORB_SRC/Thirdparty/DBoW2" -B "$ORB_SRC/Thirdparty/DBoW2/build" \
        -DCMAKE_BUILD_TYPE=Release
    cmake --build "$ORB_SRC/Thirdparty/DBoW2/build" -j "$JOBS"

    say "  libORB_SLAM3.so"
    cmake -S "$ORB_SRC" -B "$ORB_SRC/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="$PREFIX"
    # Optimizer.cc is 192K of g2o template code and at -O3 can want several GB on
    # its own. This box has 15 GB and NO SWAP, so when it lands in the same -j
    # wave as Tracking.cc/LoopClosing.cc the OOM killer takes cc1plus out and gcc
    # reports only "Killed signal terminated program cc1plus" -- no error:, which
    # makes it look like a mystery failure. Back off instead of giving up; make
    # keeps the objects already built, so each retry resumes where it stopped.
    local built=0 j
    for j in "$JOBS" 2 1; do
        if cmake --build "$ORB_SRC/build" -j "$j"; then built=1; break; fi
        warn "build failed at -j$j; retrying with fewer parallel jobs (likely OOM)"
    done
    [ "$built" = 1 ] || die "libORB_SLAM3.so failed to build even at -j1 -- see the log"

    [ -f "$ORB_SRC/lib/libORB_SLAM3.so" ] || die "no lib/libORB_SLAM3.so was produced"
    echo "    built $ORB_SRC/lib/libORB_SLAM3.so"
}

# -------------------------------------------------------------------- vocab ---
stage_vocab() {
    say "vocabulary"
    local voc="$ORB_SRC/Vocabulary/ORBvoc.txt"
    if [ -z "${FORCE:-}" ] && [ -s "$voc" ]; then
        printf '    already extracted (%s), skipping\n' "$(du -h "$voc" | cut -f1)"
        return
    fi
    require_space 400
    [ -f "$ORB_SRC/Vocabulary/ORBvoc.txt.tar.gz" ] || die "no ORBvoc.txt.tar.gz -- was ORB-SLAM3 cloned?"
    tar -xzf "$ORB_SRC/Vocabulary/ORBvoc.txt.tar.gz" -C "$ORB_SRC/Vocabulary"
    [ -s "$voc" ] || die "vocabulary extraction produced nothing"
    rm -f "$ORB_SRC/Vocabulary/ORBvoc.txt.tar.gz"   # 42 MB, recoverable from git
    printf '    extracted %s\n' "$(du -h "$voc" | cut -f1)"
}

# ------------------------------------------------------------------ cleanup ---
stage_cleanup() {
    say "cleanup"
    local before after
    before="$(free_mb)"
    rm -rf "$ORB_SRC/build" "$ORB_SRC/Thirdparty/DBoW2/build" "$PANGOLIN_SRC/build"
    # Debug symbols in these are large and we are not stepping through ORB-SLAM3.
    for so in "$ORB_SRC/lib/libORB_SLAM3.so" \
              "$ORB_SRC/Thirdparty/DBoW2/lib/libDBoW2.so" \
              "$ORB_SRC/Thirdparty/g2o/lib/libg2o.so"; do
        [ -f "$so" ] && strip --strip-unneeded "$so" 2>/dev/null || true
    done
    after="$(free_mb)"
    printf '    reclaimed %s MB (now %s MB free)\n' "$(( after - before ))" "$after"

    say "result"
    for so in "$PREFIX/lib/libpango_display.so" \
              "$ORB_SRC/lib/libORB_SLAM3.so" \
              "$ORB_SRC/Thirdparty/DBoW2/lib/libDBoW2.so" \
              "$ORB_SRC/Thirdparty/g2o/lib/libg2o.so" \
              "$ORB_SRC/Vocabulary/ORBvoc.txt"; do
        if [ -f "$so" ]; then printf '    ok      %s\n' "$so"
        else                  printf '    MISSING %s\n' "$so"; fi
    done
    # An unresolved dependency here shows up much later as a confusing colcon
    # link error, so surface it now.
    if ldd "$ORB_SRC/lib/libORB_SLAM3.so" 2>/dev/null | grep -q 'not found'; then
        warn "libORB_SLAM3.so has unresolved dependencies:"
        ldd "$ORB_SRC/lib/libORB_SLAM3.so" | grep 'not found' >&2
    fi
    printf '\n    next: colcon build --packages-select orbslam3_ros2\n'
}

# --------------------------------------------------------------------- main ---
STAGES=("$@")
[ ${#STAGES[@]} -eq 0 ] && STAGES=(preflight pangolin orbslam3 vocab cleanup)

for s in "${STAGES[@]}"; do
    case "$s" in
        preflight|pangolin|orbslam3|vocab|cleanup) "stage_$s" ;;
        *) die "unknown stage '$s' (want: preflight pangolin orbslam3 vocab cleanup)" ;;
    esac
done

say "done"
