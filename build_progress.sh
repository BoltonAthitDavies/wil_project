#!/usr/bin/env bash
# Where is the ORB-SLAM3 build up to? Safe to run any time, from any terminal.
# Reads the filesystem only -- starts nothing, changes nothing.
#
#   ./build_progress.sh          one snapshot
#   ./build_progress.sh -w       refresh every 5 s until you Ctrl-C
#   tail -f thirdparty/build.log raw compiler output

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORB="$ROOT/thirdparty/ORB_SLAM3"
LOG="$ROOT/thirdparty/build.log"

snapshot() {
    printf '\n\033[1m ORB-SLAM3 build  %s\033[0m\n\n' "$(date +%H:%M:%S)"

    # Is it actually alive? pgrep -f on the script name self-matches, so ask
    # about the compiler and make instead -- they cannot lie.
    local ncc; ncc=$(pgrep -xc cc1plus 2>/dev/null | head -1); ncc=${ncc:-0}
    if [ "$ncc" -gt 0 ]; then
        printf '  state    \033[1;32mBUILDING\033[0m  (%s compiler processes)\n' "$ncc"
    elif pgrep -x make >/dev/null 2>&1 || pgrep -x gmake >/dev/null 2>&1; then
        printf '  state    \033[1;32mBUILDING\033[0m  (linking / configuring)\n'
    elif pgrep -f 'bash .*build_orbslam3\.sh' >/dev/null 2>&1; then
        printf '  state    \033[1;32mBUILDING\033[0m  (script alive, between steps)\n'
    else
        printf '  state    \033[1;33midle\033[0m -- no compiler running\n'
    fi
    printf '  load     %s   (a live -j8 build sits near 8)\n' "$(cut -d' ' -f1-3 /proc/loadavg)"

    if [ -f "$ORB/CMakeLists.txt" ]; then
        local pending=() total=0 f
        while read -r f; do
            total=$((total + 1))
            [ -f "$ORB/build/CMakeFiles/ORB_SLAM3.dir/${f}.o" ] || pending+=("$(basename "$f")")
        done < <(awk '/^add_library/,/Settings.h\)/' "$ORB/CMakeLists.txt" 2>/dev/null \
                 | grep -oE '^(src|include)/[^ ]*\.(cc|cpp)')
        printf '  objects  %s / %s compiled\n' "$((total - ${#pending[@]}))" "$total"
        [ ${#pending[@]} -gt 0 ] && printf '  pending  %s\n' "${pending[*]}"
    fi

    printf '\n  artifacts\n'
    local f
    for f in "$ROOT/thirdparty/install/lib/libpango_display.so" \
             "$ORB/Thirdparty/DBoW2/lib/libDBoW2.so" \
             "$ORB/Thirdparty/g2o/lib/libg2o.so" \
             "$ORB/lib/libORB_SLAM3.so" \
             "$ORB/Vocabulary/ORBvoc.txt"; do
        if [ -f "$f" ]; then
            printf '    \033[32mok\033[0m       %-20s %s\n' "$(basename "$f")" "$(du -h "$f" | cut -f1)"
        else
            printf '    \033[31mmissing\033[0m  %s\n' "$(basename "$f")"
        fi
    done

    if [ -f "$LOG" ]; then
        printf '\n  log      %s  (last write %s ago)\n' "$LOG" \
            "$(( $(date +%s) - $(stat -c %Y "$LOG") ))s"
        printf '  tail     %s\n' "$(grep -E '^\[ *[0-9]+%\]' "$LOG" | tail -1 | cut -c1-72)"
        grep -qE 'Error|error:' "$LOG" && printf '    \033[31m^^ log contains errors -- grep -nE "error:" %s\033[0m\n' "$LOG"
    fi
    echo
}

if [ "${1:-}" = "-w" ]; then
    while true; do clear; snapshot; sleep 5; done
else
    snapshot
fi
