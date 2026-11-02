#!/usr/bin/env bash
G=/tmp/claude-1000/-home-ambushee-wil-project/88846a24-8e90-425d-beed-b9cdd3dcb9c4/scratchpad/crash_rv5
: > $G/loop_progress.txt
for k in 1 2 3; do
  $G/gdb_repro.sh; cp $G/gdb_run/gdb.log $G/gdb_attempt$k.log
  n=$(grep -c SIGSEGV $G/gdb_run/gdb.log); echo "attempt $k sigsegv=$n reloc=$(grep -c Relocalized $G/gdb_run/gdb.log) $(date +%T)" >> $G/loop_progress.txt
  [ "$n" -gt 0 ] && break
done
echo "loop done" >> $G/loop_progress.txt
