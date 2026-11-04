#!/usr/bin/env python3
"""Copy files off a failing disk, tolerating unreadable blocks.

A plain cp/rsync aborts on the first EIO and leaves nothing. This reads in 1 MiB
blocks, retries a failed block at 4 KiB granularity, zero-fills whatever stays
unreadable, and records every bad range -- so a damaged bag still yields a file
whose good parts are intact and whose holes are known.

    script/rescue_copy.py SRC_DIR DST_DIR
"""
import os, sys, time

BLOCK = 1 << 20
FINE  = 4096
RETRY = 3

def rescue(src, dst):
    size = os.path.getsize(src)
    fd = os.open(src, os.O_RDONLY)
    bad = []
    done = 0
    t0 = time.time()
    with open(dst, 'wb') as out:
        while done < size:
            want = min(BLOCK, size - done)
            data = None
            for _ in range(RETRY):
                try:
                    os.lseek(fd, done, os.SEEK_SET)
                    data = os.read(fd, want)
                    if len(data) == want:
                        break
                    data = None
                except OSError:
                    data = None
                time.sleep(0.1)
            if data is None:                      # descend to 4 KiB
                buf = bytearray()
                for off in range(0, want, FINE):
                    n = min(FINE, want - off)
                    chunk = None
                    for _ in range(RETRY):
                        try:
                            os.lseek(fd, done + off, os.SEEK_SET)
                            chunk = os.read(fd, n)
                            if len(chunk) == n:
                                break
                            chunk = None
                        except OSError:
                            chunk = None
                        time.sleep(0.05)
                    if chunk is None:
                        buf += b'\0' * n
                        bad.append((done + off, n))
                    else:
                        buf += chunk
                data = bytes(buf)
            out.write(data)
            done += want
            if done % (256 << 20) < BLOCK:
                el = time.time() - t0
                print(f"    {done/1e9:5.2f}/{size/1e9:.2f} GB  "
                      f"{done/1e6/max(el,.01):5.1f} MB/s  bad={len(bad)}", flush=True)
        out.flush()
        os.fsync(out.fileno())
    os.close(fd)
    # merge adjacent bad ranges
    merged = []
    for off, n in bad:
        if merged and merged[-1][0] + merged[-1][1] == off:
            merged[-1][1] += n
        else:
            merged.append([off, n])
    return size, merged

def main():
    srcdir, dstdir = sys.argv[1], sys.argv[2]
    os.makedirs(dstdir, exist_ok=True)
    report = []
    for name in sorted(os.listdir(srcdir)):
        s = os.path.join(srcdir, name)
        if not os.path.isfile(s):
            continue
        d = os.path.join(dstdir, name)
        print(f"  {name} ({os.path.getsize(s)/1e9:.2f} GB)", flush=True)
        size, bad = rescue(s, d)
        lost = sum(n for _o, n in bad)
        report.append((name, size, lost, len(bad)))
        print(f"    -> {size-lost:,} / {size:,} bytes recovered"
              f"  ({(size-lost)/size*100:.4f}%), {len(bad)} bad range(s),"
              f" {lost:,} bytes zero-filled", flush=True)
        if bad:
            for off, n in bad[:10]:
                print(f"       hole at {off:,} +{n:,}", flush=True)
    print("\nSUMMARY")
    for name, size, lost, nb in report:
        tag = "CLEAN" if lost == 0 else f"{lost:,} B LOST in {nb} range(s)"
        print(f"  {name:<46} {size/1e9:6.2f} GB  {tag}")

main()
