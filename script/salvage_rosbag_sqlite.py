#!/usr/bin/env python3
"""Salvage a damaged rosbag2 sqlite3 store into a fresh, playable bag.

rosbag2's metadata.yaml is written when recording starts, so it is never
evidence a bag is intact -- verify by opening the store and counting rows.
When a .db3 is truncated (a copy that stopped short), SQLite refuses the whole
file with "database disk image is malformed" even though almost all of it is
readable.  This walks the table b-trees directly, skipping any page or overflow
chain that runs past EOF, and rebuilds whatever survived.

    # report only (read-only)
    script/salvage_rosbag_sqlite.py BAG_0.db3
    # rebuild into a new bag directory, then: ros2 bag reindex DIR -s sqlite3
    script/salvage_rosbag_sqlite.py BAG_0.db3 --out NEWDIR/NEWBAG_0.db3

Recovery is truncation, not interpolation: check the reported per-topic counts
and timestamp span before using a salvaged bag in a comparison.
"""
import os, struct, sys, mmap, argparse, sqlite3

def varint(buf, off):
    v = 0
    for i in range(9):
        b = buf[off + i]
        if i == 8:
            v = (v << 8) | b
            return v, off + 9
        v = (v << 7) | (b & 0x7F)
        if not (b & 0x80):
            return v, off + i + 1

class Store:
    def __init__(self, path):
        self.f = open(path, 'rb')
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        h = self.mm[:100]
        self.ps = struct.unpack('>H', h[16:18])[0] or 65536
        self.reserved = h[20]
        self.usable = self.ps - self.reserved
        self.size = len(self.mm)
        self.ondisk = self.size // self.ps
        self.hdr_pages = struct.unpack('>I', h[28:32])[0]
        self.missing = set()
        # Highest page still believed to hold real data. Pages above it are
        # present but shredded (in-place corruption rather than truncation),
        # so they must be refused exactly like pages past EOF -- otherwise a
        # garbage page decodes into a garbage row with no error raised.
        self.max_page = self.ondisk

    def page(self, n):          # 1-based
        if n < 1 or n > self.max_page:
            self.missing.add(n)
            return None
        off = (n - 1) * self.ps
        return self.mm[off:off + self.ps]

    # --- b-tree traversal -------------------------------------------------
    def leaf_cells(self, root, seen=None):
        """Yield (rowid, payload_bytes_or_None) for every table-leaf cell."""
        if seen is None:
            seen = set()
        stack = [root]
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            pg = self.page(n)
            if pg is None:
                continue
            base = 100 if n == 1 else 0
            ptype = pg[base]
            ncell = struct.unpack('>H', pg[base + 3:base + 5])[0]
            if ptype == 5:                       # interior table
                cp = base + 12
                kids = [struct.unpack('>I', pg[base + 8:base + 12])[0]]
                for i in range(ncell):
                    co = struct.unpack('>H', pg[cp + 2 * i:cp + 2 * i + 2])[0]
                    kids.append(struct.unpack('>I', pg[co:co + 4])[0])
                stack.extend(k for k in kids if k)
            elif ptype == 13:                    # leaf table
                cp = base + 8
                for i in range(ncell):
                    co = struct.unpack('>H', pg[cp + 2 * i:cp + 2 * i + 2])[0]
                    if co < base or co >= self.ps:
                        continue
                    try:
                        plen, o = varint(pg, co)
                        rowid, o = varint(pg, o)
                        yield rowid, self._payload(pg, o, plen)
                    except Exception:
                        continue
            # ptype 2/10 = index pages: not needed, we rebuild indexes

    def _payload(self, pg, off, plen):
        X = self.usable - 35
        if plen <= X:
            return bytes(pg[off:off + plen])
        M = ((self.usable - 12) * 32 // 255) - 23
        K = M + ((plen - M) % (self.usable - 4))
        local = K if K <= X else M
        out = bytearray(pg[off:off + local])
        nxt = struct.unpack('>I', pg[off + local:off + local + 4])[0]
        remaining = plen - local
        while remaining > 0 and nxt:
            op = self.page(nxt)
            if op is None:
                return None                      # chain runs into the void
            nxt = struct.unpack('>I', op[:4])[0]
            take = min(remaining, self.usable - 4)
            out += op[4:4 + take]
            remaining -= take
        return bytes(out) if remaining == 0 else None

    # --- orphan sweep ------------------------------------------------------
    def orphan_message_cells(self, skip_pages, ts_lo, ts_hi):
        """Scan every page for table-leaf cells that look like `messages` rows
        but are unreachable from the current root (stale interior pages)."""
        for n in range(1, self.max_page + 1):
            if n in skip_pages:
                continue
            pg = self.page(n)
            if pg is None or pg[0] != 13:
                continue
            ncell = struct.unpack('>H', pg[3:5])[0]
            if ncell == 0 or 8 + 2 * ncell >= self.ps:
                continue
            for i in range(ncell):
                co = struct.unpack('>H', pg[8 + 2 * i:10 + 2 * i])[0]
                if co < 8 + 2 * ncell or co >= self.ps:
                    continue
                try:
                    plen, o = varint(pg, co)
                    if plen < 4 or plen > 40_000_000:
                        continue
                    rowid, o = varint(pg, o)
                    if rowid < 1 or rowid > 100_000_000:
                        continue
                    pay = self._payload(pg, o, plen)
                    if pay is None:
                        continue
                    v = decode(pay)
                    if len(v) != 4:
                        continue
                    if not isinstance(v[1], int) or not 0 < v[1] < 1000:
                        continue
                    if not isinstance(v[2], int) or not ts_lo <= v[2] <= ts_hi:
                        continue
                    if not isinstance(v[3], (bytes, bytearray)):
                        continue
                    yield rowid, v
                except Exception:
                    continue

def decode(rec):
    """Decode a SQLite record into python values."""
    hlen, o = varint(rec, 0)
    types, p = [], o
    while p < hlen:
        t, p = varint(rec, p)
        types.append(t)
    vals, p = [], hlen
    for t in types:
        if t == 0:   vals.append(None)
        elif t <= 4: n = t;  vals.append(int.from_bytes(rec[p:p+n], 'big', signed=True)); p += n
        elif t == 5: vals.append(int.from_bytes(rec[p:p+6], 'big', signed=True)); p += 6
        elif t == 6: vals.append(int.from_bytes(rec[p:p+8], 'big', signed=True)); p += 8
        elif t == 7: vals.append(struct.unpack('>d', rec[p:p+8])[0]); p += 8
        elif t == 8: vals.append(0)
        elif t == 9: vals.append(1)
        elif t >= 12 and t % 2 == 0: n = (t-12)//2; vals.append(bytes(rec[p:p+n])); p += n
        elif t >= 13: n = (t-13)//2; vals.append(rec[p:p+n].decode('utf-8','replace')); p += n
        else: vals.append(None)
    return vals

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('--out')
    ap.add_argument('--max-page', type=int, default=None, metavar='N',
                    help='treat pages above N as unreadable: for a store whose '
                         'tail was overwritten with garbage in place, where the '
                         'pages exist but their contents are not the bag')
    a = ap.parse_args()
    s = Store(a.src)
    if a.max_page is not None:
        s.max_page = min(a.max_page, s.ondisk)
        print(f"treating pages above {s.max_page:,} as unreadable "
              f"({s.ondisk - s.max_page:,} of {s.ondisk:,} refused)")
    print(f"pages on disk {s.ondisk:,} / header {s.hdr_pages:,} "
          f"(missing {s.hdr_pages - s.ondisk:,})", flush=True)

    # schema from page 1
    master = {}
    for rowid, rec in s.leaf_cells(1):
        if rec is None: continue
        typ, name, tbl, root, sql = decode(rec)[:5]
        master[name] = (typ, tbl, root, sql)
    for k, v in master.items():
        print(f"  {v[0]:5} {k:16} root={v[2]}")

    tables = {k: v for k, v in master.items() if v[0] == 'table'}
    data = {}
    for name, (_t, _tbl, root, sql) in tables.items():
        rows, lost = [], 0
        for rowid, rec in s.leaf_cells(root):
            if rec is None:
                lost += 1
                continue
            rows.append((rowid, decode(rec)))
        data[name] = rows
        print(f"{name}: {len(rows):,} rows recovered, {lost:,} lost to missing overflow",
              flush=True)
    print(f"pages requested beyond the readable region: {len(s.missing):,}")

    if 'messages' not in tables:
        print('\nno `messages` table in sqlite_master: the schema page itself is\n'
              'gone, nothing can be walked. This store is unrecoverable.')
        return

    msgs = data.get('messages', [])
    topics = {r[0]: r[1] for r in data.get('topics', [])}
    percount = {}
    for _rid, v in msgs:
        percount[v[1]] = percount.get(v[1], 0) + 1
    print("\nper-topic recovered:")
    for tid, n in sorted(percount.items()):
        nm = topics.get(tid, ['?', '?'])[1] if tid in topics else f'id{tid}'
        print(f"  {nm:34} {n:,}")
    if msgs:
        ts = [v[2] for _r, v in msgs]
        print(f"\ntimestamp span: {min(ts)} .. {max(ts)}  "
              f"({(max(ts)-min(ts))/1e9:.2f} s)")


    # orphan sweep for rows the stale tree no longer references
    reach = set()
    list(s.leaf_cells(tables['messages'][2], reach))
    have = {rid for rid, _v in msgs}
    ts_lo, ts_hi = 1_000_000_000_000_000_000, 2_000_000_000_000_000_000
    extra = {}
    for rid, v in s.orphan_message_cells(reach, ts_lo, ts_hi):
        if rid not in have and rid not in extra:
            extra[rid] = v
    print(f"\norphan sweep: {len(extra):,} additional messages "
          f"(scanned {s.ondisk - len(reach):,} unreachable pages)", flush=True)
    if extra:
        pc2 = {}
        for v in extra.values():
            pc2[v[1]] = pc2.get(v[1], 0) + 1
        for tid, n in sorted(pc2.items()):
            nm = topics.get(tid, ['?', '?'])[1] if tid in topics else f'id{tid}'
            print(f"  + {nm:32} {n:,}")
        msgs.extend(sorted(extra.items()))
        data['messages'] = msgs
        allts = [v[2] for _r, v in msgs]
        print(f"combined: {len(msgs):,} messages, span "
              f"{(max(allts)-min(allts))/1e9:.2f} s")

    if not a.out:
        return
    write(s, master, tables, a.out)


def write(s, master, tables, out):
    """Stream the salvaged rows into a fresh, healthy store."""
    if os.path.exists(out):
        sys.exit(f"refusing to overwrite existing {out}")
    db = sqlite3.connect(out)
    db.execute("pragma journal_mode=off")
    db.execute("pragma synchronous=off")
    for name, (typ, _tbl, _root, sql) in master.items():
        if typ == 'table':
            db.execute(sql)
    for name in ('schema', 'metadata', 'topics'):
        if name not in tables:
            continue
        rows = []
        for rid, rec in s.leaf_cells(tables[name][2]):
            if rec is None:
                continue
            v = decode(rec)
            v[0] = rid if v[0] is None else v[0]
            rows.append(v)
        if rows:
            db.executemany(
                f"insert into {name} values({','.join('?' * len(rows[0]))})", rows)
            print(f"{name}: {len(rows)} rows")
    db.commit()

    n, lost, batch = 0, 0, []
    for rid, rec in s.leaf_cells(tables['messages'][2]):
        if rec is None:
            lost += 1
            continue
        v = decode(rec)
        batch.append((rid, v[1], v[2], v[3]))
        if len(batch) >= 500:
            db.executemany("insert into messages (id,topic_id,timestamp,data) "
                           "values(?,?,?,?)", batch)
            n += len(batch)
            batch.clear()
            if n % 20000 == 0:
                db.commit()
                print(f"  {n:,} messages...", flush=True)
    if batch:
        db.executemany("insert into messages (id,topic_id,timestamp,data) "
                       "values(?,?,?,?)", batch)
        n += len(batch)
    db.commit()
    print(f"messages: {n:,} written, {lost} unrecoverable")

    for name, (typ, _tbl, _root, sql) in master.items():
        if typ == 'index' and sql:
            db.execute(sql)
    db.commit()
    print("integrity_check:", db.execute("pragma integrity_check").fetchone()[0])
    db.close()
    print(f"wrote {out} ({os.path.getsize(out):,} bytes)")
    print("now run:  ros2 bag reindex <dir> -s sqlite3")



if __name__ == "__main__":
    main()
