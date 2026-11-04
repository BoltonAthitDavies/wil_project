#!/usr/bin/env python3
"""Audit rosbag2 sqlite3 bags: open every store, count rows, compare to metadata.

metadata.yaml is written when recording starts and describes intent, not what
survived -- so it is never evidence a bag is intact.  This opens each .db3,
checks the SQLite header's page count against the pages actually on disk
(catching a truncated copy), then counts rows per topic and reports the span.
"""
import os, re, struct, sqlite3, sys, glob, time

def claimed(bag):
    """Pull counts out of metadata.yaml without a yaml dependency."""
    p = os.path.join(bag, 'metadata.yaml')
    if not os.path.exists(p):
        return None
    t = open(p, encoding='utf-8', errors='replace').read()
    out = {'topics': {}, 'files': []}
    m = re.search(r'^  message_count: (\d+)', t, re.M)
    out['total'] = int(m.group(1)) if m else None
    m = re.search(r'^  duration:\s*\n\s*nanoseconds: (\d+)', t, re.M)
    out['dur'] = int(m.group(1)) / 1e9 if m else None
    # topic blocks: name ... then the message_count that closes the block
    for blk in re.finditer(r'name: (\S+)\n.*?^      message_count: (\d+)',
                           t, re.M | re.S):
        out['topics'][blk.group(1)] = int(blk.group(2))
    out['files'] = re.findall(r'^    - (\S+\.db3)$', t, re.M)
    return out

def header(path):
    sz = os.path.getsize(path)
    with open(path, 'rb') as f:
        h = f.read(100)
    if h[:15] != b'SQLite format 3':
        return {'bad_magic': True, 'size': sz}
    ps = struct.unpack('>H', h[16:18])[0] or 65536
    return {'size': sz, 'ps': ps, 'ondisk': sz // ps, 'rem': sz % ps,
            'hdr_pages': struct.unpack('>I', h[28:32])[0],
            'freelist': struct.unpack('>I', h[36:40])[0]}

def probe(path):
    """Row counts straight from the store. Returns (info, error)."""
    try:
        db = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
        tnames = {i: n for i, n in db.execute("select id,name from topics")}
        rows = db.execute("select topic_id,count(*),min(timestamp),max(timestamp) "
                          "from messages group by topic_id").fetchall()
        total = db.execute("select count(*) from messages").fetchone()[0]
        db.close()
        per = {tnames.get(tid, f'<id {tid}>'): (n, lo, hi) for tid, n, lo, hi in rows}
        return {'total': total, 'per': per, 'ntopics': len(tnames)}, None
    except Exception as e:
        return None, str(e)

def main():
    bags = sorted(sys.argv[1:])
    width = max(len(os.path.basename(b.rstrip('/'))) for b in bags)
    verdicts = []
    for bag in bags:
        bag = bag.rstrip('/')
        name = os.path.basename(bag)
        c = claimed(bag)
        dbs = sorted(glob.glob(os.path.join(bag, '*.db3')))
        print(f"\n{'='*78}\n{name}")
        if c is None:
            print("  no metadata.yaml")
        listed = set(c['files']) if c else set()
        present = {os.path.basename(d) for d in dbs}
        if listed and listed != present:
            if listed - present:
                print(f"  !! metadata lists missing files: {sorted(listed-present)}")
            if present - listed:
                print(f"  !! files not listed in metadata: {sorted(present-listed)}")
        tot_rows, tot_claim = 0, (c['total'] if c else None)
        status = 'OK'
        spans = []
        for d in dbs:
            h = header(d)
            if h.get('bad_magic'):
                kind = ('EMPTY STUB' if h['size'] == 0 else 'NOT a SQLite file')
                print(f"  {os.path.basename(d)}: {kind} ({h['size']:,} B)")
                # a 0-byte stub that metadata points at blocks playback even
                # though the real data file beside it is fine
                status = 'STUB' if h['size'] == 0 else 'BROKEN'
                continue
            trunc = h['hdr_pages'] - h['ondisk']
            flag = ''
            if trunc > 0:
                flag = (f"  <-- TRUNCATED: {trunc:,} pages "
                        f"({trunc*h['ps']:,} B) missing, freelist {h['freelist']}")
                status = 'BROKEN'
            elif h['rem']:
                flag = f"  <-- partial page ({h['rem']} B)"
                status = 'BROKEN'
            print(f"  {os.path.basename(d)}: {h['size']:,} B, "
                  f"pages {h['ondisk']:,}/{h['hdr_pages']:,}{flag}")
            t0 = time.time()
            info, err = probe(d)
            if err:
                print(f"    OPEN FAILED: {err}")
                status = 'BROKEN'
                continue
            tot_rows += info['total']
            for n, (cnt, lo, hi) in info['per'].items():
                spans.append((lo, hi))
            print(f"    opened ok, {info['total']:,} messages, "
                  f"{info['ntopics']} topics  ({time.time()-t0:.1f}s)")
        if spans:
            lo, hi = min(s[0] for s in spans), max(s[1] for s in spans)
            print(f"  span: {(hi-lo)/1e9:.2f} s" +
                  (f"  (metadata claims {c['dur']:.2f} s)" if c and c['dur'] else ""))
        if tot_claim is not None:
            d = tot_rows - tot_claim
            verdict = ('exact match' if d == 0 else
                       f"{d:+,} vs metadata ({tot_rows/tot_claim*100:.1f}% present)")
            print(f"  messages: {tot_rows:,} actual / {tot_claim:,} claimed  -> {verdict}")
            if d != 0 and status == 'OK':
                status = 'MISMATCH'
        # per-topic detail only when something is off
        if c:
            for d in dbs:
                info, err = probe(d)
                if not info: continue
                for tname, cl in sorted(c['topics'].items()):
                    got = info['per'].get(tname, (0, 0, 0))[0]
                    mark = '' if got == cl else f"   <-- {got-cl:+,}"
                    print(f"      {tname:34} {got:>7,} / {cl:>7,}{mark}")
        print(f"  VERDICT: {status}")
        verdicts.append((name, status, tot_rows, tot_claim))

    print(f"\n\n{'='*78}\nSUMMARY\n{'='*78}")
    print(f"{'bag':<{width}}  {'status':<9} {'actual':>10} {'claimed':>10}")
    for name, st, a, cl in verdicts:
        claim = '-' if cl is None else format(cl, ',')
        print(f"{name:<{width}}  {st:<9} {a:>10,} {claim:>10}")
    bad = [v for v in verdicts if v[1] != 'OK']
    print(f"\n{len(verdicts)-len(bad)}/{len(verdicts)} healthy; "
          f"problems: {', '.join(n for n,_s,_a,_c in bad) or 'none'}")

main()
