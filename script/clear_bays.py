#!/usr/bin/env python3
"""Strip every bay prop from a world, leaving the building and the wall furniture.

    python3 script/clear_bays.py --world <a.world> [--world <b.world>] [--dry-run]

Removes all live Bucket / ClutteringA / ClutteringC / ClutteringD includes -- the
four models reshuffle_bays.py and fill_bays.py place inside bays. Shelves, desks,
pallet jacks, the trash cans, the robot and the building shell are untouched, so
this turns a stocked world into an empty-bay one (the `*_objonwallonly` variants).

Every prop is checked to be inside a bay rectangle before anything is written. A
prop outside every bay is not a bay prop, and finding one aborts the run rather
than silently deleting scenery the world needed.

Commented-out includes are left alone: the text is comment-blanked before
scanning, so a commented block cannot be matched, and a live block following one
cannot be swallowed (a commented block ends `</include> -->`, so a match starting
at its `<include>` runs on to the next LIVE `</include>`).
"""
import re, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reshuffle_bays as R

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRE = R.PRE


def in_a_bay(x, y):
    return any(x0 <= x <= x1 and y0 <= y <= y1 for _k, x0, x1, y0, y1 in R.BAYS)


def main():
    worlds = [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == '--world']
    dry = '--dry-run' in sys.argv
    if not worlds:
        raise SystemExit(__doc__)

    for w in worlds:
        path = w if os.path.isabs(w) else os.path.join(ROOT, w)
        text = open(path).read()
        clean = R.blank_comments(text)

        spans, kept, counts = [], [], {}
        for m in re.finditer(r'[ \t]*<include>.*?</include>\n', clean, re.S):
            b = m.group(0)
            u = re.search(r'<uri>model://([^<]+)</uri>', b)
            if not u:
                continue
            full = u.group(1)
            model = full[len(PRE):] if full.startswith(PRE) else full
            if model not in R.PROP:
                continue
            p = [float(v) for v in re.search(r'<pose>([^<]*)</pose>', b).group(1).split()]
            if not in_a_bay(p[0], p[1]):
                kept.append((model, p[0], p[1]))
                continue
            spans.append((m.start(), m.end()))
            counts[model] = counts.get(model, 0) + 1

        if kept:
            for k in kept:
                print('  %s at (%.2f, %.2f) is outside every bay' % k)
            raise SystemExit('aborting %s: not every prop is a bay prop'
                             % os.path.basename(path))

        for a, b in reversed(spans):
            text = text[:a] + text[b:]

        print('  %s' % os.path.basename(path))
        for k in sorted(counts):
            print('      removed %2d %s' % (counts[k], k))
        print('      removed %d bay props in total' % len(spans))
        if not dry:
            open(path, 'w').write(text)

    if dry:
        print('\n(dry run -- nothing written)')


if __name__ == '__main__':
    main()
