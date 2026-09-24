#!/usr/bin/env python3
"""Convert a GIF to H.264 MP4, preserving its frame timing.

    python3 script/gif_to_video.py gif/a.gif gif/b.gif
    python3 script/gif_to_video.py gif/a.gif --out-dir video --crf 20

WHY NOT JUST RE-RENDER
    render_dynamic_gif.py already writes MP4 with --out x.mp4, but re-rendering
    re-reads the bag, and the dynamic bags live on a USB drive where that is the
    slow part. Converting the finished GIF costs seconds and is bit-identical in
    content to what is already on disk and already checked.

WHY imageio_ffmpeg AND NOT cv2.VideoWriter
    There is no system ffmpeg here, but imageio_ffmpeg ships a static build, so
    real H.264 is available. cv2's mp4v fallback is an MPEG-4 Part 2 codec: it
    produces files several times larger for the same quality and is refused by
    some browsers and slide tools, which is exactly where these end up.

    H.264 requires even frame dimensions. A GIF has no such rule, so an odd width
    or height is padded by one pixel rather than being silently rescaled -- the
    panels are drawn to scale and resampling them would falsify a metre.

FRAME TIMING
    Taken from the GIF's own per-frame duration rather than assumed. These are
    written at a uniform 200 ms (5 fps); a GIF with mixed durations is reported
    and encoded at the rate of its most common frame, since MP4 here is
    constant-rate.
"""

import argparse
import collections
import os
import sys

import numpy as np
from PIL import Image

try:
    import imageio_ffmpeg
except ImportError:
    sys.exit('imageio_ffmpeg is required (it provides the bundled ffmpeg binary)')


def convert(src, dst, crf, quiet=False):
    im = Image.open(src)
    n = getattr(im, 'n_frames', 1)

    durations = collections.Counter()
    for k in range(n):
        im.seek(k)
        durations[im.info.get('duration') or 100] += 1
    ms, _ = durations.most_common(1)[0]
    fps = 1000.0 / ms
    if len(durations) > 1 and not quiet:
        print('  note: %d distinct frame durations %s; encoding at %.2f fps'
              % (len(durations), dict(durations), fps))

    im.seek(0)
    w, h = im.size
    # H.264 needs even dimensions. Pad rather than resize: these frames are drawn
    # to a stated metres-per-pixel and rescaling them would change that scale.
    pw, ph = w + (w & 1), h + (h & 1)

    # pix_fmt goes through the named argument, not output_params: passing it in
    # both places makes ffmpeg warn that it is overriding its own flag.
    writer = imageio_ffmpeg.write_frames(
        dst, (pw, ph), fps=fps, codec='libx264', quality=None,
        pix_fmt_out='yuv420p', output_params=['-crf', str(crf)],
        macro_block_size=1)
    writer.send(None)
    try:
        for k in range(n):
            im.seek(k)
            a = np.asarray(im.convert('RGB'))
            if (pw, ph) != (w, h):
                pad = np.zeros((ph, pw, 3), dtype=a.dtype)
                pad[:h, :w] = a
                a = pad
            writer.send(np.ascontiguousarray(a))
    finally:
        writer.close()
    return n, (pw, ph), fps


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('gifs', nargs='+')
    ap.add_argument('--out-dir', help='default: alongside each source')
    ap.add_argument('--crf', type=int, default=23,
                    help='x264 quality, lower is better. 18 is near-lossless, '
                         '23 the x264 default, 28 visibly soft. Default 23.')
    a = ap.parse_args()

    for src in a.gifs:
        if not os.path.isfile(src):
            print('  [skip] %s: not found' % src)
            continue
        base = os.path.splitext(os.path.basename(src))[0] + '.mp4'
        dst = os.path.join(a.out_dir or os.path.dirname(os.path.abspath(src)), base)
        os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
        n, size, fps = convert(src, dst, a.crf)
        sz_in, sz_out = os.path.getsize(src), os.path.getsize(dst)
        print('  %-46s %4d frames  %dx%d  %.1f fps   %6.1f MB -> %5.1f MB  (%.0f%% smaller)'
              % (os.path.basename(src), n, size[0], size[1], fps,
                 sz_in / 1e6, sz_out / 1e6, 100 * (1 - sz_out / sz_in)))
        print('      %s' % dst)


if __name__ == '__main__':
    main()
