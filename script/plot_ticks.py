"""Readable tick labels for logarithmic axes.

matplotlib's default log formatter prints 10^-1, 6x10^-2 and, on an axis that
spans less than a decade, sometimes a single labelled tick.  The report's
figures print plain numbers instead (0.05, 0.1, 0.2, 0.5, 1, 2, 5, ...), so a
reader can read a value off a log axis without converting a power of ten.

    from plot_ticks import plain_log
    ax.set_yscale("log"); plain_log(ax)            # y axis
    plain_log(ax, axis="x", subs=(1, 2, 5))        # any axis, chosen mantissas

Call it AFTER set_yscale / set_ylim, since both reset the locator.
"""
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter


def _fmt(v, _pos=None):
    if v <= 0:
        return ""
    return "%g" % float("%.3g" % v)


def plain_log(ax, axis="y", subs=(1.0, 2.0, 5.0)):
    """Label a log axis at mantissas `subs` of every decade, as plain numbers."""
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(LogLocator(base=10.0, subs=subs, numticks=30))
    a.set_major_formatter(FuncFormatter(_fmt))
    a.set_minor_locator(LogLocator(base=10.0, subs=[s for s in range(1, 10) if s not in subs], numticks=30))
    a.set_minor_formatter(NullFormatter())
