"""Draw a recording's warehouse as a 50 %-transparent underlay on a top-down axis.

    from map_underlay import draw_map
    draw_map(ax, "simulation/dataset_allsensor_realistic_002")

The footprints are viewer.load_obstacles(world): the convex hull of every
collision mesh sliced at robot height, the same geometry the occupancy bake and
the scene figures (chapter 3) use, in the world frame the ground truth is in --
so a trajectory aligned to ground truth sits on the map with no transform. The
world file comes from the run manifests via render_dataset_scene.world_for(),
with the same missing-floor-variant substitution that module documents.

Dynamic props are drawn where the world file authors them; the underlay is the
layout, not where a prop was at any instant. The wall is drawn as a ring.
"""
import os
import sys

import numpy as np
from matplotlib.patches import Polygon, Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import render_dataset_scene as rds   # noqa: E402  world_for / substitute_world
import viewer                         # noqa: E402  load_obstacles

_CACHE = {}
FILL = "#2f3542"


def footprints(dataset):
    """(obstacles, wall) for a dataset, or None when no world is recorded."""
    if not dataset.startswith(("simulation/", "real/")):
        dataset = "simulation/" + dataset
    world = rds.world_for(dataset)
    if world is None:
        return None
    if not os.path.isfile(world):
        sub = rds.substitute_world(world)
        if sub is None:
            return None
        world = sub[0]
    if world not in _CACHE:
        _CACHE[world] = viewer.load_obstacles(world)
    return _CACHE[world]


def draw_map(ax, dataset, alpha=0.5, fill=FILL, zorder=0.5):
    """Underlay the warehouse footprints at `alpha`; returns True if drawn."""
    fp = footprints(dataset)
    if fp is None:
        return False
    obstacles, wall = fp
    for _label, poly, _pose in obstacles:
        poly = np.asarray(poly, dtype=float)
        if len(poly) >= 3:
            ax.add_patch(Polygon(poly, closed=True, facecolor=fill, edgecolor="none", alpha=alpha, zorder=zorder))
    if wall is not None:
        x0, y0, x1, y1 = wall
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=fill, lw=2.0, alpha=alpha, zorder=zorder))
    return True
