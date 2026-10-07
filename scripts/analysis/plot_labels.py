"""Non-overlapping point labels for matplotlib scatter plots.

``place_labels`` tries offsets on growing rings around each point and keeps
the first one whose text box stays inside the axes and clears every other
label, every marker and any text box already on the axes (legends, corner
notes). A label that has to move past the first ring gets a thin leader line
back to its point. Call it after the layout is final (``tight_layout``),
because the check works in display coordinates.
"""

from __future__ import annotations

from matplotlib.transforms import Bbox

# (dx, dy) unit offsets, tried in this order on each ring.
_DIRECTIONS = [(1, 0.6), (1, -0.6), (-1, 0.6), (-1, -0.6),
               (0, 1), (0, -1), (1, 0), (-1, 0)]
_RING_PT = 7.0
_MARKER_PT = 5.0


def _ha_va(dx, dy):
    ha = 'left' if dx > 0 else 'right' if dx < 0 else 'center'
    va = 'bottom' if dy > 0 else 'top' if dy < 0 else 'center'
    return ha, va


def place_labels(ax, items, rings=6, extra_obstacles=()):
    """Annotate ``items`` on ``ax`` without overlaps.

    Each item is a dict with ``x``, ``y``, ``text`` and optional ``color``,
    ``size`` (font size), ``priority`` (higher is placed first) and
    ``required`` (default True). A required label that finds no free spot is
    drawn at the last ring with a leader line. An optional one is dropped.
    ``extra_obstacles`` are display-space boxes to keep clear (e.g. an inset).
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px_per_pt = fig.dpi / 72.0
    ax_box = ax.get_window_extent(renderer)
    obstacles = [t.get_window_extent(renderer).expanded(1.05, 1.1) for t in ax.texts]
    obstacles += list(extra_obstacles)
    legend = ax.get_legend()
    if legend is not None:
        obstacles.append(legend.get_window_extent(renderer))
    r = _MARKER_PT * px_per_pt
    markers = []
    for it in items:
        px, py = ax.transData.transform((it['x'], it['y']))
        markers.append(Bbox.from_extents(px - r, py - r, px + r, py + r))
    placed = []
    order = sorted(range(len(items)), key=lambda i: -items[i].get('priority', 0))
    for i in order:
        it = items[i]
        style = dict(fontsize=it.get('size', 7), color=it.get('color', 'k'), zorder=6)
        others = obstacles + placed + [m for j, m in enumerate(markers) if j != i]
        chosen = None
        for ring in range(1, rings + 1):
            for dx, dy in _DIRECTIONS:
                off = (dx * _RING_PT * ring, dy * _RING_PT * ring)
                ha, va = _ha_va(dx, dy)
                ann = ax.annotate(it['text'], (it['x'], it['y']), xytext=off,
                                  textcoords='offset points', ha=ha, va=va, **style)
                box = ann.get_window_extent(renderer)
                ann.remove()
                inside = (box.x0 >= ax_box.x0 and box.x1 <= ax_box.x1
                          and box.y0 >= ax_box.y0 and box.y1 <= ax_box.y1)
                if inside and not any(box.overlaps(o) for o in others):
                    chosen = (off, ha, va, ring, box)
                    break
            if chosen:
                break
        if chosen is None:
            if not it.get('required', True):
                continue
            off = (_RING_PT * rings, _RING_PT * rings * 0.6)
            ha, va = 'left', 'bottom'
            chosen = (off, ha, va, rings, None)
        off, ha, va, ring, box = chosen
        arrow = (dict(arrowstyle='-', color=style['color'], lw=0.5, alpha=0.7,
                      shrinkA=0, shrinkB=_MARKER_PT * 0.6) if ring > 1 else None)
        ann = ax.annotate(it['text'], (it['x'], it['y']), xytext=off, textcoords='offset points',
                          ha=ha, va=va, arrowprops=arrow, **style)
        placed.append(box if box is not None else ann.get_window_extent(renderer))
