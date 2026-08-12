"""
Structured Football Concepts for FootballVision

Once players live in pitch metres (see src/homography.py), we can derive the
same structured concepts an analyst reasons about — the layer Arsenal's JD
calls "structured football concepts". This module computes, per frame, from
the list of players (each with team + pitch position):

    - team centroid          : average team position (block location)
    - compactness            : mean distance of players to their centroid
    - width / depth          : lateral / longitudinal spread of the block
    - line height            : how high up the pitch the block sits (centroid x)
    - convex hull            : the team's shape polygon
    - pitch control (Voronoi): share of the pitch each team is nearest to
    - pressing proxy         : mean distance from each player to its nearest
                               opponent (smaller = tighter engagement)

Pitch control uses the nearest-player Voronoi model — every point on the
pitch is "controlled" by whichever team has the closest player. We evaluate
it on a coarse grid (fast, robust, and easy to render as a shaded overlay).

All distances are in metres. Inputs are plain dicts so this module has no
dependency on the tracker/pipeline types.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

TEAM_A = 0
TEAM_B = 1


def compute_metrics(
    players: Sequence[Dict],
    pitch_length: float,
    pitch_width: float,
    grid_cols: int = 32,
    grid_rows: int = 20,
) -> Dict:
    """
    Compute structured football concepts for one frame.

    Args:
        players: list of {"id", "team", "x", "y"} with x,y in pitch metres.
        pitch_length, pitch_width: pitch dimensions in metres.
        grid_cols, grid_rows: resolution of the pitch-control grid.

    Returns:
        {"teams": {...}, "control": {...}, "control_grid": {...},
         "pressing_m": float | None}
    """
    by_team: Dict[int, List[Dict]] = {TEAM_A: [], TEAM_B: []}
    for p in players:
        if p.get("team") in (TEAM_A, TEAM_B):
            by_team[p["team"]].append(p)

    teams = {
        str(t): _team_shape(by_team[t], pitch_length)
        for t in (TEAM_A, TEAM_B)
    }
    control, grid = _pitch_control(
        by_team, pitch_length, pitch_width, grid_cols, grid_rows
    )
    pressing = _pressing_proxy(by_team)

    return {
        "teams": teams,
        "control": control,
        "control_grid": grid,
        "pressing_m": pressing,
    }


# ---------------------------------------------------------------------------
# Per-team shape
# ---------------------------------------------------------------------------
def _team_shape(team_players: List[Dict], pitch_length: float) -> Optional[Dict]:
    if not team_players:
        return None

    pts = np.array([[p["x"], p["y"]] for p in team_players], dtype=np.float64)
    centroid = pts.mean(axis=0)
    dists = np.linalg.norm(pts - centroid, axis=1)

    xs, ys = pts[:, 0], pts[:, 1]
    shape = {
        "n": len(team_players),
        "centroid": [round(float(centroid[0]), 2), round(float(centroid[1]), 2)],
        "compactness_m": round(float(dists.mean()), 2),
        "spread_m": round(float(dists.std()), 2),
        "width_m": round(float(ys.max() - ys.min()), 2),
        "depth_m": round(float(xs.max() - xs.min()), 2),
        "line_height_m": round(float(centroid[0]), 2),
        "hull": _convex_hull(pts),
    }
    return shape


def _convex_hull(pts: np.ndarray) -> List[List[float]]:
    """Ordered convex-hull polygon (metres). Degrades gracefully for < 3 pts."""
    if len(pts) < 3:
        return [[round(float(x), 2), round(float(y), 2)] for x, y in pts]
    try:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(pts)
        return [
            [round(float(pts[v, 0]), 2), round(float(pts[v, 1]), 2)]
            for v in hull.vertices
        ]
    except Exception:
        # Collinear points or scipy hiccup — fall back to the point set.
        return [[round(float(x), 2), round(float(y), 2)] for x, y in pts]


# ---------------------------------------------------------------------------
# Pitch control (nearest-player Voronoi over a coarse grid)
# ---------------------------------------------------------------------------
def _pitch_control(
    by_team: Dict[int, List[Dict]],
    pitch_length: float,
    pitch_width: float,
    cols: int,
    rows: int,
):
    a_pts = np.array([[p["x"], p["y"]] for p in by_team[TEAM_A]], dtype=np.float64)
    b_pts = np.array([[p["x"], p["y"]] for p in by_team[TEAM_B]], dtype=np.float64)

    empty_grid = {"cols": cols, "rows": rows, "cells": []}
    if len(a_pts) == 0 and len(b_pts) == 0:
        return {"0": 0.0, "1": 0.0}, empty_grid

    # Grid cell centres in metres.
    xs = (np.arange(cols) + 0.5) / cols * pitch_length
    ys = (np.arange(rows) + 0.5) / rows * pitch_width
    gx, gy = np.meshgrid(xs, ys)  # (rows, cols)
    grid_pts = np.stack([gx.ravel(), gy.ravel()], axis=1)  # (rows*cols, 2)

    da = _min_dist_to(grid_pts, a_pts)
    db = _min_dist_to(grid_pts, b_pts)

    # cell owner: 0 -> team A, 1 -> team B
    cells = np.where(da <= db, TEAM_A, TEAM_B).astype(int)
    total = cells.size
    a_share = float((cells == TEAM_A).sum()) / total
    b_share = 1.0 - a_share

    control = {"0": round(a_share * 100, 1), "1": round(b_share * 100, 1)}
    grid = {"cols": cols, "rows": rows, "cells": cells.tolist()}
    return control, grid


def _min_dist_to(query: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Distance from each query point to the nearest point in `pts`."""
    if len(pts) == 0:
        return np.full(len(query), np.inf)
    try:
        from scipy.spatial import cKDTree

        d, _ = cKDTree(pts).query(query, k=1)
        return d
    except Exception:
        # Vectorised brute force fallback.
        diff = query[:, None, :] - pts[None, :, :]
        return np.sqrt((diff ** 2).sum(axis=2)).min(axis=1)


# ---------------------------------------------------------------------------
# Pressing proxy
# ---------------------------------------------------------------------------
def _pressing_proxy(by_team: Dict[int, List[Dict]]) -> Optional[float]:
    """Mean distance from each player to its nearest opponent (metres)."""
    a = np.array([[p["x"], p["y"]] for p in by_team[TEAM_A]], dtype=np.float64)
    b = np.array([[p["x"], p["y"]] for p in by_team[TEAM_B]], dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        return None
    da = _min_dist_to(a, b)
    db = _min_dist_to(b, a)
    return round(float(np.concatenate([da, db]).mean()), 2)


if __name__ == "__main__":
    # Two compact blocks on opposite halves of a 105x68 pitch.
    players = []
    for i in range(5):
        players.append({"id": i, "team": 0, "x": 30 + i, "y": 20 + i * 6})
    for i in range(5):
        players.append({"id": 10 + i, "team": 1, "x": 75 - i, "y": 20 + i * 6})

    m = compute_metrics(players, 105.0, 68.0)
    ta, tb = m["teams"]["0"], m["teams"]["1"]
    print("Team A centroid:", ta["centroid"], "| compactness:", ta["compactness_m"], "m")
    print("Team B centroid:", tb["centroid"], "| compactness:", tb["compactness_m"], "m")
    print("Pitch control %:", m["control"])
    print("Pressing (mean nearest-opponent):", m["pressing_m"], "m")
    print("Grid cells:", len(m["control_grid"]["cells"]),
          f"({m['control_grid']['cols']}x{m['control_grid']['rows']})")
    assert abs(m["control"]["0"] + m["control"]["1"] - 100.0) < 0.1
    assert ta["centroid"][0] < tb["centroid"][0]  # A on the left half
    print("\nMETRICS OK")
