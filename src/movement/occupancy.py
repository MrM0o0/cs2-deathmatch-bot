"""Read the radar as a local occupancy map.

The CS2 radar renders the playable area lit and out-of-bounds/walls as black.
By casting rays out from the player dot we learn, per direction, how far it is
walkable before hitting black -- i.e. where the bot can actually go. This is the
"radar as a live map" idea: steer toward open directions, away from black, with
no per-map waypoints and no fragile heading read.

Angles use the same convention as the rest of the nav stack: degrees via
atan2(dy, dx) in minimap pixels (x right, y down) -- 0 = east, +90 = south.
"""

import math


def radar_clearances(gray, cx, cy, n_rays=24, max_r=46, start_r=5, black_thresh=45):
    """For each of n_rays directions, the walkable distance (px) from (cx, cy)
    before the radar goes black (or the edge). gray is a single-channel
    brightness image of the minimap region. Returns a list indexed by ray k,
    whose angle is 360*k/n_rays degrees."""
    h, w = gray.shape[:2]
    out = []
    for k in range(n_rays):
        a = 2.0 * math.pi * k / n_rays
        dx, dy = math.cos(a), math.sin(a)
        dist = max_r
        for rr in range(start_r, max_r + 1):
            x = int(round(cx + dx * rr))
            y = int(round(cy + dy * rr))
            if x < 0 or y < 0 or x >= w or y >= h:
                dist = rr
                break
            if gray[y, x] < black_thresh:  # hit out-of-bounds / wall
                dist = rr
                break
        out.append(dist)
    return out


def clearance_at(clrs, deg):
    """Walkable distance in the direction `deg` (nearest ray)."""
    n = len(clrs)
    k = int(round((deg % 360) / 360.0 * n)) % n
    return clrs[k]


def best_direction(clrs, motion_deg, turn_penalty=0.18, extra=None):
    """Pick the direction to head: the most open ray, penalised for how far it is
    from the current travel direction (prefer to keep going / gentle course
    changes, not spin around). `extra`, if given, is a per-ray additive score --
    used to bias toward unexplored areas (visit memory) so it doesn't circle.
    Returns degrees."""
    n = len(clrs)
    best_deg, best_score = motion_deg, -1e9
    for k in range(n):
        deg = 360.0 * k / n
        ang_dist = abs(((deg - motion_deg + 180) % 360) - 180)
        score = clrs[k] - turn_penalty * ang_dist
        if extra is not None:
            score += extra[k]
        if score > best_score:
            best_score, best_deg = score, deg
    return best_deg
