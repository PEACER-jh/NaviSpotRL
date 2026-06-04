"""Global planner: contour extraction, topological graph, A*/Dijkstra path search.

Inspired by FAR Planner's contour-graph approach, adapted for 2D occupancy grids.

Key differences from the old implementation:
  1. Contour vertices are NOT pushed outward — they stay on the obstacle boundary.
  2. Visibility edges are checked via geometric line-polygon intersection, not grid sampling.
  3. No max-edge-length limit — any two free nodes with clean line-of-sight are connected.
  4. A* search (with optional Dijkstra fallback) replaces plain Dijkstra.
"""

from __future__ import annotations

import math
import os
import heapq
import random
from collections import defaultdict
from typing import Optional

import numpy as np
import cv2
from scipy import ndimage


# ===================== Geometry Utilities (adapted from FAR Planner intersection.h) =====================

def _orientation(px: float, py: float, qx: float, qy: float, rx: float, ry: float) -> int:
    """Return 0=colinear, 1=clockwise, 2=counter-clockwise for ordered triplet (p,q,r)."""
    val = (qy - py) * (rx - qx) - (qx - px) * (ry - qy)
    if abs(val) < 1e-9:
        return 0
    return 1 if val > 0 else 2


def _on_segment(px: float, py: float, qx: float, qy: float, rx: float, ry: float) -> bool:
    """Return True if point q lies on segment pr (assuming colinear)."""
    return (qx <= max(px, rx) + 1e-9 and qx >= min(px, rx) - 1e-9 and
            qy <= max(py, ry) + 1e-9 and qy >= min(py, ry) - 1e-9)


def segments_intersect(p1: tuple[float, float], q1: tuple[float, float],
                       p2: tuple[float, float], q2: tuple[float, float]) -> bool:
    """Return True if line segments p1q1 and p2q2 intersect."""
    o1 = _orientation(p1[0], p1[1], q1[0], q1[1], p2[0], p2[1])
    o2 = _orientation(p1[0], p1[1], q1[0], q1[1], q2[0], q2[1])
    o3 = _orientation(p2[0], p2[1], q2[0], q2[1], p1[0], p1[1])
    o4 = _orientation(p2[0], p2[1], q2[0], q2[1], q1[0], q1[1])
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_segment(p1[0], p1[1], p2[0], p2[1], q1[0], q1[1]):
        return True
    if o2 == 0 and _on_segment(p1[0], p1[1], q2[0], q2[1], q1[0], q1[1]):
        return True
    if o3 == 0 and _on_segment(p2[0], p2[1], p1[0], p1[1], q2[0], q2[1]):
        return True
    if o4 == 0 and _on_segment(p2[0], p2[1], q1[0], q1[1], q2[0], q2[1]):
        return True
    return False


def _line_collides_polygon(poly: list[tuple[float, float]],
                           edge: tuple[tuple[float, float], tuple[float, float]]) -> bool:
    """Return True if the line edge intersects any side of polygon."""
    m = len(poly)
    if m < 2:
        return False
    for i in range(m):
        side = (poly[i], poly[(i + 1) % m])
        if segments_intersect(edge[0], edge[1], side[0], side[1]):
            return True
    return False


def _reproject_away(x: float, y: float, poly: list[tuple[float, float]], dist: float) -> tuple[float, float]:
    """Push a point on a polygon edge outward (away from obstacle interior) by *dist*.

    The outward direction is estimated from the two incident polygon edges.
    """
    m = len(poly)
    # Find which edge the point is closest to (by projection distance to each edge)
    best_idx = 0
    best_dist = float('inf')
    for i in range(m):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % m]
        abx, aby = bx - ax, by - ay
        ab_len_sq = abx * abx + aby * aby + 1e-12
        t = max(0.0, min(1.0, ((x - ax) * abx + (y - ay) * aby) / ab_len_sq))
        px, py = ax + t * abx, ay + t * aby
        d = (x - px) ** 2 + (y - py) ** 2
        if d < best_dist:
            best_dist = d
            best_idx = i

    # Compute edge normal (perpendicular, pointing outward from polygon center)
    ax, ay = poly[best_idx]
    bx, by = poly[(best_idx + 1) % m]
    edge_x, edge_y = bx - ax, by - ay
    # Perpendicular (counter-clockwise = outward for CCW polygon)
    nx, ny = -edge_y, edge_x
    n_len = math.hypot(nx, ny) + 1e-12
    nx, ny = nx / n_len, ny / n_len

    # Determine if outward or inward by checking polygon centroid
    cx = sum(p[0] for p in poly) / m
    cy = sum(p[1] for p in poly) / m
    mid_x, mid_y = (ax + bx) / 2.0, (ay + by) / 2.0
    # If normal points toward centroid, flip it
    to_center = (cx - mid_x, cy - mid_y)
    if nx * to_center[0] + ny * to_center[1] > 0:
        nx, ny = -nx, -ny

    return x + nx * dist, y + ny * dist


def _point_inside_polygon(point: tuple[float, float], poly: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test."""
    x, y = point
    inside = False
    n = len(poly)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


# ===================== Contour Extraction =====================

def extract_obstacle_contours(occupied_grid: np.ndarray, origin_x: float, origin_y: float,
                              resolution: float, min_pts: int = 4) -> list[list[tuple[float, float]]]:
    """Extract obstacle boundary polygons from an occupancy grid (True=occupied).

    Contour vertices stay on the obstacle boundary (no outward pushing).  Each
    contour is approximated via ``cv2.approxPolyDP`` with an epsilon of 2% of
    the arc length.

    Returns:
        List of contours; each contour is a list of (x, y) world-coordinate vertices.
    """
    h, w = occupied_grid.shape
    img = (occupied_grid.astype(np.uint8) * 255)
    contours_cv, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is None:
        return []

    grid_area = occupied_grid.size
    contours: list[list[tuple[float, float]]] = []
    for cnt in contours_cv:
        area = cv2.contourArea(cnt)
        # Skip contours that are too small or too large
        if area < 3.0 or area > grid_area * 0.5:
            continue

        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        pts_world = []
        for pt in approx:
            px, py = int(pt[0][0]), int(pt[0][1])
            wx = origin_x + (px + 0.5) * resolution
            wy = origin_y + (py + 0.5) * resolution
            pts_world.append((float(wx), float(wy)))

        # Remove duplicate closing vertex
        if len(pts_world) >= 2:
            d = math.hypot(pts_world[0][0] - pts_world[-1][0],
                           pts_world[0][1] - pts_world[-1][1])
            if d < resolution * 0.5:
                pts_world.pop()

        if len(pts_world) >= min_pts:
            contours.append(pts_world)

    return contours


# ===================== Topological Graph =====================

class TopoNode:
    """Node in the topological visibility graph."""
    __slots__ = ('x', 'y', 'id', 'neighbors', 'is_start', 'is_goal',
                 'free_direct', 'contour_id', 'poly_ptr')

    def __init__(self, x: float, y: float, node_id: int,
                 is_start: bool = False, is_goal: bool = False,
                 free_direct: str = "unknown",
                 contour_id: int = -1,
                 poly_ptr: list[tuple[float, float]] | None = None):
        self.x = x
        self.y = y
        self.id = node_id
        self.neighbors: list[TopoNode] = []
        self.is_start = is_start
        self.is_goal = is_goal
        self.free_direct = free_direct
        self.contour_id = contour_id
        self.poly_ptr = poly_ptr


def _build_graph(contours: list[list[tuple[float, float]]],
                 start_xy: tuple[float, float],
                 goal_xy: tuple[float, float]) -> list[TopoNode]:
    """Build the topological graph: nodes from contours + start + goal.

    Every contour vertex becomes a node.  Consecutive vertices on the same
    contour are connected (intra-contour edges).
    """
    graph: list[TopoNode] = []
    node_id = 0

    # Start node
    start_node = TopoNode(start_xy[0], start_xy[1], node_id, is_start=True, free_direct="start")
    node_id += 1
    graph.append(start_node)

    # Goal node
    goal_node = TopoNode(goal_xy[0], goal_xy[1], node_id, is_goal=True, free_direct="goal")
    node_id += 1
    graph.append(goal_node)

    # Contour nodes — keep original boundary positions
    for cid, contour in enumerate(contours):
        m = len(contour)
        for i, (px, py) in enumerate(contour):
            node = TopoNode(px, py, node_id, free_direct="unknown",
                           contour_id=cid, poly_ptr=contour)
            node_id += 1
            graph.append(node)

    # Build intra-contour edges (front/back pointers along the polygon boundary)
    # All contour nodes: group by contour_id
    contour_groups: dict[int, list[TopoNode]] = defaultdict(list)
    for node in graph:
        if node.contour_id >= 0:
            contour_groups[node.contour_id].append(node)

    for cid, nodes in contour_groups.items():
        m = len(nodes)
        poly = nodes[0].poly_ptr
        if poly is None:
            continue
        # Connect consecutive vertices along the polygon boundary
        for i in range(m):
            a = nodes[i]
            b = nodes[(i + 1) % m]
            if b not in a.neighbors:
                a.neighbors.append(b)
            if a not in b.neighbors:
                b.neighbors.append(a)

        # Mark convexity (simplified): a node is convex if the interior angle < 180°
        for i in range(m):
            a = poly[(i - 1) % m]
            b = poly[i]
            c = poly[(i + 1) % m]
            # Cross product: (b-a) × (c-b)  — positive = left turn (convex for CCW polygon)
            cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
            nodes[i].free_direct = "convex" if cross > 0 else "concave"

    return graph


def _build_visibility_edges(graph: list[TopoNode],
                            raw_contours: list[list[tuple[float, float]]],
                            raw_grid: np.ndarray,
                            origin_x: float, origin_y: float,
                            resolution: float,
                            max_edge_len: float) -> None:
    """Add visibility edges between any two free nodes whose line-of-sight
    does not intersect any **raw** obstacle contour polygon or occupied grid cell.

    Hybrid check: geometric contour collision + grid sampling along the edge
    (catches over-simplified polygons that miss actual obstacles).
    """
    hard_polys = list(raw_contours)
    h, w = raw_grid.shape

    n = len(graph)
    for i in range(n):
        ni = graph[i]
        for j in range(i + 1, n):
            nj = graph[j]

            # Skip same-contour pairs (already have intra-contour edges)
            if ni.contour_id >= 0 and ni.contour_id == nj.contour_id:
                continue

            # Use ORIGINAL coordinates (no reprojection).  Self-collision with
            # the source contour is avoided by skipping the owner polygon below.
            edge = ((ni.x, ni.y), (nj.x, nj.y))

            # Check 1: geometric collision — skip polygons that own either endpoint
            blocked = False
            for poly in hard_polys:
                if poly is ni.poly_ptr or poly is nj.poly_ptr:
                    continue
                if _line_collides_polygon(poly, edge):
                    blocked = True
                    break

            # Check 2: grid sampling along the edge (catches approxPolyDP gaps)
            if not blocked:
                edge_len = math.hypot(nj.x - ni.x, nj.y - ni.y)
                steps = max(2, int(edge_len / (resolution * 0.5)))
                for k in range(steps + 1):
                    t = k / steps
                    sx = ni.x + t * (nj.x - ni.x)
                    sy = ni.y + t * (nj.y - ni.y)
                    gx = int((sx - origin_x) / resolution)
                    gy = int((sy - origin_y) / resolution)
                    if gx < 0 or gx >= w or gy < 0 or gy >= h:
                        blocked = True
                        break
                    if raw_grid[gy, gx]:
                        blocked = True
                        break

            if not blocked:
                # Final dense verification — double the sampling density
                fine_steps = max(4, int(edge_len / (resolution * 0.25)))
                fine_blocked = False
                for k in range(fine_steps + 1):
                    t = k / fine_steps
                    sx = ni.x + t * (nj.x - ni.x)
                    sy = ni.y + t * (nj.y - ni.y)
                    gx = int((sx - origin_x) / resolution)
                    gy = int((sy - origin_y) / resolution)
                    if gx < 0 or gx >= w or gy < 0 or gy >= h:
                        fine_blocked = True
                        break
                    if raw_grid[gy, gx]:
                        fine_blocked = True
                        break
                if not fine_blocked:
                    ni.neighbors.append(nj)
                    nj.neighbors.append(ni)


# ===================== Path Planning =====================

def _heuristic(a: TopoNode, b: TopoNode) -> float:
    """Euclidean distance heuristic for A*."""
    return math.hypot(a.x - b.x, a.y - b.y)


def a_star(graph: list[TopoNode]) -> Optional[list[tuple[float, float]]]:
    """A* shortest path from start to goal."""
    return _astar_impl(graph, use_heuristic=True)


def dijkstra_shortest_path(graph: list[TopoNode]) -> Optional[list[tuple[float, float]]]:
    """Dijkstra shortest path from start to goal (no heuristic)."""
    return _astar_impl(graph, use_heuristic=False)


def _astar_impl(graph: list[TopoNode], use_heuristic: bool) -> Optional[list[tuple[float, float]]]:
    """A* (or Dijkstra if use_heuristic=False) from start to goal node.

    Returns a list of (x, y) waypoints from start to goal, or None if no path.
    """
    start_node = None
    goal_node = None
    for node in graph:
        if node.is_start:
            start_node = node
        if node.is_goal:
            goal_node = node
    if start_node is None or goal_node is None:
        return None

    # g-score and parent map
    g_score: dict[int, float] = {node.id: float('inf') for node in graph}
    g_score[start_node.id] = 0.0
    parent: dict[int, Optional[int]] = {node.id: None for node in graph}

    # Priority queue: (f_score, tiebreaker, node_id)
    # tiebreaker prevents comparing TopoNode directly (which may be unorderable)
    tiebreaker = 0
    open_set: set[int] = {start_node.id}
    open_heap: list[tuple[float, int, int]] = [(0.0, tiebreaker, start_node.id)]
    tiebreaker += 1

    while open_heap:
        f_val, _, uid = heapq.heappop(open_heap)
        if uid not in open_set:
            continue
        open_set.discard(uid)

        node = graph[uid]
        if uid == goal_node.id:
            # Found — reconstruct path
            path: list[tuple[float, float]] = []
            cur: Optional[int] = uid
            while cur is not None:
                n = graph[cur]
                path.append((n.x, n.y))
                cur = parent[cur]
            path.reverse()
            return path

        for nb in node.neighbors:
            tentative_g = g_score[uid] + math.hypot(nb.x - node.x, nb.y - node.y)
            if tentative_g < g_score[nb.id]:
                parent[nb.id] = uid
                g_score[nb.id] = tentative_g
                if nb.id not in open_set:
                    h = _heuristic(nb, goal_node) if use_heuristic else 0.0
                    heapq.heappush(open_heap, (tentative_g + h, tiebreaker, nb.id))
                    tiebreaker += 1
                    open_set.add(nb.id)

    return None


# ===================== Grid Builders =====================

def _world_to_grid(wx: float, wy: float, origin_x: float, origin_y: float,
                   resolution: float) -> tuple[int, int]:
    return int((wx - origin_x) / resolution), int((wy - origin_y) / resolution)


def _fill_grid_rect(grid: np.ndarray, cx: float, cy: float, half_sx: float, half_sy: float,
                    origin_x: float, origin_y: float, resolution: float) -> None:
    """Fill a rectangular region in the grid."""
    h, w = grid.shape
    xmin, ymin = _world_to_grid(cx - half_sx, cy - half_sy, origin_x, origin_y, resolution)
    xmax, ymax = _world_to_grid(cx + half_sx, cy + half_sy, origin_x, origin_y, resolution)
    xmin = max(0, xmin); ymin = max(0, ymin)
    xmax = min(w - 1, xmax); ymax = min(h - 1, ymax)
    if xmin <= xmax and ymin <= ymax:
        grid[ymin:ymax + 1, xmin:xmax + 1] = True


def _fill_wall_band(grid: np.ndarray, half: float, wall_margin: float,
                    origin_x: float, origin_y: float, resolution: float) -> None:
    """Draw arena wall as thin frame (1 cell) + inflation band of ``wall_margin``."""
    arena_min = -half
    arena_max = half
    h, w = grid.shape

    def wg(wx: float, wy: float) -> tuple[int, int]:
        gx = int((wx - origin_x) / resolution)
        gy = int((wy - origin_y) / resolution)
        return max(0, min(w - 1, gx)), max(0, min(h - 1, gy))

    ax0, ay0 = wg(arena_min, arena_min)
    ax1, ay1 = wg(arena_max, arena_max)

    # Thin wall frame (always drawn)
    grid[ay0, ax0:ax1 + 1] = True
    grid[ay1, ax0:ax1 + 1] = True
    grid[ay0:ay1 + 1, ax0] = True
    grid[ay0:ay1 + 1, ax1] = True

    if wall_margin <= 0:
        return
    wm_in = wg(0, arena_min + wall_margin)[1]
    wm_out = wg(0, arena_max - wall_margin)[1]
    wm_left = wg(arena_min + wall_margin, 0)[0]
    wm_right = wg(arena_max - wall_margin, 0)[0]
    grid[ay0 + 1:wm_in + 1, ax0:ax1 + 1] = True
    grid[wm_out:ay1, ax0:ax1 + 1] = True
    grid[wm_in + 1:wm_out, ax0:wm_left + 1] = True
    grid[wm_in + 1:wm_out, wm_right:ax1 + 1] = True


def _build_planner_grid(obs_data: list,
                         half: float,
                         origin_x: float, origin_y: float,
                         resolution: float, shape: tuple,
                         planner_margin: float) -> np.ndarray:
    """Build occupancy grid covering arena with wall frame + inflation margin."""
    h, w = shape
    grid = np.zeros((h, w), dtype=bool)
    _fill_wall_band(grid, half, wall_margin=planner_margin,
                    origin_x=origin_x, origin_y=origin_y, resolution=resolution)
    for _, ox, oy, sx, sy, _ in obs_data:
        _fill_grid_rect(grid, ox, oy,
                        sx / 2.0 + planner_margin,
                        sy / 2.0 + planner_margin,
                        origin_x, origin_y, resolution)
    return grid


# ===================== Main Entry Point =====================

def plan_global_path(
    env,
    env_id: int,
    origin_x: float,
    origin_y: float,
    resolution: float,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    save_map: bool = True,
    use_astar: bool = True,
) -> Optional[list[tuple[float, float]]]:
    """Main global planning function.

    Extracts obstacle contours from an inflated planner grid, builds a
    topological visibility graph, and searches for a shortest path via A*
    (or Dijkstra).

    Args:
        env: The NaviSpotRL environment instance.
        env_id: Which parallel environment to plan for.
        origin_x, origin_y: Env-local coords of grid[0, 0] (= -half).
        resolution: Cell size in meters.
        start_xy: (x, y) env-local start position.
        goal_xy: (x, y) env-local goal position.
        save_map: If True, write ``topo_map.png`` to disk.
        use_astar: If True, use A*; otherwise use Dijkstra.

    Returns:
        List of (x, y) waypoints from start to goal, or None if no path.
    """
    cfg = env.cfg
    obs_data = env._obstacle_data.get(env_id, []) if hasattr(env, '_obstacle_data') else []
    half = cfg.arena_side_length / 2.0

    # Compute grid shape from arena dimensions
    padding = max(0.5, cfg.wall_safe_margin + cfg.obstacle_safe_margin + 0.2)
    arena_span = 2.0 * half + 2.0 * padding
    cells = max(1, int(arena_span / resolution + 0.5))
    grid_shape = (cells, cells)

    # Build raw grid (walls + raw obstacles, no inflation)
    raw_grid = _build_planner_grid(
        obs_data, half, origin_x, origin_y, resolution,
        grid_shape, planner_margin=0.0)

    # Build planner grid (walls + obstacles + inflation margin)
    planner_grid = _build_planner_grid(
        obs_data, half, origin_x, origin_y, resolution,
        grid_shape, cfg.planner_inflate_margin)

    # Step 1: Extract contours from both raw and inflated grids
    raw_contours = extract_obstacle_contours(
        raw_grid, origin_x, origin_y, resolution,
        min_pts=cfg.planner_contour_min_pts,
    )
    inflated_contours = extract_obstacle_contours(
        planner_grid, origin_x, origin_y, resolution,
        min_pts=cfg.planner_contour_min_pts,
    )
    # Step 2: Build topological graph from inflated contours (safety margin)
    graph = _build_graph(inflated_contours, start_xy, goal_xy)

    # Step 3: Add visibility edges — check against RAW contours (black = hard boundary)
    _build_visibility_edges(graph, raw_contours, raw_grid,
                            origin_x, origin_y, resolution,
                            cfg.planner_max_edge_len)

    # Step 4: Path search (A* or Dijkstra)
    if use_astar:
        path = a_star(graph)
    else:
        path = dijkstra_shortest_path(graph)

    # Debug info (commented out for training)
    # n_contour_nodes = sum(1 for n in graph if n.contour_id >= 0)
    # n_edges = sum(len(n.neighbors) for n in graph) // 2
    # label = "A*" if use_astar else "Dijkstra"
    # if path:
    #     print(f"[Planner] {label}: {len(path)} waypoints, {len(contours)} contours, "
    #           f"{n_contour_nodes} nodes, {n_edges} edges")
    # else:
    #     print(f"[Planner] {label}: NO PATH, {len(contours)} contours, "
    #           f"{n_contour_nodes} nodes")

    # Step 5: Save visualization
    if save_map:
        save_topo_map(raw_grid, planner_grid, inflated_contours, graph, path,
                      start_xy, goal_xy, origin_x, origin_y, resolution, obs_data)

    return path


# ===================== Map Visualization =====================

def save_topo_map(
    raw_grid: np.ndarray,
    planner_grid: np.ndarray,
    contours: list[list[tuple[float, float]]],
    graph: list[TopoNode],
    path: Optional[list[tuple[float, float]]],
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    origin_x: float,
    origin_y: float,
    resolution: float,
    obs_data: list,
) -> None:
    """Save topological map as PNG to ``utils/map/topo_map.png``.

    Color legend:
      - Black:      raw obstacles (no inflation)
      - Dark gray:  inflation margin
      - White:      free space
      - Green dots: contour / graph nodes
      - Yellow lines: visibility edges
      - Red dot:    start position
      - Blue dot:   goal position
      - Magenta line: shortest path
      - Deep blue dots: path waypoints
    """
    if os.environ.get("NAVISPOTRL_SAVE_MAP", "1") != "1":
        return

    map_dir = os.path.join(os.path.dirname(__file__), "map")
    os.makedirs(map_dir, exist_ok=True)

    h, w = planner_grid.shape
    img = np.full((h, w, 3), 255, dtype=np.uint8)

    # Layer 1: raw obstacles in black
    img[raw_grid] = [0, 0, 0]

    # Layer 2: inflation margin in dark gray
    margin_only = planner_grid & (~raw_grid)
    img[margin_only] = [170, 170, 170]

    def w2px(wx: float, wy: float) -> tuple[int, int]:
        return int((wx - origin_x) / resolution), int((wy - origin_y) / resolution)

    # Skip edge drawing (too much time for many nodes)
    # just draw contour nodes and path

    # Draw contour nodes (green)
    for node in graph:
        if node.is_start or node.is_goal:
            continue
        px, py = w2px(node.x, node.y)
        if 0 <= px < w and 0 <= py < h:
            cv2.circle(img, (px, py), 2, (0, 255, 0), -1)

    # Draw visibility edges (yellow) — sample a few random edges to keep fast
    drawn: set[tuple[int, int]] = set()
    for node in graph:
        px1, py1 = w2px(node.x, node.y)
        for nb in node.neighbors:
            ek = (min(node.id, nb.id), max(node.id, nb.id))
            if ek in drawn:
                continue
            drawn.add(ek)
            px2, py2 = w2px(nb.x, nb.y)
            if (0 <= px1 < w and 0 <= py1 < h and 0 <= px2 < w and 0 <= py2 < h):
                cv2.line(img, (px1, py1), (px2, py2), (255, 255, 0), 1)

    # Start (red)
    px_s, py_s = w2px(start_xy[0], start_xy[1])
    if 0 <= px_s < w and 0 <= py_s < h:
        cv2.drawMarker(img, (px_s, py_s), (255, 0, 0), cv2.MARKER_STAR, 10, 2)

    # Goal (blue)
    px_g, py_g = w2px(goal_xy[0], goal_xy[1])
    if 0 <= px_g < w and 0 <= py_g < h:
        cv2.drawMarker(img, (px_g, py_g), (0, 0, 255), cv2.MARKER_STAR, 10, 2)

    # Path (magenta line + deep blue dots)
    if path and len(path) >= 2:
        img_pts = [w2px(p[0], p[1]) for p in path]
        for px, py in img_pts:
            cv2.circle(img, (px, py), 5, (139, 0, 0), -1)
        for k in range(len(img_pts) - 1):
            cv2.line(img, img_pts[k], img_pts[k + 1], (255, 0, 255), 2)

    # Flip for correct orientation
    img = img[::-1, :, :]
    cv2.imwrite(os.path.join(map_dir, "topo_map.png"), img)
