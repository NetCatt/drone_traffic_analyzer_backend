import numpy as np


def get_velocity(track_history, obj_id, lookback=5):
    """Compute velocity vector from trajectory history.

    Uses a lookback window (default 5 frames) instead of single-frame
    displacement to smooth out detection jitter. Returns (vx, vy) where
    the magnitude is proportional to speed in pixels/frame.
    """
    history = track_history.get(obj_id, [])
    if len(history) < 2:
        return (0, 0)
    recent = history[-lookback:] if len(history) >= lookback else history
    vx = recent[-1][0] - recent[0][0]
    vy = recent[-1][1] - recent[0][1]
    return (vx, vy)


def get_crowd_flow(track_history, active_ids, lookback=5):
    """Compute average flow direction of all tracked objects.

    Filters out near-stationary objects (displacement < 1px) to avoid
    diluting the flow vector with noise from parked vehicles.
    """
    velocities = [get_velocity(track_history, i, lookback) for i in active_ids]
    velocities = [(vx, vy) for vx, vy in velocities if abs(vx) + abs(vy) > 1]  # filter stationary
    if not velocities:
        return (0, 0)
    avg_vx = np.mean([v[0] for v in velocities])
    avg_vy = np.mean([v[1] for v in velocities])
    return (avg_vx, avg_vy)


def is_anomaly(obj_velocity, crowd_velocity, angle_threshold=100):
    """True if object is moving significantly against crowd flow.

    Uses dot product to compute the angle between the object's velocity
    vector and the average crowd flow vector. An angle > 100° indicates
    the vehicle is moving against traffic (wrong-way detection).

    Math:
        cos(θ) = (v1 · v2) / (|v1| × |v2|)
        θ = arccos(cos(θ))
        anomaly if θ > threshold
    """
    vx1, vy1 = obj_velocity
    vx2, vy2 = crowd_velocity
    mag1 = np.sqrt(vx1**2 + vy1**2)
    mag2 = np.sqrt(vx2**2 + vy2**2)
    if mag1 < 2 or mag2 < 2:  # ignore near-stationary objects
        return False
    cos_angle = np.dot([vx1, vy1], [vx2, vy2]) / (mag1 * mag2)
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
    return angle > angle_threshold


def get_speed(velocity):
    """Magnitude of velocity vector — proxy for speed in px/frame."""
    return np.sqrt(velocity[0]**2 + velocity[1]**2)


def draw_heatmap(frame, track_history, active_ids, alpha=0.3):
    """Overlay trajectory density heatmap on frame.

    Accumulates Gaussian-blurred circles at every tracked position,
    normalizes to 0-255, and applies JET colormap. Hot zones indicate
    where vehicles spend the most time (congestion indicators).
    """
    import cv2
    heatmap = np.zeros(frame.shape[:2], dtype=np.float32)
    for obj_id in active_ids:
        for point in track_history.get(obj_id, []):
            cv2.circle(heatmap, point, 25, 1, -1)
    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_colored = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, 1 - alpha, heatmap_colored, alpha, 0)


def get_direction(vx, vy):
    """Return cardinal direction based on velocity vector.

    Rules:
    - If overall magnitude < 2 -> 'Stationary'
    - Dominant axis determines cardinal direction:
      - vy dominant and negative -> 'North'
      - vy dominant and positive -> 'South'
      - vx dominant and positive -> 'East'
      - vx dominant and negative -> 'West'
    """
    mag = np.sqrt(vx * vx + vy * vy)
    if mag < 2:
        return "Stationary"

    if abs(vy) > abs(vx):
        return "North" if vy < 0 else "South"
    else:
        return "East" if vx > 0 else "West"


def get_dwell_seconds(track_history, obj_id, fps):
    """Return dwell time in seconds for an object based on its track history.

    Formula: number of frames in history divided by fps. Rounded to 1 decimal.
    """
    history = track_history.get(obj_id, [])
    if not history:
        return 0.0
    seconds = len(history) / float(fps or 1)
    return round(seconds, 1)


def create_zones(width, height):
    """Create four quadrant polygon zones as numpy arrays (int32).

    Returns a dict with keys: 'zone_1', 'zone_2', 'zone_3', 'zone_4'.
    """
    mid_x = width // 2
    mid_y = height // 2
    zones = {
        "zone_1": np.array([[0, 0], [mid_x, 0], [mid_x, mid_y], [0, mid_y]], dtype=np.int32),
        "zone_2": np.array([[mid_x, 0], [width, 0], [width, mid_y], [mid_x, mid_y]], dtype=np.int32),
        "zone_3": np.array([[0, mid_y], [mid_x, mid_y], [mid_x, height], [0, height]], dtype=np.int32),
        "zone_4": np.array([[mid_x, mid_y], [width, mid_y], [width, height], [mid_x, height]], dtype=np.int32),
    }
    return zones


def get_zone_for_point(zones, cx, cy):
    """Return the zone key that contains point (cx, cy) or None."""
    import cv2

    for key, poly in zones.items():
        # cv2.pointPolygonTest returns >0 for inside, 0 for on edge, <0 for outside
        if cv2.pointPolygonTest(poly, (int(cx), int(cy)), False) >= 0:
            return key
    return None


def create_boundary_boxes(width, height, mode="one_way"):
    """Create boundary boxes for vehicle crossing detection.
    
    Args:
        width: Frame width in pixels
        height: Frame height in pixels
        mode: 'one_way' or 'four_way'
            - 'one_way': Single boundary box at frame exit (right edge)
            - 'four_way': Four boundary boxes (top-left, top-right, bottom-left, bottom-right)
    
    Returns:
        dict with boundary box metadata:
        - 'boxes': dict of box definitions {name: {'x1': int, 'y1': int, 'x2': int, 'y2': int}}
        - 'mode': str ('one_way' or 'four_way')
        - 'crossing_counts': dict of crossing counts per box {name: int}
    """
    if mode == "one_way":
        # Single vertical line at right edge of frame (vehicle exit boundary)
        # Box is 100px wide at the right edge
        x_offset = 100
        boxes = {
            "exit": {
                "x1": width - x_offset,
                "y1": 0,
                "x2": width,
                "y2": height,
                "direction": "exit",
                "color": (0, 255, 255)  # cyan
            }
        }
    elif mode == "four_way":
        # Four boxes at quadrant boundaries (entry points)
        # Each box is 80px×80px positioned at corners for intersection detection
        box_size = 80
        boxes = {
            "north": {
                "x1": width // 2 - box_size // 2,
                "y1": 0,
                "x2": width // 2 + box_size // 2,
                "y2": box_size,
                "direction": "north",
                "color": (255, 0, 0)  # blue
            },
            "south": {
                "x1": width // 2 - box_size // 2,
                "y1": height - box_size,
                "x2": width // 2 + box_size // 2,
                "y2": height,
                "direction": "south",
                "color": (0, 255, 0)  # green
            },
            "east": {
                "x1": width - box_size,
                "y1": height // 2 - box_size // 2,
                "x2": width,
                "y2": height // 2 + box_size // 2,
                "direction": "east",
                "color": (0, 165, 255)  # orange
            },
            "west": {
                "x1": 0,
                "y1": height // 2 - box_size // 2,
                "x2": box_size,
                "y2": height // 2 + box_size // 2,
                "direction": "west",
                "color": (255, 0, 255)  # magenta
            }
        }
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return {
        "boxes": boxes,
        "mode": mode,
        "crossing_counts": {name: 0 for name in boxes.keys()}
    }


def point_in_box(point, box):
    """Check if point (x, y) is inside box.
    
    Args:
        point: tuple (x, y)
        box: dict with keys 'x1', 'y1', 'x2', 'y2'
    
    Returns:
        bool
    """
    x, y = point
    return box["x1"] <= x <= box["x2"] and box["y1"] <= y <= box["y2"]


def check_line_crossing(prev_point, curr_point, box):
    """Check if a point crossed from outside to inside a box.
    
    Uses a simple approach: check if the center point transitioned from
    outside the box to inside the box between two frames.
    
    Args:
        prev_point: tuple (prev_x, prev_y) from previous frame
        curr_point: tuple (curr_x, curr_y) from current frame
        box: dict with keys 'x1', 'y1', 'x2', 'y2'
    
    Returns:
        bool: True if crossing detected (prev outside, curr inside)
    """
    prev_inside = point_in_box(prev_point, box)
    curr_inside = point_in_box(curr_point, box)
    
    # Crossing detected if point transitioned from outside to inside
    return (not prev_inside) and curr_inside
