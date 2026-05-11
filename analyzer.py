"""Full CV pipeline: detect → track → analyze → render.

Processes each frame of the input video through YOLOv8 + ByteTrack,
computes velocity vectors, detects anomalies, draws overlays (bounding
boxes, trajectory trails, velocity arrows, density heatmap, HUD), and
writes the annotated result to an output video file.
"""

from ultralytics import YOLO
import cv2
import numpy as np
import os
from collections import defaultdict
from tracker_utils import (
    get_velocity, get_crowd_flow, is_anomaly,
    get_speed, draw_heatmap, get_direction, get_dwell_seconds,
    create_zones, get_zone_for_point, create_boundary_boxes, check_line_crossing
)
from video_utils import create_video_writer, transcode_to_browser_mp4

# COCO class indices for vehicles we care about
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck (no bicycle - too small for aerial)
VEHICLE_CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}  # for tracking per-class counts
FALSE_POSITIVE_CLASSES = [9, 10, 58, 62]  # traffic light, fire hydrant, and unknown classes often misdetected in aerial
TRACK_HISTORY_LEN = 60          # max trajectory points to keep per object

# Drone/aerial detection is challenging: objects are tiny and often overlap
CONFIDENCE_THRESHOLD = 0.15     # allow more low-confidence detections for aerial
IOU_THRESHOLD = 0.4             # reduce NMS threshold to handle tight clusters
INFERENCE_IMGSZ = 1408          # larger inference for small object detection
TRACKER_CONFIG = "bytetrack_low_conf.yaml"

# Use YOLOv8m (medium) instead of nano for better aerial generalization
# YOLOv8n struggles with top-down vehicle perspectives. YOLOv8m has ~3x better
# vehicle detection on aerial footage in testing
model = YOLO("yolov8m.pt")


def run_analysis(input_path: str, output_path: str, on_frame=None) -> dict:
    """Run the full analysis pipeline on a video.

    Args:
        input_path: Path to the uploaded video file.
        output_path: Path where the annotated output video will be saved.

    Returns:
        dict with summary stats:
            - total_vehicles: unique vehicle IDs seen
            - anomaly_count: vehicles flagged as moving against flow
            - avg_speed_px_per_frame: average speed across all frames
        If on_frame is provided, it will be called with each annotated
        frame before writing to the output video.
    """
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    raw_output_path = output_path.replace(".mp4", "_raw.mp4")
    out, selected_codec = create_video_writer(raw_output_path, fps, width, height)

    track_history = defaultdict(list)
    all_ids_seen = set()
    anomaly_ids = set()
    class_counts = {"car": set(), "motorcycle": set(), "bus": set(), "truck": set()}
    direction_counts = {"North": 0, "South": 0, "East": 0, "West": 0, "Stationary": 0}
    finalized_ids = set()
    dwell_times = {}
    congested_ids = set()
    zone_counts = {"zone_1": set(), "zone_2": set(), "zone_3": set(), "zone_4": set()}
    zones = create_zones(width, height)
    
    # Initialize boundary boxes for crossing detection (one-way mode by default)
    # User can configure this to 'four_way' for intersection scenarios
    boundary_data = create_boundary_boxes(width, height, mode="one_way")
    vehicle_boundary_crossed = {}  # Track {obj_id: True} if vehicle has been counted
    
    speeds = []
    max_detected_vehicles = 0
    frame_idx = 0
    prev_active_ids = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 with ByteTrack — persist=True keeps tracker state
        # across frames so IDs are consistent
        results = model.track(
            frame,
            persist=True,
            tracker=TRACKER_CONFIG,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=INFERENCE_IMGSZ,
            verbose=False
        )

        active_ids = []
        active_vehicle_count = 0

        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            if results[0].boxes.id is not None:
                ids = results[0].boxes.id.cpu().numpy().astype(int)
            else:
                ids = [None] * len(boxes)

            # Filter out obvious false positives in aerial detection
            valid_indices = [
                i for i, cls in enumerate(classes)
                if cls not in FALSE_POSITIVE_CLASSES
                and (cls in VEHICLE_CLASSES or cls not in [9, 10])
            ]
            boxes = boxes[valid_indices]
            classes = classes[valid_indices]
            ids = [ids[i] if i < len(ids) else None for i in valid_indices]

            # First pass: update trajectory histories
            for box, obj_id, cls in zip(boxes, ids, classes):
                if cls not in VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                active_vehicle_count += 1

                if obj_id is None:
                    continue

                # Append center point to trajectory
                track_history[obj_id].append((cx, cy))
                if len(track_history[obj_id]) > TRACK_HISTORY_LEN:
                    track_history[obj_id].pop(0)

                all_ids_seen.add(obj_id)
                active_ids.append(obj_id)
                
                # Check for boundary box crossings (one-way or four-way mode)
                if obj_id not in vehicle_boundary_crossed:
                    vehicle_boundary_crossed[obj_id] = False
                    
                # Get previous center point if available
                prev_center = track_history[obj_id][-2] if len(track_history[obj_id]) >= 2 else (cx, cy)
                curr_center = (cx, cy)
                
                # Check crossing for each boundary box
                if not vehicle_boundary_crossed[obj_id]:
                    for box_name, box_def in boundary_data["boxes"].items():
                        if check_line_crossing(prev_center, curr_center, box_def):
                            boundary_data["crossing_counts"][box_name] += 1
                            vehicle_boundary_crossed[obj_id] = True
                            break  # Count only once per vehicle, on first crossing
                
                # Track this vehicle ID in the appropriate class set
                class_name = VEHICLE_CLASS_NAMES.get(cls)
                if class_name:
                    class_counts[class_name].add(obj_id)

                # Zone membership for this vehicle (unique per ID)
                zone_key = get_zone_for_point(zones, cx, cy)
                if zone_key and obj_id is not None:
                    zone_counts[zone_key].add(obj_id)

            # Compute crowd flow (average velocity of all active objects)
            crowd_flow = get_crowd_flow(track_history, active_ids)

            # Second pass: compute per-object metrics and draw overlays
            for box, obj_id, cls in zip(boxes, ids, classes):
                if cls not in VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                if obj_id is not None:
                    # velocity and speed
                    velocity = get_velocity(track_history, obj_id)
                    speed = get_speed(velocity)
                    speeds.append(speed)

                    # dwell time for this tracked id (seconds)
                    dwell = get_dwell_seconds(track_history, obj_id, fps)
                    dwell_times[obj_id] = dwell

                    # mark congested if dwell > 8s AND speed < 3 px/frame
                    if dwell > 8.0 and speed < 3.0:
                        congested_ids.add(obj_id)

                    anomaly = is_anomaly(velocity, crowd_flow)
                    if anomaly:
                        anomaly_ids.add(obj_id)
                else:
                    velocity = (0, 0)
                    anomaly = False

                # Color coding: red = anomaly, green = normal, orange = congested
                color = (0, 0, 255) if anomaly else (0, 255, 0)
                if obj_id is not None and obj_id in congested_ids:
                    color = (0, 165, 255)  # orange (BGR)

                # Bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID:{obj_id}" if obj_id is not None else "DET"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

                # Dwell time label (if available)
                if obj_id is not None and obj_id in dwell_times:
                    dwell_label = f"{dwell_times[obj_id]}s"
                    cv2.putText(frame, dwell_label, (x1, y1 - 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)

                # Velocity arrow — only draw if object is moving
                if abs(velocity[0]) + abs(velocity[1]) > 2:
                    end = (cx + int(velocity[0] * 2), cy + int(velocity[1] * 2))
                    cv2.arrowedLine(frame, (cx, cy), end, color, 2, tipLength=0.35)

                # Trajectory trail (yellow)
                if obj_id is not None:
                    history = track_history[obj_id]
                    for i in range(1, len(history)):
                        cv2.line(frame, history[i-1], history[i], (255, 255, 0), 1)

            max_detected_vehicles = max(max_detected_vehicles, active_vehicle_count)

        # Determine which IDs left the frame this iteration and finalize
        # their direction counts (only once per object)
        current_active_set = set(active_ids)
        left_ids = prev_active_ids - current_active_set
        for lid in left_ids:
            if lid in finalized_ids:
                continue
            history = track_history.get(lid, [])
            if not history:
                continue
            # average velocity over full history (per-frame)
            denom = max(1, len(history) - 1)
            avg_vx = (history[-1][0] - history[0][0]) / denom
            avg_vy = (history[-1][1] - history[0][1]) / denom
            direction = get_direction(avg_vx, avg_vy)
            if direction in direction_counts:
                direction_counts[direction] += 1
            finalized_ids.add(lid)

        prev_active_ids = current_active_set

        # Density heatmap overlay
        frame = draw_heatmap(frame, track_history, active_ids)

        # Zone overlays: semi-transparent quadrant fills and labels
        overlay = frame.copy()
        zone_colors = {
            "zone_1": (60, 120, 200),
            "zone_2": (60, 200, 120),
            "zone_3": (200, 60, 90),
            "zone_4": (180, 140, 60),
        }
        for zkey, poly in zones.items():
            cv2.fillPoly(overlay, [poly], zone_colors.get(zkey, (80, 80, 80)))

        frame = cv2.addWeighted(overlay, 0.08, frame, 0.92, 0)

        # Zone labels (corner counts)
        for zkey, poly in zones.items():
            x, y, w, h = cv2.boundingRect(poly)
            count = len(zone_counts.get(zkey, set()))
            label = f"{zkey.upper().replace('_',' ')}: {count} vehicles"
            cv2.putText(frame, label, (x + 6, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Draw boundary boxes for crossing detection
        for box_name, box_def in boundary_data["boxes"].items():
            x1 = box_def["x1"]
            y1 = box_def["y1"]
            x2 = box_def["x2"]
            y2 = box_def["y2"]
            color = box_def["color"]
            
            # Draw box outline
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw crossing count label
            crossing_count = boundary_data["crossing_counts"][box_name]
            label = f"{box_name}: {crossing_count}"
            cv2.putText(frame, label, (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # HUD — vehicle count and anomaly count
        cv2.putText(frame, f"Vehicles: {active_vehicle_count}", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Anomalies: {len(anomaly_ids)}", (15, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Boundary Crossings: {sum(boundary_data['crossing_counts'].values())}", (15, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        if on_frame is not None:
            on_frame(frame)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    # Ensure browser-compatible output (H.264 / yuv420p). If transcoding is
    # unavailable, keep the raw file so the pipeline still returns a video.
    transcoded = transcode_to_browser_mp4(raw_output_path, output_path)
    if transcoded:
        os.remove(raw_output_path)
    else:
        os.replace(raw_output_path, output_path)

    total_vehicles = len(all_ids_seen) if all_ids_seen else max_detected_vehicles
    # Compute average dwell time across all seen IDs
    if all_ids_seen:
        dwell_vals = [get_dwell_seconds(track_history, i, fps) for i in all_ids_seen]
        avg_dwell_time = round(float(np.mean(dwell_vals)), 1) if dwell_vals else 0.0
    else:
        avg_dwell_time = 0.0

    return {
        "total_vehicles": total_vehicles,
        "anomaly_count": len(anomaly_ids),
        "avg_speed_px_per_frame": round(float(np.mean(speeds)), 2) if speeds else 0,
        "video_codec": "h264" if transcoded else selected_codec,
        "class_counts": {
            "car": len(class_counts["car"]),
            "motorcycle": len(class_counts["motorcycle"]),
            "bus": len(class_counts["bus"]),
            "truck": len(class_counts["truck"]),
        },
        "direction_counts": direction_counts,
        "dwell_times": {str(k): v for k, v in dwell_times.items()},
        "avg_dwell_time": avg_dwell_time,
        "congested_count": len(congested_ids),
        "zone_counts": {
            "zone_1": len(zone_counts["zone_1"]),
            "zone_2": len(zone_counts["zone_2"]),
            "zone_3": len(zone_counts["zone_3"]),
            "zone_4": len(zone_counts["zone_4"]),
        },
        "boundary_crossings": boundary_data["crossing_counts"],
        "boundary_mode": boundary_data["mode"],
    }
