# Drone Traffic Analyzer

A computer vision system that analyzes aerial drone footage to extract real-time traffic intelligence.

**Live Demo: [your-demo-link-here]**

---

## What it does

Most drone traffic tools stop at counting. This system goes further:

| Feature | What it detects |
|---|---|
| Vehicle Detection | Cars, motorcycles, buses, trucks via YOLOv8 |
| Multi-object Tracking | Persistent ID per vehicle across frames (ByteTrack) |
| Flow Direction | North/South/East/West movement per vehicle |
| Anomaly Detection | Vehicles moving against traffic flow |
| Dwell Time | How long each vehicle stays in frame, flags congestion |
| Zone Analysis | Per-quadrant vehicle counts across the frame |
| Density Heatmap | Where vehicles spend the most time, shows bottlenecks |
| Class Breakdown | Separate counts for cars, motorcycles, buses, trucks |

---

## Why this matters

Manual traffic analysis from drone footage is slow and subjective. This system produces an automated traffic intelligence report from any aerial clip, useful for city planners identifying congestion hotspots, traffic authorities monitoring intersections, event organizers managing vehicle flow, and infrastructure inspection teams.

---

## Tech Stack

Backend: Python 3.10, FastAPI, YOLOv8x (Ultralytics), ByteTrack, OpenCV, NumPy
Frontend: React (Vite), hosted on Vercel
Infrastructure: Hugging Face Spaces with Docker 

---

## How it works

Pipeline:

1. Drone video input
2. YOLOv8x detects vehicles per frame (bounding boxes, class labels, confidence scores)
3. ByteTrack assigns persistent IDs across frames (IoU matching + low-confidence recovery)
4. Velocity vectors computed from trajectory history (displacement of bounding box center over N frames)
5. Behavior analysis: anomaly via dot product angle, dwell time via track length/fps, direction via dominant velocity axis, zone via pointPolygonTest
6. Heatmap: Gaussian blur over cumulative position history
7. Annotated video + JSON summary returned

### Anomaly Detection

```python
cos_angle = dot(v_vehicle, v_crowd) / (|v_vehicle| * |v_crowd|)
angle = degrees(arccos(cos_angle))
is_anomaly = angle > 100
```

If the angle between a vehicle velocity vector and the average traffic flow exceeds 100 degrees, it is flagged as an anomaly and highlighted red.

### Why ByteTrack over SORT

SORT discards low-confidence detections entirely. ByteTrack keeps them in a secondary buffer and attempts re-matching, which is critical in drone footage where vehicles get partially occluded. Better ID retention means more accurate counts and trajectories.

---

## Project Structure

```
drone-analyzer/
├── main.py            FastAPI app, routes, background tasks
├── analyzer.py        Full CV pipeline
├── tracker_utils.py   Velocity, anomaly, dwell time, direction logic
├── requirements.txt
└── Dockerfile

drone-frontend/
├── src/
│   ├── App.jsx
│   ├── components/
│   │   ├── UploadZone.jsx
│   │   ├── ProcessingView.jsx
│   │   ├── ResultsView.jsx
│   │   ├── StatCard.jsx
│   │   ├── ClassBreakdown.jsx
│   │   ├── DirectionCompass.jsx
│   │   └── ZoneGrid.jsx
│   ├── hooks/
│   │   └── useAnalysis.js
│   └── utils/
│       └── api.js
```

---

## API

### POST /analyze
Upload a video. Returns job ID immediately, processing runs in background.

Request: multipart/form-data with file field

Response:
```json
{ "job_id": "abc-123" }
```

### GET /status/{job_id}

Response when done:
```json
{
  "status": "done",
  "summary": {
    "total_vehicles": 47,
    "anomaly_count": 3,
    "avg_speed_px_per_frame": 8.4,
    "avg_dwell_time": 6.2,
    "congested_count": 5,
    "class_counts": { "car": 31, "motorcycle": 12, "bus": 2, "truck": 2 },
    "direction_counts": { "North": 8, "South": 11, "East": 19, "West": 9 },
    "zone_counts": { "zone_1": 14, "zone_2": 18, "zone_3": 9, "zone_4": 6 }
  }
}
```

### GET /video/{job_id}
Returns annotated output video as video/mp4.

---

## Run Locally

### Backend
```bash
git clone https://github.com/YOUR-USERNAME/drone-analyzer
cd drone-analyzer
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Backend at http://localhost:8000
API docs at http://localhost:8000/docs

### Frontend
```bash
git clone https://github.com/YOUR-USERNAME/drone-frontend
cd drone-frontend
npm install
```

Create .env:
```
VITE_API_URL=http://localhost:8000
```

```bash
npm run dev
```

Frontend at http://localhost:5173

---

## Limitations

Stationary drone assumed: zone analysis and flow direction are relative to the frame. A moving drone shifts the entire frame making pixel-space zones unreliable. Real deployment would require GPS/IMU geo-referencing to anchor zones in world coordinates.

Speed is a proxy: measured in pixels per frame, not km/h. Real speed estimation requires known altitude and camera FOV calibration.

Occlusion: heavily occluded vehicles may lose tracking ID temporarily. ByteTrack recovers most but not all.

---

## What I Would Add Next

- Optical flow stabilization to handle moving drone footage
- Real-world speed estimation using drone altitude and camera FOV
- Geo-referenced zones anchored to GPS coordinates
- Time-series graphs of vehicle count and speed exportable as CSV
- Multi-camera support for stitching feeds from multiple drones

---

## Dataset

Test videos from the VisDrone Dataset, drone footage collected for computer vision research across multiple cities.

https://github.com/VisDrone/VisDrone-Dataset

---

## Author

Your Name
GitHub: github.com/NetCatt

