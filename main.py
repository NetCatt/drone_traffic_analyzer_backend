"""FastAPI backend for Drone Traffic Analyzer.

Endpoints:
    POST /analyze        — Upload a video, start background processing, return job_id
    GET  /status/{id}    — Poll job status (processing / done / error)
    GET  /video/{id}     — Download the annotated output video

Architecture notes:
    - Video processing runs in a BackgroundTask so the upload request
      returns immediately with a job_id (avoids HTTP timeout).
    - Job state is stored in-memory (dict) — fine for a demo/single-server
      deployment. For production, use Redis or a database.
    - CORS is wide-open (*) for local dev; tighten for production.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import shutil
import uuid
import os
import time
import threading
import cv2
from analyzer import run_analysis

app = FastAPI(
    title="Drone Traffic Analyzer",
    description="Aerial drone footage → annotated traffic intelligence video",
    version="1.0.0",
)

# CORS — allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create temp directory for uploaded/processed videos
os.makedirs("temp", exist_ok=True)

# In-memory job store (sufficient for demo)
jobs = {}

# Max upload size: 50 MB
MAX_FILE_SIZE_MB = 50


@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    """Accept a video upload and start background analysis.

    Returns a job_id that the client uses to poll for status.
    """
    # Read file content to check size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(400, f"File too large. Max {MAX_FILE_SIZE_MB}MB.")

    job_id = str(uuid.uuid4())
    input_path = f"temp/{job_id}_input.mp4"
    output_path = f"temp/{job_id}_output.mp4"

    # Write uploaded file to disk
    with open(input_path, "wb") as buffer:
        buffer.write(content)

    jobs[job_id] = {
        "status": "processing",
        "latest_frame": None,
        "frame_seq": 0,
    }
    worker = threading.Thread(
        target=process_job,
        args=(job_id, input_path, output_path),
        daemon=True,
    )
    worker.start()

    return {"job_id": job_id}


def process_job(job_id: str, input_path: str, output_path: str):
    """Background task — runs the full CV pipeline and updates job status."""
    try:
        def publish_frame(frame):
            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                return

            job = jobs.get(job_id)
            if not job:
                return

            job["latest_frame"] = encoded.tobytes()
            job["frame_seq"] = job.get("frame_seq", 0) + 1

        summary = run_analysis(input_path, output_path, on_frame=publish_frame)
        job = jobs.get(job_id, {})
        job["status"] = "done"
        job["summary"] = summary
        jobs[job_id] = job
    except Exception as e:
        job = jobs.get(job_id, {})
        job["status"] = "error"
        job["message"] = str(e)
        jobs[job_id] = job


@app.get("/status/{job_id}")
def get_status(job_id: str):
    """Poll the status of a processing job.

    Returns:
        - {"status": "processing"} while running
        - {"status": "done", "summary": {...}} when complete
        - {"status": "error", "message": "..."} on failure
        - {"status": "not_found"} for invalid job_id
    """
    job = jobs.get(job_id)
    if not job:
        return {"status": "not_found"}

    response = {"status": job.get("status", "processing")}
    if "summary" in job:
        response["summary"] = job["summary"]
    if "message" in job:
        response["message"] = job["message"]
    if "frame_seq" in job:
        response["frame_seq"] = job["frame_seq"]

    return response


@app.get("/video/{job_id}")
def get_video(job_id: str):
    """Serve the annotated output video for a completed job."""
    path = f"temp/{job_id}_output.mp4"
    if not os.path.exists(path):
        return JSONResponse(status_code=404, content={"error": "Video not found"})
    return FileResponse(path, media_type="video/mp4")


@app.get("/stream/{job_id}")
def stream_video(job_id: str):
    """Live MJPEG stream of annotated frames while processing."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    def frame_generator():
        last_seq = -1
        idle_cycles = 0

        while True:
            job = jobs.get(job_id)
            if not job:
                break

            frame = job.get("latest_frame")
            seq = job.get("frame_seq", -1)

            if frame is not None and seq != last_seq:
                last_seq = seq
                idle_cycles = 0
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
            else:
                idle_cycles += 1

            if job.get("status") in {"done", "error"} and idle_cycles > 20:
                break

            time.sleep(0.1)

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


# Serve React build as static files in production (after frontend build)
# Uncomment the line below after building the frontend:
# app.mount("/", StaticFiles(directory="../frontend/dist", html=True), name="static")
