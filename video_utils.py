"""Video I/O helpers — read, write, and inspect video files.

Keeps codec-handling and resource-management logic out of the main
analysis pipeline so analyzer.py stays focused on CV logic.
"""

import cv2
import shutil
import subprocess
from pathlib import Path

try:
    import imageio_ffmpeg
except ImportError:  # Optional fallback for environments without imageio-ffmpeg
    imageio_ffmpeg = None


def get_video_info(path: str) -> dict:
    """Extract metadata from a video file.

    Returns:
        dict with keys: fps, width, height, frame_count, duration_sec
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")

    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 25,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    info["duration_sec"] = round(info["frame_count"] / info["fps"], 2)
    cap.release()
    return info


def create_video_writer(output_path: str, fps: float, width: int, height: int):
    """Create an OpenCV VideoWriter with codec fallback.

    Returns:
        tuple[cv2.VideoWriter, str]: writer and the selected codec tag.
    """
    codec_candidates = ["avc1", "H264", "mp4v"]

    for codec in codec_candidates:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if writer.isOpened():
            return writer, codec

    raise RuntimeError(f"Failed to create video writer for: {output_path}")


def get_ffmpeg_executable() -> str | None:
    """Return a usable ffmpeg executable path if available."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    if imageio_ffmpeg is not None:
        return imageio_ffmpeg.get_ffmpeg_exe()

    return None


def transcode_to_browser_mp4(input_path: str, output_path: str) -> bool:
    """Transcode video to browser-friendly H.264 MP4.

    Produces H.264 + yuv420p + faststart output, which is compatible with
    modern browsers and allows playback before full file download.
    """
    ffmpeg = get_ffmpeg_executable()
    if not ffmpeg:
        return False

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        input_path,
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0 and output_file.exists() and output_file.stat().st_size > 0


def iter_frames(path: str):
    """Generator that yields (frame_index, frame) tuples from a video.

    Handles resource cleanup automatically when the generator is
    exhausted or garbage-collected.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")

    idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield idx, frame
            idx += 1
    finally:
        cap.release()
